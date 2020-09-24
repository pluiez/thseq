import contextlib
import datetime
import logging
import sys
from typing import *

import numpy as np
import torch
import torch.nn
from torch.nn.parallel import DistributedDataParallel as DDP

import thseq.data.iterators as iterators
import thseq.optim as optim
import thseq.utils as utils
from thseq.data import Vocabulary

logger = logging.getLogger(__name__)


def _merge_step_logs(logs: List[Dict]):
    log = {}
    for k in logs[0].keys():
        if isinstance(logs[0][k], (int, float)):
            log[k] = utils.aggregate_values(logs, k, reduce=sum)
    log['loss'] /= log['normalizer']
    log['nll_loss'] = log['nll_loss'] / np.log(2) / log['true_tokens_y']
    log['ppl'] = np.exp2(log['nll_loss'])
    log['gnorm'] /= len(logs)
    log['wps'] /= len(logs)
    log['ups'] /= len(logs)
    return log


def _merge_batch_stats(log, batch, loss, nll_loss):
    log['loss'] = log.get('loss', 0) + loss
    log['nll_loss'] = log.get('nll_loss', 0) + nll_loss
    for key in batch.keys():
        if isinstance(batch[key], (int, float)):
            log[key] = log.get(key, 0) + batch[key]
    return log


def _zero_log(log, keys):
    for k in keys:
        log[k] -= log[k]


class Trainer(object):
    def __init__(self, args,
                 model,
                 criterion,
                 vocabularies: Union[List[Vocabulary], Tuple[Vocabulary, ...]],
                 optimizer: Union[optim._Optimizer, optim.FP16Optimizer],
                 lr_scheduler,
                 train_itr: iterators.EpochIterator):
        super().__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.vocabularies = vocabularies
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_itr = train_itr

        self.fp16 = args.fp16
        self.fp16_enabled = args.fp16 is not None and args.fp16 != 'none'
        self.grad_scaler = optim.GradScaler(enabled=self.fp16_enabled)

        self._global_step = 0
        self._epoch = 0

        self.saver = utils.checkpoint.Saver(
            args.checkpoint,
            args.save_checkpoint_secs,
            args.save_checkpoint_steps,
            args.save_checkpoint_epochs,
            args.keep_checkpoint_max,
            args.keep_epoch_checkpoint_max,
            args.keep_best_checkpoint_max,
            self.state_dict
        )

        self.timer = utils.ElapsedTimeMeter()
        self.wps = utils.SpeedMeter()
        self.ups = utils.SpeedMeter()
        self._last_validate_step = -1

        self._overflow = 0

    @property
    def epoch(self):
        return self._epoch

    @property
    def global_step(self):
        return self._global_step

    @property
    def elapse(self):
        return self.timer.elapse

    @property
    def best_at(self):
        return self.saver.best_at

    @property
    def best_score(self):
        return self.saver.best_score

    def inc_epoch(self):
        self._epoch += 1
        return self.epoch

    def inc_global_step(self):
        self._global_step += 1
        return self.global_step

    def fb(self, batch: dict, dummy=False):
        self.model.train()
        batch = utils.move_cuda(batch)
        with self._amp_autocast():
            loss, nll_loss = self.criterion(self.model(batch), batch['y'])
            if dummy:
                loss = loss * 0.0

        self.grad_scaler.scale(loss).backward()
        loss = loss.item()
        nll_loss = nll_loss.item()

        if self.args.profile:
            utils.profile_nan(self.model)

        return loss, nll_loss

    def update(self, normalizer):
        if self.fp16 == 'half':
            self.optimizer.fp16_grads_to_fp32_grads()
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = utils.scale_clip_grad_(
            self.optimizer.parameters(),
            max(self.args.dist_world_size, 1) / normalizer,
            self.args.clip_norm
        )
        self.grad_scaler.step(self.optimizer)
        return grad_norm

    def step(self, batches: List):
        if not self.wps.started() and self.global_step > 0:
            self.wps.start()
        if not self.ups.started() and self.global_step > 0:
            self.ups.start()

        self.optimizer.zero_grad()
        log = {}
        oom = 0
        for i, batch in enumerate(batches):
            with self._ddp_sync(i, len(batches)):
                try:
                    loss, nll_loss = self.fb(batch)
                except RuntimeError as e:
                    if 'out of memory' in str(e) or 'CUDNN_STATUS_EXECUTION_FAILED' in str(e):
                        logger.warning(f'OOM @ ({i}/{len(batches)}) during forward/backward, trying to ignore.')
                        self.optimizer.zero_grad()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        oom = i + 1
                        log.clear()
                    else:
                        logger.error(str(e))
                        raise e
                else:
                    _merge_batch_stats(log, batch, loss, nll_loss)
        if not log:
            _merge_batch_stats(log, batches[0], 0.0, 0.0)
            _zero_log(log, ('num_samples', 'true_tokens_x', 'true_tokens_y', 'batch_tokens_x', 'batch_tokens_y'))

        if torch.cuda.is_available() and self.global_step == 0:
            torch.cuda.empty_cache()

        try:
            normalizer = float(log['num_samples'] if self.args.sentence_avg else log['true_tokens_y'])
        except KeyError as e:
            logger.error(f'Normalizer key not found in log, '
                         f'probably because the key is not specified in the input batch. '
                         f'Keys in log: {tuple(log.keys())}.')
            raise e

        log['normalizer'] = normalizer
        log['oom'] = oom
        log = utils.all_reduce_dict(log, sum)
        normalizer = log['normalizer']
        oom = log["oom"]
        if normalizer == 0:
            total_batches = self.args.accumulate * self.args.dist_world_size
            logger.warning(f'Normalizer is 0, optimisation step skipped. (oom={oom}/{total_batches})')
            log = None
        else:
            with torch.autograd.profiler.record_function('optimizer'):
                try:
                    grad_norm = self.update(normalizer)
                    self.wps.stop(log['batch_tokens_y'])
                    self.ups.stop(1)
                    log['gnorm'] = grad_norm.item()
                    log['wps'] = self.wps.avg
                    log['ups'] = self.ups.avg
                    self._overflow = 0
                except OverflowError:
                    logger.info(f'Overflow detected, skipping update. Reduced scale={self.grad_scaler.get_scale()}.')
                    self.optimizer.zero_grad()
                    self._overflow += 1
                    if self._overflow >= 10:
                        raise OverflowError
                    log = self.step(batches)
        return log

    def maybe_validate(self, validate_callback, end_of_epoch: bool = False, force: bool = False):
        if not utils.is_master():
            return None
        if not validate_callback:
            return None
        if self._last_validate_step == self.global_step:
            return None
        val_steps = self.args.val_steps
        val_epochs = self.args.val_epochs
        if end_of_epoch:
            if val_epochs > 0 and (self.epoch + 1) % val_epochs == 0:
                return self._validate(validate_callback)
        else:
            if val_steps > 0 and self.global_step % val_steps == 0:
                return self._validate(validate_callback)
        if force:
            return self._validate(validate_callback)
        return None

    def train(self, validate_callback: Callable[[], float] = None,
              before_epoch_callback: Callable[[], None] = None, after_epoch_callback: Callable[[], None] = None,
              summary_writer=None):
        logs = []
        max_epoch = self.args.max_epoch or sys.maxsize
        max_step = self.args.max_step or sys.maxsize
        terminated = False
        for _ in range(self.epoch, max_epoch):
            self._seed_by_epoch()
            if before_epoch_callback is not None:
                before_epoch_callback()
            self.lr_scheduler.step_epoch(self.epoch)
            epoch_itr = self.train_itr.next_epoch_itr()
            for batches in epoch_itr:
                self._seed_by_step()
                lr = self.lr_scheduler.step(self.global_step + 1)

                with torch.autograd.profiler.record_function(f'E{self.epoch}S{self.global_step}'):
                    log = self.step(batches)

                if not log or not log['loss']:
                    continue

                logs.append(log)
                global_step = self.inc_global_step()

                summary_writer.add_scalar('loss', log['loss'] / log['true_tokens_y'], global_step)
                summary_writer.add_scalar('gnorm', log['gnorm'], global_step)
                summary_writer.add_scalar('lr', lr, global_step)
                if global_step % self.args.log_interval == 0:
                    msg = self._compose_message(_merge_step_logs(logs), lr, epoch_itr)
                    logger.info(msg)
                    logs.clear()

                score = self.maybe_validate(validate_callback)
                if score is not None:
                    summary_writer.add_scalar(f'validate/score', score, global_step)
                self.saver.try_save(self.epoch, self.global_step, score)

                if self.global_step >= max_step:
                    terminated = True
                    break
            if terminated:
                break

            self.inc_epoch()
            if after_epoch_callback is not None:
                after_epoch_callback()

            score = self.maybe_validate(validate_callback, end_of_epoch=True)
            if score is not None:
                summary_writer.add_scalar(f'validate/score', score, self.global_step)
            self.saver.save_epoch(self.epoch, self.global_step, score)
        score = self.maybe_validate(validate_callback, force=True)
        if score is not None:
            summary_writer.add_scalar(f'validate/score', score, self.global_step)
        self.saver.try_save(self.epoch, self.global_step, score, force=True)

    def state_dict(self) -> dict:
        model = self.model
        if isinstance(self.model, DDP):
            model = self.model.module
        if self.args.checkpoint_shrink:
            state = {
                'args': self.args,
                'model': model.state_dict(),
                'vocabularies': self.vocabularies,
            }
        else:
            state = {
                'args': self.args,
                'model': model.state_dict(),
                'grad_scaler': self.grad_scaler.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'train_itr': self.train_itr.state_dict(),
                'vocabularies': self.vocabularies,
                'timer': self.timer,
                'global_step': self.global_step,
                'epoch': self.epoch,
            }
        return state

    def load_state_dict(self, state: dict, no_load_model=False) -> None:
        if not no_load_model:
            model = self.model
            if isinstance(self.model, DDP):
                model = self.model.module
            model.load_state_dict(state['model'])
        if 'grad_scaler' in state:
            self.grad_scaler.load_state_dict(state['grad_scaler'])
        if 'optimizer' in state and not self.args.reset_optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
        if 'lr_scheduler' in state and not self.args.reset_lr:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            self._global_step = state.get('global_step', self._global_step)
            self._epoch = state.get('epoch', self._epoch)
        if 'train_itr' in state and not self.args.reset_dataloader:
            self.train_itr.load_state_dict(state['train_itr'])
        if 'timer' in state:
            self.timer = state['timer']
        if self.args.reset_dataloader:
            self._epoch = 0

    def _ddp_sync(self, i, n):
        if hasattr(self.model, 'no_sync') and i < n - 1:
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    def _amp_autocast(self):
        if self.fp16 == 'amp':
            import torch.cuda.amp
            return torch.cuda.amp.autocast(self.fp16_enabled)
        else:
            return contextlib.ExitStack()

    def _validate(self, callback: Callable[[], float]):
        timer = utils.ElapsedTimeMeter()
        timer.start()
        score = callback()
        logger.info(f'Validation score @ {self.global_step}: {score:.4f}, '
                    f'took {datetime.timedelta(seconds=timer.elapse // 1)}')
        self._last_validate_step = self.global_step
        return score

    def _seed_by_epoch(self):
        utils.seed(self.args.seed + self.epoch)

    def _seed_by_step(self):
        utils.seed(self.args.seed + self.global_step)

    def _compose_message(self, log, lr, epoch_itr):
        loss = log['loss']
        nll_loss = log['nll_loss']
        ppl = log['ppl']
        gnorm = log['gnorm']
        num_sample = log['num_samples']
        true_tokens_x, true_tokens_y = log['true_tokens_x'], log['true_tokens_y']
        batch_tokens_x, batch_tokens_y = log['batch_tokens_x'], log['batch_tokens_y']
        wps, ups = log['wps'], log['ups']
        oom = log['oom']

        step_in_epoch = ''
        if self.train_itr.supports_len:
            epoch_size = len(epoch_itr)
            step_in_epoch = f'({epoch_itr.count}/{epoch_size})'

        msg = f'Epoch: {self.epoch + 1}{step_in_epoch}, ' \
              f'step={self.global_step}, ' \
              f'loss={loss:.3f}, ' \
              f'nll_loss={nll_loss:.3f}, ' \
              f'ppl={ppl:.3f}, ' \
              f'lr={lr:.4e}, ' \
              f'gnorm={gnorm:.2f}, ' \
              f'samples={num_sample}, ' \
              f'tokens=({true_tokens_x},{true_tokens_y})/({batch_tokens_x},{batch_tokens_y}), ' \
              f'wps={wps:.2f}, ' \
              f'ups={ups:.2f}, ' \
              f'oom={oom}/{self.args.dist_world_size * self.args.accumulate}'
        return msg
