import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List

import torch

from .distributed import enable_on_master

__all__ = ['list_checkpoints', 'load_checkpoint', 'load_best', 'load_latest', 'Saver']

_META_CHECKPOINT_PATH = 'checkpoints.json'

logger = logging.getLogger(__name__)


def _format_name(by_epoch, by_global_step, by_time, epoch, global_step):
    if by_epoch:
        return f'ckp.E{epoch:04d}.S{global_step:08d}.pt'
    elif by_global_step:
        return f'ckp.S{global_step:08d}.pt'
    else:
        timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
        return f'ckp.T{timestamp}.S{global_step:08d}.pt'


class Checkpoint(object):

    def __init__(self, filename: str, global_step: int, epoch: int, score: float = None,
                 is_periodic=False, timestamp: float = None, end_of_epoch=None) -> None:
        super().__init__()
        self.filename = filename
        self.global_step = global_step
        self.epoch = epoch
        self.score = score
        self.is_periodic = is_periodic
        self.timestamp = timestamp or time.time()
        self.end_of_epoch = end_of_epoch

    def __hash__(self) -> int:
        return self.global_step


def list_checkpoints(dirname) -> List[Checkpoint]:
    ckps = []
    meta = Path(dirname, _META_CHECKPOINT_PATH)
    if meta.exists():
        with meta.open() as r:
            ckps = [Checkpoint(**attrs) for attrs in json.loads(r.read())]
    return ckps


def load_checkpoint(path, map_location='cpu'):
    return torch.load(path, map_location)


def load_best(dirname):
    ckps = list_checkpoints(dirname)
    ckps = list(filter(lambda c: c.score is not None, ckps))
    ckps.sort(key=lambda c: c.score)
    if ckps:
        return load_checkpoint(Path(dirname) / ckps[-1].filename)
    return {}


def load_latest(dirname):
    ckps = list_checkpoints(dirname)
    ckps.sort(key=lambda c: c.global_step)
    if ckps:
        return load_checkpoint(Path(dirname) / ckps[-1].filename)
    return {}


class Saver(object):
    def __init__(
            self, save_dir,
            save_checkpoints_secs, save_checkpoints_steps, save_checkpoint_epochs,
            keep_checkpoint_max, keep_epoch_checkpoint_max, keep_best_checkpoint_max,
            get_state_fn
    ):
        assert callable(get_state_fn), type(get_state_fn)
        assert keep_epoch_checkpoint_max >= 0
        assert keep_checkpoint_max >= 0
        assert keep_best_checkpoint_max >= 0
        self._best = [None, None]
        checkpoints = []
        if Path(save_dir, _META_CHECKPOINT_PATH).exists():
            checkpoints = list_checkpoints(save_dir)
            best = max([c for c in checkpoints if c.score], key=lambda c: c.score, default=None)
            if best:
                self._best = [(best.global_step, best.epoch), best.score]
        self._tic = time.time()

        self._save_dir = save_dir
        self._save_checkpoints_secs = save_checkpoints_secs
        self._save_checkpoints_steps = save_checkpoints_steps
        self._save_checkpoint_epochs = save_checkpoint_epochs
        self._keep_checkpoint_max = keep_checkpoint_max or int(1e10)
        self._keep_epoch_checkpoint_max = keep_epoch_checkpoint_max
        self._keep_best_checkpoint_max = keep_best_checkpoint_max or int(1e10)
        self._checkpoints: List[Checkpoint] = checkpoints
        self._get_state_fn = get_state_fn
        self._max_global_step = max(checkpoints, key=lambda c: c.global_step).global_step if checkpoints else 0

    @property
    def best_at(self):
        return self._best[0]

    @property
    def best_score(self):
        return self._best[1]

    def _update_best_score(self, score, global_step, epoch):
        save = False
        if score is None:
            return save
        bests = [c for c in self._checkpoints if c.score is not None]
        scores = [c.score for c in bests] + [score]
        scores.sort(reverse=True)
        if score in scores[:self._keep_best_checkpoint_max]:
            save = True
        if score == scores[0]:
            self._best = [(global_step, epoch), score]
        return save

    @enable_on_master()
    def try_save(self, epoch: int, global_step: int, eval_score: float = None, force: bool = False):
        if global_step < self._max_global_step:
            raise RuntimeError(f'Expected global_step ({global_step}) greater than or equal to '
                               f'maximum global step ({self._max_global_step}) seen in saved checkpoints so far.')
        toc = time.time()

        save_checkpoints_secs = self._save_checkpoints_secs
        save_checkpoints_steps = self._save_checkpoints_steps
        save, by_global_step, by_time = False, False, False
        if save_checkpoints_steps and save_checkpoints_steps > 0 and global_step % save_checkpoints_steps == 0:
            save = True
            by_global_step = True
        elif save_checkpoints_secs is not None and 0 < save_checkpoints_secs <= (toc - self._tic):
            self._tic = toc
            save = True
            by_time = True

        save = self._update_best_score(eval_score, global_step, epoch) or save or force
        is_periodic = by_global_step or by_time or force

        if save:
            if by_time:
                name = _format_name(False, False, True, epoch, global_step)
            else:
                name = _format_name(False, True, False, epoch, global_step)
            ckp = Checkpoint(name, global_step, epoch, eval_score, is_periodic)

            success = self._persist(self._get_state_fn(), ckp)

            if success:
                if eval_score:
                    logger.info(f'Saved validation checkpoint: {ckp.filename}')
                else:
                    logger.info(f'Saved checkpoint: {ckp.filename}')

    @enable_on_master()
    def save_epoch(self, epoch: int, global_step: int, eval_score: float = None):
        if self._save_checkpoint_epochs < 1 or epoch % self._save_checkpoint_epochs != 0:
            return
        name = _format_name(True, False, False, epoch, global_step)
        self._update_best_score(eval_score, global_step, epoch)
        ckp = Checkpoint(name, global_step, epoch, eval_score, True, end_of_epoch=True)
        self._persist(self._get_state_fn(), ckp)

    def _persist(self, state, checkpoint: Checkpoint):
        dup = [c for c in self._checkpoints if c.global_step == checkpoint.global_step]

        assert len(dup) <= 1

        dup = dup[0] if dup else None
        if dup is not None:
            if dup.score is not None and checkpoint.score is None:
                checkpoint.score = dup.score

        # save checkpoint
        target = Path(self._save_dir, checkpoint.filename)
        if not target.parent.exists():
            os.makedirs(str(target.parent), exist_ok=True)

        fd, name = tempfile.mkstemp(dir=self._save_dir)
        torch.save(state, name)
        os.close(fd)
        shutil.move(name, target)
        self._checkpoints.append(checkpoint)

        # remove checkpoints
        if dup is not None:
            self._checkpoints.remove(dup)
        discard = []

        evals = sorted([c for c in self._checkpoints if c.score is not None], key=lambda c: c.score)
        periods = sorted(
            [c for c in self._checkpoints if c.is_periodic and not c.end_of_epoch],
            key=lambda c: c.global_step
        )
        epochs = sorted(
            [c for c in self._checkpoints if c.is_periodic and c.end_of_epoch],
            key=lambda c: c.global_step
        )

        retains = set(
            evals[-self._keep_best_checkpoint_max:] +
            periods[-self._keep_checkpoint_max:] +
            epochs[-self._keep_epoch_checkpoint_max:]
        )
        retains = sorted(retains, key=lambda c: c.global_step)

        discard += [c for c in self._checkpoints if c not in retains]
        discard = list(set(discard))

        for c in discard:
            path = Path(self._save_dir, c.filename)
            if path.exists():
                path.unlink()

        self._checkpoints = retains
        # save meta statistics
        with Path(self._save_dir, _META_CHECKPOINT_PATH).open('w') as w:
            w.write(json.dumps(
                [c.__dict__ for c in retains],
                indent=4,
                sort_keys=True)
            )
        return True
