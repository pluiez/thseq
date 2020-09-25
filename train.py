import contextlib
import datetime
import json
import logging
import re
import socket
import sys
import time
from pathlib import Path
from typing import *

import lunas
import numpy as np
import torch.distributed as dist
import torch.multiprocessing
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import thseq.data.iterators as iterators
import thseq.models as models
import thseq.modules.losses as losses
import thseq.optim as optim
import thseq.optim.lr_schedulers as lr_schedulers
import thseq.options as options
import thseq.utils as utils
from thseq.data import Vocabulary
from thseq.trainer import Trainer as Trainer

logging.root.handlers = []
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s ', level=logging.DEBUG,
                    stream=sys.stderr)

logger = logging.getLogger('train')


def get_validator(args, model, sv, tv, beam_size=1):
    @torch.no_grad()
    def validate_bleu(model, data_loader, beam_size: int = 1):
        if not utils.is_master():
            return None

        model.eval()
        with model.no_sync() if isinstance(model, DDP) else contextlib.ExitStack():
            hyps = []
            refs = []

            infer = model.module.infer if isinstance(model, DDP) else model.infer

            for i, batch in enumerate(data_loader):
                batch = utils.move_cuda(batch)
                batch_hyps = infer(batch['x'], beam_size)
                batch_hyps = [k_best[0] for k_best in batch_hyps]

                hyps.extend([h['tokens'] for h in batch_hyps])
                refs.extend(batch['refs'])

            results = {'hyp': hyps, 'ref': refs}
            bleu = utils.bleu(results['ref'], results['hyp'])
            return bleu

    @torch.no_grad()
    def validate_logp(model, data_loader):
        if not utils.is_master():
            return None

        model.eval()
        num_tokens = 0
        total_loss = 0
        with model.no_sync() if isinstance(model, DDP) else contextlib.ExitStack():
            for i, batch in enumerate(data_loader):
                batch = utils.move_cuda(batch)
                logits = model(batch)
                nll_loss = F.cross_entropy(logits.transpose(2, 1).float(), batch['y'], reduction='sum')
                total_loss += nll_loss.float().item()
                num_tokens += batch['true_tokens_y']
        return -total_loss / num_tokens

    if args.val_method == 'bleu':
        dev_itr = get_dev_iterator_bleu(args, sv, tv)
        return lambda: validate_bleu(model, dev_itr, beam_size)
    elif args.val_method == 'logp':
        dev_itr = get_dev_iterator_logp(args, sv, tv)
        return lambda: validate_logp(model, dev_itr)
    else:
        raise NotImplementedError(f'validation method not implemented: {args.val_method}')


def check_devices(dist_world_size, require_cuda=None):
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError('CUDA is not available')
        else:
            logger.warning('Training on CPU')
    else:
        logger.info(f'Training on {dist_world_size} GPUs')


def stat_parameters(model):
    num_params = 0
    num_params_trainable = 0
    for n, p in model.named_parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    logger.info(f'Parameters: {num_params} (trainable: {num_params_trainable})')


class Text2Id(object):

    def __init__(self, vocab) -> None:
        super().__init__()
        self.vocab = vocab

    def __call__(self, line: str):
        return self.vocab.lookup(line.split(), np.int64)


class SourceTargetZip(object):
    def __call__(self, x, y):
        return {
            'x': x,
            'y': y,
            'num_tokens_x': x.size,
            'num_tokens_y': y.size,
            'num_tokens': max(x.size, y.size)
        }


class SampleSize(object):
    def __call__(self, d):
        return d['num_tokens']


class DictCollator(object):

    def __init__(self, sv: Vocabulary, tv: Vocabulary) -> None:
        super().__init__()
        self._source_pad = sv.pad_id
        self._target_pad = tv.pad_id

    def __call__(self, samples: List):
        x = utils.pack(utils.aggregate_values(samples, 'x'), self._source_pad, torch.long)
        y = utils.pack(utils.aggregate_values(samples, 'y'), self._target_pad, torch.long)
        return {
            'x': x,
            'y': y,
            'true_tokens_x': utils.aggregate_values(samples, 'num_tokens_x', reduce=sum),
            'true_tokens_y': utils.aggregate_values(samples, 'num_tokens_y', reduce=sum),
            'batch_tokens_x': x.shape[0] * x.shape[1],
            'batch_tokens_y': y.shape[0] * y.shape[1],
            'num_samples': len(samples)
        }


class MMapCollator(object):

    def __init__(self, sv: Vocabulary, tv: Vocabulary) -> None:
        super().__init__()
        self._source_pad = sv.pad_id
        self._target_pad = tv.pad_id

    def __call__(self, samples):
        xs, ys = list(zip(*samples))
        x = utils.pack(xs, self._source_pad, torch.long)
        y = utils.pack(ys, self._target_pad, torch.long)
        return {
            'x': x,
            'y': y,
            'true_tokens_x': sum(x.numel() for x in xs),
            'true_tokens_y': sum(y.numel() for y in ys),
            'batch_tokens_x': x.shape[0] * x.shape[1],
            'batch_tokens_y': y.shape[0] * y.shape[1],
            'num_samples': len(samples)
        }


def get_train_iterator_bucketing(args, num_shards, shard_idx, sv: Vocabulary, tv: Vocabulary):
    ds_x = lunas.TextLine(args.train[0]).map(Text2Id(sv))
    ds_y = lunas.TextLine(args.train[1]).map(Text2Id(tv))
    ds = lunas.Zip([ds_x, ds_y]).map(SourceTargetZip(), unpack_args=True)

    if args.shuffle:
        ds = ds.shuffle(args.shuffle)
    if args.max_tokens:
        itr = lunas.BucketIterator(
            ds,
            args.max_tokens * args.dist_world_size,
            SampleSize(),
            lunas.get_bucket_boundaries(1, 8, 8, args.max_length),
            min_length=args.min_length,
            max_length=args.max_length,
            required_batch_size_multiple=8
        )
    else:
        itr = lunas.ConstantIterator(ds, args.max_sentences * args.dist_world_size)

    if num_shards > 1:
        itr = lunas.ShardedIterator(itr, num_shards, shard_idx)

    data_loader = lunas.DataLoader(itr, args.num_workers, collate_fn=DictCollator(sv, tv),
                                   pin_memory=False)
    return data_loader


def get_dev_iterator_bleu(args, sv, tv):
    datasets = [lunas.TextLine(f) for f in args.dev]
    src, refs = datasets[0], datasets[1:]

    src = src.map(Text2Id(sv))
    refs = lunas.Zip(refs).map(lambda *ys: [y.split() for y in ys], unpack_args=True)

    ds = lunas.Zip([src, refs]).map(
        lambda x, ys: {
            'x': x,
            'size_x': x.size,
            'refs': ys
        }, unpack_args=True
    )

    def collate_dev(samples: List):
        return {
            'x': utils.pack(utils.aggregate_values(samples, 'x'), sv.pad_id, torch.long),
            'size_x': utils.aggregate_values(samples, 'size_x', reduce=sum),
            'refs': utils.aggregate_values(samples, 'refs'),
        }

    ds = ds.sort(len(ds), key=lambda x: -x['size_x'])

    itr = lunas.BucketIterator(
        ds,
        args.val_max_tokens,
        lambda x: x['size_x'],
        lunas.get_bucket_boundaries(1, 8, 8, args.max_length),
        required_batch_size_multiple=8
    )

    return lunas.DataLoader(itr, 0, collate_fn=collate_dev)


def get_dev_iterator_logp(args, sv, tv):
    src = lunas.TextLine(args.dev[0]).map(Text2Id(sv))
    ref = lunas.TextLine(args.dev[1]).map(Text2Id(tv))

    ds = lunas.Zip([src, ref]).map(SourceTargetZip(), unpack_args=True)
    ds = ds.sort(len(ds), key=lambda x: x['num_tokens'])
    itr = lunas.BucketIterator(
        ds,
        args.max_tokens,
        lambda x: x['num_tokens'],
        lunas.get_bucket_boundaries(1, 8, 8, args.max_length),
        min_length=args.min_length,
        max_length=args.max_length,
        required_batch_size_multiple=8
    )

    data_loader = lunas.DataLoader(itr, 0, collate_fn=DictCollator(sv, tv))

    return data_loader


def load_vocab(args) -> List[Vocabulary]:
    paths = [Path(args.train_bin) / Path(filename).name for filename in args.vocab]
    vocabularies = [Vocabulary(path) for path in paths]
    return vocabularies


def build_train_itr(args, svoc, tvoc) -> iterators.EpochIterator:
    shuffle = args.shuffle
    num_shards = max(1, args.dist_world_size)
    shard_idx = 0 if num_shards <= 1 else dist.get_rank()
    kwargs = {
        'num_shards': num_shards,
        'shard_index': shard_idx,
        'chunk_size': args.accumulate,
        'num_workers': args.num_workers,
        'buffer_size': args.buffer_size,
    }
    logger.info(f'Loading binary data from directory: {args.train_bin}')
    train_itr = iterators.MMapEpochIterator(
        args.train_bin,
        args.train,
        bool(args.train_bin_ordered),
        args.max_tokens,
        args.max_sentences,
        shuffle=bool(shuffle),
        collate=MMapCollator(svoc, tvoc),
        min_length=args.min_length,
        max_length=args.max_length,
        **kwargs
    )
    return train_itr


def try_prepare_finetune(args, model):
    state_dict = None
    if args.finetune:
        state_dict = utils.checkpoint.load_checkpoint(args.finetune)
        incompatible_keys = model.load_state_dict(state_dict['model'], False)
        if incompatible_keys.missing_keys:
            logger.warning(f'Missing keys: {incompatible_keys.missing_keys}')
        if incompatible_keys.unexpected_keys:
            logger.warning(f'Unexpected keys: {incompatible_keys.unexpected_keys}')

    if args.finetune_filters:
        keys = []
        for k, p in model.named_parameters():
            requires_grad = False
            for pattern in args.finetune_filters:
                if re.match(pattern, k):
                    requires_grad = True
                    keys.append(k)
                    break
            p.requires_grad = requires_grad
        logger.info(f'Fine-tuning parameters: {keys}')
    if state_dict:
        args.reset_dataloader = True
    return state_dict


def run(args):
    disable_numba_logging()
    if utils.is_master():
        with Path(args.checkpoint, 'config.json').open('w') as w:
            w.write(json.dumps(args.__dict__, indent=4, sort_keys=True))
    utils.seed(args.seed)
    sv, tv = load_vocab(args)

    logger.info('Building model')
    model = utils.move_cuda(models.build(args, [sv, tv]))
    criterion = utils.move_cuda(losses.CrossEntropyLoss(args.label_smoothing, ignore_index=tv.pad_id))
    optimizer = utils.move_cuda(optim.build(args, model.parameters()))
    lr_scheduler = lr_schedulers.build(args, args.lr, optimizer)

    state_dict = utils.checkpoint.load_latest(args.checkpoint)

    if not state_dict:
        logger.info(f'Model: \n{model}')
    elif args.finetune:
        raise ValueError(f'fine-tuning is not available while trying to resume training from an existing checkpoint.')

    logger.info(f'FP16: {args.fp16}')
    if args.fp16 and args.fp16 != 'none':
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 7:
            logger.warning('Target device does not support acceleration with --fp16')
        if args.fp16 == 'half':
            model = model.half()
            criterion = criterion.half()

    train_itr = build_train_itr(args, sv, tv)
    trainer = Trainer(args, model, criterion, [sv, tv], optimizer, lr_scheduler, train_itr)

    logger.info(f'Max sentences = {args.max_sentences}, '
                f'max tokens = {args.max_tokens} ')

    state_dict = state_dict or try_prepare_finetune(args, model)

    stat_parameters(model)

    # Restore training process
    if state_dict:
        logger.info('Resuming from given checkpoint')
        trainer.load_state_dict(state_dict, no_load_model=bool(args.finetune))
        del state_dict

    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            find_unused_parameters=True
        )
        trainer.model = model

    check_devices(args.dist_world_size)

    def after_epoch_callback():
        logger.info(f'Finished epoch {trainer.epoch}. ')

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning('Tensorboard is not available.')
        SummaryWriter = utils.DummySummaryWriter

    if not utils.is_master():
        SummaryWriter = utils.DummySummaryWriter

    writer = SummaryWriter(log_dir=Path(args.checkpoint, 'tensorboard'),
                           purge_step=trainer.global_step or None)

    validate_callback = None
    if args.dev:
        validate_callback = get_validator(args, model, sv, tv)
    trainer.timer.start()

    with writer:
        with torch.autograd.profiler.record_function('train_loop'):
            trainer.train(validate_callback=validate_callback,
                          before_epoch_callback=None,
                          after_epoch_callback=after_epoch_callback,
                          summary_writer=writer)

    logger.info(f'Training finished @ {time.strftime("%b %d, %Y, %H:%M:%S", time.localtime())}, '
                f'took {datetime.timedelta(seconds=trainer.elapse // 1)}')

    logger.info(f'Best validation score: {trainer.best_score}, @ {trainer.best_at}')


def init_process_group(local_rank, args):
    start_rank = args.dist_start_rank
    dist_local_rank = args.dist_start_rank + local_rank
    dist.init_process_group(
        args.dist_backend,
        args.dist_init_method,
        rank=start_rank + dist_local_rank,
        world_size=args.dist_world_size
    )
    torch.cuda.set_device(local_rank)
    if not utils.is_master():
        logging.disable()

    workers = utils.all_gather_list([dist.get_rank(), socket.gethostname()])
    workers.sort(key=lambda info: info[0])
    for rank, host in workers:
        logger.info(f'Initialized host {host} as rank {rank}')
    run(args)


def check_address(unix_address: str, try_free_port: bool = True):
    scheme, host, port = re.findall(r'(.+)://(.+):(\d+)', unix_address)[0]
    port = int(port)
    if scheme != 'tcp':
        return unix_address

    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        in_use = s.connect_ex((host, port)) == 0

    if in_use:
        if try_free_port:
            logger.warning(f'Address already in use: {unix_address}')
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.bind(('', 0))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                free_port = s.getsockname()[1]
            logger.warning(f'Found alternative port: {free_port}')
            return f'{scheme}://{host}:{free_port}'
        else:
            raise RuntimeError(f'Address already in use: {unix_address}')
    else:
        return unix_address

    dist.is_available()
    dist.is_nccl_available()


def main(args):
    if args.dist_world_size > 1:
        if torch.cuda.device_count() >= args.dist_world_size:
            args.dist_init_method = check_address(args.dist_init_method)
        nprocs = min(torch.cuda.device_count(), args.dist_world_size)
        torch.multiprocessing.spawn(init_process_group, args=(args,), nprocs=nprocs)
    else:
        run(args)


def disable_numba_logging():
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)


if __name__ == '__main__':
    args = options.parse_training_args()

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                main(args)
    else:
        main(args)
