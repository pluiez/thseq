import argparse
import copy
import functools
import itertools
import json
import logging
import os
from pathlib import Path
from typing import Union, Dict

import thseq.models as models
import thseq.optim as optim
import thseq.optim.lr_schedulers as lr_schedulers
import thseq.utils as utils

logger = logging.getLogger(__name__)
__all__ = ['parse_training_args', 'parse_generation_args']


def get_training_parser():
    parser = get_parser('Trainer', 'resolve')
    add_distributed_args(parser)
    add_dynamic_args(parser)
    add_dataset_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    add_validation_args(parser)

    return parser


def get_generation_parser():
    parser = get_parser('Decoding')
    add_decoding_args(parser)
    return parser


def get_parser(desc, conflict_handler='error'):
    parser = argparse.ArgumentParser(desc, conflict_handler=conflict_handler)
    parser.add_argument('--seed', type=int, default=9527,
                        help='Random seed. (default: 9527)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='Logs message every N steps. (default: 1)')

    return parser


def add_dataset_args(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument('--train-bin',
                       help='Directory containing binarized training data and vocab files.')
    group.add_argument('--train-bin-ordered', type=int,
                       help='Whether the binaries are ordered. '
                            'Enables sequential access to the binary buffer during batching.')
    group.add_argument('--train', nargs='+', default=['train'],
                       help='Filename prefixes of the training set. (default: [train])')
    group.add_argument('--vocab', nargs='+', default=['vocab'],
                       help='Filename prefixes of the vocabularies. (default: [vocab])')
    group.add_argument('--buffer-size', type=int, default=20)
    group.add_argument('--dev', nargs='+',
                       help='Full paths or path prefix to the development set. '
                            'When there are more than 2 arguments given, '
                            'remaining files starting from the second are used as references.')
    group.add_argument('--langs', nargs=2, help='Language suffixes')
    group.add_argument('--shuffle', type=int, default=1,
                       help='Whether to shuffle the dataset during training. '
                            'Additionally, this value also indicates shuffling buffer size for lunas (default: 10000)')
    group.add_argument('--num-workers', type=int, default=6,
                       help='Parallel workers for DataLoader. (default: 6)')
    group.add_argument('--min-length', type=int, default=1,
                       help='Drop samples with length lower than this value. (default: 1)')
    group.add_argument('--max-length', type=int, default=512,
                       help='Drop samples with length higher than this value. (default: 512)')


def add_validation_args(parser):
    group = parser.add_argument_group('Validation')
    group.add_argument('--val-steps', type=int, default=5000, help='(default: 5000)')
    group.add_argument('--val-epochs', type=int, default=1, help='(default: 1)')
    group.add_argument('--val-max-tokens', type=int, default=4096, help='(default: 4096)')
    group.add_argument('--val-method', choices=['bleu', 'logp'], default='bleu', help='(default: bleu)')
    group.add_argument('--patience', type=int)


def add_dynamic_args(parser):
    # 1. model
    model = parser.add_argument_group('Model')

    model.add_argument(
        '--model', '-m', metavar='MODEL',
        choices=list(models.MODELS.keys()),
        help=f'Available model architectures: {", ".join(models.MODELS.keys())}',
        default='transformer'
    )

    model.add_argument('--configs', type=str, nargs='+',
                       help='Support multiple config files, the priority '
                            'is determined by their orders from low to high.')

    # 2. optimizer
    optimizer = parser.add_argument_group('Optimizer')
    optimizer.add_argument(
        '--opt', '--optimizer', '-o',
        dest='optimizer',
        metavar='OPTIMIZER',
        choices=list(optim.OPTIMIZERS.keys()),
        help=f'Available optimizers: {", ".join(optim.OPTIMIZERS.keys())}',
        default='adam'
    )

    # 3. lr-scheduler
    scheduler = parser.add_argument_group('LR Scheduler')
    scheduler.add_argument(
        '--lr-scheduler', metavar='LR_SCHEDULER',
        choices=list(lr_schedulers.SCHEDULERS.keys()),
        help=f'Available lr schedulers: {", ".join(lr_schedulers.SCHEDULERS.keys())}',
        default='inverse_sqrt'
    )

    return model


def add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument('--fp16', type=str, choices=['amp', 'half', 'none'],
                       help='Enables mixed-precision training. (default: disabled)')
    group.add_argument('--fp16-flatten', type=int, default=1,
                       help='Enables fp16 flattening. (default: 1)')
    group.add_argument('--max-epoch', type=int, metavar='N',
                       help='Train at most N epochs before termination. '
                            'When both --max-epoch and --max-step are presented,'
                            'the minimum requirement will be determined to terminate training procedure.'
                            '(default: None)')
    group.add_argument('--max-step', '--mu', type=int, metavar='N', default=100000,
                       help='Train at most N steps before termination. '
                            'When both --max-epoch and --max-step are presented,'
                            'the minimum requirement will be determined to terminate training procedure.'
                            '(default: 100000)')
    group.add_argument('--clip-norm', type=float, metavar='NORM', default=0.0,
                       help='Scale gradient norm to be no more NORM. (default: 0.0)')
    group.add_argument('--sentence-avg', type=int, default=0,
                       help='Normalize gradients by the number of sentences. (default: 0)')
    group.add_argument('--normalize-before-reduce', type=int, default=0,
                       help='Normalize gradients before all-reduce. (defualt: 0)')
    group.add_argument('--max-tokens', type=int, metavar='N', default=4096,
                       help='Constrain the number of tokens in a batch. (default: 4096)')
    group.add_argument('--max-sentences', type=int, metavar='BATCH_SIZE',
                       help='Constrain the number of samples in a btach. (default: None)')
    group.add_argument('--accumulate', type=int, metavar='N', default=1,
                       help='Accumulate gradients of N batches for one parameter update. (default: 1)')
    group.add_argument('--lr', '--learning-rate', type=float, default=1.0,
                       help='Learning rate. (default: 1.0)')
    group.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Smoothing value for label-smoothed cross entropy. (default is 0.1)')
    group.add_argument('--momentum', type=float,
                       help='momentum factor. (default: None)')
    group.add_argument('--weight-decay', '--wd', type=float, default=0,
                       help='Weight decay (different from L2-norm regularisation). (default is 0)')
    group.add_argument('--use-weissi', type=int, default=0,
                       help='See: "Improve Generalization and Robustness of Neural Networks '
                            'via Weight Scale Shifting Invariant Regularization"')
    group.add_argument('--cos-reg', type=float, default=0,
                       help='Cosine regularization on target embedding')
    group.add_argument('--min-lr-bound', type=float, default=0.0,
                       help='Set lr as min_lr when scheduler attempts to set a value lower than min_lr.')
    group.add_argument('--finetune', metavar='FT', type=str,
                       help='Initialize model parameters from checkpoint FT.')
    group.add_argument('--finetune-filters', type=str, nargs='+',
                       help='If provided, fine-tune parameters matching given regex filters. (default: None)')
    group.add_argument('--reset-lr', type=int, default=0,
                       help='Reset lr. (default: 0)')
    group.add_argument('--reset-optimizer', type=int, default=0,
                       help='Reset optimizer state. (default: 0)')
    group.add_argument('--reset-dataloader', type=int, default=0,
                       help='Reset dataloader. (default: 0)')
    group.add_argument('--profile', type=int,
                       help='Enables profiling including nan detect and so on.')
    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group('Checkpoint')
    group.add_argument('--checkpoint', required=True,
                       help='A directory to save checkpoints. '
                            'Training will resume from latest checkpoint in the directory.')
    group.add_argument('--save-checkpoint-secs', metavar='N', type=int,
                       help='Saves checkpoint every N seconds. A "0" disables checkpointing. (default: None)')
    group.add_argument('--save-checkpoint-steps', metavar='N', type=int, default=5000,
                       help='Saves checkpoint every N steps. A "0" disables checkpointing. (default: 5000)')
    group.add_argument('--save-checkpoint-epochs', metavar='N', type=int, default=1,
                       help='Saves checkpoint every N epochs. A "0" disables checkpointing. (default: 1)')
    group.add_argument('--keep-checkpoint-max', metavar='N', type=int, default=10,
                       help='Retains at most N latest checkpoints in terms of steps or seconds. '
                            'A "0" keeps all checkpoints. (default: 10)')
    group.add_argument('--keep-epoch-checkpoint-max', metavar='N', type=int, default=5,
                       help='Retains at most N latest checkpoints in terms of epochs. '
                            'A "0" disables checkpointing. (default: 5)')
    group.add_argument('--keep-best-checkpoint-max', metavar='N', type=int, default=1,
                       help='Retains at most N best checkpoints in terms of validation score on the dev set. '
                            'A "0" keeps all checkpoints. (default: 1)')
    group.add_argument('--checkpoint-shrink', type=int, default=0,
                       help='Minimize checkpoint storage to reduce disk usage. (default: 0)')


def add_decoding_args(parser):
    group = parser.add_argument_group('Decoding')
    group.add_argument('input', help='Input file. Use "-" to read from standard input '
                                     'Note: --num-workers is required to be 0 explicitly '
                                     'when read from standard input.')
    group.add_argument('--checkpoints', required=True, nargs='+',
                       help='Either model directories or specific checkpoint files.')
    group.add_argument('-k', type=int, default=4,
                       help='Beam width. (default: 4)')
    group.add_argument('--topk', type=int, default=1,
                       help='List top k outputs and scores. (default: 1)')
    group.add_argument('--penalty', choices=['gnmt', 'fairseq'], default='fairseq',
                       help='Choose length penalty to use. (default: fairseq)')
    group.add_argument('--alpha', type=float, default=1.0,
                       help='Length penalty parameter. Set 0 to disable length penalisation. (default: 1.0)')
    group.add_argument('--step-wise-penalty', action='store_true',
                       help='Enable step-wise length penalty, otherwise apply penalisation on sequence-level.')
    group.add_argument('--min-len-a', type=int, default=0, help='(default: 0)')
    group.add_argument('--min-len-b', type=int, default=1, help='(default: 1)')
    group.add_argument('--max-len-a', type=int, default=0, help='(default: 0)')
    group.add_argument('--max-len-b', type=int, default=200, help='(default: 200)')
    group.add_argument('--max-tokens', type=int, default=0, help='(default: 0)')
    group.add_argument('--max-sentences', type=int, default=200, help='(default: 200)')
    group.add_argument('--buffer-size', type=int, default=1024, help='(default: 1024)')
    group.add_argument('--num-workers', type=int, default=0, help='(default: 0)')
    group.add_argument('--bpe', action='store_true', help='Enable post-processing with BPE.')
    group.add_argument('--sp', action='store_true', help='Enable post-processing with sentence-piece.')
    group.add_argument('--reverse', action='store_true',
                       help='Enable post-processing by reversing output tokens before BPE/sentence-piece restoration.')
    group.add_argument('--verbose', '-v', action='store_true')


def add_scoring_args(parser):
    group = parser.add_argument_group('Scoring')
    group.add_argument('source')
    group.add_argument('target')
    group.add_argument('--checkpoint', required=True,
                       help='Either model directories or a specific checkpoint file.')
    group.add_argument('--max-tokens', type=int, default=1024, help='(default: 1024)')
    group.add_argument('--max-sentences', type=int, default=0, help='(default: 0)')
    group.add_argument('--num-workers', type=int, default=0, help='(default: 0)')


def add_distributed_args(parser):
    group = parser.add_argument_group('distributed')
    group.add_argument("--dist-world-size", type=int, default=1,
                       help='Distributed world size. (default: 1)')
    group.add_argument("--dist-start-rank", type=int, default=0,
                       help='Distributed starting rank for current host. (default: 0)')
    group.add_argument("--dist-backend", type=str, default='nccl',
                       help='Distributed backend.')
    group.add_argument("--dist-init-method", type=str, default='tcp://localhost:9527',
                       help='Distributed protocol')


def get_defaults(parser: argparse.ArgumentParser, exclude=None) -> argparse.Namespace:
    if exclude is None:
        exclude = []
    exclude += ['help']
    exclude = set(exclude)

    args = argparse.Namespace()
    for action in parser._actions:
        if action.dest not in exclude:
            args.__dict__[action.dest] = action.default
    return args


def parser_clear_default(parser: argparse.ArgumentParser, excludes=None) -> argparse.ArgumentParser:
    parser = copy.deepcopy(parser)
    if excludes is None:
        excludes = []

    for action in parser._actions:
        if action.dest not in excludes:
            action.default = None
    return parser


def override(args_x: Union[argparse.Namespace, Dict], args_y: Union[argparse.Namespace, Dict]) -> argparse.Namespace:
    if args_x and not isinstance(args_x, argparse.Namespace):
        args = argparse.Namespace()
        args.__dict__.update(args_x)
        args_x = args
    if args_y and not isinstance(args_y, argparse.Namespace):
        args = argparse.Namespace()
        args.__dict__.update(args_y)
        args_y = args

    if not args_y:
        return args_x
    if not args_x:
        return args_y

    for k, v in args_y.__dict__.items():
        if v is None and k in args_x.__dict__:
            continue
        setattr(args_x, k, v)
    return args_x


def _add_dynamic_group(args, parser, group_name, desc, add_args_fn):
    val = getattr(args, group_name, None)
    if val:
        group = parser.add_argument_group(desc)
        add_args_fn(val, group)


def _parse_dynamic_args(parser):
    args, unk_args = parser.parse_known_args()
    group = parser.add_argument_group('awdasd')
    models.MODELS['transformer'].add_args(group)
    groups = [
        {
            'group_name': 'model',
            'desc': 'Model-specific options',
            'add_args_fn': lambda val, parser: models.MODELS[val].add_args(parser)
        },
        {
            'group_name': 'optimizer',
            'desc': 'Optimizer-specific options',
            'add_args_fn': lambda val, parser: optim.OPTIMIZERS[val].add_args(parser)
        },
        {
            'group_name': 'lr_scheduler',
            'desc': 'LR-scheduler-specific options',
            'add_args_fn': lambda val, parser: lr_schedulers.SCHEDULERS[val].add_args(parser)
        }
    ]

    for g in groups:
        _add_dynamic_group(args, parser, **g)

    cli_args = parser_clear_default(parser).parse_args(None)
    default_args = get_defaults(parser)
    return cli_args, default_args


def parse_training_args() -> argparse.Namespace:
    """
    Parse training args and optionally create checkpoint directory.
    :return:
    """
    parser = get_training_parser()
    cli_args, default_args = _parse_dynamic_args(parser)

    checkpoint_dir = Path(cli_args.checkpoint)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # load config args
    config_args = []
    if cli_args.configs:
        cli_args.configs = list(map(lambda f: json.loads(Path(f).read_text()), cli_args.configs))
    # load checkpoint args
    state_dict = utils.load_latest(checkpoint_dir)

    checkpoint_args = state_dict['args'] if state_dict else None
    ft_checkpoint_args = None
    if cli_args.finetune and Path(cli_args.finetune).exists():
        state_dict = utils.load_checkpoint(cli_args.finetune)
        ft_checkpoint_args = state_dict['args'] if state_dict else None
    if ft_checkpoint_args and checkpoint_args:
        raise ValueError(f'fine-tuning is not available while trying to resume training from an existing checkpoint.')
    checkpoint_args = checkpoint_args or ft_checkpoint_args
    args = functools.reduce(override, [default_args] + config_args + [checkpoint_args, cli_args])

    invalid_args = [k for k in args.__dict__.keys() if k not in default_args.__dict__.keys()]

    for k in invalid_args:
        delattr(args, k)

    def _expand(names):
        if not isinstance(names, list):
            raise ValueError(f'Unexpected values: {names}')

        if len(names) == 1 and args.langs and len(args.langs) > 1:
            prefix = names[0]
            return [f'{prefix}.{lang}' for lang in args.langs]
        return names

    args.train = _expand(args.train)
    args.vocab = _expand(args.vocab)
    if args.dev:
        args.dev = _expand(args.dev)

    for file in itertools.chain(
            [args.train_bin],
            [Path(args.train_bin) / name for name in args.vocab],
            args.dev or []
    ):
        if not Path(file).exists():
            raise FileNotFoundError(file)

    if args.val_method == 'logp':
        args.val_max_tokens = max(args.val_max_tokens, args.max_tokens)

    if not (args.max_length <= 1024):
        raise ValueError(f'Expected max_length <= 1024, got {args.max_length}')
    if not (args.min_length < args.max_length):
        raise ValueError(f'Expected min_length <= max_length, got ({args.min_length}) and ({args.max_length})')
    return args


def parse_generation_args() -> argparse.Namespace:
    args = get_generation_parser().parse_args()
    if args.max_tokens > 0 and args.max_sentences > 0:
        raise ValueError(f'max_tokens ({args.max_tokens}) and max_sentences ({args.max_sentences}) '
                         f'are exclusive options, please ensure only one option is enabled.')
    if args.input == '-' and args.num_workers != 0:
        raise ValueError(
            f'Reading from standard input requires explicitly specifying --num-workers 0, '
            f'got --num-workers {args.num_workers} instead.'
        )
    return args


def add_default_model_args(args):
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    models.MODELS[args.model].add_args(parser)
    defaults = parser.parse_args([])
    override(defaults, args)
    return defaults
