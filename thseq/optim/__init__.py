from pathlib import Path

import thseq.utils as utils
from .abs import _Optimizer
from .fp16_optimizer import FP16Optimizer, GradScaler

OPTIMIZERS = {}


def build(args, params):
    if args.fp16 == 'half':
        return FP16Optimizer(args, params, lambda fp32_params: OPTIMIZERS[args.optimizer].build(args, fp32_params))
    else:
        return OPTIMIZERS[args.optimizer].build(args, params)


def register(name):
    def register_(cls):
        if name in OPTIMIZERS:
            raise ValueError(f'Cannot register duplicate optimizer ({name})')
        if not issubclass(cls, _Optimizer):
            raise ValueError(f'Optimizer ({name}: {cls.__name__}) must extend {_Optimizer}')
        OPTIMIZERS[name] = cls
        return cls

    return register_


utils.recursive_import(Path(__file__).parent, 'thseq.optim')
