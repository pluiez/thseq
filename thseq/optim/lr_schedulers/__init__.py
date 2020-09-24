from pathlib import Path

import thseq.utils as utils
from .abs import _Scheduler

SCHEDULERS = {}


def build(args, lr, optimizer):
    return SCHEDULERS[args.lr_scheduler].build(args, lr, optimizer)


def register(name):
    def register_(cls):
        if name in SCHEDULERS:
            raise ValueError(f'Cannot register duplicate optimizer ({name})')
        if not issubclass(cls, _Scheduler):
            raise ValueError(f'Optimizer ({name}: {cls.__name__}) must extend {_Scheduler}')
        SCHEDULERS[name] = cls
        return cls

    return register_


utils.recursive_import(Path(__file__).parent, 'thseq.optim.lr_scheduler')
