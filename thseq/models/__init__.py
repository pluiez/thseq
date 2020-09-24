from pathlib import Path

import thseq.utils as utils
from .abs import _Model
from .ensemble import AverageLogProb

MODELS = {}


def build(args, vocabularies):
    model = MODELS[args.model].build(args, vocabularies)
    return model


def register(name):
    def register_(cls):
        if name in MODELS:
            raise ValueError(f'Cannot register duplicate optimizer ({name})')
        if not issubclass(cls, _Model):
            raise ValueError(f'Model ({name}: {cls.__name__}) must extend {_Model}')
        MODELS[name] = cls
        return cls

    return register_


utils.recursive_import(Path(__file__).parent, 'thseq.models')
