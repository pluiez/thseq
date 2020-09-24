import importlib
import itertools
import logging
import multiprocessing
import operator
import os
import random
from typing import Callable, Any, List

import numpy as np
import torch

__all__ = ['DummySummaryWriter', 'Pool', 'recursive_import', 'seed', 'get_state_dict', 'load_state_dict', 'stat_cuda',
           'aggregate_values', 'profile_nan', 'has_overflow']

_func_obj = None


class DummySummaryWriter(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __getattr__(self, item):
        return self

    def __setattr__(self, key, value):
        setattr(self, key, value)

    def __call__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): ...


class Pool(object):
    def __init__(self, processes) -> None:
        super().__init__()
        self._processes = processes

    def _worker_init(self, fn):
        global _func_obj
        _func_obj = fn

    def _worker(self, data, index):
        end = len(data)
        chunk_size = end // self._processes
        offset = chunk_size * index
        end = min(offset + chunk_size, end)
        return [_func_obj(x) for x in data[offset:end]]

    def map_reduce(self, data, mapper, reducer=operator.concat):
        results = []
        with multiprocessing.Pool(self._processes, self._worker_init, initargs=(mapper,)) as pool:
            for i in range(self._processes):
                results.append(pool.apply_async(self._worker, (data, i)))
            import copy
            result = results.pop(0).get()
            result = copy.deepcopy(result)
            while results:
                result = reducer(result, results.pop(0).get())
        return result


def recursive_import(dir, module_prefix=''):
    for file in os.listdir(dir):
        fullname = os.path.join(dir, file)
        if file.endswith('.py') and not file.startswith('_'):
            name = fullname[:fullname.find('.py')]
            relative_dir = module_prefix.replace('.', os.sep)
            module = name[name.rfind(relative_dir):].replace(os.sep, '.')
            module = '{}'.format(module)
            importlib.import_module(module)
        if os.path.isdir(fullname) and not file.startswith('_'):
            recursive_import(fullname, "{}.{}".format(module_prefix, file))


def seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_state_dict(obj, recursive=True, exclusions=None, inclusions=None):
    keys = vars(obj).keys()
    state_dict = {}
    if exclusions and inclusions:
        raise TypeError('Argument `exclusions` and `inclusions` '
                        'should not be both non-empty at the same time.')

    if exclusions:
        keys = [k for k in keys if k not in exclusions]
    elif inclusions:
        keys = [k for k in keys if k in inclusions]

    props = map(lambda k: getattr(obj, k), keys)

    for key, prop in itertools.zip_longest(keys, props):
        if recursive:
            to_state_dict = getattr(prop, 'state_dict', None)
            if callable(to_state_dict):
                prop = to_state_dict()
        state_dict[key] = prop

    return state_dict


def load_state_dict(obj, state_dict):
    for key, val in state_dict.items():
        prop = getattr(obj, key)
        load = getattr(prop, 'load_state_dict', None)
        if callable(load):
            load(val)
        else:
            setattr(obj, key, val)


def stat_cuda(msg):
    print('-------', msg)
    print(torch.cuda.memory_allocated() / 1024 / 1024, torch.cuda.max_memory_allocated() / 1024 / 1024)
    print(torch.cuda.memory_cached() / 1024 / 1024, torch.cuda.max_memory_cached() / 1024 / 1024)


def aggregate_values(dicts, key: str = None, val_fn: Callable[[dict], Any] = None,
                     reduce: Callable[[List], Any] = None):
    assert key is None or val_fn is None
    assert not (key is None and val_fn is None)
    values = []
    if key is not None:
        values = [d[key] for d in dicts]
    if val_fn is not None:
        values = [val_fn(d) for d in dicts]

    if reduce:
        return reduce(values)
    return values


def profile_nan(model):
    logger = logging.getLogger('nan_detector')
    nan_found = False
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm()
                if norm == float('inf') or norm != norm:
                    nan_found = True
                    logger.warning(f'NaN/inf detected: {name}')
    if nan_found:
        raise OverflowError()


def has_overflow(parameters):
    for p in parameters:
        if p.grad is not None:
            norm = p.grad.norm()
            if norm == float('inf') or norm != norm:
                return True
    return False
