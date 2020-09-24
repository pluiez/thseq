import functools

import numpy as np
import torch
import torch.nn as nn

from .nested import map as nested_map

__all__ = ['select', 'shape', 'clone', 'move_cuda', 'to_half', 'scale_clip_grad_', 'get_parameters', 'share_parameters',
           'pack']
_uint_types = {
    'uint8': 'int16', 'uint16': 'int32', 'uint32': 'int64'
}


def is_tensor_or_module(x):
    return torch.is_tensor(x) or isinstance(x, nn.Module)


def select(x, idxs, axis=0):
    return nested_map(lambda y: y.index_select(axis, idxs), x, torch.is_tensor)


def shape(x):
    return nested_map(lambda y: tuple(y.size()), x, torch.is_tensor, None, True)


def clone(x):
    return nested_map(lambda y: y, x, torch.is_tensor)


def move_cuda(x):
    if not torch.cuda.is_available():
        return x
    else:
        return nested_map(
            lambda y: y.cuda(),
            x,
            lambda y: is_tensor_or_module(y)
        )


def to_half(x):
    return nested_map(lambda y: y.half(), x, lambda y: torch.is_tensor(y) and y.dtype == torch.float32)


def scale_clip_grad_(params, scale, clip_norm):
    params = list(filter(lambda x: x.grad is not None, params))
    for p in params:
        p.grad.data.mul_(scale)
    if clip_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(params, clip_norm)
    else:
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.data, 2) for p in params]), 2)
    return grad_norm


def get_parameters(outer, *exclude_from):
    paras = []
    for para in outer.parameters():
        exclude = False
        for item in exclude_from:
            if isinstance(item, nn.Module):
                other = list(item.parameters())
            else:
                other = [item]
            for para2 in other:
                if para is para2:
                    exclude = True
                    break
            if exclude:
                break
        if not exclude:
            paras.append(para)

    return paras


def rset_attr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rget_attr(obj, pre) if pre else obj, post, val)


def rget_attr(obj, attr, *args):
    """
    using wonder's beautiful simplification:
    https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def share_parameters(module, share_to, strict=True):
    assert isinstance(module, nn.Module)
    assert isinstance(share_to, nn.Module)

    is_parameter = lambda name, module: any(name == name_ for name_, _ in module.named_parameters())

    for name, para in module.named_parameters():
        if strict and not is_parameter(name, share_to):
            raise RuntimeError(f'{name} is not an attribute of to_module'
                               f' or it\'s not an instance of nn.Parameter')
        rset_attr(share_to, attr=name, val=para)


def pack(tensors, padding_value, dtype):
    if not isinstance(tensors, (tuple, list)):
        raise ValueError(f'Expected a list or tuple, got {type(tensors)}.')
    if not isinstance(tensors[0], (np.ndarray, torch.Tensor)):
        raise ValueError(f'Invalid type: {type(tensors[0])}.')

    dtype = dtype or tensors[0].dtype
    if isinstance(tensors[0], np.ndarray):
        np_dtype = tensors[0].dtype.name
        if np_dtype in _uint_types:
            tensors = [t.astype(_uint_types[np_dtype]) for t in tensors]
        tensors = [torch.from_numpy(t) for t in tensors]

    tensors = [t.type(dtype) for t in tensors]
    max_len = max(t.size(0) for t in tensors)
    a = tensors[0].new_full((len(tensors), max_len), padding_value)
    for i, t in enumerate(tensors):
        a[i][:t.size(0)].copy_(t)
    return a
