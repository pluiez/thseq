import torch
import torch.nn.functional as F

__all__ = ['softmax', 'log_softmax']


def softmax(x, dim, use_float32=True):
    if use_float32:
        return F.softmax(x, dim, dtype=torch.float32)
    else:
        return F.softmax(x, dim)


def log_softmax(x, dim, use_float32=True):
    if use_float32:
        return F.log_softmax(x, dim, dtype=torch.float32)
    else:
        return F.log_softmax(x, dim)
