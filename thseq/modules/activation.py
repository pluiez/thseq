import math

import torch
import torch.nn as nn

_gelu_constant = math.sqrt(2.0 / math.pi)


def factory(name, default='relu'):
    _mapping = {
        'relu': nn.ReLU,
        'gelu': GELU,
        'swish': Swish
    }
    default = _mapping['relu']
    return _mapping.get(name, default)


def swish(x, beta=1.0):
    return x * torch.sigmoid(beta * x)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(_gelu_constant * (x + 0.044715 * x.pow(3))))


class Swish(nn.Module):

    def __init__(self, beta=1.0, trainable=False):
        super().__init__()
        if trainable:
            beta = nn.Parameter(torch.full((1,), beta))
        self.beta = beta

    def forward(self, x):
        return swish(x, self.beta)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)
