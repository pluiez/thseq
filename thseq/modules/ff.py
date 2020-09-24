import torch
import torch.nn as nn
import torch.nn.functional as F

from thseq.modules.activation import factory


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0, bias=True, nonlinear='relu'):
        super().__init__()
        self.linear_i = nn.Linear(input_size, hidden_size, bias)
        self.nonlinear = factory(nonlinear)()
        self.linear_o = nn.Linear(hidden_size, output_size, bias)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_i.weight)
        nn.init.xavier_uniform_(self.linear_o.weight)
        if self.linear_i.bias is not None:
            nn.init.zeros_(self.linear_i.bias)
        if self.linear_o.bias is not None:
            nn.init.zeros_(self.linear_o.bias)

    def forward(self, x):
        x = self.linear_i(x)
        x = self.nonlinear(x)
        x = F.dropout(x, self.dropout, self.training)
        x = self.linear_o(x)
        return x


class PositionWiseFeedForwardShared(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0, bias=True, nonlinear='relu'):
        super().__init__()
        assert input_size == output_size
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bias_i = None
        self.bias_o = None
        if bias:
            self.bias_i = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_o = nn.Parameter(torch.Tensor(output_size))
        self.nonlinear = factory(nonlinear)()
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias_i is not None:
            nn.init.zeros_(self.bias_i)
        if self.bias_o is not None:
            nn.init.zeros_(self.bias_o)

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias_i)
        x = self.nonlinear(x)
        x = F.dropout(x, self.dropout, self.training)
        x = F.linear(x, self.weight.t(), self.bias_o)
        return x


class Maxout(nn.Module):

    def __init__(self, input_size, hidden_size, pool_size):
        super().__init__()
        self.input_size, self.hidden_size, self.pool_size = input_size, hidden_size, pool_size
        self.lin = nn.Linear(input_size, hidden_size * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.hidden_size
        shape.append(self.pool_size)
        out = self.lin(inputs)
        m, i = out.view(*shape).max(-1)
        return m


class DropNet(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, x, y):
        if self.training:
            dropout_prob = torch.empty(1).uniform_()[0]
            if dropout_prob < self.p / 2:
                return x
            if dropout_prob > 1 - self.p / 2:
                return y
            return 0.5 * (x + y)
        else:
            return 0.5 * (x + y)

    def extra_repr(self):
        return 'p={}, inplace={}'.format(self.p, self.inplace)
