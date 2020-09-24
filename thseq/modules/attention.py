import torch
import torch.nn as nn
import torch.nn.functional as F

import thseq.utils as utils
from .embedding import RelativePositionEmbedding

__all__ = ['MultiHeadAttention']


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 is_incremental,
                 q_size=None, k_size=None, v_size=None,
                 output_size=None,
                 dropout=0., bias=True, add_bias_kv=False,
                 relative_pos_k=0, relative_pos_v=False):
        super().__init__()
        q_size = q_size or hidden_size
        k_size = k_size or hidden_size
        v_size = v_size or hidden_size
        output_size = output_size or hidden_size
        relative_pos_v = bool(relative_pos_v)

        head_size = hidden_size // num_heads
        assert head_size * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.linear_q = nn.Linear(q_size, hidden_size, bias)
        self.linear_k = nn.Linear(k_size, hidden_size, bias)
        self.linear_v = nn.Linear(v_size, hidden_size, bias)

        self.linear_o = nn.Linear(hidden_size, output_size, bias)

        self.bias_k = self.bias_v = None
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.bias_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
            raise NotImplementedError

        self.relative_embedding_k = None
        self.relative_embedding_v = None
        if relative_pos_k > 0:
            self.relative_embedding_k = RelativePositionEmbedding(relative_pos_k, head_size)
            if bool(relative_pos_v):
                self.relative_embedding_v = RelativePositionEmbedding(relative_pos_k, head_size)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.is_incremental = is_incremental
        self.q_size = q_size
        self.k_size = k_size
        self.v_size = v_size
        self.output_size = output_size
        self.dropout = dropout
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.relative_pos_k = relative_pos_k
        self.relative_pos_v = relative_pos_v

        self.head_size = head_size
        self.scaling = head_size ** -0.5

        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.0
        if self.q_size == self.v_size == self.v_size:
            gain = 1 / 2 ** 0.5
        nn.init.xavier_uniform_(self.linear_q.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_k.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_v.weight, gain=gain)

        nn.init.xavier_uniform_(self.linear_o.weight)
        nn.init.zeros_(self.linear_o.bias)
        if self.bias_k is not None:
            nn.init.xavier_uniform_(self.bias_k)
            nn.init.xavier_uniform_(self.bias_v)

    def forward(self, q, k, v, mask=None, state=None, avoid_nan=False):
        """
        Compute multi-head attention with give q/k/v.
        Args:
            q: (T x B x d_q)
            k: (S x B x d_k)
            v: (S x B x d_v)
            mask: (B x T x S), the masked element should be marked with either -inf or True.
            state: a dictionary
            avoid_nan: bool
        Returns:
            (T x B x H)
        """
        T, B, _ = q.size()
        num_head, head_size = self.num_heads, self.head_size

        k_ = state['K'].view(B * num_head, -1, head_size) if state and 'K' in state else None
        v_ = state['V'].view(B * num_head, -1, head_size) if state and 'V' in state else None

        if not self.is_incremental:
            assert not (k is None and k_ is None)
            assert not (v is None and v_ is None)

        q = self.linear_q(q)
        k = self.linear_k(k) if self.is_incremental or k_ is None else None
        v = self.linear_v(v) if self.is_incremental or v_ is None else None
        q = q * self.scaling
        q = q.view(T, B * num_head, head_size).transpose(0, 1)
        if k is not None:
            k = k.view(-1, B * num_head, head_size).transpose(0, 1)
        if v is not None:
            v = v.view(-1, B * num_head, head_size).transpose(0, 1)

        if self.is_incremental:
            if k_ is not None:
                k = torch.cat((k_, k), 1)
            if v_ is not None:
                v = torch.cat((v_, v), 1)
        else:
            k = k_ if k is None else k
            v = v_ if v is None else v
        S = k.size(1)

        if state is not None:
            state['K'] = k.view(B, num_head, -1, head_size)
            state['V'] = v.view(B, num_head, -1, head_size)

        score = torch.bmm(q, k.transpose(1, 2))
        if self.relative_embedding_k is not None:
            score = score + torch.bmm(
                q.transpose(0, 1),
                self.relative_embedding_k(k).transpose(1, 2)
            ).transpose(0, 1)

        if mask is not None:
            score = score.view(B, self.num_heads, T, -1)
            mask = mask.unsqueeze(1)

            if mask.dtype == torch.bool:
                score.masked_fill_(mask, float('-inf'))
            else:
                score = score + mask
            score = score.view(B * self.num_heads, T, -1)
            mask = mask.squeeze(1)

        if mask is not None and avoid_nan:
            nan_mask = mask.detach().clone()
            if nan_mask.dtype == torch.bool:
                nan_mask = nan_mask.masked_fill_(nan_mask, float('-inf'))
            nan_mask = nan_mask.max(2)[0].lt(0).unsqueeze(2)
            if nan_mask.any():
                score.masked_fill_(nan_mask, 1.0)

        score = utils.softmax(score, dim=-1)
        score = score.type_as(v)

        if mask is not None and avoid_nan:
            if nan_mask.any():
                score = score.masked_fill(nan_mask, 0.0)

        score = F.dropout(score, self.dropout, self.training)
        o = torch.bmm(score, v)
        if self.relative_embedding_v is not None:
            o = o + torch.matmul(
                score.transpose(0, 1),
                self.relative_embedding_v(k)
            ).transpose(0, 1)
        o = o.transpose(0, 1).contiguous().view(T, B, -1)
        o = self.linear_o(o)
        return o

    def extra_repr(self) -> str:
        repr = ['{hidden_size}, {output_size}', 'num_heads={num_heads}', 'head_size={head_size}']
        if self.q_size == self.v_size == self.v_size:
            repr.append('q/k/v={q_size}')
        else:
            repr.append('q={q_size}, k={k_size}, v={k_size}')
        if self.relative_pos_k:
            repr.append('relative_k/v={relative_pos_k}/{relative_pos_v}')

        return ', '.join(repr).format(**self.__dict__)
