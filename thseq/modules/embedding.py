import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timing_signal(input_size, length, min_timescale=1.0, max_timescale=1e4):
    channels = input_size
    # e_{pos,2i} = sin( pos / 10000 ^ (2i/d) )
    # e_{pos,2i+1} = cos( pos / 10000 ^ (2i/d) )
    # sin(t+k) = sin(t)cos(k)+cos(t)sin(k)
    # cos(t+k) = cos(a)cos(b) - sin(a)sin(b)
    position = torch.arange(length).float()
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).float() * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
    if input_size % 2 > 0:
        signal = F.pad(signal, (0, 1))
    signal = signal.view([1, length, channels])
    return signal


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, input_size, length=1026, min_timescale=1.0, max_timescale=1e4):
        """Initialises the embeddings.
        Args:
            input_size: Embedding_size
            length: Maximum length plus 2 dummy positions at front.
            min_timescale:
            max_timescale:
        """
        super().__init__()

        signal = get_timing_signal(input_size, length, min_timescale, max_timescale)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.pe = signal

    def forward(self, input):
        self.pe = self.pe.to(self._float_tensor)
        length = input.size(1)
        if length > self.pe.size(1) - 2:
            raise RuntimeError(f'Expected input length <= {self.pe.size(1) - 2}, got {length}.')
        # note: hard-coded to ignore the 0-th and 1-th positional embedding,
        # which corresponds to a padding used in fairseq.
        return self.pe[:, 2:length + 2]


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, input_size, max_len):
        super().__init__()

        self.pe = nn.Embedding(max_len, input_size)

        self.max_len = max_len

    def forward(self, input):
        length = input.size(1)
        positions = torch.arange(length).to(input.device).long()
        if length > self.max_len:
            positions[self.max_len:] = self.max_len - 1

        pe = self.pe(positions).unsqueeze(0)

        return pe


class RelativePositionEmbedding(nn.Embedding):

    def __init__(self, k: int, embedding_dim: int) -> None:
        super().__init__(2 * k + 1, embedding_dim)
        self.k = k

    def forward(self, x):
        assert x.ndim >= 3
        length = x.size(-2)
        range = torch.arange(length).to(x.device)
        distance = range[None, :] - range[:, None]
        clipped_distance = distance.clamp(-self.k, self.k)
        clipped_distance = clipped_distance + self.k
        return super(RelativePositionEmbedding, self).forward(clipped_distance)
