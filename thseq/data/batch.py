import logging

import numba

__all__ = ['batch_by_size']

logger = logging.getLogger(__name__)


@numba.njit
def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if 0 < max_sentences == len(batch):
        return 1
    if 0 < max_tokens < num_tokens:
        return 1
    return 0


@numba.njit
def batch_by_size(indices, sizes, max_tokens, max_sentences, min_length, max_length):
    batch = numba.typed.List.empty_list(numba.int64)
    sample_sizes = numba.typed.List.empty_list(numba.int64)
    max_size = 0
    dropped = 0
    batch_stops = numba.typed.List.empty_list(numba.int64)
    for i, index in enumerate(indices):
        size = sizes[index]
        if min_length or max_length:
            if not (min_length <= size <= max_length):
                dropped += 1
                continue
        max_size = max(size, max_size)
        num_tokens = (len(batch) + 1) * max_size
        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                8 * (len(batch) // 8),
                len(batch) % 8,
            )
            batch_stops.append(batch[mod_len - 1] + 1)
            batch = batch[mod_len:]
            sample_sizes = sample_sizes[mod_len:]
            max_size = max(sample_sizes) if sample_sizes else 0
        sample_sizes.append(max_size)
        batch.append(i)
    if batch_stops[-1] >= len(indices):
        batch_stops.pop()
    return batch_stops, dropped
