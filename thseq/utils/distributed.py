import pickle
from typing import Dict, List

import torch
import torch.distributed as dist

from .misc import aggregate_values

__all__ = ['is_master', 'enable_on_master', 'all_gather_list', 'all_reduce_dict', 'broadcast']


def is_master() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def enable_on_master(default_rv=None):
    def f(func):
        def g(*args, **kwargs):
            if not is_master():
                return default_rv
            else:
                return func(*args, **kwargs)

        return g

    return f


def all_gather_list(data, group=dist.group.WORLD, max_size=16384) -> List:
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    buffer_size = max_size * world_size
    # attach a buffer to the function to avoid expensive tensor allocation
    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
    buffer = all_gather_list._buffer
    buffer.zero_()

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255 * 256

    buffer_rank = buffer[rank * max_size: (rank + 1) * max_size]
    buffer_rank[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer_rank[1] = enc_size % 255
    buffer_rank[2:enc_size + 2] = torch.ByteTensor(list(enc))

    dist.all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = (255 * out_buffer[0].item()) + out_buffer[1].item()
            if size > 0:
                result.append(
                    pickle.loads(bytes(out_buffer[2:size + 2].tolist()))
                )
        return result
    except pickle.UnpicklingError as e:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )


def all_reduce_dict(d: Dict, reduce=sum) -> Dict:
    assert isinstance(d, dict)
    if dist.is_initialized():
        dicts = all_gather_list(d)
        d = {}
        for k in dicts[0]:
            d[k] = aggregate_values(dicts, k, reduce=reduce)
    return d


def broadcast(data, src=0, group=dist.group.WORLD, max_size=65534):
    """Broadcast arbitrary data to all nodes.
    """

    buffer_size = max_size
    if not hasattr(broadcast, '_buffer') or \
            broadcast._buffer.numel() < buffer_size:
        broadcast._buffer = torch.cuda.ByteTensor(buffer_size)
    buffer = broadcast._buffer
    buffer.zero_()

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))

    buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer[1] = enc_size % 255
    buffer[2:enc_size + 2] = torch.ByteTensor(list(enc))

    dist.broadcast(buffer, src, group=group)

    try:
        size = (255 * buffer[0].item()) + buffer[1].item()
        if size > 0:
            result = pickle.loads(bytes(buffer[2:size + 2].tolist()))
            return result
        return None
    except pickle.UnpicklingError as e:
        raise Exception(
            'Unable to unpickle data from other workers. broadcast requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )
