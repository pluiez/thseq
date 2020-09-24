import itertools
import logging
import math
import os
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from .batch import batch_by_size

__all__ = ['write_binary_1darray', 'MemMapDataset', 'ZipDataset', 'BatchSampler']

logger = logging.getLogger(__name__)
code2dtype = {
    0: np.uint8,
    1: np.uint16,
    2: np.uint32,
    3: np.uint64,
    4: np.int8,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    8: np.float16,
    9: np.float32,
    10: np.float64,
}

_HEADER = b'BINIDX\x00\x00'


def write_binary_1darray(input_name, output_name, converter) -> None:
    sizes = [0]
    dtype = None
    output_dir = os.path.dirname(output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(input_name, 'r') as r:
        with open(f'{output_name}.bin', 'wb') as w:
            for l in r:
                array = converter(l)
                if torch.is_tensor(array):
                    array = array.numpy()
                if not isinstance(array, np.ndarray):
                    raise RuntimeError(f'Expected array as a np.ndarray object, got {type(array)}.')
                if array.ndim > 1:
                    raise RuntimeError(f'Expected array as 1d-array, got {array.ndim}.')
                if dtype is None:
                    dtype = array.dtype
                sizes.append(array.size)
                w.write(array.tobytes())
    if len(sizes) == 1:
        raise RuntimeError(f'sizes is empty, input_name is possibly an empty file: {input_name}')

    with open(f'{output_name}.idx', 'wb') as w:
        w.write(_HEADER)
        w.write(struct.pack('<B', _get_dtype_code(dtype)))  # uchar
        w.write(struct.pack('<q', len(sizes) - 1))  # int64
        sizes = np.array(sizes, dtype=np.uint32)
        positions = np.cumsum(sizes, dtype=np.uint64)
        sizes = sizes[1:]
        positions = positions[:-1]
        w.write(sizes.tobytes())
        w.write(positions.tobytes())
    return sizes.size


def _get_dtype_code(dtype):
    for k in code2dtype:
        if code2dtype[k] == dtype:
            return k
    raise ValueError(dtype)


def _guess_name(input_name, suffix):
    names = [f'{input_name}.{suffix}', input_name]
    for name in names:
        if Path(name).exists():
            return name
    raise ValueError(f'File does not exist, available names: {" or ".join(str(name) for name in names)}.')


def _warmup_memmap(path):
    """
    Force a read on newly input file to make sure that OS pre-cache it in
    memory. This can be useful to avoid concurrent disk access when the
    same data array is passed to different child processes.
    """
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class _MemMapIndex(object):
    def __init__(self, filename):
        filename = _guess_name(filename, 'idx')
        with open(filename, 'rb') as r:
            header = r.read(len(_HEADER))
            if header != _HEADER:
                raise ValueError(f'Index file ({filename}) is either corrupted or invalid.')
            code, = struct.unpack('<B', r.read(1))
            dtype = code2dtype[code]
            length, = struct.unpack('<q', r.read(8))
            offset = r.tell()
        _warmup_memmap(filename)
        self._mmap_buffer = np.memmap(filename, mode='r')
        buffer = memoryview(self._mmap_buffer)
        self._sizes = np.frombuffer(buffer, dtype=np.uint32, count=length, offset=offset)
        self._positions = np.frombuffer(buffer, dtype=np.uint64, count=length, offset=offset + self._sizes.nbytes)
        self._dtype = dtype
        self._length = length

    def __del__(self):
        del self._mmap_buffer

    @property
    def dtype(self):
        return self._dtype

    @property
    def sizes(self):
        return self._sizes

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        return self._sizes[index], self._positions[index]

    def __len__(self):
        return self._length


class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    @property
    def sizes(self):
        raise NotImplementedError


class MemMapDataset(Dataset):

    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self._run_init()

    def __getstate__(self):
        return self._filename

    def __setstate__(self, state):
        self._filename = state
        self._run_init()

    def _run_init(self):
        # for torch.multiprocessing.spwan together with multi-processing dataloader,
        # memoryview can't be copied to dataloader's workers
        index_name = _guess_name(self._filename, 'idx')
        bin_name = _guess_name(self._filename, 'bin')
        self._index = _MemMapIndex(index_name)
        _warmup_memmap(bin_name)
        self._mmap_buffer = np.memmap(bin_name, mode='r', order='C')
        self._buffer = memoryview(self._mmap_buffer)

    def __del__(self):
        del self._mmap_buffer
        del self._index

    def __len__(self):
        return len(self._index)

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        size, pos = self._index[index]
        dtype = self._index.dtype
        offset = int(pos * dtype().itemsize)
        array = np.frombuffer(self._buffer, dtype=self._index.dtype, count=size, offset=offset)
        if array.dtype != np.int64:
            array = array.astype(np.int64)

        return torch.from_numpy(array)

    @property
    def sizes(self):
        return self._index.sizes


class TextDataset(Dataset):
    def __init__(self, filename, lookup_fn):
        super().__init__()
        with open(filename) as r:
            self._data = list(map(lookup_fn, r.readlines()))
        self._sizes = np.array([item.size for item in self._data])
        self._filename = filename

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index) -> np.ndarray:
        return self._data[index]

    @property
    def sizes(self):
        return self._sizes


class ZipDataset(Dataset):
    def __init__(self, datasets, is_ordered=False):
        assert isinstance(datasets, (tuple, list))
        sizes = list(map(len, datasets))
        assert max(sizes) == min(sizes)
        self._datasets = datasets
        self._is_ordered = is_ordered
        self._size = sizes[0]
        self._max_sizes = np.max(np.stack([d.sizes for d in datasets], 0), 0)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return tuple(d[index] for d in self._datasets)

    @property
    def sizes(self):
        return self._max_sizes

    def max_size(self, index):
        return max(self.size(index))

    def min_size(self, index):
        return min(self.size(index))

    def size(self, index):
        return tuple(d.sizes[index] if d.sizes is not None else 0 for d in self._datasets)

    def ordered_indices(self, sort_by_keys=None, sort_by_max=None, shuffle=False):
        assert not (sort_by_keys and sort_by_max)
        assert not (self._is_ordered and bool(shuffle))
        if self._is_ordered or not shuffle:
            indices = np.arange(len(self))
        else:
            indices = np.random.permutation(len(self))
            if sort_by_keys:
                for key in sort_by_keys:
                    indices = indices[np.argsort(self._datasets[key].sizes[indices], kind='mergesort')]
            elif sort_by_max:
                indices = indices[np.argsort(self._max_sizes[indices], kind='mergesort')]
        return indices


class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, indices, max_tokens, max_sentences, shuffle, num_shards, shard_index, min_length=0,
                 max_length=0) -> None:
        super().__init__(dataset)
        max_tokens = max_tokens or 0
        max_sentences = max_sentences or 0
        if max_tokens == 0 and max_tokens == 0:
            raise ValueError()
        self._dataset = dataset
        self._indices = indices
        self._max_tokens = max_tokens or 0
        self._max_sentences = max_sentences or 0
        self._shuffle = shuffle
        self._num_shards = num_shards
        self._shard_index = shard_index
        self._min_length = min_length
        self._max_length = max_length

        self._batches = self._arrange_batches()
        self._size = int(math.ceil(len(self._batches) * 1.0 / num_shards))

    def _arrange_batches(self):
        batch_stops, dropped = batch_by_size(
            self._indices,
            self._dataset.sizes,
            self._max_tokens,
            self._max_sentences,
            self._min_length,
            self._max_length,
        )
        batches = np.split(self._indices, batch_stops)
        logger.info(f'Num. of batches: {len(batches)}')
        logger.info(f'Num. of dropped samples: {dropped}')
        return batches

    def __len__(self):
        return self._size

    def __iter__(self):
        if self._shuffle:
            np.random.shuffle(self._batches)

        for batch in itertools.islice(self._batches, self._shard_index, None, self._num_shards):
            yield batch
        extra = self._size - len(self._batches)
        if extra > 0:
            for batch in np.random.choice(self._batches, extra, False):
                yield batch

    def __getitem__(self, index):
        return self._batches[index]
