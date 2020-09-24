import itertools
import logging
import math
import queue
import time
from pathlib import Path
from threading import Thread
from typing import *

import lunas
from torch.utils.data import DataLoader

from .datasets import MemMapDataset, ZipDataset, BatchSampler

__all__ = ['MMapEpochIterator', 'BucketingEpochIterator']

logger = logging.getLogger(__name__)

_sentinel = object()

_dummy_length = 1 << 32


def _support_length(x):
    if isinstance(x, (lunas.Dataset, lunas.BatchIterator, lunas.DataLoader)):
        return False
    elif hasattr(x, '__len__'):
        return True
    else:
        return False


def _get_length(x):
    return len(x) if _support_length(x) else _dummy_length


class BackgroundConsumer(Thread):
    def __init__(self, queue, source):
        Thread.__init__(self)

        self._queue = queue
        self._source = source

    def run(self):
        try:
            for x in self._source:
                self._queue.put(x)

            self._queue.put(_sentinel)
        except Exception as e:
            self._queue.put(e)


class BufferedIterator(object):
    def __init__(self, buffer_size, iterable):
        self._queue = queue.Queue(buffer_size)
        self._iterable = iterable
        self._consumer = None

        self.start_time = time.time()
        self.warning_time = None

    def _create_consumer(self):
        self._consumer = BackgroundConsumer(self._queue, self._iterable, )
        # ensures Python program exits regardless the state of daemon threads
        self._consumer.daemon = True
        self._consumer.start()

    def __iter__(self):
        return self

    def __len__(self):
        return _get_length(self._iterable)

    def __next__(self):
        if self._consumer is None:
            self._create_consumer()

        item = self._queue.get(True)
        if isinstance(item, Exception):
            raise item
        if item is _sentinel:
            raise StopIteration()
        return item


class SkipIterator(object):
    def __init__(self, iterator, n):
        self._iterator = iterator
        self._n = n

    def __len__(self):
        return _get_length(self._iterator)

    def __iter__(self):
        for x in itertools.islice(iter(self._iterator), self._n, None):
            yield x


class CountingIterator(object):
    def __init__(self, iterator, start=0):
        self._iterator = iterator
        self._count = start

    @property
    def count(self):
        return self._count

    def __len__(self):
        return _get_length(self._iterator)

    def __iter__(self):
        for x in self._iterator:
            self._count += 1
            yield x


class ChunkIterator(object):

    def __init__(self, iterator, chunk_size: int) -> None:
        super().__init__()
        self._iterator = iterator
        self._chunk_size = chunk_size
        size = _dummy_length
        if _support_length(iterator):
            size = math.ceil(len(iterator) / chunk_size)
        self._size = size

    def __len__(self):
        return self._size

    def __iter__(self):
        chunk = []
        size = self._chunk_size
        for x in self._iterator:
            chunk.append(x)
            if len(chunk) == size:
                yield chunk
                chunk.clear()
        if chunk:
            yield chunk


class EpochIterator(object):
    def __init__(self,
                 num_shards: int = 1,
                 shard_index: int = 0,
                 chunk_size: int = 1,
                 collate: Callable = None,
                 num_workers: int = 0,
                 buffer_size: int = 0,
                 shuffle: bool = False) -> None:
        super().__init__()
        assert 0 <= shard_index < num_shards
        self._num_shards = num_shards
        self._shard_index = shard_index
        self._chunk_size = chunk_size
        self._collate = collate
        self._num_workers = num_workers
        self._buffer_size = buffer_size
        self._shuffle = shuffle

        self._epoch = 0
        self._itr: CountingIterator = None
        self._offset = 0
        self._need_resume = False

    @property
    def supports_len(self):
        try:
            length = len(self)
            return True
        except NotImplementedError:
            return False

    def __len__(self):
        raise NotImplementedError

    def _get_dataloader(self, *args, **kwargs) -> Union[DataLoader, lunas.DataLoader]:
        raise NotImplementedError

    def state_dict(self):
        return {
            'epoch': self._epoch,
            'offset': 0 if self._itr is None else self._itr.count
        }

    def load_state_dict(self, state):
        self._epoch = state['epoch']
        self._offset = state['offset']
        self._need_resume = True

    def next_epoch_itr(self, *args, **kwargs) -> CountingIterator:
        itr = self._get_dataloader(*args, **kwargs)
        if not isinstance(itr, (DataLoader, lunas.DataLoader)):
            raise ValueError(
                f'Expected instance of (torch.utils.data.DataLoader, lunas.DataLoader), '
                f'got {type(itr)}.'
            )

        if self._buffer_size > 0:
            itr = BufferedIterator(self._buffer_size, itr)
        itr = ChunkIterator(itr, self._chunk_size)
        offset = 0
        if self._need_resume:
            itr = SkipIterator(itr, self._offset)
            self._need_resume = False
            offset = self._offset
        itr = CountingIterator(itr, offset)
        self._itr = itr
        return itr


class MMapEpochIterator(EpochIterator):

    def __init__(self, data_dir,
                 names,
                 is_ordered: bool,
                 max_tokens: int,
                 max_sentences: int,
                 num_shards: int = 1,
                 shard_index: int = 0,
                 chunk_size: int = 1,
                 collate: Callable = None,
                 num_workers: int = 0,
                 buffer_size: int = 0,
                 shuffle: bool = False,
                 min_length: int = 0,
                 max_length: int = 0) -> None:
        super().__init__(num_shards, shard_index, chunk_size, collate, num_workers, buffer_size, shuffle)
        logger.info('Loading binary data')
        datasets = [MemMapDataset(Path(data_dir) / Path(name).name) for name in names]
        ds = ZipDataset(datasets, is_ordered=is_ordered)

        logger.info('Building indices')
        ordered_indices = ds.ordered_indices(sort_by_max=True, shuffle=False if is_ordered else shuffle)
        logger.info('Building batches')
        sampler = BatchSampler(ds, ordered_indices, max_tokens, max_sentences, shuffle, num_shards, shard_index,
                               min_length, max_length)
        self._dataset = ds
        self._sampler = sampler

    def __len__(self):
        return len(self._sampler)

    def _get_dataloader(self, *args, **kwargs) -> Union[DataLoader, lunas.DataLoader]:
        dl = DataLoader(self._dataset, batch_sampler=self._sampler, num_workers=self._num_workers,
                        collate_fn=self._collate)
        return dl


class BucketingEpochIterator(EpochIterator):

    def __init__(self, dl_builder, num_shards: int = 1, shard_index: int = 0, chunk_size: int = 1,
                 collate: Callable = None, num_workers: int = 0, buffer_size: int = 0, shuffle: bool = False) -> None:
        super().__init__(num_shards, shard_index, chunk_size, collate, num_workers, buffer_size, shuffle)
        self._dl_builder = dl_builder

    def __len__(self):
        raise NotImplementedError

    def _get_dataloader(self, *args, **kwargs) -> Union[DataLoader, lunas.DataLoader]:
        return self._dl_builder()
