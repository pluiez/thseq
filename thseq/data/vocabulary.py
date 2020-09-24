import collections
import logging
from pathlib import Path
from typing import List, Union

import numpy as np

__all__ = ['Vocabulary']

logger = logging.getLogger(__name__)


def strip_int_size(max_value):
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    for dtype in dtypes:
        if np.iinfo(dtype).max + 1 >= max_value:
            return dtype


def load_vocab_from_files(list_of_files):
    symbol2freq = collections.defaultdict(int)
    for file in list_of_files:
        with Path(file).open(encoding="utf-8") as r:
            for line in r:
                symbol, freq = line.rsplit(None, 1)
                symbol2freq[symbol] += int(freq)
    return symbol2freq


class Vocabulary(object):

    def __init__(self, files: Union[str, Path, List[str], List[Path]] = None, symbols: List[str] = None,
                 size: int = None,
                 *,
                 min_freq: int = None,
                 pad="<pad>", eos="</s>", unk="<unk>", bos="<s>",
                 insert_symbols=True) -> None:
        super().__init__()
        size = size or None

        self.additional_symbols = [bos, pad, eos, unk]

        files = [files] if isinstance(files, (str, Path)) else files

        if size is not None and size <= 4:
            raise ValueError("size too small")

        self._symbol2index, self._index2symbol, self._symbol2freq = {}, [], {}

        symbol2freq = None
        if files is not None:
            symbol2freq = load_vocab_from_files(files)
            if min_freq:
                symbols = [symbol for symbol in symbol2freq if symbol2freq[symbol] >= min_freq]
            else:
                symbols = list(symbol2freq.keys())
        else:
            symbols = list(symbols)

        if insert_symbols:
            for additional in self.additional_symbols:
                if additional in symbols:
                    logger.warning(f'Token conflicts: {additional}, ignored.')
                    del symbols[symbols.index(additional)]
            symbols = self.additional_symbols + symbols

        symbols = symbols if size is None else symbols[:size]

        self.add_symbols(symbols, symbol2freq)

        self._bos_id = self._symbol2index[bos]
        self._pad_id = self._symbol2index[pad]
        self._eos_id = self._symbol2index[eos]
        self._unk_id = self._symbol2index[unk]

        self._num_padding = 0
        self.pad_to_multiple_(8)
        self._dtype = strip_int_size(len(self))

    def __len__(self):
        return len(self._symbol2index)

    @property
    def bos_id(self):
        return self._bos_id

    @property
    def pad_id(self):
        return self._pad_id

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def unk_id(self):
        return self._unk_id

    def add_symbols(self, symbols, symbol2freq=None):
        start = len(self._symbol2index)
        for i, symbol in enumerate(symbols, start):
            if symbol in self._symbol2index:
                print(symbol)
            self._symbol2index[symbol] = i
            self._index2symbol.append(symbol)
            if symbol2freq:
                self._symbol2freq[symbol] = symbol2freq[symbol]

        assert len(self._symbol2index) == len(self._index2symbol)

    def lookup(self, symbols: Union[List[str], str], dtype=np.int64) -> np.ndarray:
        if isinstance(symbols, str):
            symbols = symbols.split()
        prepends = []
        appends = [self.eos_id]

        ids = [self._symbol2index.get(symbol, self.unk_id) for symbol in symbols]
        return np.array(prepends + ids + appends, dtype)

    def revert(self, ids) -> List[str]:
        return [self._index2symbol[index] for index in ids]

    def pad_to_multiple_(self, padding_factor=8):
        pad_symbol = "<pad_symbol_{}>"
        remain = len(self) % 8
        if remain > 0:
            for i in range(remain, padding_factor):
                self.add_symbols([pad_symbol.format(i - remain)])
                self._num_padding += 1
        assert len(self) % 8 == 0

    def write(self, filename, keep_additional=False):
        with open(filename, 'w') as w:
            offset = len(self.additional_symbols)
            if keep_additional:
                offset = 0
            for i, symbol in enumerate(self._index2symbol[offset:]):
                w.write(f'{symbol} {self._symbol2freq.get(symbol, 9527)}\n')
