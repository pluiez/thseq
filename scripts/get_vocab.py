import argparse
import collections
import multiprocessing
import os
import sys
import time


def drop_incomplete_line(r):
    pos = r.tell()
    while True:
        try:
            return r.readline()
        except UnicodeDecodeError:
            pos -= 1
            r.seek(pos)


def get_vocab(filename, worker_id=0, num_workers=1):
    counter = collections.Counter()
    with open(filename, encoding='utf-8') as r:
        size = os.fstat(r.fileno()).st_size
        chunk_size = size // num_workers
        offset = worker_id * chunk_size
        end = offset + chunk_size
        r.seek(offset)
        if offset > 0:
            drop_incomplete_line(r)
        while True:
            line = r.readline()
            if not line:
                break
            counter.update(line.split())
            if r.tell() > end:
                break
        return counter


def pad_to_multiple(vocab, padding_format, padding_factor=8):
    remain = len(vocab) % 8
    if remain > 0:
        for i in range(remain, padding_factor):
            vocab[padding_format.format(i - remain)] = 0
    assert len(vocab) % 8 == 0


def main(args):
    tic = time.time()
    if args.parallel <= 1:
        counter = get_vocab(args.corpus)
    else:
        pool = multiprocessing.Pool(processes=args.parallel)
        counters = []
        for i in range(args.parallel):
            counters.append(
                pool.apply_async(get_vocab, (args.corpus, i, args.parallel))
            )
        pool.close()
        pool.join()
        counter = counters[0].get()
        for other in counters[1:]:
            counter.update(other.get())
    vocab = collections.OrderedDict(counter.most_common(args.limit or sys.maxsize))
    if args.pad_to_multiple:
        pad_to_multiple(vocab, args.padding_format, args.pad_to_multiple)
    for k in vocab:
        print(f'{k} {vocab[k]}')

    sys.stderr.write(f'Took {time.time() - tic}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('--parallel', '-P', type=int, default=6)
    parser.add_argument('--limit', '-L', type=int)
    parser.add_argument('--tokens', nargs='+', help='Prepend additional tokens.')
    parser.add_argument('--pad-to-multiple', type=int, default=None)
    parser.add_argument('--padding-format', default='madeupword{:04d}')

    args = parser.parse_args()
    main(args)
