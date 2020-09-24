import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thseq.data.vocabulary import Vocabulary
from thseq.data.datasets import write_binary_1darray

logging.root.handlers = []
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s ', level=logging.DEBUG,
                    stream=sys.stderr)

logger = logging.getLogger('binarize')


def binarize_text(index, input_name, target_dir, vocab_name, size, min_freq, name):
    input_name = Path(input_name)
    output_name = Path(target_dir) / input_name.name
    if name:
        suffix = input_name.suffix[1:]
        if not suffix:
            raise RuntimeError(f'Invalid suffix: {suffix}')
        output_name = Path(target_dir) / f'{name}.{suffix}'
    output_vocab_name = Path(target_dir) / Path(vocab_name).name

    vocab = Vocabulary(vocab_name, size=size, min_freq=min_freq)

    logger.info(f'({index}) input = {input_name}')
    logger.info(f'({index}) output = {output_name}')
    logger.info(f'({index}) vocab = {output_vocab_name} ')

    vocab.write(output_vocab_name)

    logger.info(f'({index}) dtype = {vocab._dtype} ')
    size = write_binary_1darray(
        input_name,
        output_name,
        lambda x: vocab.lookup(x.split(), vocab._dtype)
    )
    logger.info(f'({index}) dataset size = {size}')
    return size


def main(args):
    logger.info(f'Training inputs: {args.train}')
    logger.info(f'Vocabulary files: {args.vocab}')
    logger.info(f'Target directory: {args.target_dir}')

    target_dir = Path(args.target_dir)

    if target_dir.exists() and len(list(target_dir.iterdir())) > 1:
        logger.error('Target directory not empty!')
        exit(1)
    os.makedirs(target_dir, exist_ok=True)
    pool = mp.Pool(processes=len(args.train))
    sizes = []

    for i, (filename, vocab_name, vocab_size, vocab_min_freq) in enumerate(
            zip(args.train, args.vocab, args.vocab_size, args.vocab_min_freq)):
        sizes.append(
            pool.apply_async(
                binarize_text,
                (i, filename, target_dir, vocab_name, vocab_size, vocab_min_freq, args.name)
            )
        )

    pool.close()
    pool.join()
    sizes = [size.get() for size in sizes]
    if max(sizes) != min(sizes):
        raise RuntimeError(f'Expected consistent sizes across inputs: got {tuple(sizes)}.')
    logger.info(f'size = {max(sizes)}')

    with (target_dir / 'config.json').open('w') as w:
        w.write(json.dumps(args.__dict__, indent=4, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, nargs=2, required=True)
    parser.add_argument('--vocab', type=str, nargs=2, required=True)
    parser.add_argument('--vocab-size', type=int, nargs=2, required=True)
    parser.add_argument('--vocab-min-freq', type=int, nargs=2, required=True)
    parser.add_argument('--target-dir', type=str, required=True)
    parser.add_argument('--name', type=str, default='train')
    args = parser.parse_args()
    main(args)
