import argparse
import logging
import multiprocessing
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.root.handlers = []
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s ', level=logging.DEBUG,
                    stream=sys.stderr)

logger = logging.getLogger('slim_checkpoint')


def parse_args():
    parser = argparse.ArgumentParser('Minimise the storage of a checkpoint by removing training states.')
    parser.add_argument('--inputs', '-i', nargs='+', required=True, help='Checkpoint files to slim')
    parser.add_argument('--outputs', '-o', nargs='+', help='Slimmed checkpoints')
    parser.add_argument('--inplace', action='store_true', help='Enable in-placed slimming')
    parser.add_argument('--parallel', type=int, default=6)
    return parser.parse_args()


def slim(input, output):
    logger.info(f'Loading checkpoint: {input}')
    checkpoint = torch.load(input, 'cpu')
    preserved_keys = ['model', 'args', 'vocabularies']
    remove_keys = [key for key in checkpoint if key not in preserved_keys]
    for key in remove_keys:
        del checkpoint[key]
    logger.info(f'Removed keys: {remove_keys}')
    if input == output:
        logger.info(f'Overwriting checkpoint ...')
    else:
        logger.info(f'Saving checkpoint to: {output} ...')
    torch.save(checkpoint, output)


def main(args):
    parallel = max(args.parallel, 1)
    pool = multiprocessing.Pool(processes=parallel)
    for input, output in zip(args.inputs, args.outputs):
        pool.apply_async(slim, (input, output))
    pool.close()
    pool.join()
    logger.info('Success')


if __name__ == '__main__':
    args = parse_args()
    if args.inplace:
        args.outputs = args.inputs
    elif not args.outputs or len(args.outputs) != len(args.inputs):
        logger.error(f'Number of inputs and outputs does not match: '
                     f'({len(args.inputs)},{len(args.outputs)})')
        sys.exit(1)

    main(args)
