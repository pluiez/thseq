import argparse
import logging
import sys

import torch

logging.root.handlers = []
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s ', level=logging.DEBUG,
                    stream=sys.stderr)

logger = logging.getLogger('average_checkpoints')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', '-i', nargs='+', required=True, help='Checkpoint files to average')
    parser.add_argument('--output', '-o', required=True, help='Output averaged checkpoint')
    return parser.parse_args()


def load_checkpoint(path):
    checkpoint = torch.load(path, 'cpu')
    return {'args': checkpoint['args'], 'model': checkpoint['model'], 'vocabularies': checkpoint['vocabularies']}


def main(args):
    logger.info(f'Loading {len(args.inputs)} checkpoints ...')
    weight = 1.0 / len(args.inputs)

    logger.info(f'Scaling parameters with factors: {weight} ...')
    averaged_checkpoint = load_checkpoint(args.inputs[0])
    for k in averaged_checkpoint['model']:
        averaged_checkpoint['model'][k] = averaged_checkpoint['model'][k].float()
        averaged_checkpoint['model'][k] *= weight

    for i in range(1, len(args.inputs)):
        other = load_checkpoint(args.inputs[i])
        for k in averaged_checkpoint['model']:
            averaged_checkpoint['model'][k] += (other['model'][k].float() * weight)
    logger.info(f'Saving averaged checkpoint to file: {args.output}')
    torch.save(averaged_checkpoint, args.output)
    logger.info('Success')


if __name__ == '__main__':
    args = parse_args()
    main(args)
