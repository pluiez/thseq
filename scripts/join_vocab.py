import argparse
from collections import Counter


def load(f):
    counter = Counter()
    with open(f) as r:
        for l in r:
            token, freq = l.split()
            counter[token] = int(freq)
    return counter


def main(args):
    counter = Counter()
    counter.update(load(args.sv))
    counter.update(load(args.tv))
    for token, freq in counter.most_common():
        print(f'{token} {freq}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sv', type=str)
    parser.add_argument('tv', type=str)
    args = parser.parse_args()
    main(args)
