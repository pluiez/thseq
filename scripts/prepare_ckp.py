import argparse

import torch


def main(args):
    ckp = torch.load(args.ckp, map_location='cpu')
    ckp2 = {'args': ckp['args'], 'model': ckp['model']}
    print(f'args: {ckp2["args"]}')
    for k, v in ckp2['model'].items():
        print(f'{k}: {tuple(v.shape)}')

    torch.save(ckp2, f'{args.ckp}.prepare')
    sv, tv = ckp['vocabularies']
    sv.write(f'{args.ckp}.sv', True)
    tv.write(f'{args.ckp}.tv', True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckp', type=str)
    args = parser.parse_args()
    main(args)
