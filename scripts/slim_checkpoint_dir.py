import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import thseq.utils.checkpoint as checkpoint_util


def slim(input, output):
    checkpoint = torch.load(input, 'cpu')
    preserved_keys = ['model', 'args', 'vocabularies']
    remove_keys = [key for key in checkpoint if key not in preserved_keys]
    for key in remove_keys:
        del checkpoint[key]
    torch.save(checkpoint, output)


def main(args):
    watch_dir = Path(args.watch_dir)
    watch_log = Path(args.watch_log) if args.watch_log else None
    slimmed = []
    mod_time = watch_log.stat().st_mtime if watch_log else 0
    while True:
        ckps = checkpoint_util.list_checkpoints(watch_dir)
        if ckps:
            ckps.sort(key=lambda c: c.global_step)
            latest = ckps[-1]
            best_ckps = list(filter(lambda c: c.score is not None, ckps))
            best = None
            if best_ckps:
                best_ckps.sort(key=lambda c: c.score)
                best = best_ckps[-1]
            for ckp in ckps:
                if not (ckp in slimmed or ckp in (latest, best)):
                    slim(watch_dir / ckp.filename, watch_dir / ckp.filename)
                    slimmed.append(ckp)
        time.sleep(args.interval)
        new_mtime = watch_log.stat().st_mtime if watch_log else 0
        if new_mtime == mod_time:
            break
        mod_time = new_mtime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch-dir', type=str)
    parser.add_argument('--watch-log', type=str)
    parser.add_argument('--interval', type=int, default=1)
    args = parser.parse_args()
    main(args)
