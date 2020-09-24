import datetime
import logging
import os
import sys
from typing import List

import lunas
import torch

import thseq.models as models
import thseq.models.ensemble as ensemble
import thseq.options as options
import thseq.utils as utils
from thseq.data.vocabulary import Vocabulary

logging.root.handlers = []
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s ', level=logging.DEBUG,
                    stream=sys.stderr)

logger = logging.getLogger('generate')


def load_single(state, verbose=False):
    vocabularies = state['vocabularies']
    args = options.add_default_model_args(state['args'])
    model = models.build(args, vocabularies)
    model.load_state_dict(state['model'])
    if verbose:
        logger.info(f'args: {args}')
    return model


def load(paths, select='best', n=1, verbose=False):
    states = []
    for c in paths:
        if os.path.isdir(c):
            state = utils.load_latest(c)
        else:
            state = utils.load_checkpoint(c)
        if not state:
            raise RuntimeError(f'Failed to load checkpoint from path: {c}')
        states.append(state)

    models = [load_single(state, verbose) for state in states]

    if len(models) == 1:
        model = models[0]
    else:
        model = ensemble.AverageLogProb(models)

    return model, states[0]['vocabularies']


def get_iterator(args, source_vocab: Vocabulary):
    max_tokens, buffer_size = args.max_tokens, args.buffer_size

    def map_fn(text):
        x = torch.as_tensor(source_vocab.lookup(text.split()))
        return {
            'x': x,
            'text': text,
            'size_x': x.size(0)
        }

    def collate_fn(samples: List):
        return {
            'text': utils.aggregate_values(samples, 'text'),
            'index': utils.aggregate_values(samples, 'index'),
            'x': utils.pack(utils.aggregate_values(samples, 'x'), source_vocab.pad_id, torch.long),
            'size_x': utils.aggregate_values(samples, 'size_x', reduce=sum)
        }

    if args.input == '-':
        ds = lunas.Stdin()
    else:
        ds = lunas.TextLine(args.input)
    ds = ds.map(map_fn)
    ds = lunas.Enumerate(ds)
    ds = ds.map(
        lambda i, x: {
            'index': i,
            'text': x['text'],
            'x': x['x'],
            'size_x': x['size_x']
        }, unpack_args=True
    )
    if buffer_size > 1:
        ds = ds.sort(buffer_size, key=lambda x: -x['size_x'])
    if args.max_sentences > 0:
        itr = lunas.ConstantIterator(
            ds,
            args.max_sentences
        )
    else:
        itr = lunas.BucketIterator(
            ds,
            max_tokens,
            lambda x: x['size_x'],
            lunas.get_bucket_boundaries(1, 8, 8, 4096),
            required_batch_size_multiple=8
        )
    return lunas.DataLoader(itr, args.num_workers, collate_fn=collate_fn)


class Translator(object):

    def __init__(self, infer_fn, bpe, reverse, topk, sv, verbose) -> None:
        super().__init__()
        self.infer_fn = infer_fn
        self.bpe = bpe
        self.reverse = reverse
        self.topk = topk
        self.sv = sv
        self.verbose = verbose

        self.cache = {}
        self.ptr = 0

    def translate_batch(self, batch, update_cache: bool):
        hypos = self.infer_fn(batch)

        cache = {}
        if update_cache:
            cache = self.cache
        for i, j in enumerate(batch['index']):
            x = batch['x'][i]
            cache[j] = {
                'index': j,
                'x': x,
                'hypos': hypos[i],  # a dictionary with keys ('tokens': List[str], 'score':torch.tensor)
            }
            if self.verbose:
                print(f's-{j}\t{" ".join(self.sv.revert(x))}\t{x.tolist()}', file=sys.stderr)
                for hyp in hypos[i]:
                    print(f'h-{j}\t{" ".join(hyp["tokens"])}\t{hyp["score"]}', file=sys.stderr)

        return cache

    def translate(self, batch):
        self.translate_batch(batch, update_cache=True)

        while self.ptr in self.cache:
            entry = self.cache.pop(self.ptr)
            x, hypos = entry['x'], entry['hypos']

            if self.topk == 1:
                try:
                    hyp_str = ' '.join(hypos[0]['tokens'])
                    sys.stdout.write(f'{hyp_str}\n')
                except IndexError as e:
                    print(entry)
                    raise e
            else:
                for hyp in hypos:
                    hyp_str = ' '.join(hyp['tokens'])
                    sys.stdout.write(f'{hyp["score"]}\t{hyp_str}\n')
            self.ptr += 1


def main(args):
    logger.info('Loading checkpoints ...')
    model, vocabularies = load(args.checkpoints, verbose=args.verbose)
    s_vocab, t_vocab = vocabularies
    model = utils.move_cuda(model)

    def infer(batch):
        batch = utils.move_cuda(batch)
        return model.infer(
            batch['x'], args.k,
            args.penalty,
            args.alpha,
            args.step_wise_penalty,
            args.min_len_a,
            args.min_len_b,
            args.max_len_a,
            args.max_len_b,
            topk=args.topk
        )

    translator = Translator(infer, args.bpe, args.reverse, args.topk, s_vocab, args.verbose)

    meter = utils.SpeedMeter()

    logger.info('Building iterator ...')
    it = get_iterator(args, s_vocab)

    n_tok = 0
    n_snt = 0
    meter.start()

    logger.info('Start generation ...')
    for batch in it:
        translator.translate(batch)
        n_tok += batch['size_x']
        n_snt += len(batch['index'])
        meter.stop(batch['size_x'])

    sys.stderr.write(
        f'Sentences = {n_snt}, Tokens = {n_tok}, \n'
        f'Time = {datetime.timedelta(seconds=meter.duration)}, \n'
        f'Speed = {meter.avg:.2f} tok/s, {n_snt / meter.duration:.2f} snt/s\n'
    )


if __name__ == '__main__':
    args = options.parse_generation_args()
    main(args)
