"""
NOTE: ONNX doesn't support model exporting with non-tensor
args and return values in `forward()` function.
"""
import abc
from typing import List, Dict

import torch.nn as nn
import torch.nn.functional as F

import thseq
import thseq.search
import thseq.utils as utils
from thseq.data import Vocabulary


class _Encoder(nn.Module, abc.ABC):
    def __init__(self, args, embedding, vocabulary: Vocabulary = None):
        super().__init__()
        self.args = args
        self.vocabulary = vocabulary
        self.embedding = embedding

    def reset_parameters(self):
        ...

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError


class _Decoder(nn.Module, abc.ABC):
    def __init__(self, args, embedding, vocabulary: Vocabulary = None):
        super().__init__()
        self.logit = nn.Linear(embedding.embedding_dim, embedding.num_embeddings)

        if args.share_decoder_input_output_embedding or args.share_all_embedding:
            self.logit.weight = embedding.weight

        self.args = args
        self.vocabulary = vocabulary
        self.embedding = embedding

    def reset_parameters(self):
        ...

    @abc.abstractmethod
    def forward(self, y, state):
        raise NotImplementedError


class _Model(nn.Module, abc.ABC):
    def __init__(self, args, vocabularies: List[Vocabulary], *argv, **kwargs):
        super().__init__()
        self.args = args
        self.source_vocab: Vocabulary = vocabularies[0]
        self.target_vocab: Vocabulary = vocabularies[1]

    @classmethod
    def build(cls, args, vocabularies):
        model = cls(args, vocabularies)
        model.initialize()
        return model

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--input-size', type=int, default=512, help='(default: 512)')
        parser.add_argument('--hidden-size', type=int, default=512, help='(default: 512)')
        parser.add_argument('--share-encoder-decoder-input-embedding', type=int, default=0, help='(default: 0)')
        parser.add_argument('--share-decoder-input-output-embedding', type=int, default=0, help='(default: 0)')
        parser.add_argument('--share-all-embedding', type=int, default=0, help='(default: 0)')
        parser.add_argument('--force-share', type=int, default=0, help='(default: 0)')
        parser.add_argument('--relative-pos-k', metavar='K', type=int, default=0,
                            help='Adds relative position with window size K. (default: 0)')

    @abc.abstractmethod
    def encode(self, x):
        """
        Encode x as representation
        :param x:
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, y, state):
        """
        Decode given preceding tokens and states
        :param y:
        :param state:
        :return: a tuple containing logit of shape (B x T x V)
        and a state dictionary maintaining decoding state
        """
        raise NotImplementedError

    def forward(self, inputs):
        """
        Compute loss
        :param inputs: a dictionary
        :return:
        """
        x, y = inputs['x'], inputs['y']

        state = self.encode(x)
        # reuses eos as bos to left-pad the target.
        y = F.pad(y, [1, 0], mode='constant', value=self.target_vocab.eos_id)
        logit, _ = self.decode(y[:, :-1], state)
        return logit

    @thseq.inference_mode(True)
    def infer(
            self, x, beam_width,
            penalty: str = None,
            alpha: float = 1.0,
            step_wise_penalty: bool = None,
            min_len_a: float = 0,
            min_len_b: float = 1,
            max_len_a: float = 0,
            max_len_b: float = 200,
            block_ngram: float = 0,
            suppress_unk: bool = False,
            topk=1
    ) -> List[List[Dict]]:
        self.eval()

        def fn(y, state):
            logit, state = self.decode(y, state)
            return utils.log_softmax(logit, -1), state

        state = self.encode(x)
        src_mask = x.ne(self.source_vocab.pad_id) & x.ne(self.source_vocab.eos_id)
        src_lens = src_mask.sum(1)
        results = thseq.search.beamsearch.generate(
            fn, state,
            src_lens=src_lens, batch_size=x.size(0),
            beam_width=beam_width,
            target_vocab=self.target_vocab,
            penalty=penalty,
            alpha=alpha,
            step_wise_penalty=step_wise_penalty,
            min_len_a=min_len_a,
            min_len_b=min_len_b,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            block_ngram=block_ngram,
            suppress_unk=suppress_unk,
            topk=topk
        )

        return results

    def initialize(self):
        """
        Initialize model parameters
        :return:
        """
        raise NotImplementedError
