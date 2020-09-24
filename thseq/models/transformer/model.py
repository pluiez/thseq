import torch.nn as nn

from thseq.models.abs import _Model
from .decoder import Decoder
from .encoder import Encoder
from .. import register


@register('transformer')
class Transformer(_Model):
    def __init__(self, args, vocabularies):
        super().__init__(args, vocabularies)
        source_vocab, target_vocab = self.source_vocab, self.target_vocab
        source_embedding = nn.Embedding(len(source_vocab), args.input_size, padding_idx=source_vocab.pad_id)
        target_embedding = nn.Embedding(len(target_vocab), args.input_size, padding_idx=target_vocab.pad_id)
        if args.share_encoder_decoder_input_embedding or args.share_all_embedding:
            if not args.force_share and len(vocabularies[0]) != len(vocabularies[1]):
                raise RuntimeError(f'Expected the same vocabulary size, '
                                   f'got ({len(vocabularies[0])}, {len(vocabularies[1])}). '
                                   f'Use --force-share 1 to force weight tying.')
            if len(vocabularies[0]) <= len(vocabularies[1]):
                source_embedding.weight = target_embedding.weight
            else:
                target_embedding.weight = source_embedding.weight

        self.encoder = Encoder(args, source_embedding, source_vocab)
        self.decoder = Decoder(args, target_embedding, target_vocab)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y, state):
        return self.decoder(y, state)

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--hidden-size', type=int, default=512)
        parser.add_argument('--ffn-hidden-size', type=int, default=2048)
        parser.add_argument('--residual-dropout', type=float, default=0.1)
        parser.add_argument('--ffn-dropout', type=float, default=0.0)
        parser.add_argument('--attention-dropout', type=float, default=0.0)
        parser.add_argument('--num-encoder-layers', type=int, default=6)
        parser.add_argument('--num-decoder-layers', type=int, default=6)
        parser.add_argument('--num-heads', type=int, default=8)
        parser.add_argument('--attention-hidden-size', type=int, default=0)
        parser.add_argument('--encoder-post-norm', type=int, default=0)
        parser.add_argument('--decoder-post-norm', type=int, default=0)
        parser.add_argument('--encoder-nonlinear', type=str, choices=['relu', 'swish', 'gelu'], default='relu')
        parser.add_argument('--decoder-nonlinear', type=str, choices=['relu', 'swish', 'gelu'], default='relu')

    def initialize(self):
        for e in [self.encoder.embedding, self.decoder.embedding]:
            nn.init.normal_(e.weight, mean=0, std=e.embedding_dim ** -0.5)
        nn.init.normal_(self.decoder.logit.weight, mean=0, std=self.args.hidden_size ** -0.5)
