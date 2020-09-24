import torch
import torch.nn as nn
import torch.nn.functional as F

import thseq
from thseq.data import Vocabulary
from thseq.models.abs import _Decoder
from thseq.modules import SinusoidalPositionEmbedding, MultiHeadAttention, PositionWiseFeedForward, maybe_norm


def transpose(hs):
    return tuple(h.transpose(0, 1) for h in hs)


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_size = args.hidden_size
        attention_hidden_size = args.attention_hidden_size
        ffn_hidden_size = args.ffn_hidden_size
        self.post_norm = args.decoder_post_norm
        self.ln0 = nn.LayerNorm(hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.encoder_decoder_attention = MultiHeadAttention(
            attention_hidden_size or hidden_size,
            args.num_heads,
            False,
            q_size=hidden_size, k_size=hidden_size, v_size=hidden_size,
            output_size=hidden_size,
            dropout=args.attention_dropout
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = PositionWiseFeedForward(
            hidden_size,
            ffn_hidden_size,
            hidden_size,
            args.ffn_dropout,
            nonlinear=args.decoder_nonlinear
        )
        self.residual_dropout = args.residual_dropout

    def forward(self, x, encoder_output, state):
        H = encoder_output['H']
        H_mask = encoder_output['mask']
        if state is not None:
            state['encoder'] = state.get('encoder', {})

        residual = x
        x = maybe_norm(self.ln0, x, True, self.post_norm)
        prev_h = None
        if state is not None and 'prev_h' in state:
            prev_h = transpose(state['prev_h'])
        x, prev_h = self.rnn(x, prev_h)
        if state is not None:
            state['prev_h'] = transpose(prev_h)
        x = F.dropout(x, self.residual_dropout, self.training)
        x = residual + x
        x = maybe_norm(self.ln0, x, False, self.post_norm)

        residual = x
        x = maybe_norm(self.ln1, x, True, self.post_norm)
        encoder_state = state['encoder'] if state is not None else None
        x = self.encoder_decoder_attention(
            q=x, k=H, v=H, mask=H_mask,
            state=encoder_state,
        )
        x = F.dropout(x, self.residual_dropout, self.training)
        x = residual + x
        x = maybe_norm(self.ln1, x, False, self.post_norm)

        residual = x
        x = maybe_norm(self.ln2, x, True, self.post_norm)
        x = self.ffn(x)
        x = F.dropout(x, self.residual_dropout, self.training)
        x = residual + x
        x = maybe_norm(self.ln2, x, False, self.post_norm)
        return x


class Decoder(_Decoder):
    def __init__(self, args, embedding, vocabulary: Vocabulary = None):
        super().__init__(args, embedding, vocabulary)
        hidden_size = args.hidden_size
        num_layers = args.num_decoder_layers

        self.pe = None
        if args.decoder_embedding_add_pos:
            self.pe = SinusoidalPositionEmbedding(hidden_size)
        self.scaling = hidden_size ** 0.5
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(num_layers)])
        self.post_norm = args.decoder_post_norm
        self.ln = None
        if not self.post_norm:
            self.ln = nn.LayerNorm(hidden_size)
        self.residual_dropout = args.residual_dropout

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._self_atn_mask = torch.empty(0, 0, 0)

    def forward(self, y, state):
        """
        Args:
            y: (B, T)
            state: a dictionary.
        Returns:
            (logit, state)
        """
        pe = None
        if self.pe is not None:
            pe = self.pe(y)
        if thseq.is_inference():
            y = self.embedding(y[:, -1:])
            pe = pe[:, -1:]
        else:
            y = self.embedding(y)
        if pe is not None:
            y = y * self.scaling + pe

        y = F.dropout(y, self.residual_dropout, self.training)
        y = y.transpose(1, 0)
        for i, layer in enumerate(self.layers):
            key = f'l{i}'
            state[key] = state.get(key, {})
            y = layer(
                y,
                encoder_output=state['encoder'],
                state=state[key] if thseq.is_inference() else None
            )
        if thseq.is_inference():
            del state['encoder']['H']
            state['encoder']['H'] = None
        if self.ln is not None:
            y = self.ln(y)
        y = y.transpose(1, 0)
        y = self.logit(y)
        return y, state
