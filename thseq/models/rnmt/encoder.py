import torch.nn as nn
import torch.nn.functional as F

from thseq.data import Vocabulary
from thseq.models.abs import _Encoder
from thseq.modules import SinusoidalPositionEmbedding, MultiHeadAttention, PositionWiseFeedForward, maybe_norm


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_size = args.hidden_size
        attention_hidden_size = args.attention_hidden_size
        ffn_hidden_size = args.ffn_hidden_size
        self.post_norm = args.encoder_post_norm
        self.ln0 = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            attention_hidden_size or hidden_size,
            args.num_heads,
            False,
            q_size=hidden_size, k_size=hidden_size, v_size=hidden_size,
            output_size=hidden_size,
            dropout=args.attention_dropout,
            relative_pos_k=args.relative_pos_k,
            relative_pos_v=bool(args.relative_pos_v)
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = PositionWiseFeedForward(
            hidden_size,
            ffn_hidden_size,
            hidden_size,
            args.ffn_dropout,
            nonlinear=args.encoder_nonlinear
        )
        self.residual_dropout = args.residual_dropout

    def forward(self, x, mask):
        residual = x
        x = maybe_norm(self.ln0, x, True, self.post_norm)
        x = self.self_attention(q=x, k=x, v=x, mask=mask)
        x = F.dropout(x, self.residual_dropout, self.training)
        x = residual + x
        x = maybe_norm(self.ln0, x, False, self.post_norm)

        residual = x
        x = maybe_norm(self.ln1, x, True, self.post_norm)
        x = self.ffn(x)
        x = F.dropout(x, self.residual_dropout, self.training)
        x = residual + x
        x = maybe_norm(self.ln1, x, False, self.post_norm)
        return x


class Encoder(_Encoder):
    def __init__(self, args, embedding, vocabulary: Vocabulary = None):
        super().__init__(args, embedding, vocabulary)
        hidden_size = args.hidden_size
        num_layers = args.num_encoder_layers

        self.pe = None
        if args.encoder_embedding_add_pos:
            self.pe = SinusoidalPositionEmbedding(hidden_size)
        self.scaling = hidden_size ** 0.5
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(num_layers)])
        self.ln = None
        if not args.encoder_post_norm:
            self.ln = nn.LayerNorm(hidden_size)
        self.residual_dropout = args.residual_dropout

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        """
        Args:
            x: (B, T)

        Returns:
            state: a dictionary.
        """
        mask = x.eq(self.embedding.padding_idx).unsqueeze(1)
        x = self.embedding(x)
        if self.pe is not None:
            x = x * self.scaling + self.pe(x)
        x = F.dropout(x, self.residual_dropout, self.training)
        x = x.transpose(1, 0)
        for layer in self.layers:
            x = layer(x, mask)
        if self.ln is not None:
            x = self.ln(x)
        state = {
            'encoder': {
                'H': x,
                'mask': mask
            }
        }
        return state
