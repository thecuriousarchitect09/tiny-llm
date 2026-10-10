
import torch.nn as nn

from attention import CausalSelfAttention
from feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model, n_heads, block_size, dropout
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, dropout)

    def forward(self, x):
        # Pre-layer normalization + residual connection.
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x
