
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # True entries are positions that must be masked.
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(block_size, block_size, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # (B, T, C) -> (B, heads, T, head_dim)
        def split_heads(tensor):
            return tensor.view(
                batch_size, seq_len,
                self.n_heads, self.head_dim
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Attention scores: Q multiplied by K transpose.
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Hide future tokens.
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        output = weights @ v

        # Merge attention heads.
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        return self.resid_dropout(self.proj(output))
