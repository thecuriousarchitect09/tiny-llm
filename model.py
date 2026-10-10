
import torch
import torch.nn as nn
import torch.nn.functional as F

from embeddings import TokenPositionEmbedding
from transformer_block import TransformerBlock


class TinyLLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        block_size,
        dropout,
    ):
        super().__init__()

        self.block_size = block_size

        self.embeddings = TokenPositionEmbedding(
            vocab_size, d_model, block_size, dropout
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Share the token embedding and output projection weights.
        self.lm_head.weight = self.embeddings.token_embedding.weight

    def forward(self, token_ids, targets=None):
        x = self.embeddings(token_ids)
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch_size, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, vocab_size),
                targets.reshape(batch_size * seq_len),
            )

        return logits, loss
