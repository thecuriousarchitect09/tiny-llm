
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)
