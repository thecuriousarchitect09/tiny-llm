
import torch
from config import (
    DATA_PATH,
    BLOCK_SIZE,
    BATCH_SIZE,
    TRAIN_SPLIT,
    DEVICE,
)
from tokenizer import CharTokenizer


def load_dataset():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    if len(text) < BLOCK_SIZE + 2:
        raise ValueError(
            f"Add more training text to {DATA_PATH}. "
            f"Need at least {BLOCK_SIZE + 2} characters."
        )

    tokenizer = CharTokenizer(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split = int(len(tokens) * TRAIN_SPLIT)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    if len(val_tokens) < BLOCK_SIZE + 1:
        raise ValueError("Dataset is too small for the validation split.")

    return tokenizer, train_tokens, val_tokens


def get_batch(tokens):
    # Randomly choose starting positions.
    starts = torch.randint(
        0, len(tokens) - BLOCK_SIZE, (BATCH_SIZE,)
    )

    x = torch.stack([
        tokens[i:i + BLOCK_SIZE] for i in starts
    ])
    y = torch.stack([
        tokens[i + 1:i + BLOCK_SIZE + 1] for i in starts
    ])

    return x.to(DEVICE), y.to(DEVICE)
