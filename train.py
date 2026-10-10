
import torch

from config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    CHECKPOINT_PATH,
    D_MODEL,
    DROPOUT,
    EVAL_BATCHES,
    EVAL_INTERVAL,
    LEARNING_RATE,
    MAX_STEPS,
    N_HEADS,
    N_LAYERS,
    WEIGHT_DECAY,
    DEVICE,
)
from dataset import get_batch, load_dataset
from model import TinyLLM


@torch.no_grad()
def estimate_loss(model, train_tokens, val_tokens):
    model.eval()
    results = {}

    for name, tokens in [
        ("train", train_tokens),
        ("val", val_tokens),
    ]:
        losses = torch.zeros(EVAL_BATCHES)

        for i in range(EVAL_BATCHES):
            x, y = get_batch(tokens)
            _, loss = model(x, y)
            losses[i] = loss.item()

        results[name] = losses.mean().item()

    model.train()
    return results


def main():
    torch.manual_seed(42)

    tokenizer, train_tokens, val_tokens = load_dataset()

    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    parameters = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Parameters: {parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for step in range(MAX_STEPS):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(
                model, train_tokens, val_tokens
            )
            print(
                f"Step {step:5d} | "
                f"train loss: {losses['train']:.4f} | "
                f"val loss: {losses['val']:.4f}"
            )

        x, y = get_batch(train_tokens)

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Limit unusually large gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    # Save model weights and the exact tokenizer vocabulary.
    torch.save(
        {
            "model_state": model.state_dict(),
            "chars": tokenizer.chars,
            "config": {
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "block_size": BLOCK_SIZE,
                "dropout": DROPOUT,
            },
        },
        CHECKPOINT_PATH,
    )

    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
