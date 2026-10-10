
import torch

from config import (
    CHECKPOINT_PATH,
    DEVICE,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)
from model import TinyLLM
from tokenizer import CharTokenizer


def load_model():
    # Only load checkpoints you trust.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    chars = checkpoint["chars"]
    tokenizer = CharTokenizer("".join(chars))
    cfg = checkpoint["config"]

    model = TinyLLM(
        vocab_size=len(chars),
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        block_size=cfg["block_size"],
        dropout=cfg["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, tokenizer, cfg["block_size"]


@torch.no_grad()
def generate(model, tokenizer, prompt, block_size):
    if not prompt:
        raise ValueError("Prompt must not be empty.")

    context = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE
    )

    for _ in range(MAX_NEW_TOKENS):
        # Keep only the most recent context window.
        context_window = context[:, -block_size:]

        logits, _ = model(context_window)
        next_logits = logits[:, -1, :] / max(TEMPERATURE, 1e-5)

        probabilities = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        context = torch.cat([context, next_token], dim=1)

    return tokenizer.decode(context[0].tolist())


def main():
    model, tokenizer, block_size = load_model()

    prompt = input("Enter a prompt: ")
    print("\nGenerated text:\n")
    print(generate(model, tokenizer, prompt, block_size))


if __name__ == "__main__":
    main()
