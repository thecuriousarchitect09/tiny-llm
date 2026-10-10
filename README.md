# Tiny LLM From Scratch

A small GPT-style, decoder-only Transformer implemented in Python and PyTorch.

The project demonstrates how a language model tokenizes text, learns contextual representations, predicts the next token, and generates text autoregressively.

## Architecture

1. **Tokenizer** — maps characters to integer IDs.
2. **Dataset** — creates shifted input and target sequences.
3. **Embeddings** — combines token and position representations.
4. **Causal self-attention** — lets tokens attend to preceding tokens.
5. **Feed-forward network** — transforms token representations.
6. **Transformer block** — combines attention, normalization, and residual connections.
7. **Language model** — predicts vocabulary logits and computes cross-entropy loss.
8. **Training loop** — optimizes model parameters with AdamW.
9. **Generation** — samples tokens repeatedly to produce text.

## Requirements

- Python 3.10+
- PyTorch 2.0+

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

## Prepare the dataset

Place your training corpus in `data/input.txt`.

Use enough text for both the training and validation splits. The tokenizer vocabulary is built from the entire corpus, while the actual training examples come from the training split.

## Train

From the project root:

```bash
python train.py
```

The script prints training and validation losses and saves the model checkpoint to `tiny_llm.pt`.

## Generate text

```bash
python generate.py
```

Enter a prompt made from characters present in the training corpus.

## Configuration

Edit `config.py` to change:

- `D_MODEL`: embedding dimension
- `N_HEADS`: number of attention heads
- `N_LAYERS`: number of Transformer blocks
- `BLOCK_SIZE`: maximum context length
- `BATCH_SIZE`: examples per training batch
- `MAX_STEPS`: number of training updates
- `LEARNING_RATE`: optimizer learning rate
- `TEMPERATURE`: sampling randomness

Ensure `D_MODEL` is divisible by `N_HEADS`.

## Limitations

This is an educational implementation, not a production language model. Character-level tokenization is inefficient for natural language, and a small dataset and model will produce limited results.

The project does not use pretrained weights or an external language model API. PyTorch provides tensor operations, automatic differentiation, and optimization primitives.

## Learning objective

Understand the complete path from raw text to token IDs, Transformer computations, next-token loss, parameter updates, and autoregressive generation.