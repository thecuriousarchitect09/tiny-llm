
import torch

# Reproducibility
SEED = 42
torch.manual_seed(SEED)

# Paths
DATA_PATH = "data/input.txt"
CHECKPOINT_PATH = "tiny_llm.pt"

# Tokenizer / dataset
BATCH_SIZE = 16
BLOCK_SIZE = 64  # Maximum context length
TRAIN_SPLIT = 0.9

# Transformer architecture
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1

# Training
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_STEPS = 3000
EVAL_INTERVAL = 300
EVAL_BATCHES = 20

# Generation
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
