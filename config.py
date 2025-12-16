"""
config.py

Configuration settings and hyperparameters for MrityunjayaGPT.
Centralizing these values allows for easy tuning and experimentation.
"""
import os

# --- Hardware Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Tokenizer and Data ---
VOCAB_SIZE = 50257  # Standard GPT-2 BPE vocabulary size
BLOCK_SIZE = 256    # Context window size (max sequence length)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# --- Model Hyperparameters (GPT-Small) ---
GPT_CONFIG = {
    "n_layer": 6,          # Number of transformer blocks
    "n_head": 6,           # Number of attention heads
    "n_embd": 384,         # Embedding dimension (d_model)
    "dropout": 0.2,        # Dropout rate for regularization
    "bias": False,         # Whether to use bias in linear layers
    "vocab_size": VOCAB_SIZE,
    "block_size": BLOCK_SIZE,
}

# --- Training Configuration ---
MAX_ITERS = 5000       # Total number of training steps
EVAL_INTERVAL = 500    # How often to run validation
LOG_INTERVAL = 100     # How often to log loss to console
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-1
GRADIENT_CLIP = 0.5
BATCH_SIZE = 64        # Training batch size
GRADIENT_ACCUMULATION_STEPS = 1 # Use 1 for standard, increase for larger effective batch size

# --- Checkpoint Configuration ---
CHECKPOINT_PATH = "best_model_params.pt"
