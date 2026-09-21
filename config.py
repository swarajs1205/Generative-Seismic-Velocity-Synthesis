# Configuration for Seismic Diffusion Model

import torch

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training parameters
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 2e-4

# Dataset parameters
DATASET_LIMIT = 5000  # Limit dataset for faster training (None for all)

# Diffusion parameters
IMG_SIZE = 70
NOISE_STEPS = 300  # Reduced from 1000 for efficiency
BETA_START = 1e-4
BETA_END = 0.02

# Paths
DATA_DIR = "./data/raw/CurveVel"
RESULTS_DIR = "./results"
CHECKPOINT_DIR = "./checkpoints"
