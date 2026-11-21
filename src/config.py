from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directories
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Image settings
IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Device
DEVICE = "cpu"

print("Project configuration loaded successfully!")
print("Project path:", PROJECT_ROOT)
print("Device:", DEVICE)