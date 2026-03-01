import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data", "chest_xray")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Image parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Model saving
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.h5")