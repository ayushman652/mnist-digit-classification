from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEP_LEARNING_ROOT = PROJECT_ROOT.parent
WORKSPACE_ROOT = DEEP_LEARNING_ROOT.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# Dataset configuration
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH  # 784 pixels

NUM_CLASSES = 10
VALIDATION_SIZE = 0.10


# Neural network configuration
HIDDEN_LAYER_1_UNITS = 784
HIDDEN_LAYER_2_UNITS = 100

HIDDEN_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "softmax"


# Training configuration
RANDOM_STATE = 42
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

OPTIMIZER = "adam"
LOSS_FUNCTION = "categorical_crossentropy"
METRICS = ["accuracy"]