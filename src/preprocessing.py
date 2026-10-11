import numpy as np
from tensorflow.keras.utils import to_categorical

from src.config import (
    INPUT_SIZE,
    NUM_CLASSES,
)


def preprocess_data(X_train, y_train, X_test, y_test):
    # Flatten images from 28x28 into 784 pixel vectors
    X_train = X_train.reshape(-1, INPUT_SIZE)
    X_test = X_test.reshape(-1, INPUT_SIZE)

    # Convert pixel values from [0, 255] to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Convert digit labels into one-hot encoded vectors
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    print("Preprocessing completed.")
    print(f"Training images: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test images: {X_test.shape}")
    print(f"Test labels: {y_test.shape}")

    return X_train, y_train, X_test, y_test