from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

from src.config import (
    INPUT_SIZE,
    HIDDEN_LAYER_1_UNITS,
    HIDDEN_LAYER_2_UNITS,
    NUM_CLASSES,
    HIDDEN_ACTIVATION,
    OUTPUT_ACTIVATION,
)


def build_model():
    model = Sequential([
        Input(shape=(INPUT_SIZE,)),

        Dense(HIDDEN_LAYER_1_UNITS, activation=HIDDEN_ACTIVATION),
        Dense(HIDDEN_LAYER_2_UNITS, activation=HIDDEN_ACTIVATION),

        Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION)
    ])

    return model