
from tensorflow.keras.optimizers import Adam

from src.config import (
    LEARNING_RATE,
    LOSS_FUNCTION,
    METRICS,
    EPOCHS,
    BATCH_SIZE,
)


def train_model(model, X_train, y_train):
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=LOSS_FUNCTION,
        metrics=METRICS,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=2,
    )

    return history
