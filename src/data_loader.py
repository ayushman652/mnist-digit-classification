

from tensorflow.keras.datasets import mnist


def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print("Dataset loaded successfully.")
    print(f"Training images: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test images: {X_test.shape}")
    print(f"Test labels: {y_test.shape}")

    return X_train, y_train, X_test, y_test