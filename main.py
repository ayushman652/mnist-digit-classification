from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import build_model
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.visualizer import plot_training_history

def main():
    # 1. Load MNIST dataset
    X_train, y_train, X_test, y_test = load_data()

    # 2. Preprocess images and labels
    X_train, y_train, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test
    )

    # 3. Build neural network
    model = build_model()
    model.summary()

    # 4. Train the model
    history = train_model(model, X_train, y_train)
    plot_training_history(history)

    # 5. Evaluate on the separate test set
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
