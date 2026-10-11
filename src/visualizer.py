import matplotlib.pyplot as plt

from src.config import OUTPUTS_DIR


def plot_training_history(history):
    # Create outputs directory if it doesn't exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUTS_DIR / "training_loss.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUTS_DIR / "training_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Training plots saved to: {OUTPUTS_DIR}")