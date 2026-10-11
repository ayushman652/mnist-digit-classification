
def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    return test_loss, test_accuracy
