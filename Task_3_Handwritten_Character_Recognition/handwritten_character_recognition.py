"""
CodeAlpha ML Internship - Task 3: Handwritten Character Recognition
Dataset: MNIST handwritten digits
Model: Convolutional Neural Network (CNN)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to [0, 1] and add channel dimension.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print("Training data:", x_train.shape)
    print("Testing data:", x_test.shape)

    model = build_model()
    model.summary()

    history = model.fit(
        x_train, y_train,
        validation_split=0.10,
        epochs=5,
        batch_size=128,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save model
    model.save("mnist_cnn_model.keras")

    # Plot training history
    plt.figure()
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=200)
    plt.close()

    # Predictions and classification report
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("MNIST Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.close()

    # Show sample predictions
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].squeeze(), cmap="gray")
        plt.title(f"Actual: {y_test[i]}\nPred: {y_pred[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=200)
    plt.close()

    print("\nFiles created:")
    print("- mnist_cnn_model.keras")
    print("- accuracy_curve.png")
    print("- confusion_matrix.png")
    print("- sample_predictions.png")

if __name__ == "__main__":
    main()
