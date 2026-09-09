"""
Training Pipeline for Eye State Convolutional Neural Network (CNN).
Trains a binary classification model (Open vs. Closed Eyes) on empirical video frame crops.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.cnn_classifier import build_eye_cnn

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "eye_dataset")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_dataset(target_size=(64, 64)):
    print(f"[1/4] Loading eye dataset from: {DATA_DIR}")
    images = []
    labels = []

    closed_files = glob.glob(os.path.join(DATA_DIR, "Closed", "*.*"))
    open_files = glob.glob(os.path.join(DATA_DIR, "Open", "*.*"))

    for f in closed_files:
        img = cv2.imread(f)
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img.astype(np.float32) / 255.0)
            labels.append(0)  # 0 = Closed

    for f in open_files:
        img = cv2.imread(f)
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img.astype(np.float32) / 255.0)
            labels.append(1)  # 1 = Open

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    print(f"  -> Total Loaded: {len(X)} images ({len(closed_files)} Closed, {len(open_files)} Open)")
    return X, y


def train():
    print("=" * 65)
    print("  TRAINING CNN EYE-STATE CLASSIFIER (TensorFlow / Keras)")
    print("  Fulfilling: 'leveraging CNN-based eye-state classification on video frames'")
    print("=" * 65)

    X, y = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[2/4] Split data: {len(X_train)} Train, {len(X_val)} Validation")
    model = build_eye_cnn(input_shape=(64, 64, 3))
    model.summary()

    save_path = os.path.join(MODELS_DIR, "eye_state_cnn.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True)
    ]

    print("\n[3/4] Training CNN for up to 15 epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[4/4] Evaluating CNN model on validation test set...")
    val_probs = model.predict(X_val, verbose=0).ravel()
    val_preds = (val_probs >= 0.5).astype(int)

    acc = np.mean(val_preds == y_val)
    auc = roc_auc_score(y_val, val_probs)
    print(f"\n--- Validation Performance ---")
    print(f"  Test Accuracy: {acc * 100:.2f}%")
    print(f"  ROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, target_names=["Closed", "Open"]))

    # Plot and save training history using Matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history["accuracy"], label="Train Acc", color="#0284C7")
    ax1.plot(history.history["val_accuracy"], label="Val Acc", color="#10B981")
    ax1.set_title("CNN Accuracy Progression")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train Loss", color="#EF4444")
    ax2.plot(history.history["val_loss"], label="Val Loss", color="#F59E0B")
    ax2.set_title("CNN Loss Progression")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    curve_path = os.path.join(MODELS_DIR, "cnn_training_curves.png")
    plt.savefig(curve_path, dpi=120)
    plt.close(fig)
    print(f"Saved training curves to: {curve_path}")
    print(f"Model successfully saved to: {save_path}")


if __name__ == "__main__":
    train()
