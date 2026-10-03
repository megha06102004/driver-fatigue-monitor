"""
Convolutional Neural Network (CNN) for Eye State Classification (Open vs. Closed).
Optimized binary classifier for real-time video frame eye patches.
"""

import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "eye_state_cnn.keras"
)
DEFAULT_META_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "cnn_metadata.json"
)


def build_eye_cnn(input_shape=(64, 64, 3)) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=input_shape, name="eye_image_input"),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.20),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.40),
        layers.Dense(1, activation="sigmoid", name="eye_state_prob")
    ], name="EyeState_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def crop_eye_region(frame: np.ndarray, landmark_points: np.ndarray, margin: float = 0.35, target_size=(64, 64)) -> np.ndarray:
    h, w = frame.shape[:2]
    x_min, y_min = np.min(landmark_points, axis=0).astype(int)
    x_max, y_max = np.max(landmark_points, axis=0).astype(int)

    box_w = max(1, x_max - x_min)
    box_h = max(1, y_max - y_min)
    pad_x = int(box_w * margin)
    pad_y = int(box_h * margin)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((*target_size, 3), dtype=np.float32)

    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


class EyeCNNClassifier:
    """Production runtime inference engine for the Eye State CNN."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, meta_path: str = DEFAULT_META_PATH):
        self.model_path = model_path
        self.model = None
        self.threshold = 0.5

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    self.threshold = meta.get("optimal_threshold", 0.5)
            except Exception:
                pass

        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"[INFO] Eye State CNN loaded successfully (Threshold={self.threshold:.4f}).")
        else:
            print(f"[WARN] CNN weights not found at {model_path}.")

    def predict_eye(self, eye_crop: np.ndarray):
        if self.model is None:
            return "Open", 1.0

        batch = np.expand_dims(eye_crop, axis=0)
        prob = float(self.model.predict(batch, verbose=0)[0][0])
        state = "Open" if prob >= self.threshold else "Closed"
        return state, prob
