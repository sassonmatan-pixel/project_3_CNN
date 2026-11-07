"""Shared prediction core for Fruit Vision.

Both the FastAPI server (app/main.py) and the Streamlit app (streamlit_app.py)
use this module, so the preprocessing is guaranteed to be identical everywhere
and identical to what the model saw during training in Colab:

    RGB -> resize to 100x100 -> float array -> divide by 255 -> add batch dim
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.keras"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"

IMG_SIZE = (100, 100)  # same as IMG_SIZE in the training notebook
UNKNOWN_THRESHOLD = 0.60  # below this confidence we answer "unknown"
UNKNOWN_LABEL = "unknown"


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes are not a readable image."""


def build_result(probs, class_names, threshold: float = UNKNOWN_THRESHOLD) -> dict:
    """Turn the model's raw probabilities into the JSON-friendly result.

    Kept as a pure function (no model needed) so the 60% rule is easy to test.
    The original probabilities are always returned, even when the label is
    "unknown".
    """
    probs = [float(p) for p in probs]
    if len(probs) != len(class_names):
        raise ValueError(
            f"Model returned {len(probs)} values but there are {len(class_names)} class names"
        )
    best = max(range(len(probs)), key=probs.__getitem__)
    confidence = probs[best]
    is_unknown = confidence < threshold
    return {
        "prediction": UNKNOWN_LABEL if is_unknown else class_names[best],
        "best_guess": class_names[best],  # what the model leaned towards
        "confidence": round(confidence, 4),
        "is_unknown": is_unknown,
        "probabilities": {n: round(p, 4) for n, p in zip(class_names, probs)},
    }


class FruitPredictor:
    """Loads the Keras model once and classifies images."""

    def __init__(self, model_path: Path = MODEL_PATH, class_names_path: Path = CLASS_NAMES_PATH):
        # Quiet TensorFlow's very chatty startup logs (must be set before import).
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import keras  # imported here so importing this module stays fast

        with open(class_names_path, encoding="utf-8") as f:
            self.class_names: list[str] = json.load(f)
        self.model = keras.models.load_model(model_path)

        out = self.model.output_shape[-1]
        if out != len(self.class_names):
            raise RuntimeError(
                f"Model outputs {out} classes but class_names.json has {len(self.class_names)}"
            )

        # The input size comes from the model itself (100x100 for the old CNN, 224x224 for
        # MobileNetV2), so swapping model.keras never needs a code change here.
        h, w = self.model.input_shape[1:3]
        self.img_size = (int(w), int(h))  # PIL wants (width, height)

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """bytes -> array of shape (1, H, W, 3) with values in [0, 1]."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()  # force a real decode so corrupt files fail here
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("The file is not a valid image") from exc

        # Phones store rotation in EXIF metadata instead of rotating pixels.
        # Without this, a portrait photo would reach the model sideways.
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB").resize(self.img_size)
        arr = np.asarray(img, dtype="float32") / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, image_bytes: bytes) -> dict:
        batch = self.preprocess(image_bytes)
        probs = self.model.predict(batch, verbose=0)[0]
        return build_result(probs, self.class_names)
