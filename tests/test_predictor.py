from pathlib import Path

import numpy as np
import pytest

from app.predictor import InvalidImageError, build_result

SAMPLES = Path(__file__).parent / "samples"
NAMES = ["Apple", "Avocado", "Banana", "Eggplant", "Tomato"]


def test_model_loads_and_class_count_matches(predictor):
    assert predictor.model.output_shape[-1] == len(predictor.class_names) == 5


def test_class_order_is_the_training_order(predictor):
    assert predictor.class_names == NAMES


def test_preprocess_shape_and_range(predictor):
    arr = predictor.preprocess((SAMPLES / "Apple.jpg").read_bytes())
    w, h = predictor.img_size
    assert arr.shape == (1, h, w, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


@pytest.mark.parametrize("fruit", NAMES)
def test_valid_fruit_image_returns_prediction(predictor, fruit):
    result = predictor.predict((SAMPLES / f"{fruit}.jpg").read_bytes())
    assert set(result["probabilities"]) == set(NAMES)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01
    assert result["prediction"] == fruit


def test_non_image_bytes_are_rejected(predictor):
    with pytest.raises(InvalidImageError):
        predictor.predict(b"this is definitely not an image")


# --- the 60% rule (pure function, no model needed) -------------------------

def test_confident_prediction_is_not_unknown():
    r = build_result([0.05, 0.02, 0.90, 0.02, 0.01], NAMES)
    assert r["prediction"] == "Banana" and r["is_unknown"] is False


def test_low_confidence_becomes_unknown_but_keeps_probabilities():
    r = build_result([0.30, 0.10, 0.45, 0.10, 0.05], NAMES)
    assert r["prediction"] == "unknown" and r["is_unknown"] is True
    assert r["best_guess"] == "Banana"
    assert r["probabilities"]["Banana"] == 0.45


def test_threshold_is_inclusive_at_60_percent():
    assert build_result([0.60, 0.10, 0.10, 0.10, 0.10], NAMES)["is_unknown"] is False
    assert build_result([0.59, 0.11, 0.10, 0.10, 0.10], NAMES)["is_unknown"] is True


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        build_result([0.5, 0.5], NAMES)
