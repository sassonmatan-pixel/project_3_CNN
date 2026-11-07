import os
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@pytest.fixture(scope="session")
def predictor():
    """Load the real model once for the whole test session."""
    from app.predictor import FruitPredictor
    return FruitPredictor()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:  # the 'with' runs the lifespan (loads the model)
        yield c
