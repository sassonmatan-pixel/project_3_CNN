from pathlib import Path

from app.main import MAX_UPLOAD_BYTES

SAMPLES = Path(__file__).parent / "samples"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert len(body["classes"]) == 5


def test_home_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert 'dir="rtl"' in r.text


def test_predict_valid_image(client):
    data = (SAMPLES / "Banana.jpg").read_bytes()
    r = client.post("/predict", files={"file": ("banana.jpg", data, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "Banana"
    assert body["is_unknown"] is False
    assert len(body["probabilities"]) == 5


def test_non_image_file_is_rejected(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_fake_image_with_image_content_type_is_rejected(client):
    r = client.post("/predict", files={"file": ("fake.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.status_code == 400


def test_empty_file_is_rejected(client):
    r = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_too_large_image_is_rejected(client):
    big = b"\xff\xd8\xff" + b"0" * (MAX_UPLOAD_BYTES + 10)
    r = client.post("/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413
