"""API contract tests (in-process TestClient; synthetic PNG upload).

Hermetic by design: a tiny SmallCNN stands in for the production weights
(no models/ needed — that dir is gitignored), so these run anywhere.
Production wiring (real weights file) is verified by the Docker parity check,
not unit tests.
"""
import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import serve
from src.gradcam import GradCAM
from src.models import SmallCNN

app = serve.app


@pytest.fixture(autouse=True)
def tiny_model_stand_in():
    """Install a random SmallCNN so no test touches disk artifacts."""
    net = SmallCNN().eval()
    serve._model = net
    serve._explainer = GradCAM(net, "features.6")
    yield
    serve._model = None
    serve._explainer = None


client = TestClient(app)


def make_png(color=128, size=(200, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(color, color, color)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health():
    assert client.get("/health").json() == {"status": "healthy"}


def test_predict_returns_class_confidence_overlay():
    r = client.post(
        "/predict", files={"file": ("xray.png", make_png(), "image/png")}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_class"] in ("NORMAL", "PNEUMONIA")
    assert 0.0 <= body["confidence"] <= 1.0
    raw = base64.b64decode(body["gradcam_png_base64"])
    overlay = Image.open(io.BytesIO(raw))
    assert overlay.size == (200, 200)  # overlay matches uploaded resolution


def test_predict_rejects_non_image():
    r = client.post(
        "/predict", files={"file": ("x.txt", io.BytesIO(b"not an image"), "text/plain")}
    )
    assert r.status_code == 400
