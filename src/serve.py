"""Day 38: image serving — class + confidence + Grad-CAM overlay.

POST /predict (multipart image upload) ->
  {"predicted_class", "confidence", "gradcam_png_base64"}.
The confidence lets radiologists apply their own threshold and spot
uncertain cases; the overlay shows *where* the model looked.
"""
import base64
import io
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import eval_transforms
from dataset import LABEL_NAMES
from gradcam import GradCAM, blend_heatmap
from models import mobilenet_frozen

MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_finetuned.pt"
TARGET_LAYER = "features.18"

app = FastAPI(title="Chest X-Ray Pneumonia API")

model = mobilenet_frozen()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True)["state_dict"])
model.eval()
model.requires_grad_(True)  # flags come frozen from the constructor (Day 36)
explainer = GradCAM(model, TARGET_LAYER)
preprocess = eval_transforms()


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    gradcam_png_base64: str


@app.get("/health")
def health():
    return {"status": "healthy"}


def encode_png(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):  # noqa: B008 - FastAPI idiom
    try:
        raw = await file.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001 - unreadable upload -> 400
        raise HTTPException(status_code=400, detail=f"unreadable image: {e}")
    try:
        with torch.enable_grad():
            x = preprocess(pil_img).unsqueeze(0)
            with torch.no_grad():
                proba = model(x).softmax(1)[0]
            cls = int(proba.argmax())
            cam = explainer.heatmap(x, target_class=cls)
        overlay = blend_heatmap(pil_img, cam)
        return PredictionResponse(
            predicted_class=LABEL_NAMES[cls],
            confidence=round(float(proba[cls]), 4),
            gradcam_png_base64=encode_png(overlay),
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - scoring failures -> 400, never 500
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
