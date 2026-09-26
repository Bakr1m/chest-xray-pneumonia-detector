FROM python:3.12-slim

WORKDIR /app

# Torch needs the OpenMP runtime on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Serving deps first for better layer caching (CPU-only torch via extra index).
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code only (no data/, notebooks/, tests/).
COPY src/ ./src/
COPY api/ ./api/

# Production model: fetched by exact release version, SHA256-verified.
# Never a moving tag, never baked from a developer laptop. Provenance:
# https://github.com/Bakr1m/chest-xray-pneumonia-detector/releases/tag/v1.0.0
ARG MODEL_TAG=v1.0.0
ARG MODEL_SHA256=5b3ed8cabc532ce262ad0b6d59262ac571cfb1f8fc809f204b6e10c525d20b1f
RUN mkdir -p models && \
    curl -fsSL -o models/mobilenet_finetuned.pt \
      "https://github.com/Bakr1m/chest-xray-pneumonia-detector/releases/download/${MODEL_TAG}/mobilenet_finetuned.pt" && \
    echo "${MODEL_SHA256}  models/mobilenet_finetuned.pt" | sha256sum -c - && \
    python -c "import torch; m=torch.load('models/mobilenet_finetuned.pt', map_location='cpu', weights_only=True); print('artifact OK:', sorted(m.keys()))"

EXPOSE 8000

CMD ["python", "api/main.py"]
