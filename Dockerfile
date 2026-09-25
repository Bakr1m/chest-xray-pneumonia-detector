FROM python:3.12-slim

WORKDIR /app

# Torch needs the OpenMP runtime on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Serving deps first for better layer caching (CPU-only torch via extra index).
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code + production model only (no data/, notebooks/, tests/).
COPY src/ ./src/
COPY api/ ./api/
COPY models/mobilenet_finetuned.pt ./models/mobilenet_finetuned.pt

EXPOSE 8000

CMD ["python", "api/main.py"]
