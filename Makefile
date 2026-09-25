.PHONY: install test lint train serve docker-build docker-run clean

install:            ## Install full local stack
	.venv/bin/pip install -r requirements.txt

test:               ## Hermetic test suite (no data/ needed)
	.venv/bin/python -m pytest tests/ -q

lint:               ## Lint everything
	.venv/bin/ruff check src api tests

train:              ## Baseline -> transfer learning -> fine-tune (needs data/)
	.venv/bin/python src/train_baseline.py
	.venv/bin/python src/train_transfer.py
	.venv/bin/python src/finetune.py

serve:              ## API on :8000
	.venv/bin/python api/main.py

docker-build:       ## Build serving image (serving deps only)
	docker build -t pneumonia-api .

docker-run:         ## Run serving image on host :8001 (host :8000 often taken)
	docker run -p 8001:8000 pneumonia-api

clean:              ## Caches only (never data/, models/, mlruns/)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
