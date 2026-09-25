# Contributing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test   # hermetic, no data/ needed
make lint
```

1. **No large files in git**: `data/`, `models/`, `mlruns/` stay ignored.
   Only code, docs, and small figures (`docs/*.png`) are versioned.
2. **Tests import real code** (`src.*`) and run on synthetic images —
   the suite must pass with no dataset present.
3. **Lint clean** (`ruff check src api tests`) before every push.
4. **Docs with behavior changes**: `/predict` changes update README,
   the demo runbook (`notebooks/10_wrapup.ipynb`), and the model card.
