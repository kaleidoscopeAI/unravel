# Unravel

Minimal build/run steps:
1) `python -m venv .venv`
2) Activate: `source .venv/bin/activate` (Unix/macOS) or `.venv\Scripts\activate` (Windows)
3) `pip install -r requirements.txt`
4) From the repository root: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`
