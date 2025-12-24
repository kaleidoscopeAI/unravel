# Unravel

Minimal build/run steps:
1) `python -m venv .venv && source .venv/bin/activate`
2) `pip install -r requirements.txt` (points to backend-requirements.txt)
3) `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`
