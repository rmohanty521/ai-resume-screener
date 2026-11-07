# AI Resume Screener

Screen resumes against a job description using **Claude**, **MongoDB Atlas Vector Search**, and **Python**.

Pipeline:

1. **Parse** raw resume text into structured documents with Claude (structured outputs).
2. **Embed** each resume and store it in MongoDB Atlas.
3. **Retrieve** the closest candidates for a job description with `$vectorSearch`.
4. **Score** role fit for each candidate with Claude, with a short rationale.
5. **Evaluate** against a hand-labeled benchmark (top-5 agreement) and track cost and latency per request.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in your keys
```

## Status

Work in progress.
