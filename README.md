# AI Resume Screener

Rank resumes against a job description using **Claude**, **MongoDB Atlas Vector Search**, and **Python**.

```
                 ingest (once)                                   screen (per job)
┌────────────┐   ┌──────────────┐   ┌──────────┐      ┌──────────────┐   ┌──────────────┐
│ Kaggle CSV │──▶│ clean + PII  │──▶│ Claude   │      │ job desc     │──▶│ Voyage query │
│ 2,483 rows │   │ redaction    │   │ parse to │      └──────────────┘   │ embedding    │
└────────────┘   └──────────────┘   │ schema   │                         └──────┬───────┘
                                    └────┬─────┘                                ▼
                                         ▼                         ┌─────────────────────┐
                                  ┌─────────────┐   ┌──────────┐   │ Atlas $vectorSearch │
                                  │ Voyage doc  │──▶│ MongoDB  │◀──│ top-k candidates    │
                                  │ embedding   │   │ Atlas    │   └──────────┬──────────┘
                                  └─────────────┘   └──────────┘              ▼
                                                                   ┌─────────────────────┐
                                                                   │ Claude fit scoring  │
                                                                   │ (parallel, 0-100 +  │
                                                                   │  rationale) → top 5 │
                                                                   └─────────────────────┘
```

1. **Parse**: Claude (structured outputs) turns raw resume text into a typed `ParsedResume`: headline, summary, skills, years, roles, education. Emails, phones and URLs are redacted first.
2. **Embed and store**: a summary of each parsed resume is embedded with Voyage AI (`voyage-3.5`, 1024-dim) and stored in MongoDB Atlas.
3. **Retrieve**: the job description is embedded as a query; Atlas `$vectorSearch` returns the nearest candidates, optionally filtered by category.
4. **Score**: Claude scores each candidate's fit (0-100) with matched and missing skills and a short rationale, then candidates are re-ranked.
5. **Measure**: every request logs cost and latency; a hand-labeled benchmark measures top-5 agreement with a human.

## Results

| | |
|---|---|
| Resumes ingested | **520** across 24 categories ([Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)) |
| Parse cost | ~$0.0007 per resume (Claude Haiku 5.5), ~$0.40 for all 520 |
| Screening cost | RESULTS_COST per request (10 candidates retrieved and scored) |
| Screening latency | RESULTS_LATENCY |
| Top-5 agreement | Run `screener eval` after labeling (see [Benchmark](#benchmark)) |

### Model choice

Measured on the same 10 candidates for a backend-engineer job:

| Scoring model | One call | Parallel, groups of 2 |
|---|---|---|
| Claude Opus 5.5 | 10.5 s, $0.063 | 4.0 s, $0.083 |
| Claude Sonnet 5.5 | 8.2 s, $0.033 | 4.8 s, $0.046 |
| **Claude Haiku 5.5** | 6.5 s, $0.002 | **3.0 s, $0.002** |

Latency is driven by output length, so scoring candidates in small parallel groups cut latency by roughly half to two thirds. Haiku 5.5 is the default for parsing and scoring; set `CLAUDE_MODEL` in `.env` to use a stronger model.

## Setup

Requires Python 3.10+, a MongoDB Atlas cluster (the free M0 tier works), an Anthropic API key and a Voyage AI key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env    # fill in ANTHROPIC_API_KEY, VOYAGE_API_KEY, MONGODB_URI
```

Download `Resume.csv` from the [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) into `data/raw/Resume.csv`.

> Voyage's free tier without a payment method is limited to 3 requests/min. The code waits and retries, but ingest takes ~18 minutes instead of ~1. Adding a payment method keeps the 200M free tokens and removes the limit.

## Usage

```bash
screener ingest --n 520        # parse, embed and store resumes (cached, safe to re-run)
screener init-index            # create/update the Atlas Vector Search index

screener screen --title "Senior Accountant" \
  --description "GAAP, month-end close, reconciliations, QuickBooks. CPA preferred."

screener stats                 # cost and latency summary of logged requests
```

## Benchmark

`data/benchmark/` holds 8 job descriptions (`jobs.json`) and a fixed pool of 100 resumes (`pool.txt`, balanced across categories). Searches during evaluation are restricted to that pool.

```bash
screener label                 # grade ~12 candidates per job: 0 none, 1 weak, 2 good, 3 strong
screener eval                  # top-5 agreement, cost and latency p95
```

**Top-5 agreement** for a job is the share of the screener's top-5 picks that the human graded as a match (grade ≥ 2), divided by `min(5, human matches)` so a job with only three good candidates in the pool can still reach 100%. It is reported for vector search alone and after Claude re-ranking, which shows what the LLM step adds. Candidates are shown to the labeler in shuffled order so the vector ranking doesn't bias the grades.

## Design notes

- **Structured outputs** (`client.beta.messages.parse` with Pydantic models) guarantee schema-valid JSON for both parsing and scoring; no prompt-and-pray JSON parsing.
- **Fairness**: the parser omits name, age, gender and ethnicity from summaries, contact details are redacted before any model call, and the scoring rubric tells the model to judge skills and experience only.
- **Robustness**: the parse cache is keyed by resume id and content hash and written atomically; batch failures are recorded, not fatal; ingest stores in chunks so progress survives errors; the scorer re-asks for any candidate the model skipped and drops ids it invented.
- **Refusal fallback**: for models that support it (Opus, Sonnet, Fable), requests opt into server-side fallback so a declined request is retried on another model.

## Project layout

```
src/resume_screener/
  config.py        settings from .env
  schemas.py       Pydantic models (ParsedResume, FitScore, ...)
  dataset.py       Kaggle CSV loader + stratified sampling
  text.py          normalization + PII redaction
  parser.py        Claude resume parser
  batch.py         parallel parsing with on-disk cache
  embeddings.py    Voyage embeddings with rate-limit handling
  db.py            Atlas connection + resume repository
  vector_index.py  Atlas Vector Search index definition
  ingest.py        end-to-end ingest pipeline
  retrieval.py     $vectorSearch retrieval
  scorer.py        Claude fit scoring
  screener.py      retrieve + score service
  metrics.py       cost/latency tracking
  benchmark.py     benchmark files and labels
  evaluate.py      top-5 agreement
  cli.py           `screener` command
tests/             unit tests (no network): `pytest`
```
