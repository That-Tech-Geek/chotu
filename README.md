# chotu — Autonomous Negotiation Engine

Backend-only, ultra-low-latency negotiation API. **No LLM in the engine** — it
talks to external LLM agents over HTTP (`User ↔ LLM ↔ Negotiation API`).

## Architecture (serverless/Vercel, 250MB RAM)

Hot path lives on **Vercel Python functions** (`api/index.py`), so everything
is sized for a small bundle and no Docker:

| Concern | Plan | Serverless choice |
|---|---|---|
| API | FastAPI | FastAPI via Vercel Python runtime |
| Vector retrieval | FAISS-HNSW | **pgvector on Supabase** (server-side ANN) with in-memory `NumpyCosineIndex` fallback (capped corpus, RAM-safe) |
| State/cache | Redis | **Upstash Redis over REST** (httpx, no client dep) or in-memory LRU |
| Persistence | Supabase/Postgres | Supabase REST (`persistence/repository.py`) + `schema.sql` with RLS |
| Async work | workers/queues | **Vercel Cron** → `/v1/internal/cron/evolve` (guarded by `CRON_SECRET`), or Supabase Edge Functions |
| Bundle diet | numpy/scipy/faiss | **numpy + fastapi + httpx only**; scipy dropped (closed-form gaussian), faiss dropped, no torch/sklearn |

## Core loop

```
OBSERVE → MODEL → GENERATE → SIMULATE → OPTIMIZE → GATE → ACT → OBSERVE → LEARN → EVOLVE
```

## Modules (`src/economic_engine/`)

- `state/canonical.py` — merchant/supplier/product/negotiation/... canonical models
- `models/cost_engine.py` — landed cost as a distribution (P10..P95)
- `text/{extractor,embedder}.py` — lexicon signals + deterministic hashing embedder
- `retrieval/{numpy_index,pgvector_index}.py` — cosine retrieval backends
- `negotiation/{strategies,opponent,solver,engine}.py` — strategy population,
  first-class `OpponentState` latent posterior, `BargainingSolver` over
  alternating offers, decision loop
- `simulation/{monte_carlo,opponent,benchmark}.py` — vectorized MC with
  FAST/STANDARD/DEEP modes, synthetic opponent generator, benchmark harness
  playing Chotu vs baselines (fixed price, linear concession, tit-for-tat,
  random, Nash heuristic)
- `optimization/{objectives,information}.py` — CVaR, fractional Kelly, sim-
  estimated VOI
- `evolutionary/replicator.py` — replicator dynamics over strategy population
- `relationships/engine.py` — LTV-aware personalization
- `policy/gates.py` — money-action gates → ALLOWED/REQUIRES_APPROVAL/BLOCKED
- `learning/loop.py` — offline learning/replay/promotion
- `persistence/{schema.sql,repository.py}` — Supabase schema + REST repository
- `connectors/{providers,razorpay,shopify}.py` — provider interfaces + adapters
- `api/main.py` — FastAPI endpoints incl. cron

## API

```
POST /v1/negotiations
POST /v1/negotiations/{id}/events
POST /v1/negotiations/{id}/decide
GET  /v1/negotiations/{id}
GET  /v1/suppliers/{id}/profile
GET  /v1/suppliers/{id}/metrics
POST /v1/datasets
POST /v1/datasets/{id}/train
POST /v1/webhooks/{provider}
POST /v1/internal/cron/evolve    # Vercel Cron, X-Cron-Secret
```

## Run locally

```
pip install -e ".[dev]"
python -m pytest
uvicorn api.index:app --reload
```

## Benchmark

`simulation/benchmark.py` plays Chotu against synthetic opponents generated
with independent cost/reservation/patience/urgency parameters, and compares
it against baselines (fixed price, linear concession, tit-for-tat, random,
Nash heuristic):

```python
from economic_engine.simulation.benchmark import benchmark, ChotuPolicy
benchmark(n=100, policies={"chotu": ChotuPolicy()})
```

This is the empirical backbone of the engine: if Chotu is meant to beat
humans/simple strategies, this is where that claim is tested, measured in
surplus, win-rate, avg rounds (deal time) and walkaway rate.

## Deploy to Vercel

```
vercel
vercel --prod
```

`vercel.json` sets function memory to 256MB, a 30s max duration, and a nightly
cron hitting `/v1/internal/cron/evolve` for offline strategy evolution.
