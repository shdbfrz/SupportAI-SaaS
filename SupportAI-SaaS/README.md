# SupportAI SaaS — Multi-Tenant Intent Classification Gateway

A self-serve SaaS version of the SupportAI routing gateway: any company can
sign up, train a classifier on their own support queries (or try it
instantly on bundled demo data), and get an API that bypasses expensive
LLM calls for queries it's confident about — with per-tenant usage
tracking and cost-savings estimates, like a real usage-based API product.

Built with FastAPI, scikit-learn, ONNX Runtime (INT8 quantized), FAISS,
SQLite, and a live signup/train/classify/billing dashboard.

## What makes this a SaaS (not just an API)

| Capability | How it's implemented |
|---|---|
| Self-serve signup | `POST /signup` → instant API key, no manual provisioning |
| Bring-your-own-data | `POST /train` with your own labeled queries, or `use_demo_data: true` to try it in seconds |
| Full tenant isolation | Each tenant gets its own ONNX model, FAISS index, and training data on disk — verified by tests that two tenants' models and usage never cross-contaminate |
| Usage-based billing model | `GET /usage` — running totals + estimated cost vs. an all-LLM baseline, the same shape as a real API billing page |
| Self-service dashboard | Full flow (signup → train → live classify → usage) in one browser page, no separate admin tool needed |

## Try it in under a minute

```bash
pip install -r requirements.txt
PYTHONPATH=. uvicorn app.main:app --reload
```
Open `http://localhost:8000` — sign up, click "try with demo data", then
classify a few queries and watch the usage/cost numbers update live.

Or via API directly:
```bash
# 1. Sign up
curl -X POST http://localhost:8000/signup -H "Content-Type: application/json" \
  -d '{"company_name": "Acme Corp"}'
# -> {"tenant_id": 1, "company_name": "Acme Corp", "api_key": "sk_..."}

# 2. Train (demo data, or your own texts/labels)
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" \
  -H "X-API-Key: sk_..." -d '{"use_demo_data": true}'

# 3. Classify
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" \
  -H "X-API-Key: sk_..." -d '{"text": "why was I charged twice"}'

# 4. Check usage/cost
curl http://localhost:8000/usage -H "X-API-Key: sk_..."
```

## Measured results (from the underlying single-tenant model — see the
non-SaaS `SupportAI` project for the full benchmark)

| Metric | Result |
|---|---|
| LLM bypass rate | ~90% on the bundled demo dataset |
| ONNX INT8 size reduction | ~60-70% (varies by tenant's dataset size) |
| Fast-path latency | <2ms | LLM-path latency (simulated) | ~350ms |

This SaaS layer adds multi-tenancy, isolation, and billing on top of that
same core classification/routing engine — it doesn't change the
underlying model's accuracy characteristics, which depend on each
tenant's own training data.

> **Small-dataset honesty note:** if a tenant trains on a tiny custom
> dataset (say, 12 examples), there isn't enough data for a genuine
> held-out test split. `train_tenant_model` detects this and reports
> accuracy measured on the training set itself — clearly labeled as such
> in the response (`accuracy_measured_on`) — rather than silently passing
> off a memorization score as generalization.

## How the isolation works

Every tenant gets:
- A row in `tenants` (SQLite) with a unique `api_key`
- A private directory `tenant_data/tenant_{id}/` containing their own
  `model_int8.onnx`, `pipeline.joblib` (for LIME if extended), FAISS
  index, and training data
- Usage rows in `usage_log` scoped to their `tenant_id`

`tests/test_saas.py` includes a dedicated isolation test: two tenants
train different models, classify different queries, and the test asserts
each tenant's `/usage` only reflects their own activity.

## Billing model (illustrative)

`/usage` computes:
- `estimated_cost_usd` = (fast-path classifications × $0.00005) + (LLM-routed × $0.002)
- `cost_if_all_llm_usd` = what it would have cost if every query went to an LLM
- `estimated_savings_usd` / `estimated_savings_pct` = the difference

These per-call prices are illustrative (a rough real-world LLM API cost
vs. the near-zero cost of running your own small model) — the point is
the *shape* of a usage-based SaaS pricing page, not a real invoice. Wiring
in actual provider pricing and a payment processor (Stripe, etc.) would be
the next step for a production version.

## Running it

**Locally:**
```bash
pip install -r requirements.txt
PYTHONPATH=. uvicorn app.main:app --reload
```

**With Docker:**
```bash
docker build -t supportai-saas .
docker run -p 8000:8000 -v $(pwd)/tenant_data:/srv/tenant_data supportai-saas
```
(The volume mount keeps tenant data across container restarts — without
it, all signups/models are lost when the container stops.)

> **Linux note:** same as the single-tenant SupportAI project — ONNX
> Runtime's text-preprocessing needs the `en_US.UTF-8` locale on Linux.
> The Dockerfile handles this; if running `train_tenant_model` directly on
> a bare Linux machine, `export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8` first.

## API

| Endpoint | Auth | Description |
|---|---|---|
| `GET /` | — | Self-serve dashboard (signup → train → classify → usage) |
| `POST /signup` | — | `{"company_name": "..."}` → tenant ID + API key |
| `POST /train` | `X-API-Key` | `{"use_demo_data": true}` or `{"texts": [...], "labels": [...]}` |
| `POST /classify` | `X-API-Key` | `{"text": "..."}` → intent, route, confidence |
| `GET /usage` | `X-API-Key` | Usage totals + estimated cost/savings |
| `GET /health` | — | Liveness check |
| `GET /docs` | — | Swagger UI |

## Testing

```bash
PYTHONPATH=. pytest tests/ -v
```
9 tests covering: signup, auth rejection (missing/invalid key),
train-before-classify guard, demo-data training, custom-data training,
malformed-input validation, **two-tenant isolation**, and cost calculation.

## Project structure

```
SupportAI-SaaS/
├── app/
│   ├── db.py              # tenant + usage SQLite store
│   ├── dataset.py         # bundled demo dataset generator
│   ├── tenant_model.py    # per-tenant train/export/quantize + inference
│   ├── main.py            # FastAPI app: signup/train/classify/usage
│   └── static/index.html  # self-serve dashboard
├── tests/test_saas.py
├── tenant_data/            # created at runtime — one folder per tenant
├── Dockerfile
└── requirements.txt
```

## What a real production SaaS would add

- Real payment processing (Stripe) instead of illustrative pricing
- Rate limiting per API key / plan tier
- Background training jobs (for large datasets) instead of synchronous
  training in the request
- Real LLM fallback (an actual API call) instead of the simulated delay
- Auth beyond a bearer API key (OAuth for the dashboard, key rotation)
- Postgres instead of SQLite for concurrent multi-instance deployments
