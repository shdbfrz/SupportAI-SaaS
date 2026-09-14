"""
SupportAI SaaS — multi-tenant intent-classification gateway.

Flow:
  1. POST /signup           -> get an API key
  2. POST /train            -> train your own model (your data, or demo data to try instantly)
  3. POST /classify         -> classify queries using YOUR trained model
  4. GET  /usage            -> see usage + estimated cost / cost saved vs. an all-LLM baseline

Every tenant's model, training data, and usage log are fully isolated
(separate ONNX model + FAISS index per tenant, separate DB rows).
"""
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.db import (
    create_tenant, get_tenant_by_api_key, mark_trained,
    log_usage, get_usage_summary,
)
from app.tenant_model import train_tenant_model, TenantModel

app = FastAPI(title="SupportAI SaaS")

# Simple in-process cache so we don't reload each tenant's ONNX model from
# disk on every request. Keyed by tenant_id.
_model_cache: dict[int, TenantModel] = {}

# Illustrative, not-a-real-invoice pricing used only to compute the
# "estimated cost" / "cost saved" figures shown in /usage — mirrors how a
# typical usage-based SaaS + LLM API pricing model would look.
COST_PER_FAST_CLASSIFICATION = 0.00005   # $ — running your own small model
COST_PER_LLM_CALL = 0.002                # $ — an approximate real LLM API call cost


def _require_tenant(x_api_key: Optional[str]) -> dict:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    tenant = get_tenant_by_api_key(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant


def _get_model(tenant_id: int) -> TenantModel:
    if tenant_id not in _model_cache:
        try:
            _model_cache[tenant_id] = TenantModel(tenant_id)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Model not trained yet — call POST /train first")
    return _model_cache[tenant_id]


class SignupRequest(BaseModel):
    company_name: str


@app.post("/signup")
async def signup(req: SignupRequest):
    tenant = create_tenant(req.company_name)
    return tenant


class TrainRequest(BaseModel):
    texts: Optional[list[str]] = None
    labels: Optional[list[str]] = None
    use_demo_data: bool = False


@app.post("/train")
async def train(req: TrainRequest, x_api_key: Optional[str] = Header(None)):
    tenant = _require_tenant(x_api_key)

    if req.use_demo_data:
        texts, labels = None, None
    else:
        if not req.texts or not req.labels or len(req.texts) != len(req.labels):
            raise HTTPException(
                status_code=422,
                detail="Provide matching 'texts' and 'labels' arrays, or set use_demo_data=true to try instantly"
            )
        texts, labels = req.texts, req.labels

    meta = train_tenant_model(tenant["id"], texts, labels)
    mark_trained(tenant["id"])
    _model_cache.pop(tenant["id"], None)  # force reload on next classify
    return meta


class ClassifyRequest(BaseModel):
    text: str


@app.post("/classify")
async def classify(req: ClassifyRequest, x_api_key: Optional[str] = Header(None)):
    tenant = _require_tenant(x_api_key)
    model = _get_model(tenant["id"])

    start = time.perf_counter()
    result = model.classify(req.text)
    if result["route"] == "llm_fallback":
        # Simulates a real LLM API round-trip. In production this would be
        # an actual call to GPT-4/Claude/an internal model — kept here so
        # the latency figures in /usage reflect a realistic cost, not just
        # this gateway's own (near-zero) routing overhead.
        import asyncio
        await asyncio.sleep(0.35)
    latency_ms = (time.perf_counter() - start) * 1000
    result["latency_ms"] = round(latency_ms, 3)

    log_usage(tenant["id"], result["route"], latency_ms)
    return result


@app.get("/usage")
async def usage(x_api_key: Optional[str] = Header(None)):
    tenant = _require_tenant(x_api_key)
    summary = get_usage_summary(tenant["id"])

    fast_count = summary["fast_classifier"]["count"]
    llm_count = summary["llm_fallback"]["count"]
    total = fast_count + llm_count

    actual_cost = fast_count * COST_PER_FAST_CLASSIFICATION + llm_count * COST_PER_LLM_CALL
    baseline_cost_if_all_llm = total * COST_PER_LLM_CALL
    saved = baseline_cost_if_all_llm - actual_cost
    saved_pct = (saved / baseline_cost_if_all_llm * 100) if baseline_cost_if_all_llm > 0 else 0

    return {
        "company_name": tenant["company_name"],
        "total_classifications": total,
        "bypassed_llm": fast_count,
        "routed_to_llm": llm_count,
        "bypass_rate_pct": round(fast_count / total * 100, 1) if total else 0,
        "estimated_cost_usd": round(actual_cost, 5),
        "cost_if_all_llm_usd": round(baseline_cost_if_all_llm, 5),
        "estimated_savings_usd": round(saved, 5),
        "estimated_savings_pct": round(saved_pct, 1),
        "route_breakdown": summary,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


_DASHBOARD_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return _DASHBOARD_HTML
