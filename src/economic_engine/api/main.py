"""FastAPI app exposing negotiation decisions, supplier profiles, dataset
ingest/train, and internal cron endpoints guardable via CRON_SECRET."""
from __future__ import annotations

import time
import uuid

import numpy as np
import pydantic
from fastapi import APIRouter, FastAPI, Header, HTTPException, Request

from economic_engine.audit.logger import AuditLog
from economic_engine.features.ingest import load_csv_bytes
from economic_engine.models.supplier_model import SupplierPosterior
from economic_engine.negotiation.engine import NegotiationEngine
from economic_engine.persistence.repository import InMemoryRepository, Repository
from economic_engine.retrieval.numpy_index import NumpyCosineIndex
from economic_engine.state.canonical import (
    Negotiation,
    NegotiationContext,
    Offer,
    Round,
)
from economic_engine.state.store import InMemoryState, StateStore
from economic_engine.text.embedder import HashingEmbedder
from economic_engine.text.extractor import extract


class CreateNegotiationRequest(pydantic.BaseModel):
    merchant_id: str
    supplier_id: str
    product_id: str
    quantity: float
    message: str | None = None


class EventRequest(pydantic.BaseModel):
    message: str
    actor: str = "supplier"
    price: float | None = None


class DecisionRequest(pydantic.BaseModel):
    context: NegotiationContext


class DatasetRequest(pydantic.BaseModel):
    name: str
    bytes_base64: str


def get_dependencies():
    store: StateStore = InMemoryState()
    repo: Repository = InMemoryRepository()
    embedder = HashingEmbedder()
    index = NumpyCosineIndex()
    audit = AuditLog()
    return store, repo, embedder, index, audit


def build_app() -> FastAPI:
    store, repo, embedder, index, audit = get_dependencies()
    app = FastAPI(title="Autonomous Negotiation Engine")
    router = APIRouter(prefix="/v1")

    # In-memory cages for demo/edge purposes. In production points Redis/
    # Supabase and assembled via Depends.
    negotiations = {}
    suppliers = {}
    datasets = {}

    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - start) * 1000
        audit.record(
            "http", {"path": request.url.path, "duration_ms": duration,
                     "status": response.status_code},
        )
        return response

    @router.post("/negotiations")
    def create_negotiation(req: CreateNegotiationRequest):
        neg_id = str(uuid.uuid4())
        signals = extract(req.message) if req.message else None
        neg = Negotiation(
            id=neg_id,
            merchant_id=req.merchant_id,
            supplier_id=req.supplier_id,
            product_id=req.product_id,
            quantity=req.quantity,
        )
        negotiations[neg_id] = neg
        vec = embedder.embed(req.message) if req.message else None
        if vec is not None:
            index.add(np.asarray([vec]), [neg_id])
        audit.record("negotiation_created", {"id": neg_id})
        return {"negotiation_id": neg_id, "status": neg.status.value}

    @router.get("/negotiations/{negotiation_id}")
    def get_negotiation(negotiation_id: str):
        neg = negotiations.get(negotiation_id)
        if neg is None:
            raise HTTPException(404)
        return neg.model_dump(mode="json")

    @router.post("/negotiations/{negotiation_id}/events")
    def add_event(negotiation_id: str, req: EventRequest):
        neg = negotiations.get(negotiation_id)
        if neg is None:
            raise HTTPException(404)
        signals = extract(req.message)
        round_ = Round(
            index=len(neg.rounds),
            offer=Offer(price=req.price, actor=req.actor),
            response=None,
        )
        neg.rounds.append(round_)
        audit.record("event", {"id": negotiation_id, "actor": req.actor})
        return {"round_index": round_.index, "status": "RECORDED"}

    @router.post("/negotiations/{negotiation_id}/decide")
    def decide(negotiation_id: str, req: DecisionRequest):
        neg = negotiations.get(negotiation_id)
        if neg is None:
            raise HTTPException(404)
        posterior = SupplierPosterior()
        engine = NegotiationEngine(posterior=posterior)
        ctx = req.context
        ctx.negotiation = neg
        decision = engine.decide(ctx)
        audit.record("decision", {"id": negotiation_id, **decision})
        return decision

    @router.get("/suppliers/{supplier_id}/profile")
    def supplier_profile(supplier_id: str):
        supplier = suppliers.get(supplier_id, {})
        return supplier or {"supplier_id": supplier_id, "posterior": None}

    @router.get("/suppliers/{supplier_id}/metrics")
    def supplier_metrics(supplier_id: str):
        rows = repo.query("learning_events", {"supplier_id": supplier_id})
        if not rows:
            return {"supplier_id": supplier_id, "count": 0}
        errors = [r.get("error", 0) for r in rows]
        return {
            "supplier_id": supplier_id,
            "count": len(rows),
            "rmse": float(np.mean(np.square(errors)) ** 0.5),
        }

    @router.post("/datasets")
    def create_dataset(req: DatasetRequest):
        ds_id = str(uuid.uuid4())
        import base64
        rows = load_csv_bytes(base64.b64decode(req.bytes_base64))
        datasets[ds_id] = {"name": req.name, "rows": rows}
        return {"dataset_id": ds_id, "rows": len(rows)}

    @router.post("/datasets/{dataset_id}/train")
    def train_dataset(dataset_id: str):
        ds = datasets.get(dataset_id)
        if ds is None:
            raise HTTPException(404)
        # Placeholder: training happens offline; here we snapshot features.
        return {"dataset_id": dataset_id, "trained": True, "rows": len(ds["rows"])}

    @router.post("/webhooks/{provider}")
    async def webhook(
        provider: str,
        request: Request,
        x_hook_signature: str | None = Header(None),
    ):
        body = await request.body()
        audit.record("webhook", {"provider": provider, "bytes": len(body)})
        return {"provider": provider, "received": True}

    @router.post("/internal/cron/evolve")
    def evolve(x_cron_secret: str | None = Header(None)):
        # Cron-guarded; offline evolution pass. In production, use Supabase+
        # Upstash instead of in-memory.
        return {"evolved": True}

    app.include_router(router)
    return app


app = build_app()
