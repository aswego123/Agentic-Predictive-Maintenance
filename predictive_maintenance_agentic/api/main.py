"""
FastAPI surface for the agentic predictive-maintenance system.

Endpoints (matches section 2 of the build prompt):
  POST   /analyze                 — kick off a cycle for one sensor batch
  GET    /fleet/status            — fleet memory summary
  GET    /cycles/{cycle_id}       — trace + last state for a cycle
  POST   /engineer/approve        — resume a graph that hit the interrupt

Every response labels the simulation source (`"synthetic"` today) so
callers can never confuse a synthetic result for a real solver result.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..agents import engineer_resume_payload
from ..agents._shared import (
    get_fleet_memory,
    get_simulation_layer,
    sensor_batch_to_dict,
)
from ..data.synthetic_generator import (
    generate_normal_batch,
    generate_sensor_data,
    inject_anomaly,
)
from ..graph import GraphBundle, build_graph
from ..models import AssetType


app = FastAPI(
    title="Predictive Maintenance — Agentic Digital Twin",
    version="0.1.0",
    description=(
        "LangGraph-based multi-agent flow wrapping the existing physics + "
        "ML predictive-maintenance modules. The simulation layer is "
        "SYNTHETIC — no real ANSYS/Abaqus/Creo/NASTRAN calls are made."
    ),
)


# ---- Global graph bundle (built once at startup) ----
_BUNDLE: Optional[GraphBundle] = None
# In-memory index of {cycle_id -> thread_id} so we can look up runs.
# Note: this dict is transient — it gets rebuilt from the persisted
# fleet_memory SQLite on the first call to `_get_bundle()` after a
# restart so users don't "lose" their prior cycles across restarts.
_CYCLE_INDEX: Dict[str, Dict[str, Any]] = {}
_INDEX_REHYDRATED: bool = False


def _rehydrate_cycle_index() -> None:
    """
    Repopulate `_CYCLE_INDEX` from the persisted fleet_memory SQLite so
    that cycles from previous process runs still show up in `/fleet/status`
    (and therefore the Cycles list view in the UI).

    Prior to this fix `_CYCLE_INDEX` was reset to `{}` on every uvicorn
    restart, which made the Cycles page appear empty even though the
    LangGraph checkpoints + audit trail were still on disk.
    """
    global _INDEX_REHYDRATED
    if _INDEX_REHYDRATED:
        return
    _INDEX_REHYDRATED = True  # set first — even a failed rehydrate shouldn't loop

    try:
        memory = get_fleet_memory()
        with memory._connect() as conn:  # type: ignore[attr-defined]
            rows = conn.execute(
                """
                SELECT cycle_id, asset_id, MIN(created_at) AS created_at
                FROM fleet_memory
                WHERE cycle_id IS NOT NULL AND cycle_id != ''
                GROUP BY cycle_id
                """
            ).fetchall()
        for row in rows:
            cid = row["cycle_id"]
            if cid and cid not in _CYCLE_INDEX:
                _CYCLE_INDEX[cid] = {
                    "asset_id": row["asset_id"],
                    "created_at": row["created_at"],
                }
    except Exception:
        # Rehydration is best-effort — if the schema is missing or the
        # DB is locked we just start with an empty index rather than
        # blocking the server from booting.
        pass


def _get_bundle() -> GraphBundle:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = build_graph()
    _rehydrate_cycle_index()
    return _BUNDLE


# ============================================================
# Request / response models
# ============================================================
class SensorBatch(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="Row-oriented sensor data")
    columns: Optional[List[str]] = None


class AnalyzeRequest(BaseModel):
    asset_id: str
    asset_type: str = Field(..., description="AssetType enum value, e.g. 'aircraft_engine'")
    material_name: Optional[str] = None  # inferred from asset_type or CSV if omitted
    component: Optional[str] = None
    cycle_id: Optional[str] = None  # client may supply so the UI can redirect immediately
    # Either provide a sensor_batch OR request a synthetic one.
    sensor_batch: Optional[SensorBatch] = None
    generate_synthetic: bool = False
    synthetic_n_samples: int = 100
    force_anomaly: bool = False
    force_normal: bool = False


class EngineerApproveRequest(BaseModel):
    cycle_id: str
    approved: bool
    engineer_id: str
    notes: str = ""
    extra: Optional[Dict[str, Any]] = None


# ============================================================
# Helpers
# ============================================================
def _thread_config(cycle_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": cycle_id}}


def _run_from(state_updates: Optional[Dict[str, Any]], cycle_id: str, resume: bool = False) -> Dict[str, Any]:
    """
    Run (or resume) the graph, then return the current state snapshot.

    For resume, LangGraph expects `graph.invoke(None, config)` after the
    caller has patched state via `graph.update_state`.
    """
    bundle = _get_bundle()
    graph = bundle.graph
    cfg = _thread_config(cycle_id)

    if resume:
        result_state = graph.invoke(None, cfg)
    else:
        result_state = graph.invoke(state_updates, cfg)

    return _snapshot(cycle_id)


def _snapshot(cycle_id: str) -> Dict[str, Any]:
    bundle = _get_bundle()
    snap = bundle.graph.get_state(_thread_config(cycle_id))
    values = snap.values if snap else {}
    next_nodes = list(snap.next) if snap and snap.next else []
    return {
        "cycle_id": cycle_id,
        "status": values.get("status", "running") if values else "unknown",
        "is_anomalous": bool(values.get("is_anomalous", False)) if values else False,
        "asset_id": values.get("asset_id") if values else None,
        "asset_type": values.get("asset_type") if values else None,
        "component": values.get("component") if values else None,
        "anomaly_result": values.get("anomaly_result") if values else None,
        "stress_features": values.get("stress_features") if values else None,
        "physics_prediction": _strip_recs(values.get("physics_prediction")) if values else None,
        "ml_correction": values.get("ml_correction") if values else None,
        "judge_verdict": values.get("judge_verdict") if values else None,
        "calibration_result": values.get("calibration_result") if values else None,
        "engineer_decision": values.get("engineer_decision") if values else None,
        "negotiation_round": values.get("negotiation_round", 0) if values else 0,
        "negotiation_history": values.get("negotiation_history") if values else [],
        "dialogue_history": values.get("dialogue_history") if values else [],
        "fetched_features": values.get("fetched_features") if values else {},
        "resimulation_round": values.get("resimulation_round", 0) if values else 0,
        "action": values.get("action") if values else None,
        "critic_review": values.get("critic_review") if values else None,
        "work_order": values.get("work_order") if values else None,
        "trace": values.get("trace") if values else [],
        "fleet_memory_refs": values.get("fleet_memory_refs") if values else [],
        "next": next_nodes,
        "awaiting_engineer_approval": bool(next_nodes and "engineer_approval" in next_nodes),
        "simulation_adapter": get_simulation_layer().name,
        "simulation_is_synthetic": True,
    }


def _strip_recs(pred: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not pred:
        return pred
    out = dict(pred)
    # keep recommendations but drop the huge legacy dataclass if present
    out.pop("_raw", None)
    return out


# ---------------------------------------------------------------
# Material resolution
#
# The CLI script (anamoly-detection.py) hard-codes a canonical
# material per asset family (aircraft_wing -> Al7075-T6, etc). We
# mirror those defaults here so cycles started without an explicit
# `material_name` still get realistic material physics + supplier
# recommendations instead of silently falling back to Steel4340.
# ---------------------------------------------------------------
DEFAULT_MATERIAL_FOR_ASSET_TYPE: Dict[AssetType, str] = {
    AssetType.AIRCRAFT_ENGINE:         "Inconel718",
    AssetType.AIRCRAFT_LANDING_GEAR:   "Steel4340",
    AssetType.AIRCRAFT_BRAKE:          "Steel4340",
    AssetType.AIRCRAFT_WING:           "Al7075-T6",
    AssetType.AIRCRAFT_FUSELAGE:       "Al2024-T3",
    AssetType.TRAIN_BOGIE:             "CastIron",
    AssetType.TRAIN_BRAKE:             "Steel4340",
    AssetType.TRAIN_WHEEL:             "Steel4340",
    AssetType.TRAIN_TRACTION_MOTOR:    "Steel4340",
}


def _material_from_batch(df: pd.DataFrame) -> Optional[str]:
    """Pick the first non-empty `material_name` / `material` value in the batch."""
    for col in ("material_name", "material"):
        if col in df.columns:
            for value in df[col].dropna():
                text = str(value).strip()
                if text:
                    return text
    return None


def _resolve_material(
    explicit: Optional[str],
    df: pd.DataFrame,
    asset_type: AssetType,
) -> str:
    if explicit:
        return explicit
    from_batch = _material_from_batch(df)
    if from_batch:
        return from_batch
    return DEFAULT_MATERIAL_FOR_ASSET_TYPE.get(asset_type, "Steel4340")


# ============================================================
# Endpoints
# ============================================================
@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <!doctype html>
    <html><head><title>Predictive Maintenance — Agentic Digital Twin</title>
    <style>
      body{font-family:system-ui,sans-serif;max-width:820px;margin:40px auto;padding:0 20px;line-height:1.5;color:#222}
      code{background:#f4f4f4;padding:2px 6px;border-radius:4px}
      a{color:#0b64d0}
      .note{background:#fff8e1;border-left:4px solid #f5c000;padding:10px 14px;border-radius:4px}
    </style></head>
    <body>
      <h1>Predictive Maintenance — Agentic Digital Twin</h1>
      <p>LangGraph + LangChain multi-agent flow. Simulation layer is
      <b>synthetic</b> — no ANSYS/Abaqus/Creo/NASTRAN calls.</p>
      <ul>
        <li><a href="/docs">Interactive API docs (Swagger UI)</a></li>
        <li><a href="/redoc">ReDoc</a></li>
        <li><a href="/fleet/status">GET /fleet/status</a></li>
      </ul>
      <h3>Endpoints</h3>
      <ul>
        <li><code>POST /analyze</code> — start a cycle for a sensor batch</li>
        <li><code>GET  /fleet/status</code> — fleet memory summary</li>
        <li><code>GET  /cycles/{cycle_id}</code> — trace + last state</li>
        <li><code>POST /engineer/approve</code> — resume a paused graph</li>
      </ul>
      <p class="note">Prefer a UI? Run
      <code>streamlit run predictive_maintenance_agentic/ui/dashboard.py</code>
      in another terminal.</p>
    </body></html>
    """


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "simulation_adapter": get_simulation_layer().name}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    try:
        asset_type = AssetType(req.asset_type)
    except ValueError:
        raise HTTPException(400, f"Unknown asset_type {req.asset_type!r}")

    # ------- Build the sensor batch -------
    if req.sensor_batch is not None:
        df = pd.DataFrame(req.sensor_batch.records)
    elif req.generate_synthetic:
        if req.force_normal:
            df = generate_normal_batch(req.asset_id, asset_type, req.synthetic_n_samples)
        else:
            df = generate_sensor_data(req.asset_id, asset_type, req.synthetic_n_samples)
        if req.force_anomaly:
            df = inject_anomaly(df, channel="vibration", multiplier=6.0)
    else:
        raise HTTPException(400, "Provide sensor_batch or set generate_synthetic=true")

    if "asset_id" not in df.columns:
        df["asset_id"] = req.asset_id

    # Resolve the material for this run in this order of preference:
    #   1. explicit `material_name` in the request
    #   2. a `material_name` / `material` column in the sensor CSV (first non-empty row)
    #   3. the per-asset-type default from DEFAULT_MATERIAL_FOR_ASSET_TYPE
    resolved_material = _resolve_material(req.material_name, df, asset_type)

    cycle_id = req.cycle_id or f"cycle-{uuid.uuid4().hex[:12]}"
    initial: Dict[str, Any] = {
        "cycle_id": cycle_id,
        "asset_id": req.asset_id,
        "asset_type": asset_type.value,
        "component": req.component or asset_type.value,
        "material_name": resolved_material,
        "sensor_batch": sensor_batch_to_dict(df),
    }

    _CYCLE_INDEX[cycle_id] = {
        "created_at": time.time(),
        "asset_id": req.asset_id,
    }

    snapshot = _run_from(initial, cycle_id, resume=False)
    return snapshot


@app.get("/fleet/status")
def fleet_status() -> Dict[str, Any]:
    # Ensure prior-run cycles are in the index. This handler doesn't call
    # `_get_bundle()` (which would build the whole graph unnecessarily),
    # so we must trigger rehydration directly — otherwise the /cycles
    # list view is empty after every backend restart.
    _rehydrate_cycle_index()
    mem = get_fleet_memory()
    # Snapshot the known cycles (this process) and asset summaries.
    asset_ids = {info["asset_id"] for info in _CYCLE_INDEX.values() if info.get("asset_id")}
    return {
        "known_assets": sorted(asset_ids),
        "asset_summaries": [mem.summarize_asset(a) for a in sorted(asset_ids)],
        "known_cycles": [
            {"cycle_id": cid, **info} for cid, info in _CYCLE_INDEX.items()
        ],
        "simulation_adapter": get_simulation_layer().name,
        "simulation_is_synthetic": True,
    }


@app.get("/cycles/{cycle_id}")
def get_cycle(cycle_id: str) -> Dict[str, Any]:
    _rehydrate_cycle_index()
    if cycle_id not in _CYCLE_INDEX:
        # Still try — the checkpointer may know about it across restarts.
        try:
            snap = _snapshot(cycle_id)
            # If _snapshot returned an "unknown" placeholder for a truly
            # unknown cycle, treat it as a pending cycle the client just
            # created (the POST /analyze is likely in flight). Returning
            # a soft pending payload keeps the detail page's poll alive
            # instead of tripping the react-query error path on 404.
            if snap.get("status") in {None, "unknown"}:
                return {**snap, "status": "pending"}
            return snap
        except Exception:
            # Same rationale — return a pending stub so the UI can keep
            # polling until the /analyze POST reaches the server and
            # actually registers the cycle in _CYCLE_INDEX.
            return {
                "cycle_id": cycle_id,
                "status": "pending",
                "is_anomalous": False,
                "trace": [],
                "next": [],
                "awaiting_engineer_approval": False,
            }
    return _snapshot(cycle_id)


@app.post("/engineer/approve")
def engineer_approve(req: EngineerApproveRequest) -> Dict[str, Any]:
    bundle = _get_bundle()
    cfg = _thread_config(req.cycle_id)
    snap = bundle.graph.get_state(cfg)
    if snap is None:
        raise HTTPException(404, f"No such cycle {req.cycle_id!r}")

    next_nodes = list(snap.next or [])
    if "engineer_approval" not in next_nodes:
        raise HTTPException(
            409,
            f"Cycle {req.cycle_id!r} is not waiting on engineer approval "
            f"(next nodes: {next_nodes}).",
        )

    resume_state = engineer_resume_payload(
        approved=req.approved,
        engineer_id=req.engineer_id,
        notes=req.notes,
        extra=req.extra,
    )
    # Patch engineer_decision INTO the pending state, then resume.
    bundle.graph.update_state(cfg, resume_state)
    return _run_from(None, req.cycle_id, resume=True)
