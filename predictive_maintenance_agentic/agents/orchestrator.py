"""
Orchestrator nodes.

The Orchestrator Agent in the diagram owns the cycle: starts it,
reads fleet memory, hands off to Anomaly Detection. It also gates the
short-circuit path (rule #1 in the build prompt).
"""
from __future__ import annotations

import uuid
from typing import Any, Dict

from ..memory.fleet_memory import FleetMemoryStore
from ..models import AssetType, HealthStatus
from ._shared import (
    append_trace,
    get_anomaly_engine,
    get_fleet_memory,
    sensor_batch_from_dict,
    to_native,
)


def initialize_cycle_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    First node: assign cycle_id, load prior fleet-memory context for
    this asset (last 10 entries), and initialize per-cycle counters.
    """
    cycle_id = state.get("cycle_id") or f"cycle-{uuid.uuid4().hex[:12]}"
    memory: FleetMemoryStore = get_fleet_memory()
    history = memory.get_history(state["asset_id"], limit=10)

    trace = append_trace(
        state,
        node="orchestrator.init",
        note=f"Loaded {len(history)} prior fleet-memory entries for {state['asset_id']}",
        data={"fleet_memory_history_size": len(history)},
    )
    return {
        "cycle_id": cycle_id,
        "negotiation_round": 0,
        "resimulation_round": 0,
        "status": "running",
        "fleet_memory_refs": [h.entry_id for h in history],
        "trace": trace,
        "errors": [],
        "negotiation_history": [],
    }


def ingest_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Data Ingestion Layer + Anomaly Detection Gate.

    Runs anomaly detection on the incoming sensor batch. If NOT
    anomalous, writes a "cycle_normal" entry to fleet memory and
    marks the cycle for termination (see conditional edge in
    build_graph.py).
    """
    sensor_df = sensor_batch_from_dict(state["sensor_batch"])
    asset_type = AssetType(state["asset_type"])
    material = state.get("material_name")

    engine = get_anomaly_engine()
    result = engine.detect(asset_type, material, sensor_df)

    updates: Dict[str, Any] = {
        "anomaly_result": to_native(result.to_dict()),
        "is_anomalous": bool(result.is_anomalous),
    }

    trace = append_trace(
        state,
        node="ingestion.anomaly_gate",
        note=(
            f"Anomaly gate: {'ANOMALOUS' if result.is_anomalous else 'NORMAL'} "
            f"(count={result.anomaly_count}, severity={result.max_severity})"
        ),
        data=result.to_dict(),
    )
    updates["trace"] = to_native(trace)

    # Rule #1: short-circuit for non-anomalous cycles.
    if not result.is_anomalous:
        memory = get_fleet_memory()
        entry = memory.add_entry(
            asset_id=state["asset_id"],
            kind="cycle_normal",
            cycle_id=state.get("cycle_id"),
            payload={
                "anomaly_result": result.to_dict(),
                "note": "Anomaly gate did not trip; no physics/ML calls made.",
            },
        )
        refs = list(state.get("fleet_memory_refs") or [])
        refs.append(entry.entry_id)
        updates["fleet_memory_refs"] = refs
        updates["status"] = "normal_end"

    return updates
