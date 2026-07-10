"""
Engineer Approval — real LangGraph interrupt.

We DON'T auto-approve here. When the graph reaches this node, the
StateGraph's `interrupt_before` (set in build_graph.py) suspends
execution. The FastAPI `/engineer/approve` endpoint resumes by updating
state with the engineer's decision and calling `graph.invoke(None, ...)`.

If `state["engineer_decision"]` is already populated when this node
runs, we treat that as the resume payload and validate it — this covers
both the API resume path and unit-test injection paths.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from ..config import LIMITS
from ..memory.fleet_memory import FleetMemoryStore
from ._shared import append_trace, get_fleet_memory, to_native


def engineer_resume_payload(
    approved: bool,
    engineer_id: str,
    notes: str = "",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Helper the API layer uses to build the resume payload."""
    return {
        "engineer_decision": {
            "approved": bool(approved),
            "engineer_id": engineer_id,
            "notes": notes,
            "timestamp": time.time(),
            **(extra or {}),
        }
    }


def engineer_approval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    decision = state.get("engineer_decision")

    if decision is None:
        # This is only reached if interrupt_before is not configured
        # (defensive). Populate a "pending" record; the graph will
        # halt at the interrupt anyway.
        pending = {
            "approved": None,
            "engineer_id": None,
            "notes": "AWAITING_ENGINEER_APPROVAL",
            "timestamp": time.time(),
        }
        trace = append_trace(
            state,
            node="engineer_approval",
            note="Awaiting engineer input (interrupt).",
            data=pending,
        )
        return {"engineer_decision": pending, "trace": to_native(trace)}

    # Post-resume path — log to fleet memory.
    memory: FleetMemoryStore = get_fleet_memory()
    entry = memory.add_entry(
        asset_id=state["asset_id"],
        kind="engineer_decision",
        cycle_id=state.get("cycle_id"),
        payload={
            "decision": decision,
            "calibration_result": state.get("calibration_result"),
            "resimulation_round": state.get("resimulation_round", 0),
            "resim_cap": LIMITS.max_resimulation_rounds,
        },
    )
    refs = list(state.get("fleet_memory_refs") or [])
    refs.append(entry.entry_id)

    trace = append_trace(
        state,
        node="engineer_approval",
        note=(
            f"Engineer {'APPROVED' if decision.get('approved') else 'REJECTED'} "
            f"calibration (round {state.get('resimulation_round', 0)}/{LIMITS.max_resimulation_rounds})."
        ),
        data=decision,
    )

    return {
        "engineer_decision": decision,
        "fleet_memory_refs": refs,
        "trace": to_native(trace),
    }
