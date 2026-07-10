"""
Action Agent — the last real "decision" node in the graph.

Reads physics + ML + judge verdict, decides one of:
  * continue_monitoring
  * schedule_inspection
  * recommend_repair
  * generate_work_order

Also surfaces material-change recommendations from EnhancedMaterialDatabase
when RUL is low (rule #6 in the build prompt), and generates a stubbed
MRO/ERP work-order payload via the SAP-PM adapter.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from ..config import LIMITS, THRESHOLDS
from ..integrations import SAPPMAdapter
from ._shared import append_trace, get_fleet_memory, get_material_engine, to_native


def _decide_action(
    physics: Dict[str, Any],
    ml: Dict[str, Any],
    verdict: Dict[str, Any],
    forced_unresolved: bool,
) -> Dict[str, Any]:
    if forced_unresolved:
        return {
            "type": "generate_work_order",
            "reason": (
                "Non-convergence after 5 re-simulation rounds — routing to "
                "engineering with unresolved-divergence flag."
            ),
            "unresolved_divergence": True,
        }

    health_status = physics.get("health_status") or "normal"
    rul_hours = physics.get("rul_hours")
    ml_rul = ml.get("rul_hours") or 0.0

    # Choose the more conservative RUL for decisioning.
    effective_rul = min([r for r in (rul_hours, ml_rul) if r], default=None)

    if health_status in ("critical", "failed") or (effective_rul is not None and effective_rul < 168):
        return {
            "type": "generate_work_order",
            "reason": f"Health={health_status}, RUL≈{effective_rul} h — urgent maintenance.",
        }
    if health_status == "warning" or (effective_rul is not None and effective_rul < THRESHOLDS.low_rul_hours):
        return {
            "type": "recommend_repair",
            "reason": f"Health={health_status}, RUL≈{effective_rul} h — schedule repair within 2-4 weeks.",
        }
    if verdict.get("divergence"):
        return {
            "type": "schedule_inspection",
            "reason": "Physics/ML divergence resolved by calibration; schedule NDT inspection.",
        }
    return {
        "type": "continue_monitoring",
        "reason": "Anomaly acknowledged but predictions agree; continue monitoring.",
    }


def action_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    physics = state.get("physics_prediction") or {}
    ml = state.get("ml_correction") or {}
    verdict = state.get("judge_verdict") or {}
    resim_round = int(state.get("resimulation_round", 0))
    engineer_dec = state.get("engineer_decision") or {}

    # Rule #3: hitting the 5-round cap forces the "unresolved_divergence"
    # flag AND requires an explicit engineer sign-off note.
    forced_unresolved = (
        resim_round >= LIMITS.max_resimulation_rounds
        and (verdict.get("divergence") or not engineer_dec.get("approved"))
    )

    action = _decide_action(physics, ml, verdict, forced_unresolved)

    # Rule #6: material recommendations only when RUL is low.
    material_recs: Optional[Dict[str, Any]] = None
    rul_hours = physics.get("rul_hours")
    material_name = state.get("material_name")
    if (rul_hours is not None and rul_hours < THRESHOLDS.low_rul_hours) or forced_unresolved:
        if material_name:
            material_recs = get_material_engine().recommend(material_name)
            action["material_recommendations"] = material_recs

    # MRO work-order stub, only for the two action types that need one.
    work_order: Optional[Dict[str, Any]] = None
    if action["type"] in ("generate_work_order", "recommend_repair"):
        adapter = SAPPMAdapter()
        wo_payload = {
            "asset_id": state["asset_id"],
            "asset_type": state["asset_type"],
            "component": state.get("component"),
            "action_type": action["type"],
            "reason": action["reason"],
            "physics_summary": {
                "rul_hours": physics.get("rul_hours"),
                "health_status": physics.get("health_status"),
                "failure_mode": physics.get("failure_mode"),
            },
            "ml_summary": {
                "rul_hours": ml.get("rul_hours"),
                "predicted_stress_mpa": ml.get("predicted_stress_mpa"),
            },
            "judge_verdict": verdict,
            "engineer_decision": engineer_dec or None,
            "material_recommendations": material_recs,
            "cycle_id": state.get("cycle_id"),
            "created_at": time.time(),
            "id_hint": f"WO-{uuid.uuid4().hex[:8]}",
            "unresolved_divergence": bool(forced_unresolved),
        }
        if forced_unresolved and not engineer_dec.get("notes"):
            wo_payload["engineer_signoff_required"] = True
        work_order = adapter.create_work_order(wo_payload)

    # Write to fleet memory (rule #5: write at end of every cycle).
    memory = get_fleet_memory()
    entry = memory.add_entry(
        asset_id=state["asset_id"],
        kind="cycle_action",
        cycle_id=state.get("cycle_id"),
        payload={
            "action": action,
            "work_order": work_order,
            "physics": {"rul_hours": physics.get("rul_hours"), "health_status": physics.get("health_status")},
            "ml": {"rul_hours": ml.get("rul_hours")},
            "verdict": verdict,
            "resim_round": resim_round,
        },
    )
    refs = list(state.get("fleet_memory_refs") or [])
    refs.append(entry.entry_id)

    status = "unresolved_divergence" if forced_unresolved else "action_taken"

    trace = append_trace(
        state,
        node="action_agent",
        note=f"Action: {action['type']} — {action['reason']}",
        data={"action": action, "work_order_id": (work_order or {}).get("work_order_id")},
    )

    return {
        "action": to_native(action),
        "work_order": to_native(work_order),
        "fleet_memory_refs": refs,
        "status": status,
        "trace": to_native(trace),
    }
