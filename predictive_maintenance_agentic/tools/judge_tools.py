"""
Tools bound to the Judge Agent via LangChain `@tool`.

The Judge is invoked as a ReAct-style loop: the LLM sees the tool
signatures + the sensor CSV, calls up to N tools, then emits a final
structured verdict. All tools are pure functions of state + arguments;
they never mutate state directly. The graph node captures each tool
call into `state["trace"]` so the UI can render the ReAct chain.

Tools are constructed per-cycle via `build_judge_tools(state)` so each
tool has closure access to the current asset_id, stress features, and
physics/ML predictions without leaking a global.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .._legacy_imports import anomaly_mod as _canon
from ..agents._shared import get_fleet_memory, get_physics_engine


# ---------------------------------------------------------------
# Argument schemas (pydantic — required by LangChain tools)
# ---------------------------------------------------------------
class _AssetHistoryArgs(BaseModel):
    limit: int = Field(10, ge=1, le=50, description="How many past entries to summarize.")


class _SimilarFailuresArgs(BaseModel):
    stress_tolerance_mpa: float = Field(
        50.0, ge=1.0, le=500.0,
        description="Match past cycles whose stress is within ±tolerance of current.",
    )
    limit: int = Field(20, ge=1, le=100)


class _MaterialLimitsArgs(BaseModel):
    material_name: str = Field(
        ...,
        description="Material identifier (e.g. 'Steel4340', 'Ti-6Al-4V', 'Al7075-T6').",
    )


class _WhatIfArgs(BaseModel):
    stress_delta_mpa: float = Field(
        0.0, ge=-500.0, le=500.0,
        description="Signed delta added to the current stress amplitude.",
    )
    crack_size_delta_mm: float = Field(
        0.0, ge=-5.0, le=10.0,
        description="Signed delta added to the current crack size.",
    )


class _RecentMaintArgs(BaseModel):
    days: int = Field(30, ge=1, le=365, description="Look-back window in days.")


# ---------------------------------------------------------------
# Tool implementations (closures over cycle state)
# ---------------------------------------------------------------
def _summarize_entry(entry_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compact projection of a fleet-memory entry — small enough for
    an LLM tool observation."""
    physics = entry_payload.get("physics_prediction") or entry_payload.get("physics") or {}
    ml = entry_payload.get("ml_correction") or entry_payload.get("ml") or {}
    verdict = entry_payload.get("judge_verdict") or entry_payload.get("verdict") or {}
    return {
        "cycle_id": entry_payload.get("cycle_id"),
        "created_at": entry_payload.get("created_at") or entry_payload.get("timestamp"),
        "physics_stress_mpa": physics.get("stress_amplitude_mpa"),
        "physics_rul_hours": physics.get("rul_hours"),
        "ml_stress_mpa": ml.get("predicted_stress_mpa"),
        "ml_rul_hours": ml.get("rul_hours"),
        "route": verdict.get("route"),
        "maintenance_required": verdict.get("maintenance_required"),
    }


def build_judge_tools(state: Dict[str, Any]) -> List[StructuredTool]:
    """Return a list of LangChain StructuredTool objects for this cycle."""
    asset_id: str = state.get("asset_id", "")
    stress_features: Dict[str, Any] = state.get("stress_features") or {}
    physics_pred: Dict[str, Any] = state.get("physics_prediction") or {}
    # Carry the deterministic operational-cycle reading through so
    # simulate_what_if returns the same health_score family the main
    # physics_agent produced (instead of the random legacy placeholder).
    operational_cycles_actual: Any = physics_pred.get("operational_cycles_actual")
    material_db = _canon.EnhancedMaterialDatabase()

    # ---- tool 1 -------------------------------------------------
    def _get_fleet_memory_stats(limit: int = 10) -> Dict[str, Any]:
        try:
            memory = get_fleet_memory()
            summary = memory.summarize_asset(asset_id)
            history = memory.get_history(asset_id, limit=limit)
            recent = [_summarize_entry(e.payload) for e in history]
            return {
                "asset_id": asset_id,
                "counts_by_kind": summary.get("counts", {}),
                "last_entry_at": summary.get("last_entry_at"),
                "recent": recent[:limit],
            }
        except Exception as exc:
            return {"error": str(exc)}

    # ---- tool 2 -------------------------------------------------
    def _find_similar_failures(stress_tolerance_mpa: float = 50.0, limit: int = 20) -> Dict[str, Any]:
        try:
            target_stress = float(physics_pred.get("stress_amplitude_mpa") or 0.0)
            if target_stress <= 0:
                return {"error": "no current physics stress available for comparison"}
            memory = get_fleet_memory()
            history = memory.get_history(asset_id, limit=limit, kinds=["cycle_action"])
            matches: List[Dict[str, Any]] = []
            for entry in history:
                payload = entry.payload
                phys = payload.get("physics_prediction") or payload.get("physics") or {}
                past_stress = phys.get("stress_amplitude_mpa")
                if past_stress is None:
                    continue
                if abs(float(past_stress) - target_stress) <= stress_tolerance_mpa:
                    row = _summarize_entry(payload)
                    row["stress_delta_mpa"] = round(float(past_stress) - target_stress, 2)
                    matches.append(row)
            return {
                "asset_id": asset_id,
                "target_stress_mpa": round(target_stress, 2),
                "tolerance_mpa": stress_tolerance_mpa,
                "match_count": len(matches),
                "matches": matches,
            }
        except Exception as exc:
            return {"error": str(exc)}

    # ---- tool 3 -------------------------------------------------
    def _query_material_limits(material_name: str) -> Dict[str, Any]:
        try:
            params = material_db.get_material(material_name)
            if params is None:
                available = list(material_db.materials.keys()) if hasattr(material_db, "materials") else []
                return {"error": f"unknown material {material_name!r}", "available": available[:20]}
            current_stress = float(physics_pred.get("stress_amplitude_mpa") or stress_features.get("stress_amplitude_mpa") or 0.0)
            ratio_to_S_e = current_stress / params.S_e if params.S_e else None
            ratio_to_yield = current_stress / params.yield_strength if params.yield_strength else None
            return {
                "material_name": material_name,
                "ultimate_tensile_strength_mpa": params.S_ut,
                "endurance_limit_mpa": params.S_e,
                "yield_strength_mpa": params.yield_strength,
                "fracture_toughness_mpa_sqrt_m": params.K_IC,
                "threshold_stress_intensity_mpa_sqrt_m": params.K_th,
                "youngs_modulus_gpa": params.E,
                "current_stress_mpa": round(current_stress, 2),
                "stress_over_endurance_ratio": round(ratio_to_S_e, 3) if ratio_to_S_e is not None else None,
                "stress_over_yield_ratio": round(ratio_to_yield, 3) if ratio_to_yield is not None else None,
                "supplier": params.supplier,
            }
        except Exception as exc:
            return {"error": str(exc)}

    # ---- tool 4 -------------------------------------------------
    def _simulate_what_if(stress_delta_mpa: float = 0.0, crack_size_delta_mm: float = 0.0) -> Dict[str, Any]:
        try:
            engine = get_physics_engine()
            base_stress = float(stress_features.get("stress_amplitude_mpa") or 0.0)
            base_range = float(stress_features.get("stress_range_mpa") or 2.0 * base_stress)
            base_mean = float(stress_features.get("mean_stress_mpa") or 0.0)
            base_crack = stress_features.get("crack_size_mm")
            base_crack_val = float(base_crack) if base_crack is not None else None

            new_stress = max(0.0, base_stress + float(stress_delta_mpa))
            new_range = max(0.0, base_range + 2.0 * float(stress_delta_mpa))
            new_crack = None
            if base_crack_val is not None:
                new_crack = max(0.001, base_crack_val + float(crack_size_delta_mm))

            result = engine.analyze(
                material_name=stress_features.get("material_name") or "Steel4340",
                stress_amplitude_mpa=new_stress,
                stress_range_mpa=new_range,
                mean_stress_mpa=base_mean,
                crack_size_mm=new_crack,
                geometry_factor=float(stress_features.get("geometry_factor", 1.0)),
                R_ratio=float(stress_features.get("R_ratio", 0.0)),
                cycles_per_hour=float(stress_features.get("cycles_per_hour", 3600.0)),
                operating_hours_per_day=float(stress_features.get("operating_hours_per_day", 16.0)),
                operational_cycles_actual=(
                    float(operational_cycles_actual)
                    if operational_cycles_actual is not None
                    else None
                ),
            )
            result.pop("_raw", None)
            return {
                "inputs": {
                    "base_stress_mpa": round(base_stress, 2),
                    "new_stress_mpa": round(new_stress, 2),
                    "base_crack_size_mm": base_crack_val,
                    "new_crack_size_mm": new_crack,
                },
                "predicted_rul_hours": result.get("rul_hours"),
                "predicted_health_status": result.get("health_status"),
                "predicted_health_score": result.get("health_score"),
                "failure_mode": result.get("failure_mode"),
            }
        except Exception as exc:
            return {"error": str(exc)}

    # ---- tool 5 -------------------------------------------------
    def _get_recent_maintenance(days: int = 30) -> Dict[str, Any]:
        try:
            memory = get_fleet_memory()
            history = memory.get_history(asset_id, limit=100, kinds=["cycle_action", "engineer_decision"])
            cutoff = datetime.utcnow() - timedelta(days=int(days))
            hits: List[Dict[str, Any]] = []
            for entry in history:
                created = entry.created_at
                try:
                    ts = datetime.fromisoformat(str(created).replace("Z", ""))
                except Exception:
                    continue
                if ts < cutoff:
                    continue
                payload = entry.payload
                verdict = payload.get("judge_verdict") or payload.get("verdict") or {}
                hits.append({
                    "cycle_id": entry.cycle_id,
                    "kind": entry.kind,
                    "created_at": str(created),
                    "route": verdict.get("route"),
                    "maintenance_required": verdict.get("maintenance_required"),
                    "action_summary": (payload.get("action_plan") or {}).get("summary")
                        or (payload.get("action") or {}).get("summary"),
                })
            return {
                "asset_id": asset_id,
                "window_days": int(days),
                "count": len(hits),
                "entries": hits[:20],
            }
        except Exception as exc:
            return {"error": str(exc)}

    return [
        StructuredTool.from_function(
            func=_get_fleet_memory_stats,
            name="get_fleet_memory_stats",
            description=(
                "Get the counts + newest N cycle summaries for the current "
                "asset from fleet memory. Use this to check whether the "
                "asset has recent history and how many action-cycles fired."
            ),
            args_schema=_AssetHistoryArgs,
        ),
        StructuredTool.from_function(
            func=_find_similar_failures,
            name="find_similar_failures",
            description=(
                "Find previous cycle_action entries for THIS asset whose "
                "physics stress is within ±tolerance MPa of the current "
                "physics prediction. Use this to see if similar stress "
                "levels required maintenance in the past."
            ),
            args_schema=_SimilarFailuresArgs,
        ),
        StructuredTool.from_function(
            func=_query_material_limits,
            name="query_material_limits",
            description=(
                "Look up ultimate tensile, endurance limit, yield strength, "
                "and fracture toughness for a material, plus current stress "
                "ratios. Use this to sanity-check whether the current stress "
                "is genuinely dangerous vs still safe within design margins."
            ),
            args_schema=_MaterialLimitsArgs,
        ),
        StructuredTool.from_function(
            func=_simulate_what_if,
            name="simulate_what_if",
            description=(
                "Re-run the physics engine with an adjusted stress "
                "amplitude and/or crack size (signed deltas). Use this to "
                "test hypotheses like 'if stress rises 20 MPa, does the "
                "RUL cliff?' before recommending action."
            ),
            args_schema=_WhatIfArgs,
        ),
        StructuredTool.from_function(
            func=_get_recent_maintenance,
            name="get_recent_maintenance",
            description=(
                "List cycle_action + engineer_decision entries for this "
                "asset within the last N days. Use this to check whether "
                "maintenance was recently performed (avoid redundant "
                "recommendations)."
            ),
            args_schema=_RecentMaintArgs,
        ),
    ]


__all__ = ["build_judge_tools"]
