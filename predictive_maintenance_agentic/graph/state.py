"""
Shared LangGraph state schema (section 4 of the build prompt).

We serialize the incoming sensor batch as an in-memory `dict` (records
form) so LangGraph checkpointing can round-trip it through SQLite
without pandas-specific pickling.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class CycleState(TypedDict, total=False):
    # --- identity ---
    cycle_id: str
    asset_id: str
    asset_type: str
    component: Optional[str]
    material_name: Optional[str]

    # --- data ingestion ---
    sensor_batch: Dict[str, Any]      # {"records": [...], "columns": [...]}

    # --- anomaly gate ---
    anomaly_result: Optional[Dict[str, Any]]
    is_anomalous: bool

    # --- simulation layer (only if anomalous) ---
    stress_features: Optional[Dict[str, Any]]

    # --- physics ⇄ ML negotiation ---
    physics_prediction: Optional[Dict[str, Any]]
    ml_correction: Optional[Dict[str, Any]]
    negotiation_round: int            # capped at LIMITS.max_negotiation_rounds
    negotiation_history: List[Dict[str, Any]]
    dialogue_history: List[Dict[str, Any]]   # structured moves per round
    pending_data_request: Optional[str]      # tool name requested in round 2
    fetched_features: Optional[Dict[str, Any]]  # data_fetch_agent output

    # --- judge / calibration / engineer loop ---
    judge_verdict: Optional[Dict[str, Any]]
    calibration_result: Optional[Dict[str, Any]]
    engineer_decision: Optional[Dict[str, Any]]
    resimulation_round: int           # capped at LIMITS.max_resimulation_rounds

    # --- action + memory ---
    action: Optional[Dict[str, Any]]
    work_order: Optional[Dict[str, Any]]
    fleet_memory_refs: List[str]

    # --- post-cycle reflection (Critic) ---
    critic_review: Optional[Dict[str, Any]]

    # --- trace (populated by every node for /cycles/{id}) ---
    trace: List[Dict[str, Any]]

    # --- terminal status flags ---
    status: str                       # "running" | "normal_end" | "action_taken" | "unresolved_divergence"
    errors: List[str]
