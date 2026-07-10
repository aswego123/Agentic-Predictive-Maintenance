"""
LangGraph StateGraph wiring.

Layout matches section 1 of the build prompt / FINAL-DIAGRAM.png:

  init  →  ingest(anomaly_gate)
                 │
                 ├─ NOT anomalous → END  (fleet-memory "cycle_normal" already written)
                 │
                 └─ anomalous → simulation → physics ─┐
                                              ↓        │
                                       ml_correction ──┴─ (round 2? yes → judge)
                                              ↓
                                            judge
                                              │
             ┌───── divergent ─────────────┐  ├── ok → action → END
             ↓                             │  │
      calibration → engineer_approval ────┘  │
             ↑                                │
             └──── resim < 5? ────── simulation loop
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from ..agents import (
    action_agent_node,
    calibration_agent_node,
    critic_agent_node,
    data_fetch_agent_node,
    engineer_approval_node,
    ingest_data_node,
    initialize_cycle_node,
    judge_agent_node,
    ml_correction_agent_node,
    physics_agent_node,
    simulation_layer_node,
)
from ..config import LIMITS, MEMORY
from .state import CycleState


# ---------------------------------------------------------------
# Conditional-edge routers
# ---------------------------------------------------------------
def _route_after_ingest(state: Dict[str, Any]) -> str:
    """Anomaly gate: short-circuit to END if not anomalous."""
    return "simulation" if state.get("is_anomalous") else END


def _route_after_ml(state: Dict[str, Any]) -> str:
    """Physics ⇄ ML negotiation: max 2 rounds.

    * If the round cap has not been hit → back to physics.
    * If the cap IS hit but the dialogue produced a `pending_data_request`
      → route through data_fetch first so the judge gets the extra
      signal.
    * Otherwise → judge.
    """
    if int(state.get("negotiation_round", 0)) < LIMITS.max_negotiation_rounds:
        return "physics"  # negotiate another round
    if state.get("pending_data_request"):
        return "data_fetch"
    return "judge"


def _route_after_judge(state: Dict[str, Any]) -> str:
    """
    Route after Judge (matches the user-requested flow):

      * Cap hit (resim_round >= 5) → Action (with unresolved_divergence).
      * Judge says maintenance_required=True → Action FAST PATH (no
        need to burn cycles on calibration; the physics/health signal
        is already clear).
      * Judge says divergence=True → Calibration → Engineer → resim.
      * Otherwise → Action (continue monitoring).
    """
    verdict = state.get("judge_verdict") or {}
    resim = int(state.get("resimulation_round", 0))
    if resim >= LIMITS.max_resimulation_rounds:
        return "action"
    if verdict.get("maintenance_required"):
        return "action"
    if verdict.get("divergence"):
        return "calibration"
    return "action"


def _route_after_engineer(state: Dict[str, Any]) -> str:
    """
    Post-approval routing:
      * approved & resim < cap → simulation (re-run with adjusted params)
      * rejected or cap reached → action (with unresolved flag if needed)
    """
    dec = state.get("engineer_decision") or {}
    resim = int(state.get("resimulation_round", 0))
    approved = bool(dec.get("approved"))
    if approved and resim < LIMITS.max_resimulation_rounds:
        return "simulation"
    return "action"


# ---------------------------------------------------------------
# Builder
# ---------------------------------------------------------------
@dataclass
class GraphBundle:
    graph: Any                       # CompiledGraph
    checkpointer: Optional[Any]      # for API cross-request resume


def build_graph(
    checkpoint_path: Optional[str] = None,
    enable_interrupt: bool = True,
) -> GraphBundle:
    """
    Build and compile the LangGraph state machine.

    * `checkpoint_path`: SQLite path for persistence. Defaults to
      config.MEMORY.checkpoint_path. Pass ":memory:" for tests.
    * `enable_interrupt`: when True (the default), the graph pauses
      before `engineer_approval` — the API layer resumes via
      `graph.invoke(None, config)` after PATCHing the decision in.
    """
    sg: StateGraph = StateGraph(CycleState)

    sg.add_node("init", initialize_cycle_node)
    sg.add_node("ingest", ingest_data_node)
    sg.add_node("simulation", simulation_layer_node)
    sg.add_node("physics", physics_agent_node)
    sg.add_node("ml", ml_correction_agent_node)
    sg.add_node("data_fetch", data_fetch_agent_node)
    sg.add_node("judge", judge_agent_node)
    sg.add_node("calibration", calibration_agent_node)
    sg.add_node("engineer_approval", engineer_approval_node)
    sg.add_node("action", action_agent_node)
    sg.add_node("critic", critic_agent_node)

    # Linear starting edges
    sg.add_edge(START, "init")
    sg.add_edge("init", "ingest")

    # Anomaly gate — hard short-circuit if normal.
    sg.add_conditional_edges("ingest", _route_after_ingest, {
        "simulation": "simulation",
        END: END,
    })

    # Simulation → Physics → ML
    sg.add_edge("simulation", "physics")
    sg.add_edge("physics", "ml")

    # Physics ⇄ ML negotiation (max 2 rounds; optional data-fetch bridge)
    sg.add_conditional_edges("ml", _route_after_ml, {
        "physics": "physics",
        "data_fetch": "data_fetch",
        "judge": "judge",
    })

    # Data-fetch always flows into judge with the enriched features.
    sg.add_edge("data_fetch", "judge")

    # Judge → Calibration | Action
    sg.add_conditional_edges("judge", _route_after_judge, {
        "calibration": "calibration",
        "action": "action",
    })

    # Calibration → Engineer Approval (human-in-the-loop interrupt)
    sg.add_edge("calibration", "engineer_approval")

    # Engineer → re-simulate | Action
    sg.add_conditional_edges("engineer_approval", _route_after_engineer, {
        "simulation": "simulation",
        "action": "action",
    })

    # Action → Critic (post-cycle reflection) → END
    sg.add_edge("action", "critic")
    sg.add_edge("critic", END)

    # ---- Checkpointer (SQLite) ----
    checkpointer = None
    path = checkpoint_path or MEMORY.checkpoint_path
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        if path == ":memory:":
            import sqlite3
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            checkpointer = SqliteSaver(conn)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            import sqlite3
            conn = sqlite3.connect(path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
    except Exception:  # pragma: no cover
        # In-memory fallback
        try:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        except Exception:
            checkpointer = None

    compile_kwargs: Dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if enable_interrupt:
        compile_kwargs["interrupt_before"] = ["engineer_approval"]

    graph = sg.compile(**compile_kwargs)
    return GraphBundle(graph=graph, checkpointer=checkpointer)
