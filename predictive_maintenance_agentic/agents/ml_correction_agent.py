"""
ML Correction Agent.

Uses the RULEngine (RandomForest + optional GP residual correction) to
produce an ML-adjusted stress + RUL. An LLM, if available, only adds a
brief rationale — the numbers come from sklearn.
"""
from __future__ import annotations

from typing import Any, Dict

from ..config import LIMITS, get_llm
from ..models import AssetType
from ._dialogue import apply_concession, run_dialogue_move
from ._shared import (
    append_trace,
    get_fleet_memory,
    get_ml_engine,
    msg_content_text,
    sensor_batch_from_dict,
    to_native,
)


def _learned_physics_weight(asset_id: str) -> float:
    """Read the Critic's last learned physics weight for this asset, if any."""
    default = 0.8
    if not asset_id:
        return default
    try:
        weights = get_fleet_memory().get_latest_critic_weights(asset_id) or {}
        return float(weights.get("physics_weight", default))
    except Exception:  # pragma: no cover
        return default


def ml_correction_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    sensor_df = sensor_batch_from_dict(state["sensor_batch"])
    asset_type = AssetType(state["asset_type"])
    physics = state.get("physics_prediction") or {}

    engine = get_ml_engine()
    physics_weight = _learned_physics_weight(state.get("asset_id", ""))
    result = engine.correct(asset_type, sensor_df, physics, physics_weight=physics_weight)

    llm = get_llm()
    rationale = (
        f"Gaussian Process prediction (method={result['method']}): "
        f"stress≈{result['predicted_stress_mpa']:.1f} ± {result.get('stress_std_mpa', 0.0):.1f} MPa, "
        f"RUL≈{result['rul_hours']:.0f} ± {result.get('rul_std_hours', 0.0):.0f} h "
        f"(95% CI: {result['confidence_lower_hours']:.0f}-{result['confidence_upper_hours']:.0f} h)."
    )
    if llm is not None:
        try:
            msg = llm.invoke(
                [
                    ("system",
                     "You are an ML-ops engineer. In <=3 sentences, describe "
                     "how the ML correction differs from the physics prediction "
                     "and what that implies about model trust."),
                    ("human", str({"physics": physics, "ml": result})),
                ]
            )
            llm_text = msg_content_text(msg)
            if llm_text:
                rationale = llm_text
        except Exception:  # pragma: no cover
            pass
    result["rationale"] = rationale

    incoming_round = int(state.get("negotiation_round", 0))
    negotiation_round = min(incoming_round + 1, LIMITS.max_negotiation_rounds)

    history = list(state.get("negotiation_history") or [])
    dialogue = list(state.get("dialogue_history") or [])
    pending_request = state.get("pending_data_request")

    # -------- Dialogue round: ML has seen physics round-2 output --------
    dialogue_move: Dict[str, Any] | None = None
    if incoming_round >= 1:
        last_physics = next(
            (h for h in reversed(history) if h.get("source") == "physics"),
            None,
        )
        if last_physics is not None:
            move = run_dialogue_move(
                self_side="ml",
                own_prediction=result,
                other_prediction=last_physics.get("prediction") or {},
                round_number=negotiation_round,
                prior_moves=dialogue,
            )
            dialogue_move = move
            if move["move"] == "concede" and move.get("revised_prediction"):
                result = apply_concession(result, move["revised_prediction"])
            elif move["move"] == "request_data" and move.get("data_request"):
                pending_request = pending_request or move["data_request"]
            dialogue.append({
                "round": negotiation_round,
                "source": "ml",
                "move": move["move"],
                "rationale": move["rationale"],
                "data_request": move.get("data_request"),
                "revised_prediction": move.get("revised_prediction"),
            })

    history.append({"round": negotiation_round, "source": "ml", "prediction": result})

    note = (
        f"ML round {negotiation_round}/{LIMITS.max_negotiation_rounds}: "
        f"stress={result['predicted_stress_mpa']:.1f} MPa, "
        f"RUL={result['rul_hours']:.0f} h"
    )
    if dialogue_move is not None:
        note += f" [dialogue: {dialogue_move['move']}]"
        if dialogue_move["move"] == "request_data":
            note += f" ({dialogue_move.get('data_request')})"

    trace = append_trace(
        state,
        node="ml_correction_agent",
        note=note,
        data={
            "round": negotiation_round,
            "predicted_stress_mpa": result["predicted_stress_mpa"],
            "rul_hours": result["rul_hours"],
        },
    )

    out: Dict[str, Any] = {
        "ml_correction": to_native(result),
        "negotiation_round": negotiation_round,
        "negotiation_history": to_native(history),
        "dialogue_history": to_native(dialogue),
        "trace": to_native(trace),
    }
    if pending_request is not None:
        out["pending_data_request"] = pending_request
    return out
