"""
Digital Twin / Simulation Layer node + Physics Agent node.

The simulation node is separate from Physics — matching the diagram —
but they always run back-to-back the first time through. On
re-simulation (after Calibration + Engineer approval) the simulation
node runs again with the calibrated parameters applied.
"""
from __future__ import annotations

from typing import Any, Dict

from ..config import get_llm
from ..models import AssetType
from ._dialogue import apply_concession, run_dialogue_move
from ._shared import (
    append_trace,
    get_physics_engine,
    get_simulation_layer,
    msg_content_text,
    sensor_batch_from_dict,
    to_native,
)


def simulation_layer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Digital-twin box in the diagram — synthetic stress features only."""
    sensor_df = sensor_batch_from_dict(state["sensor_batch"])
    asset_type = AssetType(state["asset_type"])
    material = state.get("material_name") or "Steel4340"
    component = state.get("component") or asset_type.value

    layer = get_simulation_layer()
    features = layer.compute_stress_features(
        asset_id=state["asset_id"],
        asset_type=asset_type,
        component=component,
        material_name=material,
        sensor_batch=sensor_df,
        anomaly_context=state.get("anomaly_result"),
    )

    # Calibration override: if a previous Calibration + Engineer approval
    # supplied adjusted parameters, apply them before physics runs.
    calib = state.get("calibration_result") or {}
    adj = calib.get("adjusted_stress_features") or {}
    for key, value in adj.items():
        if hasattr(features, key):
            setattr(features, key, value)

    feats_dict = {
        "asset_id": features.asset_id,
        "asset_type": features.asset_type,
        "component": features.component,
        "material_name": features.material_name,
        "stress_amplitude_mpa": features.stress_amplitude_mpa,
        "stress_range_mpa": features.stress_range_mpa,
        "mean_stress_mpa": features.mean_stress_mpa,
        "crack_size_mm": features.crack_size_mm,
        "geometry_factor": features.geometry_factor,
        "R_ratio": features.R_ratio,
        "cycles_per_hour": features.cycles_per_hour,
        "operating_hours_per_day": features.operating_hours_per_day,
        "source": features.source,
        "is_synthetic": features.is_synthetic,
        "extras": features.extras,
    }

    trace = append_trace(
        state,
        node="simulation_layer",
        note=f"[{layer.name}] Stress amplitude = {features.stress_amplitude_mpa:.1f} MPa",
        data={
            "adapter": layer.name,
            "is_synthetic": features.is_synthetic,
            "stress_amplitude_mpa": features.stress_amplitude_mpa,
        },
    )

    return {
        "stress_features": to_native(feats_dict),
        "component": component,
        "trace": to_native(trace),
    }


def physics_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Physics Agent: runs Basquin + Paris + NASGRO via PhysicsEngine.

    An LLM (if configured) is only used to produce a short natural-
    language rationale — the numerical result comes from the physics
    engine, never from the model.
    """
    feats = state["stress_features"]

    # Pull the real `operational_cycles` reading from the sensor batch
    # (max across all rows = latest counter reading). PhysicsEngine
    # uses this to compute health_score deterministically instead of
    # the legacy random placeholder.
    operational_cycles_actual: float | None = None
    sensor_payload = state.get("sensor_batch")
    if sensor_payload:
        try:
            sensor_df = sensor_batch_from_dict(sensor_payload)
            if "operational_cycles" in sensor_df.columns:
                col = sensor_df["operational_cycles"].dropna()
                if len(col) > 0:
                    operational_cycles_actual = float(col.max())
        except Exception:  # pragma: no cover — defensive; keep the pipeline moving
            operational_cycles_actual = None

    engine = get_physics_engine()
    result = engine.analyze(
        material_name=feats["material_name"],
        stress_amplitude_mpa=float(feats["stress_amplitude_mpa"]),
        stress_range_mpa=float(feats["stress_range_mpa"]),
        mean_stress_mpa=float(feats.get("mean_stress_mpa", 0.0)),
        crack_size_mm=feats.get("crack_size_mm"),
        geometry_factor=float(feats.get("geometry_factor", 1.0)),
        R_ratio=float(feats.get("R_ratio", 0.0)),
        cycles_per_hour=float(feats.get("cycles_per_hour", 3600.0)),
        operating_hours_per_day=float(feats.get("operating_hours_per_day", 16.0)),
        operational_cycles_actual=operational_cycles_actual,
    )
    # Drop the raw dataclass — not JSON-serializable.
    result.pop("_raw", None)

    llm = get_llm()
    rationale = _rule_based_physics_rationale(result)
    if llm is not None:
        try:
            msg = llm.invoke(
                [
                    ("system",
                     "You are a fatigue-analysis engineer. In <=3 sentences, "
                     "summarize what the physics prediction implies about "
                     "component health. Be quantitative and concise."),
                    ("human", str(result)),
                ]
            )
            rationale = msg_content_text(msg) or _rule_based_physics_rationale(result)
        except Exception:  # pragma: no cover
            pass
    result["rationale"] = rationale

    negotiation_round = int(state.get("negotiation_round", 0))
    history = list(state.get("negotiation_history") or [])
    dialogue = list(state.get("dialogue_history") or [])
    pending_request = state.get("pending_data_request")

    # -------- Dialogue round: physics has seen ML round-1 output --------
    dialogue_move: Dict[str, Any] | None = None
    if negotiation_round >= 1:
        last_ml = next(
            (h for h in reversed(history) if h.get("source") == "ml"),
            None,
        )
        if last_ml is not None:
            move = run_dialogue_move(
                self_side="physics",
                own_prediction=result,
                other_prediction=last_ml.get("prediction") or {},
                round_number=negotiation_round + 1,
                prior_moves=dialogue,
            )
            dialogue_move = move
            if move["move"] == "concede" and move.get("revised_prediction"):
                result = apply_concession(result, move["revised_prediction"])
            elif move["move"] == "request_data" and move.get("data_request"):
                # Only set if nothing is already queued (ML may have set it).
                pending_request = pending_request or move["data_request"]
            dialogue.append({
                "round": negotiation_round + 1,
                "source": "physics",
                "move": move["move"],
                "rationale": move["rationale"],
                "data_request": move.get("data_request"),
                "revised_prediction": move.get("revised_prediction"),
            })

    history.append({"round": negotiation_round, "source": "physics", "prediction": result})

    note = (
        f"Physics: stress={result['stress_amplitude_mpa']:.1f} MPa, "
        f"RUL={result['rul_hours']}, mode={result['failure_mode']}"
    )
    if dialogue_move is not None:
        note += f" [dialogue: {dialogue_move['move']}]"
        if dialogue_move["move"] == "request_data":
            note += f" ({dialogue_move.get('data_request')})"

    trace = append_trace(
        state,
        node="physics_agent",
        note=note,
        data={
            "rul_hours": result["rul_hours"],
            "health_score": result["health_score"],
            "dialogue_move": dialogue_move["move"] if dialogue_move else None,
        },
    )

    out: Dict[str, Any] = {
        "physics_prediction": to_native(result),
        "negotiation_history": to_native(history),
        "dialogue_history": to_native(dialogue),
        "trace": to_native(trace),
    }
    if pending_request is not None:
        out["pending_data_request"] = pending_request
    return out


def _rule_based_physics_rationale(result: Dict[str, Any]) -> str:
    status = result.get("health_status")
    mode = result.get("failure_mode")
    rul = result.get("rul_hours")
    return (
        f"Physics indicates health={status}, dominant mode={mode}, "
        f"RUL≈{rul} h. Estimate uses Basquin+Paris+NASGRO conservative envelope."
    )


__all__ = ["simulation_layer_node", "physics_agent_node"]
