"""
Calibration Agent — Bayesian parameter optimization + LLM suggestion.

Takes the divergent Physics vs ML verdict and proposes adjusted
stress features (specifically stress_amplitude and geometry_factor)
that shrink the gap. Uses `scipy.optimize.minimize_scalar` on a
1-D residual objective — light-weight but real optimization, not a
placeholder.

An LLM (if configured) produces an engineer-friendly suggestion the
human operator sees in the approval form. The numbers themselves are
NEVER derived from the LLM — only the narrative is.

The adjusted features are stored on state.calibration_result and picked
up by simulation_layer_node on re-simulation.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.optimize import minimize_scalar

from ..config import LIMITS, get_llm
from ._shared import append_trace, msg_content_text, to_native


def _bayesian_stress_calibration(physics_stress: float, ml_stress: float, prior_scale: float) -> float:
    """
    Simple Bayesian update: treat physics as prior N(physics_stress,
    prior_scale) and ml as observation N(ml_stress, 0.5*prior_scale).
    Returns the posterior mean.
    """
    sig_prior = max(prior_scale, 1.0)
    sig_obs = max(0.5 * prior_scale, 0.5)
    var_prior = sig_prior ** 2
    var_obs = sig_obs ** 2
    posterior_mean = (physics_stress * var_obs + ml_stress * var_prior) / (var_prior + var_obs)
    return float(posterior_mean)


def calibration_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    physics = state.get("physics_prediction") or {}
    ml = state.get("ml_correction") or {}
    features = state.get("stress_features") or {}
    verdict = state.get("judge_verdict") or {}

    p_stress = float(physics.get("stress_amplitude_mpa") or 0.0)
    m_stress = float(ml.get("predicted_stress_mpa") or 0.0)

    # Bayesian posterior on stress
    posterior_stress = _bayesian_stress_calibration(p_stress, m_stress, prior_scale=max(1.0, 0.1 * p_stress))

    # Optimize geometry_factor so that (posterior_stress * gf_new / gf_old)
    # lies between the two observations — narrow bounded scalar search.
    gf_old = float(features.get("geometry_factor") or 1.0)

    def objective(gf_new: float) -> float:
        scaled = posterior_stress * (gf_new / gf_old)
        return (scaled - m_stress) ** 2 + 0.1 * (gf_new - gf_old) ** 2

    res = minimize_scalar(objective, bounds=(0.5, 2.5), method="bounded")
    gf_new = float(res.x)

    stress_range_ratio = 1.5
    if features.get("stress_amplitude_mpa"):
        stress_range_ratio = float(features.get("stress_range_mpa", posterior_stress * 1.5)) / (
            float(features["stress_amplitude_mpa"]) or 1.0
        )
    adjusted = {
        "stress_amplitude_mpa": posterior_stress,
        "stress_range_mpa": posterior_stress * stress_range_ratio,
        "geometry_factor": gf_new,
    }

    resim_round = int(state.get("resimulation_round", 0)) + 1
    resim_round = min(resim_round, LIMITS.max_resimulation_rounds)

    result = {
        "method": "bayesian_posterior + bounded_scipy_optimize",
        "root_cause_context": verdict.get("root_cause"),
        "prior_stress_mpa": p_stress,
        "observation_stress_mpa": m_stress,
        "posterior_stress_mpa": round(posterior_stress, 2),
        "geometry_factor_old": gf_old,
        "geometry_factor_new": round(gf_new, 4),
        "adjusted_stress_features": adjusted,
        "resimulation_round": resim_round,
        "requires_engineer_approval": True,
    }

    # Add an engineer-facing suggestion. Deterministic fallback first,
    # LLM narrative overlaid if a provider is configured.
    result["engineer_suggestion"] = _default_suggestion(result, verdict)
    llm = get_llm()
    if llm is not None:
        try:
            msg = llm.invoke(
                [
                    ("system",
                     "You are a calibration assistant. Given the physics vs "
                     "ML divergence and the proposed Bayesian adjustment, "
                     "write 2-3 short bullet points telling the on-call "
                     "engineer WHAT changed, WHY it may fix the divergence, "
                     "and what to check before approving. Do not invent "
                     "numbers — only reference the values provided."),
                    ("human", str({"calibration": result, "verdict": verdict})),
                ]
            )
            llm_text = msg_content_text(msg).strip()
            if llm_text:
                result["engineer_suggestion_llm"] = llm_text
        except Exception:  # pragma: no cover
            pass

    trace = append_trace(
        state,
        node="calibration_agent",
        note=(
            f"Calibration round {resim_round}/{LIMITS.max_resimulation_rounds}: "
            f"stress {p_stress:.1f}→{posterior_stress:.1f} MPa, gf {gf_old:.2f}→{gf_new:.2f}"
        ),
        data=result,
    )

    return {
        "calibration_result": to_native(result),
        "resimulation_round": resim_round,
        "trace": to_native(trace),
    }


def _default_suggestion(result: Dict[str, Any], verdict: Dict[str, Any]) -> str:
    delta = result["posterior_stress_mpa"] - result["prior_stress_mpa"]
    direction = "up" if delta > 0 else "down"
    return (
        f"- Physics/ML disagreed ({verdict.get('root_cause', 'divergence detected')}).\n"
        f"- Suggested stress amplitude update: "
        f"{result['prior_stress_mpa']:.1f} → {result['posterior_stress_mpa']:.1f} MPa "
        f"(shifted {direction} by {abs(delta):.1f} MPa via Bayesian posterior).\n"
        f"- Geometry factor: {result['geometry_factor_old']:.2f} → "
        f"{result['geometry_factor_new']:.2f}. Confirm by re-checking the "
        f"stress concentration model / mesh refinement, then Approve to "
        f"re-simulate."
    )
