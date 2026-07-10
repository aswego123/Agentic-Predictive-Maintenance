"""
Critic Agent — post-cycle reflection with per-asset memory.

Runs *after* Action. Reads the last N cycles for this asset from the
FleetMemoryStore and produces:

  * a short retrospective critique (LLM-optional) describing the
    physics/ML agreement pattern, escalation rate and RUL volatility;
  * a learned `physics_weight` in [0.5, 0.9] that the ML correction
    agent will consume on the *next* cycle, replacing the hard-coded
    80/20 blend.

Everything is best-effort: if memory / LLM / stats fail we log to the
trace and let the pipeline finish cleanly. The Critic never modifies
the current cycle's action — only the belief the next cycle will use.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from ..config import get_llm
from ._shared import append_trace, get_fleet_memory, msg_content_text, to_native


# Bounds and update sensitivity for the learned physics weight.
_WEIGHT_MIN = 0.5
_WEIGHT_MAX = 0.9
_WEIGHT_DEFAULT = 0.8
_WEIGHT_STEP = 0.05
_LOOKBACK = 10


def _safe_stdev(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0


def _summarize_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute retrospective diagnostics over the last N cycles."""
    physics_ruls: List[float] = []
    ml_ruls: List[float] = []
    divergences: List[bool] = []
    escalations = 0
    total = 0

    for payload in history:
        total += 1
        physics = payload.get("physics") or {}
        ml = payload.get("ml") or {}
        verdict = payload.get("verdict") or {}
        action = payload.get("action") or {}

        p_rul = physics.get("rul_hours")
        m_rul = ml.get("rul_hours")
        if isinstance(p_rul, (int, float)):
            physics_ruls.append(float(p_rul))
        if isinstance(m_rul, (int, float)):
            ml_ruls.append(float(m_rul))
        divergences.append(bool(verdict.get("divergence")))
        if action.get("type") in ("generate_work_order", "recommend_repair"):
            escalations += 1

    # Mean signed gap (physics − ml) / physics. Positive means physics
    # is optimistic vs ML; negative means physics is pessimistic.
    signed_gaps: List[float] = []
    for p, m in zip(physics_ruls, ml_ruls):
        if p > 0:
            signed_gaps.append((p - m) / p)

    return {
        "n_cycles": total,
        "divergence_rate": (sum(divergences) / total) if total else 0.0,
        "escalation_rate": (escalations / total) if total else 0.0,
        "mean_signed_gap": float(statistics.fmean(signed_gaps)) if signed_gaps else 0.0,
        "mean_abs_gap": float(statistics.fmean([abs(g) for g in signed_gaps])) if signed_gaps else 0.0,
        "physics_rul_std": _safe_stdev(physics_ruls),
        "ml_rul_std": _safe_stdev(ml_ruls),
    }


def _update_physics_weight(
    current_weight: float, stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Bounded update rule for the physics weight based on retrospective stats.

    Semantics:
      * `signed_gap = mean((physics_rul - ml_rul) / physics_rul)`
        > 0  → physics predicts *longer* life than ML (physics optimistic)
        < 0  → physics predicts *shorter* life than ML (physics pessimistic)

    Rules (evaluated in order; first match wins):
      1. Physics optimistic AND ML consistent  → trust ML more    (weight ↓)
      2. Physics pessimistic AND ML noisy      → trust physics more (weight ↑)
      3. Physics pessimistic AND ML consistent → trust ML slightly more (weight ↓)
      4. ML dramatically noisier than physics  → trust physics more (weight ↑)
      5. Otherwise                             → keep weight
    """
    weight = current_weight
    reasons: List[str] = []

    abs_gap = float(stats.get("mean_abs_gap") or 0.0)
    signed_gap = float(stats.get("mean_signed_gap") or 0.0)
    divergence_rate = float(stats.get("divergence_rate") or 0.0)
    physics_std = float(stats.get("physics_rul_std") or 0.0)
    ml_std = float(stats.get("ml_rul_std") or 0.0)

    ml_is_noisy = ml_std > 2.0 * max(physics_std, 1.0)
    disagreement = abs_gap > 0.30

    # Rule 1: physics optimistic (predicts longer life) + ML looks consistent
    #         → physics may be missing a failure mode, trust ML more.
    if disagreement and signed_gap > 0.15 and not ml_is_noisy:
        weight -= _WEIGHT_STEP
        reasons.append(
            f"physics predicts {signed_gap:.0%} longer life than ML and ML is consistent — "
            "trusting ML more"
        )

    # Rule 2: physics pessimistic + ML volatile
    #         → ML is the erratic one, trust physics more (SAFETY-BIASED).
    elif disagreement and signed_gap < -0.15 and ml_is_noisy:
        weight += _WEIGHT_STEP
        reasons.append(
            f"physics predicts {abs(signed_gap):.0%} shorter life than ML but ML σ "
            f"({ml_std:.0f}h) >> physics σ ({physics_std:.0f}h) — trusting physics more"
        )

    # Rule 3: physics pessimistic + ML consistent
    #         → both stable but disagree; slight nudge toward ML.
    elif disagreement and signed_gap < -0.15 and not ml_is_noisy:
        weight -= _WEIGHT_STEP
        reasons.append(
            f"physics predicts {abs(signed_gap):.0%} shorter life than ML and ML is "
            "consistent — slight lean toward ML"
        )

    # Rule 4: ML noise dominates even without a large gap
    #         → physics is the stable anchor.
    elif ml_is_noisy and divergence_rate < 0.3:
        weight += _WEIGHT_STEP
        reasons.append(
            f"ML RUL volatility ({ml_std:.0f}h) >> physics ({physics_std:.0f}h) — "
            "increasing physics weight"
        )

    else:
        reasons.append("agreement stable — keeping physics weight")

    weight = max(_WEIGHT_MIN, min(_WEIGHT_MAX, round(weight, 3)))
    return {"physics_weight": weight, "reasons": reasons}


def _llm_critique(
    stats: Dict[str, Any], adjustment: Dict[str, Any], asset_id: str
) -> Optional[str]:
    llm = get_llm()
    if llm is None:
        return None
    try:
        msg = llm.invoke(
            [
                (
                    "system",
                    "You are a reliability engineering critic. Given a "
                    "summary of the last few maintenance cycles for one asset, "
                    "write a concise (<=3 sentences) retrospective that names "
                    "the dominant failure pattern (agreement, physics-optimism, "
                    "ML-noise, escalation drift), and whether the proposed "
                    "weight update seems justified.",
                ),
                (
                    "human",
                    str(
                        {
                            "asset_id": asset_id,
                            "stats": stats,
                            "adjustment": adjustment,
                        }
                    ),
                ),
            ]
        )
        return msg_content_text(msg) or None
    except Exception:  # pragma: no cover - LLM failures are non-fatal
        return None


def critic_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Post-cycle reflection: adjust learned weights for the *next* cycle."""
    asset_id = state.get("asset_id")
    if not asset_id:
        # Nothing to learn from — pass through.
        return {
            "trace": to_native(
                append_trace(state, node="critic_agent", note="skipped (no asset_id)")
            )
        }

    memory = get_fleet_memory()
    try:
        history = memory.get_recent_cycle_actions(asset_id, limit=_LOOKBACK)
    except Exception as exc:  # pragma: no cover
        return {
            "trace": to_native(
                append_trace(
                    state,
                    node="critic_agent",
                    note=f"memory read failed: {exc}",
                )
            )
        }

    prior = memory.get_latest_critic_weights(asset_id) or {}
    current_weight = float(prior.get("physics_weight") or _WEIGHT_DEFAULT)

    if len(history) < 2:
        review = {
            "n_cycles_considered": len(history),
            "physics_weight": current_weight,
            "rationale": "insufficient history (<2 cycles) — keeping default weight.",
            "stats": {},
        }
        trace = append_trace(
            state,
            node="critic_agent",
            note=f"cold start: {len(history)} cycles in memory, keeping weight={current_weight}",
            data=review,
        )
        return {"critic_review": to_native(review), "trace": to_native(trace)}

    stats = _summarize_history(history)
    adjustment = _update_physics_weight(current_weight, stats)
    new_weight = adjustment["physics_weight"]
    rationale = "; ".join(adjustment["reasons"])

    llm_note = _llm_critique(stats, adjustment, asset_id)
    if llm_note:
        rationale = f"{rationale}. {llm_note.strip()}"

    review = {
        "n_cycles_considered": len(history),
        "previous_physics_weight": current_weight,
        "physics_weight": new_weight,
        "rationale": rationale,
        "stats": stats,
    }

    # Persist the new weight so the *next* cycle's ML agent can read it.
    try:
        memory.add_entry(
            asset_id=asset_id,
            kind="critic_weights",
            cycle_id=state.get("cycle_id"),
            payload=review,
        )
    except Exception:  # pragma: no cover
        pass

    trace = append_trace(
        state,
        node="critic_agent",
        note=(
            f"physics_weight {current_weight} → {new_weight} "
            f"over last {len(history)} cycles (divergence_rate="
            f"{stats['divergence_rate']:.0%}, mean_abs_gap="
            f"{stats['mean_abs_gap']:.0%})"
        ),
        data=review,
    )
    return {"critic_review": to_native(review), "trace": to_native(trace)}
