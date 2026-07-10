"""
Wraps FatigueAnalyzer (Basquin + Paris + NASGRO) from anamoly-detection.py.

The Physics Agent calls this engine. It returns both the raw
`LifecyclePrediction` dataclass AND a JSON-friendly dict the LangGraph
state schema can serialize.

IMPORTANT — health_score override
---------------------------------
The upstream `FatigueAnalyzer.predict_lifecycle()` (both in
`anamoly-detection.py` and `prediction-managent.py`) computes
`life_used_percent` with `random.uniform(0.2, 0.6)` — a placeholder
the original authors left in place with the comment
"Calculate life used based on operational cycles".

We override it here with a **stress-based** health formula so the
score reflects the classical fatigue safety factor
(applied stress vs endurance limit). That matches the intuition an
engineer has when reading the panel: low stress ratio → high score,
stress above the endurance limit → sharply declining score.

See `_health_score_from_stress()` below for the exact band mapping.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from .._legacy_imports import anomaly_mod as _canon
from ..models import FatigueParameters, HealthStatus, LifecyclePrediction
from .material_engine import get_material_params


class PhysicsEngine:
    """Calls FatigueAnalyzer.predict_lifecycle — no math reimplemented here."""

    def analyze(
        self,
        material_name: str,
        stress_amplitude_mpa: float,
        stress_range_mpa: float,
        mean_stress_mpa: float = 0.0,
        crack_size_mm: Optional[float] = None,
        geometry_factor: float = 1.0,
        R_ratio: float = 0.0,
        cycles_per_hour: float = 3600.0,
        operating_hours_per_day: float = 16.0,
        operational_cycles_actual: Optional[float] = None,
    ) -> Dict[str, Any]:
        params: FatigueParameters = get_material_params(material_name)
        if params is None:
            raise ValueError(f"Unknown material {material_name!r}")

        analyzer = _canon.FatigueAnalyzer(params)
        crack_size_m = (crack_size_mm / 1000.0) if crack_size_mm else None

        pred: LifecyclePrediction = analyzer.predict_lifecycle(
            stress_amplitude=stress_amplitude_mpa,
            stress_range=stress_range_mpa,
            mean_stress=mean_stress_mpa,
            crack_size=crack_size_m,
            geometry_factor=geometry_factor,
            R_ratio=R_ratio,
            operating_hours_per_day=operating_hours_per_day,
            cycles_per_hour=cycles_per_hour,
        )

        total_life_cycles = pred.total_life_cycles

        # ---- deterministic RUL / cycles-used from operational_cycles ----
        rul_hours = pred.remaining_life_hours
        rul_years = pred.remaining_life_years
        cycles_used_percent = pred.cycles_used_percent

        if (
            operational_cycles_actual is not None
            and total_life_cycles is not None
            and not (isinstance(total_life_cycles, float) and (np.isinf(total_life_cycles) or np.isnan(total_life_cycles)))
            and total_life_cycles > 0
        ):
            life_used = float(operational_cycles_actual) / float(total_life_cycles)
            life_used = max(0.0, min(0.99, life_used))
            remaining_cycles = float(total_life_cycles) * (1.0 - life_used)
            rul_hours = remaining_cycles / float(cycles_per_hour)
            rul_years = rul_hours / (8760.0 * 0.8)
            cycles_used_percent = life_used * 100.0

        # ---- stress-based health_score (independent of operational cycles) ----
        endurance_limit_mpa = float(params.S_e or 0.0)
        yield_strength_mpa = float(params.yield_strength or 0.0) if hasattr(params, "yield_strength") else 0.0
        ultimate_tensile_mpa = float(params.S_ut or 0.0)
        stress_health = _health_score_from_stress(
            stress_amplitude_mpa=float(stress_amplitude_mpa),
            endurance_limit_mpa=endurance_limit_mpa,
            yield_strength_mpa=yield_strength_mpa,
            ultimate_tensile_mpa=ultimate_tensile_mpa,
        )

        # ---- life-used-based health score ----
        # The stress-based score above only measures the *current* load
        # margin (σ vs endurance limit). It doesn't know that the part
        # may already be near end of design life. We combine the two by
        # taking the worse of them so the UI never shows a "NORMAL"
        # health badge next to "urgent maintenance in 6 days".
        life_health = _health_score_from_cycles_used(cycles_used_percent)

        health_score = min(stress_health, life_health)
        health_status_value = _health_status_from_score(health_score)

        # ---- deterministic predicted_failure_date + recommendations ----
        # `FatigueAnalyzer.predict_lifecycle` in prediction-managent.py
        # computes `life_used_percent` with `random.uniform(0.2, 0.6)`,
        # which cascades into a random `predicted_failure_date` and
        # random `maintenance_recommendations` on the returned dataclass.
        # We derive both here from the deterministic RUL + stress-based
        # health status so the Lifecycle Prediction panel in the UI
        # stops showing values that change every run.
        predicted_failure_date_iso = _deterministic_failure_date(rul_hours)
        recommendations = _deterministic_recommendations(
            analyzer=analyzer,
            health_status_value=health_status_value,
            rul_hours=rul_hours,
            failure_mode=pred.failure_mode,
        )

        return {
            "source": "physics",
            "method": "basquin+paris+nasgro",
            "material_name": material_name,
            "stress_amplitude_mpa": stress_amplitude_mpa,
            "predicted_stress_mpa": stress_amplitude_mpa,  # physics takes stress as input
            "rul_hours": _finite(rul_hours),
            "rul_years": _finite(rul_years),
            "total_life_cycles": _finite(total_life_cycles),
            "total_life_hours": _finite(pred.total_life_hours),
            "total_life_years": _finite(pred.total_life_years),
            "cycles_used_percent": cycles_used_percent,
            "failure_mode": pred.failure_mode.value,
            "health_score": health_score,
            "health_status": health_status_value,
            "crack_size_m": pred.crack_size,
            "stress_intensity_factor": pred.stress_intensity_factor,
            "confidence_lower_cycles": _finite(pred.confidence_lower),
            "confidence_upper_cycles": _finite(pred.confidence_upper),
            "simulation_correlation": pred.simulation_correlation,
            "predicted_failure_date": predicted_failure_date_iso,
            "recommendations": recommendations,
            "health_score_source": "min(stress_based, life_used_based)",
            "health_score_stress": stress_health,
            "health_score_life": life_health,
            "stress_ratio_vs_endurance": (
                round(float(stress_amplitude_mpa) / endurance_limit_mpa, 3)
                if endurance_limit_mpa > 0 else None
            ),
            "endurance_limit_mpa": endurance_limit_mpa or None,
            "yield_strength_mpa": yield_strength_mpa or None,
            "ultimate_tensile_mpa": ultimate_tensile_mpa or None,
            "operational_cycles_actual": operational_cycles_actual,
            "_raw": pred,  # kept for downstream agents that want the dataclass
        }


def _health_score_from_stress(
    stress_amplitude_mpa: float,
    endurance_limit_mpa: float,
    yield_strength_mpa: float,
    ultimate_tensile_mpa: float,
) -> float:
    """
    Health-score from the classical fatigue safety-factor idea:
    the ratio of applied stress to the material's endurance limit.

    Bands (aerospace / ASME convention):

        r = stress / endurance_limit

        r ≤ 0.5   → 95 – 100   (very safe, high margin)
        r = 1.0   → 80         (at endurance boundary — theoretically infinite life)
        r = 1.5   → 50         (finite-life territory — WARNING)
        r = 2.0   → 20         (severe high-cycle fatigue — CRITICAL)
        r ≥ 3.0   → 0          (approaching yield / FAILED)

    Two extra guards:
      - If the material entry has no endurance limit, we fall back to
        a mid-band 50 so downstream logic doesn't panic.
      - If applied stress is at or above yield, score collapses toward 0
        regardless of the endurance-limit ratio (localised plastic
        deformation = component compromised).
    """
    if stress_amplitude_mpa < 0:
        stress_amplitude_mpa = 0.0
    if endurance_limit_mpa <= 0:
        return 50.0  # unknown material — neutral score

    # Instant-fail check against yield / UTS.
    if yield_strength_mpa > 0 and stress_amplitude_mpa >= yield_strength_mpa:
        return 0.0
    if ultimate_tensile_mpa > 0 and stress_amplitude_mpa >= ultimate_tensile_mpa:
        return 0.0

    r = stress_amplitude_mpa / endurance_limit_mpa

    if r <= 0.5:
        # Safe zone: 100 → 95 as r goes 0 → 0.5
        health = 100.0 - 10.0 * r
    elif r <= 1.0:
        # Approaching endurance limit: 95 → 80
        health = 95.0 - 30.0 * (r - 0.5)
    elif r <= 1.5:
        # Above endurance, finite life: 80 → 50
        health = 80.0 - 60.0 * (r - 1.0)
    elif r <= 2.0:
        # Elevated finite-life risk: 50 → 20
        health = 50.0 - 60.0 * (r - 1.5)
    elif r <= 3.0:
        # Severe HCF territory: 20 → 0
        health = 20.0 - 20.0 * (r - 2.0)
    else:
        health = 0.0

    return max(0.0, min(100.0, health))


def _health_status_from_score(score: float) -> str:
    """Same 5-band mapping as anamoly-detection.py / prediction-managent.py."""
    if score > 80:
        return HealthStatus.NORMAL.value
    if score > 60:
        return HealthStatus.MONITORING.value
    if score > 40:
        return HealthStatus.WARNING.value
    if score > 20:
        return HealthStatus.CRITICAL.value
    return HealthStatus.FAILED.value


def _health_score_from_cycles_used(pct: Optional[float]) -> float:
    """
    Life-used-based health score. Maps the fraction of total design
    life already consumed onto the same 0-100 scale the stress-based
    score uses, so the two can be combined with a simple ``min()``:

        <20% used   → 95  (well within design life)
        <40% used   → 75
        <60% used   → 55
        <80% used   → 35
        ≥80% used   → 10  (approaching end-of-life)

    Returns 100 (no penalty) if `pct` is None / NaN so components with
    unknown mileage don't get artificially downgraded.
    """
    if pct is None:
        return 100.0
    try:
        v = float(pct)
    except (TypeError, ValueError):
        return 100.0
    if np.isnan(v) or np.isinf(v):
        return 100.0
    if v < 20:
        return 95.0
    if v < 40:
        return 75.0
    if v < 60:
        return 55.0
    if v < 80:
        return 35.0
    return 10.0


def _finite(x: float) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (np.isinf(x) or np.isnan(x)):
        return None
    return float(x)


def _deterministic_failure_date(rul_hours: Optional[float]) -> str:
    """
    ISO-string predicted failure date derived from the deterministic RUL.

    Replaces `FatigueAnalyzer.predict_lifecycle().predicted_failure_date`
    which is computed from a `random.uniform(0.2, 0.6)` `life_used_percent`
    and therefore changes on every run for the same inputs.
    """
    if rul_hours is None or (isinstance(rul_hours, float) and (np.isinf(rul_hours) or np.isnan(rul_hours))):
        # Unknown / infinite life — schedule the placeholder date ~10y out.
        return (datetime.now() + timedelta(days=365 * 10)).isoformat()
    hours = max(0.0, float(rul_hours))
    return (datetime.now() + timedelta(hours=hours)).isoformat()


def _deterministic_recommendations(
    analyzer: Any,
    health_status_value: str,
    rul_hours: Optional[float],
    failure_mode: Any,
) -> List[str]:
    """
    Regenerate the recommendations list from the *deterministic* health
    status + RUL, using the legacy `FatigueAnalyzer._generate_recommendations`
    method so the emoji-decorated text stays identical to the CLI.

    The list on `LifecyclePrediction.maintenance_recommendations` returned
    by `predict_lifecycle()` is derived from the random `remaining_life_hours`
    and random `health_status` — so it also changes every run. Rebuilding
    it here keeps the UI panel consistent across identical inputs.
    """
    if isinstance(rul_hours, float) and (np.isinf(rul_hours) or np.isnan(rul_hours)):
        hours = 8760.0 * 10  # ~10 years — matches legacy fallback
    elif rul_hours is None:
        hours = 168.0  # safe fallback (1 week)
    else:
        hours = max(0.0, float(rul_hours))

    # Convert the string health status back into the legacy enum object
    # the recommender expects.
    try:
        health_status_enum = _canon.HealthStatus(health_status_value)
    except ValueError:
        health_status_enum = _canon.HealthStatus.MONITORING

    try:
        return list(
            analyzer._generate_recommendations(health_status_enum, hours, failure_mode)
        )
    except Exception:
        # Belt-and-braces: never let a recommender crash the physics node.
        return []

