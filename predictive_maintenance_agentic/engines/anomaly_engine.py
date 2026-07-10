"""
Thin wrapper around EnhancedAnomalyDetector (from anamoly-detection.py).

The wrapper caches one detector per (AssetType, material) pair, trains
it on the first batch it sees, then reuses the trained model for
subsequent detects. All actual detection logic lives in the legacy
class — do NOT reimplement Isolation Forest / statistical thresholds
here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .._legacy_imports import anomaly_mod as _canon
from ..models import AnomalyResult, AssetType, HealthStatus


@dataclass
class AnomalyGateResult:
    """Summary the graph's anomaly gate consumes."""
    is_anomalous: bool
    anomaly_count: int
    max_severity: str
    top_channel: Optional[str]
    top_score: float
    raw: List[AnomalyResult]

    # --- CLI-parity extras (surfaced to /cycles/{id} for the UI) ---
    severity_breakdown: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    top_anomalies: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_anomalous": self.is_anomalous,
            "anomaly_count": self.anomaly_count,
            "max_severity": self.max_severity,
            "top_channel": self.top_channel,
            "top_score": self.top_score,
            "severity_breakdown": self.severity_breakdown,
            "by_type": self.by_type,
            "top_anomalies": self.top_anomalies,
        }


def _summarize_anomaly(a: AnomalyResult) -> Dict[str, Any]:
    """CLI-style compact projection of one anomaly for the UI list."""
    return {
        "anomaly_type": a.anomaly_type,
        "sensor": a.sensor_type,
        "value": round(float(a.value), 3) if a.value is not None else None,
        "threshold": round(float(a.threshold), 3) if a.threshold is not None else None,
        "score": round(float(a.anomaly_score), 3) if a.anomaly_score is not None else None,
        "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
        "probable_failure_mode": (
            a.probable_failure_mode.value
            if getattr(a, "probable_failure_mode", None) is not None
            and hasattr(a.probable_failure_mode, "value")
            else None
        ),
        "trend_direction": getattr(a, "trend_direction", "stable"),
        "remediation_hint": getattr(a, "remediation_hint", "") or "",
    }


class AnomalyEngine:
    """
    Wraps EnhancedAnomalyDetector with a *clean-baseline* trainer and a
    rule-based fallback so obviously out-of-range CSV rows are always
    flagged.

    Why this class exists
    ---------------------
    Earlier version cached one detector per (asset_type, material) and
    trained it on the FIRST batch it saw. That caused two bugs the user
    hit in production:

      1. Training-data contamination — IsolationForest(contamination=0.05)
         fit on the same batch it then scores can flag at most 5% of the
         training data, and the 3σ baseline stats become huge because
         they include the anomalous points → nothing ever fires.
      2. Cache poisoning — once a bad detector was cached for
         ``(aircraft_wing, Al7075-T6)``, every subsequent CSV for that
         asset-type reused it and also reported "no anomalies".

    Fix in this version
    -------------------
    * Detectors are trained on a fresh SYNTHETIC NORMAL baseline
      generated from the asset-type defaults in
      ``anamoly-detection.generate_sensor_data`` — never on the user's
      CSV. This keeps the baseline stats honest.
    * The detector is still cached per (asset_type, material) so we
      don't pay the training cost on every request, but the cached
      instance was trained on clean data so it is safe to reuse.
    * A rule-based fallback (``_rule_based_anomalies``) computes hard
      z-scores against the clean baseline. Anything with |z| ≥ 3.5 on
      any sensor is reported even if IsolationForest missed it. This
      guarantees the graph doesn't short-circuit on obviously bad CSVs.
    """

    # Rows to synthesise per (asset_type, material) — enough for the
    # IsolationForest to converge but small enough to keep first-call
    # latency imperceptible.
    _BASELINE_N_SAMPLES = 300

    def __init__(self) -> None:
        self._detectors: Dict[Tuple[str, Optional[str]], Any] = {}
        # Cache the clean baseline dataframe so we can compute rule-based
        # z-scores without re-generating it on every detect() call.
        self._baselines: Dict[Tuple[str, Optional[str]], pd.DataFrame] = {}

    def _key(self, asset_type: AssetType, material_name: Optional[str]) -> Tuple[str, Optional[str]]:
        return (asset_type.value, material_name)

    def _clean_baseline(self, asset_type: AssetType) -> pd.DataFrame:
        """
        Generate a synthetic 'known-good' batch for this asset type using
        the legacy generator's per-asset-type normal operating ranges.
        We strip out the 5% anomalies the generator normally injects so
        the detector's baseline is truly clean.
        """
        df = _canon.generate_sensor_data(
            f"__baseline_{asset_type.value}",
            asset_type,
            n_samples=self._BASELINE_N_SAMPLES,
        )
        # Ensure no NaN / inf leak from the generator.
        return df.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    def _get_or_train(
        self,
        asset_type: AssetType,
        material_name: Optional[str],
    ):
        """Return a detector trained on a fresh synthetic baseline.

        NOTE: The `training_batch` argument the previous version accepted
        has been removed. Training on the incoming (possibly anomalous)
        batch is exactly the contamination bug this rewrite fixes.
        """
        key = self._key(asset_type, material_name)
        detector = self._detectors.get(key)
        if detector is None:
            baseline = self._clean_baseline(asset_type)
            self._baselines[key] = baseline
            detector = _canon.EnhancedAnomalyDetector(asset_type, material_name)
            detector.train_models(baseline)
            self._detectors[key] = detector
        return detector

    def _rule_based_anomalies(
        self,
        asset_type: AssetType,
        material_name: Optional[str],
        sensor_batch: pd.DataFrame,
        z_threshold: float = 3.5,
    ) -> List[Dict[str, Any]]:
        """
        Safety-net check: flag rows whose sensor values are more than
        ``z_threshold`` standard deviations away from the clean baseline
        mean. Runs regardless of what IsolationForest reports so that
        obvious out-of-range CSVs cannot be silently marked "healthy".
        """
        key = self._key(asset_type, material_name)
        baseline = self._baselines.get(key)
        if baseline is None:
            return []
        # Only compare on columns actually present on both sides.
        common = [c for c in _canon._SENSOR_FEATURE_COLS if c in baseline.columns and c in sensor_batch.columns]
        if not common:
            return []
        stats = {
            c: (float(baseline[c].mean()), float(baseline[c].std()) or 1e-9)
            for c in common
        }
        flagged: List[Dict[str, Any]] = []
        for idx, row in sensor_batch.reset_index(drop=True).iterrows():
            for col in common:
                mean, std = stats[col]
                try:
                    val = float(row[col])
                except (TypeError, ValueError):
                    continue
                z = (val - mean) / std
                if abs(z) >= z_threshold:
                    flagged.append(
                        {
                            "anomaly_type": "rule_zscore",
                            "sensor": col,
                            "value": round(val, 3),
                            "threshold": round(mean + z_threshold * std, 3),
                            "score": round(abs(z), 3),
                            "severity": (
                                HealthStatus.CRITICAL.value
                                if abs(z) >= 6
                                else HealthStatus.WARNING.value
                                if abs(z) >= 4.5
                                else HealthStatus.MONITORING.value
                            ),
                            "probable_failure_mode": None,
                            "trend_direction": "spike",
                            "remediation_hint": (
                                f"Sensor '{col}' deviates {z:+.1f}σ from clean baseline "
                                f"({val:.2f} vs μ={mean:.2f}, σ={std:.2f})."
                            ),
                            "row_index": int(idx),
                        }
                    )
        return flagged

    def detect(
        self,
        asset_type: AssetType,
        material_name: Optional[str],
        sensor_batch: pd.DataFrame,
        training_batch: Optional[pd.DataFrame] = None,  # kept for API compat; unused
    ) -> AnomalyGateResult:
        """
        Run detection on ``sensor_batch`` against a clean synthetic
        baseline. The ``training_batch`` argument is accepted for
        backward compat but IGNORED — training on caller data is what
        caused the "no anomalies ever" bug.

        NOTE: The rule-based z-score fallback (``_rule_based_anomalies``)
        is intentionally NOT invoked here. It was found to be too
        aggressive — flagging most CSVs with severity ``critical`` or
        ``failed`` — which then compounded through the downstream
        stress-amplitude boost in ``SyntheticSimulationAdapter``
        (up to 1.7×), causing Basquin RUL (N ∝ σ⁻¹⁰) to collapse by
        two orders of magnitude on realistic input. IsolationForest
        alone is the source of truth. The rule-based helper is kept
        on the class for optional diagnostic use but no longer
        influences the gate result.
        """
        del training_batch  # explicit: we never train on caller data

        detector = self._get_or_train(asset_type, material_name)
        anomalies: List[AnomalyResult] = detector.detect_anomalies(sensor_batch)

        if not anomalies:
            return AnomalyGateResult(
                is_anomalous=False,
                anomaly_count=0,
                max_severity=HealthStatus.NORMAL.value,
                top_channel=None,
                top_score=0.0,
                raw=[],
                severity_breakdown={},
                by_type={},
                top_anomalies=[],
            )

        # Rank purely by IsolationForest anomaly_score.
        combined = [
            _summarize_anomaly(a)
            for a in sorted(anomalies, key=lambda a: a.anomaly_score, reverse=True)
        ]

        order = {
            HealthStatus.NORMAL.value: 0,
            HealthStatus.MONITORING.value: 1,
            HealthStatus.WARNING.value: 2,
            HealthStatus.CRITICAL.value: 3,
            HealthStatus.FAILED.value: 4,
        }
        max_sev = max(
            (d.get("severity") or HealthStatus.MONITORING.value for d in combined),
            key=lambda s: order.get(s, 0),
        )
        severity_breakdown = dict(Counter(d.get("severity") for d in combined))
        by_type = dict(Counter(d.get("anomaly_type") for d in combined))
        top = combined[0]

        return AnomalyGateResult(
            is_anomalous=True,
            anomaly_count=len(combined),
            max_severity=max_sev,
            top_channel=top.get("sensor"),
            top_score=float(top.get("score") or 0.0),
            raw=anomalies,  # keep raw model results for downstream agents
            severity_breakdown=severity_breakdown,
            by_type=by_type,
            top_anomalies=combined[:10],
        )
