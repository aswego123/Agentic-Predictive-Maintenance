"""
ML-side RUL/stress correction wrapper.

Primary predictor is a **Gaussian Process Regressor** — it produces
both mean and std for stress (MPa) and RUL (hours) directly, so the
Judge Agent gets a native uncertainty band instead of a fabricated one.
The legacy `RULEstimator` (RandomForest) is kept only as a warm-start
prior and a fallback if scikit-learn's GP is unavailable.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .._legacy_imports import anomaly_mod as _canon
from ..models import AssetType


try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    _SK_OK = True
except ImportError:  # pragma: no cover
    _SK_OK = False


# Feature columns fed to the GP (missing columns fall back to 0.0).
_GP_FEATURES = ("vibration", "load_factor", "temperature", "pressure", "speed")


def _feature_vector(df: pd.DataFrame) -> np.ndarray:
    """Extract a fixed-order mean-feature vector from the sensor batch."""
    row = []
    for col in _GP_FEATURES:
        row.append(float(df[col].mean()) if col in df.columns else 0.0)
    return np.array(row, dtype=float)


def _build_anchor_dataset(
    x_current: np.ndarray,
    rul_prior: float,
    stress_prior: float,
    severity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a small in-context training set for the GP, anchored on the
    physics-derived priors and perturbed relatively along each feature.
    Returns (X, y_rul, y_stress).

    Targets are shaped by the *relative* perturbation index (not raw
    Euclidean distance), so features with wildly different scales
    (e.g. speed vs vibration) don't explode the prior.
    """
    n_feat = x_current.shape[0]
    rows: list[np.ndarray] = [x_current.copy()]
    # `magnitude[i]` is the normalized offset amount used to shape
    # the target for row i. 0 for the centre point, 1 for perturbed.
    magnitudes: list[float] = [0.0]

    for i in range(n_feat):
        for delta in (-0.25, 0.25):
            pert = x_current.copy()
            # Relative perturbation of ~25% (with a floor so zero-valued
            # features still move a little).
            pert[i] = pert[i] * (1.0 + delta) + delta
            rows.append(pert)
            magnitudes.append(abs(delta) * 4.0)  # 25% → normalized 1.0

    X = np.vstack(rows)
    mags = np.array(magnitudes, dtype=float)

    # RUL decreases and stress increases with normalized magnitude.
    # Scaling is small (≤15% at magnitude=1) so priors stay realistic.
    y_rul = rul_prior * np.clip(1.0 - 0.15 * mags * severity, 0.05, None)
    y_stress = stress_prior * (1.0 + 0.10 * mags * severity)
    return X, y_rul, y_stress


class RULEngine:
    """
    Gaussian-Process-first RUL + stress predictor. Cached per asset_type
    so we don't refit anchors for every request. The legacy RF is used
    only to seed a prior when physics doesn't supply one.
    """

    def __init__(self) -> None:
        self._rul_by_type: Dict[str, Any] = {}

    def _get_rul(self, asset_type: AssetType):
        key = asset_type.value
        est = self._rul_by_type.get(key)
        if est is None:
            est = _canon.RULEstimator()
            self._rul_by_type[key] = est
        return est

    def train(
        self,
        asset_type: AssetType,
        sensor_data: pd.DataFrame,
        rul_labels: pd.Series,
    ) -> None:
        self._get_rul(asset_type).train_model(sensor_data, rul_labels)

    def correct(
        self,
        asset_type: AssetType,
        sensor_batch: pd.DataFrame,
        physics_prediction: Dict[str, Any],
        physics_weight: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Return an ML-adjusted view using a Gaussian Process as the
        primary predictor for both RUL (hours) and stress (MPa).

        `physics_weight` in [0.0, 1.0] controls the blend between the
        physics RUL and the RF baseline when building the GP prior.
        Defaults to the historical 80/20 physics-heavy prior; the
        Critic agent can override it per-asset via fleet memory.

        Flow:
          1. Extract mean feature vector from the sensor batch.
          2. Build a physics-anchored in-context training set.
          3. Fit two GPRs (RUL, stress) with an RBF+White kernel and
             obtain mean + std at the query point.
          4. Report 95% CI from the GP std.
          5. Fallback to RF-only prediction if GP is unavailable.
        """
        pw = max(0.0, min(1.0, float(physics_weight)))
        est = self._get_rul(asset_type)
        rf_pred = est.predict_rul(sensor_batch)
        rf_rul = float(rf_pred.get("rul_hours") or 0.0)

        physics_rul = float(physics_prediction.get("rul_hours") or rf_rul or 1000.0)
        physics_stress = float(physics_prediction.get("stress_amplitude_mpa") or 250.0)

        # Severity multiplier: how far the sensor envelope pushes us
        # beyond nominal. Higher severity ⇒ larger anchor perturbations
        # ⇒ larger GP std ⇒ wider CI. Bounded to keep the CI meaningful.
        vib = float(sensor_batch["vibration"].mean()) if "vibration" in sensor_batch else 0.0
        load = float(sensor_batch["load_factor"].mean()) if "load_factor" in sensor_batch else 0.0
        severity = max(0.5, min(3.0, 0.5 + vib + load))

        gp_used = False
        gp_rul_std = 0.0
        gp_stress_std = 0.0
        ml_rul = rf_rul or physics_rul
        ml_stress = physics_stress * (0.95 + 0.15 * vib + 0.05 * load)
        rf_trustworthy = False
        rf_ratio: Optional[float] = None

        if _SK_OK:
            try:
                x_current = _feature_vector(sensor_batch)
                # Physics-weighted prior (Critic-tunable). Sanity-check
                # the RF baseline before letting it near the prior: the
                # legacy RULEstimator falls back to random.uniform(100,
                # 1000) when its model is untrained, which can drag the
                # prior *away* from physics for no good reason. Only
                # blend RF when it's within a plausible band of physics
                # (0.5× to 2×); otherwise use physics-only.
                if rf_rul > 0.0 and physics_rul > 0.0:
                    ratio = rf_rul / physics_rul
                    rf_ratio = float(ratio)
                    rf_trustworthy = 0.5 <= ratio <= 2.0
                else:
                    rf_trustworthy = False

                if rf_trustworthy:
                    rul_prior = pw * physics_rul + (1.0 - pw) * rf_rul
                else:
                    rul_prior = physics_rul
                X, y_rul, y_stress = _build_anchor_dataset(
                    x_current, rul_prior, physics_stress, severity
                )

                # Scale features so the RBF length-scale is well-conditioned.
                scaler = StandardScaler().fit(X)
                Xs = scaler.transform(X)
                xq = scaler.transform(x_current.reshape(1, -1))

                kernel = (
                    ConstantKernel(1.0, (1e-2, 1e2))
                    * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2))
                    + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e0))
                )

                gp_rul = GaussianProcessRegressor(
                    kernel=kernel, normalize_y=True, n_restarts_optimizer=2
                ).fit(Xs, y_rul)
                gp_stress = GaussianProcessRegressor(
                    kernel=kernel, normalize_y=True, n_restarts_optimizer=2
                ).fit(Xs, y_stress)

                rul_mean, rul_std = gp_rul.predict(xq, return_std=True)
                stress_mean, stress_std = gp_stress.predict(xq, return_std=True)

                ml_rul = float(rul_mean[0])
                ml_stress = float(stress_mean[0])
                gp_rul_std = float(rul_std[0])
                gp_stress_std = float(stress_std[0])
                gp_used = True
            except Exception:  # pragma: no cover - fall back to RF
                gp_used = False

        # 95% CI from the GP std when available, else RF fallback bounds.
        if gp_used:
            conf_lo = max(0.0, ml_rul - 1.96 * gp_rul_std)
            conf_hi = ml_rul + 1.96 * gp_rul_std
        else:
            conf_lo = float(rf_pred.get("confidence_lower") or ml_rul * 0.8)
            conf_hi = float(rf_pred.get("confidence_upper") or ml_rul * 1.2)

        return {
            "source": "ml_correction",
            "method": "gaussian_process" if gp_used else "rf_fallback",
            "predicted_stress_mpa": ml_stress,
            "stress_std_mpa": gp_stress_std,
            "rul_hours": ml_rul,
            "rul_std_hours": gp_rul_std,
            "confidence_lower_hours": max(0.0, conf_lo),
            "confidence_upper_hours": max(0.0, conf_hi),
            "feature_importance": rf_pred.get("feature_importance", {}),
            "gp_correction_applied": gp_used,
            "gp_features": list(_GP_FEATURES),
            "physics_weight_used": pw,
            "rf_baseline_rul_hours": rf_rul,
            "rf_baseline_used_in_prior": rf_trustworthy,
            "rf_physics_ratio": rf_ratio,
            # Explainability stub — real SHAP/LIME hook goes here.
            "explainability": {
                "provider": "stub",
                "note": "Wire SHAP/LIME here; feature_importance from RF prior is available.",
            },
        }

