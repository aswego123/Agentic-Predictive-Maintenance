"""
Data-Fetch Agent — services `pending_data_request` from the physics/ML
dialogue by computing lightweight signal-processing features on the
sensor batch and attaching them to state.

This exists so the physics ⇄ ML negotiation can genuinely request
additional evidence ("I need vibration FFT before I can commit") rather
than just re-running with the same numbers.

Currently supported requests:

  * "spectral_features"  — dominant vibration frequency, spectral
    energy, band-limited energy ratio.
  * "thermal_gradient"   — first/second-order temperature and
    oil-temperature trends, showing whether heat is drifting up.

If the request is unknown or the sensor batch is missing the needed
column we log to trace and return the state unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from ._shared import (
    append_trace,
    sensor_batch_from_dict,
    to_native,
)


# ---------------------------------------------------------------
# Tools (plain functions, safe to call from anywhere)
# ---------------------------------------------------------------
def spectral_features(values: List[float]) -> Dict[str, Any]:
    """
    Dominant frequency + spectral energy for a 1-D signal (assumed
    uniformly sampled). Returns Hz-agnostic bin indices — the caller
    decides how to interpret them.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size < 8:
        return {
            "available": False,
            "reason": "signal too short for FFT (<8 samples)",
        }
    arr = arr - arr.mean()
    spectrum = np.abs(np.fft.rfft(arr))
    if spectrum.size <= 1:
        return {"available": False, "reason": "flat signal"}
    # Ignore DC bin.
    body = spectrum[1:]
    total_energy = float((body ** 2).sum())
    if total_energy <= 0:
        return {"available": False, "reason": "zero energy"}
    dom_idx = int(np.argmax(body)) + 1
    top3 = np.argsort(body)[-3:][::-1] + 1  # top-3 bins
    # Band ratio: high-frequency third vs low-frequency third.
    third = max(1, body.size // 3)
    low_energy = float((body[:third] ** 2).sum())
    high_energy = float((body[-third:] ** 2).sum())
    ratio = high_energy / low_energy if low_energy > 0 else float("inf")
    return {
        "available": True,
        "n_samples": int(arr.size),
        "dominant_bin": dom_idx,
        "dominant_bin_rel_energy": float(body[dom_idx - 1] ** 2 / total_energy),
        "top3_bins": [int(b) for b in top3],
        "high_low_energy_ratio": round(ratio if math.isfinite(ratio) else 999.0, 3),
        "total_energy": round(total_energy, 4),
    }


def thermal_gradient(temp: List[float], oil_temp: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Return first-order slope and mean-diff for temperature channels so
    the physics/ML agents can tell if heat is drifting up over the
    batch.
    """
    def _stats(vals: List[float]) -> Dict[str, Any]:
        arr = np.asarray([v for v in vals if v is not None], dtype=float)
        if arr.size < 3:
            return {"available": False, "reason": "not enough samples"}
        x = np.arange(arr.size, dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0])
        return {
            "available": True,
            "n_samples": int(arr.size),
            "slope_per_step": round(slope, 4),
            "range": round(float(arr.max() - arr.min()), 3),
            "mean": round(float(arr.mean()), 3),
        }

    out: Dict[str, Any] = {"temperature": _stats(temp)}
    if oil_temp is not None:
        out["oil_temperature"] = _stats(oil_temp)
    return out


# ---------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------
_DISPATCH = {
    "spectral_features": "vibration",
    "thermal_gradient": "temperature",
}


def data_fetch_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    request = state.get("pending_data_request")
    if not request:
        # Nothing requested — passthrough (should not happen given the
        # router, but keep the node total).
        trace = append_trace(
            state,
            node="data_fetch_agent",
            note="skipped (no pending_data_request)",
        )
        return {"trace": to_native(trace)}

    try:
        sensor_df = sensor_batch_from_dict(state["sensor_batch"])
    except Exception as exc:
        trace = append_trace(
            state,
            node="data_fetch_agent",
            note=f"failed to load sensor batch: {exc}",
        )
        return {"trace": to_native(trace), "pending_data_request": None}

    result: Dict[str, Any] = {"tool": request}

    if request == "spectral_features":
        if "vibration" in sensor_df.columns:
            result.update(spectral_features(sensor_df["vibration"].tolist()))
        else:
            result.update({"available": False, "reason": "no vibration column"})

    elif request == "thermal_gradient":
        temp = sensor_df["temperature"].tolist() if "temperature" in sensor_df.columns else []
        oil = sensor_df["oil_temperature"].tolist() if "oil_temperature" in sensor_df.columns else None
        result.update(thermal_gradient(temp, oil))

    else:
        result.update({"available": False, "reason": f"unknown tool: {request}"})

    if result.get("available"):
        note = (
            f"fetched {request}: "
            + ", ".join(
                f"{k}={v}" for k, v in result.items()
                if k in ("dominant_bin", "slope_per_step", "high_low_energy_ratio", "range")
            )
        )
    else:
        note = f"{request} unavailable: {result.get('reason', '?')}"

    fetched = dict(state.get("fetched_features") or {})
    fetched[request] = result

    trace = append_trace(
        state,
        node="data_fetch_agent",
        note=note,
        data=result,
    )

    return {
        "fetched_features": to_native(fetched),
        "pending_data_request": None,
        "trace": to_native(trace),
    }


__all__ = ["data_fetch_agent_node", "spectral_features", "thermal_gradient"]
