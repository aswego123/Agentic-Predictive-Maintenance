"""
Thin wrapper around the two legacy `generate_sensor_data` implementations.

Both files define an identical-signature function; we expose the
canonical one from anamoly-detection.py and add an `inject_anomaly`
helper for driving the graph's anomaly gate in tests / demos.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .._legacy_imports import anomaly_mod as _canon
from ..models import AssetType


def generate_sensor_data(
    asset_id: str,
    asset_type: AssetType,
    n_samples: int = 100,
) -> pd.DataFrame:
    """Delegate directly to the canonical generator."""
    return _canon.generate_sensor_data(asset_id, asset_type, n_samples)


def inject_anomaly(
    df: pd.DataFrame,
    channel: str = "vibration",
    multiplier: float = 5.0,
    row_index: Optional[int] = None,
) -> pd.DataFrame:
    """
    Force a hard anomaly into a synthetic sensor batch so the graph's
    anomaly gate has something to trip on. Returns a *copy* so callers
    can compare against the original.
    """
    out = df.copy()
    if channel not in out.columns:
        raise KeyError(f"Channel {channel!r} not in sensor batch columns: {list(out.columns)}")
    idx = row_index if row_index is not None else len(out) - 1
    out.loc[idx, channel] = out[channel].mean() + multiplier * (out[channel].std() or 1.0)
    return out


def generate_normal_batch(asset_id: str, asset_type: AssetType, n_samples: int = 50) -> pd.DataFrame:
    """
    Generate a batch tuned to look 'normal' by clamping outlier spikes
    the legacy generator sprinkles in. Useful for exercising the
    anomaly-gate short-circuit path.
    """
    df = generate_sensor_data(asset_id, asset_type, n_samples)
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "operational_cycles":
            continue
        mean = df[col].mean()
        std = df[col].std() or 0.0
        upper = mean + 1.5 * std
        lower = mean - 1.5 * std
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


if __name__ == "__main__":  # pragma: no cover
    df = generate_sensor_data("ENGINE-001", AssetType.AIRCRAFT_ENGINE, n_samples=20)
    print(df.head())
    print(f"\nGenerated {len(df)} rows, columns={list(df.columns)}")
