"""
Import shim for the two existing scripts.

`anamoly-detection.py` and `prediction-managent.py` sit in the parent
directory and use hyphens (so they aren't valid Python module names).
This module loads them at import time via importlib and re-exports the
names the wrapper engines need. All physics/ML logic lives in those
files — do NOT reimplement any of it here.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType


_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)


def _load(module_alias: str, filename: str) -> ModuleType:
    path = os.path.join(_PARENT, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Legacy module {filename!r} not found at {path}. "
            "Keep anamoly-detection.py and prediction-managent.py in "
            "the repo root next to predictive-maintenance-agentic/."
        )
    spec = importlib.util.spec_from_file_location(module_alias, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"Could not build spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = module
    spec.loader.exec_module(module)
    return module


# Canonical fatigue/material engine (superset of dataclasses + suppliers).
anomaly_mod = _load("eix_legacy_anomaly", "anamoly-detection.py")

# Canonical per-part lifecycle engine (PART_CONFIG, PartLifecyclePredictor,
# resolve_asset_type, CSV loaders).
prediction_mod = _load("eix_legacy_prediction", "prediction-managent.py")
