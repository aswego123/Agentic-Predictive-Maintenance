"""CSV loading — thin re-exports from prediction-managent.py."""
from __future__ import annotations

from .._legacy_imports import prediction_mod as _pm

load_sensor_csv = _pm.load_sensor_csv
load_sensor_data_from_directory = _pm.load_sensor_data_from_directory

__all__ = ["load_sensor_csv", "load_sensor_data_from_directory"]
