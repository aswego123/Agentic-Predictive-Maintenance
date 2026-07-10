"""
Unified data models for the agentic system.

Both `anamoly-detection.py` and `prediction-managent.py` define an
overlapping set of enums / dataclasses (AssetType, FailureMode,
HealthStatus, MaintenancePriority, SensorData, FatigueParameters,
AnomalyResult, LifecyclePrediction, MaintenanceAction,
FleetHealthSummary). The two definitions are mostly identical, with
`anamoly-detection.py` carrying the SUPERSET (supplier fields,
RecommendationType, SupplierInfo, MaterialRecommendation).

Rule from the build prompt: *don't silently drop fields — diff them
and keep the union*. We therefore re-export the enums/dataclasses from
`anamoly-detection.py` as canonical, since they include every field
present in `prediction-managent.py` plus the supplier/material-change
extensions.
"""
from __future__ import annotations

from ._legacy_imports import anomaly_mod as _canon

# ------------------------------------------------------------------
# ENUMS (identical in both files -> take from canonical)
# ------------------------------------------------------------------
AssetType = _canon.AssetType
FailureMode = _canon.FailureMode
HealthStatus = _canon.HealthStatus
MaintenancePriority = _canon.MaintenancePriority
RecommendationType = _canon.RecommendationType  # only in anomaly file

# ------------------------------------------------------------------
# DATACLASSES (superset — anomaly-detection.py's versions)
# ------------------------------------------------------------------
SupplierInfo = _canon.SupplierInfo
MaterialRecommendation = _canon.MaterialRecommendation
FatigueParameters = _canon.FatigueParameters
SensorData = _canon.SensorData
AnomalyResult = _canon.AnomalyResult
LifecyclePrediction = _canon.LifecyclePrediction
MaintenanceAction = _canon.MaintenanceAction
FleetHealthSummary = _canon.FleetHealthSummary


__all__ = [
    # Enums
    "AssetType",
    "FailureMode",
    "HealthStatus",
    "MaintenancePriority",
    "RecommendationType",
    # Dataclasses
    "SupplierInfo",
    "MaterialRecommendation",
    "FatigueParameters",
    "SensorData",
    "AnomalyResult",
    "LifecyclePrediction",
    "MaintenanceAction",
    "FleetHealthSummary",
]
