"""
Digital-twin / simulation abstraction.

Real coupling to ANSYS / Abaqus / Creo / NASTRAN is out of scope for
this hackathon build. To plug one in, implement a subclass of
`SimulationLayer` and register it in place of `SyntheticSimulationAdapter`
inside `graph/build_graph.py` — no agent code needs to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from ..models import AssetType


@dataclass
class StressFeatures:
    """Contract every SimulationLayer must return for a flagged component."""
    asset_id: str
    asset_type: str
    component: str
    material_name: str
    # Core physics quantities the fatigue engine consumes
    stress_amplitude_mpa: float
    stress_range_mpa: float
    mean_stress_mpa: float
    crack_size_mm: Optional[float] = None
    geometry_factor: float = 1.0
    R_ratio: float = 0.0
    # Duty-cycle context (drives Basquin/Paris integration)
    cycles_per_hour: float = 3600.0
    operating_hours_per_day: float = 16.0
    # Provenance — every adapter MUST label whether the stress values
    # came from a real solver or a synthetic surrogate.
    source: str = "unspecified"
    is_synthetic: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)


class SimulationLayer(ABC):
    """Interface for anything that produces stress features for a component."""

    name: str = "abstract"

    @abstractmethod
    def compute_stress_features(
        self,
        asset_id: str,
        asset_type: AssetType,
        component: str,
        material_name: str,
        sensor_batch: pd.DataFrame,
        anomaly_context: Optional[Dict[str, Any]] = None,
    ) -> StressFeatures:
        """Return stress features for the flagged component only."""
