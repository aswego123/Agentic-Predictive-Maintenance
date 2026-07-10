"""
SYNTHETIC simulation adapter.

Derives stress features from the synthetic sensor data + per-part
engineering defaults (PART_CONFIG in prediction-managent.py). Clearly
labelled as synthetic in `StressFeatures.is_synthetic` and in every API
response — nothing here calls a real solver.

In addition to the scalar stress amplitude the fatigue engine consumes,
this adapter also produces a small 3D stress-intensity field
(`extras["stress_field_3d"]`) so the frontend can render a volumetric
heat-map of where the component is most stressed. The field is
synthesized from a parametric mesh — it is NOT an FEA result.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .._legacy_imports import prediction_mod as _pm
from ..models import AssetType
from .base import SimulationLayer, StressFeatures


# Per-part mesh shape (nx, ny, nz) and geometry descriptor. Kept tiny
# so the graph state stays small when checkpointed.
_MESH_BY_TYPE: Dict[AssetType, Dict[str, Any]] = {
    AssetType.AIRCRAFT_ENGINE:        {"shape": (8, 8, 5), "geometry": "cylinder", "hotspot_region": "root"},
    AssetType.AIRCRAFT_LANDING_GEAR:  {"shape": (6, 6, 8), "geometry": "strut",    "hotspot_region": "mid"},
    AssetType.AIRCRAFT_BRAKE:         {"shape": (7, 7, 3), "geometry": "disc",     "hotspot_region": "outer_ring"},
    AssetType.AIRCRAFT_WING:          {"shape": (10, 4, 3), "geometry": "beam",    "hotspot_region": "root"},
    AssetType.AIRCRAFT_FUSELAGE:      {"shape": (8, 5, 5), "geometry": "shell",    "hotspot_region": "hoop_seam"},
    AssetType.TRAIN_BOGIE:            {"shape": (6, 6, 6), "geometry": "box",      "hotspot_region": "corner"},
    AssetType.TRAIN_BRAKE:            {"shape": (7, 7, 3), "geometry": "disc",     "hotspot_region": "outer_ring"},
    AssetType.TRAIN_WHEEL:            {"shape": (7, 7, 3), "geometry": "disc",     "hotspot_region": "rim"},
    AssetType.TRAIN_TRACTION_MOTOR:   {"shape": (5, 5, 8), "geometry": "cylinder", "hotspot_region": "mid"},
}


class SyntheticSimulationAdapter(SimulationLayer):
    """Cheap stand-in for a real ANSYS/Abaqus co-simulation."""

    name = "synthetic"

    def compute_stress_features(
        self,
        asset_id: str,
        asset_type: AssetType,
        component: str,
        material_name: str,
        sensor_batch: pd.DataFrame,
        anomaly_context: Optional[Dict[str, Any]] = None,
    ) -> StressFeatures:
        config = _pm.PART_CONFIG.get(asset_type, {})

        # 1. Base stress amplitude derived from vibration + load_factor,
        #    scaled toward the material's yield strength so numbers stay
        #    physically plausible for the downstream fatigue engine.
        vib = float(sensor_batch["vibration"].mean()) if "vibration" in sensor_batch else 0.5
        load = float(sensor_batch["load_factor"].mean()) if "load_factor" in sensor_batch else 0.5

        # Lightweight surrogate: 0.25 * yield_strength weighted by load,
        # bumped by observed vibration energy. Deliberately transparent
        # so reviewers can see it is NOT a real FEA result.
        from ..engines.material_engine import get_material_params

        params = get_material_params(material_name)
        yield_mpa = float(params.yield_strength) if params else 400.0
        base_amp = 0.25 * yield_mpa * max(0.2, load) * (1.0 + 0.4 * vib)

        # 2. Anomaly context boosts the amplitude — a flagged spike
        #    means the component is being loaded outside normal envelope.
        if anomaly_context:
            severity = anomaly_context.get("severity", "monitoring")
            boost = {
                "monitoring": 1.05,
                "warning": 1.20,
                "critical": 1.45,
                "failed": 1.70,
            }.get(severity, 1.0)
            base_amp *= boost

        stress_range_ratio = float(config.get("stress_range_ratio", 1.5))
        mean_stress_ratio = float(config.get("mean_stress_ratio", 0.25))
        geometry_factor = float(config.get("geometry_factor", 1.2))
        cycles_per_hour = float(config.get("cycles_per_hour", 3600))
        operating_hours_per_day = float(config.get("operating_hours_per_day", 16))

        stress_amp = float(base_amp)
        stress_range = stress_amp * stress_range_ratio
        mean_stress = stress_amp * mean_stress_ratio

        return StressFeatures(
            asset_id=asset_id,
            asset_type=asset_type.value,
            component=component,
            material_name=material_name,
            stress_amplitude_mpa=stress_amp,
            stress_range_mpa=stress_range,
            mean_stress_mpa=mean_stress,
            crack_size_mm=None,  # let fatigue engine use initial_crack_size
            geometry_factor=geometry_factor,
            R_ratio=0.0,
            cycles_per_hour=cycles_per_hour,
            operating_hours_per_day=operating_hours_per_day,
            source="synthetic_surrogate",
            is_synthetic=True,
            extras={
                "note": "SYNTHETIC — no real ANSYS/Abaqus/Creo/NASTRAN call was made.",
                "vibration_mean": vib,
                "load_factor_mean": load,
                "yield_strength_ref_mpa": yield_mpa,
                "stress_field_3d": _build_stress_field_3d(
                    asset_type=asset_type,
                    peak_stress_mpa=stress_amp,
                    yield_stress_mpa=yield_mpa,
                    anomaly_context=anomaly_context,
                ),
            },
        )


# ------------------------------------------------------------------
# 3D stress-intensity field synthesis
# ------------------------------------------------------------------
def _build_stress_field_3d(
    asset_type: AssetType,
    peak_stress_mpa: float,
    yield_stress_mpa: float,
    anomaly_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a small 3D mesh of stress values suitable for a Plotly scatter3d
    or volume plot. All numbers are synthetic — no real solver was run.
    """
    mesh = _MESH_BY_TYPE.get(
        asset_type,
        {"shape": (6, 6, 4), "geometry": "box", "hotspot_region": "center"},
    )
    nx, ny, nz = mesh["shape"]
    geometry = mesh["geometry"]
    hotspot_region = mesh["hotspot_region"]

    # 1. Build node coordinates for the geometry archetype.
    xs, ys, zs = _mesh_coordinates(geometry, nx, ny, nz)

    # 2. Distance-from-hotspot field (0 = at hotspot, 1 = farthest).
    hx, hy, hz = _hotspot_location(geometry, hotspot_region, xs, ys, zs)
    dist = np.sqrt((xs - hx) ** 2 + (ys - hy) ** 2 + (zs - hz) ** 2)
    dist = dist / (dist.max() + 1e-9)

    # 3. Stress falls off with distance from hotspot; hotspot ≈ peak_stress.
    #    Add mild noise so the field isn't perfectly symmetric.
    rng = np.random.default_rng(int(peak_stress_mpa * 1000) % (2 ** 32))
    falloff = np.exp(-2.5 * dist)
    noise = rng.normal(0.0, 0.04, size=xs.shape)
    stress = peak_stress_mpa * (falloff + noise)
    stress = np.clip(stress, a_min=0.0, a_max=1.2 * yield_stress_mpa)

    # 4. Boost hotspot further if the anomaly severity is high.
    if anomaly_context:
        severity = anomaly_context.get("severity", "monitoring")
        boost = {
            "monitoring": 1.00,
            "warning": 1.10,
            "critical": 1.25,
            "failed": 1.40,
        }.get(severity, 1.0)
        # Boost is concentrated near the hotspot.
        stress = stress * (1.0 + (boost - 1.0) * falloff)

    # 5. Flag hotspots. Two criteria (either one triggers):
    #      (a) absolute:  above 0.85 * yield (a real engineering concern)
    #      (b) relative:  in the top 10% of stress values in this field
    #    (b) guarantees the frontend always has something to highlight.
    abs_thresh = 0.85 * yield_stress_mpa
    rel_thresh = float(np.quantile(stress, 0.90))
    hotspot_threshold = min(abs_thresh, rel_thresh)
    is_hotspot = stress >= hotspot_threshold

    points: List[Dict[str, Any]] = []
    for i in range(xs.size):
        points.append(
            {
                "x": float(xs.flat[i]),
                "y": float(ys.flat[i]),
                "z": float(zs.flat[i]),
                "stress_mpa": float(stress.flat[i]),
                "is_hotspot": bool(is_hotspot.flat[i]),
            }
        )

    return {
        "geometry": geometry,
        "hotspot_region": hotspot_region,
        "mesh_shape": [int(nx), int(ny), int(nz)],
        "min_stress_mpa": float(stress.min()),
        "max_stress_mpa": float(stress.max()),
        "hotspot_threshold_mpa": float(hotspot_threshold),
        "hotspot_count": int(is_hotspot.sum()),
        "points": points,
        "is_synthetic": True,
        "note": "SYNTHETIC field — replace with real FEA nodal output in a real adapter.",
    }


def _mesh_coordinates(geometry: str, nx: int, ny: int, nz: int):
    if geometry == "cylinder":
        # Wrap x,y onto a cylinder of unit radius.
        theta = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
        r = np.linspace(0.6, 1.0, ny)
        z = np.linspace(0.0, 1.0, nz)
        T, R, Z = np.meshgrid(theta, r, z, indexing="ij")
        return R * np.cos(T), R * np.sin(T), Z
    if geometry == "disc":
        theta = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
        r = np.linspace(0.1, 1.0, ny)
        z = np.linspace(0.0, 0.2, nz)
        T, R, Z = np.meshgrid(theta, r, z, indexing="ij")
        return R * np.cos(T), R * np.sin(T), Z
    if geometry == "beam":
        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(-0.2, 0.2, ny)
        z = np.linspace(-0.05, 0.05, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return X, Y, Z
    if geometry == "strut":
        x = np.linspace(-0.3, 0.3, nx)
        y = np.linspace(-0.3, 0.3, ny)
        z = np.linspace(0.0, 1.0, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return X, Y, Z
    if geometry == "shell":
        # Cylindrical shell (thin).
        theta = np.linspace(0.0, np.pi, nx)
        y = np.linspace(0.0, 1.0, ny)
        z = np.linspace(0.98, 1.0, nz)
        T, Y, R = np.meshgrid(theta, y, z, indexing="ij")
        return R * np.cos(T), Y, R * np.sin(T)
    # box (default)
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    z = np.linspace(0.0, 1.0, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z


def _hotspot_location(geometry: str, region: str, xs, ys, zs):
    """Pick a plausible hotspot centroid for the geometry archetype."""
    if geometry in ("cylinder", "strut") and region == "root":
        return float(xs.flat[0]), float(ys.flat[0]), 0.0
    if geometry in ("cylinder", "strut") and region == "mid":
        return 0.0, 0.0, 0.5
    if geometry == "disc" and region in ("outer_ring", "rim"):
        # Point on outer ring at theta=0
        return 1.0, 0.0, float(zs.mean())
    if geometry == "beam" and region == "root":
        return 0.0, 0.0, 0.0
    if geometry == "shell" and region == "hoop_seam":
        return float(xs.mean()), 0.5, float(zs.mean())
    if geometry == "box" and region == "corner":
        return 1.0, 1.0, 1.0
    return float(xs.mean()), float(ys.mean()), float(zs.mean())
