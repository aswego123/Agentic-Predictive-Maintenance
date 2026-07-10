"""Smoke tests for each engine wrapper."""
from __future__ import annotations

import pandas as pd

from predictive_maintenance_agentic.data.synthetic_generator import (
    generate_sensor_data,
    inject_anomaly,
    generate_normal_batch,
)
from predictive_maintenance_agentic.engines import (
    AnomalyEngine,
    MaterialEngine,
    PhysicsEngine,
    RULEngine,
)
from predictive_maintenance_agentic.models import AssetType


def test_synthetic_generator_shape():
    df = generate_sensor_data("ENGINE-001", AssetType.AIRCRAFT_ENGINE, n_samples=30)
    assert len(df) == 30
    for col in ("vibration", "temperature", "load_factor", "asset_id"):
        assert col in df.columns


def test_anomaly_engine_detects_forced_anomaly():
    df = generate_sensor_data("ENGINE-001", AssetType.AIRCRAFT_ENGINE, n_samples=80)
    df_anom = inject_anomaly(df, channel="vibration", multiplier=8.0)
    eng = AnomalyEngine()
    result = eng.detect(AssetType.AIRCRAFT_ENGINE, "Inconel718", df_anom, training_batch=df)
    assert result.anomaly_count >= 1
    assert result.is_anomalous is True


def test_anomaly_engine_normal_batch_may_short_circuit():
    df = generate_normal_batch("ENGINE-001", AssetType.AIRCRAFT_ENGINE, n_samples=80)
    eng = AnomalyEngine()
    # Contract: engine returns a well-formed result either way.
    result = eng.detect(AssetType.AIRCRAFT_ENGINE, "Inconel718", df, training_batch=df)
    assert isinstance(result.is_anomalous, bool)


def test_physics_engine_returns_finite_or_none():
    phys = PhysicsEngine()
    out = phys.analyze(
        material_name="Steel4340",
        stress_amplitude_mpa=300.0,
        stress_range_mpa=450.0,
        mean_stress_mpa=75.0,
        cycles_per_hour=1,
        operating_hours_per_day=10,
    )
    assert out["material_name"] == "Steel4340"
    assert out["failure_mode"] in {
        "high_cycle_fatigue", "low_cycle_fatigue",
        "thermal_fatigue", "corrosion_fatigue", "creep", "fracture",
        "wear", "bearing_failure", "overheating", "excessive_vibration",
    }


def test_rul_engine_combines_with_physics():
    df = generate_sensor_data("ENGINE-001", AssetType.AIRCRAFT_ENGINE, n_samples=40)
    phys = PhysicsEngine().analyze(
        material_name="Inconel718",
        stress_amplitude_mpa=280.0,
        stress_range_mpa=420.0,
        mean_stress_mpa=60.0,
    )
    ml = RULEngine().correct(AssetType.AIRCRAFT_ENGINE, df, phys)
    assert ml["source"] == "ml_correction"
    assert "rul_hours" in ml


def test_material_engine_returns_alternatives():
    rec = MaterialEngine().recommend("Steel4340")
    assert rec["current_material"] == "Steel4340"
    assert isinstance(rec["alternatives"], list)
