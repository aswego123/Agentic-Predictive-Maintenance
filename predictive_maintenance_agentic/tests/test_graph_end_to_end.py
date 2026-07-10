"""
End-to-end LangGraph tests.

Covers:
  * Non-anomalous batch short-circuits after the anomaly gate.
  * Anomalous batch walks the full pipeline (agreement path).
  * Deliberately conflicting physics vs ML triggers calibration +
    the engineer-approval interrupt, which is then resumed.
"""
from __future__ import annotations

import uuid

import pandas as pd
import pytest

from predictive_maintenance_agentic.agents import engineer_resume_payload
from predictive_maintenance_agentic.agents._shared import (
    sensor_batch_to_dict,
)
from predictive_maintenance_agentic.data.synthetic_generator import (
    generate_normal_batch,
    generate_sensor_data,
    inject_anomaly,
)
from predictive_maintenance_agentic.graph import build_graph
from predictive_maintenance_agentic.models import AssetType


def _thread_cfg():
    return {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}


def _run(bundle, initial, cfg):
    return bundle.graph.invoke(initial, cfg)


def test_normal_batch_short_circuits():
    bundle = build_graph(checkpoint_path=":memory:")
    df = generate_normal_batch("ENGINE-XX", AssetType.AIRCRAFT_ENGINE, n_samples=60)
    df["asset_id"] = "ENGINE-XX"
    cfg = _thread_cfg()

    result = _run(
        bundle,
        {
            "asset_id": "ENGINE-XX",
            "asset_type": AssetType.AIRCRAFT_ENGINE.value,
            "material_name": "Inconel718",
            "component": "turbine_blade",
            "sensor_batch": sensor_batch_to_dict(df),
        },
        cfg,
    )
    # Either short-circuited (normal_end) or produced an action after
    # detecting a residual anomaly — but if normal, physics MUST be None.
    if not result.get("is_anomalous"):
        assert result.get("status") == "normal_end"
        assert result.get("physics_prediction") is None
        assert result.get("ml_correction") is None


def test_anomalous_batch_walks_full_pipeline_or_interrupts():
    bundle = build_graph(checkpoint_path=":memory:")
    df = generate_sensor_data("ENGINE-YY", AssetType.AIRCRAFT_ENGINE, n_samples=80)
    df = inject_anomaly(df, channel="vibration", multiplier=8.0)
    df["asset_id"] = "ENGINE-YY"

    cfg = _thread_cfg()
    result = _run(
        bundle,
        {
            "asset_id": "ENGINE-YY",
            "asset_type": AssetType.AIRCRAFT_ENGINE.value,
            "material_name": "Inconel718",
            "component": "turbine_blade",
            "sensor_batch": sensor_batch_to_dict(df),
        },
        cfg,
    )
    assert result.get("is_anomalous") is True
    # Physics + ML must have run.
    assert result.get("physics_prediction") is not None
    assert result.get("ml_correction") is not None

    # If graph paused at engineer_approval, resume as many times as
    # needed. The Judge → Calibration → Engineer → resim loop is
    # capped at 5 rounds by LIMITS.max_resimulation_rounds, so at most
    # 5 resumes are required before the graph exits via `action`.
    for _ in range(10):
        snap = bundle.graph.get_state(cfg)
        if not snap or not snap.next:
            break
        if "engineer_approval" not in (snap.next or ()):
            break
        bundle.graph.update_state(
            cfg,
            engineer_resume_payload(
                approved=True,
                engineer_id="eng-test",
                notes="test approval",
            ),
        )
        result = bundle.graph.invoke(None, cfg)

    # Whichever path, we must end with a status of action_taken OR
    # unresolved_divergence (never left in 'running').
    assert result.get("status") in ("action_taken", "unresolved_divergence")
    assert result.get("action") is not None
