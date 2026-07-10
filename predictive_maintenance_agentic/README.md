# Predictive Maintenance — Engineering Intelligence

A LangGraph + LangChain multi-agent orchestration layer that wraps the
existing physics + ML predictive-maintenance code in this repo
([anamoly-detection.py](../anamoly-detection.py) and
[prediction-managent.py](../prediction-managent.py)) into the flow
shown in `FINAL-DIAGRAM.png`.

**Nothing about the fatigue physics or the RUL ML has been rewritten.**
Every engine in `engines/` is a thin wrapper around the classes already
in those two files.

**The simulation layer is SYNTHETIC.** No ANSYS / Abaqus / Creo /
NASTRAN calls are made. See the "Plugging in real simulation software"
section below for the swap point.

---

## Quick start

```bash
# From the repo root (the folder containing anamoly-detection.py)
pip install -U langgraph
pip install -r predictive_maintenance_agentic/requirements.txt

# Sanity-check the synthetic data path
python -m predictive_maintenance_agentic.data.synthetic_generator

# Run the tests
pytest -q predictive_maintenance_agentic/tests

# Boot the API (uses SQLite checkpointer at ./eix_checkpoints.sqlite)
uvicorn predictive_maintenance_agentic.api.main:app --reload
```

By default the LLM provider is `none`, so the agents run their
deterministic rule-based rationale paths (fast, offline, hackathon-safe).
To turn LLM narration on:

```bash
export EIX_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-...
# or
export EIX_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

---

## Exercising the graph end-to-end

Once the API is up:

```bash
# 1. Kick off an anomalous cycle
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
        "asset_id": "ENGINE-001",
        "asset_type": "aircraft_engine",
        "material_name": "Inconel718",
        "component": "turbine_blade",
        "generate_synthetic": true,
        "synthetic_n_samples": 100,
        "force_anomaly": true
      }' | jq .

# 2. If the response has "awaiting_engineer_approval": true, resume:
curl -s -X POST http://localhost:8000/engineer/approve \
  -H "Content-Type: application/json" \
  -d '{
        "cycle_id": "<the cycle_id from step 1>",
        "approved": true,
        "engineer_id": "eng-42",
        "notes": "Calibration looks reasonable, proceeding."
      }' | jq .

# 3. Inspect the full node-by-node trace:
curl -s http://localhost:8000/cycles/<cycle_id> | jq .trace

# 4. Fleet health:
curl -s http://localhost:8000/fleet/status | jq .
```

To trigger the anomaly-gate short-circuit path:

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
        "asset_id": "ENGINE-001",
        "asset_type": "aircraft_engine",
        "material_name": "Inconel718",
        "generate_synthetic": true,
        "force_normal": true
      }' | jq '.status, .physics_prediction'
# → "normal_end", null
```

---

## How the LangGraph nodes map to the diagram

Diagram box (from `FINAL-DIAGRAM.png`)
LangGraph node
Source file

Orchestrator Agent (cycle start, fleet-memory load)
`init`
[agents/orchestrator.py](agents/orchestrator.py) → `initialize_cycle_node`

Data Ingestion Layer + Anomaly Detection gate
`ingest`
[agents/orchestrator.py](agents/orchestrator.py) → `ingest_data_node` (uses [engines/anomaly_engine.py](engines/anomaly_engine.py))

Digital Twin / Simulation Layer
`simulation`
[simulation/synthetic_adapter.py](simulation/synthetic_adapter.py) via [agents/physics_agent.py](agents/physics_agent.py#L18) → `simulation_layer_node`

Physics Agent
`physics`
[agents/physics_agent.py](agents/physics_agent.py) → `physics_agent_node` (wraps `FatigueAnalyzer` from [anamoly-detection.py](../anamoly-detection.py))

ML Correction Agent
`ml`
[agents/ml_correction_agent.py](agents/ml_correction_agent.py) → `ml_correction_agent_node` (wraps `RULEstimator` + `GaussianProcessRegressor`)

Judge Agent (confidence, HCF/LCF, divergence, root cause)
`judge`
[agents/judge_agent.py](agents/judge_agent.py) → `judge_agent_node`

Calibration Agent (Bayesian parameter optimization)
`calibration`
[agents/calibration_agent.py](agents/calibration_agent.py) → `calibration_agent_node`

Engineer Approval (human-in-the-loop)
`engineer_approval` (interrupt_before)
[agents/engineer_approval.py](agents/engineer_approval.py) → `engineer_approval_node`

Action Agent
`action`
[agents/action_agent.py](agents/action_agent.py) → `action_agent_node`

Fleet Memory ("Agent Communication" panel)
Read in `init`, written in `ingest` / `engineer_approval` / `action`
[memory/fleet_memory.py](memory/fleet_memory.py) → `FleetMemoryStore`

MRO / ERP (SAP PM, Maximo, AMOS, Scheduling, Logbook)
Called from `action`, stub only
[integrations/sap_pm_adapter.py](integrations/sap_pm_adapter.py), [integrations/maximo_adapter.py](integrations/maximo_adapter.py)

Dashboard (React + Plotly panel)
Data exposed via API; UI intentionally not built
[api/main.py](api/main.py) endpoints `/analyze`, `/fleet/status`, `/cycles/{id}`, `/engineer/approve`

### Behavioral rules enforced by the graph

Rule
Where

Anomaly gate hard short-circuit
`_route_after_ingest` in [graph/build_graph.py](graph/build_graph.py)

Physics ⇄ ML negotiation capped at 2 rounds
`_route_after_ml` + `LIMITS.max_negotiation_rounds` in [config.py](config.py)

Judge → Calibration → Engineer → re-simulate capped at 5 rounds; force "unresolved_divergence" on cap
`_route_after_judge`, `_route_after_engineer` in [graph/build_graph.py](graph/build_graph.py); flag applied in [agents/action_agent.py](agents/action_agent.py)

Engineer approval is a real interrupt, not a mock
`interrupt_before=["engineer_approval"]` in [graph/build_graph.py](graph/build_graph.py); resume via `/engineer/approve`

Fleet memory read at cycle start, written at cycle end
`initialize_cycle_node` reads; `ingest_data_node`, `engineer_approval_node`, `action_agent_node` write

Material recommendations only surface when RUL is low
`THRESHOLDS.low_rul_hours` gate in [agents/action_agent.py](agents/action_agent.py)

---

## Reconciling the two legacy files

Both `anamoly-detection.py` and `prediction-managent.py` defined
overlapping enums / dataclasses. Per the build prompt, we keep the
**union** — [models.py](models.py) re-exports the superset from
`anamoly-detection.py` (which already carries every field
`prediction-managent.py` had, plus supplier / material-change extensions
and the `RecommendationType` enum). No fields were dropped.

Canonical mapping:

Concept
Canonical source
Wrapper

Anomaly detection (base + material-aware)
`AnomalyDetector`, `EnhancedAnomalyDetector` (anamoly-detection.py)
[engines/anomaly_engine.py](engines/anomaly_engine.py)

Fatigue physics (Basquin + Paris + NASGRO + lifecycle)
`FatigueAnalyzer` (anamoly-detection.py)
[engines/physics_engine.py](engines/physics_engine.py)

RUL estimation + GP residual correction
`RULEstimator` (anamoly-detection.py) + sklearn GP
[engines/rul_engine.py](engines/rul_engine.py)

Material + supplier database + recommendations
`EnhancedMaterialDatabase` (anamoly-detection.py)
[engines/material_engine.py](engines/material_engine.py)

Per-component RUL / part-name resolution + `PART_CONFIG`
`PartLifecyclePredictor`, `resolve_asset_type`, `PART_CONFIG` (prediction-managent.py)
Used by [simulation/synthetic_adapter.py](simulation/synthetic_adapter.py)

CSV ingestion
`load_sensor_csv`, `load_sensor_data_from_directory` (prediction-managent.py)
[data/loaders.py](data/loaders.py)

Synthetic sensor generator
`generate_sensor_data` (both files, identical)
[data/synthetic_generator.py](data/synthetic_generator.py)

The legacy files stay put — we import them at runtime via
[_legacy_imports.py](_legacy_imports.py) (they use hyphens in their
filenames so this needs `importlib.util`).

---

## Plugging in real ANSYS / Abaqus / Creo / NASTRAN

Everything simulator-specific is behind
[simulation/base.py](simulation/base.py) → `SimulationLayer`. To swap in
a real solver:

1. Implement a subclass:

   ```python
   from predictive_maintenance_agentic.simulation.base import (
       SimulationLayer, StressFeatures,
   )

   class AnsysAdapter(SimulationLayer):
       name = "ansys"
       def compute_stress_features(self, asset_id, asset_type, component,
                                   material_name, sensor_batch,
                                   anomaly_context=None):
           # Call the real ANSYS APDL / pyMAPDL job here.
           # Return a StressFeatures with is_synthetic=False and
           # source="ansys_apdl_<version>".
           ...
   ```

2. Register it before building the graph:

   ```python
   from predictive_maintenance_agentic.agents._shared import set_simulation_layer
   set_simulation_layer(AnsysAdapter())
   ```

No agent code, no graph wiring, and no API contract change is required —
the adapter's `StressFeatures.is_synthetic=False` will automatically
show up in every `/cycles/{id}` response and every MRO work-order
payload.

---

## Project layout

```
predictive_maintenance_agentic/
├── models.py                      # unified dataclasses/enums (superset)
├── config.py                      # LLM provider, thresholds, iteration limits
├── _legacy_imports.py             # importlib shim for the hyphenated files
├── data/
│   ├── synthetic_generator.py     # wraps generate_sensor_data + inject_anomaly
│   └── loaders.py                 # wraps CSV loaders
├── simulation/
│   ├── base.py                    # SimulationLayer interface + StressFeatures
│   └── synthetic_adapter.py       # SYNTHETIC adapter (only implementation today)
├── engines/
│   ├── anomaly_engine.py          # wraps EnhancedAnomalyDetector
│   ├── physics_engine.py          # wraps FatigueAnalyzer (Basquin+Paris+NASGRO)
│   ├── rul_engine.py              # wraps RULEstimator + GP residual
│   └── material_engine.py         # wraps EnhancedMaterialDatabase
├── memory/
│   └── fleet_memory.py            # SQLite-backed FleetMemoryStore
├── agents/
│   ├── orchestrator.py            # init + ingest/anomaly gate
│   ├── physics_agent.py           # simulation_layer_node + physics_agent_node
│   ├── ml_correction_agent.py     # RF+GP correction + optional LLM rationale
│   ├── judge_agent.py             # confidence / divergence / root cause
│   ├── calibration_agent.py       # Bayesian + scipy.optimize
│   ├── engineer_approval.py       # interrupt + resume payload builder
│   └── action_agent.py            # decision + MRO stub + material recs
├── graph/
│   ├── state.py                   # CycleState TypedDict
│   └── build_graph.py             # StateGraph, conditional edges, caps
├── integrations/
│   ├── sap_pm_adapter.py          # stub
│   └── maximo_adapter.py          # stub
├── api/
│   └── main.py                    # FastAPI app (/analyze, /fleet/status, /cycles/*, /engineer/approve)
└── tests/
    ├── test_engines.py            # per-engine wrapper smoke tests
    └── test_graph_end_to_end.py   # full graph incl. interrupt/resume
```
