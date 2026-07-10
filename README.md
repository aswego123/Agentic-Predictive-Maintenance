# EIx — Engineering Intelligence Predictive Maintenance

## Use Case: Anomaly Detection & Fatigue Intelligence for Aerospace and Rail Assets
LangGraph multi-agent digital twin where physics and ML negotiate,
a Gemini Judge investigate with multiple tools, and every decision leaves an audit trail.

**An agentic predictive-maintenance digital twin for industrial assets.**

https://github.com/user-attachments/assets/83a36a5a-27f7-4859-a6ac-a1cee1121a4a

## Problem Statement: Aircraft and rail operators still rely on fixed inspection intervals and manual thresholds. 
By the time a gauge trips, damage has often already accumulated - turning a routine swap into an unscheduled grounding.
Reactive, not predictive- Threshold alarms fire only after a fault is already underway.
Unplanned downtime - Emergency repairs cost 3-5x more than scheduled maintenance.
Fragmented data - Sensor CSVs, material specs, and fatigue models live in silos.

## Pipeline: CSV Input → Anomaly Detection → Physics-Based Fatigue Prediction → ML Correction → LLM Judge (Tool-Use) → Material-Swap Recommendation → Engineer Approval → Work Order

Raw sensor and inspection data enters as a CSV feed and is first screened for 
anomalies to flag components showing deviation from expected behavior. Flagged 
components are run through a physics-based fatigue prediction model to generate 
a first-principles degradation estimate, which an ML correction layer then refines 
against real historical outcomes — accounting for manufacturing variance, 
environmental effects, and sensor drift that physics alone misses. The corrected 
prediction is passed to an LLM Judge with tool-use access, which cross-examines 
both outputs, pulls in relevant context (component history, maintenance records, 
material specs), and surfaces a material-swap recommendation with full reasoning. 
Nothing is auto-executed — every recommendation is routed to an engineer for 
approval before being converted into a work order, closing the loop from raw 
sensor data to actionable maintenance decision with a fully explainable trail.

Built for the Hackathon around a LangGraph state machine (Python / FastAPI backend)
and a React + Vite frontend that visualizes the whole pipeline as it runs.

---

## Table of contents

- [What this system does](#what-this-system-does)
- [Architecture at a glance](#architecture-at-a-glance)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Quick start (backend + frontend)](#quick-start-backend--frontend)
- [Environment configuration](#environment-configuration)
- [Backend HTTP API](#backend-http-api)
- [Data model — the CSV your sensors upload](#data-model--the-csv-your-sensors-upload)
- [Pipeline stages, one by one](#pipeline-stages-one-by-one)
- [Frontend UI — what each tab shows](#frontend-ui--what-each-tab-shows)
- [Persistence & memory](#persistence--memory)
- [Simulation adapter (synthetic vs. real solver)](#simulation-adapter-synthetic-vs-real-solver)
- [Development workflow](#development-workflow)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Design decisions worth reading](#design-decisions-worth-reading)

---

## What this system does

You upload a batch of raw sensor readings for one industrial component (an
aircraft wing, a train bogie, a jet engine, etc.). The system:

1. Detects anomalies in the readings (IsolationForest, trained on a clean
   synthetic baseline for that asset type).
2. Derives a *stress amplitude* for the component from the sensor batch
   plus material properties (a lightweight surrogate — not real FEA).
3. Runs a **Basquin + Paris + NASGRO** fatigue-life analysis
   (`FatigueAnalyzer` from the legacy `prediction-managent.py`) to get
   total design cycles, current RUL, and predicted failure date.
4. Runs a small ML "correction" model that adjusts the physics numbers
   given the sensor context, and may negotiate with the physics agent
   for another round.
5. Sends the raw CSV + intermediate results to a **Gemini-based Judge Agent**
   that decides in a ReAct loop (up to 6 tool calls) whether the two models
   agree, whether maintenance is needed, and what route to take
   (`action_fast_path` / `calibration` / `action_monitoring`).
6. If the Judge routes to calibration, it pauses at an **engineer-approval
   interrupt** — a human in the UI decides whether to accept the recomputed
   parameters and re-simulate.
7. Emits a work order (SAP-PM stub) with a material-swap recommendation
   pulled from a hardcoded supplier catalogue.
8. Persists everything to two SQLite files so the fleet-history / Cycles
   page survives restarts.

The whole pipeline is a LangGraph `StateGraph` with a SQLite checkpointer.
The frontend polls `GET /cycles/{cycle_id}` every 750 ms and animates each
node lighting up as it completes.

---

## Architecture at a glance

```
             ┌──────────────────────────────────────────────────────────────────┐
             │  React + Vite (pmx-frontend)                                     │
             │  - New Analysis form  (CSV upload / manual entry / synthetic)    │
             │  - Cycles list        (asset + status)                           │
             │  - Cycle detail       (7 tabs — see UI section)                  │
             └───────────────┬──────────────────────────────────────────────────┘
                             │ HTTP + 750 ms polling
                             ▼
             ┌──────────────────────────────────────────────────────────────────┐
             │  FastAPI (predictive_maintenance_agentic.api.main)               │
             │  POST /analyze  · GET /cycles/{id}  · GET /fleet/status ·        │
             │  POST /engineer/approve                                          │
             └───────────────┬──────────────────────────────────────────────────┘
                             │ graph.invoke(state, thread_id=cycle_id)
                             ▼
             ┌──────────────────────────────────────────────────────────────────┐
             │  LangGraph StateGraph  (predictive_maintenance_agentic.graph)    │
             │                                                                  │
             │  START → init → ingest ─╮                                        │
             │                          ├─(no anomaly)→ END                     │
             │                          ▼                                       │
             │              simulation → physics ⇄ ml  (max 2 negotiation)      │
             │                                  │                               │
             │                             data_fetch (optional)                │
             │                                  ▼                               │
             │                              judge ─╮                            │
             │                                     ├→ calibration → 🛑engineer  │
             │                                     ▼         │                  │
             │                                   action ◀────╯ (approve → resim)│
             │                                     ▼                            │
             │                                   critic → END                   │
             │                                                                  │
             │  Checkpointer: SqliteSaver (eix_checkpoints.sqlite)              │
             │  Fleet memory: SQLite     (eix_fleet_memory.sqlite)              │
             └───────────────┬──────────────────────────────────────────────────┘
                             │
                             ▼
             ┌──────────────────────────────────────────────────────────────────┐
             │  Engines                                                         │
             │  · AnomalyEngine    (IsolationForest, clean baseline)            │
             │  · PhysicsEngine    (Basquin + Paris + NASGRO)                   │
             │  · MaterialEngine   (yield / endurance / UTS lookup)             │
             │  · SimulationLayer  (SyntheticSimulationAdapter — surrogate)     │
             │  · LLM              (Gemini pro-latest via langchain-google)     │
             └──────────────────────────────────────────────────────────────────┘
```

---

## Repository layout

```
Hackathon/
├── anamoly-detection.py               # Legacy: EnhancedAnomalyDetector + material recs
├── prediction-managent.py             # Legacy: FatigueAnalyzer + PART_CONFIG
├── predictive_maintenance_agentic/    # NEW agentic backend (Python)
│   ├── api/main.py                    #   FastAPI entrypoint
│   ├── graph/build_graph.py           #   StateGraph + edges + interrupt
│   ├── agents/                        #   8 LangGraph nodes
│   │   ├── orchestrator.py            #     ingest + anomaly gate
│   │   ├── physics_agent.py           #     simulation + physics
│   │   ├── ml_correction_agent.py     #     ML correction + negotiation
│   │   ├── data_fetch_agent.py        #     enrichment
│   │   ├── judge_agent.py             #     LLM ReAct Judge (Gemini)
│   │   ├── calibration_agent.py       #     recompute parameters
│   │   ├── engineer_approval.py       #     human-in-the-loop interrupt
│   │   ├── action_agent.py            #     work order + material rec
│   │   └── critic_agent.py            #     post-cycle reflection
│   ├── engines/                       #   Pure-function engines
│   │   ├── anomaly_engine.py          #     IsolationForest wrapper
│   │   ├── physics_engine.py          #     Basquin+Paris+NASGRO wrapper
│   │   └── material_engine.py         #     Material property lookup
│   ├── simulation/                    #   Synthetic FEA stand-in
│   ├── tools/judge_tools.py           #   5 tools for the Judge ReAct loop
│   ├── memory/fleet_memory.py         #   SQLite audit trail per asset
│   ├── data/synthetic_generator.py    #   Wraps legacy generate_sensor_data
│   ├── models.py                      #   Pydantic models + enums
│   ├── config.py                      #   Env-var configuration
│   ├── requirements.txt
│   └── tests/                         #   pytest suite
├── pmx-frontend/                      # NEW React UI (TS + Vite)
│   ├── src/
│   │   ├── views/                     #   Route views
│   │   ├── components/cycle/          #   Panels: verdict, pipeline, physics/ML …
│   │   ├── lib/lifecycle.ts           #   Client-side maintenance scheduler
│   │   ├── hooks/                     #   useCycle, useAnalyze, useFleetStatus
│   │   └── store/                     #   Zustand slices
│   └── package.json
├── data-points/                       # Sample CSV inputs (WING-001, BOGIE-001)
├── architecture-diagrams/             # HTML + PNG of the pipeline
├── documentation/                     # Hackathon brief
├── eix_checkpoints.sqlite             # LangGraph state (created at runtime)
├── eix_fleet_memory.sqlite            # Per-asset audit trail (runtime)
├── .env.example                       # Copy → .env and fill in secrets
└── README.md                          # ← you are here
```

---

## Prerequisites

| Tool         | Version    | Notes |
|--------------|-----------|-------|
| Python       | 3.13      | Miniconda / venv either works |
| Node.js      | 20.12+    | Frontend uses Vite + React 18 |
| npm          | 10+       | Ships with Node |
| SQLite       | (built-in)| Used by LangGraph checkpointer + fleet memory |
| Gemini API key | —      | Only if you want the real Judge LLM; otherwise the deterministic fallback runs |

Optional but recommended: a virtualenv (`python -m venv .venv && source .venv/bin/activate`).

---

## Quick start (backend + frontend)

### 1. Install dependencies

```bash
# Python backend
cd /path/to/Hackathon
pip install -r predictive_maintenance_agentic/requirements.txt
pip install langchain-google-genai   # only if using Gemini as Judge

# Frontend
cd pmx-frontend
npm install
cd ..
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and set EIX_LLM_PROVIDER=google + EIX_GOOGLE_API_KEY=<your-key>
# OR set EIX_LLM_PROVIDER=none to run purely on the deterministic fallback.
```

### 3. Run the backend (port 8000)

```bash
python -m uvicorn predictive_maintenance_agentic.api.main:app \
       --host 127.0.0.1 --port 8000
```

Expected startup output:

```
INFO:     Started server process [pid]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Verify it's healthy:

```bash
curl http://127.0.0.1:8000/health
# → {"status":"ok","simulation_adapter":"synthetic"}
```

### 4. Run the frontend (port 5173)

```bash
cd pmx-frontend
npm run dev
```

Vite opens `http://127.0.0.1:5173`. It proxies API calls to the backend on 8000.

### 5. Run an analysis

- Open `http://127.0.0.1:5173/analyze/new`
- Upload one of the sample CSVs from `data-points/` (e.g. `WING-001.csv`)
- Click **Analyze** — you're redirected to `/cycles/cycle-xxx` immediately
- Watch the pipeline graph light up node by node (~60–150 s end-to-end with Gemini)

---

## Environment configuration

All backend env vars are prefixed **`EIX_`** and read by
[`config.py`](predictive_maintenance_agentic/config.py). See `.env.example`
for the complete list. Key ones:

| Variable                       | Default                       | Purpose |
|--------------------------------|-------------------------------|---------|
| `EIX_LLM_PROVIDER`             | `none`                        | `none` \| `google` \| `openai` \| `anthropic` \| `azure` \| `azure_apim` |
| `EIX_LLM_MODEL`                | `gemini-pro-latest`           | Model name for the chosen provider |
| `EIX_LLM_TEMPERATURE`          | `0.1`                         | Kept low so Judge JSON stays parseable |
| `EIX_GOOGLE_API_KEY`           | *(required for google)*       | Gemini API key |
| `EIX_CHECKPOINT_PATH`          | `./eix_checkpoints.sqlite`    | LangGraph SqliteSaver location |
| `EIX_FLEET_MEMORY_PATH`        | `./eix_fleet_memory.sqlite`   | Per-asset audit trail |
| `EIX_SIMULATION_ADAPTER`       | `synthetic`                   | Only synthetic ships today |

If `EIX_LLM_PROVIDER=none`, the Judge falls back to a deterministic
Python function that computes `divergence` / `route` from Physics vs
ML deltas — the graph stays runnable without any LLM.

Frontend env vars live in `pmx-frontend/.env`:

```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## Backend HTTP API

Interactive OpenAPI docs are served at
[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) when the
backend is running.

| Method | Path                       | Purpose |
|--------|----------------------------|---------|
| `GET`  | `/health`                  | Liveness probe |
| `GET`  | `/fleet/status`            | List of `known_assets` + `known_cycles` (rehydrated from SQLite on startup) |
| `POST` | `/analyze`                 | Kick off a new cycle (accepts optional client-supplied `cycle_id`) |
| `GET`  | `/cycles/{cycle_id}`       | Full snapshot of a cycle's state (used by 750 ms poll) |
| `POST` | `/engineer/approve`        | Resume a cycle paused at the calibration interrupt |

### Example: start a cycle from the terminal

```bash
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "WING-001",
    "asset_type": "aircraft_wing",
    "material_name": "Al7075-T6",
    "cycle_id": "cycle-demo-01",
    "generate_synthetic": true,
    "synthetic_n_samples": 200
  }' | python -m json.tool
```

### Example: poll a cycle

```bash
curl -s http://127.0.0.1:8000/cycles/cycle-demo-01 | python -m json.tool
```

Response includes: `status`, `is_anomalous`, `anomaly_result`,
`stress_features`, `physics_prediction`, `ml_correction`,
`judge_verdict`, `calibration_result`, `engineer_decision`, `action`,
`critic_review`, `trace` (a stream of `{node, note, data, ts}`),
`awaiting_engineer_approval`.

### Example: approve a paused cycle

```bash
curl -s -X POST http://127.0.0.1:8000/engineer/approve \
  -H "Content-Type: application/json" \
  -d '{
    "cycle_id": "cycle-demo-01",
    "approved": true,
    "engineer_id": "eng-01",
    "notes": "Recalibrated parameters look reasonable"
  }'
```

---

## Data model — the CSV your sensors upload

Header row (exactly this order — see [sensor_template.csv](sensor_template.csv) if present):

```
asset_id,asset_type,timestamp,operational_cycles,vibration,temperature,pressure,load_factor,speed,acoustic_emission,oil_pressure,oil_temperature
```

| Column               | Type     | What it means |
|----------------------|----------|--------------|
| `asset_id`           | string   | Fleet-unique tag (`WING-001`) |
| `asset_type`         | enum     | `aircraft_engine` \| `aircraft_wing` \| `train_bogie` \| … |
| `timestamp`          | ISO 8601 | Row timestamp |
| `operational_cycles` | int      | Cumulative fatigue-cycle counter — drives RUL |
| `vibration`          | float    | g (peak) |
| `temperature`        | float    | °C |
| `pressure`           | float    | bar |
| `load_factor`        | float    | 0.0 – 1.0 (fraction of rated load) |
| `speed`              | float    | RPM |
| `acoustic_emission`  | float    | dB |
| `oil_pressure`       | float    | bar |
| `oil_temperature`    | float    | °C |

The optional column `material_name` (e.g. `Al7075-T6`) can be added — the
adapter reads the first non-empty value. If omitted the backend picks a
sensible default per `asset_type`.

Sample files: [data-points/WING-001.csv](data-points/WING-001.csv),
[data-points/BOGIE-001.csv](data-points/BOGIE-001.csv).

---

## Pipeline stages, one by one

| Node             | File                                                          | What it does |
|------------------|---------------------------------------------------------------|--------------|
| `init`           | [orchestrator.py](predictive_maintenance_agentic/agents/orchestrator.py) | Timestamps the cycle, initialises trace |
| `ingest`         | orchestrator.py                                               | Runs `AnomalyEngine`. If no anomaly → short-circuit to `END`. |
| `simulation`     | [physics_agent.py](predictive_maintenance_agentic/agents/physics_agent.py) | `SyntheticSimulationAdapter.compute_stress_features` — derives stress amplitude from vibration + load_factor + material yield |
| `physics`        | physics_agent.py                                              | `PhysicsEngine.analyze` — Basquin + Paris + NASGRO → RUL, health score, predicted failure date |
| `ml`             | [ml_correction_agent.py](predictive_maintenance_agentic/agents/ml_correction_agent.py) | Adjusts physics numbers; may `request_data` back to physics for one negotiation round |
| `data_fetch`     | [data_fetch_agent.py](predictive_maintenance_agentic/agents/data_fetch_agent.py) | Enriches state with extra sensor context |
| `judge`          | [judge_agent.py](predictive_maintenance_agentic/agents/judge_agent.py) | LLM ReAct loop (6 tool budget) → strict JSON verdict {`divergence`, `route`, `confidence`, `root_cause`} |
| `calibration`    | [calibration_agent.py](predictive_maintenance_agentic/agents/calibration_agent.py) | If Judge requested calibration, recomputes stress-feature parameters |
| `engineer_approval` | [engineer_approval.py](predictive_maintenance_agentic/agents/engineer_approval.py) | **`interrupt_before=["engineer_approval"]`** — graph pauses here for human review |
| `action`         | [action_agent.py](predictive_maintenance_agentic/agents/action_agent.py) | Creates work order (SAP-PM stub) + material-swap recommendation |
| `critic`         | [critic_agent.py](predictive_maintenance_agentic/agents/critic_agent.py) | Writes retrospective `critic_weights` to fleet memory |

Routing edges: see [`build_graph.py`](predictive_maintenance_agentic/graph/build_graph.py).

### Judge Agent tools

The Judge (a Gemini `pro-latest` agent) can invoke these tools during
its ReAct loop — up to 6 per verdict:

| Tool                       | Purpose |
|----------------------------|---------|
| `get_fleet_memory_stats`   | Past cycle counts, divergence rate, engineer approval rate for this asset |
| `find_similar_failures`    | Neighbours within a stress-MPa tolerance across the fleet |
| `query_material_limits`    | Endurance limit, yield, UTS for a material |
| `simulate_what_if`         | Re-run physics with a Δstress / Δcrack — returns new RUL |
| `get_recent_maintenance`   | Recent fleet-wide maintenance actions |

Full list in [`tools/judge_tools.py`](predictive_maintenance_agentic/tools/judge_tools.py).

---

## Frontend UI — what each tab shows

Cycle-detail page (`/cycles/{id}`) has 7 tabs. All tabs live in
[`views/CycleDetailView.tsx`](pmx-frontend/src/views/CycleDetailView.tsx).

| Tab                | Panels |
|--------------------|--------|
| **Overview**       | Interactive React Flow pipeline graph (nodes light up as each agent completes) + Lifecycle Prediction & maintenance schedule |
| **Detection**      | AnomalyPanel — CLI-style anomaly list with severity dots, sensor names, remediation hints |
| **Stress Intensity** | 3D scatter of the parametric mesh with per-node stress values (Turbo colourscale, diamond markers for hotspots) |
| **Physics & ML**   | Side-by-side Physics-vs-ML numeric panel with dialogue rationale |
| **Reasoning**      | Judge verdict card + list of tool calls made + full dialogue history + negotiation history |
| **Action plan**    | Material recommendation (supplier catalogue) + calibration parameters + critic review |
| **Debug**          | Full trace stream + raw state JSON |

Above the tabs sits a **Verdict Hero** banner that reads its urgency
from the same lifecycle scheduler the panel uses — so the banner,
the schedule card, and the recommendation list are always consistent.

---

## Persistence & memory

Two SQLite files are written to the repo root by default:

| File                        | Table(s)                | What lives here |
|-----------------------------|-------------------------|-----------------|
| `eix_checkpoints.sqlite`    | `checkpoints`, `writes` | LangGraph state after every node — enables cycle resume across restarts |
| `eix_fleet_memory.sqlite`   | `fleet_memory`          | Per-asset audit trail (kinds: `cycle_normal`, `cycle_action`, `engineer_decision`, `critic_weights`) |

The API's `_CYCLE_INDEX` in-memory dict is **rehydrated from
`fleet_memory.sqlite`** on the first request after startup
(see [`api/main.py::_rehydrate_cycle_index`](predictive_maintenance_agentic/api/main.py)),
so `/cycles` in the UI is never empty after a backend restart.

### Wiping the database

```bash
sqlite3 eix_fleet_memory.sqlite "DELETE FROM fleet_memory; VACUUM;"
sqlite3 eix_checkpoints.sqlite  "DELETE FROM checkpoints; DELETE FROM writes; VACUUM;"
# then restart the backend
```

---

## Simulation adapter (synthetic vs. real solver)

Stress features today come from a **surrogate** — not from real FEA:

```python
# predictive_maintenance_agentic/simulation/synthetic_adapter.py
base_amp = 0.25 * yield_strength_mpa * max(0.2, load_factor.mean()) \
           * (1.0 + 0.4 * vibration.mean())
```

The `stress_features.is_synthetic = True` flag is propagated all the
way to the UI so no viewer ever confuses this for a real ANSYS /
Abaqus / Creo / NASTRAN result.

To swap in a real solver:

1. Subclass `SimulationLayer` from `simulation/base.py`.
2. Implement `compute_stress_features()` — return a `StressFeatures`
   dataclass with `is_synthetic=False`.
3. Register the adapter and set `EIX_SIMULATION_ADAPTER=<your-adapter>`.

No agent code needs to change.

---

## Development workflow

```bash
# Backend — hot reload on save
python -m uvicorn predictive_maintenance_agentic.api.main:app \
       --host 127.0.0.1 --port 8000 --reload

# Frontend — HMR out of the box
cd pmx-frontend
npm run dev

# Regenerate TS types from the backend's OpenAPI spec
cd pmx-frontend
npm run gen:api-types    # → src/types/api.generated.ts
```

Linting / typechecking:

```bash
# Frontend
cd pmx-frontend
npm run typecheck
npm run lint
```

---

## Testing

```bash
# From the repo root
pytest predictive_maintenance_agentic/tests/
```

Key tests:

- `test_engines.py` — `AnomalyEngine`, `PhysicsEngine` invariants
- `test_graph_end_to_end.py` — full graph run with `checkpoint_path=":memory:"`
- Physics engine determinism: `PhysicsEngine.analyze(...)` returns
  identical numbers for identical inputs (verified by the
  `test_physics_engine_returns_finite_or_none` test).

---

## Troubleshooting

**"Cycles page is empty after a restart"**
The in-memory index rehydrates from SQLite the first time `/fleet/status`
is called. If the DB was wiped, no cycles will show — that's expected.

**"POST /analyze times out"**
The Gemini ReAct loop can take 60–150 s. The UI generates its
`cycle_id` client-side and navigates immediately; the pipeline fills in
via 750 ms polling. If you're calling the API from `curl`, use
`--max-time 300`.

**"Anomaly count is huge / all cycles are anomalous"**
Older versions had a training-contamination bug — the detector was
trained on the incoming batch. Now it trains on a fresh clean synthetic
baseline per (`asset_type`, `material`). Restart the backend to reset
the in-memory detector cache.

**"Predicted failure date changes between runs for identical CSVs"**
Fixed. The legacy `FatigueAnalyzer` used `random.uniform(...)` for
`life_used_percent`; our engine wrapper recomputes it deterministically
from `operational_cycles / total_life_cycles`.

**"Uvicorn refuses to start — port in use"**
```bash
fuser -k 8000/tcp 2>/dev/null
```

**"Frontend can't reach the backend"**
Check `pmx-frontend/.env` has `VITE_API_BASE_URL=http://127.0.0.1:8000`
and that CORS is open (the backend allows `http://127.0.0.1:5173`
by default via FastAPI's CORS middleware).

---

## Design decisions worth reading

- **Client-supplied `cycle_id`** so the UI can navigate to the detail
  page *before* the LLM pipeline finishes. Backend accepts an optional
  `cycle_id` on `POST /analyze` and registers the cycle in
  `_CYCLE_INDEX` before invoking the graph.
- **Health score is blended**: `min(stress_based, life_used_based)`.
  The stress-based half looks at σ / endurance-limit; the life-based
  half looks at `cycles_used_percent`. Worst-of-two prevents the UI
  showing "NORMAL 87.1" next to "Urgent within 7 days".
- **Lifecycle scheduler is RUL-aware**: even if stress health says
  "monitoring", a RUL < 7 days escalates urgency to `urgent`. The
  schedule-by date is also hard-clamped to never be later than the
  predicted-failure date.
- **Recommendations are catalogue-driven, not physics-driven**: the
  supplier data (implementation days, cost impact, part number) comes
  from a hardcoded dict in `anamoly-detection.py` — those are business
  facts about Alcoa, Carpenter, etc., not physics outputs. So they
  don't change per cycle. What *does* change is which alternative is
  chosen.
- **Judge falls back gracefully**: if Gemini is off / errors / returns
  malformed JSON, a deterministic Python function still emits a valid
  verdict from Physics vs ML deltas + fleet history. The trace shows
  `"source": "deterministic_fallback"`.
- **Everything is labelled synthetic**: the stress field, the anomaly
  training baseline, the 3D mesh, and the API responses all carry an
  `is_synthetic` flag or a `_note` string. No result masquerades as
  a real solver output.

---

## Sub-project READMEs

- Backend deep-dive: [`predictive_maintenance_agentic/README.md`](predictive_maintenance_agentic/README.md)
- Frontend deep-dive: [`pmx-frontend/README.md`](pmx-frontend/README.md)

---

Built for the hackathon. No warranty implied — the simulation adapter
is a surrogate and the material catalogue is illustrative, not
production-grade procurement data. Everything worth trusting is
computed from the CSV you upload and the physics + LLM stack above it.
