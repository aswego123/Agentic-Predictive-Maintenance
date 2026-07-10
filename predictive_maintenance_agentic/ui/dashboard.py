"""
Streamlit dashboard for the agentic predictive-maintenance graph.

Run with:
    streamlit run predictive_maintenance_agentic/ui/dashboard.py

By default it talks to the FastAPI app at http://127.0.0.1:8000.
Override with the EIX_API_URL environment variable.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


API_URL = os.getenv("EIX_API_URL", "http://127.0.0.1:8000").rstrip("/")

ASSET_TYPES = [
    "aircraft_engine",
    "aircraft_landing_gear",
    "aircraft_brake",
    "aircraft_wing",
    "aircraft_fuselage",
    "train_bogie",
    "train_brake",
    "train_wheel",
    "train_traction_motor",
]

MATERIALS = [
    "Al7075-T6",
    "Al2024-T3",
    "Steel4340",
    "AISI4130",
    "Ti-6Al-4V",
    "Inconel718",
    "CastIron",
]


# Canonical sensor-batch columns expected by the anomaly engine + physics
# stack (see anamoly-detection.py::generate_sensor_data).
SENSOR_COLUMNS: List[str] = [
    "asset_id",
    "asset_type",
    "timestamp",
    "operational_cycles",
    "vibration",
    "temperature",
    "pressure",
    "load_factor",
    "speed",
    "acoustic_emission",
    "oil_pressure",
    "oil_temperature",
]

# Reasonable per-asset-family defaults for the manual-entry editor.
_AIRCRAFT_DEFAULTS = {
    "vibration": 0.4,
    "temperature": 950.0,
    "pressure": 100.0,
    "load_factor": 0.7,
    "speed": 10000.0,
    "acoustic_emission": 28.0,
    "oil_pressure": 50.0,
    "oil_temperature": 100.0,
}
_TRAIN_DEFAULTS = {
    "vibration": 1.2,
    "temperature": 50.0,
    "pressure": 200.0,
    "load_factor": 0.6,
    "speed": 120.0,
    "acoustic_emission": 20.0,
    "oil_pressure": 150.0,
    "oil_temperature": 75.0,
}


def _defaults_for(asset_type: str) -> Dict[str, float]:
    return _TRAIN_DEFAULTS if asset_type.startswith("train_") else _AIRCRAFT_DEFAULTS


def _template_dataframe(asset_id: str, asset_type: str, n_rows: int = 5) -> pd.DataFrame:
    """Small starter DataFrame the user can edit in-place."""
    defaults = _defaults_for(asset_type)
    ts = pd.date_range(start="2026-01-01", periods=n_rows, freq="h")
    rows = []
    for i in range(n_rows):
        row = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "timestamp": ts[i].isoformat(),
            "operational_cycles": i + 1,
        }
        row.update(defaults)
        rows.append(row)
    return pd.DataFrame(rows, columns=SENSOR_COLUMNS)


def _dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a user-supplied DataFrame into JSON-safe records for the API."""
    clean = df.copy()
    # Timestamps → ISO strings; NaNs → None.
    if "timestamp" in clean.columns:
        clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
        clean["timestamp"] = clean["timestamp"].apply(
            lambda v: v.isoformat() if pd.notna(v) else None
        )
    for col in clean.select_dtypes(include=["number"]).columns:
        clean[col] = clean[col].astype(float)
    records = clean.where(pd.notna(clean), None).to_dict(orient="records")
    return records


# ============================================================
# HTTP helpers
# ============================================================
def _api_get(path: str) -> Dict[str, Any]:
    r = requests.get(f"{API_URL}{path}", timeout=60)
    r.raise_for_status()
    return r.json()


def _api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # Gemini 2.5 Pro "thinking" mode + a judge ReAct loop with up to 6
    # tool calls can push a single /analyze well past a minute. 5 min
    # gives comfortable headroom without letting stuck calls hang forever.
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def _api_available() -> bool:
    try:
        _api_get("/health")
        return True
    except Exception:
        return False


# ============================================================
# UI
# ============================================================
st.set_page_config(
    page_title="EIx — Engineering Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Predictive Maintenance — Engineering Intelligence")
st.caption(
    "LangGraph + LangChain multi-agent orchestration over the physics + ML "
    "predictive-maintenance stack. Simulation layer is **synthetic** — "
    "no ANSYS/Abaqus/Creo/NASTRAN calls are made."
)

if not _api_available():
    st.error(
        f"Cannot reach the FastAPI backend at `{API_URL}`.\n\n"
        "Start it in another terminal with:\n\n"
        "```bash\nuvicorn predictive_maintenance_agentic.api.main:app --reload\n```"
    )
    st.stop()


# ---- Session state ----
if "last_cycle_id" not in st.session_state:
    st.session_state.last_cycle_id = None
if "last_state" not in st.session_state:
    st.session_state.last_state = None


# ============================================================
# Sidebar — kick off a cycle
# ============================================================
with st.sidebar:
    st.header("Start a cycle")
    asset_id = st.text_input("Asset ID", value="ENGINE-001")
    asset_type = st.selectbox("Asset type", ASSET_TYPES, index=0)
    material = st.selectbox("Material", MATERIALS, index=MATERIALS.index("Inconel718"))
    component = st.text_input("Component (optional)", value="turbine_blade")

    st.markdown("---")
    data_source = st.radio(
        "Sensor data source",
        [
            "Generate synthetic (auto)",
            "Upload file (CSV / JSON)",
            "Manual entry (table editor)",
        ],
        index=0,
    )

    # These are only used by the synthetic path.
    n_samples = 100
    scenario = "Force anomaly"
    uploaded_records: Optional[List[Dict[str, Any]]] = None
    upload_error: Optional[str] = None

    if data_source == "Generate synthetic (auto)":
        n_samples = st.slider("Synthetic samples", 20, 300, 100, step=10)
        scenario = st.radio(
            "Scenario",
            ["Force anomaly", "Force normal (short-circuit)", "As generated"],
            index=0,
        )

    elif data_source == "Upload file (CSV / JSON)":
        st.caption(
            "Upload sensor data with the columns below. "
            "Extra numeric columns are kept, missing ones are filled with "
            "asset-family defaults."
        )
        st.download_button(
            "📥 Download template CSV",
            data=_template_dataframe(asset_id, asset_type, n_rows=5).to_csv(index=False),
            file_name="sensor_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        uploaded = st.file_uploader(
            "Sensor batch file",
            type=["csv", "json"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".json"):
                    payload_bytes = uploaded.read()
                    parsed = json.loads(payload_bytes.decode("utf-8"))
                    if isinstance(parsed, dict) and "records" in parsed:
                        df_up = pd.DataFrame(parsed["records"])
                    else:
                        df_up = pd.DataFrame(parsed)
                else:
                    df_up = pd.read_csv(uploaded)
                # Fill missing sensor columns with sensible defaults.
                for col, val in _defaults_for(asset_type).items():
                    if col not in df_up.columns:
                        df_up[col] = val
                if "asset_id" not in df_up.columns:
                    df_up["asset_id"] = asset_id
                if "asset_type" not in df_up.columns:
                    df_up["asset_type"] = asset_type
                if "operational_cycles" not in df_up.columns:
                    df_up["operational_cycles"] = range(1, len(df_up) + 1)
                if "timestamp" not in df_up.columns:
                    df_up["timestamp"] = pd.date_range(
                        start="2026-01-01", periods=len(df_up), freq="h"
                    ).astype(str)
                uploaded_records = _dataframe_to_records(df_up)
                st.success(f"Parsed {len(uploaded_records)} rows.")
                with st.expander("Preview uploaded data", expanded=False):
                    st.dataframe(df_up.head(20), use_container_width=True)
            except Exception as exc:
                upload_error = str(exc)
                st.error(f"Failed to parse file: {exc}")

    else:  # Manual entry
        st.caption(
            "Edit the table below. Add rows with the ➕ button. "
            "Any numeric column works; the anomaly engine will use it."
        )
        editor_key = f"editor_{asset_id}_{asset_type}"
        starter = _template_dataframe(asset_id, asset_type, n_rows=5)
        edited = st.data_editor(
            starter,
            key=editor_key,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.TextColumn(help="ISO timestamp"),
                "operational_cycles": st.column_config.NumberColumn(step=1),
            },
        )
        try:
            uploaded_records = _dataframe_to_records(edited)
        except Exception as exc:
            upload_error = str(exc)
            st.error(f"Table not valid: {exc}")

    st.markdown("---")
    analyze_disabled = (
        data_source != "Generate synthetic (auto)"
        and (uploaded_records is None or upload_error is not None)
    )
    if st.button(
        "▶ Analyze",
        type="primary",
        use_container_width=True,
        disabled=analyze_disabled,
    ):
        payload: Dict[str, Any] = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "material_name": material,
            "component": component or None,
        }
        if data_source == "Generate synthetic (auto)":
            payload.update(
                {
                    "generate_synthetic": True,
                    "synthetic_n_samples": int(n_samples),
                    "force_anomaly": scenario == "Force anomaly",
                    "force_normal": scenario == "Force normal (short-circuit)",
                }
            )
        else:
            if not uploaded_records:
                st.error("No sensor rows provided.")
                st.stop()
            payload["sensor_batch"] = {
                "records": uploaded_records,
                "columns": list(uploaded_records[0].keys()),
            }

        try:
            with st.spinner("Running graph..."):
                res = _api_post("/analyze", payload)
            st.session_state.last_cycle_id = res.get("cycle_id")
            st.session_state.last_state = res
            st.success(f"Cycle {res.get('cycle_id')} → status={res.get('status')}")
        except Exception as exc:
            st.error(f"POST /analyze failed: {exc}")


# ============================================================
# Fleet status
# ============================================================
col_a, col_b = st.columns([2, 3])

with col_a:
    st.subheader("Fleet status")
    try:
        fs = _api_get("/fleet/status")
        st.metric("Simulation adapter", fs.get("simulation_adapter", "n/a"))
        st.metric("Known assets", len(fs.get("known_assets", [])))
        st.metric("Known cycles", len(fs.get("known_cycles", [])))

        summaries = fs.get("asset_summaries", [])
        if summaries:
            st.markdown("**Per-asset fleet-memory counts**")
            rows: List[Dict[str, Any]] = []
            for s in summaries:
                row = {"asset_id": s.get("asset_id")}
                row.update(s.get("counts", {}))
                rows.append(row)
            st.dataframe(pd.DataFrame(rows).fillna(0), hide_index=True, use_container_width=True)

        with st.expander("Known cycles"):
            cycles = fs.get("known_cycles", [])
            if cycles:
                st.dataframe(pd.DataFrame(cycles), hide_index=True, use_container_width=True)
            else:
                st.info("No cycles yet — start one from the sidebar.")
    except Exception as exc:
        st.error(f"GET /fleet/status failed: {exc}")


# ============================================================
# Cycle detail
# ============================================================
with col_b:
    st.subheader("Cycle detail")
    cycle_id = st.text_input(
        "Cycle ID",
        value=st.session_state.last_cycle_id or "",
        help="Populated after running /analyze; you can also paste any cycle_id.",
    )

    state: Optional[Dict[str, Any]] = None
    if cycle_id:
        try:
            state = _api_get(f"/cycles/{cycle_id}")
            st.session_state.last_state = state
        except Exception as exc:
            st.error(f"GET /cycles/{cycle_id} failed: {exc}")

    if state:
        status = state.get("status", "unknown")
        awaiting = state.get("awaiting_engineer_approval")
        badge = {
            "running": "🟡",
            "normal_end": "🟢",
            "action_taken": "🔵",
            "unresolved_divergence": "🔴",
        }.get(status, "⚪")
        st.markdown(f"### {badge} Status: `{status}`")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Anomalous?", str(state.get("is_anomalous")))
        m2.metric("Negotiation round", state.get("negotiation_round", 0))
        m3.metric("Re-sim round", state.get("resimulation_round", 0))
        m4.metric("Awaiting eng.", "yes" if awaiting else "no")

        # ---- Engineer approval widget ----
        if awaiting:
            st.warning("Graph is paused at **engineer_approval** — decision required.")
            with st.form("engineer_form"):
                col1, col2 = st.columns(2)
                with col1:
                    eng_id = st.text_input("Engineer ID", value="eng-42")
                    approved = st.selectbox("Decision", ["Approve", "Reject"], index=0)
                with col2:
                    notes = st.text_area("Notes", value="Calibration looks reasonable.")
                submitted = st.form_submit_button("Submit decision", type="primary")
                if submitted:
                    try:
                        res = _api_post(
                            "/engineer/approve",
                            {
                                "cycle_id": cycle_id,
                                "approved": approved == "Approve",
                                "engineer_id": eng_id,
                                "notes": notes,
                            },
                        )
                        st.session_state.last_state = res
                        st.success(
                            f"Resumed → status={res.get('status')}, "
                            f"resim_round={res.get('resimulation_round')}"
                        )
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"POST /engineer/approve failed: {exc}")

        # ---- Node trace ----
        with st.expander("Node-by-node trace", expanded=True):
            trace = state.get("trace") or []
            if trace:
                rows = [
                    {
                        "#": i + 1,
                        "node": t.get("node"),
                        "note": t.get("note"),
                    }
                    for i, t in enumerate(trace)
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.info("No trace yet.")

        # ---- Anomaly gate visualization ----
        anomaly = state.get("anomaly_result")
        if anomaly:
            st.markdown("### Anomaly gate")
            cols = st.columns(4)
            cols[0].metric("Is anomalous?", str(anomaly.get("is_anomalous")))
            cols[1].metric("Count", anomaly.get("anomaly_count", 0))
            cols[2].metric("Severity", anomaly.get("max_severity", "-"))
            cols[3].metric("Top channel", str(anomaly.get("top_channel") or "-"))
            top_ch = anomaly.get("top_channel")
            if top_ch:
                st.caption(
                    f"Highest-scoring anomaly was on channel **`{top_ch}`** "
                    f"with score **{anomaly.get('top_score', 0):.2f}**."
                )

        # ---- 3D stress intensity visualization ----
        stress_feats = state.get("stress_features") or {}
        field = (stress_feats.get("extras") or {}).get("stress_field_3d")
        if field and field.get("points"):
            geom = field.get("geometry", "-")
            st.markdown(
                f"### 3D Stress Intensity — geometry: **`{geom}`** (synthetic)"
            )
            st.caption(
                "Geometry archetype is chosen from the selected **asset type**. "
                "aircraft_engine / traction_motor → cylinder, wing → beam, "
                "brake / wheel → disc, landing_gear → strut, fuselage → shell, "
                "bogie → box."
            )
            info_cols = st.columns(4)
            info_cols[0].metric("Geometry", geom)
            info_cols[1].metric("Hotspot region", field.get("hotspot_region", "-"))
            info_cols[2].metric("Max stress (MPa)", f"{field.get('max_stress_mpa', 0):.1f}")
            info_cols[3].metric("Hotspot nodes", field.get("hotspot_count", 0))

            pts = pd.DataFrame(field["points"])
            hotspot_thresh = float(field.get("hotspot_threshold_mpa", pts["stress_mpa"].max()))

            # Per-geometry default camera so the shape is obvious.
            camera_by_geometry = {
                "beam":     dict(eye=dict(x=1.6, y=1.6, z=0.6)),
                "disc":     dict(eye=dict(x=0.2, y=0.2, z=2.2)),
                "cylinder": dict(eye=dict(x=1.8, y=1.8, z=1.4)),
                "strut":    dict(eye=dict(x=2.0, y=2.0, z=0.9)),
                "shell":    dict(eye=dict(x=1.8, y=1.8, z=1.2)),
                "box":      dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            }
            camera = camera_by_geometry.get(geom, dict(eye=dict(x=1.6, y=1.6, z=1.2)))

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=pts["x"],
                        y=pts["y"],
                        z=pts["z"],
                        mode="markers",
                        marker=dict(
                            size=pts["stress_mpa"].apply(
                                lambda v: 4 + 8 * (v / max(pts["stress_mpa"].max(), 1e-6))
                            ),
                            color=pts["stress_mpa"],
                            colorscale="Turbo",
                            colorbar=dict(title="Stress (MPa)"),
                            opacity=0.85,
                            symbol=pts["is_hotspot"].map({True: "diamond", False: "circle"}),
                            line=dict(width=0),
                        ),
                        text=[
                            f"stress={s:.1f} MPa"
                            + (" ⚠ HOTSPOT" if h else "")
                            for s, h in zip(pts["stress_mpa"], pts["is_hotspot"])
                        ],
                        hovertemplate="%{text}<extra></extra>",
                    )
                ]
            )
            fig.update_layout(
                title=dict(
                    text=f"Geometry archetype: {geom} — {len(pts)} nodes",
                    x=0.02,
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                scene=dict(
                    xaxis=dict(title="X"),
                    yaxis=dict(title="Y"),
                    zaxis=dict(title="Z"),
                    aspectmode="data",
                    camera=camera,
                ),
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"⚠ Synthetic stress field — {field.get('note', '')} "
                f"Diamond markers are nodes above the hotspot threshold "
                f"({hotspot_thresh:.0f} MPa)."
            )

        # ---- Physics / ML side-by-side ----
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Physics prediction**")
            phys = state.get("physics_prediction")
            if phys:
                st.json({
                    k: phys.get(k)
                    for k in [
                        "stress_amplitude_mpa",
                        "rul_hours",
                        "rul_years",
                        "failure_mode",
                        "health_score",
                        "health_status",
                        "cycles_used_percent",
                        "simulation_correlation",
                    ]
                })
            else:
                st.caption("Not run (or non-anomalous short-circuit).")

        with c2:
            st.markdown("**ML correction**")
            ml = state.get("ml_correction")
            if ml:
                st.json({
                    k: ml.get(k)
                    for k in [
                        "predicted_stress_mpa",
                        "rul_hours",
                        "confidence_lower_hours",
                        "confidence_upper_hours",
                        "method",
                        "gp_correction_applied",
                    ]
                })
            else:
                st.caption("Not run.")

        # ---- Physics vs ML comparison chart ----
        phys = state.get("physics_prediction") or {}
        ml = state.get("ml_correction") or {}
        if phys and ml:
            comp_rows = [
                {"metric": "Stress amplitude (MPa)",
                 "Physics": phys.get("stress_amplitude_mpa") or 0,
                 "ML":      ml.get("predicted_stress_mpa") or 0},
                {"metric": "RUL (hours)",
                 "Physics": phys.get("rul_hours") or 0,
                 "ML":      ml.get("rul_hours") or 0},
            ]
            comp_df = pd.DataFrame(comp_rows)
            comp_long = comp_df.melt(id_vars="metric", var_name="source", value_name="value")
            fig2 = px.bar(comp_long, x="metric", y="value", color="source", barmode="group",
                          title="Physics vs ML side-by-side")
            fig2.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        # ---- Judge verdict + historical panel ----
        verdict = state.get("judge_verdict")
        if verdict:
            st.markdown("### Judge verdict")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Confidence", f"{verdict.get('confidence_score', 0):.2f}")
            v2.metric("Divergence", str(verdict.get("divergence")))
            v3.metric("Maintenance required", str(verdict.get("maintenance_required")))
            v4.metric("Route", str(verdict.get("route", "-")))
            st.info(
                verdict.get("root_cause_llm")
                or verdict.get("root_cause")
                or ""
            )
            hist = verdict.get("historical_context") or {}
            if hist:
                st.markdown("**Historical context (fleet memory)**")
                hcols = st.columns(4)
                hcols[0].metric("History size", hist.get("history_size", 0))
                hcols[1].metric("Past action cycles", hist.get("past_action_cycles", 0))
                hcols[2].metric("Past divergence rate", f"{hist.get('past_divergence_rate', 0):.2f}")
                hcols[3].metric("Engineer approval rate", f"{hist.get('past_engineer_approval_rate', 0):.2f}")
                if hist.get("last_action_type"):
                    st.caption(f"Last action for this asset: **{hist['last_action_type']}**")

        # ---- Calibration ----
        calib = state.get("calibration_result")
        if calib:
            st.markdown("### Calibration suggestion")
            ccol1, ccol2, ccol3 = st.columns(3)
            ccol1.metric("Prior stress", f"{calib.get('prior_stress_mpa', 0):.1f} MPa")
            ccol2.metric("Posterior stress", f"{calib.get('posterior_stress_mpa', 0):.1f} MPa")
            ccol3.metric("Geometry factor", f"{calib.get('geometry_factor_new', 0):.2f}",
                         f"was {calib.get('geometry_factor_old', 0):.2f}")
            suggestion = calib.get("engineer_suggestion_llm") or calib.get("engineer_suggestion")
            if suggestion:
                st.markdown(suggestion)
            with st.expander("Raw calibration payload"):
                st.json(calib)

        # ---- Action + work order ----
        action = state.get("action")
        if action:
            st.markdown("### Action")
            st.json(action)
        wo = state.get("work_order")
        if wo:
            st.markdown("### MRO/ERP work order (stub)")
            st.json(wo)

        # ---- Physics ⇄ ML dialogue (round-2 negotiation) ----
        dialogue = state.get("dialogue_history") or []
        if dialogue:
            st.markdown("### 💬 Physics ⇄ ML dialogue")
            for move in dialogue:
                source = move.get("source", "?")
                move_kind = move.get("move", "?")
                icon = {"concede": "🤝", "hold": "🛡️", "request_data": "🔍"}.get(move_kind, "•")
                header = f"{icon} **{source.upper()}** — round {move.get('round','?')} — `{move_kind}`"
                if move.get("data_request"):
                    header += f" (requested `{move['data_request']}`)"
                st.markdown(header)
                rationale = move.get("rationale")
                if rationale:
                    st.caption(rationale)
                revised = move.get("revised_prediction")
                if revised:
                    st.caption(
                        f"revised → stress={revised.get('stress_mpa','?')} MPa, "
                        f"RUL={revised.get('rul_hours','?')} h"
                    )
            fetched = state.get("fetched_features") or {}
            if fetched:
                with st.expander("Data-fetch results (from dialogue request)"):
                    st.json(fetched)

        # ---- Judge ReAct tool calls ----
        verdict = state.get("judge_verdict") or {}
        tool_calls_made = verdict.get("tool_calls_made")
        if tool_calls_made:
            trace_entries = state.get("trace") or []
            judge_tool_calls = [
                t for t in trace_entries
                if str(t.get("node", "")).startswith("judge_agent.tool_call")
            ]
            if judge_tool_calls:
                st.markdown(
                    f"### 🛠️ Judge tool calls ({len(judge_tool_calls)}) "
                    f"— source: `{verdict.get('source','?')}`"
                )
                for i, entry in enumerate(judge_tool_calls, 1):
                    data = entry.get("data") or {}
                    st.markdown(
                        f"**{i}. `{data.get('tool','?')}`** — args: `{data.get('args', {})}`"
                    )
                    snippet = data.get("observation_snippet") or entry.get("note", "")
                    st.caption(snippet)

        # ---- Critic retrospective (per-asset learning) ----
        critic = state.get("critic_review")
        if critic:
            st.markdown("### 🔎 Critic review (post-cycle reflection)")
            prev_w = critic.get("previous_physics_weight")
            new_w = critic.get("physics_weight")
            n_cycles = critic.get("n_cycles_considered", 0)

            kcols = st.columns(3)
            kcols[0].metric("Cycles considered", n_cycles)
            if prev_w is not None and new_w is not None:
                delta = round(float(new_w) - float(prev_w), 3)
                kcols[1].metric(
                    "Physics weight",
                    f"{float(new_w):.2f}",
                    delta=f"{delta:+.2f}" if delta else None,
                )
            else:
                kcols[1].metric("Physics weight", f"{float(new_w or 0.8):.2f}")
            kcols[2].metric(
                "Prev physics weight",
                f"{float(prev_w):.2f}" if prev_w is not None else "—",
            )

            rationale = critic.get("rationale")
            if rationale:
                st.info(rationale)

            stats = critic.get("stats") or {}
            if stats:
                st.markdown("**Retrospective stats (last N cycles)**")
                scols = st.columns(4)
                scols[0].metric(
                    "Divergence rate",
                    f"{float(stats.get('divergence_rate', 0)) * 100:.0f}%",
                )
                scols[1].metric(
                    "Escalation rate",
                    f"{float(stats.get('escalation_rate', 0)) * 100:.0f}%",
                )
                scols[2].metric(
                    "Physics RUL σ",
                    f"{float(stats.get('physics_rul_std', 0)):.1f} h",
                )
                scols[3].metric(
                    "ML RUL σ",
                    f"{float(stats.get('ml_rul_std', 0)):.1f} h",
                )
                gap = stats.get("mean_signed_gap")
                if isinstance(gap, (int, float)):
                    direction = (
                        "physics optimistic vs ML"
                        if gap > 0.15
                        else "physics pessimistic vs ML"
                        if gap < -0.15
                        else "physics ≈ ML"
                    )
                    st.caption(
                        f"Mean signed gap (physics−ml)/physics = "
                        f"{gap:+.2f} → {direction}"
                    )

            with st.expander("Raw critic payload"):
                st.json(critic)

        # ---- Full raw state ----
        with st.expander("Raw state"):
            st.code(json.dumps(state, indent=2, default=str), language="json")
    else:
        st.info("Start a cycle from the sidebar to see details here.")
