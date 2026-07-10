"""
Judge Agent (LLM-first, CSV-only input).

The Judge is now a genuine LLM agent:

    * Its ONLY input is the raw sensor CSV that flowed through the graph
      (state["sensor_batch"]). No physics numbers, no ML numbers, no
      history — the LLM has to reason about the component's health
      purely from the sensor readings.
    * It returns a strict JSON verdict with the fields the rest of the
      graph expects (`divergence`, `maintenance_required`,
      `confidence_score`, `route`, `root_cause`, ...).

If no LLM provider is configured (EIX_LLM_PROVIDER=none), or the LLM
call fails / returns malformed JSON, we fall back to the previous
deterministic verdict computed from Physics vs ML + fleet memory so the
graph stays runnable in offline / hackathon demos.
"""
from __future__ import annotations

import io
import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import THRESHOLDS, get_judge_llm
from ..tools import build_judge_tools
from ._shared import (
    append_trace,
    get_fleet_memory,
    sensor_batch_from_dict,
    to_native,
)


# Health statuses that mean "act now, don't spend cycles on calibration".
_URGENT_HEALTH = {"warning", "critical", "failed"}

# Cap how many rows we hand to the LLM to keep prompt tokens bounded.
# Head + tail + describe() covers the shape of realistic sensor batches.
_MAX_CSV_ROWS = 60

# ReAct loop budget: how many tool calls the Judge may make before it
# must emit a final structured verdict.
_JUDGE_TOOL_BUDGET = 6


def _msg_content_text(msg) -> str:
    """
    Normalize a LangChain message's `content` field to a plain string.

    Providers differ:
      * OpenAI / Anthropic → `content` is a str.
      * Gemini (thinking mode) → `content` is a list of parts like
        [{"type":"text","text":"..."}, {"type":"tool_use",...}].
    Non-text parts are dropped so downstream JSON parsing works.
    """
    content = getattr(msg, "content", "")
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict) and "text" in chunk:
                parts.append(str(chunk["text"]))
        return "\n".join(parts)
    return str(content)


# ============================================================
# Fleet memory (used only by the deterministic fallback)
# ============================================================
def _historical_signal(asset_id: str) -> Dict[str, Any]:
    memory = get_fleet_memory()
    history = memory.get_history(asset_id, limit=10)

    total_action_cycles = 0
    divergent_action_cycles = 0
    last_action: str = ""
    approvals = 0
    approval_total = 0

    for entry in history:
        if entry.kind == "cycle_action":
            total_action_cycles += 1
            payload = entry.payload or {}
            verdict = payload.get("verdict") or {}
            if verdict.get("divergence"):
                divergent_action_cycles += 1
            if not last_action:
                last_action = (payload.get("action") or {}).get("type") or ""
        elif entry.kind == "engineer_decision":
            approval_total += 1
            dec = (entry.payload or {}).get("decision") or {}
            if dec.get("approved"):
                approvals += 1

    divergence_rate = (
        divergent_action_cycles / total_action_cycles if total_action_cycles > 0 else 0.0
    )
    approval_rate = approvals / approval_total if approval_total > 0 else 0.0

    return {
        "history_size": len(history),
        "past_action_cycles": total_action_cycles,
        "past_divergence_rate": round(divergence_rate, 3),
        "past_engineer_approval_rate": round(approval_rate, 3),
        "last_action_type": last_action,
    }


# ============================================================
# LLM-driven verdict (primary path)
# ============================================================
_JUDGE_SYSTEM_PROMPT = """\
You are the Judge Agent in a predictive-maintenance digital-twin system.

You will be given ONLY the raw sensor telemetry for one operating cycle
of an industrial component, formatted as CSV. Numeric columns you may
see include:
  vibration, temperature, pressure, load_factor, speed,
  acoustic_emission, oil_pressure, oil_temperature, operational_cycles.

Your job is to decide, based ONLY on this CSV, whether the physics-vs-ML
predictions the rest of the system produced are trustworthy (no
divergence) or should be re-calibrated (divergence), AND whether the
component needs immediate maintenance.

Return a SINGLE valid JSON object — no prose, no code fences — with
EXACTLY these keys:

{
  "divergence":          <true|false>,
  "maintenance_required":<true|false>,
  "confidence_score":    <float in [0,1]>,
  "route":               <"action_fast_path" | "calibration" | "action_monitoring">,
  "stress_delta_mpa":    <float>,   # your estimated model disagreement
  "rul_delta_fraction":  <float>,   # your estimated model disagreement
  "hcf_lcf_note":        <one short sentence about HCF vs LCF risk>,
  "root_cause":          <one short sentence justifying your call>
}

Decision rules:
  * If any sensor channel shows large sudden spikes, sustained trend
    beyond normal band, or values well above typical operating ranges
    → set divergence=true.
  * If channels look bad enough to indicate imminent failure
    (very high vibration, temperature approaching yield, load spikes
    beyond design) → set maintenance_required=true and
    route="action_fast_path".
  * If divergence=true but not urgent → route="calibration".
  * If everything looks normal → divergence=false, maintenance_required=false,
    route="action_monitoring", confidence_score >= 0.75.
"""


def _dataframe_to_csv_prompt(df: pd.DataFrame) -> str:
    """
    Serialize the sensor batch as CSV for the LLM. Also include compact
    describe() statistics so the model gets both the raw shape and a
    summary in <= a few thousand tokens.
    """
    if len(df) <= _MAX_CSV_ROWS:
        rows_csv = df.to_csv(index=False)
    else:
        # First N/2 + last N/2 rows so we cover start and end trends.
        half = _MAX_CSV_ROWS // 2
        head_df = df.head(half)
        tail_df = df.tail(_MAX_CSV_ROWS - half)
        rows_csv = (
            head_df.to_csv(index=False)
            + tail_df.to_csv(index=False, header=False)
        )
        rows_csv = f"# NOTE: batch has {len(df)} rows; showing first {half} + last {_MAX_CSV_ROWS - half}\n" + rows_csv

    # Numeric summary.
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        buf = io.StringIO()
        numeric.describe().to_csv(buf)
        stats_csv = buf.getvalue()
    else:
        stats_csv = ""

    parts = ["=== SENSOR CSV ===", rows_csv]
    if stats_csv:
        parts += ["=== NUMERIC SUMMARY ===", stats_csv]
    return "\n".join(parts)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Pull the first JSON object out of an LLM response. Tolerates
    ```json fences, leading/trailing prose, etc.
    """
    if not text:
        return None
    # Strip common markdown fences.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    # Grab the first {...} block (non-greedy across newlines).
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _sanitize_llm_verdict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce LLM output into the schema the rest of the graph expects.
    Missing fields get safe defaults; wrong types are cast.
    """
    def _fbool(key: str, default: bool = False) -> bool:
        v = raw.get(key, default)
        if isinstance(v, str):
            return v.strip().lower() in {"true", "yes", "1"}
        return bool(v)

    def _ffloat(key: str, default: float = 0.0) -> float:
        try:
            return float(raw.get(key, default))
        except Exception:
            return default

    divergence = _fbool("divergence")
    maintenance_required = _fbool("maintenance_required")
    confidence_score = max(0.0, min(1.0, _ffloat("confidence_score", 0.5)))

    route = str(raw.get("route", "")).strip().lower()
    if route not in {"action_fast_path", "calibration", "action_monitoring"}:
        # Derive route from the two booleans if the LLM omitted / bungled it.
        if maintenance_required:
            route = "action_fast_path"
        elif divergence:
            route = "calibration"
        else:
            route = "action_monitoring"

    return {
        "confidence_score": round(confidence_score, 3),
        "divergence": divergence,
        "maintenance_required": maintenance_required,
        "route": route,
        "stress_delta_mpa": round(_ffloat("stress_delta_mpa"), 2),
        "rul_delta_fraction": round(_ffloat("rul_delta_fraction"), 3),
        "hcf_lcf_note": str(raw.get("hcf_lcf_note", "")).strip(),
        "root_cause": str(raw.get("root_cause", "")).strip()
                       or "LLM verdict provided without root cause.",
        "source": "llm_csv_only",
    }


def _llm_verdict_from_csv(sensor_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Call the LLM with only the CSV data. Return None on any failure."""
    llm = get_judge_llm()
    if llm is None:
        return None
    try:
        csv_prompt = _dataframe_to_csv_prompt(sensor_df)
        msg = llm.invoke(
            [
                ("system", _JUDGE_SYSTEM_PROMPT),
                ("human", csv_prompt),
            ]
        )
        text = _msg_content_text(msg)
        raw = _extract_json(text)
        if not raw:
            return None
        return _sanitize_llm_verdict(raw)
    except Exception:
        return None


# ============================================================
# ReAct-style verdict (primary path when tools + LLM are available)
# ============================================================
_JUDGE_REACT_SYSTEM_PROMPT = """\
You are the Judge Agent for a predictive-maintenance digital-twin
system. You have two prior model outputs (physics + ML) plus the raw
sensor CSV. You ALSO have five tools to gather extra evidence:

  * get_fleet_memory_stats(limit) — recent history counts for this asset.
  * find_similar_failures(stress_tolerance_mpa, limit) — past cycles
    with similar physics stress.
  * query_material_limits(material_name) — S_ut, S_e, yield, K_IC and
    current stress ratios.
  * simulate_what_if(stress_delta_mpa, crack_size_delta_mm) — re-run
    physics with adjusted stress or crack size.
  * get_recent_maintenance(days) — recent actions / engineer decisions.

Workflow:
  1. Call whichever tools you need (at most 6). Each call returns a JSON
     observation you can reason about.
  2. When you have enough evidence, STOP calling tools and reply with a
     SINGLE valid JSON object with EXACTLY these keys:

  {
    "divergence":           <true|false>,
    "maintenance_required": <true|false>,
    "confidence_score":     <float in [0,1]>,
    "route":                <"action_fast_path" | "calibration" | "action_monitoring">,
    "stress_delta_mpa":     <float>,
    "rul_delta_fraction":   <float>,
    "hcf_lcf_note":         <one short sentence about HCF vs LCF risk>,
    "root_cause":           <one short sentence citing what the tools revealed>
  }

Rules for the final verdict:
  * maintenance_required=true ⇒ route="action_fast_path".
  * divergence=true and not urgent ⇒ route="calibration".
  * otherwise ⇒ route="action_monitoring" and confidence_score >= 0.75.
  * Cite at least one tool observation in `root_cause` if you called any.
"""


def _llm_verdict_react(
    state: Dict[str, Any],
    sensor_df: pd.DataFrame,
    physics: Dict[str, Any],
    ml: Dict[str, Any],
) -> Optional[tuple]:
    """
    Run the Judge as a ReAct loop with tool-calling.

    Returns `(sanitized_verdict, tool_trace_entries)` on success or None
    on any failure (LLM unavailable, tool-binding not supported, no
    parseable final JSON).
    """
    llm = get_judge_llm()
    if llm is None:
        return None
    try:
        tools = build_judge_tools(state)
        llm_with_tools = llm.bind_tools(tools)
    except Exception:
        return None

    tool_by_name = {t.name: t for t in tools}
    tool_trace: List[Dict[str, Any]] = []

    csv_prompt = _dataframe_to_csv_prompt(sensor_df)
    context_prompt = (
        "Physics prediction: "
        + json.dumps({
            "stress_amplitude_mpa": physics.get("stress_amplitude_mpa"),
            "rul_hours": physics.get("rul_hours"),
            "failure_mode": physics.get("failure_mode"),
            "health_status": physics.get("health_status"),
        }, default=str)
        + "\nML correction: "
        + json.dumps({
            "predicted_stress_mpa": ml.get("predicted_stress_mpa"),
            "rul_hours": ml.get("rul_hours"),
            "method": ml.get("method"),
        }, default=str)
        + "\n\n"
        + csv_prompt
    )

    # LangChain message list (mixed tuples + Message objects work fine
    # with the OpenAI-compatible client we use).
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    messages: List[Any] = [
        SystemMessage(content=_JUDGE_REACT_SYSTEM_PROMPT),
        HumanMessage(content=context_prompt),
    ]

    for _ in range(_JUDGE_TOOL_BUDGET):
        try:
            ai_msg: AIMessage = llm_with_tools.invoke(messages)
        except Exception:
            return None
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            # No further tools requested → treat this message content
            # as the final verdict.
            text = _msg_content_text(ai_msg)
            raw = _extract_json(text)
            if not raw:
                return None
            verdict = _sanitize_llm_verdict(raw)
            verdict["source"] = "llm_react"
            verdict["tool_calls_made"] = len(tool_trace)
            return verdict, tool_trace

        # Execute each requested tool call and feed the observation
        # back to the LLM as a ToolMessage.
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            call_id = call.get("id") or name
            tool = tool_by_name.get(name)
            if tool is None:
                observation: Any = {"error": f"unknown tool {name}"}
            else:
                try:
                    observation = tool.invoke(args)
                except Exception as exc:
                    observation = {"error": str(exc)}
            # Compact observation for the trace panel.
            obs_str = json.dumps(observation, default=str)
            trace_snippet = obs_str[:240] + ("…" if len(obs_str) > 240 else "")
            tool_trace.append({
                "tool": name,
                "args": args,
                "observation_snippet": trace_snippet,
            })
            messages.append(
                ToolMessage(
                    content=obs_str[:4000],  # cap payload back to LLM
                    tool_call_id=call_id,
                )
            )

    # Budget exhausted — ask the LLM one more time for a final verdict.
    try:
        from langchain_core.messages import HumanMessage
        messages.append(
            HumanMessage(
                content=(
                    "Tool-call budget exhausted. Reply now with the FINAL "
                    "JSON verdict only, no more tool calls."
                )
            )
        )
        final_msg = llm.invoke(messages)  # unbound → no more tool calls
        text = _msg_content_text(final_msg)
        raw = _extract_json(text)
        if raw:
            verdict = _sanitize_llm_verdict(raw)
            verdict["source"] = "llm_react_budget_capped"
            verdict["tool_calls_made"] = len(tool_trace)
            return verdict, tool_trace
    except Exception:
        pass
    return None


# ============================================================
# Deterministic fallback (used only when the LLM is unavailable)
# ============================================================
def _deterministic_verdict(
    physics: Dict[str, Any],
    ml: Dict[str, Any],
    history_signal: Dict[str, Any],
) -> Dict[str, Any]:
    p_stress = float(physics.get("stress_amplitude_mpa") or 0.0)
    m_stress = float(ml.get("predicted_stress_mpa") or 0.0)
    stress_delta = abs(p_stress - m_stress)

    p_rul = physics.get("rul_hours")
    m_rul = ml.get("rul_hours") or 0.0
    if p_rul and p_rul > 0:
        rul_delta_frac = abs(p_rul - m_rul) / p_rul
    else:
        rul_delta_frac = 0.0

    sim_corr = float(physics.get("simulation_correlation") or 0.9)
    confidence = (
        sim_corr
        - min(1.0, stress_delta / max(THRESHOLDS.stress_divergence_mpa * 2, 1.0)) * 0.4
        - min(1.0, rul_delta_frac / max(THRESHOLDS.rul_divergence_fraction * 2, 0.01)) * 0.4
    )
    hist_penalty = 0.2 * history_signal.get("past_divergence_rate", 0.0)
    hist_bonus = 0.05 * history_signal.get("past_engineer_approval_rate", 0.0)
    confidence = max(0.0, min(1.0, confidence - hist_penalty + hist_bonus))

    divergent = (
        stress_delta > THRESHOLDS.stress_divergence_mpa
        or rul_delta_frac > THRESHOLDS.rul_divergence_fraction
        or confidence < THRESHOLDS.min_confidence_score
    )

    fm = physics.get("failure_mode") or "unknown"
    hcf_lcf_note = (
        f"Dominant failure mode from physics: {fm}. "
        f"Physics stress {p_stress:.1f} MPa vs ML stress {m_stress:.1f} MPa."
    )

    health = (physics.get("health_status") or "normal").lower()
    rul_hours = physics.get("rul_hours")
    ml_rul_hours = ml.get("rul_hours") or 0.0
    urgent_rul = (
        (rul_hours is not None and rul_hours < THRESHOLDS.low_rul_hours * 0.4)
        or (ml_rul_hours and ml_rul_hours < THRESHOLDS.low_rul_hours * 0.4)
    )
    maintenance_required = (health in _URGENT_HEALTH) or urgent_rul

    if divergent:
        if stress_delta > THRESHOLDS.stress_divergence_mpa:
            root_cause = (
                f"Stress mismatch of {stress_delta:.1f} MPa exceeds threshold "
                f"({THRESHOLDS.stress_divergence_mpa} MPa)."
            )
        elif rul_delta_frac > THRESHOLDS.rul_divergence_fraction:
            root_cause = (
                f"RUL disagreement {rul_delta_frac*100:.1f}% exceeds threshold."
            )
        else:
            root_cause = (
                f"Confidence {confidence:.2f} below minimum {THRESHOLDS.min_confidence_score}."
            )
    elif maintenance_required:
        root_cause = (
            f"Physics and ML agree but health={health} (RUL≈{rul_hours} h)."
        )
    else:
        root_cause = "Physics and ML agree within tolerance; asset healthy."

    if maintenance_required:
        route = "action_fast_path"
    elif divergent:
        route = "calibration"
    else:
        route = "action_monitoring"

    return {
        "confidence_score": round(confidence, 3),
        "divergence": bool(divergent),
        "maintenance_required": bool(maintenance_required),
        "route": route,
        "stress_delta_mpa": round(stress_delta, 2),
        "rul_delta_fraction": round(rul_delta_frac, 3),
        "hcf_lcf_note": hcf_lcf_note,
        "root_cause": root_cause,
        "source": "deterministic_fallback",
    }


# ============================================================
# Graph node
# ============================================================
def judge_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    physics = state.get("physics_prediction") or {}
    ml = state.get("ml_correction") or {}
    history_signal = _historical_signal(state["asset_id"])

    verdict: Optional[Dict[str, Any]] = None
    tool_trace: List[Dict[str, Any]] = []

    sensor_batch = state.get("sensor_batch")
    if sensor_batch:
        try:
            sensor_df = sensor_batch_from_dict(sensor_batch)
        except Exception:
            sensor_df = None

        if sensor_df is not None:
            # 1) Preferred: ReAct loop with tools.
            react_out = _llm_verdict_react(state, sensor_df, physics, ml)
            if react_out is not None:
                verdict, tool_trace = react_out
            # 2) Fallback: CSV-only LLM (no tools) — kept for backwards
            #    compat with legacy tests that stub bind_tools out.
            if verdict is None:
                verdict = _llm_verdict_from_csv(sensor_df)

    if verdict is None:
        # 3) Final fallback: deterministic verdict.
        verdict = _deterministic_verdict(physics, ml, history_signal)

    verdict["historical_context"] = history_signal

    trace = list(state.get("trace") or [])
    # Fold each ReAct tool call into the visible trace so the UI can
    # show the Judge's chain-of-tool-use.
    for call in tool_trace:
        trace.append({
            "node": "judge_agent.tool_call",
            "note": f"{call['tool']}({call['args']}) → {call['observation_snippet']}",
            "data": call,
        })
    trace.append({
        "node": "judge_agent",
        "note": (
            f"Judge [{verdict.get('source','?')}]: "
            f"confidence={verdict['confidence_score']}, "
            f"divergent={verdict['divergence']}, "
            f"maintenance_required={verdict['maintenance_required']}, "
            f"route={verdict['route']}"
            + (f" [tools={len(tool_trace)}]" if tool_trace else "")
        ),
        "data": verdict,
    })

    return {
        "judge_verdict": to_native(verdict),
        "trace": to_native(trace),
    }

