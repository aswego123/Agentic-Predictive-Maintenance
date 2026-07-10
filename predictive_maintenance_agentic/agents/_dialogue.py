"""
Shared dialogue helper for the physics ⇄ ML negotiation.

When either agent runs its second round it can now emit a *structured*
`dialogue_move` — one of:

  * "concede"       — accept the other side's evidence and revise its
                      own prediction.
  * "hold"          — reject the other side's revision and stand by
                      its own prediction (with an explicit rationale).
  * "request_data"  — call for extra evidence (spectral_features or
                      thermal_gradient) via the data_fetch_agent
                      before committing.

Everything here is LLM-optional: if no LLM is configured we return a
plain "hold" move so the pipeline stays deterministic.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..config import get_llm
from ._shared import msg_content_text


_ALLOWED_MOVES = {"concede", "hold", "request_data"}
_ALLOWED_DATA_TOOLS = {"spectral_features", "thermal_gradient"}


_DIALOGUE_SYSTEM_PROMPT = """\
You are a maintenance-analytics agent participating in a two-round
negotiation with the OTHER MODEL (physics or ML). You have already
produced a prediction. Now you have seen the other side's prediction
AND their rationale.

Choose ONE of three structured moves and return a SINGLE valid JSON
object — no prose, no code fences.

Schema:

{
  "move":                 <"concede" | "hold" | "request_data">,
  "revised_prediction":   {"rul_hours": <float>, "stress_mpa": <float>} | null,
  "rationale":            <one to three concise sentences>,
  "data_request":         <"spectral_features" | "thermal_gradient" | null>
}

Rules:
  * "concede" — you accept the other side's evidence. `revised_prediction`
    MUST be non-null (updated numbers close to the other side or a
    blend). `data_request` MUST be null.
  * "hold" — you keep your prediction unchanged. `revised_prediction`
    MUST be null. `data_request` MUST be null.
  * "request_data" — you need an additional signal to commit. Pick a
    single tool from the allowed set:
        spectral_features — vibration FFT (dominant frequency, HF/LF
                            energy ratio) — good for bearing / rotor
                            wear diagnosis.
        thermal_gradient  — temperature and oil-temperature trends —
                            good for creep / cooling faults.
    `revised_prediction` MUST be null.

Be honest. If your prediction is stronger, hold. If the other side's
rationale is more compelling, concede. If you are genuinely unsure,
request data.
"""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _sanitize_move(raw: Dict[str, Any]) -> Dict[str, Any]:
    move = str(raw.get("move", "")).strip().lower()
    if move not in _ALLOWED_MOVES:
        move = "hold"
    revised = raw.get("revised_prediction")
    if move != "concede":
        revised = None
    elif isinstance(revised, dict):
        try:
            revised = {
                "rul_hours": float(revised.get("rul_hours")),
                "stress_mpa": float(revised.get("stress_mpa")),
            }
        except Exception:
            revised = None
            move = "hold"
    else:
        revised = None
        move = "hold"

    data_request = raw.get("data_request")
    if move != "request_data":
        data_request = None
    elif str(data_request).strip().lower() not in _ALLOWED_DATA_TOOLS:
        data_request = None
        move = "hold"
    else:
        data_request = str(data_request).strip().lower()

    return {
        "move": move,
        "revised_prediction": revised,
        "rationale": str(raw.get("rationale", "")).strip()
                     or f"(no rationale returned; defaulting to {move})",
        "data_request": data_request,
    }


def _fallback_move(
    self_side: str,
    own_pred: Dict[str, Any],
    other_pred: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic move when LLM is unavailable or errors out."""
    own_rul = float(own_pred.get("rul_hours") or 0.0)
    other_rul = float(other_pred.get("rul_hours") or 0.0)
    denom = max(own_rul, 1.0)
    gap = abs(own_rul - other_rul) / denom if own_rul > 0 else 0.0
    if gap < 0.10:
        return {
            "move": "hold",
            "revised_prediction": None,
            "rationale": f"gap {gap:.0%} within tolerance; standing by {self_side} prediction.",
            "data_request": None,
        }
    # LLM unavailable — never concede blindly. Hold with note.
    return {
        "move": "hold",
        "revised_prediction": None,
        "rationale": (
            f"{self_side} vs other RUL gap {gap:.0%}; LLM unavailable so "
            "holding position without dialogue."
        ),
        "data_request": None,
    }


def run_dialogue_move(
    *,
    self_side: str,                 # "physics" | "ml"
    own_prediction: Dict[str, Any],
    other_prediction: Dict[str, Any],
    round_number: int,
    prior_moves: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ask the LLM (or the deterministic fallback) which dialogue move to
    play in round 2. Returns a sanitized move dict.
    """
    llm = get_llm()
    if llm is None:
        return _fallback_move(self_side, own_prediction, other_prediction)

    payload = {
        "self_side": self_side,
        "round_number": round_number,
        "own_prediction": {
            "rul_hours": own_prediction.get("rul_hours"),
            "stress_mpa": own_prediction.get("predicted_stress_mpa")
                           or own_prediction.get("stress_amplitude_mpa"),
            "rationale": own_prediction.get("rationale", ""),
            "method": own_prediction.get("method")
                       or own_prediction.get("failure_mode", ""),
        },
        "other_prediction": {
            "rul_hours": other_prediction.get("rul_hours"),
            "stress_mpa": other_prediction.get("predicted_stress_mpa")
                           or other_prediction.get("stress_amplitude_mpa"),
            "rationale": other_prediction.get("rationale", ""),
            "method": other_prediction.get("method")
                       or other_prediction.get("failure_mode", ""),
        },
        "prior_moves": prior_moves[-4:],  # keep context small
    }

    try:
        msg = llm.invoke(
            [
                ("system", _DIALOGUE_SYSTEM_PROMPT),
                ("human", json.dumps(payload, default=str)),
            ]
        )
        text = msg_content_text(msg)
        raw = _extract_json(text)
        if raw is None:
            return _fallback_move(self_side, own_prediction, other_prediction)
        return _sanitize_move(raw)
    except Exception:  # pragma: no cover
        return _fallback_move(self_side, own_prediction, other_prediction)


def apply_concession(
    own_prediction: Dict[str, Any],
    revised: Dict[str, Any],
) -> Dict[str, Any]:
    """Overwrite RUL and stress in the prediction with the revised values."""
    out = dict(own_prediction)
    if "rul_hours" in revised:
        out["rul_hours"] = float(revised["rul_hours"])
    if "stress_mpa" in revised:
        if "predicted_stress_mpa" in out:
            out["predicted_stress_mpa"] = float(revised["stress_mpa"])
        else:
            out["stress_amplitude_mpa"] = float(revised["stress_mpa"])
    out["revised_after_dialogue"] = True
    return out


__all__ = ["run_dialogue_move", "apply_concession"]
