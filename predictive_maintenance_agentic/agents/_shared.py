"""
Shared helpers for agent nodes: sensor-batch (de)serialization, trace
append, engine singletons.

Keeping this in one place avoids each agent recreating engines on every
node call and keeps the LangGraph state JSON-serializable.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd

from ..engines import (
    AnomalyEngine,
    MaterialEngine,
    PhysicsEngine,
    RULEngine,
)
from ..memory.fleet_memory import FleetMemoryStore
from ..simulation import SimulationLayer, SyntheticSimulationAdapter


# ---------------------------------------------------------------
# Engine singletons (safe within a single Python process).
# The API layer replaces `_SIM` at startup if a different adapter
# is provided.
# ---------------------------------------------------------------
_ANOMALY = AnomalyEngine()
_PHYSICS = PhysicsEngine()
_ML = RULEngine()
_MATERIAL = MaterialEngine()
_SIM: SimulationLayer = SyntheticSimulationAdapter()
_MEMORY: Optional[FleetMemoryStore] = None


def get_anomaly_engine() -> AnomalyEngine:
    return _ANOMALY


def get_physics_engine() -> PhysicsEngine:
    return _PHYSICS


def get_ml_engine() -> RULEngine:
    return _ML


def get_material_engine() -> MaterialEngine:
    return _MATERIAL


def get_simulation_layer() -> SimulationLayer:
    return _SIM


def set_simulation_layer(layer: SimulationLayer) -> None:
    global _SIM
    _SIM = layer


def get_fleet_memory() -> FleetMemoryStore:
    global _MEMORY
    if _MEMORY is None:
        from ..config import MEMORY as _CFG
        _MEMORY = FleetMemoryStore(_CFG.fleet_memory_path)
    return _MEMORY


def set_fleet_memory(mem: FleetMemoryStore) -> None:
    global _MEMORY
    _MEMORY = mem


# ---------------------------------------------------------------
# Sensor batch (de)serialization for graph-state safety.
# ---------------------------------------------------------------
def sensor_batch_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """Round-trip-safe representation for LangGraph state."""
    return {
        "columns": list(df.columns),
        "records": df.astype(object).where(df.notna(), None).to_dict(orient="records"),
    }


def sensor_batch_from_dict(payload: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(payload["records"])


# ---------------------------------------------------------------
# Numpy → native-Python sanitizer for state payloads.
# LangGraph's msgpack checkpointer can't serialize numpy scalars.
# ---------------------------------------------------------------
try:
    import numpy as _np  # noqa: WPS433
except ImportError:  # pragma: no cover
    _np = None


def to_native(obj: Any) -> Any:
    """Recursively coerce numpy scalars / arrays into native Python types."""
    if obj is None:
        return None
    if _np is not None:
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            return [to_native(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    # pandas.Timestamp -> iso string
    if hasattr(obj, "isoformat") and not isinstance(obj, (str, bytes)):
        try:
            return obj.isoformat()
        except Exception:  # pragma: no cover
            pass
    return obj


# ---------------------------------------------------------------
# Trace helper — every node appends one row so `/cycles/{id}` can
# show a Judge-friendly execution log.
# ---------------------------------------------------------------
def append_trace(state: Dict[str, Any], node: str, note: str, data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    trace = list(state.get("trace") or [])
    trace.append(
        {
            "ts": time.time(),
            "node": node,
            "note": note,
            "data": data or {},
        }
    )
    return trace


# ---------------------------------------------------------------
# LLM message-content normalization.
#
# LangChain message `.content` may be a plain `str` (OpenAI, Anthropic,
# Azure OpenAI) OR a `list[dict]` of typed parts when the underlying
# provider streams tool/thinking segments (Gemini, some Anthropic modes).
# Every LLM caller in this package should route responses through here
# so downstream regex/JSON parsing stays provider-agnostic.
# ---------------------------------------------------------------
def msg_content_text(msg: Any) -> str:
    """Normalize LangChain message content (str or list of parts) to plain str."""
    content = getattr(msg, "content", "")
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict) and "text" in chunk:
                parts.append(str(chunk["text"]))
        return "\n".join(parts)
    return str(content)
