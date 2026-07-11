"""
Cross-cycle memory ("Agent Communication" panel in the diagram).

Stores per-asset entries (inspections, divergences, calibration
adjustments, engineer decisions). Backed by SQLite by default so the
same store survives restarts. Swap the storage layer for Postgres or
Redis by subclassing `FleetMemoryStore` — the agents only touch the
`get_history` / `add_entry` methods.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FleetMemoryEntry:
    entry_id: str
    asset_id: str
    kind: str  # "cycle_normal" | "cycle_action" | "divergence" | "calibration" | "engineer_decision"
    cycle_id: Optional[str]
    payload: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


class FleetMemoryStore:
    """SQLite-backed fleet memory. Thread-safe for a single-process demo."""

    def __init__(self, db_path: str = "./sqlite-data/eix_fleet_memory.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fleet_memory (
                    entry_id   TEXT PRIMARY KEY,
                    asset_id   TEXT NOT NULL,
                    kind       TEXT NOT NULL,
                    cycle_id   TEXT,
                    payload    TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asset ON fleet_memory (asset_id, created_at DESC)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_entry(
        self,
        asset_id: str,
        kind: str,
        payload: Dict[str, Any],
        cycle_id: Optional[str] = None,
    ) -> FleetMemoryEntry:
        entry = FleetMemoryEntry(
            entry_id=str(uuid.uuid4()),
            asset_id=asset_id,
            kind=kind,
            cycle_id=cycle_id,
            payload=payload,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO fleet_memory (entry_id, asset_id, kind, cycle_id, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry.entry_id,
                    entry.asset_id,
                    entry.kind,
                    entry.cycle_id,
                    json.dumps(payload, default=str),
                    entry.created_at,
                ),
            )
            conn.commit()
        return entry

    def get_history(
        self,
        asset_id: str,
        limit: int = 20,
        kinds: Optional[List[str]] = None,
    ) -> List[FleetMemoryEntry]:
        query = "SELECT * FROM fleet_memory WHERE asset_id = ?"
        params: List[Any] = [asset_id]
        if kinds:
            placeholders = ",".join(["?"] * len(kinds))
            query += f" AND kind IN ({placeholders})"
            params.extend(kinds)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            FleetMemoryEntry(
                entry_id=r["entry_id"],
                asset_id=r["asset_id"],
                kind=r["kind"],
                cycle_id=r["cycle_id"],
                payload=json.loads(r["payload"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def summarize_asset(self, asset_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT kind, COUNT(*) as c FROM fleet_memory WHERE asset_id = ? GROUP BY kind",
                (asset_id,),
            ).fetchall()
        counts = {r["kind"]: r["c"] for r in rows}
        last = self.get_history(asset_id, limit=1)
        return {
            "asset_id": asset_id,
            "counts": counts,
            "last_entry_at": last[0].created_at if last else None,
            "last_entry_kind": last[0].kind if last else None,
        }

    def entry_ids_since(self, cycle_id: str) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT entry_id FROM fleet_memory WHERE cycle_id = ?", (cycle_id,)
            ).fetchall()
        return [r["entry_id"] for r in rows]

    # ------------------------------------------------------------------
    # Critic helpers (per-asset retrospective learning)
    # ------------------------------------------------------------------
    def get_recent_cycle_actions(
        self, asset_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return the last `limit` cycle_action payloads for this asset,
        oldest-first, ready to feed to the Critic node."""
        entries = self.get_history(
            asset_id, limit=limit, kinds=["cycle_action", "cycle_normal"]
        )
        # get_history returns newest-first; the Critic prefers chronological.
        return [e.payload for e in reversed(entries)]

    def get_latest_critic_weights(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Return the most-recent `critic_weights` payload, or None."""
        entries = self.get_history(asset_id, limit=1, kinds=["critic_weights"])
        return entries[0].payload if entries else None

    @staticmethod
    def entry_to_dict(entry: FleetMemoryEntry) -> Dict[str, Any]:
        return asdict(entry)
