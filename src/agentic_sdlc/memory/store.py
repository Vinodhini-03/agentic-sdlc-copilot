"""
Lightweight persistent memory for the agent: past review decisions,
resolved feedback, and short-term conversation summaries. Backed by
SQLite so it works with zero external infra; swap `_conn` for Postgres
in production by changing this file only.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MemoryRecord:
    repo: str
    key: str          # e.g. a stable hash of (file_path, issue_type)
    value: dict[str, Any]
    ts: float


class MemoryStore:
    def __init__(self, db_path: str | Path = "agent_memory.sqlite3") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                repo TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                ts REAL NOT NULL,
                PRIMARY KEY (repo, key)
            )
            """
        )
        self._conn.commit()

    def upsert(self, repo: str, key: str, value: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO memory (repo, key, value, ts) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(repo, key) DO UPDATE SET value=excluded.value, ts=excluded.ts",
            (repo, key, json.dumps(value), time.time()),
        )
        self._conn.commit()

    def get(self, repo: str, key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT value FROM memory WHERE repo = ? AND key = ?", (repo, key)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def all_for_repo(self, repo: str) -> list[MemoryRecord]:
        rows = self._conn.execute(
            "SELECT repo, key, value, ts FROM memory WHERE repo = ?", (repo,)
        ).fetchall()
        return [MemoryRecord(repo=r, key=k, value=json.loads(v), ts=t) for r, k, v, t in rows]

    def close(self) -> None:
        self._conn.close()
