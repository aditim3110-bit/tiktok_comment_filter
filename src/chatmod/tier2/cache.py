"""SQLite cache of LLM verdicts keyed by (backend, prompt hash).

Makes eval runs reproducible and cheap to re-run after changing only Tier 1 thresholds.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

from chatmod.schema import Tier2Result


class JudgeCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute("CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, result TEXT NOT NULL)")
        self._db.commit()

    @staticmethod
    def _key(backend: str, system: str, user: str) -> str:
        h = hashlib.sha256()
        for part in (backend, system, user):
            h.update(part.encode())
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, backend: str, system: str, user: str) -> Tier2Result | None:
        with self._lock:
            row = self._db.execute("SELECT result FROM verdicts WHERE key=?", (self._key(backend, system, user),)).fetchone()
        return Tier2Result.model_validate_json(row[0]) if row else None

    def put(self, backend: str, system: str, user: str, res: Tier2Result) -> None:
        if res.output is None:
            return  # never cache errors
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO verdicts VALUES (?, ?)",
                             (self._key(backend, system, user), res.model_dump_json()))
            self._db.commit()
