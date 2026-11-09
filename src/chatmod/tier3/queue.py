"""Tier 3: SQLite-backed human review queue."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chatmod.schema import Action, Decision, Label

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id    TEXT NOT NULL,
    text          TEXT NOT NULL,
    context       TEXT NOT NULL DEFAULT '[]',
    reason        TEXT NOT NULL,
    suggested_label    TEXT,
    suggested_severity INTEGER,
    suggested_conf     REAL,
    provisional_action TEXT NOT NULL,
    priority      INTEGER NOT NULL,
    decision_json TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',   -- pending | claimed | resolved
    reviewer      TEXT,
    final_label   TEXT,
    final_severity INTEGER,
    final_action  TEXT,
    notes         TEXT,
    created_at    TEXT NOT NULL,
    claimed_at    TEXT,
    resolved_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_items(status, priority DESC, created_at);
"""

# Higher = reviewed first. Safety-critical labels jump the queue.
_LABEL_PRIORITY = {Label.self_harm: 4, Label.threat_violence: 4, Label.hate: 2, Label.sexual: 2,
                   Label.harassment: 1, Label.scam_spam: 1, Label.benign: 0}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReviewQueue:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._db.executescript(_SCHEMA)

    def enqueue(self, decision: Decision, reason: str, context: list[str] | None = None) -> int:
        priority = decision.severity + _LABEL_PRIORITY.get(decision.label, 0)
        with self._lock:
            cur = self._db.execute(
                """INSERT INTO review_items (message_id, text, context, reason, suggested_label,
                   suggested_severity, suggested_conf, provisional_action, priority, decision_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (decision.message_id, decision.text, json.dumps(context or []), reason, decision.label.value,
                 decision.severity, decision.confidence, decision.action.value, priority,
                 decision.model_dump_json(), _now()),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def list(self, status: str | None = "pending", limit: int = 100) -> list[dict[str, Any]]:
        q = "SELECT * FROM review_items"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        q += " ORDER BY priority DESC, created_at ASC, id ASC LIMIT ?"
        with self._lock:
            rows = self._db.execute(q, (*args, limit)).fetchall()
        return [dict(r) for r in rows]

    def get(self, item_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM review_items WHERE id=?", (item_id,)).fetchone()
        return dict(row) if row else None

    def claim_next(self, reviewer: str) -> dict[str, Any] | None:
        """Atomically claim the highest-priority pending item."""
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM review_items WHERE status='pending' ORDER BY priority DESC, created_at ASC, id ASC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            self._db.execute("UPDATE review_items SET status='claimed', reviewer=?, claimed_at=? WHERE id=?",
                             (reviewer, _now(), row["id"]))
            self._db.commit()
        return self.get(row["id"])

    def resolve(self, item_id: int, label: Label, severity: int, action: Action,
                reviewer: str = "reviewer", notes: str = "") -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("SELECT status FROM review_items WHERE id=?", (item_id,)).fetchone()
            if row is None:
                raise KeyError(item_id)
            if row["status"] == "resolved":
                raise ValueError(f"item {item_id} already resolved")
            self._db.execute(
                """UPDATE review_items SET status='resolved', reviewer=?, final_label=?, final_severity=?,
                   final_action=?, notes=?, resolved_at=? WHERE id=?""",
                (reviewer, label.value, severity, action.value, notes, _now(), item_id),
            )
            self._db.commit()
        return self.get(item_id)  # type: ignore[return-value]

    def release(self, item_id: int) -> None:
        with self._lock:
            self._db.execute("UPDATE review_items SET status='pending', reviewer=NULL, claimed_at=NULL "
                             "WHERE id=? AND status='claimed'", (item_id,))
            self._db.commit()

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) n FROM review_items GROUP BY status").fetchall()
        out = {"pending": 0, "claimed": 0, "resolved": 0}
        out.update({r["status"]: r["n"] for r in rows})
        return out

    def export_labels(self) -> list[dict[str, Any]]:
        """Resolved items as gold labels in the eval dataset schema."""
        rows = self.list(status="resolved", limit=1_000_000)
        return [
            {"id": f"review-{r['id']}", "text": r["text"], "label": r["final_label"],
             "severity": r["final_severity"], "source": "human_review",
             "llm_label": r["suggested_label"], "reviewer": r["reviewer"]}
            for r in rows
        ]
