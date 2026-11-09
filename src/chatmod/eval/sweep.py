"""Tier 1 threshold sweep.

Toxicity scores are computed once; each (low_thr, high_thr) pair just re-applies the band
logic, so a full grid costs milliseconds. For each point we report how much traffic Tier 1
settles and how often those settled decisions are wrong; the goal is to settle as much as
possible while keeping settled-error and benign false positives low.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from chatmod.config import Config
from chatmod.schema import Label
from chatmod.tier1 import Tier1, get_scorer

DEFAULT_LOWS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
DEFAULT_HIGHS = [0.7, 0.8, 0.85, 0.9, 0.95, 0.98]


def sweep(cfg: Config, rows: list[dict[str, Any]], *, scorer: str | None = None,
          lows: list[float] = DEFAULT_LOWS, highs: list[float] = DEFAULT_HIGHS,
          max_settled_error: float = 0.05, max_benign_blocked: float = 0.005) -> dict[str, Any]:
    sc = get_scorer(scorer or cfg.tier1.scorer, cfg.tier1.detoxify_model)
    texts = [r["text"] for r in rows]
    tox = Tier1(sc, cfg.tier1).score(texts)
    gold = [r["label"] for r in rows]

    points = []
    for lo in lows:
        for hi in highs:
            if hi <= lo:
                continue
            t1 = Tier1(sc, cfg.tier1.model_copy(update={"low_thr": lo, "high_thr": hi}))
            res = [t1._decide(t, s) for t, s in zip(texts, tox)]
            settled = [(g, r) for g, r in zip(gold, res) if r.settled]
            wrong = sum(1 for g, r in settled if r.label is None or r.label.value != g)
            benign_settled_harm = sum(1 for g, r in settled if g == "benign" and r.label is not Label.benign)
            harm_let_through = sum(1 for g, r in settled if g != "benign" and r.label is Label.benign)
            n_benign = max(1, gold.count("benign"))
            n_harm = max(1, len(gold) - gold.count("benign"))
            points.append({
                "low_thr": lo, "high_thr": hi,
                "settle_rate": round(len(settled) / len(rows), 4),
                "settled_error": round(wrong / max(1, len(settled)), 4),
                "benign_blocked_rate": round(benign_settled_harm / n_benign, 4),
                "harmful_allowed_rate": round(harm_let_through / n_harm, 4),
            })

    ok = [p for p in points
          if p["settled_error"] <= max_settled_error and p["benign_blocked_rate"] <= max_benign_blocked]
    best = (max(ok, key=lambda p: (p["settle_rate"], -p["settled_error"])) if ok
            else min(points, key=lambda p: (p["benign_blocked_rate"], p["settled_error"])))
    return {"scorer": sc.name, "n": len(rows), "max_settled_error": max_settled_error,
            "max_benign_blocked": max_benign_blocked, "best": best, "points": points}


def write_sweep(result: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sweep.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result["points"][0]))
        w.writeheader()
        w.writerows(result["points"])
    return path
