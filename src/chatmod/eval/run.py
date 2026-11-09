"""Run the pipeline over a labelled dataset and write predictions, metrics and a report."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from chatmod.config import PROJECT_ROOT, Config
from chatmod.eval.metrics import compute_metrics
from chatmod.eval.report import write_report
from chatmod.pipeline import Mode, Pipeline
from chatmod.schema import Decision, Message


async def run_mode(pipe: Pipeline, rows: list[dict[str, Any]], mode: Mode, chunk: int = 32,
                   progress: bool = False) -> list[Decision]:
    preds: list[Decision] = []
    for i in range(0, len(rows), chunk):
        msgs = [Message(id=r["id"], text=r["text"], context=r.get("context", [])) for r in rows[i : i + chunk]]
        preds += await pipe.moderate_batch(msgs, mode)
        if progress:
            llm = [d.tier2 for d in preds if d.tier2 is not None]
            errs = sum(1 for t in llm if t.error)
            cached = sum(1 for t in llm if t.cached)
            print(f"  [{mode.value}] {len(preds)}/{len(rows)} messages, LLM calls {len(llm)} "
                  f"({cached} cached, {errs} failed)", file=sys.stderr, flush=True)
            if pipe.judge is not None and pipe.judge.exhausted and errs:
                print(f"  [{mode.value}] stopping LLM calls: {pipe.judge.exhausted}", file=sys.stderr, flush=True)
    return preds


def run_eval(cfg: Config, rows: list[dict[str, Any]], modes: list[Mode], *, scorer: str | None = None,
             provider: str | None = None, out_dir: str | Path | None = None, dataset_name: str = "") -> Path:
    out = Path(out_dir) if out_dir else PROJECT_ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    needs_llm = any(m is not Mode.tier1_only for m in modes)
    pipe = Pipeline.from_config(cfg, scorer=scorer, provider=provider if needs_llm else "none",
                                with_queue=False, patient=True)

    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        preds = asyncio.run(run_mode(pipe, rows, mode, progress=mode is not Mode.tier1_only))
        results[mode.value] = compute_metrics(rows, preds, cfg)
        with (out / f"predictions_{mode.value}.jsonl").open("w") as f:
            for g, d in zip(rows, preds):
                f.write(json.dumps({"id": g["id"], "text": g["text"], "gold": g["label"],
                                    "gold_severity": g.get("severity"), "tags": g.get("tags", []),
                                    "pred": d.label.value, "severity": d.severity, "action": d.action.value,
                                    "route": d.route.value, "confidence": d.confidence,
                                    "rationale": d.rationale, "user_message": d.user_message,
                                    "t1_reason": d.tier1.reason if d.tier1 else None,
                                    "rule_hits": d.tier1.rule_hits if d.tier1 else [],
                                    "llm_error": d.tier2.error if d.tier2 else None}, ensure_ascii=False) + "\n")

    meta = {
        "dataset": dataset_name, "n": len(rows), "modes": [m.value for m in modes],
        "scorer": pipe.tier1.scorer.name, "llm": pipe.judge.name if pipe.judge else None,
        "thresholds": cfg.tier1.model_dump() | {"llm_conf_thr": cfg.tier2.conf_thr},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    write_report(out, meta, results)
    return out
