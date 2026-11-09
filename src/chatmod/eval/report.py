"""Markdown + HTML eval report. Both are generated from the same table sections."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from chatmod.eval.metrics import LABELS

Table = tuple[str, list[str], list[list[Any]], str]  # title, headers, rows, note


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:,.1f}"
    return "—" if v is None else str(v)


def _sections(meta: dict[str, Any], results: dict[str, dict[str, Any]]) -> list[Table]:
    modes = list(results)
    T: list[Table] = []
    get = lambda m, *ks: _dig(results[m], ks)

    T.append(("Summary by mode", ["metric", *modes], [
        ["accuracy", *[get(m, "accuracy") for m in modes]],
        ["macro F1 (harmful labels)", *[get(m, "macro_f1_harmful") for m in modes]],
        ["harmful vs benign F1", *[get(m, "binary", "f1") for m in modes]],
        ["**benign actioned rate** (lower is better)", *[get(m, "viewer_impact", "benign_actioned_rate") for m in modes]],
        ["benign hidden/blocked rate", *[get(m, "viewer_impact", "benign_hidden_rate") for m in modes]],
        ["harmful caught rate", *[get(m, "viewer_impact", "harmful_caught_rate") for m in modes]],
        ["severity-3 hidden/blocked rate", *[get(m, "viewer_impact", "severe_caught_rate") for m in modes]],
        ["severity MAE (correct label)", *[get(m, "severity_mae") for m in modes]],
        ["settled at tier 1", *[get(m, "routing", "share", "tier1") for m in modes]],
        ["sent to LLM", *[get(m, "routing", "escalated_to_llm") for m in modes]],
        ["sent to human review", *[get(m, "routing", "to_human") for m in modes]],
        ["tier-1 settled accuracy", *[get(m, "routing", "tier1_settled_accuracy") for m in modes]],
        ["accuracy if humans resolve queue", *[get(m, "oracle_human_accuracy") for m in modes]],
        ["LLM calls", *[get(m, "routing", "llm_calls") for m in modes]],
        ["LLM errors", *[get(m, "routing", "llm_errors") for m in modes]],
        ["latency p50 ms", *[get(m, "latency_ms", "end_to_end_p50") for m in modes]],
        ["latency p95 ms", *[get(m, "latency_ms", "end_to_end_p95") for m in modes]],
        ["LLM latency p95 ms (uncached)", *[get(m, "latency_ms", "tier2_p95") for m in modes]],
        ["est. USD / 1k messages", *[get(m, "cost", "usd_per_1k_messages") for m in modes]],
    ], "tier1-only gives unsettled messages Tier 1's provisional guess. tier2-only sends every message to the LLM."))

    for m in modes:
        pl = results[m]["per_label"]
        T.append((f"Per-label ({m})", ["label", "precision", "recall", "f1", "support"],
                  [[l, pl[l]["precision"], pl[l]["recall"], pl[l]["f1"], pl[l]["support"]] for l in LABELS], ""))
        conf = results[m]["confusion"]
        T.append((f"Confusion matrix ({m}), rows = gold, cols = predicted", ["gold \\ pred", *LABELS],
                  [[g, *[conf.get(g, {}).get(p, 0) for p in LABELS]] for g in LABELS], ""))
        if results[m]["tags"]:
            T.append((f"Hard-case tags ({m})", ["tag", "n", "label accuracy"],
                      [[t, v["n"], v["acc"]] for t, v in results[m]["tags"].items()], ""))
    return T


def _dig(d: Any, ks: tuple) -> Any:
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _md(meta: dict[str, Any], tables: list[Table]) -> str:
    lines = ["# Moderation eval report", ""]
    for k, v in meta.items():
        lines.append(f"- **{k}**: `{v}`")
    for title, headers, rows, note in tables:
        lines += ["", f"## {title}", ""]
        if note:
            lines += [f"_{note}_", ""]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "---|" * len(headers))
        lines += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


_CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e8;--head:#f5f5f7}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--fg:#ececf0;--muted:#9a9aa2;--line:#2c2c31;--head:#1c1c20}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:22px}h2{font-size:16px;margin-top:32px}.meta{color:var(--muted);font-size:13px}
.wrap{overflow-x:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{background:var(--head)}em{color:var(--muted)}
.warn{border:1px solid #e8590c;background:color-mix(in srgb,#e8590c 12%,var(--bg));border-radius:8px;padding:10px 12px}
"""


def _html(meta: dict[str, Any], tables: list[Table]) -> str:
    e = html.escape
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Moderation eval</title><style>{_CSS}</style></head><body><h1>Moderation eval report</h1><div class='meta'>"]
    parts += [f"<div><b>{e(str(k))}</b>: {e(str(v))}</div>" for k, v in meta.items()]
    parts.append("</div>")
    for title, headers, rows, note in tables:
        parts.append(f"<h2>{e(title)}</h2>")
        if note:
            parts.append(f"<p><em>{e(note)}</em></p>")
        parts.append("<div class='wrap'><table><tr>" + "".join(f"<th>{e(h)}</th>" for h in headers) + "</tr>")
        for r in rows:
            cells = [e(_fmt(c)).replace("**", "") for c in r]
            parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        parts.append("</table></div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def llm_warnings(results: dict[str, dict[str, Any]]) -> list[str]:
    """One line per mode whose LLM calls failed. When an LLM call fails the decision falls
    back to Tier 1's guess, so those modes' metrics silently become Tier 1 metrics."""
    out = []
    for mode, r in results.items():
        ro = r.get("routing", {})
        if ro.get("llm_errors"):
            kinds = ", ".join(f"{k}: {v}" for k, v in ro.get("llm_error_kinds", {}).items())
            out.append(f"{mode}: {ro['llm_errors']}/{ro.get('llm_calls', '?')} LLM calls failed ({kinds}). "
                       f"Those messages fell back to Tier 1's guess, so this mode's quality metrics are NOT valid. "
                       f"Re-run the same command later: successful verdicts are cached, only failures are retried.")
    return out


def write_report(out: Path, meta: dict[str, Any], results: dict[str, dict[str, Any]]) -> None:
    tables = _sections(meta, results)
    warns = llm_warnings(results)
    md = _md(meta, tables)
    page = _html(meta, tables)
    if warns:
        md = md.replace("\n", "\n" + "".join(f"\n> **⚠ {w}**\n" for w in warns) + "\n", 1)
        banner = "".join(f"<p class='warn'>⚠ {html.escape(w)}</p>" for w in warns)
        page = page.replace("<div class='meta'>", banner + "<div class='meta'>", 1)
    (out / "report.md").write_text(md)
    (out / "report.html").write_text(page)
