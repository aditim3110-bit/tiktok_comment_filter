"""Evaluation metrics for the moderation pipeline. Pure Python, no sklearn needed."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from chatmod.config import Config
from chatmod.schema import Action, Decision, Label, Route

LABELS = [l.value for l in Label]
HARM = [l for l in LABELS if l != "benign"]


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return round(s[k], 2)


def _error_kind(err: str) -> str:
    e = err.lower()
    if "quota exhausted" in e or "perday" in e:
        return "daily quota exhausted"
    if "429" in e or "rate limit" in e:
        return "rate limited (429)"
    if "timeout" in e:
        return "timeout"
    if "truncated" in e:
        return "truncated (max_output_tokens)"
    if "blocked" in e:
        return "blocked by safety filter"
    if "invalid output" in e:
        return "invalid output"
    if any(c in e for c in ("500", "502", "503", "504", "unavailable")):
        return "server error (5xx)"
    return "other"


def compute_metrics(gold: list[dict[str, Any]], preds: list[Decision], cfg: Config) -> dict[str, Any]:
    assert len(gold) == len(preds)
    n = len(gold)
    y = [g["label"] for g in gold]
    yhat = [d.label.value for d in preds]

    # ---- per-label P/R/F1 + confusion ----
    confusion: dict[str, Counter] = {l: Counter() for l in LABELS}
    for a, b in zip(y, yhat):
        confusion[a][b] += 1
    per_label = {}
    for l in LABELS:
        tp = confusion[l][l]
        fp = sum(confusion[o][l] for o in LABELS if o != l)
        fn = sum(confusion[l][o] for o in LABELS if o != l)
        per_label[l] = {**_prf(tp, fp, fn), "support": tp + fn}
    present_harm = [l for l in HARM if per_label[l]["support"] > 0]
    macro_f1_harm = round(sum(per_label[l]["f1"] for l in present_harm) / max(1, len(present_harm)), 4)

    # ---- binary harmful vs benign (label level) ----
    tp = sum(1 for a, b in zip(y, yhat) if a != "benign" and b != "benign")
    fp = sum(1 for a, b in zip(y, yhat) if a == "benign" and b != "benign")
    fn = sum(1 for a, b in zip(y, yhat) if a != "benign" and b == "benign")
    binary = _prf(tp, fp, fn)

    # ---- action level: what viewers actually experience ----
    gold_benign = [d for g, d in zip(gold, preds) if g["label"] == "benign"]
    gold_harm = [d for g, d in zip(gold, preds) if g["label"] != "benign"]
    benign_fpr = sum(d.action is not Action.allow for d in gold_benign) / max(1, len(gold_benign))
    benign_hidden = sum(d.action in (Action.hide, Action.block) for d in gold_benign) / max(1, len(gold_benign))
    harm_caught = sum(d.action is not Action.allow for d in gold_harm) / max(1, len(gold_harm))
    gold_sev3 = [d for g, d in zip(gold, preds) if g.get("severity", 0) >= 3]
    sev3_caught = sum(d.action in (Action.hide, Action.block) for d in gold_sev3) / max(1, len(gold_sev3))

    # ---- severity error on correctly-labelled harmful ----
    sev_err = [abs(g.get("severity", 0) - d.severity) for g, d in zip(gold, preds)
               if g["label"] != "benign" and g["label"] == d.label.value]
    sev_mae = round(sum(sev_err) / len(sev_err), 3) if sev_err else None

    # ---- routing ----
    routes = Counter(d.route.value for d in preds)
    route_share = {r.value: round(routes.get(r.value, 0) / max(1, n), 4) for r in Route if r is not Route.human}
    escalated_t2 = sum(1 for d in preds if d.tier2 is not None)
    per_route_acc = {}
    for r in (Route.tier1, Route.tier2, Route.tier3):
        pairs = [(g, d) for g, d in zip(gold, preds) if d.route is r]
        if pairs:
            per_route_acc[r.value] = {
                "n": len(pairs),
                "label_acc": round(sum(g["label"] == d.label.value for g, d in pairs) / len(pairs), 4),
            }
    # Tier 1 settled vs "unsettled guesses" (tier1-only mode)
    t1_settled = [(g, d) for g, d in zip(gold, preds) if d.route is Route.tier1 and not d.signals.get("unsettled")]
    t1_settled_acc = round(sum(g["label"] == d.label.value for g, d in t1_settled) / len(t1_settled), 4) if t1_settled else None

    # End-to-end assuming human reviewers are always right on tier3 items.
    oracle_correct = sum(1 for g, d in zip(gold, preds) if d.route is Route.tier3 or g["label"] == d.label.value)
    t2_calls = [d.tier2 for d in preds if d.tier2 is not None]
    t2_errors = sum(1 for t in t2_calls if t.error)
    error_kinds = Counter(_error_kind(t.error) for t in t2_calls if t.error)

    # ---- latency / cost ----
    lat_all = [d.latency_ms for d in preds]
    lat_t1 = [d.tier1.latency_ms for d in preds if d.tier1]
    lat_t2 = [d.tier2.latency_ms for d in preds if d.tier2 and not d.tier2.cached]
    in_tok = sum(d.tier2.input_tokens for d in preds if d.tier2)
    out_tok = sum(d.tier2.output_tokens for d in preds if d.tier2)
    cost = in_tok / 1e6 * cfg.tier2.price_in_per_m + out_tok / 1e6 * cfg.tier2.price_out_per_m

    # ---- hard-case tags ----
    by_tag: dict[str, list[bool]] = defaultdict(list)
    for g, d in zip(gold, preds):
        for t in g.get("tags", []):
            by_tag[t].append(g["label"] == d.label.value)
    tag_acc = {t: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for t, v in sorted(by_tag.items())}

    return {
        "n": n,
        "gold_distribution": dict(Counter(y)),
        "accuracy": round(sum(a == b for a, b in zip(y, yhat)) / max(1, n), 4),
        "macro_f1_harmful": macro_f1_harm,
        "binary": binary,
        "per_label": per_label,
        "confusion": {a: dict(c) for a, c in confusion.items()},
        "viewer_impact": {
            "benign_actioned_rate": round(benign_fpr, 4),
            "benign_hidden_rate": round(benign_hidden, 4),
            "harmful_caught_rate": round(harm_caught, 4),
            "severe_caught_rate": round(sev3_caught, 4),
        },
        "severity_mae": sev_mae,
        "routing": {
            "share": route_share,
            "escalated_to_llm": round(escalated_t2 / max(1, n), 4),
            "to_human": round(routes.get("tier3", 0) / max(1, n), 4),
            "per_route_accuracy": per_route_acc,
            "tier1_settled_accuracy": t1_settled_acc,
            "llm_errors": t2_errors,
            "llm_calls": len(t2_calls),
            "llm_error_rate": round(t2_errors / len(t2_calls), 4) if t2_calls else 0.0,
            "llm_error_kinds": dict(error_kinds.most_common()),
        },
        "oracle_human_accuracy": round(oracle_correct / max(1, n), 4),
        "latency_ms": {
            "end_to_end_p50": _pct(lat_all, 0.5), "end_to_end_p95": _pct(lat_all, 0.95),
            "tier1_p50": _pct(lat_t1, 0.5), "tier1_p95": _pct(lat_t1, 0.95),
            "tier2_p50": _pct(lat_t2, 0.5), "tier2_p95": _pct(lat_t2, 0.95),
        },
        "cost": {
            "input_tokens": in_tok, "output_tokens": out_tok,
            "usd_total": round(cost, 5), "usd_per_1k_messages": round(cost / max(1, n) * 1000, 4),
        },
        "tags": tag_acc,
    }
