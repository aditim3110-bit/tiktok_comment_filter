import json

from chatmod.eval.datasets import SEED_PATH, load_jsonl
from chatmod.eval.metrics import compute_metrics
from chatmod.eval.run import run_eval
from chatmod.eval.sweep import sweep
from chatmod.pipeline import Mode
from chatmod.schema import Action, Decision, Label, Route


def test_seed_dataset_is_valid():
    rows = load_jsonl(SEED_PATH)
    assert len(rows) >= 200
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["label"] for r in rows} == {l.value for l in Label}
    assert all(0 <= r["severity"] <= 3 for r in rows)
    assert all((r["severity"] == 0) == (r["label"] == "benign") for r in rows)


def _d(label, action=Action.allow, route=Route.tier1, sev=0):
    return Decision(message_id="x", text="x", label=label, severity=sev, confidence=1, action=action, route=route)


def test_metrics_on_handmade_example(cfg):
    gold = [{"label": "benign", "severity": 0}, {"label": "benign", "severity": 0},
            {"label": "hate", "severity": 3}, {"label": "scam_spam", "severity": 2}]
    preds = [_d(Label.benign), _d(Label.harassment, Action.hide, sev=2),
             _d(Label.hate, Action.block, Route.tier2, 3), _d(Label.benign, route=Route.tier3)]
    m = compute_metrics(gold, preds, cfg)
    assert m["accuracy"] == 0.5
    assert m["viewer_impact"]["benign_actioned_rate"] == 0.5
    assert m["viewer_impact"]["harmful_caught_rate"] == 0.5
    assert m["viewer_impact"]["severe_caught_rate"] == 1.0
    assert m["per_label"]["hate"]["f1"] == 1.0
    assert m["routing"]["share"] == {"tier1": 0.5, "tier2": 0.25, "tier3": 0.25}
    assert m["oracle_human_accuracy"] == 0.75  # tier3 item counted as fixed by a human
    assert m["confusion"]["benign"]["harassment"] == 1


def test_run_eval_offline_writes_artifacts(tmp_path, cfg):
    rows = load_jsonl(SEED_PATH)
    out = run_eval(cfg, rows, list(Mode), scorer="keyword", provider="fake", out_dir=tmp_path)
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["results"]) == {"full", "tier1-only", "tier2-only"}
    assert (out / "report.html").read_text().startswith("<!doctype html>")
    assert "Summary by mode" in (out / "report.md").read_text()
    full = metrics["results"]["full"]
    assert full["n"] == len(rows) and full["routing"]["share"]["tier1"] > 0
    assert metrics["results"]["tier1-only"]["routing"]["escalated_to_llm"] == 0
    assert metrics["results"]["tier2-only"]["routing"]["escalated_to_llm"] == 1.0


def test_sweep_monotonic_settle_rate(cfg):
    rows = load_jsonl(SEED_PATH)
    res = sweep(cfg, rows, scorer="keyword", lows=[0.05, 0.3], highs=[0.95])
    a, b = res["points"]
    assert b["settle_rate"] >= a["settle_rate"]  # higher low_thr -> more allowed at tier 1
    assert res["best"] in res["points"]
