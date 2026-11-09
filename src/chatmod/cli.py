"""chatmod CLI: moderate, eval, sweep, serve, queue."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from chatmod.config import PROJECT_ROOT, load_config
from chatmod.schema import Message

app = typer.Typer(help="Tiered moderation for live-stream chat.", no_args_is_help=True)
queue_app = typer.Typer(help="Human review queue.", no_args_is_help=True)
app.add_typer(queue_app, name="queue")

ConfigOpt = typer.Option(None, "--config", "-c", help="YAML config (default: config/default.yaml)")
ScorerOpt = typer.Option(None, "--scorer", help="tier1 scorer: detoxify | keyword")
ProviderOpt = typer.Option(None, "--llm", help="tier2 provider: gemini | fake | none")


@app.command()
def moderate(text: list[str] = typer.Argument(..., help="message(s) to moderate"),
             config: Optional[Path] = ConfigOpt, scorer: Optional[str] = ScorerOpt,
             llm: Optional[str] = ProviderOpt, no_queue: bool = typer.Option(False, help="don't write to the review queue")):
    """Moderate one or more messages and print the decisions as JSON."""
    from chatmod.pipeline import Pipeline

    cfg = load_config(config)
    pipe = Pipeline.from_config(cfg, scorer=scorer, provider=llm, with_queue=not no_queue)
    decisions = asyncio.run(pipe.moderate_batch([Message(text=t) for t in text]))
    for d in decisions:
        typer.echo(d.model_dump_json(indent=2, exclude={"tier1": {"normalized_text"}}))


@app.command("eval")
def eval_cmd(dataset: str = typer.Option("seed", help="seed | civil_comments | sms_spam | tweet_eval_hate | public | path.jsonl (comma-separated to mix)"),
             mode: str = typer.Option("all", help="full | tier1-only | tier2-only | all"),
             n: int = typer.Option(400, help="sample size per public dataset"),
             seed: int = typer.Option(13), out: Optional[Path] = typer.Option(None),
             config: Optional[Path] = ConfigOpt, scorer: Optional[str] = ScorerOpt, llm: Optional[str] = ProviderOpt):
    """Run the eval harness and write runs/<timestamp>/{metrics.json,report.md,report.html,predictions_*.jsonl}."""
    from chatmod.eval.datasets import load_dataset_spec
    from chatmod.eval.run import run_eval
    from chatmod.pipeline import Mode

    cfg = load_config(config)
    rows = load_dataset_spec(dataset, n=n, seed=seed)
    modes = list(Mode) if mode == "all" else [Mode(mode)]
    path = run_eval(cfg, rows, modes, scorer=scorer, provider=llm, out_dir=out, dataset_name=dataset)
    metrics = json.loads((path / "metrics.json").read_text())["results"]
    typer.echo(f"\n{len(rows)} examples from {dataset!r}\n")
    hdr = f"{'mode':<12}{'acc':>7}{'mF1':>7}{'benignFP':>10}{'caught':>8}{'@T1':>7}{'→LLM':>7}{'→human':>8}{'$/1k':>8}{'p95ms':>8}"
    typer.echo(hdr)
    for m, r in metrics.items():
        vi, ro = r["viewer_impact"], r["routing"]
        typer.echo(f"{m:<12}{r['accuracy']:>7.3f}{r['macro_f1_harmful']:>7.3f}{vi['benign_actioned_rate']:>10.3f}"
                   f"{vi['harmful_caught_rate']:>8.3f}{ro['share']['tier1']:>7.2f}{ro['escalated_to_llm']:>7.2f}"
                   f"{ro['to_human']:>8.2f}{r['cost']['usd_per_1k_messages']:>8.3f}{r['latency_ms']['end_to_end_p95']:>8.1f}")
    from chatmod.eval.report import llm_warnings

    for w in llm_warnings(metrics):
        typer.secho(f"\n⚠ {w}", fg=typer.colors.YELLOW)
    typer.echo(f"\nreport: {path / 'report.html'}")


@app.command()
def sweep(dataset: str = typer.Option("seed"), n: int = typer.Option(400), seed: int = typer.Option(13),
          max_error: float = typer.Option(0.05, help="max acceptable error rate on tier-1-settled messages"),
          max_benign_blocked: float = typer.Option(0.005, help="max share of benign messages tier 1 may block"),
          config: Optional[Path] = ConfigOpt, scorer: Optional[str] = ScorerOpt):
    """Sweep tier 1 low/high thresholds; report settle rate vs. settled error."""
    from chatmod.eval.datasets import load_dataset_spec
    from chatmod.eval.sweep import sweep as run_sweep, write_sweep

    cfg = load_config(config)
    rows = load_dataset_spec(dataset, n=n, seed=seed)
    res = run_sweep(cfg, rows, scorer=scorer, max_settled_error=max_error, max_benign_blocked=max_benign_blocked)
    out = PROJECT_ROOT / "runs" / f"sweep-{datetime.now():%Y%m%d-%H%M%S}"
    csv_path = write_sweep(res, out)
    typer.echo(f"{'low':>6}{'high':>6}{'settled':>9}{'err':>7}{'benignBlk':>11}{'harmAllow':>11}")
    for p in res["points"]:
        mark = "  ←" if p is res["best"] else ""
        typer.echo(f"{p['low_thr']:>6.2f}{p['high_thr']:>6.2f}{p['settle_rate']:>9.3f}{p['settled_error']:>7.3f}"
                   f"{p['benign_blocked_rate']:>11.3f}{p['harmful_allowed_rate']:>11.3f}{mark}")
    b = res["best"]
    typer.echo(f"\nbest with settled error ≤ {max_error} and benign blocked ≤ {max_benign_blocked}: low_thr={b['low_thr']} high_thr={b['high_thr']} "
               f"(settles {b['settle_rate']:.0%}, error {b['settled_error']:.1%})\ncsv: {csv_path}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, config: Optional[Path] = ConfigOpt,
          scorer: Optional[str] = ScorerOpt, llm: Optional[str] = ProviderOpt):
    """Run the moderation API + reviewer UI (http://host:port/review)."""
    import uvicorn

    from chatmod.api.app import create_app
    from chatmod.pipeline import Pipeline

    cfg = load_config(config)
    pipe = Pipeline.from_config(cfg, scorer=scorer, provider=llm)
    uvicorn.run(create_app(pipe, cfg), host=host, port=port)


def _queue():
    from chatmod.tier3 import ReviewQueue

    cfg = load_config()
    return ReviewQueue(cfg.resolve(cfg.tier3.db_path))


@queue_app.command("stats")
def queue_stats():
    typer.echo(json.dumps(_queue().stats()))


@queue_app.command("list")
def queue_list(status: str = "pending", limit: int = 20):
    for it in _queue().list(status=status, limit=limit):
        typer.echo(f"#{it['id']:<5} [{it['suggested_label']}/{it['suggested_severity']} conf={it['suggested_conf']:.2f}] "
                   f"{it['text'][:70]!r}  ({it['reason']})")


@queue_app.command("export")
def queue_export(out: Path = typer.Option(PROJECT_ROOT / "data" / "reviewed.jsonl")):
    """Export human-resolved items as gold labels (usable with `chatmod eval --dataset <file>`)."""
    rows = _queue().export_labels()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    typer.echo(f"wrote {len(rows)} labels to {out}")


if __name__ == "__main__":
    app()
