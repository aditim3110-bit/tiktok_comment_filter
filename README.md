# chatmod: tiered moderation for live-stream chat

Every chat message passes through up to three tiers:

| Tier | What | Settles |
|---|---|---|
| 1 | Detoxify toxicity model + regex rules for scams, spam, risk phrases and escalation hints | Clear benign, clear scams, clear targeted abuse |
| 2 | LLM judge (Gemini Flash) with the written policy, returning structured JSON: `label, severity, confidence, rationale, user_message` | What Tier 1 can't settle |
| 3 | Human review queue + web UI | LLM confidence below threshold, LLM errors, and every self-harm / threat case |

Messages waiting for review are soft-hidden unless the LLM leaned benign. When the LLM fails, the pipeline fails safe: it hides the message and queues it for review.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[all]'      # or '.[dev]' for the offline core only
export GEMINI_API_KEY=...              # needed for Tier 2 with --llm gemini (the default)
```

## Usage

```bash
chatmod moderate "FREE COINS at coinsdrop.xyz" "gg ez"
chatmod serve                          # http://127.0.0.1:8000/review, /playground, /docs
chatmod eval --dataset seed            # runs full, tier1-only and tier2-only modes and compares them
chatmod eval --dataset seed,public --n 400
chatmod sweep --dataset seed           # tune tier1 low_thr / high_thr
chatmod queue list | stats | export    # export = human labels as an eval dataset
```

Useful flags:
- `--llm fake`: offline stand-in that exercises the plumbing only. Its numbers don't measure LLM quality.
- `--llm none`: Tier 1 only.
- `--scorer keyword`: no torch needed.

Each eval run writes `runs/<timestamp>/` containing `metrics.json`, `report.md`, `report.html` and `predictions_<mode>.jsonl`.

## Layout

- `policy/policy.md`: the written policy. The LLM and the reviewers both read it.
- `config/default.yaml`: thresholds, model id, the label×severity → action table, prices.
- `src/chatmod/tier1/`: normalization (leetspeak, homoglyphs, `t . me`, `d m m e`), rules, scorers, band logic.
- `src/chatmod/tier2/`: prompt, Gemini adapter, retry/timeout, SQLite verdict cache.
- `src/chatmod/tier3/`: SQLite review queue.
- `src/chatmod/api/`: FastAPI app and reviewer UI.
- `src/chatmod/eval/`: datasets, metrics, runner, sweep, report.
- `data/seed/seed.jsonl`: 220 hand-written live-chat examples, weighted toward hard cases (`tags`).

## Metrics to watch

- **benign actioned rate**: how often normal chat gets warned, hidden or blocked.
- **severity-3 hidden/blocked rate**: how often the dangerous messages are actually caught.
- **settled at tier 1 / sent to LLM / sent to human**: drives cost and reviewer load.
- **tier-1 settled accuracy**: Tier 1 errors are never seen by the LLM or a human, so this must stay high.

## Known limitations

- Tier 1 latency is about 200 ms per message on CPU (Detoxify `unbiased` is RoBERTa-base). To reach single-digit ms, batch messages, use a GPU, export to ONNX, or switch to `original-small`.
- Detoxify `unbiased` was trained on civil_comments, so Tier 1 scores on that dataset are optimistic.
- The seed set is small and adversarial. Re-sweep thresholds on real, labelled traffic, which `chatmod queue export` provides over time.
- Copy-paste flood and per-user rate rules need per-user state; that isn't implemented yet.
- The review UI has no authentication. Run it locally or behind auth.
