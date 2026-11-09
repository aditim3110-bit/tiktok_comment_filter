"""Eval dataset loaders. Every loader yields the common schema:
{id, text, label, severity, tags, source}

Public datasets are sampled to a fixed size with a fixed seed and cached to data/public/.
CAVEAT: Detoxify ``unbiased`` was trained on Jigsaw Unintended Bias (= civil_comments), so
Tier 1 numbers on civil_comments are optimistic. Use it for Tier 2 / pipeline comparisons
and for scale, and trust the seed set for Tier 1 quality.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

from chatmod.config import PROJECT_ROOT
from chatmod.schema import Label

SEED_PATH = PROJECT_ROOT / "data" / "seed" / "seed.jsonl"
PUBLIC_DIR = PROJECT_ROOT / "data" / "public"

Example = dict[str, Any]


def load_jsonl(path: str | Path) -> list[Example]:
    rows = []
    with Path(path).open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            r.setdefault("id", f"{Path(path).stem}-{i:05d}")
            r.setdefault("severity", 0 if r["label"] == "benign" else 2)
            r.setdefault("tags", [])
            r.setdefault("source", Path(path).stem)
            Label(r["label"])  # validate
            rows.append(r)
    return rows


def _write_jsonl(rows: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- public dataset mappers ----------------------------------------------------------

def _civil_comments(n: int, seed: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("google/civil_comments", split="test")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    head_map = [("threat", Label.threat_violence), ("identity_attack", Label.hate),
                ("sexual_explicit", Label.sexual), ("insult", Label.harassment)]
    want_harm, want_benign = n // 2, n - n // 2
    out: list[Example] = []
    nh = nb = 0
    for i in idx:
        if nh >= want_harm and nb >= want_benign:
            break
        r = ds[i]
        text = r["text"].strip()
        if not text or len(text) > 300:  # keep chat-length messages
            continue
        tox = r["toxicity"]
        if tox < 0.1 and nb < want_benign:
            out.append({"text": text, "label": "benign", "severity": 0}); nb += 1
        elif tox >= 0.5 and nh < want_harm:
            head, label = max(head_map, key=lambda hl: r[hl[0]])
            if r[head] < 0.5:
                label = Label.harassment
            sev = 3 if r["severe_toxicity"] >= 0.3 or tox >= 0.9 else 2
            out.append({"text": text, "label": label.value, "severity": sev}); nh += 1
    return out


def _sms_spam(n: int, seed: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("ucirvine/sms_spam", split="train")
    rng = random.Random(seed)
    spam = [r["sms"].strip() for r in ds if r["label"] == 1]
    ham = [r["sms"].strip() for r in ds if r["label"] == 0]
    rng.shuffle(spam); rng.shuffle(ham)
    out = [{"text": t, "label": "scam_spam", "severity": 2} for t in spam[: n // 2]]
    out += [{"text": t, "label": "benign", "severity": 0} for t in ham[: n - n // 2]]
    return out


def _tweet_eval_hate(n: int, seed: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("cardiffnlp/tweet_eval", "hate", split="test")
    rng = random.Random(seed)
    rows = list(ds)
    rng.shuffle(rows)
    hate = [r["text"] for r in rows if r["label"] == 1][: n // 2]
    ok = [r["text"] for r in rows if r["label"] == 0][: n - n // 2]
    return ([{"text": t, "label": "hate", "severity": 2} for t in hate]
            + [{"text": t, "label": "benign", "severity": 0} for t in ok])


PUBLIC: dict[str, Callable[[int, int], list[Example]]] = {
    "civil_comments": _civil_comments,
    "sms_spam": _sms_spam,
    "tweet_eval_hate": _tweet_eval_hate,
}


def load_public(name: str, n: int = 400, seed: int = 13, refresh: bool = False) -> list[Example]:
    if name not in PUBLIC:
        raise ValueError(f"unknown public dataset {name!r}; choose from {sorted(PUBLIC)}")
    cache = PUBLIC_DIR / f"{name}_n{n}_s{seed}.jsonl"
    if cache.exists() and not refresh:
        return load_jsonl(cache)
    try:
        rows = PUBLIC[name](n, seed)
    except ImportError as e:
        raise RuntimeError("HF datasets not installed. Run: pip install -e '.[data]'") from e
    for i, r in enumerate(rows):
        r.update(id=f"{name}-{i:05d}", tags=[], source=name)
    _write_jsonl(rows, cache)
    return rows


def load_dataset_spec(spec: str, n: int = 400, seed: int = 13) -> list[Example]:
    """spec: 'seed' | a public dataset name | a path to a .jsonl file | comma-separated mix."""
    rows: list[Example] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if part == "seed":
            rows += load_jsonl(SEED_PATH)
        elif part == "public":
            for name in PUBLIC:
                rows += load_public(name, n, seed)
        elif part in PUBLIC:
            rows += load_public(part, n, seed)
        elif Path(part).exists():
            rows += load_jsonl(part)
        else:
            raise ValueError(f"unknown dataset: {part}")
    return rows
