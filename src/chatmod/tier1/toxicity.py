"""Toxicity scorers for Tier 1.

All scorers return, per text, a dict of head -> probability using Detoxify's head names:
toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit.
Missing heads are simply absent (e.g. Detoxify ``original`` has no sexual_explicit).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

from chatmod.tier1.normalize import normalize

HEADS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]


class ToxicityScorer(Protocol):
    name: str

    def score(self, texts: list[str]) -> list[dict[str, float]]: ...


class DetoxifyScorer:
    """Wraps the `detoxify` package (pip install 'chatmod[ml]'). CPU is fine for chat-length text."""

    def __init__(self, model: str = "unbiased", device: str = "cpu", batch_size: int = 64):
        try:
            from detoxify import Detoxify
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Detoxify is not installed. Run: pip install -e '.[ml]'") from e
        self.name = f"detoxify-{model}"
        self.batch_size = batch_size
        self._model = Detoxify(model, device=device)

    def score(self, texts: list[str]) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            res = self._model.predict(batch)
            # Detoxify returns {head: [scores...]}; some versions name the sexual head differently.
            res = {("sexual_explicit" if k == "sexually_explicit" else k): v for k, v in res.items()}
            # Detoxify returns bare floats (not lists) when the batch has a single text.
            res = {k: (v if isinstance(v, (list, tuple)) or hasattr(v, "__len__") else [v]) for k, v in res.items()}
            for j in range(len(batch)):
                out.append({k: float(v[j]) for k, v in res.items() if k in HEADS})
        return out


class KeywordScorer:
    """Tiny lexicon-based stand-in with Detoxify's output shape.

    It's for tests, CI and running the harness without torch. It is deliberately crude,
    so don't use it as a production Tier 1.
    """

    name = "keyword"

    _LEX: dict[str, list[str]] = {
        "insult": [
            r"idiot", r"stupid", r"loser", r"moron", r"dumb", r"pathetic", r"worthless", r"ugly",
            r"clown", r"trash\s+(?:human|person)", r"nobody\s+likes\s+you", r"you\s+suck", r"shut\s+up",
            r"fat", r"disgusting", r"freak", r"bitch", r"retard\w*", r"kys",
        ],
        "obscene": [r"fuck\w*", r"shit\w*", r"damn", r"ass(?:hole)?", r"bitch", r"crap", r"wtf"],
        "threat": [
            r"(?:i'?ll|i\s+will|gonna|going\s+to)\s+(?:kill|hurt|stab|shoot|beat|find)\s+you",
            r"i\s+know\s+where\s+you\s+live", r"watch\s+your\s+back", r"you(?:'re|\s+are)\s+dead",
        ],
        "sexual_explicit": [r"nudes?", r"naked", r"sex\w*", r"horny", r"boobs?", r"dick", r"porn\w*", r"onlyfans"],
        "identity_attack": [
            r"all\s+(?:\w+\s+)?(?:immigrants|muslims|jews|christians|gays|women|men|blacks|whites|asians|mexicans|trans\s+people)\s+(?:are|should)",
            r"(?:go\s+back\s+to\s+your\s+country)", r"(?:women|girls)\s+(?:can'?t|shouldn'?t)\s+(?:game|play|stream)",
            r"(?:gays?|trans)\s+(?:are|is)\s+(?:a\s+)?(?:disease|sick|mental)",
        ],
    }
    _COMPILED = {h: [re.compile(rf"\b{p}\b") for p in ps] for h, ps in _LEX.items()}

    def score(self, texts: list[str]) -> list[dict[str, float]]:
        return [self._score_one(t) for t in texts]

    def _score_one(self, text: str) -> dict[str, float]:
        t = normalize(text).deobf
        heads = {h: 0.02 for h in HEADS}
        for head, pats in self._COMPILED.items():
            n = sum(1 for p in pats if p.search(t))
            if n:
                heads[head] = min(0.99, 0.6 + 0.2 * n)
        targeted = bool(re.search(r"\b(?:you|ur|u|your|you'?re)\b", t))
        if heads["insult"] > 0.5 and targeted:
            heads["insult"] = min(0.99, heads["insult"] + 0.3)
        heads["toxicity"] = max(heads[h] for h in ("insult", "obscene", "threat", "identity_attack", "sexual_explicit"))
        heads["severe_toxicity"] = 0.9 if heads["toxicity"] > 0.95 and heads["obscene"] > 0.5 else 0.02
        return heads


@lru_cache(maxsize=4)
def get_scorer(kind: str = "detoxify", model: str = "unbiased") -> ToxicityScorer:
    if kind == "keyword":
        return KeywordScorer()
    if kind == "detoxify":
        return DetoxifyScorer(model)
    raise ValueError(f"unknown scorer: {kind}")
