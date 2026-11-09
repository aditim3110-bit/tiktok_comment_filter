"""Tier 1: settle the easy cases with toxicity scores + rules, escalate the rest.

Decision bands (all thresholds in config.tier1):
1. Any ``risk`` rule (self-harm, threats) -> escalate, always.
2. Scam rule score >= scam_strong -> settle scam_spam.
3. A *targeted* toxicity head (insult / identity_attack / sexual_explicit / severe_toxicity)
   >= high_thr -> settle harassment / hate / sexual.
   ``threat`` and ``obscene`` never settle on their own: in-game "I'll kill you" and
   untargeted swearing are usually benign and need context.
4. All heads < low_thr, rule score < scam_escalate and no hint rules -> settle benign.
5. Otherwise escalate.
"""

from __future__ import annotations

import time

from chatmod.config import Tier1Config
from chatmod.schema import Label, Tier1Result
from chatmod.tier1.normalize import normalize
from chatmod.tier1.rules import run_rules
from chatmod.tier1.toxicity import ToxicityScorer

# Head -> label for settling clear violations. Order matters only for ties.
_SETTLE_HEADS: list[tuple[str, Label]] = [
    ("identity_attack", Label.hate),
    ("sexual_explicit", Label.sexual),
    ("insult", Label.harassment),
    ("severe_toxicity", Label.harassment),
]


class Tier1:
    def __init__(self, scorer: ToxicityScorer, cfg: Tier1Config):
        self.scorer = scorer
        self.cfg = cfg

    def classify(self, text: str) -> Tier1Result:
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: list[str]) -> list[Tier1Result]:
        t0 = time.perf_counter()
        scores = self.score(texts)
        per_msg_ms = (time.perf_counter() - t0) * 1000 / max(1, len(texts))
        out = []
        for text, tox in zip(texts, scores):
            t1 = time.perf_counter()
            res = self._decide(text, tox)
            res.latency_ms = per_msg_ms + (time.perf_counter() - t1) * 1000
            out.append(res)
        return out

    def score(self, texts: list[str]) -> list[dict[str, float]]:
        """Toxicity heads per text. When de-obfuscation changes a text ("y0u 1d10t"), the
        clean version is scored too and the per-head max is kept."""
        if not texts:
            return []
        deobf = [normalize(t).deobf for t in texts]
        extra_idx = [i for i, (t, d) in enumerate(zip(texts, deobf)) if d != t.lower().strip()]
        scores = self.scorer.score(list(texts) + [deobf[i] for i in extra_idx])
        base, extra = scores[: len(texts)], scores[len(texts):]
        for i, s in zip(extra_idx, extra):
            base[i] = {h: max(v, s.get(h, 0.0)) for h, v in base[i].items()}
        return base

    def _decide(self, text: str, tox: dict[str, float]) -> Tier1Result:
        cfg = self.cfg
        rules = run_rules(text)
        base = dict(
            tox_scores={k: round(v, 4) for k, v in tox.items()},
            rule_hits=rules.hits,
            rule_score=rules.score,
            normalized_text=rules.normalized.deobf,
        )
        max_tox = max(tox.values(), default=0.0)

        if rules.risk_hits:
            return Tier1Result(settled=False, reason=f"risk rule: {','.join(rules.risk_hits)}", **base)

        if rules.score >= cfg.scam_strong:
            sev = 3 if rules.severe else 2
            conf = min(0.99, 0.6 + 0.08 * rules.score)
            return Tier1Result(settled=True, label=Label.scam_spam, severity=sev, confidence=conf,
                               reason=f"scam rules score {rules.score:.1f}", **base)

        head, label = max(_SETTLE_HEADS, key=lambda hl: tox.get(hl[0], 0.0))
        top = tox.get(head, 0.0)
        if top >= cfg.high_thr:
            # Insult usually co-fires with identity attacks / sexual content; prefer the more
            # specific category when it is clearly elevated.
            if tox.get("identity_attack", 0.0) >= 0.5:
                label = Label.hate
            elif tox.get("sexual_explicit", 0.0) >= 0.5:
                label = Label.sexual
            sev = 3 if top >= 0.98 or tox.get("severe_toxicity", 0) >= cfg.high_thr else 2
            return Tier1Result(settled=True, label=label, severity=sev, confidence=top,
                               reason=f"{head}={top:.2f} >= {cfg.high_thr}", **base)

        if max_tox < cfg.low_thr and rules.score < cfg.scam_escalate and not rules.hint_hits:
            return Tier1Result(settled=True, label=Label.benign, severity=0, confidence=1 - max_tox,
                               reason="all scores low", **base)

        why = []
        if rules.hint_hits:
            why.append(f"hints: {','.join(rules.hint_hits)}")
        if max_tox >= cfg.low_thr:
            why.append(f"max toxicity {max_tox:.2f} in uncertain band")
        if rules.score >= cfg.scam_escalate:
            why.append(f"rule score {rules.score:.1f}")
        return Tier1Result(settled=False, reason="; ".join(why), **base)
