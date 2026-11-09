"""Orchestrates Tier 1 -> Tier 2 -> Tier 3."""

from __future__ import annotations

import asyncio
from enum import Enum

from chatmod.config import Config, load_config
from chatmod.schema import Action, Decision, Label, Message, Route, Tier1Result, Tier2Result
from chatmod.tier1 import Tier1, get_scorer
from chatmod.tier2 import FakeBackend, JudgeCache, LLMJudge, RateLimiter
from chatmod.tier3 import ReviewQueue


class Mode(str, Enum):
    full = "full"
    tier1_only = "tier1-only"   # no LLM: unsettled messages get Tier 1's provisional guess
    tier2_only = "tier2-only"   # LLM on every message (quality/cost upper-bound baseline)


_RISK_LABEL = {"self_harm_phrase": Label.self_harm, "encourage_self_harm": Label.harassment,
               "threat_phrase": Label.threat_violence}
_HEAD_LABEL = {"identity_attack": Label.hate, "sexual_explicit": Label.sexual, "threat": Label.threat_violence,
               "insult": Label.harassment, "severe_toxicity": Label.harassment, "toxicity": Label.harassment}


def provisional(t1: Tier1Result) -> tuple[Label, int]:
    """Tier 1's best guess for a message it could not settle (used in tier1-only mode)."""
    if t1.settled and t1.label is not None:
        return t1.label, t1.severity
    for hit in t1.rule_hits:
        if hit in _RISK_LABEL:
            return _RISK_LABEL[hit], 2
    if t1.rule_score >= 1.0:
        return Label.scam_spam, 1
    heads = {h: v for h, v in t1.tox_scores.items() if h in _HEAD_LABEL}
    if heads:
        head = max(heads, key=heads.get)
        if heads[head] >= 0.5:
            return _HEAD_LABEL[head], 1
    return Label.benign, 0


class Pipeline:
    def __init__(self, cfg: Config, tier1: Tier1, judge: LLMJudge | None, queue: ReviewQueue | None = None):
        self.cfg = cfg
        self.tier1 = tier1
        self.judge = judge
        self.queue = queue
        self._sems: dict[int, asyncio.Semaphore] = {}

    def _sem(self) -> asyncio.Semaphore:
        # One semaphore per event loop: eval runs call asyncio.run() once per mode.
        loop_id = id(asyncio.get_running_loop())
        if loop_id not in self._sems:
            self._sems = {loop_id: asyncio.Semaphore(self.cfg.tier2.max_concurrency)}
        return self._sems[loop_id]

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Config | None = None, *, scorer: str | None = None, provider: str | None = None,
                    with_queue: bool = True, use_cache: bool = True, patient: bool = False) -> "Pipeline":
        cfg = cfg or load_config()
        t1 = Tier1(get_scorer(scorer or cfg.tier1.scorer, cfg.tier1.detoxify_model), cfg.tier1)
        judge = build_judge(cfg, provider, use_cache, patient=patient)
        queue = ReviewQueue(cfg.resolve(cfg.tier3.db_path)) if with_queue else None
        return cls(cfg, t1, judge, queue)

    # ---- public API ---------------------------------------------------------------------
    async def moderate(self, msg: Message, mode: Mode = Mode.full) -> Decision:
        return (await self.moderate_batch([msg], mode))[0]

    async def moderate_batch(self, msgs: list[Message], mode: Mode = Mode.full) -> list[Decision]:
        t1s = self.tier1.classify_batch([m.text for m in msgs])
        tasks = [self._route(m, t1, mode) for m, t1 in zip(msgs, t1s)]
        return list(await asyncio.gather(*tasks))

    # ---- routing ------------------------------------------------------------------------
    async def _route(self, msg: Message, t1: Tier1Result, mode: Mode) -> Decision:
        if mode is not Mode.tier2_only and t1.settled:
            return self._decision(msg, t1.label, t1.severity, t1.confidence, Route.tier1, t1,
                                  rationale=f"tier1: {t1.reason}")

        if mode is Mode.tier1_only or self.judge is None:
            label, sev = provisional(t1)
            return self._decision(msg, label, sev, 0.5, Route.tier1, t1,
                                  rationale=f"tier1 provisional: {t1.reason}", signals={"unsettled": True})

        async with self._sem():
            t2 = await self.judge.judge(msg, None if mode is Mode.tier2_only else t1)
        return self._after_tier2(msg, t1, t2)

    def _after_tier2(self, msg: Message, t1: Tier1Result, t2: Tier2Result) -> Decision:
        cfg = self.cfg.tier2
        out = t2.output
        if out is None:
            # LLM failed: fail safe by soft-hiding pending human review.
            label, sev = provisional(t1)
            d = self._decision(msg, label, sev, 0.0, Route.tier3, t1, t2,
                               rationale=f"tier2 error: {t2.error}", action=Action.hide)
            return self._enqueue(d, f"llm error: {t2.error}", msg)

        reasons = []
        if out.confidence < cfg.conf_thr:
            reasons.append(f"low confidence {out.confidence:.2f}")
        if out.label in cfg.always_review and out.severity > 0:
            reasons.append(f"{out.label.value} always reviewed")

        if not reasons:
            return self._decision(msg, out.label, out.severity, out.confidence, Route.tier2, t1, t2,
                                  rationale=out.rationale, user_message=out.user_message)

        final = self.cfg.action_for(out.label, out.severity)
        pending_action = Action.allow if final is Action.allow else Action.hide
        d = self._decision(msg, out.label, out.severity, out.confidence, Route.tier3, t1, t2,
                           rationale=out.rationale, user_message=out.user_message, action=pending_action)
        return self._enqueue(d, "; ".join(reasons), msg)

    def _enqueue(self, d: Decision, reason: str, msg: Message) -> Decision:
        d.needs_review = True
        d.signals["review_reason"] = reason
        if self.queue is not None:
            d.review_id = self.queue.enqueue(d, reason, msg.context)
        return d

    def _decision(self, msg: Message, label: Label, sev: int, conf: float, route: Route,
                  t1: Tier1Result, t2: Tier2Result | None = None, *, rationale: str = "",
                  user_message: str = "", action: Action | None = None, signals: dict | None = None) -> Decision:
        latency = t1.latency_ms + (t2.latency_ms if t2 else 0.0)
        return Decision(
            message_id=msg.id, text=msg.text, label=label, severity=sev, confidence=round(conf, 4),
            action=action or self.cfg.action_for(label, sev), route=route, rationale=rationale,
            user_message=user_message or _default_user_message(label, sev), latency_ms=round(latency, 2),
            tier1=t1, tier2=t2, signals=signals or {},
        )


_DEFAULT_MESSAGES = {
    Label.scam_spam: "Links, outside contact info and giveaway offers aren't allowed in chat. This helps protect viewers from scams.",
    Label.harassment: "Please keep chat respectful. Insults or attacks aimed at others aren't allowed.",
    Label.hate: "Comments attacking people for who they are aren't allowed here.",
    Label.sexual: "Sexual content isn't allowed in this chat.",
    Label.threat_violence: "Threats of violence aren't allowed in chat.",
    Label.self_harm: "It sounds like you might be going through a lot. You're not alone. If you're in danger, please reach out to a local crisis line.",
}


def _default_user_message(label: Label, sev: int) -> str:
    return "" if label is Label.benign or sev == 0 else _DEFAULT_MESSAGES.get(label, "")


def build_judge(cfg: Config, provider: str | None = None, use_cache: bool = True,
                patient: bool = False) -> LLMJudge | None:
    """``patient`` (evals): long timeouts, wait out rate limits, retry timeouts.
    Otherwise (serving): fail fast so a message never waits long for a verdict."""
    t2 = cfg.tier2
    provider = provider or t2.provider
    if provider == "none":
        return None
    if provider == "fake":
        backend = FakeBackend()
    elif provider == "gemini":
        from chatmod.tier2.gemini import GeminiBackend

        backend = GeminiBackend(t2.model, t2.temperature, t2.thinking_budget,
                                thinking_level=t2.thinking_level, max_output_tokens=t2.max_output_tokens,
                                safety_threshold=t2.safety_threshold)
    else:
        raise ValueError(f"unknown tier2 provider: {provider}")
    cache = JudgeCache(cfg.resolve(t2.cache_path)) if use_cache and provider != "fake" else None
    limiter = RateLimiter(t2.rpm_limit) if t2.rpm_limit > 0 and provider != "fake" else None
    return LLMJudge(backend, cfg.policy_text(), t2.eval_timeout_s if patient else t2.timeout_s, cache,
                    limiter=limiter, max_retries=t2.max_retries,
                    max_wait_s=t2.eval_max_wait_s if patient else t2.live_max_wait_s, retry_timeouts=patient)
