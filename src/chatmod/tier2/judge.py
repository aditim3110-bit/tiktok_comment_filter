"""LLM judge interface, rate limiting + retry/timeout wrapper, and an offline fake judge."""

from __future__ import annotations

import asyncio
import random
import threading
import time
from typing import Protocol

from pydantic import ValidationError

from chatmod.schema import JudgeOutput, Label, Message, Tier1Result, Tier2Result
from chatmod.tier2.prompt import build_system, build_user


class RawJudgeResponse:
    def __init__(self, text: str, input_tokens: int = 0, output_tokens: int = 0):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class LLMBackend(Protocol):
    """One raw call to a model. Adapters (Gemini, Claude, ...) implement this."""

    name: str

    async def complete(self, system: str, user: str) -> RawJudgeResponse: ...


# ---- backend-agnostic errors: adapters translate provider errors into these ----------

class RateLimited(Exception):
    """Short-term limit hit (e.g. requests/minute). ``retry_after`` = server-suggested wait."""

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class TransientError(RateLimited):
    """Overloaded / 5xx: worth retrying with backoff."""


class BadResponse(Exception):
    """The model answered but gave no usable verdict (truncated, safety-blocked, ...).
    Not retried: the same deterministic request would fail the same way."""


class QuotaExhausted(Exception):
    """Long-term quota (e.g. requests/day) is gone. Retrying is pointless until it resets."""


class RateLimiter:
    """Client-side requests-per-minute pacing, safe to share across event loops.

    Slots are reserved under a thread lock (no awaiting inside), then awaited with
    asyncio.sleep, so eval runs that call asyncio.run() once per mode can share it.
    ``penalize`` pushes every future slot back when the server says "retry in Ns".
    """

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    async def acquire(self, max_wait: float = float("inf")) -> None:
        if not self.interval and self._next <= time.monotonic():
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            if slot - now > max_wait:
                raise RateLimited(f"client rate limit: next slot in {slot - now:.0f}s", retry_after=slot - now)
            self._next = slot + self.interval
        if slot > now:
            await asyncio.sleep(slot - now)

    def penalize(self, delay: float) -> None:
        with self._lock:
            self._next = max(self._next, time.monotonic() + delay)


class LLMJudge:
    """Builds the prompt, calls a backend with pacing, timeout and retries, validates the JSON.

    Retry policy:
    - invalid JSON / schema: retried once.
    - RateLimited / TransientError: retried up to ``max_retries`` times, waiting the
      server's retry-after (else exponential backoff with jitter), but only while the wait
      is <= ``max_wait_s``. Live traffic uses a small max_wait (fail fast into the review
      queue); evals use a large one (just wait it out).
    - QuotaExhausted: no retry, and every later call fails immediately without hitting
      the API, so a run doesn't burn requests against an exhausted daily quota.
    - Timeouts: retried only when ``retry_timeouts`` (evals).
    """

    def __init__(self, backend: LLMBackend, policy: str, timeout_s: float = 8.0, cache=None, *,
                 limiter: RateLimiter | None = None, max_retries: int = 3, max_wait_s: float = 2.0,
                 retry_timeouts: bool = False):
        self.backend = backend
        self.system = build_system(policy)
        self.timeout_s = timeout_s
        self.cache = cache
        self.limiter = limiter
        self.max_retries = max_retries
        self.max_wait_s = max_wait_s
        self.retry_timeouts = retry_timeouts
        self.exhausted: str | None = None

    @property
    def name(self) -> str:
        return self.backend.name

    async def judge(self, msg: Message, t1: Tier1Result | None = None) -> Tier2Result:
        user = build_user(msg, t1)
        t0 = time.perf_counter()
        if self.cache is not None:
            hit = self.cache.get(self.backend.name, self.system, user)
            if hit is not None:
                hit.cached = True
                return hit
        if self.exhausted:
            return Tier2Result(error=self.exhausted, latency_ms=_ms(t0))

        err: str | None = None
        in_tok = out_tok = 0
        invalid_left, retries_left, backoff_n = 1, self.max_retries, 0
        while True:
            try:
                if self.limiter is not None:
                    await self.limiter.acquire(self.max_wait_s)
                raw = await asyncio.wait_for(self.backend.complete(self.system, user), self.timeout_s)
                in_tok += raw.input_tokens
                out_tok += raw.output_tokens
                out = JudgeOutput.model_validate_json(_strip_fences(raw.text))
                res = Tier2Result(output=out, latency_ms=_ms(t0), input_tokens=in_tok, output_tokens=out_tok)
                if self.cache is not None:
                    self.cache.put(self.backend.name, self.system, user, res)
                return res
            except ValidationError as e:
                err = f"invalid output: {e.errors()[0].get('msg', e)}"
                if invalid_left:
                    invalid_left -= 1
                    continue
                break
            except BadResponse as e:
                err = f"bad response: {e}"
                break
            except QuotaExhausted as e:
                self.exhausted = err = f"quota exhausted: {e}"
                break
            except (RateLimited, asyncio.TimeoutError) as e:
                if isinstance(e, asyncio.TimeoutError):
                    err = f"timeout after {self.timeout_s}s"
                    if not self.retry_timeouts:
                        break
                    wait = 0.0
                else:
                    err = str(e)
                    wait = e.retry_after if e.retry_after is not None else min(60.0, 2.0 ** backoff_n)
                    if e.retry_after is not None and self.limiter is not None:
                        self.limiter.penalize(e.retry_after)
                if retries_left <= 0 or wait > self.max_wait_s:
                    break
                retries_left -= 1
                backoff_n += 1
                await asyncio.sleep(wait + random.uniform(0, 0.25 * wait + 0.05))
            except Exception as e:  # unknown network / API error: one quick retry
                err = f"{type(e).__name__}: {str(e)[:300]}"
                if retries_left <= 0:
                    break
                retries_left = 0
                await asyncio.sleep(0.5)
        return Tier2Result(error=err, latency_ms=_ms(t0), input_tokens=in_tok, output_tokens=out_tok)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.rsplit("```", 1)[0]
    return t.strip()


class FakeBackend:
    """Deterministic offline stand-in that derives a verdict from the Tier 1 signals in the
    prompt. Lets the full pipeline, API and harness run without an API key. Its numbers are
    NOT a measure of LLM quality."""

    name = "fake"

    async def complete(self, system: str, user: str) -> RawJudgeResponse:
        import json
        import re

        m = re.search(r"Automated signals \(hints, may be wrong\): (\{.*\})", user)
        sig = json.loads(m.group(1)) if m else {}
        tox = sig.get("toxicity_scores", {})
        hits = sig.get("rule_hits", [])
        label, sev, conf = Label.benign, 0, 0.8
        if "self_harm_phrase" in hits:
            label, sev, conf = Label.self_harm, 2, 0.75
        elif "encourage_self_harm" in hits:
            label, sev, conf = Label.harassment, 3, 0.85
        elif "threat_phrase" in hits:
            label, sev, conf = Label.threat_violence, 2, 0.7
        elif any(h not in ("all_caps", "flood") and not h.startswith("hint_") for h in hits):
            label, sev, conf = Label.scam_spam, 2, 0.75
        elif tox:
            head = max(tox, key=tox.get)
            top = tox[head]
            label = {"identity_attack": Label.hate, "sexual_explicit": Label.sexual, "threat": Label.threat_violence}.get(head, Label.harassment)
            if head == "obscene" and tox.get("insult", 0) < 0.3:
                label = Label.benign
            sev = 0 if label is Label.benign else (2 if top > 0.6 else 1)
            conf = 0.55 + abs(top - 0.5) * 0.6
        out = JudgeOutput(label=label, severity=sev, confidence=round(conf, 2),
                          rationale="fake judge: derived from tier1 signals",
                          user_message="" if label is Label.benign else "This message breaks the chat rules.")
        return RawJudgeResponse(out.model_dump_json(), input_tokens=len(system + user) // 4, output_tokens=60)
