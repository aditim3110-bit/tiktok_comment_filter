import asyncio
import json
import time

import pytest

from chatmod.eval.report import llm_warnings
from chatmod.schema import Message
from chatmod.tier2 import LLMJudge, QuotaExhausted, RateLimited, RateLimiter, RawJudgeResponse, TransientError
from chatmod.tier2.gemini import classify_api_error

OK = json.dumps({"label": "benign", "severity": 0, "confidence": 0.9, "rationale": "r", "user_message": ""})


def _payload(quota_id: str, delay: str = "50s"):
    # Shape copied from a real gemini-3.8-flash free-tier 429 (2026-11-10).
    return {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
             "quotaId": quota_id}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}}


def test_classify_per_minute_429():
    e = classify_api_error(429, _payload("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"))
    assert type(e) is RateLimited and e.retry_after == 50.0


def test_classify_per_day_429():
    e = classify_api_error(429, _payload("GenerateRequestsPerDayPerProjectPerModel-FreeTier"), "limit: 20, model")
    assert isinstance(e, QuotaExhausted) and "limit 20" in str(e)


def test_classify_retry_delay_from_message_text():
    e = classify_api_error(429, {"error": {"details": []}}, "Please retry in 12.5s.")
    assert isinstance(e, RateLimited) and e.retry_after == 12.5


def test_classify_5xx_and_other():
    assert isinstance(classify_api_error(503, {"error": {"status": "UNAVAILABLE"}}), TransientError)
    assert classify_api_error(400, {"error": {"status": "INVALID_ARGUMENT"}}) is None


class Scripted:
    name = "scripted"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    async def complete(self, system, user):
        self.calls += 1
        r = self.responses.pop(0) if self.responses else OK
        if isinstance(r, BaseException):
            raise r
        if r == "SLOW":
            await asyncio.sleep(1)
        return RawJudgeResponse(r)


async def test_rate_limited_then_success_honours_retry_after():
    b = Scripted(RateLimited("429", retry_after=0.05), OK)
    t0 = time.monotonic()
    r = await LLMJudge(b, "p", max_wait_s=1.0).judge(Message(text="x"))
    assert r.output is not None and b.calls == 2 and time.monotonic() - t0 >= 0.05


async def test_live_mode_fails_fast_when_wait_too_long():
    b = Scripted(RateLimited("429 (retry in 50s)", retry_after=50))
    t0 = time.monotonic()
    r = await LLMJudge(b, "p", max_wait_s=2.0).judge(Message(text="x"))
    assert r.error and "429" in r.error and b.calls == 1 and time.monotonic() - t0 < 1


async def test_transient_uses_backoff_and_gives_up_after_max_retries():
    b = Scripted(*[TransientError("503")] * 5)
    r = await LLMJudge(b, "p", max_retries=2, max_wait_s=0.0).judge(Message(text="x"))
    # backoff 2**0 = 1s > max_wait 0 -> gives up after the first attempt
    assert r.error == "503" and b.calls == 1
    b2 = Scripted(TransientError("503", retry_after=0.01), TransientError("503", retry_after=0.01), OK)
    r2 = await LLMJudge(b2, "p", max_retries=2, max_wait_s=1.0).judge(Message(text="x"))
    assert r2.output is not None and b2.calls == 3


async def test_quota_exhausted_stops_all_further_calls():
    b = Scripted(QuotaExhausted("429 daily quota"))
    j = LLMJudge(b, "p", max_wait_s=100)
    r1 = await j.judge(Message(text="a"))
    r2 = await j.judge(Message(text="b"))
    assert "quota exhausted" in r1.error and r2.error == r1.error and b.calls == 1


async def test_patient_mode_retries_timeouts():
    b = Scripted("SLOW", OK)
    r = await LLMJudge(b, "p", timeout_s=0.05, retry_timeouts=True, max_wait_s=5).judge(Message(text="x"))
    assert r.output is not None and b.calls == 2


async def test_rate_limiter_spaces_requests():
    lim = RateLimiter(rpm=600)  # one request per 0.1s
    t0 = time.monotonic()
    for _ in range(3):
        await lim.acquire()
    assert time.monotonic() - t0 >= 0.19


async def test_rate_limiter_penalty_and_fail_fast():
    lim = RateLimiter(rpm=0)
    lim.penalize(30)
    with pytest.raises(RateLimited):
        await lim.acquire(max_wait=1.0)


async def test_server_retry_after_pushes_back_shared_limiter():
    lim = RateLimiter(rpm=6000)
    b = Scripted(RateLimited("429", retry_after=0.2), OK)
    j = LLMJudge(b, "p", limiter=lim, max_wait_s=1.0)
    t0 = time.monotonic()
    await j.judge(Message(text="x"))
    assert time.monotonic() - t0 >= 0.2


def test_report_warns_when_llm_calls_failed():
    ok = {"routing": {"llm_errors": 0, "llm_calls": 10}}
    bad = {"routing": {"llm_errors": 7, "llm_calls": 10, "llm_error_kinds": {"rate limited (429)": 7}}}
    w = llm_warnings({"tier1-only": ok, "full": bad})
    assert len(w) == 1 and w[0].startswith("full: 7/10") and "NOT valid" in w[0]
