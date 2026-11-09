"""Gemini Flash backend using the google-genai SDK with schema-constrained JSON output.

Auth: set GEMINI_API_KEY (or GOOGLE_API_KEY). Install with: pip install -e '.[llm]'
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from chatmod.schema import Label
from chatmod.tier2.judge import BadResponse, QuotaExhausted, RateLimited, RawJudgeResponse, TransientError


def _parse_duration(s: Any) -> float | None:
    m = re.fullmatch(r"\s*([\d.]+)s\s*", str(s)) if s is not None else None
    return float(m.group(1)) if m else None


def classify_api_error(code: int | None, payload: Any, text: str = "") -> Exception | None:
    """Map a Gemini API error to RateLimited / TransientError / QuotaExhausted (else None).

    Google's 429 body carries google.rpc.QuotaFailure (which quota: per-minute vs per-day)
    and google.rpc.RetryInfo (retryDelay, e.g. "50s").
    """
    err = payload.get("error", payload) if isinstance(payload, dict) else {}
    details = err.get("details") or [] if isinstance(err, dict) else []
    retry_after = None
    quota_ids: list[str] = []
    for d in details:
        if not isinstance(d, dict):
            continue
        if d.get("@type", "").endswith("RetryInfo"):
            retry_after = _parse_duration(d.get("retryDelay"))
        for v in d.get("violations", []) or []:
            if isinstance(v, dict) and v.get("quotaId"):
                quota_ids.append(v["quotaId"])
    if retry_after is None:
        m = re.search(r"retry in ([\d.]+)s", text)
        retry_after = float(m.group(1)) if m else None

    if code == 429:
        if any("PerDay" in q for q in quota_ids):
            limit = re.search(r"limit: (\d+)", text)
            return QuotaExhausted(f"429 daily quota ({', '.join(quota_ids)}"
                                  f"{', limit ' + limit.group(1) if limit else ''})")
        what = ", ".join(quota_ids) or "rate limit"
        return RateLimited(f"429 {what}" + (f" (retry in {retry_after:.0f}s)" if retry_after else ""), retry_after)
    if code in (500, 502, 503, 504):
        return TransientError(f"{code} {err.get('status', '') if isinstance(err, dict) else ''}".strip(), retry_after)
    return None


class _WireVerdict(BaseModel):
    """Response schema sent to Gemini. Kept free of numeric constraints for schema
    compatibility; the result is re-validated against JudgeOutput (with bounds) afterwards."""

    label: Label
    severity: int
    confidence: float
    rationale: str
    user_message: str


_TEXT_HARM_CATEGORIES = ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                         "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]


def check_response(resp: Any) -> None:
    """Raise BadResponse when Gemini returned no usable verdict, naming the reason, instead
    of letting it surface later as an opaque JSON parse error."""
    feedback = getattr(resp, "prompt_feedback", None)
    block = getattr(feedback, "block_reason", None)
    if block:
        raise BadResponse(f"prompt blocked by Gemini ({_enum_name(block)})")
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        raise BadResponse("no candidates returned")
    reason = _enum_name(getattr(cands[0], "finish_reason", None))
    usage = getattr(resp, "usage_metadata", None)
    thoughts = getattr(usage, "thoughts_token_count", 0) or 0
    if reason == "MAX_TOKENS":
        raise BadResponse(f"truncated at max_output_tokens ({thoughts} thinking tokens used); "
                          f"raise tier2.max_output_tokens or lower tier2.thinking_level")
    if reason not in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
        ratings = [f"{_enum_name(r.category)}={_enum_name(r.probability)}"
                   for r in (getattr(cands[0], "safety_ratings", None) or []) if getattr(r, "blocked", False)]
        raise BadResponse(f"blocked by safety filter ({reason}{': ' + ', '.join(ratings) if ratings else ''})"
                          if reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII")
                          else f"response ended with finish_reason={reason}")


def _enum_name(v: Any) -> str | None:
    return None if v is None else getattr(v, "name", str(v))


class GeminiBackend:
    def __init__(self, model: str = "gemini-3.8-flash", temperature: float = 0.0,
                 thinking_budget: int | None = None, api_key: str | None = None, *,
                 thinking_level: str | None = "low", max_output_tokens: int = 2048,
                 safety_threshold: str | None = "BLOCK_NONE"):
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("google-genai is not installed. Run: pip install -e '.[llm]'") from e
        self._types = types
        self._errors = errors
        # Disable the SDK's own retries: LLMJudge owns retry/backoff so it can honour
        # retryDelay, respect the latency budget and stop on daily-quota exhaustion.
        http = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
        kw: dict[str, Any] = {"http_options": http}
        if api_key:
            kw["api_key"] = api_key
        self._client = genai.Client(**kw)
        self.model = model
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.safety_threshold = safety_threshold
        # The name is part of the verdict-cache key, so include settings that change answers.
        thinking = f"level={thinking_level}" if thinking_level else f"budget={thinking_budget}"
        self.name = f"gemini:{model}:{thinking}"

    async def complete(self, system: str, user: str) -> RawJudgeResponse:
        types = self._types
        cfg: dict[str, Any] = dict(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=_WireVerdict,
            # Thinking tokens count against this limit; too low a cap truncates the JSON.
            max_output_tokens=self.max_output_tokens,
        )
        # Gemini 3 models use thinking_level; older ones use thinking_budget. Never send both.
        if self.thinking_level:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level.upper())
        elif self.thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        if self.safety_threshold:
            # A moderation judge must be able to read the abuse it classifies; the default
            # filters can return an empty response for exactly the messages we care about.
            cfg["safety_settings"] = [types.SafetySetting(category=c, threshold=self.safety_threshold)
                                      for c in _TEXT_HARM_CATEGORIES]
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=user, config=types.GenerateContentConfig(**cfg)
            )
        except self._errors.APIError as e:
            mapped = classify_api_error(getattr(e, "code", None), getattr(e, "details", None), str(e))
            if mapped is None:
                raise
            raise mapped from e
        check_response(resp)
        usage = getattr(resp, "usage_metadata", None)
        return RawJudgeResponse(
            text=resp.text or "",
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=(getattr(usage, "candidates_token_count", 0) or 0)
            + (getattr(usage, "thoughts_token_count", 0) or 0),
        )
