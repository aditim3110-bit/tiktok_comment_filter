import pytest

pytest.importorskip("google.genai")

from google.genai import types  # noqa: E402

from chatmod.schema import Message  # noqa: E402
from chatmod.tier2 import BadResponse, LLMJudge  # noqa: E402
from chatmod.tier2.gemini import GeminiBackend, check_response  # noqa: E402


def _resp(finish=None, text='{"x": 1}', thoughts=0, block=None, ratings=None):
    cand = types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]) if text else None,
        finish_reason=finish, safety_ratings=ratings)
    return types.GenerateContentResponse(
        candidates=[cand],
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block) if block else None,
        usage_metadata=types.GenerateContentResponseUsageMetadata(thoughts_token_count=thoughts))


def test_stop_is_fine():
    check_response(_resp(types.FinishReason.STOP))


def test_max_tokens_is_reported_with_thinking_usage():
    with pytest.raises(BadResponse, match=r"truncated at max_output_tokens \(380 thinking tokens"):
        check_response(_resp(types.FinishReason.MAX_TOKENS, text='{"label": "ha', thoughts=380))


def test_safety_block_names_category():
    rating = types.SafetyRating(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                probability=types.HarmProbability.HIGH, blocked=True)
    with pytest.raises(BadResponse, match="blocked by safety filter.*HATE_SPEECH=HIGH"):
        check_response(_resp(types.FinishReason.SAFETY, text=None, ratings=[rating]))


def test_prompt_block():
    with pytest.raises(BadResponse, match="prompt blocked"):
        check_response(_resp(text=None, block=types.BlockedReason.SAFETY))


def test_request_config_validates_with_sdk():
    b = GeminiBackend(api_key="dummy", thinking_level="low", max_output_tokens=2048, safety_threshold="BLOCK_NONE")
    cfg = types.GenerateContentConfig(
        max_output_tokens=b.max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_level=b.thinking_level.upper()),
        safety_settings=[types.SafetySetting(category=c, threshold=b.safety_threshold)
                         for c in ("HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_HARASSMENT")])
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert "level=low" in b.name  # settings are part of the cache key


async def test_bad_response_is_not_retried():
    class B:
        name = "b"
        calls = 0

        async def complete(self, s, u):
            B.calls += 1
            raise BadResponse("truncated at max_output_tokens (400 thinking tokens used)")

    r = await LLMJudge(B(), "p", max_wait_s=100).judge(Message(text="x"))
    assert B.calls == 1 and r.error.startswith("bad response: truncated")
