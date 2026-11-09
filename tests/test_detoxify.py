"""Integration tests against the real Detoxify model. Skipped if the [ml] extra is not installed."""

import pytest

pytest.importorskip("detoxify")

from chatmod.tier1 import DetoxifyScorer  # noqa: E402


@pytest.fixture(scope="module")
def scorer():
    return DetoxifyScorer("unbiased")


def test_single_text_batch(scorer):
    # Detoxify returns bare floats for a one-item batch; this is the API's common path.
    (s,) = scorer.score(["you are a worthless idiot"])
    assert s["insult"] > 0.5 and set(s) >= {"toxicity", "insult", "identity_attack"}


def test_multi_text_batch(scorer):
    a, b = scorer.score(["hello chat, love the stream", "you are a worthless idiot"])
    assert a["toxicity"] < 0.1 < b["toxicity"]
