import pytest

from chatmod.config import Config, Tier1Config, Tier2Config, load_config
from chatmod.tier1 import KeywordScorer, Tier1


class StubScorer:
    """Returns fixed head scores per text (default: all low)."""

    name = "stub"

    def __init__(self, table: dict[str, dict[str, float]] | None = None):
        self.table = table or {}

    def score(self, texts):
        base = {"toxicity": 0.01, "severe_toxicity": 0.0, "obscene": 0.0, "threat": 0.0,
                "insult": 0.0, "identity_attack": 0.0, "sexual_explicit": 0.0}
        return [{**base, **self.table.get(t, {})} for t in texts]


@pytest.fixture
def cfg() -> Config:
    c = load_config()
    c.tier1 = Tier1Config(scorer="keyword")
    c.tier2 = Tier2Config(provider="fake", conf_thr=0.7)
    return c


@pytest.fixture
def keyword_tier1(cfg) -> Tier1:
    return Tier1(KeywordScorer(), cfg.tier1)
