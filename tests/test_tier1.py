from chatmod.config import Tier1Config
from chatmod.schema import Label
from chatmod.tier1 import Tier1

from conftest import StubScorer


def make(table):
    return Tier1(StubScorer(table), Tier1Config(low_thr=0.2, high_thr=0.9, scam_strong=3.0, scam_escalate=1.0))


def test_low_scores_no_rules_settles_benign():
    r = make({}).classify("what game is this?")
    assert r.settled and r.label is Label.benign and r.confidence > 0.9


def test_high_insult_settles_harassment():
    r = make({"you idiot": {"toxicity": 0.97, "insult": 0.95}}).classify("you idiot")
    assert r.settled and r.label is Label.harassment and r.severity >= 2


def test_high_identity_attack_settles_hate():
    r = make({"x": {"toxicity": 0.95, "insult": 0.6, "identity_attack": 0.93}}).classify("x")
    assert r.settled and r.label is Label.hate


def test_high_threat_alone_does_not_settle():
    # in-game "I'm gonna kill you" often scores high on threat; it must go to the LLM
    r = make({"x": {"toxicity": 0.92, "threat": 0.95}}).classify("x")
    assert not r.settled


def test_high_obscene_alone_does_not_settle():
    r = make({"x": {"toxicity": 0.93, "obscene": 0.96}}).classify("x")
    assert not r.settled


def test_middle_band_escalates():
    r = make({"x": {"toxicity": 0.5, "insult": 0.45}}).classify("x")
    assert not r.settled and "uncertain band" in r.reason


def test_strong_scam_settles():
    r = make({}).classify("FREE COINS at coinsdrop.xyz")
    assert r.settled and r.label is Label.scam_spam and r.severity == 3


def test_weak_scam_escalates():
    r = make({}).classify("dm me")
    assert not r.settled and "rule score" in r.reason


def test_risk_rule_always_escalates_even_if_scores_low():
    r = make({}).classify("I want to die")
    assert not r.settled and "self_harm_phrase" in r.reason


def test_hint_blocks_auto_allow():
    r = make({}).classify("ur voice is so annoying")
    assert not r.settled and "hint_targeted_negative" in r.reason


def test_identity_head_wins_over_insult():
    r = make({"x": {"toxicity": 0.97, "insult": 0.97, "identity_attack": 0.7}}).classify("x")
    assert r.settled and r.label is Label.hate


def test_deobfuscated_text_is_also_scored():
    t1 = make({"you worthless idiot": {"toxicity": 0.98, "insult": 0.97}})
    r = t1.classify("y0u w0rthl3ss 1d10t")
    assert r.settled and r.label is Label.harassment


def test_batch_matches_single():
    t1 = make({"bad": {"insult": 0.95, "toxicity": 0.95}})
    batch = t1.classify_batch(["hi", "bad", "dm me"])
    assert [b.settled for b in batch] == [True, True, False]
    assert all(b.latency_ms >= 0 for b in batch)


def test_keyword_scorer_end_to_end(keyword_tier1):
    assert keyword_tier1.classify("you're a worthless idiot").label is Label.harassment
    assert keyword_tier1.classify("lets gooo").label is Label.benign
