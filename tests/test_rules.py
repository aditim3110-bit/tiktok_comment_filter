import pytest

from chatmod.tier1.normalize import normalize
from chatmod.tier1.rules import run_rules


@pytest.mark.parametrize("text,expected", [
    ("fr33 c0ins", "free coins"),
    ("t . me/abc", "t.me/abc"),
    ("site dot com", "site.com"),
    ("giftz[.]xyz", "giftz.xyz"),
    ("fгее соins", "free coins"),          # cyrillic homoglyphs
    ("fre​e coins", "free coins"),     # zero-width space
    ("loooooool", "lool"),
])
def test_normalize_deobfuscates(text, expected):
    assert expected in normalize(text).deobf


def test_normalize_keeps_plain_numbers():
    assert "2 to 1" in normalize("the score was 2 to 1").deobf
    assert normalize("+1 415 555 0132").plain == "+1 415 555 0132"


def test_normalize_does_not_glue_sentences():
    assert "thanks.me" not in normalize("thanks. me too").deobf


@pytest.mark.parametrize("text,rule", [
    ("go to coinsdrop.xyz now", "url"),
    ("https://example.org/x", "url"),
    ("bit.ly/3xYz9", "shortener"),
    ("join t.me/hotdeals", "messenger_link"),
    ("wa.me/15551234567", "messenger_link"),
    ("add me on telegram", "messenger_mention"),
    ("dm me for collab", "contact_solicit"),
    ("link in my bio", "contact_solicit"),
    ("add me on telegram @fastmoney", "handle_with_contact"),
    ("FREE COINS for everyone", "free_stuff"),
    ("free vbucks", "free_stuff"),
    ("verify your account here", "giveaway_phish"),
    ("double your bitcoin", "investment"),
    ("buy cheap followers", "sell_engagement"),
    ("text me +44 7700 900123", "phone_number"),
    ("0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE", "crypto_wallet"),
    ("check out my channel", "self_promo"),
    ("FOLLOW MY CHANNEL FOLLOW MY CHANNEL", "all_caps"),
    ("aaaaaaaaaaaaaaaa", "flood"),
    ("I want to die", "self_harm_phrase"),
    ("kys", "encourage_self_harm"),
    ("i know where you live", "threat_phrase"),
    # obfuscated variants
    ("d m m e for the l i n k", "contact_solicit"),
    ("fr33 c0ins at t . me/c0inz", "messenger_link"),
    # escalation hints
    ("all immigrants are criminals", "hint_identity"),
    ("s3nd n00dz", "hint_sexual"),
    ("her real name is X and she works at Target", "hint_personal_info"),
    ("thinking about ending it all", "hint_self_harm"),
    ("nobody likes you, stop streaming", "hint_targeted_negative"),
    ("ur voice is so annoying", "hint_targeted_negative"),
    ("wh4tsapp me", "messenger_mention"),
    ("cl@im y0ur prize", "giveaway_phish"),
])
def test_rule_fires(text, rule):
    assert rule in run_rules(text).hits


@pytest.mark.parametrize("text", [
    "gg ez", "what game is this?", "thanks. me too", "nice line, made me laugh",
    "there's a snap in the audio sometimes", "the score was 2 to 1", "I paid 100 dollars for this headset",
    "@streamer you dropped your shield", "join the discord, it's in the about section",
    "I'm gonna kill you in this round", "I'm dying of laughing", "LETS GO", "1v1 me, you won't",
])
def test_benign_messages_fire_no_scam_rules(text):
    res = run_rules(text)
    assert res.score < 1.0, res.hits
    assert not res.risk_hits, res.risk_hits


def test_scores_add_up():
    strong = run_rules("FREE COINS at coinsdrop.xyz")
    assert strong.score >= 3.0 and strong.severe
    weak = run_rules("dm me")
    assert 1.0 <= weak.score < 3.0
