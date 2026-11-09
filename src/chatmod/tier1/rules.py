"""Cheap regex rules for scams, spam and high-risk phrases.

Rule kinds:
- ``scam``: off-platform contact, links, fake giveaways, crypto. Weights add up; a high
  total settles the message as scam_spam in Tier 1, a lower one escalates.
- ``spam``: low-grade noise (floods, all caps). Low weight; on their own they rarely
  reach the escalation threshold.
- ``risk``: self-harm or threat phrases that the toxicity model is poor at. Any risk hit
  always escalates to the LLM, regardless of toxicity scores.
- ``hint``: subtle-harm cues (identity mentions, sexual terms, personal info, put-downs
  aimed at someone). A hint never decides a label; it only blocks Tier 1 from settling
  the message as benign.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from chatmod.tier1.normalize import Normalized, normalize

Kind = Literal["scam", "spam", "risk", "hint"]

_TLD = r"(?:com|net|org|io|me|ly|gg|co|xyz|top|link|site|online|info|app|shop|ru|cn|tk|us|live|tv|biz|click|win|vip)"

URL = re.compile(rf"(?:https?://|www\.)\S+|\b[a-z0-9-]{{2,}}\.{_TLD}\b(?:/\S*)?")
SHORTENER = re.compile(r"\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|cutt\.ly|rb\.gy|is\.gd|shorturl\.at|tiny\.cc)\b")
MESSENGER_LINK = re.compile(r"\b(?:t\.me|telegram\.me|wa\.me|chat\.whatsapp\.com|discord\.gg|discord\.com/invite)/\S+")
# Only unambiguous app names: "line", "signal", "snap", "discord" are too common in normal
# chat (streamers advertise their own discord). Links to them are covered by MESSENGER_LINK.
MESSENGER = re.compile(r"\b(?:telegram|telegrm|whatsapp|whats\s?app|watsapp|snapchat|kik|wechat|signal\s+app)\b")
CONTACT = re.compile(
    r"\b(?:dm|pm)\s*me\b|\b(?:inbox|message|msg|text|contact|hmu|hit\s+me\s+up|add|reach|find)\s+me\b"
    r"|\bmy\s+(?:dms?|inbox)\b|\bcheck\s+(?:my\s+)?(?:dms?|inbox|bio)\b|\blink\s+in\s+(?:my\s+)?bio\b"
)
HANDLE = re.compile(r"(?<!\w)@[a-z0-9_.]{3,}")
FREE_STUFF = re.compile(
    r"\bfree\s+(?:\w+\s+)?(?:coins?|gifts?|diamonds?|followers?|likes?|views?|robux|v-?bucks|gems?|skins?|money|cash|crypto|nitro)\b"
)
GIVEAWAY = re.compile(r"\b(?:giveaway|give\s?away|claim\s+(?:your|ur|now)|you(?:'ve|\s+have)?\s+won(?!['’]t)|winner|verify\s+(?:your|ur)\s+account|account\s+(?:will\s+be\s+)?(?:suspended|banned))\b")
INVEST = re.compile(r"\b(?:double\s+your|guaranteed\s+(?:profit|returns?)|investment\s+plan|forex|crypto\s+signals?|passive\s+income|earn\s+\$?\d+\S*\s+(?:a|per)\s+(?:day|week|hour))\b")
SELL = re.compile(r"\b(?:buy|cheap|sell(?:ing)?)\s+(?:\w+\s+)?(?:followers|views|likes|accounts?)\b")
PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")
WALLET = re.compile(r"\b0x[a-f0-9]{40}\b|\b(?:bc1|[13])[a-hj-np-z0-9]{25,39}\b")
SELF_PROMO = re.compile(r"\b(?:check\s+out|sub(?:scribe)?\s+to|follow)\s+my\s+(?:channel|page|stream|profile|account)\b")

SELF_HARM = re.compile(
    r"\b(?:kill(?:ing)?\s+myself|kms|end\s+(?:it\s+all|my\s+life)|suicid\w*|want\s+to\s+die|wanna\s+die"
    r"|cut(?:ting)?\s+(?:myself|again)|self[\s-]?harm|don'?t\s+want\s+to\s+(?:live|be\s+here))\b"
)
ENCOURAGE_SELF_HARM = re.compile(r"\b(?:kys|kill\s+(?:your|ur)\s?self|go\s+die|unalive\s+(?:your|ur)\s?self)\b")
THREAT = re.compile(
    r"\b(?:i\s+know\s+where\s+you\s+live|(?:i'?ll|i\s+will|gonna|going\s+to)\s+(?:find|hurt|stab|shoot)\s+you"
    r"|see\s+you\s+(?:irl|outside)|your\s+(?:home\s+)?address\s+is|watch\s+your\s+back)\b"
)

# --- escalation hints: subtle harm the toxicity model scores low. They never decide a
# label; they only stop Tier 1 from auto-allowing, so the LLM takes a look.
IDENTITY = re.compile(
    r"\b(?:immigrants?|refugees?|muslims?|islam|jews?|jewish|christians?|hindus?|sikhs?|black(?:s| people)|white(?:s| people)"
    r"|asians?|mexicans?|latin[oa]s?|africans?|arabs?|indians?|chinese|gays?|lesbians?|queer|trans(?:gender)?|lgbt\w*"
    r"|women|girls|females?|disabled|autis\w*|retard\w*|1488|14 words)\b"
)
SEXUAL_HINT = re.compile(r"\b(?:nudes?|noodz?|naked|body|thighs?|boobs?|ass|feet\s+pics?|sexy|hot\s+pics?|onlyfans|what\s+she\s+said|sounded\s+wrong|ayo)\b")
PERSONAL_INFO = re.compile(r"\b(?:real\s+name|works?\s+at|lives?\s+(?:at|on|in)|home\s+address|her\s+address|his\s+address|phone\s+number\s+is|goes\s+to\s+\w+\s+(?:school|high))\b")
SELF_HARM_HINT = re.compile(
    r"\b(?:ending\s+it|end\s+it|no\s+reason\s+to\s+live|haven'?t\s+eaten|starv\w*|purg\w*|hurt\s+myself|hate\s+myself"
    r"|can'?t\s+go\s+on|goodbye\s+(?:everyone|chat)|last\s+stream)\b"
)
TARGETED_NEG = re.compile(
    r"\b(?:you|ur|u|you'?re|your|streamer|this\s+guy|this\s+girl|she|he)\b.{0,40}"
    r"\b(?:annoying|cringe|bad|trash|garbage|ugly|stupid|dumb|worthless|useless|pathetic|loser|quit|leave|shut\s+up|nobody|proud|talented|disgusting|weird|fake)\b"
    r"|\b(?:nobody|no\s+one)\s+(?:likes|cares|watches)\b|\bimagine\s+being\b|\bwho\s+(?:even\s+)?watches\b|\bjust\s+quit\b|\bstop\s+streaming\b"
)

_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")


@dataclass(frozen=True)
class Rule:
    id: str
    kind: Kind
    weight: float
    check: Callable[[Normalized], bool]
    severe: bool = False  # implies severity 3 when the scam total settles in Tier 1


def _re(pattern: re.Pattern[str], view: str = "deobf") -> Callable[[Normalized], bool]:
    return lambda n: bool(pattern.search(getattr(n, view)))


def _caps(n: Normalized) -> bool:
    letters = [c for c in n.original if c.isalpha()]
    return len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def _flood(n: Normalized) -> bool:
    return bool(re.search(r"(.)\1{9,}", n.plain)) or len(_EMOJI.findall(n.original)) >= 12


def _handle_with_contact(n: Normalized) -> bool:
    return bool(HANDLE.search(n.plain)) and bool(MESSENGER.search(n.deobf) or CONTACT.search(n.deobf))


RULES: list[Rule] = [
    Rule("url", "scam", 2.0, lambda n: bool(URL.search(n.deobf))),
    Rule("shortener", "scam", 1.0, _re(SHORTENER)),
    Rule("messenger_link", "scam", 3.0, _re(MESSENGER_LINK)),
    Rule("messenger_mention", "scam", 1.0, _re(MESSENGER)),
    Rule("contact_solicit", "scam", 1.5, _re(CONTACT)),
    Rule("handle_with_contact", "scam", 1.0, _handle_with_contact),
    Rule("free_stuff", "scam", 2.0, _re(FREE_STUFF), severe=True),
    Rule("giveaway_phish", "scam", 1.5, _re(GIVEAWAY), severe=True),
    Rule("investment", "scam", 1.5, _re(INVEST), severe=True),
    Rule("sell_engagement", "scam", 2.0, _re(SELL)),
    Rule("phone_number", "scam", 2.0, _re(PHONE, "plain")),
    Rule("crypto_wallet", "scam", 2.5, _re(WALLET, "plain"), severe=True),
    Rule("self_promo", "spam", 1.0, _re(SELF_PROMO)),
    Rule("all_caps", "spam", 0.5, _caps),
    Rule("flood", "spam", 1.0, _flood),
    Rule("self_harm_phrase", "risk", 0.0, _re(SELF_HARM)),
    Rule("encourage_self_harm", "risk", 0.0, _re(ENCOURAGE_SELF_HARM)),
    Rule("threat_phrase", "risk", 0.0, _re(THREAT)),
    Rule("hint_identity", "hint", 0.0, _re(IDENTITY)),
    Rule("hint_sexual", "hint", 0.0, _re(SEXUAL_HINT)),
    Rule("hint_personal_info", "hint", 0.0, _re(PERSONAL_INFO)),
    Rule("hint_self_harm", "hint", 0.0, _re(SELF_HARM_HINT)),
    Rule("hint_targeted_negative", "hint", 0.0, _re(TARGETED_NEG)),
]


@dataclass
class RuleResult:
    normalized: Normalized
    hits: list[str] = field(default_factory=list)
    score: float = 0.0  # scam + spam weight total
    risk_hits: list[str] = field(default_factory=list)
    hint_hits: list[str] = field(default_factory=list)
    severe: bool = False


def run_rules(text: str, rules: list[Rule] = RULES) -> RuleResult:
    n = normalize(text)
    res = RuleResult(normalized=n)
    for rule in rules:
        if rule.check(n):
            res.hits.append(rule.id)
            if rule.kind == "risk":
                res.risk_hits.append(rule.id)
            elif rule.kind == "hint":
                res.hint_hits.append(rule.id)
            else:
                res.score += rule.weight
                res.severe |= rule.severe
    return res
