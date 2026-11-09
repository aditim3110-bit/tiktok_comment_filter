"""Text normalization to defeat cheap obfuscation before rules run.

Two views of the text are produced:
- ``plain``:  NFKC, zero-width stripped, lowercased, whitespace collapsed. Used for
  numeric patterns (phone numbers, wallets) where leetspeak mapping would corrupt digits.
- ``deobf``:  ``plain`` + de-obfuscated dots ("dot com", "[.]", "t . me"), leetspeak
  mapped inside word-like tokens, homoglyphs folded, and runs of 3+ repeated letters
  collapsed to 2. Used for keyword / link rules.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_ZERO_WIDTH = re.compile(r"[​-‏⁠-⁤﻿­]")
_WS = re.compile(r"\s+")

# Cyrillic/Greek lookalikes that NFKC does not fold.
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
        "ј": "j", "ѕ": "s", "к": "k", "м": "m", "т": "t", "н": "h", "в": "b", "г": "r", "ԁ": "d",
        "ո": "n", "ӏ": "l", "ɡ": "g",
        "α": "a", "ο": "o", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "τ": "t", "ρ": "p",
    }
)

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})

_BRACKET_DOT = re.compile(r"\s*[\[\(\{]\s*\.\s*[\]\)\}]\s*")
# "t . me", "bit .ly": whitespace *before* the dot. Ordinary sentences ("thanks. me too")
# have no space before the period, so they are left alone.
_SPACED_DOT = re.compile(r"(?<=\w)\s+\.\s*(?=\w)")
_REPEATS = re.compile(r"([a-z])\1{2,}")
# "d m m e" / "l i n k": three or more single letters separated by single spaces.
_SPACED_LETTERS = re.compile(r"(?<!\S)(?:[a-z] ){2,}[a-z](?!\S)")
_TOKEN = re.compile(r"\S+")
_TLDS = r"(?:com|net|org|io|me|ly|gg|co|xyz|top|link|site|online|info|app|shop|ru|cn|tk|us|live|tv)"


@dataclass(frozen=True)
class Normalized:
    original: str
    plain: str
    deobf: str


def _plain(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = _ZERO_WIDTH.sub("", t)
    t = t.translate(_HOMOGLYPHS)
    t = t.lower()
    return _WS.sub(" ", t).strip()


def _deleet_token(tok: str) -> str:
    # Only map leetspeak inside tokens that contain at least one letter, so that plain
    # numbers ("gg 2 ez", "100%") and handles keep their digits meaningful.
    if not re.search(r"[a-z]", tok):
        return tok
    if tok.startswith("@") and len(tok) > 1:
        return "@" + tok[1:].translate(_LEET)
    return tok.translate(_LEET)


def _undot(t: str) -> str:
    # "example dot com" -> "example.com" (only when followed by a TLD-ish word, so the
    # ordinary word "dot" in a sentence is left alone).
    t = re.sub(rf"(?<=\w)\s*(?:\(|\[|\{{)?\s*d[o0]t\s*(?:\)|\]|\}})?\s*(?={_TLDS}\b)", ".", t)
    t = _BRACKET_DOT.sub(".", t)
    t = _SPACED_DOT.sub(".", t)
    return t


def normalize(text: str) -> Normalized:
    plain = _plain(text)
    deobf = _undot(plain)
    deobf = " ".join(_deleet_token(tok) for tok in _TOKEN.findall(deobf))
    deobf = _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), deobf)
    deobf = _REPEATS.sub(r"\1\1", deobf)
    return Normalized(original=text, plain=plain, deobf=deobf)
