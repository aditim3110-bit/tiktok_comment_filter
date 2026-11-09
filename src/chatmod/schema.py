"""Shared data model used by every tier, the API and the eval harness."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Label(str, Enum):
    benign = "benign"
    scam_spam = "scam_spam"
    harassment = "harassment"
    hate = "hate"
    sexual = "sexual"
    threat_violence = "threat_violence"
    self_harm = "self_harm"


HARMFUL_LABELS = [l for l in Label if l is not Label.benign]


class Action(str, Enum):
    allow = "allow"
    warn = "warn"
    hide = "hide"
    block = "block"


class Route(str, Enum):
    """Where a decision was finally settled."""

    tier1 = "tier1"
    tier2 = "tier2"
    tier3 = "tier3"  # pending human review (provisional action applied)
    human = "human"  # resolved by a reviewer


class Message(BaseModel):
    text: str
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_id: str | None = None
    channel_id: str | None = None
    context: list[str] = Field(default_factory=list)  # a few preceding chat lines, optional


class Tier1Result(BaseModel):
    settled: bool
    label: Label | None = None
    severity: int = 0
    confidence: float = 0.0
    tox_scores: dict[str, float] = Field(default_factory=dict)
    rule_hits: list[str] = Field(default_factory=list)
    rule_score: float = 0.0
    normalized_text: str = ""
    reason: str = ""
    latency_ms: float = 0.0


class JudgeOutput(BaseModel):
    """Structured JSON the LLM must return."""

    label: Label
    severity: int = Field(ge=0, le=3)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    user_message: str


class Tier2Result(BaseModel):
    output: JudgeOutput | None = None
    error: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


class Decision(BaseModel):
    message_id: str
    text: str
    label: Label
    severity: int
    confidence: float
    action: Action
    route: Route
    rationale: str = ""
    user_message: str = ""
    needs_review: bool = False
    review_id: int | None = None
    latency_ms: float = 0.0
    tier1: Tier1Result | None = None
    tier2: Tier2Result | None = None
    signals: dict[str, Any] = Field(default_factory=dict)
