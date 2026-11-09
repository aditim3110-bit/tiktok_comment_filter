"""Configuration loading (YAML -> pydantic)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from chatmod.schema import Action, Label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


class Tier1Config(BaseModel):
    scorer: str = "detoxify"
    detoxify_model: str = "unbiased"
    low_thr: float = 0.30
    high_thr: float = 0.95
    scam_strong: float = 3.0
    scam_escalate: float = 1.0


class Tier2Config(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-3.8-flash"
    temperature: float = 0.0
    thinking_budget: int | None = None
    thinking_level: str | None = "low"
    max_output_tokens: int = 2048
    safety_threshold: str | None = "BLOCK_NONE"
    timeout_s: float = 8.0
    eval_timeout_s: float = 30.0
    max_concurrency: int = 8
    rpm_limit: float = 0.0          # client-side requests/minute pacing; 0 = off
    max_retries: int = 3
    live_max_wait_s: float = 2.0    # serving: longest rate-limit wait before falling back to review
    eval_max_wait_s: float = 120.0  # evals: wait out per-minute limits
    conf_thr: float = 0.70
    always_review: list[Label] = Field(default_factory=lambda: [Label.self_harm, Label.threat_violence])
    cache_path: str = ".cache/llm_cache.sqlite"
    price_in_per_m: float = 0.30
    price_out_per_m: float = 2.50


class Tier3Config(BaseModel):
    db_path: str = ".cache/review_queue.sqlite"


class Config(BaseModel):
    policy_path: str = "policy/policy.md"
    tier1: Tier1Config = Field(default_factory=Tier1Config)
    tier2: Tier2Config = Field(default_factory=Tier2Config)
    tier3: Tier3Config = Field(default_factory=Tier3Config)
    actions: dict[Label, list[Action]] = Field(default_factory=dict)

    def action_for(self, label: Label, severity: int) -> Action:
        table = self.actions.get(label)
        if not table:
            return Action.allow if label is Label.benign else Action.hide
        return table[max(0, min(severity, len(table) - 1))]

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def policy_text(self) -> str:
        return self.resolve(self.policy_path).read_text()


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    return Config.model_validate(data or {})
