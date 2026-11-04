"""Model settings: defaults, allowed ranges, curated model list and env loading.

`ModelSettings` is a frozen (immutable, hashable) Pydantic model so it can be
used directly as a cache key: every distinct combination of settings maps to
its own cached agent.
"""

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL_ID = "openai/gpt-6-luna"

# Curated OpenRouter model ids for the sidebar selectbox (verified against the
# public OpenRouter model list). The first entry is the default.
CURATED_MODELS: tuple[str, ...] = (
    DEFAULT_MODEL_ID,
    "anthropic/claude-sonnet-5.5",
    "google/gemini-3.8-flash",
    "deepseek/deepseek-v4.1-flash",
    "qwen/qwen3.8-flash",
)

# Models that OpenRouter lists as not supporting `temperature` / `top_p`
# (reasoning models). They silently ignore those settings.
SAMPLING_IGNORED_MODELS: frozenset[str] = frozenset(
    {
        "openai/gpt-6-luna",
        "openai/gpt-6-luna-pro",
        "anthropic/claude-sonnet-5.5",
        "anthropic/claude-sonnet-5",
    }
)
# Heuristic for custom ids not in the set above.
_REASONING_HINT = re.compile(r"(^openai/(o\d|gpt-[56])|reason|think|-r1\b)", re.IGNORECASE)

# OpenRouter ids look like "<vendor>/<model>", optionally with a ":variant".
MODEL_ID_PATTERN = r"^[\w.\-]+/[\w.\-]+(:[\w.\-]+)?$"


@dataclass(frozen=True)
class Range:
    """Allowed range for a numeric setting (also used for UI sliders)."""

    min: float
    max: float
    step: float


TEMPERATURE = Range(0.0, 2.0, 0.05)
TOP_P = Range(0.0, 1.0, 0.05)
MAX_TOKENS = Range(256, 32_768, 256)
PENALTY = Range(-2.0, 2.0, 0.1)
TIMEOUT_S = Range(5, 300, 5)


class ModelSettings(BaseModel):
    """Generation settings for the chat model. Immutable and hashable."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    model_id: str = Field(
        default=DEFAULT_MODEL_ID,
        pattern=MODEL_ID_PATTERN,
        description="OpenRouter model id, e.g. 'openai/gpt-6-luna'.",
    )
    temperature: float = Field(default=0.3, ge=TEMPERATURE.min, le=TEMPERATURE.max)
    top_p: float = Field(default=1.0, ge=TOP_P.min, le=TOP_P.max)
    max_tokens: int = Field(default=8192, ge=MAX_TOKENS.min, le=MAX_TOKENS.max)
    frequency_penalty: float = Field(default=0.0, ge=PENALTY.min, le=PENALTY.max)
    presence_penalty: float = Field(default=0.0, ge=PENALTY.min, le=PENALTY.max)
    seed: int | None = Field(default=None, ge=0, description="Fixed seed; None = random.")
    timeout: int = Field(
        default=60, ge=TIMEOUT_S.min, le=TIMEOUT_S.max, description="Request timeout in seconds."
    )


def load_env() -> None:
    """Load variables from `.env` into the process environment."""
    load_dotenv()


def default_model_id() -> str:
    """Model id from `OPENROUTER_MODEL`, falling back to the built-in default."""
    return os.getenv("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL_ID


def default_settings() -> ModelSettings:
    """Default settings, honouring `OPENROUTER_MODEL` from the environment."""
    return ModelSettings(model_id=default_model_id())


def has_api_key() -> bool:
    """Whether `OPENROUTER_API_KEY` is set (without exposing its value)."""
    return bool(os.getenv("OPENROUTER_API_KEY"))


def may_ignore_sampling(model_id: str) -> bool:
    """True if the model is (likely) a reasoning model that ignores temperature/top_p."""
    base_id = model_id.split(":", 1)[0]
    return base_id in SAMPLING_IGNORED_MODELS or bool(_REASONING_HINT.search(base_id))
