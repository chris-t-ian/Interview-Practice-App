"""Build the chat model and agent from `ModelSettings`.

We instantiate `ChatOpenRouter` directly instead of passing an
"openrouter:<id>" string, so every setting is passed explicitly. All fields
below exist on `ChatOpenRouter` in langchain-openrouter 0.2.9 and are
forwarded to the OpenRouter API. Note that its `timeout` is in milliseconds.
"""

from typing import Any

from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from langgraph.graph.state import CompiledStateGraph

from langchain_streamlit_app.settings import ModelSettings

# Retries of the OpenRouter SDK on 5xx and connection errors. Each retry adds
# up to ~150 s of backoff (SDK default is 2), so keep it low: otherwise the
# total wait is far longer than the per-request timeout the user sets.
MAX_RETRIES = 1

# ChatOpenRouter fields shown as the "effective config" in the UI. An explicit
# allow-list, so the API key can never end up in it.
_EFFECTIVE_FIELDS = (
    "model_name",
    "temperature",
    "top_p",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "request_timeout",
    "max_retries",
)

# Type of the compiled agent returned by `create_agent`.
Agent = CompiledStateGraph


def build_chat_model(settings: ModelSettings) -> ChatOpenRouter:
    """Create a ChatOpenRouter instance. Reads `OPENROUTER_API_KEY` from the env.

    Raises `ValueError` if the key is not set.
    """
    return ChatOpenRouter(
        model=settings.model_id,
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_tokens=settings.max_tokens,
        frequency_penalty=settings.frequency_penalty,
        presence_penalty=settings.presence_penalty,
        seed=settings.seed,
        timeout=settings.timeout * 1000,  # seconds -> milliseconds
        max_retries=MAX_RETRIES,
    )


def build_agent(
    chat_model: ChatOpenRouter,
    system_prompt: str,
    response_format: type[Any] | None = None,
) -> Agent:
    """Create a tool-less agent around the given chat model.

    With `response_format` (e.g. a Pydantic model) the agent returns the parsed
    result under `result["structured_response"]`.
    """
    return create_agent(
        model=chat_model, tools=[], system_prompt=system_prompt, response_format=response_format
    )


def effective_config(chat_model: ChatOpenRouter) -> dict[str, Any]:
    """The settings actually set on the chat model (no secrets)."""
    return chat_model.model_dump(include=set(_EFFECTIVE_FIELDS))
