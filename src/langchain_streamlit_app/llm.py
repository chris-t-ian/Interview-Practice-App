"""Build the chat model and agent from `ModelSettings`.

We instantiate `ChatOpenRouter` directly instead of passing an
"openrouter:<id>" string, so every setting is passed explicitly. All fields
below exist on `ChatOpenRouter` in langchain-openrouter 0.2.9 and are
forwarded to the OpenRouter API. Note that its `timeout` is in milliseconds.
"""

from typing import Any

from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter

from langchain_streamlit_app.settings import ModelSettings

# Placeholder until the prompt registry (prompts.py) exists.
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."

# ChatOpenRouter fields shown as the "effective config" in the UI.
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


def build_chat_model(settings: ModelSettings) -> ChatOpenRouter:
    """Create a ChatOpenRouter instance. Reads `OPENROUTER_API_KEY` from the env."""
    return ChatOpenRouter(
        model=settings.model_id,
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_tokens=settings.max_tokens,
        frequency_penalty=settings.frequency_penalty,
        presence_penalty=settings.presence_penalty,
        seed=settings.seed,
        timeout=settings.timeout * 1000,  # seconds -> milliseconds
    )


def build_agent(chat_model: ChatOpenRouter, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """Create a tool-less agent around the given chat model."""
    return create_agent(model=chat_model, tools=[], system_prompt=system_prompt)


def effective_config(chat_model: ChatOpenRouter) -> dict[str, Any]:
    """The settings actually set on the chat model (no secrets)."""
    return chat_model.model_dump(include=set(_EFFECTIVE_FIELDS))
