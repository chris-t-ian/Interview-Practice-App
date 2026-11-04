import pytest
from pydantic import ValidationError

from langchain_streamlit_app.llm import MAX_RETRIES, build_chat_model, effective_config
from langchain_streamlit_app.settings import (
    REDACTED,
    ModelSettings,
    has_api_key,
    may_ignore_sampling,
    redact_secrets,
)

FAKE_KEY = "sk-or-v1-0123456789abcdef-test"


def test_defaults_and_hashable():
    s = ModelSettings()
    assert s.model_id == "openai/gpt-6-luna"
    assert hash(s) == hash(ModelSettings())
    assert s != ModelSettings(temperature=1.0)


def test_frozen():
    with pytest.raises(ValidationError):
        ModelSettings().temperature = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 2.5},
        {"top_p": 1.5},
        {"max_tokens": 10},
        {"model_id": "no slash"},
        {"seed": -1},
    ],
)
def test_rejects_out_of_range(kwargs):
    with pytest.raises(ValidationError):
        ModelSettings(**kwargs)


def test_model_id_is_stripped_and_accepts_variant():
    assert ModelSettings(model_id="  x/y:free ").model_id == "x/y:free"


def test_may_ignore_sampling():
    assert may_ignore_sampling("openai/gpt-6-luna")
    assert may_ignore_sampling("openai/o4-mini")
    assert not may_ignore_sampling("google/gemini-3.8-flash")


def test_build_chat_model_passes_settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    s = ModelSettings(
        model_id="google/gemini-3.8-flash",
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        frequency_penalty=0.5,
        presence_penalty=-0.5,
        seed=42,
        timeout=30,
    )
    cfg = effective_config(build_chat_model(s))
    assert cfg["model_name"] == "google/gemini-3.8-flash"
    assert cfg["temperature"] == 0.7
    assert cfg["top_p"] == 0.9
    assert cfg["max_tokens"] == 1024
    assert cfg["frequency_penalty"] == 0.5
    assert cfg["presence_penalty"] == -0.5
    assert cfg["seed"] == 42
    assert cfg["request_timeout"] == 30_000
    assert cfg["max_retries"] == MAX_RETRIES
    assert "openrouter_api_key" not in cfg


def test_api_key_never_exposed_by_chat_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    model = build_chat_model(ModelSettings())
    assert FAKE_KEY not in str(effective_config(model))
    assert FAKE_KEY not in repr(model)
    assert FAKE_KEY not in str(model.model_dump())


@pytest.mark.parametrize(("value", "expected"), [(FAKE_KEY, True), ("   ", False), ("", False)])
def test_has_api_key(monkeypatch, value, expected):
    monkeypatch.setenv("OPENROUTER_API_KEY", value)
    assert has_api_key() is expected


def test_has_api_key_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert not has_api_key()


def test_redact_secrets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "custom-secret-123")
    text = "failed with custom-secret-123 and Bearer sk-or-v1-deadbeefdeadbeef"
    redacted = redact_secrets(text)
    assert "custom-secret-123" not in redacted
    assert "sk-or-v1-deadbeef" not in redacted
    assert redacted.count(REDACTED) == 2


def test_redact_secrets_without_key_keeps_text(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert redact_secrets("plain message") == "plain message"
