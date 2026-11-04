import pytest
from pydantic import ValidationError

from langchain_streamlit_app.llm import build_chat_model, effective_config
from langchain_streamlit_app.settings import ModelSettings, may_ignore_sampling


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
    [{"temperature": 2.5}, {"top_p": 1.5}, {"max_tokens": 10}, {"model_id": "no slash"}, {"seed": -1}],
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
    assert "openrouter_api_key" not in cfg
