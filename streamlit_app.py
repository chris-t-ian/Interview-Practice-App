"""Streamlit Q&A app that calls a model through LangChain + OpenRouter."""

import streamlit as st
from pydantic import ValidationError

from langchain_streamlit_app import settings as cfg
from langchain_streamlit_app.llm import (
    DEFAULT_SYSTEM_PROMPT,
    build_agent,
    build_chat_model,
    effective_config,
)
from langchain_streamlit_app.settings import ModelSettings

# Loads OPENROUTER_API_KEY (and optionally OPENROUTER_MODEL) from .env.
cfg.load_env()

DEFAULTS = cfg.default_settings()
MODEL_OPTIONS = list(dict.fromkeys([DEFAULTS.model_id, *cfg.CURATED_MODELS]))

# Session-state key for each widget, with its default value.
WIDGET_DEFAULTS = {
    "s_model_choice": DEFAULTS.model_id,
    "s_custom_model": "",
    "s_temperature": DEFAULTS.temperature,
    "s_top_p": DEFAULTS.top_p,
    "s_max_tokens": DEFAULTS.max_tokens,
    "s_frequency_penalty": DEFAULTS.frequency_penalty,
    "s_presence_penalty": DEFAULTS.presence_penalty,
    "s_seed": DEFAULTS.seed,
    "s_timeout": DEFAULTS.timeout,
}
for key, value in WIDGET_DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_settings() -> None:
    st.session_state.update(WIDGET_DEFAULTS)


def _settings_key(settings: ModelSettings) -> str:
    return settings.model_dump_json()


# One chat model / agent per distinct (settings, system prompt).
@st.cache_resource(max_entries=16, hash_funcs={ModelSettings: _settings_key})
def get_chat_model(settings: ModelSettings):
    return build_chat_model(settings)


@st.cache_resource(max_entries=16, hash_funcs={ModelSettings: _settings_key})
def get_agent(settings: ModelSettings, system_prompt: str):
    return build_agent(get_chat_model(settings), system_prompt)


# ---- Sidebar: model settings ------------------------------------------------
with st.sidebar:
    st.header("Model settings")
    st.selectbox(
        "Model",
        MODEL_OPTIONS,
        key="s_model_choice",
        help="Curated OpenRouter model ids. Use the field below for any other id.",
    )
    st.text_input(
        "Custom model id",
        key="s_custom_model",
        placeholder="vendor/model-name",
        help="Overrides the selection above when filled, e.g. `mistralai/mistral-large-4-0`.",
    )
    model_id = st.session_state.s_custom_model.strip() or st.session_state.s_model_choice

    st.slider(
        "Temperature",
        min_value=cfg.TEMPERATURE.min,
        max_value=cfg.TEMPERATURE.max,
        step=cfg.TEMPERATURE.step,
        key="s_temperature",
        help="Higher = more varied output, lower = more focused and deterministic.",
    )
    st.slider(
        "Top p",
        min_value=cfg.TOP_P.min,
        max_value=cfg.TOP_P.max,
        step=cfg.TOP_P.step,
        key="s_top_p",
        help="Nucleus sampling: only sample from the most likely tokens whose "
        "probabilities add up to this value. Usually tune this or temperature, not both.",
    )
    if cfg.may_ignore_sampling(model_id):
        st.caption(
            ":material/info: This looks like a reasoning model; it may ignore "
            "temperature and top p."
        )
    st.number_input(
        "Max tokens",
        min_value=int(cfg.MAX_TOKENS.min),
        max_value=int(cfg.MAX_TOKENS.max),
        step=int(cfg.MAX_TOKENS.step),
        key="s_max_tokens",
        help="Upper limit on generated tokens. For reasoning models this also "
        "covers the hidden reasoning tokens.",
    )
    st.slider(
        "Frequency penalty",
        min_value=cfg.PENALTY.min,
        max_value=cfg.PENALTY.max,
        step=cfg.PENALTY.step,
        key="s_frequency_penalty",
        help="Positive values discourage repeating the same tokens often.",
    )
    st.slider(
        "Presence penalty",
        min_value=cfg.PENALTY.min,
        max_value=cfg.PENALTY.max,
        step=cfg.PENALTY.step,
        key="s_presence_penalty",
        help="Positive values encourage introducing new topics.",
    )
    st.number_input(
        "Seed",
        min_value=0,
        step=1,
        key="s_seed",
        placeholder="Random",
        help="Fixed seed for more reproducible output (not all models support it). "
        "Leave empty for random.",
    )
    st.slider(
        "Timeout (seconds)",
        min_value=int(cfg.TIMEOUT_S.min),
        max_value=int(cfg.TIMEOUT_S.max),
        step=int(cfg.TIMEOUT_S.step),
        key="s_timeout",
        help="How long to wait for a single model request.",
    )
    st.button(
        "Reset to defaults",
        icon=":material/restart_alt:",
        on_click=reset_settings,
        width="stretch",
    )

    try:
        settings = ModelSettings(
            model_id=model_id,
            temperature=st.session_state.s_temperature,
            top_p=st.session_state.s_top_p,
            max_tokens=st.session_state.s_max_tokens,
            frequency_penalty=st.session_state.s_frequency_penalty,
            presence_penalty=st.session_state.s_presence_penalty,
            seed=st.session_state.s_seed,
            timeout=st.session_state.s_timeout,
        )
    except ValidationError as exc:
        problems = "\n".join(f"- {e['loc'][0]}: {e['msg']}" for e in exc.errors())
        st.error(problems, title="Invalid model settings")
        st.stop()

# ---- Main area --------------------------------------------------------------
st.title("Ask a question")
st.caption(f"Model: `{settings.model_id}` via OpenRouter")

if not cfg.has_api_key():
    st.error("Add `OPENROUTER_API_KEY` to your `.env` file.", title="Missing API key")
    st.stop()

try:
    chat_model = get_chat_model(settings)
    agent = get_agent(settings, DEFAULT_SYSTEM_PROMPT)
except Exception as exc:  # e.g. invalid parameters rejected by ChatOpenRouter
    st.error(str(exc), title="Could not build the model")
    st.stop()

with st.sidebar.expander("Effective config", icon=":material/tune:"):
    st.json(effective_config(chat_model))

with st.form("question_form"):
    question = st.text_area("Your question")
    submitted = st.form_submit_button("Ask", icon=":material/send:")

if submitted and question.strip():
    try:
        with st.spinner("Thinking..."):
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    except Exception as exc:  # surface API/model errors in the UI
        st.error(str(exc), title="The model request failed")
    else:
        with st.container(border=True):
            st.markdown(result["messages"][-1].text)
