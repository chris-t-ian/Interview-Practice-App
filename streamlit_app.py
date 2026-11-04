"""Job Description Analyser: Streamlit UI.

All logic lives in `langchain_streamlit_app`; this file only handles layout,
widgets, caching and rendering.
"""

import json
import re

import streamlit as st
from pydantic import ValidationError

from langchain_streamlit_app import settings as cfg
from langchain_streamlit_app.analyzer import (
    DEFAULT_HOURS_PER_DAY,
    DEFAULT_STUDY_DAYS,
    MAX_STUDY_DAYS,
    MIN_STUDY_DAYS,
    AnalysisError,
    AnalysisResult,
    analyze_with_metadata,
    build_analysis_agent,
    time_budget_for,
    to_markdown,
    total_study_hours,
)
from langchain_streamlit_app.llm import build_chat_model, effective_config
from langchain_streamlit_app.prompts import DEFAULT_PROMPT_KEY, PROMPTS
from langchain_streamlit_app.settings import ModelSettings

st.set_page_config(page_title="Job description analyser", page_icon=":material/work:", layout="wide")

# Loads OPENROUTER_API_KEY (and optionally OPENROUTER_MODEL) from .env.
cfg.load_env()

DEFAULTS = cfg.default_settings()
MODEL_OPTIONS = list(dict.fromkeys([DEFAULTS.model_id, *cfg.CURATED_MODELS]))

# Session-state key for each settings widget, with its default value.
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
st.session_state.setdefault("prompt_key", DEFAULT_PROMPT_KEY)
# Last successful AnalysisResult; kept so reruns/tab switches don't re-call the API.
st.session_state.setdefault("last_result", None)


def reset_settings() -> None:
    st.session_state.update(WIDGET_DEFAULTS)


def _settings_key(settings: ModelSettings) -> str:
    return settings.model_dump_json()


# One chat model / agent per distinct (settings, prompt, time budget).
@st.cache_resource(max_entries=16, hash_funcs={ModelSettings: _settings_key})
def get_chat_model(settings: ModelSettings):
    return build_chat_model(settings)


@st.cache_resource(max_entries=16, hash_funcs={ModelSettings: _settings_key})
def get_analysis_agent(settings: ModelSettings, prompt_key: str, time_budget_hours: float):
    return build_analysis_agent(settings, prompt_key, time_budget_hours)


# ---- Sidebar: prompt technique + model settings -----------------------------
with st.sidebar:
    st.header("Prompt technique")
    prompt_key = st.selectbox(
        "System prompt",
        list(PROMPTS),
        key="prompt_key",
        format_func=lambda k: PROMPTS[k].name,
    )
    spec = PROMPTS[prompt_key]
    st.caption(f"**{spec.technique}**: {spec.description}")

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

# ---- Main area: input form --------------------------------------------------
st.title("Job description analyser")
st.caption(
    f"Paste a job description to get key skills, likely interview topics and a study plan. "
    f"Model: `{settings.model_id}` via OpenRouter · Prompt: {spec.name}"
)

if not cfg.has_api_key():
    st.error("Add `OPENROUTER_API_KEY` to your `.env` file.", title="Missing API key")
    st.stop()

try:
    chat_model = get_chat_model(settings)
except Exception as exc:  # e.g. invalid parameters rejected by ChatOpenRouter
    st.error(str(exc), title="Could not build the model")
    st.stop()

with st.sidebar.expander("Effective config", icon=":material/tune:"):
    st.json(effective_config(chat_model))

with st.form("jd_form"):
    jd_text = st.text_area(
        "Job description",
        height=320,
        placeholder="Paste the full job description here…",
    )
    study_days = st.number_input(
        "Days until interview",
        min_value=MIN_STUDY_DAYS,
        max_value=MAX_STUDY_DAYS,
        value=DEFAULT_STUDY_DAYS,
        step=1,
        help=f"The study plan assumes about {DEFAULT_HOURS_PER_DAY:g} hours of study per day.",
    )
    submitted = st.form_submit_button("Analyse", icon=":material/search:", type="primary")

if submitted:
    if not jd_text.strip():
        st.error("Paste a job description first.", title="Nothing to analyse")
    else:
        try:
            hours = time_budget_for(int(study_days))
            agent = get_analysis_agent(settings, prompt_key, hours)
            with st.spinner(f"Analysing with {spec.name}…"):
                st.session_state.last_result = analyze_with_metadata(
                    jd_text, settings, prompt_key, int(study_days), agent=agent
                )
        except AnalysisError as exc:
            st.session_state.last_result = None
            st.error(str(exc), title="Analysis failed")
        except Exception as exc:  # anything unexpected, e.g. while building the agent
            st.session_state.last_result = None
            st.error(f"{type(exc).__name__}: {exc}", title="Analysis failed")

# ---- Results ----------------------------------------------------------------
result: AnalysisResult | None = st.session_state.last_result
if result is None:
    st.stop()

analysis = result.analysis
usage = result.usage
planned_hours = total_study_hours(analysis)

st.divider()
st.subheader(analysis.role_title)
with st.container(horizontal=True):
    st.badge(analysis.seniority, icon=":material/badge:", color="blue")
    st.badge(PROMPTS[result.prompt_key].name, icon=":material/psychology:", color="violet")
    st.badge(f"Interview in {result.study_days} d", icon=":material/event:", color="gray")


def _fmt_int(value: int | None) -> str:
    return f"{value:,}" if value is not None else "n/a"


with st.container(horizontal=True):
    st.metric("Latency", f"{usage.latency_s:.1f} s", border=True)
    st.metric(
        "Tokens (in / out)",
        f"{_fmt_int(usage.input_tokens)} / {_fmt_int(usage.output_tokens)}",
        help=(
            f"Total: {_fmt_int(usage.total_tokens)}. "
            f"Reasoning: {_fmt_int(usage.reasoning_tokens)}. "
            f"Model calls: {usage.model_calls}."
        ),
        border=True,
    )
    st.metric(
        "Cost",
        f"${usage.cost_usd:.4f}" if usage.cost_usd is not None else "n/a",
        help="As reported by OpenRouter.",
        border=True,
    )
    st.metric(
        "Study hours",
        f"{planned_hours:g} / {result.time_budget_hours:g}",
        help="Planned hours / time budget.",
        border=True,
    )
if usage.model_name or usage.provider:
    st.caption(
        f"Served by `{usage.model_name or 'unknown model'}`"
        + (f" via {usage.provider}" if usage.provider else "")
    )

if analysis.reasoning:
    with st.expander("Reasoning", icon=":material/lightbulb:"):
        st.markdown(analysis.reasoning)

skills_tab, topics_tab, plan_tab, json_tab = st.tabs(
    [
        f":material/checklist: Skills ({len(analysis.skills)})",
        f":material/forum: Interview topics ({len(analysis.interview_topics)})",
        ":material/calendar_month: Study plan",
        ":material/data_object: Raw JSON",
    ]
)

with skills_tab:
    must = sum(s.priority == "must-have" for s in analysis.skills)
    st.caption(f"{must} must-have · {len(analysis.skills) - must} nice-to-have")
    st.dataframe(
        [
            {
                "Skill": s.name,
                "Category": s.category,
                "Priority": s.priority,
                "Inferred": s.inferred,
                "Evidence": s.evidence,
            }
            for s in analysis.skills
        ],
        hide_index=True,
        column_config={
            "Skill": st.column_config.TextColumn(width="medium", pinned=True),
            "Category": st.column_config.TextColumn(width="small"),
            "Priority": st.column_config.TextColumn(width="small"),
            "Inferred": st.column_config.CheckboxColumn(
                width="small", help="Only implied by the JD, not stated explicitly."
            ),
            "Evidence": st.column_config.TextColumn(width="large"),
        },
        alt="Key skills from the job description",
    )

DIFFICULTY_COLORS = {"easy": "green", "medium": "orange", "hard": "red"}

with topics_tab:
    for i, topic in enumerate(analysis.interview_topics):
        color = DIFFICULTY_COLORS.get(topic.difficulty, "gray")
        with st.expander(f"{topic.topic} :{color}-badge[{topic.difficulty}]", expanded=i == 0):
            st.markdown(f"*{topic.why_likely}*")
            st.markdown(
                "\n".join(f"{n}. {q}" for n, q in enumerate(topic.example_questions, start=1))
            )

with plan_tab:
    st.dataframe(
        [
            {
                "Block": item.day_or_block,
                "Focus": item.focus,
                "Activities": "\n".join(f"• {a}" for a in item.activities),
                "Resources": ", ".join(item.resources),
                "Hours": item.estimated_hours,
            }
            for item in analysis.study_plan
        ],
        hide_index=True,
        column_config={
            "Block": st.column_config.TextColumn(width="small"),
            "Activities": st.column_config.TextColumn(width="large"),
            "Hours": st.column_config.NumberColumn(format="%.1f h", width="small"),
        },
        alt="Study plan",
    )
    st.markdown(f"**Total: {planned_hours:g} h** of a {result.time_budget_hours:g} h budget.")
    if planned_hours > result.time_budget_hours:
        st.warning("The plan exceeds the time budget.", icon=":material/warning:")

with json_tab:
    st.json(analysis.model_dump(mode="json"))

slug = re.sub(r"[^a-z0-9]+", "-", analysis.role_title.lower()).strip("-") or "analysis"
with st.container(horizontal=True):
    st.download_button(
        "Download JSON",
        data=json.dumps(analysis.model_dump(mode="json"), indent=2, ensure_ascii=False),
        file_name=f"{slug}.json",
        mime="application/json",
        icon=":material/download:",
        on_click="ignore",
    )
    st.download_button(
        "Download Markdown",
        data=to_markdown(result),
        file_name=f"{slug}.md",
        mime="text/markdown",
        icon=":material/download:",
        on_click="ignore",
    )
