"""Job Description Analyser: Streamlit UI.

All logic lives in `langchain_streamlit_app`; this file only handles layout,
widgets, caching and rendering.
"""

import json
import re

import pandas as pd
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
    InputRejected,
    analyze_with_metadata,
    build_analysis_agent,
    time_budget_for,
    to_markdown,
    total_study_hours,
)
from langchain_streamlit_app.guards import (
    GuardResult,
    check_rate_limit,
    record_request,
    run_guards,
)
from langchain_streamlit_app.llm import build_chat_model, effective_config
from langchain_streamlit_app.prompts import DEFAULT_PROMPT_KEY, PROMPTS
from langchain_streamlit_app.settings import ModelSettings

st.set_page_config(
    page_title="Job description analyser", page_icon=":material/work:", layout="wide"
)

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
# Increments per successful analysis so checklist widget keys reset for a new result.
st.session_state.setdefault("result_id", 0)
# Open state of the JD input expander (collapses after a valid submit).
st.session_state.setdefault("jd_expander", True)
# Timestamps of recent analyses, for the per-session rate limit (guards.py).
st.session_state.setdefault("analysis_times", [])


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



def collapse_jd_input() -> None:
    """Submit callback: collapse the JD input once a JD passes the guards."""
    if run_guards(st.session_state.get("jd_text", "")).ok:
        st.session_state.jd_expander = False


def jd_expander_label() -> str:
    jd = st.session_state.get("jd_text", "").strip()
    if st.session_state.jd_expander or not jd:
        return "Job description"
    first_line = jd.splitlines()[0]
    preview = first_line if len(first_line) <= 80 else first_line[:80].rstrip() + "…"
    return f"Job description: {preview}"


with st.expander(
    jd_expander_label(), key="jd_expander", on_change="rerun", icon=":material/description:"
):
    with st.form("jd_form", border=False):
        jd_text = st.text_area(
            "Job description",
            key="jd_text",
            height=320,
            placeholder="Paste the full job description here…",
            label_visibility="collapsed",
        )
        study_days = st.number_input(
            "Days until interview",
            min_value=MIN_STUDY_DAYS,
            max_value=MAX_STUDY_DAYS,
            value=DEFAULT_STUDY_DAYS,
            step=1,
            help=f"The study plan assumes about {DEFAULT_HOURS_PER_DAY:g} hours of study per day.",
        )
        submitted = st.form_submit_button(
            "Analyse", icon=":material/search:", type="primary", on_click=collapse_jd_input
        )

def show_blocked(guard: GuardResult, title: str) -> None:
    st.error(guard.reason, title=title, icon=":material/block:")
    st.caption(f"Rule: `{guard.rule}`")


if submitted:
    # Guards run before any agent is built or model is called.
    rate = check_rate_limit(st.session_state.analysis_times)
    guard = run_guards(jd_text)
    if not rate.ok:
        show_blocked(rate, "Too many requests")
    elif not guard.ok:
        st.session_state.last_result = None
        title = "Input blocked" if guard.rule.startswith("injection") else "Invalid input"
        show_blocked(guard, title)
    else:
        for warning in guard.warnings:
            st.warning(warning, icon=":material/warning:")
        try:
            hours = time_budget_for(int(study_days))
            agent = get_analysis_agent(settings, prompt_key, hours)
            record_request(st.session_state.analysis_times)
            with st.spinner(f"Analysing with {spec.name}…"):
                st.session_state.last_result = analyze_with_metadata(
                    guard.sanitized_text, settings, prompt_key, int(study_days), agent=agent
                )
            st.session_state.result_id += 1
            for old in [k for k in st.session_state if str(k).startswith("chk_")]:
                del st.session_state[old]  # checklist ticks belong to the previous result
        except InputRejected as exc:
            st.session_state.last_result = None
            show_blocked(exc.guard, "Input blocked")
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
result_spec = PROMPTS[result.prompt_key]

CATEGORY_COLORS = {"technical": "blue", "soft": "violet", "domain": "orange", "tool": "green"}
PRIORITY_COLORS = {"must-have": "red", "nice-to-have": "gray"}
PROFICIENCY_COLORS = {
    "beginner": "green",
    "fluent": "yellow",
    "professional": "red",
    "unknown": "gray",
}
MENTIONED_COLORS = {"Explicit": "blue", "Implied": "orange"}
DIFFICULTY_COLORS = {"easy": "green", "medium": "orange", "hard": "red"}


def _fmt_int(value: int | None) -> str:
    return f"{value:,}" if value is not None else "n/a"


def _mentioned(inferred: bool) -> str:
    return "Implied" if inferred else "Explicit"


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+!|~:$])")


def _md_escape(text: str) -> str:
    """Escape Markdown so model text renders literally inside st.table cells."""
    return _MD_SPECIAL.sub(r"\\\1", text.replace("\n", " "))


def _md_badge(label: str, color: str) -> str:
    return f":{color}-badge[{_md_escape(label)}]"


# Tooltip look for the evidence shown when hovering a skill name.
TOOLTIP_PROPS = [
    ("visibility", "hidden"),
    ("position", "absolute"),
    ("z-index", "10"),
    ("background-color", "rgba(30, 30, 30, 0.95)"),
    ("color", "white"),
    ("padding", "6px 10px"),
    ("border-radius", "6px"),
    ("font-size", "0.85em"),
    ("white-space", "normal"),
    ("width", "max-content"),
    ("max-width", "320px"),
    ("transform", "translate(-20px, -110%)"),
]


def _css_safe(text: str) -> str:
    """Text for a CSS `content:` string (pandas puts tooltips there unescaped)."""
    return text.replace("\\", "/").replace('"', "'").replace("\n", " ")


def skills_table(skills) -> "pd.io.formats.style.Styler":
    """Skill table with colored badges; the evidence is a hover tooltip on the skill name."""

    def badges(values: list[str], colors: dict[str, str], labels=None) -> list[str]:
        labels = labels or values
        return [_md_badge(lbl, colors.get(v, "gray")) for v, lbl in zip(values, labels)]

    proficiency = [s.proficiency for s in skills]
    mentioned = [_mentioned(s.inferred) for s in skills]
    df = pd.DataFrame(
        {
            "Skill": [_md_escape(s.name) for s in skills],
            "Category": badges([s.category for s in skills], CATEGORY_COLORS),
            "Priority": badges([s.priority for s in skills], PRIORITY_COLORS),
            "Proficiency level": badges(
                proficiency, PROFICIENCY_COLORS, [p.capitalize() for p in proficiency]
            ),
            "How it's mentioned": badges(mentioned, MENTIONED_COLORS),
        }
    )
    tooltips = pd.DataFrame("", index=df.index, columns=df.columns)
    tooltips["Skill"] = [f"Evidence: “{_css_safe(s.evidence)}”" for s in skills]
    return df.style.set_tooltips(tooltips, props=TOOLTIP_PROPS)


@st.dialog("Raw JSON", width="large", icon=":material/data_object:")
def show_raw_json(data: dict) -> None:
    st.json(data)


# Header: role + toolbar
analysis_json = analysis.model_dump(mode="json")
slug = re.sub(r"[^a-z0-9]+", "-", analysis.role_title.lower()).strip("-") or "analysis"

st.divider()
head_left, head_right = st.columns([3, 2], vertical_alignment="center")
with head_left:
    st.subheader(analysis.role_title)
    with st.container(horizontal=True):
        st.badge(analysis.seniority, icon=":material/badge:", color="blue")
        st.badge(f"Interview in {result.study_days} d", icon=":material/event:", color="gray")
with head_right:
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("View JSON", icon=":material/data_object:"):
            show_raw_json(analysis_json)
        st.download_button(
            "JSON",
            data=json.dumps(analysis_json, indent=2, ensure_ascii=False),
            file_name=f"{slug}.json",
            mime="application/json",
            icon=":material/download:",
            on_click="ignore",
        )
        st.download_button(
            "Markdown",
            data=to_markdown(result),
            file_name=f"{slug}.md",
            mime="text/markdown",
            icon=":material/download:",
            on_click="ignore",
        )

# Dashboard metrics
with st.container(horizontal=True):
    st.metric("Prompt", result_spec.name, help=f"Technique: {result_spec.technique}", border=True)
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
if usage.model_name or usage.provider:
    st.caption(
        f"Served by `{usage.model_name or 'unknown model'}`"
        + (f" via {usage.provider}" if usage.provider else "")
    )

if analysis.reasoning:
    with st.expander("Reasoning", icon=":material/lightbulb:"):
        st.markdown(analysis.reasoning)

# Skills (left) and interview topics (right)
skills_col, topics_col = st.columns(2, gap="large")

with skills_col:
    st.subheader(f":material/checklist: Key skills ({len(analysis.skills)})", divider="gray")
    st.table(
        skills_table(analysis.skills),
        hide_index=True,
        border="horizontal",
        alt="Key skills with category, priority, proficiency level and how they are mentioned",
    )
    st.caption(":material/info: Hover over a skill to see the evidence from the job description.")

with topics_col:
    st.subheader(
        f":material/forum: Likely interview topics ({len(analysis.interview_topics)})",
        divider="gray",
    )
    for i, topic in enumerate(analysis.interview_topics):
        color = DIFFICULTY_COLORS.get(topic.difficulty, "gray")
        with st.expander(f"{topic.topic} :{color}-badge[{topic.difficulty}]", expanded=i == 0):
            st.markdown(f"*{topic.why_likely}*")
            st.markdown(
                "\n".join(f"{n}. {q}" for n, q in enumerate(topic.example_questions, start=1))
            )


# Study plan: checklist cards. A fragment, so ticking a box reruns only this part.
@st.fragment
def render_study_plan(result: AnalysisResult, result_id: int) -> None:
    plan = result.analysis.study_plan
    planned = total_study_hours(result.analysis)
    keys = [
        [f"chk_{result_id}_{i}_{j}" for j in range(len(item.activities))]
        for i, item in enumerate(plan)
    ]
    total = sum(len(ks) for ks in keys)
    done = sum(bool(st.session_state.get(k)) for ks in keys for k in ks)

    st.subheader(
        f":material/calendar_month: Study plan · {planned:g} / {result.time_budget_hours:g} h",
        divider="gray",
        help="Planned hours / time budget.",
    )
    if planned > result.time_budget_hours:
        st.warning("The plan exceeds the time budget.", icon=":material/warning:")
    if total:
        st.progress(done / total, text=f"{done} of {total} activities done")

    per_row = 3
    for start in range(0, len(plan), per_row):
        cols = st.columns(per_row)
        for col, i in zip(cols, range(start, min(start + per_row, len(plan)))):
            item = plan[i]
            block_done = bool(keys[i]) and all(st.session_state.get(k) for k in keys[i])
            with col.container(border=True, height="stretch"):
                status = " :green[:material/check_circle:]" if block_done else ""
                st.caption(f"{item.day_or_block} · {item.estimated_hours:g} h{status}")
                st.markdown(f"**{item.focus}**")
                for activity, key in zip(item.activities, keys[i]):
                    st.checkbox(activity, key=key)
                if item.resources:
                    st.caption(f":material/menu_book: {', '.join(item.resources)}")


render_study_plan(result, st.session_state.result_id)
