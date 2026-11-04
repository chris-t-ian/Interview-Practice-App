"""Analysis pipeline: prompt -> model -> parsed `JobAnalysis`.

No Streamlit imports, so this module can be tested on its own. The content
guards (guards.py) run at the start of `analyze_with_metadata`, before the agent
is built or called; only the sanitised text reaches the model.

The UI caches the agent itself (see `build_analysis_agent`) and passes it in;
without an agent argument, a fresh one is built for each call.
"""

import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from langchain_streamlit_app.guards import GuardResult, run_guards
from langchain_streamlit_app.llm import build_agent, build_chat_model
from langchain_streamlit_app.prompts import (
    MAX_TIME_BUDGET_HOURS,
    get_prompt,
    wrap_job_description,
)
from langchain_streamlit_app.schemas import JobAnalysis
from langchain_streamlit_app.settings import ModelSettings

DEFAULT_HOURS_PER_DAY = 2.0
DEFAULT_STUDY_DAYS = 7
MIN_STUDY_DAYS = 1
MAX_STUDY_DAYS = 60


class AnalysisError(Exception):
    """A readable error for the UI (bad input, failed request, unparsable output)."""


class InputRejected(AnalysisError):
    """The input guards blocked the job description; `guard` says which rule fired."""

    def __init__(self, guard: GuardResult):
        super().__init__(guard.reason or "The job description was rejected.")
        self.guard = guard


@dataclass(frozen=True)
class UsageInfo:
    """Token usage, cost and latency of one analysis run (fields may be missing)."""

    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    model_name: str | None = None
    provider: str | None = None
    model_calls: int = 0


@dataclass(frozen=True)
class AnalysisResult:
    """Parsed analysis plus run metadata."""

    analysis: JobAnalysis
    usage: UsageInfo
    prompt_key: str
    study_days: int
    time_budget_hours: float


def time_budget_for(study_days: int, hours_per_day: float = DEFAULT_HOURS_PER_DAY) -> float:
    """Total study hours for `study_days`, capped at the prompt's maximum budget."""
    if not MIN_STUDY_DAYS <= study_days <= MAX_STUDY_DAYS:
        raise AnalysisError(
            f"Days until the interview must be between {MIN_STUDY_DAYS} and {MAX_STUDY_DAYS}."
        )
    if hours_per_day <= 0:
        raise AnalysisError("Study hours per day must be greater than 0.")
    return min(study_days * hours_per_day, MAX_TIME_BUDGET_HOURS)


def build_user_message(jd: str, study_days: int, time_budget_hours: float) -> str:
    """User message: interview timing (trusted) + the JD wrapped in delimiter tags."""
    day_word = "day" if study_days == 1 else "days"
    return (
        f"The interview is in {study_days} {day_word}. "
        f"Total study time budget: {time_budget_hours:g} hours.\n\n"
        f"{wrap_job_description(jd)}"
    )


def build_analysis_agent(settings: ModelSettings, prompt_key: str, time_budget_hours: float):
    """Agent with the rendered system prompt and `JobAnalysis` as structured output."""
    system_prompt = get_prompt(prompt_key).render(time_budget_hours)
    return build_agent(
        build_chat_model(settings), system_prompt=system_prompt, response_format=JobAnalysis
    )


def analyze_with_metadata(
    jd: str,
    settings: ModelSettings,
    prompt_key: str,
    study_days: int,
    *,
    hours_per_day: float = DEFAULT_HOURS_PER_DAY,
    agent: Any = None,
) -> AnalysisResult:
    """Run the full pipeline and return the analysis with usage/latency metadata.

    Raises `InputRejected` if a guard blocks the JD and `AnalysisError` with a
    readable message on any other failure.
    """
    guard = run_guards(jd)
    if not guard.ok:
        raise InputRejected(guard)
    jd = guard.sanitized_text
    try:
        get_prompt(prompt_key)
    except KeyError as exc:
        raise AnalysisError(str(exc.args[0])) from None
    hours = time_budget_for(study_days, hours_per_day)

    if agent is None:
        agent = build_analysis_agent(settings, prompt_key, hours)

    message = build_user_message(jd, study_days, hours)
    start = time.perf_counter()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    except Exception as exc:
        raise AnalysisError(f"The model request failed: {_describe(exc)}") from exc
    latency = time.perf_counter() - start

    analysis = result.get("structured_response")
    if not isinstance(analysis, JobAnalysis):
        raise AnalysisError(
            "The model did not return a structured analysis. Try again, raise max tokens, "
            "or pick a model that supports structured output."
        )
    return AnalysisResult(
        analysis=analysis,
        usage=collect_usage(result.get("messages", []), latency),
        prompt_key=prompt_key,
        study_days=study_days,
        time_budget_hours=hours,
    )


def analyze(jd: str, settings: ModelSettings, prompt_key: str, study_days: int) -> JobAnalysis:
    """Analyse a job description and return only the parsed `JobAnalysis`."""
    return analyze_with_metadata(jd, settings, prompt_key, study_days).analysis


def collect_usage(messages: list[Any], latency_s: float) -> UsageInfo:
    """Sum token usage and cost over all model calls in the run."""
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    totals: dict[str, int] = {}
    reasoning = 0
    has_reasoning = False
    cost = 0.0
    has_cost = False
    for msg in ai_messages:
        usage = msg.usage_metadata or {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if key in usage:
                totals[key] = totals.get(key, 0) + usage[key]
        details = usage.get("output_token_details") or {}
        if "reasoning" in details:
            reasoning += details["reasoning"]
            has_reasoning = True
        if isinstance(msg.response_metadata.get("cost"), (int, float)):
            cost += msg.response_metadata["cost"]
            has_cost = True
    last_meta = ai_messages[-1].response_metadata if ai_messages else {}
    return UsageInfo(
        latency_s=latency_s,
        input_tokens=totals.get("input_tokens"),
        output_tokens=totals.get("output_tokens"),
        total_tokens=totals.get("total_tokens"),
        reasoning_tokens=reasoning if has_reasoning else None,
        cost_usd=cost if has_cost else None,
        model_name=last_meta.get("model_name"),
        provider=last_meta.get("provider"),
        model_calls=len(ai_messages),
    )


def _describe(exc: Exception) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def total_study_hours(analysis: JobAnalysis) -> float:
    return sum(item.estimated_hours for item in analysis.study_plan)


def to_markdown(result: AnalysisResult) -> str:
    """Render the analysis as a Markdown report."""
    a = result.analysis
    lines = [
        f"# Interview prep: {a.role_title}",
        "",
        f"- Seniority: {a.seniority}",
        f"- Interview in: {result.study_days} day(s)",
        f"- Study budget: {result.time_budget_hours:g} h "
        f"(planned: {total_study_hours(a):g} h)",
        "",
        "## Key skills",
        "",
        "| Skill | Category | Priority | Proficiency | How it's mentioned | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for s in a.skills:
        mentioned = "Implied" if s.inferred else "Explicit"
        lines.append(
            f"| {_cell(s.name)} | {s.category} | {s.priority} | {s.proficiency} | {mentioned} "
            f"| {_cell(s.evidence)} |"
        )
    lines += ["", "## Likely interview topics", ""]
    for t in a.interview_topics:
        lines += [f"### {t.topic} ({t.difficulty})", "", t.why_likely, ""]
        lines += [f"{i}. {q}" for i, q in enumerate(t.example_questions, start=1)]
        lines.append("")
    lines += ["## Study plan", ""]
    for item in a.study_plan:
        lines += [f"### {item.day_or_block}: {item.focus} ({item.estimated_hours:g} h)", ""]
        lines += [f"- {act}" for act in item.activities]
        if item.resources:
            lines.append(f"- Resources: {', '.join(item.resources)}")
        lines.append("")
    if a.reasoning:
        lines += ["## Reasoning", "", a.reasoning, ""]
    return "\n".join(lines).rstrip() + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
