"""Analysis pipeline: guards -> prompt -> model -> parsed `JobAnalysis`.

No Streamlit imports, so this module can be tested on its own. The content
guards (guards.py) run at the start of `analyze_with_metadata`, before the agent
is built or called; only the sanitised text reaches the model.

The UI caches the agent itself (see `build_analysis_agent`) and passes it in;
without an agent argument, a fresh one is built for each call.

Every error leaves this module as an `AnalysisError` with a readable message
that has been passed through `redact_secrets`.
"""

import time
from dataclasses import dataclass
from typing import Any

import httpx
from langchain.agents.structured_output import StructuredOutputError
from langchain_core.messages import AIMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.errors import GraphRecursionError
from openrouter.errors import NoResponseError

from langchain_streamlit_app.guards import GuardResult, run_guards
from langchain_streamlit_app.llm import Agent, build_agent, build_chat_model
from langchain_streamlit_app.prompts import (
    MAX_TIME_BUDGET_HOURS,
    get_prompt,
    render_system_prompt,
    wrap_job_description,
)
from langchain_streamlit_app.schemas import JobAnalysis
from langchain_streamlit_app.settings import (
    API_KEY_ENV,
    ModelSettings,
    has_api_key,
    redact_secrets,
)

DEFAULT_HOURS_PER_DAY = 2.0
DEFAULT_STUDY_DAYS = 7
MIN_STUDY_DAYS = 1
MAX_STUDY_DAYS = 60

# Structured output fails -> retry once. LangChain picks the strategy per model:
# - provider-native JSON schema: a parse failure raises `StructuredOutputError`,
#   and we retry once ourselves (SCHEMA_ATTEMPTS).
# - tool-call fallback: LangChain retries inside the graph. A run whose N-th
#   model call succeeds needs N + 1 graph steps, so a limit of 3 allows exactly
#   one in-graph retry; hitting it raises `GraphRecursionError`, which we treat
#   as final (the retry has already happened).
SCHEMA_ATTEMPTS = 2
RECURSION_LIMIT = 3
MAX_ERROR_DETAIL_CHARS = 300


class AnalysisError(Exception):
    """A readable error for the UI (bad input, failed request, unparsable output)."""


class InputRejected(AnalysisError):
    """The input guards blocked the job description; `guard` says which rule fired."""

    def __init__(self, guard: GuardResult) -> None:
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
    attempts: int = 1


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


def build_analysis_agent(
    chat_model: ChatOpenRouter, prompt_key: str, time_budget_hours: float
) -> Agent:
    """Agent with the rendered system prompt and `JobAnalysis` as structured output."""
    return build_agent(
        chat_model,
        system_prompt=render_system_prompt(prompt_key, time_budget_hours),
        response_format=JobAnalysis,
    )


def analyze_with_metadata(
    jd: str,
    settings: ModelSettings,
    prompt_key: str,
    study_days: int,
    *,
    hours_per_day: float = DEFAULT_HOURS_PER_DAY,
    agent: Agent | None = None,
) -> AnalysisResult:
    """Run the full pipeline and return the analysis with usage/latency metadata.

    If the model's output does not match the schema, the request is retried
    once. Raises `InputRejected` if a guard blocks the JD and `AnalysisError`
    with a readable (secret-free) message on any other failure.
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
        agent = build_analysis_agent(_build_chat_model_or_raise(settings), prompt_key, hours)

    payload = {"messages": [{"role": "user", "content": build_user_message(jd, study_days, hours)}]}
    messages: list[Any] = []  # all messages over all attempts, for usage/cost
    problem = ""
    start = time.perf_counter()
    for attempt in range(1, SCHEMA_ATTEMPTS + 1):
        try:
            result = agent.invoke(payload, config={"recursion_limit": RECURSION_LIMIT})
        except StructuredOutputError as exc:
            messages.append(exc.ai_message)
            problem = str(exc)
            continue
        except GraphRecursionError:
            problem = "the model repeatedly returned output that does not match the schema"
            break
        except Exception as exc:
            raise AnalysisError(describe_request_error(exc, settings)) from exc
        messages.extend(result.get("messages", []))
        analysis = result.get("structured_response")
        if isinstance(analysis, JobAnalysis):
            return AnalysisResult(
                analysis=analysis,
                usage=collect_usage(messages, time.perf_counter() - start, attempts=attempt),
                prompt_key=prompt_key,
                study_days=study_days,
                time_budget_hours=hours,
            )
        problem = "the response contained no structured analysis"

    raise AnalysisError(
        "The model's output did not match the expected format, even after one retry "
        f"({_shorten(redact_secrets(problem))}). Try again, raise max tokens, or pick a "
        "model that supports structured output."
    )


def _build_chat_model_or_raise(settings: ModelSettings) -> ChatOpenRouter:
    """Build the chat model, turning a missing key or bad setting into `AnalysisError`."""
    if not has_api_key():
        raise AnalysisError(f"{API_KEY_ENV} is not set. Add it to your .env file.")
    try:
        return build_chat_model(settings)
    except ValueError as exc:
        raise AnalysisError(
            f"Could not build the model: {redact_secrets(_format_exception(exc))}"
        ) from exc


def describe_request_error(exc: BaseException, settings: ModelSettings) -> str:
    """User-facing message for a failed model request (secrets redacted)."""
    status = getattr(exc, "status_code", None)
    detail = _shorten(_format_exception(exc))
    if isinstance(exc, httpx.TimeoutException) or status in (408, 524):
        message = (
            f"The model request timed out after {settings.timeout} s. Raise the timeout in "
            "the sidebar, lower max tokens, or try a faster model."
        )
    elif status == 401:
        message = f"OpenRouter rejected the API key (401). Check {API_KEY_ENV} in your .env file."
    elif status == 402:
        message = "Your OpenRouter account has too few credits for this request (402)."
    elif status == 404:
        message = (
            f"Model `{settings.model_id}` was not found on OpenRouter (404). Check the model id."
        )
    elif status == 400:
        message = (
            f"OpenRouter rejected the request (400): {detail}. Check the model id and "
            "whether the model supports the chosen settings."
        )
    elif status == 429:
        message = "Rate limited by OpenRouter or the model provider (429). Wait a moment and retry."
    elif isinstance(status, int) and status >= 500:
        message = (
            f"OpenRouter or the model provider returned an error ({status}). "
            "Try again or pick another model."
        )
    elif isinstance(exc, (httpx.NetworkError, NoResponseError)):
        message = "Could not reach OpenRouter. Check your internet connection and try again."
    else:
        message = f"The model request failed: {detail}"
    return redact_secrets(message)


def collect_usage(messages: list[Any], latency_s: float, *, attempts: int = 1) -> UsageInfo:
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
        attempts=attempts,
    )


def _format_exception(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _shorten(text: str, limit: int = MAX_ERROR_DETAIL_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def total_study_hours(analysis: JobAnalysis) -> float:
    """Sum of `estimated_hours` over the study plan."""
    return sum(item.estimated_hours for item in analysis.study_plan)


def to_markdown(result: AnalysisResult) -> str:
    """Render the analysis as a Markdown report."""
    analysis = result.analysis
    lines = [
        f"# Interview prep: {analysis.role_title}",
        "",
        f"- Seniority: {analysis.seniority}",
        f"- Interview in: {result.study_days} day(s)",
        f"- Study budget: {result.time_budget_hours:g} h "
        f"(planned: {total_study_hours(analysis):g} h)",
        "",
        "## Key skills",
        "",
        "| Skill | Category | Priority | Proficiency | How it's mentioned | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for skill in analysis.skills:
        lines.append(
            f"| {_cell(skill.name)} | {skill.category} | {skill.priority} | "
            f"{skill.proficiency} | {skill.mentioned} | {_cell(skill.evidence)} |"
        )
    lines += ["", "## Likely interview topics", ""]
    for topic in analysis.interview_topics:
        lines += [f"### {topic.topic} ({topic.difficulty})", "", topic.why_likely, ""]
        lines += [f"{n}. {q}" for n, q in enumerate(topic.example_questions, start=1)]
        lines.append("")
    lines += ["## Study plan", ""]
    for item in analysis.study_plan:
        lines += [f"### {item.day_or_block}: {item.focus} ({item.estimated_hours:g} h)", ""]
        lines += [f"- {act}" for act in item.activities]
        if item.resources:
            lines.append(f"- Resources: {', '.join(item.resources)}")
        lines.append("")
    if analysis.reasoning:
        lines += ["## Reasoning", "", analysis.reasoning, ""]
    return "\n".join(lines).rstrip() + "\n"


def _cell(text: str) -> str:
    """Escape text for a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")
