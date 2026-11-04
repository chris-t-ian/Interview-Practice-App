# Project rules

- Never read, print, or echo the contents of `.env` or any API key/secret (e.g. `OPENROUTER_API_KEY`). Refer to keys only by variable name.

# Overview

**Job Description Analyser**: a Streamlit app for students preparing for interviews. The user pastes a job description (JD), and the app returns:

1. **Key skills**: technical and soft skills, each with a priority.
2. **Likely interview topics**, each with example questions.
3. **A short study plan** to brush up on those topics.

The model is called through LangChain (`create_agent`, no tools) via OpenRouter. Main model: `openai/gpt-6-luna`.

Features:
- **Sidebar model settings**: model id, temperature, max tokens, top_p and similar, as sliders/fields.
- **At least 5 selectable system prompts**, each using a different prompting technique (e.g. zero-shot, few-shot, chain-of-thought, role/persona, structured-output/JSON schema).
- **Security guards**: input validation (empty/too short/too long, not a JD) and a prompt-injection filter that runs before the JD reaches the model.
- **Later (optional)**: a vector-DB novelty check on JDs (embeddings: `qwen/qwen3-embedding-8b` via OpenRouter), and LLM-as-a-judge evaluation of the output.

## Target module layout

`streamlit_app.py` stays the UI entry point (layout, sidebar, form, rendering, `st.error`). Logic lives in the package `src/langchain_streamlit_app/`:

- `schemas.py`: Pydantic models for the structured result (skills, interview topics + questions, study plan) and for the model settings.
- `settings.py`: defaults and allowed ranges for model settings, plus env loading (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`).
- `llm.py`: builds the chat model / agent from the settings (`openrouter:<model-id>`).
- `prompts.py`: the system prompt registry (name → technique description + prompt text), at least 5 entries.
- `guards.py`: input validation and the prompt-injection filter; returns a clear pass/fail result with a reason.
- `analyzer.py`: the pipeline (guards → prompt → model → parsed result); no Streamlit imports, so it can be tested on its own.
- Later: `novelty.py` (vector DB + embeddings) and `judge.py` (LLM-as-a-judge).

# Setup & running

- Python 3.14 (`.python-version`), managed with **uv**.
- Install deps: `uv sync`
- Run: `uv run streamlit run streamlit_app.py`
- Config comes from `.env`, loaded with `python-dotenv`:
  - `OPENROUTER_API_KEY` (required)
  - `OPENROUTER_MODEL` (optional, any OpenRouter model id, default `openai/gpt-6-luna`)

# Conventions

- Dependencies are pinned to exact versions in `pyproject.toml`. Add new ones with `uv add <pkg>==<version>` so `uv.lock` stays in sync.
- Models are passed as `openrouter:<model-id>` strings to `create_agent` (needs `langchain-openrouter`).
- Cache the agent with `@st.cache_resource`, keyed on the model settings and the selected prompt, so changing a setting builds a new agent. Show errors in the UI with `st.error`.
- Keep Streamlit code in `streamlit_app.py`. Package modules don't import `streamlit`.
- Use the LangChain and Streamlit skills in `.claude/skills/` when changing LangChain or Streamlit code.
