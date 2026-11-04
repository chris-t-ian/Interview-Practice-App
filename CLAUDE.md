# Project rules

- Never read, print, or echo the contents of `.env` or any API key/secret (e.g. `OPENROUTER_API_KEY`). Refer to keys only by variable name.

# Overview

Minimal Streamlit Q&A app: the user types a question, and the answer comes from a model called through LangChain (`create_agent`, no tools) via OpenRouter.

- `streamlit_app.py`: the whole app (form, cached agent, error handling).
- `src/langchain_streamlit_app/`: placeholder package from `uv init` that isn't used by the app.

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
- Cache the agent with `@st.cache_resource`. Show errors in the UI with `st.error`.
- Use the LangChain and Streamlit skills in `.claude/skills/` when changing LangChain or Streamlit code.
