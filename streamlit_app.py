"""Minimal Streamlit Q&A app that calls a model through LangChain + OpenRouter."""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent

# Loads OPENROUTER_API_KEY (and optionally OPENROUTER_MODEL) from .env at runtime.
load_dotenv()

# Any OpenRouter model id works; override with OPENROUTER_MODEL in .env.
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.5")


@st.cache_resource
def get_agent():
    return create_agent(
        model=f"openrouter:{MODEL}",
        tools=[],
        system_prompt="You are a helpful assistant. Answer clearly and concisely.",
    )


st.title("Ask a question")
st.caption(f"Model: `{MODEL}` via OpenRouter")

if not os.getenv("OPENROUTER_API_KEY"):
    st.error("Add `OPENROUTER_API_KEY` to your `.env` file.", title="Missing API key")
    st.stop()

with st.form("question_form"):
    question = st.text_area("Your question")
    submitted = st.form_submit_button("Ask", icon=":material/send:")

if submitted and question.strip():
    try:
        with st.spinner("Thinking..."):
            result = get_agent().invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
    except Exception as exc:  # surface API/model errors in the UI
        st.error(str(exc), title="The model request failed")
    else:
        with st.container(border=True):
            st.markdown(result["messages"][-1].text)
