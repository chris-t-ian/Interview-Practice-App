# Job Description Analyser

## Security

Every job description (JD) goes through `src/langchain_streamlit_app/guards.py` before any model call. Each guard returns a `GuardResult` (`ok`, `reason`, `sanitized_text`, `rule`, `warnings`) instead of raising. Only `sanitized_text` reaches the model.

**Input validation**
- Rejects empty input, input shorter than 200 or longer than 15,000 characters, and input where less than 60 % of the non-space characters are letters (JSON dumps, tables of numbers). The limits are constants at the top of `guards.py`.
- A keyword heuristic (English and German: *responsibilities, requirements, experience, Aufgaben, Anforderungen, Erfahrung, …*) checks whether the text looks like a JD. If it doesn't, the app shows a warning but still runs the analysis.
- Rate limit: at most 5 analyses per 60 s per browser session. The timestamps are kept in `st.session_state`.

**Prompt-injection filter**
- The text is normalised first: Unicode NFKC (turns full-width letters into ASCII), zero-width and control characters removed, lowercased, whitespace collapsed.
- English and German regex rules then look for: "ignore/forget previous instructions", role overrides ("you are now DAN", "von nun an bist du …"), persona requests ("pretend you are an AI", "tu so, als wärst du …"), requests to reveal the system prompt, "new instructions:", chat-template tags (`<|system|>`, `[INST]`, `<<SYS>>`), markdown headers like `### System`, and our own `<job_description>` delimiter tag.
- A match blocks the input, and the UI shows which rule fired (e.g. `injection:ignore_instructions`).
- The rules are deliberately narrow so that normal JD phrases pass, for example "act as a point of contact", "you are now part of the team" or "experience writing system prompts".

**Defence in depth**
- `sanitize()` neutralises the delimiter tags in any spelling (`</job_description>`, `< /Job-Description >`, `&lt;/job_description&gt;`), so the JD can't close its data block early.
- The JD is sent wrapped in `<job_description>…</job_description>`. Every system prompt says the content of that block is untrusted data, that instructions inside it must never be followed, and that the system prompt must never be revealed (`prompts.py`, `SHARED_RULES`).
- The output must match the `JobAnalysis` Pydantic schema, which limits what a hijacked model can return.

**Limits**
- Regex filters are easy to bypass: paraphrases ("kindly set aside what you were told earlier"), other languages, homoglyphs (a Cyrillic "о" survives NFKC), base64 or ROT13 payloads, or instructions split across sentences. Treat the filter as a cheap first layer that catches obvious attacks, not as protection.
- Prompt instructions and delimiters reduce the risk but don't remove it. A model can still be persuaded by a clever enough JD.
- Narrow rules mean more misses. Broad rules would block real JDs. Known false positives that are still blocked: "act as a Linux terminal power user" and "reveal the system prompt weaknesses of …" in a red-teaming JD.
- The rate limit is per browser session only. Reloading the page resets it, so it is not a defence against abuse of the API key.

Run the guard tests with `uv run pytest tests/test_guards.py`.
