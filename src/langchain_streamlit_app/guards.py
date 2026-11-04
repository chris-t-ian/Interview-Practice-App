"""Input guards that run before any model call.

Three layers, each returning a `GuardResult` instead of raising:

1. Input validation: empty, too short/long, mostly non-text, "looks like a JD"
   heuristic (the last one only warns).
2. Prompt-injection filter: normalise the text (NFKC, drop invisible format
   characters, lowercase, collapse whitespace) and match high-confidence attack
   patterns in English and German. A match blocks the input and names the rule.
3. Rate limit: max N analyses per time window. The UI keeps the timestamp list
   in `st.session_state`; this module only does the bookkeeping.

Regex filters are easy to bypass (paraphrases, other languages, homoglyphs), so
this is only the first layer. The JD is also wrapped in delimiter tags (which
`neutralize_delimiters` keeps intact) and every system prompt tells the model to
treat the JD as untrusted data.

No Streamlit imports.
"""

import re
import time
import unicodedata
from dataclasses import dataclass, field

# ---- Limits ----------------------------------------------------------------
MIN_CHARS = 200
MAX_CHARS = 15_000
# Share of letters among non-whitespace characters. Typical JDs are ~0.8+.
MIN_LETTER_RATIO = 0.6
# Distinct JD keywords needed before we stop warning "this may not be a JD".
JD_KEYWORD_THRESHOLD = 3
RATE_LIMIT_MAX = 5
RATE_LIMIT_WINDOW_S = 60.0

TAG_REPLACEMENT = "[tag removed]"


@dataclass(frozen=True)
class GuardResult:
    """Outcome of a guard check.

    `ok` False means block; `reason` is a user-facing message and `rule` the id of
    the rule that fired. `warnings` are non-blocking notices. `sanitized_text` is
    the text to send on to the model (empty when blocked).
    """

    ok: bool
    reason: str | None = None
    sanitized_text: str = ""
    rule: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class InjectionMatch:
    rule: str
    description: str
    matched_text: str


# ---- Normalisation and sanitising ------------------------------------------


def _strip_invisible(text: str) -> str:
    """Drop format chars (zero-width space/joiner, BOM, soft hyphen, bidi marks)
    and control chars except newline/tab."""
    return "".join(
        ch
        for ch in text
        if ch in "\n\t" or unicodedata.category(ch) not in ("Cf", "Cc", "Co", "Cs")
    )


def normalize(text: str) -> str:
    """Matching form: NFKC, invisible chars removed, lowercase, single spaces."""
    text = _strip_invisible(unicodedata.normalize("NFKC", text))
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _normalize_lines(text: str) -> list[str]:
    """Like `normalize`, but per line (for line-anchored rules like '### system')."""
    text = _strip_invisible(unicodedata.normalize("NFKC", text)).casefold()
    return [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]


# Our delimiter tag in any spelling: </job_description>, < job-description >,
# <JOB DESCRIPTION attr="x">, and HTML-escaped &lt;/job_description&gt;.
_DELIMITER_RE = re.compile(
    r"(?:<|&lt;)\s*/?\s*job[\s_\-]*description\b[^>&]*(?:>|&gt;)", re.IGNORECASE
)


def neutralize_delimiters(text: str) -> str:
    """Replace our delimiter tags so the text cannot break out of the JD block."""
    return _DELIMITER_RE.sub(TAG_REPLACEMENT, text)


def sanitize(text: str) -> str:
    """Text forwarded to the model: NFKC, invisible chars removed, line endings
    normalised, delimiter tags neutralised. Case and line breaks are kept."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return neutralize_delimiters(_strip_invisible(text)).strip()


# ---- Prompt-injection rules ------------------------------------------------
# Each rule: (id, description, compiled pattern). Patterns run on `normalize()`
# output (lowercase, single spaces). They are deliberately narrow: common JD
# phrases such as "act as a point of contact", "you are now part of the team" or
# "experience writing system prompts" must NOT match.

_FILLER_WORDS = r"(?:[\w'’-]+ ){0,3}"  # up to three filler words

_TARGETS_EN = (
    r"(?:instructions?|prompts?|rules|guidelines|directives|directions|commands|context|"
    r"constraints)"
)
_TARGETS_DE = (
    r"(?:anweisungen|instruktionen|regeln|vorgaben|befehle|richtlinien|prompts?|anordnungen)"
)

_INJECTION_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        "Asks the model to ignore/forget its previous instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget) (?:" + _FILLER_WORDS + r")?"
            r"(?:all|your|these|those|previous|prior|above|earlier|preceding|"
            r"foregoing|initial|original|system)\b ?" + _FILLER_WORDS + _TARGETS_EN + r"\b"
        ),
    ),
    (
        "ignore_instructions_de",
        "Fordert das Modell auf, bisherige Anweisungen zu ignorieren",
        re.compile(
            r"\b(?:ignorier(?:e|en|t)?|vergiss|vergesst|vergessen sie|missachte(?:n)?|"
            r"übergeh(?:e|en)?|überschreib(?:e|en)?) (?:" + _FILLER_WORDS + r")?"
            r"(?:alle|sämtliche|deine|eure|vorherigen?|bisherigen?|obigen?|"
            r"vorangegangenen?|früheren?|ursprünglichen?|system)\b ?"
            + _FILLER_WORDS
            + _TARGETS_DE
            + r"\b"
        ),
    ),
    (
        "role_override",
        "Tries to give the model a new identity ('you are now …', 'from now on you …')",
        re.compile(
            r"\byou are now (?:(?:a|an|the|my) )?(?:[\w-]+ ){0,2}"
            r"(?:assistant|ai|chatbot|bot|model|llm|gpt|dan|jailbroken|unrestricted|"
            r"unfiltered|uncensored|evil|hacker|in (?:developer|god|dan|jailbreak) mode)(?![\w-])"
            r"(?! (?:engineer|researcher|developer|specialist|expert|team|product|lead))"
            r"|\byou are no longer (?:an? |the )?(?:assistant|ai|bound|restricted|limited|"
            r"required|a job description analy[sz]er)\b"
            r"|\bfrom now on,? you (?:will |must |shall |should )?(?:only (?:answer|respond|reply|"
            r"output|say)|(?:answer|respond|reply) (?:only|with)|ignore|obey|act as|"
            r"are (?:an? )?(?:ai|assistant|bot|dan|unrestricted|free|no longer))\b"
            r"|\b(?:enable|enter|activate|switch to) (?:developer|god|dan|jailbreak) mode\b"
            r"|\bdo anything now\b"
        ),
    ),
    (
        "role_override_de",
        "Versucht dem Modell eine neue Rolle zu geben ('du bist jetzt …')",
        re.compile(
            r"\b(?:du bist|sie sind) (?:jetzt|nun|ab sofort|ab jetzt) (?:(?:ein|eine|der|die) )?"
            r"(?:[\w-]+ ){0,2}(?:assistent(?:in)?|ki|chatbot|bot|modell|sprachmodell|dan|"
            r"uneingeschränkt|ungefiltert|hacker)\b"
            r"|\b(?:ab sofort|von nun an|ab jetzt),? (?:antwortest du|ignorierst du|"
            r"(?:sollst|wirst|musst) du nur|bist du (?:ein(?:e)? )?(?:[\w-]+ )?"
            r"(?:ki|assistent|bot|pirat|dan|modell|hacker|uneingeschränkt))\b"
            r"|\bdu bist kein (?:assistent|analy[sz]er|modell) mehr\b"
            r"|\b(?:entwickler|developer|god|dan)[- ]?modus\b"
        ),
    ),
    (
        "act_as_persona",
        "Asks the model to role-play as something else",
        re.compile(
            r"\bact as (?:if|though) you (?:are|were|have|had) (?:no|not|an?|unrestricted|"
            r"unlimited|free)\b"
            r"|\bact as (?:an? )?(?:unrestricted|unfiltered|uncensored|jailbroken|evil|"
            r"different|new) (?:ai|assistant|model|chatbot|bot|llm)\b"
            r"|\bact as (?:an? )?(?:dan|linux terminal|python interpreter|shell)\b"
            r"|\bpretend (?:to be|you are|you're|that you are) (?:an? |the )?(?:[\w-]+ ){0,2}"
            r"(?:ai|assistant|model|bot|dan|unrestricted|human|developer|system)\b"
            r"|\b(?:roleplay|role-play) as\b"
        ),
    ),
    (
        "act_as_persona_de",
        "Fordert Rollenspiel als etwas anderes",
        re.compile(
            r"\b(?:tu|tue|tun sie) so,? als (?:ob du|wärst du|ob sie|wären sie|seist du)\b"
            r"|\bhandle als (?:ein(?:e)? )?(?:uneingeschränkte|andere|neue)?\s?"
            r"(?:ki|assistent|modell|bot)\b"
            r"|\bgib dich als .{0,40} aus\b"
            r"|\bspiele? die rolle (?:eine[sr]?|des|der) .{0,30}"
            r"(?:ki|assistent|modell|bot|hacker)\b"
        ),
    ),
    (
        "prompt_exfiltration",
        "Asks the model to reveal its system prompt or instructions",
        re.compile(
            r"\b(?:reveal|print|show|display|output|repeat|leak|dump|disclose|expose|recite|"
            r"tell me|give me|write out|spell out|what (?:is|are|were)) (?:me |us )?"
            + _FILLER_WORDS
            + r"(?:your|thy) (?:system prompt|system message|initial prompt|hidden prompt|"
            r"original prompt|instructions|prompt)\b"
            r"|\b(?:reveal|print|repeat|leak|dump|disclose|expose|recite|output) (?:me |us )?"
            r"(?:"
            + _FILLER_WORDS
            + r")?(?:the |this )?(?:above |previous |hidden |secret |initial |original )?"
            r"(?:system prompt|system message|initial prompt|hidden prompt|original prompt|"
            r"(?:above|hidden|secret|initial|original) instructions)\b"
            r"|\b(?:everything|all text|the text|all) (?:above|before) (?:this|these) "
            r"(?:line|message|text|sentence)\b"
        ),
    ),
    (
        "prompt_exfiltration_de",
        "Fordert das Modell auf, seinen Systemprompt preiszugeben",
        re.compile(
            r"\b(?:zeig(?:e|en sie)?|gib|nenne?|verrate?|wiederhole?|druck(?:e)?|enthülle?) "
            r"(?:mir |uns )?"
            + _FILLER_WORDS
            + r"(?:deine[n]?|ihre[n]?|den|die (?:geheimen|versteckten|ursprünglichen|obigen|"
            r"vorherigen)) (?:system[- ]?prompts?|system[- ]?nachricht|anweisungen|instruktionen|"
            r"regeln|vorgaben|prompts?)\b"
        ),
    ),
    (
        "new_instructions",
        "Introduces a new set of instructions",
        re.compile(
            r"\b(?:new|updated|real|actual|revised|override|admin|developer) "
            r"(?:system )?(?:instructions|prompt) ?:"
            r"|\b(?:neue|aktualisierte|echte|eigentliche|geänderte) "
            r"(?:anweisungen|instruktionen) ?:"
        ),
    ),
    (
        "chat_role_tag",
        "Contains chat-template role tags (e.g. <|system|>, [INST], <<SYS>>)",
        re.compile(
            r"<\|\s*(?:system|assistant|user|developer|im_start|im_end|endoftext|"
            r"start_header_id|end_header_id|eot_id)\s*\|>"
            r"|\[/?inst\]|<</?sys>>"
            r"|</?\s*(?:system_prompt|system_instructions|system_message)\s*>"
        ),
    ),
    (
        "delimiter_breakout",
        "Contains our <job_description> delimiter tag (attempt to break out of the data block)",
        _DELIMITER_RE,
    ),
]

# Line-anchored rule: a markdown header that opens a fake system/assistant turn.
_ROLE_HEADER_RE = re.compile(
    r"^#{1,6} ?(?:system|assistant|developer|system prompt|system message|"
    r"new instructions?|neue anweisungen|systemanweisung(?:en)?) ?:?$"
)
_ROLE_HEADER_RULE = ("markdown_role_header", "Markdown header posing as a system/assistant turn")


def detect_injection(text: str) -> InjectionMatch | None:
    """Return the first high-confidence injection pattern found, or None."""
    norm = normalize(text)
    for rule, description, pattern in _INJECTION_RULES:
        if m := pattern.search(norm):
            return InjectionMatch(rule, description, m.group(0))
    for line in _normalize_lines(text):
        if _ROLE_HEADER_RE.match(line):
            return InjectionMatch(*_ROLE_HEADER_RULE, line)
    return None


# ---- JD heuristic ----------------------------------------------------------
# Stems, matched as substrings of the normalised text so German compounds
# ("Berufserfahrung", "Anforderungsprofil") count too.
JD_KEYWORDS = (
    # English
    "responsibilit",
    "requirement",
    "experience",
    "qualification",
    "skills",
    "you will",
    "what you'll",
    "we offer",
    "benefits",
    "about the role",
    "about you",
    "degree",
    "candidate",
    "apply",
    "salary",
    "full-time",
    "part-time",
    "position",
    "nice to have",
    "must have",
    "job description",
    # German
    "aufgaben",
    "anforderung",
    "erfahrung",
    "qualifikation",
    "kenntnisse",
    "profil",
    "wir bieten",
    "bewerb",
    "stelle",
    "studium",
    "ausbildung",
    "(m/w/d)",
    "vollzeit",
    "teilzeit",
    "verantwortung",
    "das bringst du mit",
    "was dich erwartet",
)


def jd_keyword_hits(text: str) -> list[str]:
    norm = normalize(text)
    return [kw for kw in JD_KEYWORDS if kw in norm]


def letter_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(c.isalpha() for c in chars) / len(chars)


# ---- Public entry points ---------------------------------------------------


def _block(rule: str, reason: str) -> GuardResult:
    return GuardResult(ok=False, reason=reason, rule=rule)


def validate_input(text: str | None) -> GuardResult:
    """Length / content checks. Passes with warnings if it doesn't look like a JD."""
    if text is None or not text.strip():
        return _block("empty", "Please paste a job description first.")
    stripped = text.strip()
    if len(stripped) > MAX_CHARS:
        return _block(
            "too_long",
            f"The job description is too long ({len(stripped):,} characters, "
            f"max {MAX_CHARS:,}). Remove boilerplate such as company history or legal text.",
        )
    if len(stripped) < MIN_CHARS:
        return _block(
            "too_short",
            f"The job description is too short ({len(stripped)} characters, "
            f"min {MIN_CHARS}). Paste the full posting.",
        )
    ratio = letter_ratio(stripped)
    if ratio < MIN_LETTER_RATIO:
        return _block(
            "not_text",
            f"This doesn't look like normal text (only {ratio:.0%} letters). "
            "Paste the job description as plain text.",
        )
    warnings: tuple[str, ...] = ()
    hits = jd_keyword_hits(stripped)
    if len(hits) < JD_KEYWORD_THRESHOLD:
        warnings = (
            "This may not be a job description: few typical terms such as "
            "responsibilities, requirements or experience were found. The analysis "
            "may be less useful.",
        )
    return GuardResult(ok=True, sanitized_text=sanitize(stripped), warnings=warnings)


def check_injection(text: str) -> GuardResult:
    match = detect_injection(text)
    if match is None:
        return GuardResult(ok=True, sanitized_text=sanitize(text))
    return _block(
        f"injection:{match.rule}",
        f"Possible prompt injection detected ({match.description}). "
        f"Matched: “{match.matched_text[:80]}”. Remove that part and try again.",
    )


def run_guards(text: str | None) -> GuardResult:
    """All content guards in order. The first failure wins.

    The injection filter runs before the length checks so that a short attack
    string reports the injection rule rather than just "too short".
    """
    if text is None or not text.strip():
        return validate_input(text)
    if len(text.strip()) > MAX_CHARS:  # bound the work before running regexes
        return validate_input(text)
    injection = check_injection(text)
    if not injection.ok:
        return injection
    return validate_input(text)


def check_rate_limit(
    history: list[float],
    *,
    now: float | None = None,
    max_requests: int = RATE_LIMIT_MAX,
    window_s: float = RATE_LIMIT_WINDOW_S,
) -> GuardResult:
    """Check (without recording) whether another request fits in the window.

    `history` holds the timestamps of earlier requests; entries older than the
    window are pruned in place.
    """
    now = time.monotonic() if now is None else now
    history[:] = [t for t in history if now - t < window_s]
    if len(history) < max_requests:
        return GuardResult(ok=True)
    retry = window_s - (now - history[0])
    return _block(
        "rate_limit",
        f"Rate limit reached: at most {max_requests} analyses per {window_s:g} s. "
        f"Try again in {max(1, round(retry))} s.",
    )


def record_request(history: list[float], *, now: float | None = None) -> None:
    history.append(time.monotonic() if now is None else now)
