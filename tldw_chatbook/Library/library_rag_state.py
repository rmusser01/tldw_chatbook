"""Pure display-state contracts for Library-native Search/RAG."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import html
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from rich.markup import escape as escape_markup

from tldw_chatbook.Library.library_rag_answer_service import LibraryRagAnswer
from tldw_chatbook.Utils.input_validation import (
    sanitize_string,
    validate_number_range,
    validate_text_input,
    validate_url,
)


LIBRARY_RAG_SOURCE_TYPES: tuple[tuple[str, str], ...] = (
    ("notes", "Notes"),
    ("media", "Media"),
    ("conversations", "Conversations"),
    ("prompts", "Prompts"),
    ("workspaces", "Workspaces"),
    ("collections", "Collections"),
)
# The one display-label vocabulary for raw source-type identifiers, shared
# by the Sources toggles (`scope_toggle_label`, "✓ Notes"/"✓ Media (1)"),
# the scope-summary strip (`library_rag_scope_summary`), and the Evidence
# region's coverage note (`library_rag_coverage_note`) -- before this table
# existed, the coverage note joined RAW identifiers ("notes, conversations")
# two lines below toggles reading capitalized labels, two vocabularies on
# one screen (controller amendment to Task 8, folded into RAG-32).
_LIBRARY_RAG_SOURCE_TYPE_LABELS: Mapping[str, str] = dict(LIBRARY_RAG_SOURCE_TYPES)
# The subset of LIBRARY_RAG_SOURCE_TYPES with a real per-source toggle in the
# Search canvas scope region (B2): workspaces/collections have no retrieval
# seam of their own yet, so they get no toggle row.
LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES: tuple[str, ...] = (
    "notes",
    "media",
    "conversations",
    "prompts",
)
LIBRARY_RAG_DEFAULT_TOP_K = 5
LIBRARY_RAG_RUN_ACTION_ID = "library-rag-run-query"
LIBRARY_RAG_USE_IN_CONSOLE_ACTION_ID = "library-rag-use-in-console"
LIBRARY_RAG_SERVICE_ERROR_SELECTOR = "library-rag-service-error"
LIBRARY_RAG_EMPTY_STATE_SELECTOR = "library-rag-empty-state"
LIBRARY_RAG_USE_IN_CONSOLE_DISABLED_REASON = (
    "Run a query and select usable evidence before sending to Console."
)
# The "#library-rag-scope-summary" strip text for the common case (every
# available source selected). `library_rag_scope_summary` below is the ONE
# source of truth both the panel's compose path
# (library_search_rag_panel._scope_summary) and the screen's incremental
# refresh path (LibraryScreen._library_rag_scope_summary) delegate to, so
# the two can't drift apart -- this constant is that builder's common-case
# return value, kept verbatim (RAG-32 review: it is the pre-existing,
# already-pinned copy). Per-source counts are deliberately absent from the
# strip in that case -- the scope toggle buttons directly below already
# carry them (L6); a real subset instead gets the explicit source list (and
# what's off) built by `library_rag_scope_summary`.
LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY = "Scope: all local sources"
LIBRARY_RAG_QUERY_MAX_LENGTH = 2_000
LIBRARY_RAG_DISPLAY_MAX_LENGTH = 1_000
LIBRARY_RAG_SNIPPET_MAX_LENGTH = 4_000
# The on-screen projection of a result row's snippet (`display_snippet`) is
# far shorter than the stored 4,000-char handoff/evidence payload above --
# live UAT found a low-relevance result rendering 25+ unclamped lines and
# burying later results (RAG-30/31). `snippet` itself is never clamped.
LIBRARY_RAG_SNIPPET_DISPLAY_MAX_CHARS = 320
# The Evidence region's quiet no-match line (RAG-33/Task 11) quotes the
# submitted query inline in one sentence -- a far shorter budget than the
# snippet clamp above, since this needs to stay a single row, not a
# multi-line card.
LIBRARY_RAG_EMPTY_QUERY_QUOTE_MAX_CHARS = 80
LIBRARY_RAG_TOP_K_MAX = 50
# Match-band thresholds (RAG-34): honest bands instead of a raw
# three-decimal cosine score. Both boundaries are inclusive on their upper
# band -- a score of exactly LIBRARY_RAG_MATCH_STRONG_THRESHOLD lands in
# "strong", and exactly LIBRARY_RAG_MATCH_MODERATE_THRESHOLD lands in
# "moderate" -- so `library_rag_score_suffix`/`library_rag_all_matches_weak`
# below are the single seam a future refactor must touch to shift them.
LIBRARY_RAG_MATCH_STRONG_THRESHOLD = 0.5
LIBRARY_RAG_MATCH_MODERATE_THRESHOLD = 0.2
LIBRARY_RAG_PROVENANCE_KEYS = frozenset(
    {
        "active_context_eligible",
        "active_workspace_id",
        "authority_label",
        "evidence_status",
        "eligibility_reason",
        "item_type",
        "source_type",
        "type",
        "workspace_id",
        "workspace_ids",
    }
)
_SCRIPT_BLOCK_PATTERN = re.compile(
    r"<script\b[^>]*>.*?</script\s*>",
    re.IGNORECASE | re.DOTALL,
)
LIBRARY_SEARCH_HISTORY_LIMIT = 10
LIBRARY_SEARCH_HISTORY_ENTRY_MAX_CHARS = 200
# Disabled-reason text for the two run-gate blockers that render as a single
# quiet line (A1) instead of the full callout + recovery-copy presentation.
# Both strings are unique across the gate ladder in
# `LibraryRagQueryState.from_values`, so `blocked_is_empty_query` /
# `blocked_is_no_scope` below can key off them directly.
_EMPTY_QUERY_DISABLED_REASON = "Enter a question or search query."
_NO_SCOPE_DISABLED_REASON = "Select at least one Library source."
# The whole no-sources presentation (L3a quiet-gate principle): ONE muted
# line plus the single "Open Import media" action -- never the old
# Unavailable/Why/Next/Recovery/Owner dump plus checklist, whose internal
# jargon ("Owner: Library source index") regressed the 2026-07 core-loop UAT.
LIBRARY_RAG_NO_SOURCES_GATE_COPY = (
    "No Library sources yet — import media or create notes, then search."
)
_NO_SOURCES_NEXT_ACTION = "Import media or create notes, then search."
LIBRARY_RAG_SEARCHING_LABEL = "Searching…"
#: PR-3 Task 3: the RAG Answer worker's in-flight label (mirrors
#: `LIBRARY_RAG_SEARCHING_LABEL`) -- reached once retrieval has already
#: landed and the single grounded-answer provider call is running.
LIBRARY_RAG_ANSWERING_LABEL = "Answering…"
_OPEN_SOURCE_TYPE_MAP = {
    "note": "notes",
    "notes": "notes",
    "media": "media",
    "media_chunk": "media",
    "conversation": "conversations",
    "conversations": "conversations",
    "chat": "conversations",
    # Deliberately singular, unlike the other three (whose canonical/open
    # value coincides with the plural scope-toggle key "prompts"):
    # `_open_library_item_by_id`'s dispatch key for prompts is "prompt" (see
    # its docstring), distinct from the "prompts" scope-toggle/source key
    # used for search selection and the rail row.
    "prompt": "prompt",
}


def update_search_history(history: Sequence[str], query: str) -> tuple[str, ...]:
    """Return search history with `query` prepended, deduped, capped at 10.

    Args:
        history: Existing history entries, most recent first.
        query: Newly submitted query; blank input leaves history unchanged.

    Returns:
        New history tuple, entries truncated to 200 chars, length <= 10.
    """
    entry = (query or "").strip()[:LIBRARY_SEARCH_HISTORY_ENTRY_MAX_CHARS]
    if not entry:
        return tuple(str(item) for item in history)
    deduped = [entry] + [str(item) for item in history if str(item) != entry]
    return tuple(deduped[:LIBRARY_SEARCH_HISTORY_LIMIT])


def searching_status_line(source_types: Sequence[str]) -> str:
    """Build the visible in-flight status line for a running search.

    Args:
        source_types: Selected source type IDs for the in-flight query.

    Returns:
        User-facing status line, e.g. `searching · notes, media…`.
    """
    labels = ", ".join(str(s) for s in source_types if str(s).strip())
    return f"searching · {labels}…" if labels else "searching…"


def _clean_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = " ".join(str(value).strip().split())
    return text or fallback


def _source_type_display_label(source_type: str) -> str:
    """Map a raw source-type identifier to its `LIBRARY_RAG_SOURCE_TYPES`
    display label (e.g. "notes" -> "Notes").

    Falls back to the raw identifier, unchanged, when it isn't one of the
    known types -- diagnostics payloads are service-supplied data, not a
    closed enum this module controls, so an unrecognized value still
    renders rather than disappearing or raising.
    """
    key = _clean_text(source_type).lower()
    return _LIBRARY_RAG_SOURCE_TYPE_LABELS.get(key, source_type)


def _remove_dangerous_display_patterns(value: str) -> tuple[str, bool]:
    scrubbed = _SCRIPT_BLOCK_PATTERN.sub("", value)
    changed = scrubbed != value
    for pattern in ("javascript:", "onclick=", "onerror="):
        if pattern in scrubbed.lower():
            scrubbed = re.sub(re.escape(pattern), "", scrubbed, flags=re.IGNORECASE)
            changed = True
    return scrubbed, changed


def _collapse_text(value: str, *, preserve_newlines: bool) -> str:
    if not preserve_newlines:
        return " ".join(value.strip().split())
    lines = (" ".join(line.strip().split()) for line in value.strip().splitlines())
    return "\n".join(line for line in lines if line)


def _sanitize_display_text(
    value: Any,
    fallback: str,
    *,
    max_length: int = LIBRARY_RAG_DISPLAY_MAX_LENGTH,
    preserve_newlines: bool = False,
    escape: bool = True,
) -> str:
    if value is None:
        return fallback
    sanitized = sanitize_string(str(value), max_length=max_length)
    scrubbed, _ = _remove_dangerous_display_patterns(sanitized)
    if not validate_text_input(scrubbed, max_length=max_length, allow_html=False):
        return fallback
    text = _collapse_text(scrubbed, preserve_newlines=preserve_newlines)
    if not text:
        return fallback
    if not escape:
        return text
    return escape_markup(_unescape_and_rescrub(text))


def _unescape_and_rescrub(text: str) -> str:
    """Undo HTML-entity encoding on already-collapsed display text, safely.

    The shared tail of `_sanitize_display_text(escape=True)`, factored out
    so `LibraryRagResultRow.from_result` can reuse it to derive the
    UNESCAPED-but-otherwise-fully-processed snippet text `display_snippet`
    strips Markdown structure from (RAG-30/31 C1 fix) -- escaping must stay
    display_snippet's terminal step, so the Markdown-stripping pass needs
    this function's output, not `_sanitize_display_text`'s own
    `escape_markup`-terminated one.

    RAG-30/31 originally re-escaped here (html.escape(html.unescape(text)))
    so a "R&amp;D" upstream source rendered as a single "&amp;D" instead
    of doubling into "R&amp;amp;D". That kept the escaping *symmetric* but
    broke the actual display surface: `Static` widgets render Rich markup,
    not HTML, so Rich never decodes "&amp;" back to "&" -- a user typing a
    literal "&" saw the literal string "&amp;" on screen (live UAT,
    2026-08-03 task-15 finding 1).

    Un-escaping first and NOT re-escaping fixes that display bug, but
    naively deleting `html.escape` here is unsafe on its own: this
    function's caller's dangerous-pattern scrubber
    (`_remove_dangerous_display_patterns`, dropping `<script>` blocks/
    `javascript:`/`onclick=`/`onerror=`) already ran ABOVE, on `sanitized`,
    BEFORE any unescaping. An entity-encoded payload
    (`&lt;script&gt;alert(1)&lt;/script&gt;`) sails straight past that
    scrubber -- it doesn't look like `<script>` yet -- and then, on this
    line, `html.unescape` would decode it into a LIVE `<script>` tag.
    `escape_markup` only neutralizes Rich's own `[`/`]` markup syntax; it
    does nothing for `<script`. So the scrubber has to run AGAIN, after
    unescaping, on the now-decoded text, before any final markup-escape.

    Args:
        text: Already sanitized/scrubbed/collapsed display text (the
            `_sanitize_display_text(escape=False)` output).

    Returns:
        `text` with HTML entities decoded and dangerous patterns re-scrubbed
        against the decoded form. Still UNESCAPED -- callers that render
        this must run it through `escape_markup` (or a stricter equivalent)
        as their own terminal step.
    """
    unescaped = html.unescape(text)
    unescaped, _ = _remove_dangerous_display_patterns(unescaped)
    return unescaped


# `_strip_markdown_syntax` structural patterns. This is deliberately a small
# regex pass, not a Markdown parser: it removes structural notation (link
# syntax keeps its visible text) while leaving the underlying text content
# alone. Applied ONLY to the UNESCAPED sanitized snippet text (RAG-30/31 C1
# fix) -- escaping must be `display_snippet`'s terminal step, so `[` here is
# never already escape_markup-escaped; the link pattern below no longer
# needs to tolerate an optional leading backslash (an earlier version did,
# back when this ran on already-escaped text -- see the C1 fix notes on
# `LibraryRagResultRow.display_snippet` for why that ordering was unsafe:
# stripping Markdown syntax AFTER escaping could turn an inert escaped
# bracket into a live one, e.g. `\[*/etc/hosts*\]`-shaped input never
# reached this function, but `[*/etc/hosts*]` did, and stripping its `*`
# delimiters exposed a bracket `escape_markup` had not touched).
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKDOWN_HEADING_PATTERN = re.compile(r"(?m)^#{1,6}[ \t]+")
_MARKDOWN_LIST_MARKER_PATTERN = re.compile(r"(?m)^(?:[-*+]|\d+[.)])[ \t]+")
_MARKDOWN_BACKTICK_PATTERN = re.compile(r"`+")
# Matched pair of emphasis delimiters (1-3 `*`/`_`), simplified CommonMark
# left/right-flanking: the delimiter run must not be immediately preceded or
# followed by a word char (or another delimiter char) on the "outside" and
# must not be immediately adjacent to whitespace on the "inside". This is a
# capture-and-keep substitution (like the link pattern above), not a
# delete-all -- a bare `_`/`*` embedded in an identifier (`top_k`,
# `my_notes_2026.md`, `OPENAI_API_KEY`) never satisfies the flanking rule on
# both sides, so it is left untouched rather than deleted (RAG-30/31 review:
# the earlier delete-all version corrupted technical content).
_MARKDOWN_EMPHASIS_PATTERN = re.compile(
    r"(?<![\w*_])([*_]{1,3})(?!\s)(.+?)(?<!\s)\1(?![\w*_])"
)


def _strip_markdown_syntax(text: str) -> str:
    """Strip common Markdown structural syntax, preserving the text content.

    Not a full Markdown parser -- a small, pure regex pass that removes
    heading markers, list bullets, emphasis markers (`**`/`__`/`*`/`_`), code
    fences/backticks, and link syntax (keeping the link's visible text, e.g.
    `[label](url)` -> `label`), so raw Markdown from ingested notes/media
    reads as plain prose on screen instead of literal notation (RAG-30/31).

    Args:
        text: Text to strip Markdown structure from.

    Returns:
        `text` with Markdown structural syntax removed.
    """
    if not text:
        return text
    stripped = _MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    stripped = _MARKDOWN_HEADING_PATTERN.sub("", stripped)
    stripped = _MARKDOWN_LIST_MARKER_PATTERN.sub("", stripped)
    stripped = _MARKDOWN_BACKTICK_PATTERN.sub("", stripped)
    stripped = _MARKDOWN_EMPHASIS_PATTERN.sub(r"\2", stripped)
    return stripped


# `escape_markup` (rich.markup.escape) only escapes a `[` immediately
# followed by what RICH's own markup grammar treats as a tag start
# (`[a-z#/@]`) -- by design, since Rich's own renderer only recognizes tags
# matching that same pattern. Every on-screen `Static`/`Label` in this app,
# however, renders through TEXTUAL's `Content` markup tokenizer
# (`textual.markup`), not Rich's -- and Textual's tokenizer opens a tag on
# ANY unescaped `[` (`expect_markup.open_tag = r"(?<!\\)\["`), regardless of
# what follows. A bracket `escape_markup` leaves untouched because it
# doesn't look like a Rich tag -- `[TODO]`, `[/etc/hosts]`, any bracket not
# immediately followed by a lowercase letter/`#`/`/`/`@` -- is therefore
# STILL live markup to Textual: `Content.from_markup('[TODO] finish this')`
# silently drops the bracketed span (renders `' finish this'`, eating the
# visible text), and `Content.from_markup('config [/etc/hosts]')` raises
# `MarkupError` outright (verified against the installed Textual 8.2.7).
#
# `display_snippet` strips Markdown structure before its terminal escape
# step (RAG-30/31 C1 fix): stripping emphasis delimiters can turn a bracket
# `escape_markup` would have covered on the raw text (`[*/etc/hosts*]`, not
# tag-shaped -- the `*` right after `[` isn't in `[a-z#/@]` either) into
# exactly this exposed shape (`[/etc/hosts]`) once the `*`s are gone. Using
# `escape_markup` as the terminal step there would still leave the `[TODO]`
# shape (from stripping `[_TODO_]`) unescaped and eaten -- so that terminal
# step needs the fuller guarantee below, not `escape_markup`'s narrower one.
_UNESCAPED_BRACKET_PATTERN = re.compile(r"(\\*)(\[)")


def _escape_all_brackets(text: str) -> str:
    """Escape every `[` in `text`, not only Rich-tag-shaped ones.

    Same backslash-doubling algorithm `escape_markup` uses (existing
    backslashes before a `[` are doubled, then one more is added, keeping
    the resulting run odd -- Textual's tokenizer treats an odd backslash
    run immediately before `[` as "already escaped", an even run as
    literal backslashes followed by a live tag open) -- just applied to
    every `[`, not only ones that look like a Rich/Textual tag start. For
    any bracket `escape_markup` already escapes, this produces the
    identical output (see the module comment above); it additionally
    covers the brackets `escape_markup` does not.

    Args:
        text: Text to escape for safe rendering in a Textual `Content`
            markup string (e.g. a `Static(...)` argument).

    Returns:
        `text` with every `[` neutralized against Textual's markup
        tokenizer.
    """

    def _double_backslashes(match: re.Match[str]) -> str:
        backslashes = match.group(1)
        return f"{backslashes}{backslashes}\\["

    escaped = _UNESCAPED_BRACKET_PATTERN.sub(_double_backslashes, text)
    if escaped.endswith("\\") and not escaped.endswith("\\\\"):
        escaped += "\\"
    return escaped


#: `ANSWER_MAX_TOKENS` (1200, `library_rag_answer_service.py`) at up to ~5
#: chars/token leaves headroom for a full grounded answer without
#: truncating it mid-sentence or mid-citation.
LIBRARY_RAG_ANSWER_DISPLAY_MAX_LENGTH = 8_000


def library_rag_answer_display_text(text: Any) -> str:
    """Sanitize and terminally escape one Library RAG answer's raw text.

    The shared escaping seam for `LibraryRagAnswer.text`/`.error` and
    `AnswerCitationValidation.recovery` copy on their way into a `Static` --
    all three are untrusted (`.text`/`.error` are model/provider output;
    `.recovery` is run through here too, defensively, even though it is
    presently always one of a handful of fixed sentences).

    Unlike `LibraryRagResultRow.display_snippet`, this does NOT strip
    Markdown structure: the answer is short generated prose, not an
    arbitrary-length ingested document, and `LIBRARY_RAG_ANSWER_SYSTEM_
    PROMPT` already asks the model for plain prose with no preamble.

    It DOES need the fuller `_escape_all_brackets` guarantee rather than
    plain `escape_markup`, and for a reason `display_snippet` does not
    share: a WORKING answer is expected to routinely carry bracketed
    citation markers by design -- the system prompt asks the model to cite
    with a label like `[S1]`. Rich's own `escape_markup` only escapes a `[`
    that looks tag-shaped by ITS OWN narrower grammar (`[a-z#/@]`); an
    uppercase citation label like `[S1]` does not match it and would sail
    through unescaped. `Static` renders through Textual's own tokenizer,
    though, which opens a tag on ANY unescaped `[` regardless of what
    follows -- so the citation markers this whole feature exists to show
    are exactly what a narrower escape would leave live and dangerous.
    `_escape_all_brackets` neutralizes every `[`, tag-shaped or not.

    Args:
        text: Raw text to sanitize and escape for display.

    Returns:
        Sanitized, escaped, display-ready text; `""` for empty or unsafe
        input (e.g. an embedded `<script>` block).
    """
    plain = _sanitize_display_text(
        text,
        "",
        max_length=LIBRARY_RAG_ANSWER_DISPLAY_MAX_LENGTH,
        preserve_newlines=True,
        escape=False,
    )
    if not plain:
        return ""
    return _escape_all_brackets(plain)


def _clamp_display_text(
    text: str, max_chars: int = LIBRARY_RAG_SNIPPET_DISPLAY_MAX_CHARS
) -> str:
    """Clamp `text` to `max_chars`, breaking at a word boundary.

    Args:
        text: Text to clamp.
        max_chars: Maximum length of the returned string, ellipsis included.

    Returns:
        `text` unchanged when it already fits; otherwise a word-boundary
        prefix followed by a single trailing "…".
    """
    if len(text) <= max_chars:
        return text
    budget = max(max_chars - 1, 0)
    truncated = text[:budget].rstrip()
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return f"{truncated.rstrip()}…"


def _sanitize_query(value: Any) -> tuple[str, bool]:
    if value is None:
        return "", False
    sanitized = sanitize_string(str(value), max_length=LIBRARY_RAG_QUERY_MAX_LENGTH)
    scrubbed, changed = _remove_dangerous_display_patterns(sanitized)
    valid = validate_text_input(
        scrubbed,
        max_length=LIBRARY_RAG_QUERY_MAX_LENGTH,
        allow_html=False,
    )
    if changed or not valid:
        return "", True
    return _collapse_text(scrubbed, preserve_newlines=False), False


def _sanitize_url(value: Any) -> str:
    text = _sanitize_display_text(value, "", max_length=2_000, escape=False)
    if not text:
        return ""
    if text.startswith("file://"):
        return text
    return text if validate_url(text) else ""


def _sentence(value: str) -> str:
    text = value.strip()
    if not text or text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _coerce_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _coerce_positive_int(value: Any, fallback: int) -> int:
    if not validate_number_range(value, min_val=1, max_val=LIBRARY_RAG_TOP_K_MAX):
        return fallback
    coerced = int(value)
    return coerced if coerced > 0 else fallback


def _coerce_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def library_rag_score_suffix(score: float | None) -> str:
    """Return an evidence row's title-line score suffix as an honest band.

    Raw three-decimal cosine scores (e.g. "| score 0.091") are meaningless
    to users -- this renders a match band instead (RAG-34). `None` (the
    service hard-sets `score=None` for keyword-mode rows because FTS
    relevance was judged misleading; see
    `library_local_rag_search_service.py`) renders as an empty string, not
    "unknown". The weak band keeps the raw two-decimal number for
    transparency, so a user can tell "this is the best of a bad lot" from
    "this barely matched at all".

    Band boundaries are inclusive on their upper band: a score exactly at
    `LIBRARY_RAG_MATCH_STRONG_THRESHOLD` is "strong", and a score exactly at
    `LIBRARY_RAG_MATCH_MODERATE_THRESHOLD` is "moderate".

    Args:
        score: Retrieval score, or `None` for keyword-mode rows.

    Returns:
        `""` for `None`; otherwise `" | match: strong"`,
        `" | match: moderate"`, or `" | match: weak (0.xx)"`.
    """
    if score is None:
        return ""
    if score >= LIBRARY_RAG_MATCH_STRONG_THRESHOLD:
        return " | match: strong"
    if score >= LIBRARY_RAG_MATCH_MODERATE_THRESHOLD:
        return " | match: moderate"
    return f" | match: weak ({score:.2f})"


def _normalize_mode(value: Any) -> str:
    mode = _sanitize_display_text(value, "rag", max_length=32, escape=False).lower()
    return mode if mode in {"rag", "search"} else "rag"


def _recovery_copy(
    *,
    status_label: str,
    unavailable_what: str,
    why: str,
    next_action: str,
    recovery_action: str,
    owner: str,
) -> str:
    return "\n".join(
        (
            _sentence(status_label),
            f"Unavailable: {_sentence(unavailable_what)}",
            f"Why: {_sentence(why)}",
            f"Next: {_sentence(next_action)}",
            f"Recovery: {_sentence(recovery_action)}",
            f"Owner: {_sentence(owner)}",
        )
    )


@dataclass(frozen=True)
class LibraryRagActionState:
    """Display state for one Library Search/RAG action."""

    label: str
    enabled: bool
    widget_id: str
    disabled_reason: str = ""

    @property
    def tooltip(self) -> str:
        return "" if self.enabled else self.disabled_reason


@dataclass(frozen=True)
class LibraryRagSourceOption:
    """One source-scope option in Library Search/RAG."""

    source_type: str
    label: str
    count: int
    selected: bool
    status: str
    recovery: str = ""

    @property
    def available(self) -> bool:
        return self.count > 0

    @property
    def count_label(self) -> str:
        suffix = "source" if self.count == 1 else "sources"
        return f"{self.count} {suffix}"


@dataclass(frozen=True)
class LibraryRagScopeState:
    """Display state for Library Search/RAG source scope."""

    heading: str
    options: tuple[LibraryRagSourceOption, ...]
    selected_source_types: tuple[str, ...]
    total_count: int
    status: str = "ready"
    recovery_copy: str = ""

    @classmethod
    def from_source_counts(
        cls,
        *,
        notes: Any = 0,
        media: Any = 0,
        conversations: Any = 0,
        prompts: Any = 0,
        workspaces: Any = 0,
        collections: Any = 0,
        selected: Sequence[str] | None = None,
        heading: str = "Source Scope: All local sources",
    ) -> "LibraryRagScopeState":
        """Build source-scope display state from loose source counts.

        Args:
            notes: Available note source count.
            media: Available media source count.
            conversations: Available conversation source count.
            prompts: Available prompt source count.
            workspaces: Available workspace source count.
            collections: Available collection source count.
            selected: Selected source type IDs. `None` selects all available sources;
                an empty sequence represents an explicit empty selection.
            heading: User-facing source-scope heading.

        Returns:
            Display state for the Library Search/RAG source scope.
        """

        counts = {
            "notes": _coerce_non_negative_int(notes),
            "media": _coerce_non_negative_int(media),
            "conversations": _coerce_non_negative_int(conversations),
            "prompts": _coerce_non_negative_int(prompts),
            "workspaces": _coerce_non_negative_int(workspaces),
            "collections": _coerce_non_negative_int(collections),
        }
        available_source_types = {
            source_type for source_type, count in counts.items() if count > 0
        }
        selected_source_types = available_source_types if selected is None else selected
        selected_values = {
            _clean_text(source_type).lower() for source_type in selected_source_types
        }
        options = tuple(
            LibraryRagSourceOption(
                source_type=source_type,
                label=label,
                count=counts[source_type],
                selected=source_type in selected_values and counts[source_type] > 0,
                status="ready" if counts[source_type] > 0 else "empty",
                recovery=(
                    ""
                    if counts[source_type] > 0
                    else f"No {source_type} available. Add or import {source_type} before querying."
                ),
            )
            for source_type, label in LIBRARY_RAG_SOURCE_TYPES
        )
        total_count = sum(counts.values())
        recovery_copy = ""
        status = "ready"
        if total_count == 0:
            status = "blocked"
            # A single quiet gate line (plus the scope region's one
            # "Open Import media" action) is the entire no-sources state.
            recovery_copy = LIBRARY_RAG_NO_SOURCES_GATE_COPY
        elif not any(option.selected for option in options):
            status = "blocked"
            recovery_copy = _recovery_copy(
                status_label="No source selected",
                unavailable_what="Library Search/RAG",
                why="No Library source scope is selected",
                next_action="Select at least one Library source before querying",
                recovery_action="Library source scope",
                owner="Library source scope",
            )
        return cls(
            heading=heading,
            options=options,
            selected_source_types=tuple(
                option.source_type for option in options if option.selected
            ),
            total_count=total_count,
            status=status,
            recovery_copy=recovery_copy,
        )

    @property
    def has_available_sources(self) -> bool:
        return self.total_count > 0

    @property
    def has_selected_sources(self) -> bool:
        return bool(self.selected_source_types)

    def option_by_type(self, source_type: str) -> LibraryRagSourceOption:
        normalized_source_type = _clean_text(source_type).lower()
        for option in self.options:
            if option.source_type == normalized_source_type:
                return option
        raise KeyError(source_type)


def library_rag_scope_summary(scope: LibraryRagScopeState) -> str:
    """Return the "#library-rag-scope-summary" strip text (RAG-32).

    Live UAT (critique RAG-32): the strip printed the hardcoded "all local
    sources" copy directly above the Sources toggles even when a user had
    switched a source off -- e.g. deselecting Media still read "all local
    sources". This is the ONE builder both `LibrarySearchRagPanel`'s
    compose path and `LibraryScreen`'s incremental refresh path delegate
    to (`Tests/UI/test_library_shell.py::
    test_library_shell_search_scope_strip_refresh_path_uses_shared_copy`
    pins the two seams agree).

    Only sources with a real toggle row
    (`LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES` -- notes/media/conversations/
    prompts; workspaces/collections have no retrieval seam of their own
    yet and are never user-togglable) are considered "available" here, so
    the strip never mentions a source the user has no control over.

    Three cases:
      * No source is available at all (the empty-library edge, already
        owned by `LIBRARY_RAG_NO_SOURCES_GATE_COPY` elsewhere on screen)
        or every available source is selected (the common case): returns
        `LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY` verbatim, unchanged.
      * No available source is selected (deselect-all, already surfaced by
        the run gate's own "Select at least one Library source." quiet
        line): returns "Scope: no sources selected" rather than listing
        every available source as "off", which would just restate the
        same fact as noise.
      * A genuine subset is selected: returns the selected sources'
        display labels in canonical `LIBRARY_RAG_SOURCE_TYPES` order,
        followed by the deselected sources parenthesized -- e.g.
        "Scope: Notes, Conversations (Media, Prompts off)". The
        parenthetical is deliberate, not incidental: RAG-32's own
        complaint was a *missing negative* ("still reads all local
        sources" after deselecting Media), so the fix names what's off,
        not just what's on.

    Args:
        scope: Current Library Search/RAG source scope display state.

    Returns:
        The scope-summary strip's user-facing text.
    """
    toggle_types = [
        source_type
        for source_type in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES
        if scope.option_by_type(source_type).available
    ]
    if not toggle_types:
        return LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY
    selected_values = set(scope.selected_source_types)
    selected_types = [
        source_type for source_type in toggle_types if source_type in selected_values
    ]
    if len(selected_types) == len(toggle_types):
        return LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY
    if not selected_types:
        return "Scope: no sources selected"
    off_types = [
        source_type for source_type in toggle_types if source_type not in selected_values
    ]
    selected_labels = ", ".join(
        _source_type_display_label(source_type) for source_type in selected_types
    )
    off_labels = ", ".join(
        _source_type_display_label(source_type) for source_type in off_types
    )
    return f"Scope: {selected_labels} ({off_labels} off)"


def library_rag_empty_state_quiet_copy(query: str, scope: LibraryRagScopeState) -> str:
    """Return the Evidence region's quiet no-match copy (RAG-33/Task 11).

    Live UAT (critique RAG-33): a routine "your library has nothing
    matching this query" search rendered the full Unavailable/Why/Next/
    Recovery/Owner dump, ending in the internal-process line "Owner:
    Library retrieval" -- ceremony for what is honestly one sentence. The
    render seam (`library_rag_results_body_children` in
    `library_search_rag_panel.py`) reserves this quiet copy for the
    routine no-match case only (`retrieval_status == "empty"`); real
    failures (missing dependencies, empty index, provider unavailable,
    policy denial) still render the full recovery dump -- this function
    does NOT build that dump and is never called for those statuses.

    The second line adapts to whether a real, available Library source is
    still switched off (RAG-27/B2 toggles): when one is, "turn on more
    sources" is true, actionable advice; when every available source is
    already selected, offering to enable sources that don't exist would be
    a false claim -- this project's established honesty bar for retrieval
    copy (RAG-29's coverage note, RAG-34's match bands) -- so that clause
    is dropped rather than shown regardless of whether it's true.

    Args:
        query: The submitted query (`LibraryRagQueryState.query`, already
            sanitized/collapsed plain text). Escaped and clamped for
            display here -- this is the one place in the panel that
            renders raw query text inside a `Static` rather than passing
            it through as an `Input` value or an already-escaped history
            `Button` label (mirrors that builder's rich-markup escaping,
            e.g. a query of `[bold]x` must not inject markup).
        scope: Current source-scope display state, consulted only to
            check whether a real source is still switched off.

    Returns:
        Two-line quiet copy: `"No evidence matched '<query>'."` then the
        adaptive follow-up line, joined by a single newline.
    """
    display_query = _clamp_display_text(query, LIBRARY_RAG_EMPTY_QUERY_QUOTE_MAX_CHARS)
    escaped_query = escape_markup(display_query)
    has_more_sources = any(
        scope.option_by_type(source_type).available
        for source_type in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES
        if source_type not in scope.selected_source_types
    )
    second_line = (
        "Try broader terms or turn on more sources."
        if has_more_sources
        else "Try broader terms."
    )
    return f"No evidence matched '{escaped_query}'.\n{second_line}"


@dataclass(frozen=True)
class LibraryRagQueryState:
    """Display state for Library Search/RAG query controls."""

    query: str
    mode: str
    mode_label: str
    top_k: int
    include_citations: bool
    status: str
    run_action: LibraryRagActionState
    recovery_copy: str = ""

    @property
    def blocked_is_empty_query(self) -> bool:
        """True when the run gate's blocker is a missing query (A1).

        The Search canvas renders a single muted line for this case instead
        of the full callout + recovery-copy presentation.
        """
        return self.run_action.disabled_reason == _EMPTY_QUERY_DISABLED_REASON

    @property
    def blocked_is_no_scope(self) -> bool:
        """True when the run gate's blocker is an empty source scope (A1/B2).

        Reached when no Library source is selected -- either no sources are
        available at all, or the user deselected every scope toggle. Like
        `blocked_is_empty_query`, this renders as a single muted line.
        """
        return self.run_action.disabled_reason == _NO_SCOPE_DISABLED_REASON

    @classmethod
    def from_values(
        cls,
        *,
        query: Any = "",
        mode: Any = "rag",
        top_k: Any = LIBRARY_RAG_DEFAULT_TOP_K,
        include_citations: bool = True,
        has_source_scope: bool = True,
        dependencies_ready: bool = True,
        index_ready: bool = True,
        provider_ready: bool = True,
    ) -> "LibraryRagQueryState":
        """Build query-control display state from UI or service values.

        Args:
            query: User query text.
            mode: Search mode, either `rag` or `search`; invalid values default to `rag`.
            top_k: Requested result count. Values outside the allowed range use the default.
            include_citations: Whether citation metadata should be requested/displayed.
            has_source_scope: Whether at least one source is selected.
            dependencies_ready: Whether Search/RAG optional dependencies are available.
            index_ready: Whether the selected source scope has an index.
            provider_ready: Whether a provider/model is ready for RAG-answer mode.

        Returns:
            Display state for query controls and the run action.
        """

        normalized_query, unsafe_query = _sanitize_query(query)
        normalized_mode = _normalize_mode(mode)
        mode_label = "Search" if normalized_mode == "search" else "RAG Answer"
        normalized_top_k = _coerce_positive_int(top_k, LIBRARY_RAG_DEFAULT_TOP_K)
        disabled_reason = ""
        owner = ""
        next_action = ""
        recovery_action = ""
        if unsafe_query:
            disabled_reason = "Enter a safe question or search query."
            owner = "user"
            next_action = "Remove markup or script content before running Search/RAG"
            recovery_action = "Query input"
        elif not has_source_scope:
            disabled_reason = _NO_SCOPE_DISABLED_REASON
            owner = "Library source scope"
            next_action = "Select or import a source before querying"
            recovery_action = "Library source scope"
        elif not normalized_query:
            disabled_reason = _EMPTY_QUERY_DISABLED_REASON
            owner = "user"
            next_action = "Type a query before running Search/RAG"
            recovery_action = "Query input"
        elif not dependencies_ready:
            disabled_reason = "Install or enable Search/RAG dependencies."
            owner = "optional dependency"
            next_action = "Install Search/RAG dependencies and restart"
            recovery_action = "Settings or package extras"
        elif not index_ready:
            disabled_reason = "Index selected Library sources before querying."
            owner = "Library source index"
            next_action = "Build or refresh the Library index"
            recovery_action = "Library indexing"
        elif normalized_mode == "rag" and not provider_ready:
            disabled_reason = "Select a provider/model before asking for a RAG answer."
            owner = "LLM provider"
            next_action = "Select a provider and model before running a RAG answer"
            recovery_action = "Console controls"

        enabled = not disabled_reason
        recovery_copy = ""
        if disabled_reason:
            recovery_copy = _recovery_copy(
                status_label="Blocked",
                unavailable_what="Run Library Search/RAG",
                why=disabled_reason,
                next_action=next_action,
                recovery_action=recovery_action,
                owner=owner,
            )
        return cls(
            query=normalized_query,
            mode=normalized_mode,
            mode_label=mode_label,
            top_k=normalized_top_k,
            include_citations=include_citations,
            status="ready" if enabled else "blocked",
            run_action=LibraryRagActionState(
                label="Run",
                enabled=enabled,
                widget_id=LIBRARY_RAG_RUN_ACTION_ID,
                disabled_reason=disabled_reason,
            ),
            recovery_copy=recovery_copy,
        )


@dataclass(frozen=True)
class LibraryRagCitation:
    """Normalized citation metadata for a Library Search/RAG result."""

    label: str
    url: str = ""
    source_id: str = ""
    chunk_id: str = ""


@dataclass(frozen=True)
class LibraryRagResultRow:
    """Normalized evidence row for Library Search/RAG results."""

    result_id: str
    title: str
    snippet: str
    score: float | None
    source_id: str
    chunk_id: str
    citations: tuple[LibraryRagCitation, ...]
    provenance: Mapping[str, Any]
    runtime_backend: str = ""
    #: Sanitized/collapsed/HTML-entity-decoded snippet text, still UNESCAPED
    #: (RAG-30/31 C1 fix) -- `display_snippet` strips Markdown structure from
    #: this, never from `snippet` (which is already `escape_markup`-escaped
    #: for the Console-handoff/evidence-bundle surface), so escaping stays
    #: `display_snippet`'s terminal step instead of running before a text
    #: transform that can expose an unescaped bracket. Excluded from
    #: equality/repr: it is a pure function of the same source data `snippet`
    #: is built from, never independently meaningful.
    _snippet_plain: str = field(default="", repr=False, compare=False)

    @classmethod
    def from_result(cls, result: Mapping[str, Any] | Any) -> "LibraryRagResultRow":
        """Normalize a retrieval result into immutable evidence display state.

        Args:
            result: Retrieval result mapping from a local or remote Search/RAG adapter.

        Returns:
            Normalized evidence row with sanitized display text, citations, score, IDs,
            backend metadata, and immutable provenance.
        """

        values = result if isinstance(result, Mapping) else {}
        source_id = _sanitize_display_text(values.get("source_id"), "", escape=False)
        chunk_id = _sanitize_display_text(values.get("chunk_id"), "", escape=False)
        title = _sanitize_display_text(
            values.get("document_title")
            or values.get("title")
            or values.get("source_title"),
            "Untitled source",
        )
        # Collapsed-but-unescaped first (mirrors `_sanitize_display_text`'s
        # own `escape=True` path one step at a time, see `_unescape_and_rescrub`)
        # so `snippet_plain` -- what `display_snippet` strips Markdown syntax
        # from -- and `snippet` -- the escaped Console-handoff/evidence-bundle
        # value -- are derived from the exact same processed text, just with
        # escaping applied (or not) as the very last step for each.
        snippet_collapsed = _sanitize_display_text(
            values.get("snippet") or values.get("text") or values.get("content"),
            "No snippet available.",
            max_length=LIBRARY_RAG_SNIPPET_MAX_LENGTH,
            preserve_newlines=True,
            escape=False,
        )
        snippet_plain = _unescape_and_rescrub(snippet_collapsed)
        snippet = escape_markup(snippet_plain)
        citations = tuple(
            _normalize_citation(citation)
            for citation in _as_sequence(values.get("citations"))
        )
        provenance_value = values.get("provenance")
        provenance = (
            dict(provenance_value) if isinstance(provenance_value, Mapping) else {}
        )
        for key in LIBRARY_RAG_PROVENANCE_KEYS:
            if key in values and key not in provenance:
                provenance[key] = values[key]
        result_id = _result_id(source_id, chunk_id, title)
        return cls(
            result_id=result_id,
            title=title,
            snippet=snippet,
            score=_coerce_score(values.get("score")),
            source_id=source_id,
            chunk_id=chunk_id,
            citations=citations,
            provenance=MappingProxyType(provenance),
            runtime_backend=_sanitize_display_text(
                values.get("runtime_backend"),
                "",
                escape=False,
            ),
            _snippet_plain=snippet_plain,
        )

    @property
    def citation_labels(self) -> tuple[str, ...]:
        return tuple(citation.label for citation in self.citations)

    @property
    def display_snippet(self) -> str:
        """On-screen projection of `snippet`: Markdown-stripped and clamped.

        `snippet` itself -- and its 4,000-char cap used for Console
        handoff/evidence bundles -- is unchanged; this is a display-only
        derivation (RAG-30/31): raw Markdown structure is stripped to plain
        prose and the result is flattened to one line and clamped to
        `LIBRARY_RAG_SNIPPET_DISPLAY_MAX_CHARS` at a word boundary, so a
        single low-relevance result can't render 25+ lines and bury the
        rest of the evidence list.

        Derived from `_snippet_plain` -- the same sanitized/collapsed text
        `snippet` is built from, but still UNESCAPED -- never from `snippet`
        itself (2026-08-03 task-15 finding C1 fix): Markdown-stripping ran
        on the already-`escape_markup`-escaped `snippet` before this fix,
        which could resurrect live markup an earlier stripped delimiter had
        been shielding (e.g. `[*/etc/hosts*]` is inert -- `escape_markup`
        does not need to touch it, since Rich's tag-look-alike check fails
        on the leading `*` -- but stripping its `*` emphasis markers first
        exposes `[/etc/hosts]`, a bracket `escape_markup` DOES leave alone
        because `/` passes that same check, and Textual's renderer used to
        see it live and crash). Escaping now runs strictly last, over the
        fully stripped/flattened/clamped text, via `_escape_all_brackets`
        rather than `escape_markup` -- `escape_markup`'s narrower
        tag-look-alike check still leaves some stripped shapes (e.g.
        `[TODO]`, from stripping `[_TODO_]`) unescaped, and Textual's own
        markup tokenizer opens a tag on ANY unescaped `[`, not only
        tag-shaped ones (see `_escape_all_brackets`'s module comment).
        """
        stripped = _strip_markdown_syntax(self._snippet_plain)
        flattened = _collapse_text(stripped, preserve_newlines=False)
        clamped = _clamp_display_text(flattened)
        return _escape_all_brackets(clamped)

    @property
    def source_type_badge_label(self) -> str:
        """Compact source type label for evidence rows."""
        return (
            _provenance_text(self.provenance, "source_type")
            or _provenance_text(self.provenance, "item_type")
            or _provenance_text(self.provenance, "type")
            or "source"
        )

    @property
    def workspace_badge_label(self) -> str:
        """Compact workspace authority label for evidence rows."""
        workspace_ids = _provenance_text_tuple(self.provenance, "workspace_ids")
        workspace_id = _provenance_text(self.provenance, "workspace_id")
        if workspace_id and workspace_id not in workspace_ids:
            workspace_ids = (*workspace_ids, workspace_id)
        if not workspace_ids:
            return "all workspaces"
        if len(workspace_ids) == 1:
            return workspace_ids[0]
        return f"{len(workspace_ids)} workspaces"

    @property
    def citation_count_badge_label(self) -> str:
        """Compact citation count label for evidence rows."""
        count = len(self.citations)
        suffix = "citation" if count == 1 else "citations"
        return f"{count} {suffix}"

    @property
    def eligibility_badge_label(self) -> str:
        """Compact active-context eligibility label for evidence rows."""
        explicit_eligible = _coerce_optional_bool(
            self.provenance.get("active_context_eligible")
        )
        if explicit_eligible is True:
            return "eligible"
        if explicit_eligible is False:
            return "blocked"
        workspace_ids = _provenance_text_tuple(self.provenance, "workspace_ids")
        if workspace_ids and not _provenance_text(
            self.provenance, "active_workspace_id"
        ):
            return "blocked"
        return "eligible"

    @property
    def row_badge_label(self) -> str:
        """One-line source authority summary for result list scanning.

        Humanized composition (UX wave M5): badges that would only restate
        the default/no-signal case are dropped rather than listed
        unconditionally, and the remainder is joined with the app-wide
        " · " separator (not "|"). The source-type badge always appears;
        the workspace badge is dropped when it is the default "all
        workspaces"; the citation-count badge appears only when there are
        citations; eligibility contributes nothing when "eligible" and
        "excluded from context" when "blocked". Examples: "media",
        "media · 2 citations", "media · excluded from context". The
        individual badge properties above are unchanged -- other call
        sites/tests depend on their current behavior.
        """
        parts = [self.source_type_badge_label]
        workspace_label = self.workspace_badge_label
        if workspace_label != "all workspaces":
            parts.append(workspace_label)
        if len(self.citations) > 0:
            parts.append(self.citation_count_badge_label)
        if self.eligibility_badge_label == "blocked":
            parts.append("excluded from context")
        return " · ".join(parts)

    @property
    def source_identity_label(self) -> str:
        """User-facing source/chunk identity for selected evidence inspection."""
        if self.source_id and self.chunk_id:
            return f"Source: {self.source_id} / {self.chunk_id}"
        if self.source_id:
            return f"Source: {self.source_id}"
        if self.chunk_id:
            return f"Chunk: {self.chunk_id}"
        return "Source: unavailable"

    @property
    def runtime_label(self) -> str:
        """User-facing runtime/backend identity for selected evidence inspection."""
        return f"Runtime: {self.runtime_backend or 'local'}"

    @property
    def authority_display_label(self) -> str:
        """User-facing authority label aligned with evidence handoff metadata."""
        explicit_label = _provenance_text(self.provenance, "authority_label")
        if explicit_label:
            return f"Authority: {explicit_label}"

        workspace_ids = _provenance_text_tuple(self.provenance, "workspace_ids")
        workspace_id = _provenance_text(self.provenance, "workspace_id")
        if workspace_id and workspace_id not in workspace_ids:
            workspace_ids = (*workspace_ids, workspace_id)
        if workspace_ids:
            return f"Authority: Workspace: {', '.join(workspace_ids)}"

        runtime_backend = self.runtime_backend.lower()
        source_authority = (
            "server"
            if runtime_backend.startswith("server") or "server" in runtime_backend
            else "local"
        )
        return f"Authority: Source authority: {source_authority}"

    @property
    def eligibility_label(self) -> str:
        """User-facing active-context eligibility for selected evidence inspection."""
        explicit_eligible = _coerce_optional_bool(
            self.provenance.get("active_context_eligible")
        )
        explicit_reason = _provenance_text(self.provenance, "eligibility_reason")
        if explicit_eligible is True:
            return "Eligibility: available for active workspace"
        if explicit_eligible is False:
            reason = explicit_reason.replace("_", " ") if explicit_reason else "blocked"
            return f"Eligibility: blocked for active workspace ({reason})"

        workspace_ids = _provenance_text_tuple(self.provenance, "workspace_ids")
        if workspace_ids and not _provenance_text(
            self.provenance, "active_workspace_id"
        ):
            return "Eligibility: blocked until an active workspace is available"
        return "Eligibility: available for active context"

    @property
    def handoff_label(self) -> str:
        """User-facing statement of what the Console handoff preserves."""
        return "Handoff: snippet + citations + source/chunk IDs"

    @property
    def open_source_type(self) -> str:
        """Library canvas target this result can open, or empty string."""
        raw = (
            str(
                self.provenance.get("source_type")
                or self.provenance.get("item_type")
                or self.provenance.get("type")
                or ""
            )
            .strip()
            .lower()
        )
        return _OPEN_SOURCE_TYPE_MAP.get(raw, "")

    @property
    def can_open(self) -> bool:
        """True when the row carries a resolvable parent id and known type."""
        return bool(self.open_source_type and self.source_id)


def library_rag_all_matches_weak(rows: Sequence[LibraryRagResultRow]) -> bool:
    """True when every scored row among `rows` bands weak (RAG-34/Task 8).

    Feeds Task 8's evidence-list coverage note. Unscored rows (keyword-mode,
    `score is None`) are ignored entirely -- neither counted toward "all"
    nor treated as weak. True only when there is at least one scored row and
    all of them fall below `LIBRARY_RAG_MATCH_MODERATE_THRESHOLD`; a result
    set with no scored rows at all (e.g. pure keyword mode) returns `False`,
    not `True` -- "everything is weak" is a claim about actual scores, not
    about their absence.

    Args:
        rows: Evidence rows to inspect.

    Returns:
        Whether every scored row among `rows` bands weak.
    """
    scored = [row.score for row in rows if row.score is not None]
    if not scored:
        return False
    return all(score < LIBRARY_RAG_MATCH_MODERATE_THRESHOLD for score in scored)


# Task 8: the Evidence region's one-line semantic-coverage note prefix when
# every scored row bands weak (`library_rag_all_matches_weak`). Kept as its
# own constant (rather than inlined into `library_rag_coverage_note`) so the
# exact wording has one source of truth.
LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX = (
    "No strong semantic matches — results below are weak."
)


def library_rag_coverage_note(
    diagnostics: Mapping[str, Any] | None,
    rows: Sequence[LibraryRagResultRow],
) -> str:
    """Return the Evidence region's one-line semantic coverage note (Task 8).

    Live UAT (RAG-29): a "cake" query in rag mode returned unrelated media
    fixtures and no conversation, even though a conversation plainly
    discussing cake existed -- with nothing on screen distinguishing "your
    notes contain nothing relevant" from "semantic search never looked at
    your notes". `_search_semantic` (`library_local_rag_search_service.py`)
    now reports, per query, which of the *requested* source types are
    actually present in the returned rows' provenance under
    `diagnostics["semantic_scope_coverage"]`; this renders that into one
    honest, specific sentence -- or stays silent when there is nothing to
    say.

    Args:
        diagnostics: The outcome's `LibraryRagSearchOutcome.diagnostics`
            mapping. Only the `"semantic_scope_coverage"` slot is
            consulted; any other shape (keyword mode's scope-exclusion
            slot, an empty mapping, `None`) renders no coverage claim.
        rows: The panel's current, already-normalized evidence rows (i.e.
            what is about to be shown).

    Returns:
        `""` when `rows` is empty (edge case: zero results overall is the
        no-match/empty state's territory, not a coverage note enumerating
        every requested source as "uncovered"), when every requested source
        type is covered and no row bands weak, or when `diagnostics` carries
        no `semantic_scope_coverage` entry at all (e.g. keyword mode).
        Otherwise `"Semantic search found nothing from: <types>."` (types
        in the order `semantic_scope_coverage["uncovered"]` lists them,
        rendered through `_source_type_display_label` -- the same
        `LIBRARY_RAG_SOURCE_TYPES` vocabulary the Sources toggles use, e.g.
        "Notes" not "notes" -- rather than the raw identifiers the
        diagnostics payload carries, so this note and the toggles two lines
        above it speak one vocabulary), with
        `LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX` prepended (space-joined) when
        `library_rag_all_matches_weak(rows)` is True -- or just the
        weak-prefix alone when nothing is uncovered.
    """
    if not rows:
        return ""
    coverage = (
        diagnostics.get("semantic_scope_coverage")
        if isinstance(diagnostics, Mapping)
        else None
    )
    uncovered = (
        tuple(str(item) for item in coverage.get("uncovered", ()) or ())
        if isinstance(coverage, Mapping)
        else ()
    )
    # `_source_type_display_label` falls back to the raw, unrecognized
    # `source_type` verbatim when it isn't one of `LIBRARY_RAG_SOURCE_TYPES`
    # -- and `uncovered` above is `str(item)` from the service's
    # `semantic_scope_coverage` diagnostics mapping, a swappable attribute
    # this module does not control the shape of. Every other user-visible
    # string this module builds is `escape_markup`-escaped before reaching a
    # `Static`; these labels were the one gap (task-15 finding M8).
    uncovered_labels = tuple(
        escape_markup(_source_type_display_label(source_type))
        for source_type in uncovered
    )
    message = (
        f"Semantic search found nothing from: {', '.join(uncovered_labels)}."
        if uncovered_labels
        else ""
    )
    if library_rag_all_matches_weak(rows):
        return (
            f"{LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX} {message}"
            if message
            else LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX
        )
    return message


@dataclass(frozen=True)
class LibraryRagPanelState:
    """Display state for the destination-native Library Search/RAG panel."""

    scope: LibraryRagScopeState
    query_state: LibraryRagQueryState
    results: tuple[LibraryRagResultRow, ...]
    retrieval_status: str
    next_action: str
    use_in_console_action: LibraryRagActionState
    selected_result_id: str = ""
    selected_result: LibraryRagResultRow | None = None
    recovery_copy: str = ""
    recovery_selector: str = ""
    history: tuple[str, ...] = ()
    history_collapsed: bool = False
    #: Evidence region one-line semantic coverage note (Task 8), built by
    #: `library_rag_coverage_note` from the retrieval outcome's
    #: `diagnostics["semantic_scope_coverage"]` and the panel's own
    #: `results`. Empty string when there is nothing to say.
    coverage_note: str = ""
    #: The query the CURRENT `retrieval_status`/`results` were actually
    #: retrieved for -- independent of `query_state.query`, which tracks
    #: live, not-yet-submitted input text (task-15 finding I3). The two
    #: coincide immediately after a search lands, but `query_state.query`
    #: keeps moving on every keystroke (in-panel query box) or rail-search-
    #: box edit while `retrieval_status` can still read "empty" from the
    #: last completed search -- the quiet no-match line must quote the query
    #: that produced that outcome, not whatever text is sitting in a box at
    #: render time. Defaults to `query` (via `from_values`'s `None` sentinel)
    #: so every pre-existing call site that never distinguished the two
    #: keeps its exact prior behavior.
    searched_query: str = ""
    #: PR-3 Task 1's grounded-answer outcome (`generate_library_rag_answer`),
    #: when one has landed for the query `results` were retrieved for.
    #: `None` before any rag-mode answer has been generated, and always
    #: `None` in keyword (search) mode -- rag mode is the only mode Task 1's
    #: answer service is ever invoked for.
    answer: LibraryRagAnswer | None = None
    #: `answer.status` mirrored onto the panel state directly, so render
    #: code (`library_rag_answer_children`) can branch on it without
    #: dereferencing `answer` (which is `None` before any answer lands).
    #: `""` when `answer` is `None`.
    answer_status: str = ""

    @classmethod
    def from_values(
        cls,
        *,
        source_counts: Mapping[str, Any] | None = None,
        query: Any = "",
        searched_query: Any = None,
        mode: Any = "rag",
        results: Sequence[LibraryRagResultRow | Mapping[str, Any]] = (),
        selected_result_id: Any = "",
        retrieval_status: Any = "",
        recovery_copy: Any = "",
        recovery_selector: Any = "",
        dependencies_ready: bool = True,
        index_ready: bool = True,
        provider_ready: bool = True,
        selected_source_types: Sequence[str] | None = None,
        history: Sequence[str] = (),
        history_collapsed: bool = False,
        diagnostics: Mapping[str, Any] | None = None,
        answer: LibraryRagAnswer | None = None,
    ) -> "LibraryRagPanelState":
        """Build full Library Search/RAG panel display state.

        Args:
            source_counts: Available source counts keyed by source type.
            query: User query text (live, not-yet-submitted input).
            searched_query: The query the current `retrieval_status`/
                `results` were actually retrieved for (task-15 finding I3).
                `None` (the default -- every call site that predates this
                fix) falls back to `query`, so a caller that never tracked
                the two separately keeps identical prior behavior; a caller
                that does (the screen) passes the query it last actually
                searched, which can differ from `query` once the user edits
                the box (or a separate rail search box) without re-running.
            mode: Search mode, either `rag` or `search`.
            results: Retrieval result rows or mappings.
            selected_result_id: Result ID selected for inspector/Console handoff.
            retrieval_status: Explicit retrieval status override.
            recovery_copy: Explicit retrieval recovery copy from a service outcome.
            recovery_selector: Stable selector used for explicit retrieval recovery.
            dependencies_ready: Whether Search/RAG optional dependencies are available.
            index_ready: Whether the selected source scope has an index.
            provider_ready: Whether a provider/model is ready for RAG-answer mode.
            selected_source_types: Selected source type IDs. `None` selects all available
                source types; an empty sequence represents no selected sources.
            history: Prior submitted queries, most recent first.
            history_collapsed: Whether the `Recent searches` collapsible should
                render collapsed (D1). The caller owns this decision -- it is
                only forced on the results-arrival transition, not on every
                render -- so this is a plain passthrough, not derived here.
            diagnostics: The retrieval outcome's non-result-shaped notices
                (`LibraryRagSearchOutcome.diagnostics`), e.g.
                `semantic_scope_coverage` (Task 8). `None` (the default --
                every call site that predates Task 8) renders no coverage
                note.
            answer: PR-3 Task 1's grounded-answer outcome for the current
                `results`, or `None` before one has landed (every call site
                that predates Task 3).

        Returns:
            Display state for the destination-native Library Search/RAG panel.
        """

        counts = dict(source_counts or {})
        scope = LibraryRagScopeState.from_source_counts(
            notes=counts.get("notes", 0),
            media=counts.get("media", 0),
            conversations=counts.get("conversations", 0),
            prompts=counts.get("prompts", 0),
            workspaces=counts.get("workspaces", 0),
            collections=counts.get("collections", 0),
            selected=selected_source_types,
        )
        query_state = LibraryRagQueryState.from_values(
            query=query,
            mode=mode,
            has_source_scope=scope.has_selected_sources,
            dependencies_ready=dependencies_ready,
            index_ready=index_ready,
            provider_ready=provider_ready,
        )
        normalized_searched_query, _ = _sanitize_query(
            query if searched_query is None else searched_query
        )
        result_rows = tuple(
            result
            if isinstance(result, LibraryRagResultRow)
            else LibraryRagResultRow.from_result(result)
            for result in results
        )
        coverage_note = library_rag_coverage_note(diagnostics, result_rows)
        normalized_selected_result_id = _clean_text(selected_result_id)
        selected_result = next(
            (
                result
                for result in result_rows
                if result.result_id == normalized_selected_result_id
            ),
            None,
        )
        explicit_status = _clean_text(retrieval_status).lower()
        explicit_recovery_copy = _sanitize_display_text(
            recovery_copy,
            "",
            preserve_newlines=True,
        )
        explicit_recovery_selector = _sanitize_display_text(
            recovery_selector,
            "",
            max_length=128,
            escape=False,
        )
        active_recovery_selector = ""
        if query_state.status == "blocked":
            normalized_status = "blocked"
            recovery_copy = scope.recovery_copy or query_state.recovery_copy
            next_action = _blocked_next_action(recovery_copy)
        elif explicit_status == "searching":
            normalized_status = "searching"
            recovery_copy = ""
            next_action = "Wait for retrieval results."
        elif explicit_status == "answering":
            # PR-3 Task 3: reached once retrieval already landed and the RAG
            # Answer worker's single provider call is in flight -- one more
            # explicit-status branch alongside "searching" above, not a
            # forked copy of it (see the run-action override below, which
            # extends the same `if` rather than adding a second one).
            normalized_status = "answering"
            recovery_copy = ""
            next_action = "Wait for the RAG answer."
        elif explicit_status in {"blocked", "failed"}:
            normalized_status = explicit_status
            recovery_copy = explicit_recovery_copy or _recovery_copy(
                status_label="Retrieval unavailable",
                unavailable_what="Library Search/RAG retrieval",
                why="Library retrieval could not complete",
                next_action="Retry the query or check Library indexing",
                recovery_action="Retry",
                owner="Library retrieval",
            )
            next_action = _blocked_next_action(recovery_copy)
            active_recovery_selector = (
                explicit_recovery_selector or LIBRARY_RAG_SERVICE_ERROR_SELECTOR
            )
        elif explicit_status == "empty" or (
            explicit_status == "ready" and not result_rows
        ):
            normalized_status = "empty"
            recovery_copy = explicit_recovery_copy or _recovery_copy(
                status_label="No results",
                unavailable_what="Library Search/RAG evidence",
                why="No evidence matched the current query",
                next_action="Revise the query or broaden the source scope",
                recovery_action="Query input or source scope",
                owner="Library retrieval",
            )
            next_action = "Revise the query or broaden the source scope."
            active_recovery_selector = (
                explicit_recovery_selector or LIBRARY_RAG_EMPTY_STATE_SELECTOR
            )
        elif result_rows:
            normalized_status = "ready"
            recovery_copy = ""
            next_action = (
                "Review cited evidence or send the selected result to Console."
            )
        else:
            normalized_status = "ready"
            recovery_copy = ""
            next_action = "Run Search/RAG over the selected Library sources."

        if normalized_status in {"searching", "answering"}:
            # C2: the run action itself carries the in-flight state -- label
            # "Searching…"/"Answering…" (an ellipsis character, one unit),
            # disabled, so the canvas never shows an enabled Run button while
            # a query (or its RAG answer) is already running. Only reachable
            # when the run gate was open (query_state.status != "blocked"),
            # so there is always a well-formed prior run_action to replace.
            in_flight_label, in_flight_reason = (
                (LIBRARY_RAG_ANSWERING_LABEL, "Answer generation in progress.")
                if normalized_status == "answering"
                else (LIBRARY_RAG_SEARCHING_LABEL, "Search in progress.")
            )
            query_state = replace(
                query_state,
                run_action=LibraryRagActionState(
                    label=in_flight_label,
                    enabled=False,
                    widget_id=LIBRARY_RAG_RUN_ACTION_ID,
                    disabled_reason=in_flight_reason,
                ),
            )

        can_use_console = normalized_status == "ready" and selected_result is not None
        answer_status = answer.status if answer is not None else ""
        return cls(
            scope=scope,
            query_state=query_state,
            results=result_rows,
            retrieval_status=normalized_status,
            next_action=next_action,
            use_in_console_action=LibraryRagActionState(
                label="Use in Console",
                enabled=can_use_console,
                widget_id=LIBRARY_RAG_USE_IN_CONSOLE_ACTION_ID,
                disabled_reason=(
                    ""
                    if can_use_console
                    else LIBRARY_RAG_USE_IN_CONSOLE_DISABLED_REASON
                ),
            ),
            selected_result_id=normalized_selected_result_id,
            selected_result=selected_result,
            recovery_copy=recovery_copy,
            recovery_selector=active_recovery_selector,
            history=tuple(str(h) for h in history),
            history_collapsed=bool(history_collapsed),
            coverage_note=coverage_note,
            searched_query=normalized_searched_query,
            answer=answer,
            answer_status=answer_status,
        )


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    if value:
        return (value,)
    return ()


def _normalize_citation(value: Any) -> LibraryRagCitation:
    if isinstance(value, Mapping):
        label = _sanitize_display_text(
            value.get("label")
            or value.get("title")
            or value.get("url")
            or value.get("source_id"),
            "Citation",
        )
        return LibraryRagCitation(
            label=label,
            url=_sanitize_url(value.get("url")),
            source_id=_sanitize_display_text(value.get("source_id"), "", escape=False),
            chunk_id=_sanitize_display_text(value.get("chunk_id"), "", escape=False),
        )
    return LibraryRagCitation(label=_sanitize_display_text(value, "Citation"))


def _provenance_text(provenance: Mapping[str, Any], key: str) -> str:
    return _sanitize_display_text(
        provenance.get(key),
        "",
        max_length=LIBRARY_RAG_DISPLAY_MAX_LENGTH,
    )


def _provenance_text_tuple(provenance: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = provenance.get(key)
    values = _as_sequence(value)
    normalized = tuple(
        _sanitize_display_text(
            item,
            "",
            max_length=LIBRARY_RAG_DISPLAY_MAX_LENGTH,
        )
        for item in values
    )
    return tuple(text for text in normalized if text)


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "enabled", "eligible", "available"}:
        return True
    if text in {"0", "false", "no", "n", "disabled", "blocked", "ineligible"}:
        return False
    return None


def _result_id(source_id: str, chunk_id: str, title: str) -> str:
    if source_id and chunk_id:
        return f"{source_id}:{chunk_id}"
    if source_id:
        return source_id
    if chunk_id:
        return chunk_id
    return f"result:{title.lower().replace(' ', '-')}"


def _blocked_next_action(recovery_copy: str) -> str:
    if recovery_copy == LIBRARY_RAG_NO_SOURCES_GATE_COPY:
        return _NO_SOURCES_NEXT_ACTION
    for line in recovery_copy.splitlines():
        if line.startswith("Next: "):
            return line.removeprefix("Next: ")
    return "Resolve the blocker before running Search/RAG."
