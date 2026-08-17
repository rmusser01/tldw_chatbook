"""Pure display-state contracts for Library-native Search/RAG.

One deliberate exception to "pure" (TASK-15020/B3): `library_rag_profile_
top_k` reads the active RAG profile's result count, because the window's
evidence depth is a user setting and a display state that cannot see it
would have to keep hardcoding 5. The read is lazy, exception-safe, torch-
free by construction (see that function), and everything downstream of it
stays a pure function of the value it returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import html
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from loguru import logger
from rich.markup import escape as escape_markup

from tldw_chatbook.Library.library_rag_answer_service import LibraryRagAnswer
from tldw_chatbook.Library.library_rag_score_kinds import (
    LIBRARY_RAG_SCORE_KIND_HYBRID_FUSION,
    LIBRARY_RAG_SCORE_KIND_RERANKER,
    LIBRARY_RAG_SCORE_KIND_VECTOR_SIMILARITY,
    coerce_optional_float as _coerce_score,
    library_rag_result_score_kind,
    library_rag_similarity_input,
    normalize_library_rag_score_kind as _normalize_score_kind,
)
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
#: The evidence depth used when the active RAG profile cannot be read at all
#: (TASK-15020/B3). This used to be `LIBRARY_RAG_DEFAULT_TOP_K` -- the window's
#: actual, unconfigurable default -- while the Console's Library RAG entry
#: points already read the profile (TASK-406/TASK-3170). Renamed with the
#: behavior: it is now only the degraded answer, reached when
#: `library_rag_profile_top_k` cannot resolve a positive number. Kept equal to
#: the Console seam's own fallback (`CONSOLE_LIBRARY_RAG_FALLBACK_TOP_K`,
#: pinned by the coupling test) so both surfaces degrade to the same depth.
LIBRARY_RAG_FALLBACK_TOP_K = 5
LIBRARY_RAG_RUN_ACTION_ID = "library-rag-run-query"
LIBRARY_RAG_SERVICE_ERROR_SELECTOR = "library-rag-service-error"
LIBRARY_RAG_EMPTY_STATE_SELECTOR = "library-rag-empty-state"
LIBRARY_RAG_USE_IN_CONSOLE_DISABLED_REASON = (
    "Run a query and select usable evidence before sending to Console."
)
# Task-2852: pre-navigation notice shown when "Use in Console" is pressed
# while Console is still locked behind first-run provider setup. The
# handoff still proceeds (evidence is genuinely staged and a matching
# receipt appears on the locked Console surface -- see
# `console_setup_staged_receipt` in `Chat/console_live_work.py`), so this is
# advisory, not a block: it exists because the pre-fix UAT repro found the
# navigation landing on a silent setup screen with zero trace of the
# selection.
LIBRARY_RAG_USE_IN_CONSOLE_LOCKED_NOTICE = (
    "Library evidence staged — finish provider setup in Console to use it."
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
#: The scope-summary line's leading noun on the Library screen. The Console
#: passes its own ("Sources") to `library_rag_source_scope_summary` -- see
#: that function -- because "Scope" is already spent there on the retrieval
#: item scope ("Scope: 2 items"), a different concept.
LIBRARY_RAG_SCOPE_SUMMARY_PREFIX = "Scope"
_ALL_LOCAL_SOURCES_SUMMARY_TAIL = "all local sources"
LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY = (
    f"{LIBRARY_RAG_SCOPE_SUMMARY_PREFIX}: {_ALL_LOCAL_SOURCES_SUMMARY_TAIL}"
)
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
# (RAG-port P0/Task 6) The band thresholds above are a claim about COSINE
# SIMILARITY, and nothing else the retrieval stack produces lives on that
# scale:
#   * hybrid (RRF) fuses by RANK -- a fused score's theoretical maximum is
#     `1/(rrf_k + 1)`, i.e. ~0.17 at the shipped `rrf_k = 5` (TASK-4110) and
#     ~0.016 at the previous `rrf_k = 60`. Below the 0.2 weak boundary
#     either way, but the kind -- not the magnitude -- is what disqualifies
#     it: a rank blend is not a similarity at any k. Banding it on the
#     thresholds above rendered a wall of "match: weak (0.02)" on every
#     hybrid search, including perfect matches.
#   * reranker scores are unbounded (cross-encoder logits, 0-10 LLM
#     scales); a value that happens to land inside [0, 1] is not a
#     similarity either.
# So the band's INPUT is chosen by score kind (`library_rag_score_suffix`):
# hybrid rows band on the vector leg Task 2 preserves in
# `hybrid_fusion["vector_score"]`, and rows with no similarity at all
# disclose their kind instead of inventing one. The kind vocabulary and its
# resolution rule live in `library_rag_score_kinds` (imported above) so the
# Console evidence-bundle builder can share them without closing an import
# cycle; only the DISPLAY copy for the two no-similarity kinds is here.
#: Title-line suffixes for the two kinds that carry no similarity to band.
LIBRARY_RAG_KEYWORD_MATCH_SUFFIX = " | keyword match"
LIBRARY_RAG_RERANKED_SUFFIX = " | reranked"
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
# Raw provenance `source_type`/`item_type`/`type` identifiers -> the
# scope-toggle vocabulary `LibraryRagScopeState.selected_source_types`
# speaks ("notes"/"media"/"conversations"/"prompts"). D4/task-5: a scope
# toggle flipped OFF after retrieval already landed used to leave that
# source's rows displayed, selectable, and stageable into Console --
# `LibraryRagPanelState.from_values` filters already-landed rows against
# the CURRENT scope using this map. Distinct from `_OPEN_SOURCE_TYPE_MAP`
# just above: that map's "prompt" deliberately stays singular
# (`_open_library_item_by_id`'s dispatch key) -- this one's "prompt" must
# canonicalize to the plural "prompts" scope-toggle key, or toggling
# Prompts off would never hide a prompt row. Mirrors
# `_SEMANTIC_SOURCE_TYPE_MAP` in `library_local_rag_search_service.py`
# (the retrieval-time analogue of this same filter, applied to rag mode's
# rows before they land). That map used to omit "prompt"/"prompts" because
# nothing on the rag path could emit one; TASK-15020/B2's prompts keyword
# sub-leg does, so the two maps now agree on prompts as well, and this one
# keeps the extra "workspace"/"collection" entries no retrieval path emits.
# Prompts still have no SEMANTIC seam -- that fact moved to
# `_SEMANTICALLY_COVERABLE_SOURCE_TYPES`, which is about the vector index
# rather than about canonicalization.
_SCOPE_SOURCE_TYPE_MAP = {
    "note": "notes",
    "notes": "notes",
    "media": "media",
    "media_chunk": "media",
    "conversation": "conversations",
    "conversations": "conversations",
    "chat": "conversations",
    "prompt": "prompts",
    "prompts": "prompts",
    "workspace": "workspaces",
    "workspaces": "workspaces",
    "collection": "collections",
    "collections": "collections",
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
        User-facing status line, e.g. `searching · Notes, Media…`. Source
        type IDs are rendered through `_source_type_display_label` (task-7
        PR-2 leftover) so this line matches the capitalized vocabulary the
        Sources toggles and scope-summary strip already use, instead of
        the raw lowercase identifier.
    """
    labels = ", ".join(
        _source_type_display_label(str(s)) for s in source_types if str(s).strip()
    )
    return f"searching · {labels}…" if labels else "searching…"


def library_rag_paid_mode_notice(provider: str) -> str:
    """Return the quiet line's ready-state paid-mode notice (PR-T2 Task 4).

    Until this task, the ONLY provider-adjacent copy on the Library
    Search/RAG panel was the *blocked* branch's "Select a provider/model
    before asking for a RAG answer." text (`LibraryRagQueryState.
    from_values`) -- it vanishes the instant a provider IS configured,
    the exact inversion of what a keyboard-fast user needs before
    pressing a button that spends real money. This fills the query
    region's reserved quiet-line row (`library_search_rag_panel.py`'s
    `library_rag_query_status_children`) in the one state that row was
    otherwise left empty for -- ready, `rag` mode, a provider configured
    -- naming the provider that would actually be billed. No confirmation
    dialog, no gate: a statement, not a speed bump.

    Args:
        provider: The provider `resolve_library_rag_answer_provider`
            would call if Run were pressed right now
            (`LibraryRagQueryState.ready_answer_provider`).

    Returns:
        One sentence naming `provider`, e.g. `RAG Answer sends your
        question and the evidence to openai. Search stays local.`
    """
    return (
        f"RAG Answer sends your question and the evidence to {provider}. "
        "Search stays local."
    )


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


def library_rag_profile_top_k() -> int:
    """Return the ACTIVE RAG profile's result count (TASK-15020/B3).

    The Search/RAG window's evidence depth used to be the literal 5 while the
    Console's two Library RAG entry points already honored the profile
    (TASK-406/TASK-3170) -- two surfaces over one retrieval stack disagreeing
    about how deep a search goes, with only one of them configurable. This is
    the ONE seam both now read: `chat_screen._console_library_rag_profile_
    top_k` delegates here, and `Tests/Library/test_library_rag_state.py` pins
    that they agree on both branches.

    Reads `resolve_active_rag_top_k` -- the depth-only resolution -- NOT
    `resolve_active_rag_config()`: the full resolution's env layer probes the
    embedding device and imports torch (~0.9s), and this is called from a
    display-state builder that the Library screen rebuilds on every render
    and documents as never importing an optional Search/RAG dependency. See
    that function's docstring; the no-torch property is pinned in
    `Tests/RAG/test_active_config_resolution.py`.

    Imported lazily for the same reason the Console seam does: the profile
    manager is not something this module has any other reason to load.

    Returns:
        The profile's `search.default_top_k` when it resolves to a positive
        integer, else `LIBRARY_RAG_FALLBACK_TOP_K` -- a broken/absent profile
        must degrade to a usable depth, never raise inside a render.
    """
    try:
        from ..RAG_Search.simplified.active_config import resolve_active_rag_top_k

        value = int(resolve_active_rag_top_k())
    except Exception as exc:
        logger.debug(
            "Library Search/RAG could not read the active RAG profile's "
            "top_k (exception_category={}); using the fallback.",
            type(exc).__name__,
        )
        return LIBRARY_RAG_FALLBACK_TOP_K
    return value if value > 0 else LIBRARY_RAG_FALLBACK_TOP_K


def _resolve_window_top_k(value: Any) -> int:
    """Resolve the window's evidence depth: explicit value, else the profile.

    B3 changes the DEFAULT only. An in-range caller-supplied count wins
    unchanged and the profile is never consulted for it; anything unset or
    outside `1..LIBRARY_RAG_TOP_K_MAX` resolves to the active profile's depth,
    clamped to that same bound.

    The clamp is deliberate: Settings accepts a profile `default_top_k` up to
    100 (`settings_library_rag_defaults.MAX_RAG_RESULT_COUNT`) while this
    window's own bound is 50, and a 100-deep profile means "as deep as you
    can" -- discarding it back to the fallback 5 (what the pre-B3 coercion did
    with any out-of-range number) would invert the user's intent. The
    Console seam has no such bound and stays uncapped; this is the one
    deliberate difference between the two, and it only exists above 50.

    Args:
        value: The caller-supplied count, or `None`/invalid for "unset".

    Returns:
        A depth within `1..LIBRARY_RAG_TOP_K_MAX`.
    """
    if validate_number_range(value, min_val=1, max_val=LIBRARY_RAG_TOP_K_MAX):
        coerced = int(value)
        if coerced > 0:
            return coerced
    return min(library_rag_profile_top_k(), LIBRARY_RAG_TOP_K_MAX)


def library_rag_score_suffix(
    score: float | None,
    *,
    score_kind: str = LIBRARY_RAG_SCORE_KIND_VECTOR_SIMILARITY,
    vector_score: float | None = None,
) -> str:
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

    The band is a claim about cosine similarity, so `score_kind` selects
    what is banded (RAG-port P0/Task 6 -- see the thresholds' own comment
    above):

    * `vector_similarity` (the default, and every pre-existing call site):
      band `score`, exactly as before.
    * `hybrid_fusion` with a `vector_score`: band the VECTOR LEG. The fused
      RRF number is a rank blend (maxing out at `1/(rrf_k + 1)`) and is
      never banded, whatever k is configured.
      The label is unchanged -- "match: strong" means the same thing it
      always did, because it is computed from the same kind of number.
    * `hybrid_fusion` with no `vector_score` (an FTS-leg-only row): no
      similarity exists, so this discloses `" | keyword match"` -- never a
      fabricated band, and never the fused 0.0x number.
    * `reranker`: `" | reranked"`. Reranker outputs are unbounded (logits,
      0-10 LLM scales); the kind is disclosed instead of banding them.

    Args:
        score: Retrieval score, or `None` for keyword-mode rows.
        score_kind: The score's kind
            (`library_rag_score_kinds.LIBRARY_RAG_SCORE_KINDS`).
        vector_score: Preserved vector-leg similarity for hybrid rows.

    Returns:
        `""` for an unscored similarity row;
        `LIBRARY_RAG_RERANKED_SUFFIX` for reranked rows;
        `LIBRARY_RAG_KEYWORD_MATCH_SUFFIX` for FTS-only hybrid rows;
        otherwise `" | match: strong"`, `" | match: moderate"`, or
        `" | match: weak (0.xx)"`.
    """
    kind = _normalize_score_kind(score_kind)
    if kind == LIBRARY_RAG_SCORE_KIND_RERANKER:
        return LIBRARY_RAG_RERANKED_SUFFIX
    similarity = library_rag_similarity_input(
        score, score_kind=kind, vector_score=vector_score
    )
    if similarity is None:
        return (
            LIBRARY_RAG_KEYWORD_MATCH_SUFFIX
            if kind == LIBRARY_RAG_SCORE_KIND_HYBRID_FUSION
            else ""
        )
    if similarity >= LIBRARY_RAG_MATCH_STRONG_THRESHOLD:
        return " | match: strong"
    if similarity >= LIBRARY_RAG_MATCH_MODERATE_THRESHOLD:
        return " | match: moderate"
    return f" | match: weak ({similarity:.2f})"


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

    The grammar itself lives in `library_rag_source_scope_summary` below,
    which this delegates to -- the Console's Library RAG surfaces hold a
    bare tuple of source types (no counts, so no `LibraryRagScopeState`)
    and share that same builder rather than growing a second grammar.

    Args:
        scope: Current Library Search/RAG source scope display state.

    Returns:
        The scope-summary strip's user-facing text.
    """
    return library_rag_source_scope_summary(
        scope.selected_source_types,
        available_source_types=[
            source_type
            for source_type in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES
            if scope.option_by_type(source_type).available
        ],
    )


def library_rag_source_scope_summary(
    selected_source_types: Sequence[str],
    *,
    available_source_types: Sequence[str] = LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES,
    prefix: str = LIBRARY_RAG_SCOPE_SUMMARY_PREFIX,
) -> str:
    """Return the source-scope summary line for a raw source-type selection.

    The grammar owned here -- canonical order, the named negative, the
    "all local sources" common case -- is `library_rag_scope_summary`'s
    (see its docstring for why each case reads the way it does). This is
    that same builder reached without a `LibraryRagScopeState`: the
    Console's Library RAG surfaces (the Inspector readiness card's label
    and the RAG settings modal's toggle row, RAG-44) hold a plain tuple of
    source types and no per-source counts, so they cannot construct one.
    Two seams, one builder -- the PR-2 lesson that produced RAG-32's fix.

    Args:
        selected_source_types: The selected raw source-type identifiers.
        available_source_types: The source types the caller can actually
            toggle. Defaults to every real toggle source; the Library
            screen narrows it to the ones with sources on disk.
        prefix: The line's leading noun. Defaults to Library's "Scope";
            the Console passes "Sources" because its own "Scope:" already
            names the retrieval ITEM scope (conversation ∩ workspace),
            which this line has nothing to do with.

    Returns:
        The user-facing summary line, e.g. `Scope: Notes, Conversations
        (Media, Prompts off)`.
    """
    available_values = {
        _clean_text(source_type).lower() for source_type in available_source_types
    }
    toggle_types = [
        source_type
        for source_type in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES
        if source_type in available_values
    ]
    if not toggle_types:
        return f"{prefix}: {_ALL_LOCAL_SOURCES_SUMMARY_TAIL}"
    selected_values = {
        _clean_text(source_type).lower() for source_type in selected_source_types
    }
    selected_types = [
        source_type for source_type in toggle_types if source_type in selected_values
    ]
    if len(selected_types) == len(toggle_types):
        return f"{prefix}: {_ALL_LOCAL_SOURCES_SUMMARY_TAIL}"
    if not selected_types:
        return f"{prefix}: no sources selected"
    off_types = [
        source_type for source_type in toggle_types if source_type not in selected_values
    ]
    selected_labels = ", ".join(
        _source_type_display_label(source_type) for source_type in selected_types
    )
    off_labels = ", ".join(
        _source_type_display_label(source_type) for source_type in off_types
    )
    return f"{prefix}: {selected_labels} ({off_labels} off)"


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
    #: The provider that would be billed if Run were pressed right now
    #: (PR-T2 Task 4) -- `""` whenever there is nothing pending to
    #: announce: Search mode never calls a provider, and any blocked `rag`
    #: state (empty query, no scope, missing deps/index, no provider) has
    #: no run about to happen. Non-empty if and only if `mode == "rag"` and
    #: `status == "ready"` -- `from_values` derives BOTH readiness and this
    #: field from the same single `provider_name` argument (review round:
    #: an earlier revision took readiness and the name as two independently
    #: settable parameters, which let them disagree -- "ready to spend
    #: money" while the quiet line stayed silent about it). Feeds the quiet
    #: line's `library_rag_paid_mode_notice` (`library_search_rag_panel.
    #: py`'s `library_rag_query_status_children`).
    ready_answer_provider: str = ""

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
        top_k: Any = None,
        include_citations: bool = True,
        has_source_scope: bool = True,
        dependencies_ready: bool = True,
        index_ready: bool = True,
        provider_name: str | None = None,
        provider_credential_recovery: str = "",
    ) -> "LibraryRagQueryState":
        """Build query-control display state from UI or service values.

        Args:
            query: User query text.
            mode: Search mode, either `rag` or `search`; invalid values default to `rag`.
            top_k: Requested result count. `None` (the Library screen's own
                case -- the canvas carries no depth control) and any value
                outside `1..LIBRARY_RAG_TOP_K_MAX` resolve to the ACTIVE RAG
                PROFILE's `default_top_k` (TASK-15020/B3), clamped to that
                bound, and to `LIBRARY_RAG_FALLBACK_TOP_K` only when the
                profile itself is unresolvable. An in-range explicit value is
                used unchanged and never consults the profile -- see
                `_resolve_window_top_k`.
            include_citations: Whether citation metadata should be requested/displayed.
            has_source_scope: Whether at least one source is selected.
            dependencies_ready: Whether Search/RAG optional dependencies are available.
            index_ready: Whether the selected source scope has an index.
            provider_name: The provider `resolve_library_rag_answer_provider`
                resolved, or `None`/blank when none is configured. This is
                the SOLE source of RAG-mode provider readiness (PR-T2 Task 4
                review) -- an earlier revision took a separate `provider_
                ready: bool` alongside this, which let a caller pass
                `provider_ready=True` with no name (or vice versa) and
                silently produce "the mode can spend money and the quiet
                line says nothing", the exact inversion this task exists to
                fix. Collapsing to one parameter makes that combination
                impossible to construct: readiness is derived here as
                `bool((provider_name or "").strip())`, so "ready" and "has
                a name to show" can no longer disagree.
            provider_credential_recovery: The remedy for a provider that IS
                named in config but cannot authenticate (`Library/library_
                rag_answer_service.LibraryRagProviderGate.credential_
                recovery`, ultimately `ProviderReadiness.recovery`, e.g.
                "Set ANTHROPIC_API_KEY or add api_key under [api_settings.
                anthropic]."). `""` for the genuinely-unselected case, and
                `""` whenever `provider_name` is set. It is a MESSAGE, not
                a second readiness input: readiness stays derived solely
                from `provider_name` above, so passing this can never make
                a blocked state look ready -- it only decides WHICH blocked
                copy the RAG-mode branch shows. Without it the collapse to
                one `provider_name` erased the distinction between "no
                provider selected" and "selected, no credential", and the
                only surviving copy told the second user to select the
                provider they had already selected, pointing at Console
                controls instead of at the credential (PR-T2 review round
                3, finding I1).

        Returns:
            Display state for query controls and the run action.
        """

        normalized_provider_name = (provider_name or "").strip()
        provider_ready = bool(normalized_provider_name)
        # Escaped like every other config-sourced display string in this
        # module: the remedy text embeds a TOML table name in brackets
        # ("... add api_key under [api_settings.anthropic].") and both of
        # its sinks -- the run button's tooltip and the blocked callout /
        # recovery `Static`s -- render Rich markup, which would eat the
        # bracketed half of the instruction. Never shown when a provider
        # IS ready: the ladder below only reads it in the blocked branch.
        credential_recovery = (
            ""
            if provider_ready
            else _sanitize_display_text(provider_credential_recovery, "")
        )
        normalized_query, unsafe_query = _sanitize_query(query)
        normalized_mode = _normalize_mode(mode)
        mode_label = "Search" if normalized_mode == "search" else "RAG Answer"
        normalized_top_k = _resolve_window_top_k(top_k)
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
            # Two different blockers wear this one branch (finding I1). A
            # provider that is NAMED but cannot authenticate gets the
            # readiness object's own remedy -- naming the env var and the
            # config table -- because telling that user to "select a
            # provider/model" names a step they already completed.
            if credential_recovery:
                disabled_reason = (
                    f"The configured provider has no usable API key. "
                    f"{credential_recovery}"
                )
                owner = "LLM provider credential"
                next_action = "Add the provider credential, then run again"
                recovery_action = credential_recovery
            else:
                disabled_reason = (
                    "Select a provider/model before asking for a RAG answer."
                )
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
        # Reuses `normalized_provider_name`/`enabled` rather than
        # re-deriving readiness: in `rag` mode, `enabled` is `True` only
        # when the gate ladder above never hit the provider branch, which
        # itself only happens when `provider_ready` (== `bool(normalized_
        # provider_name)`) was already `True` -- so "ready" and "has a name
        # to show" cannot diverge by construction (PR-T2 Task 4 review).
        ready_answer_provider = (
            normalized_provider_name if normalized_mode == "rag" and enabled else ""
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
            ready_answer_provider=ready_answer_provider,
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
    #: What scale `score` is on (RAG-port P0/Task 6). Defaults to
    #: `vector_similarity` -- the only kind that existed before hybrid and
    #: reranking became reachable -- so every pre-existing construction and
    #: every `library_rag_score_suffix(row.score)` call site keeps its exact
    #: prior behavior. Resolved once here, in `from_result`, rather than at
    #: each display site, so the band, the all-weak coverage sentence and
    #: the Console evidence bundle cannot disagree about one row.
    score_kind: str = LIBRARY_RAG_SCORE_KIND_VECTOR_SIMILARITY
    #: The vector leg's preserved cosine similarity for `hybrid_fusion`
    #: rows (Task 2's `metadata["hybrid_fusion"]["vector_score"]`), or
    #: `None` -- including for an FTS-leg-only hybrid row, which has no
    #: similarity at all. Always `None` for the other kinds.
    vector_score: float | None = None
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
        # Score-kind resolution reads the ORIGINAL `provenance_value` and
        # engine `metadata` blocks, not the sanitized `provenance` copy
        # above: `LIBRARY_RAG_PROVENANCE_KEYS` is a display allowlist, and
        # the fusion/reranker channels are retrieval provenance, not
        # display provenance.
        score_kind, vector_score = library_rag_result_score_kind(
            provenance_value,
            values.get("metadata"),
            values,
        )
        return cls(
            result_id=result_id,
            title=title,
            snippet=snippet,
            score=_coerce_score(values.get("score")),
            score_kind=score_kind,
            vector_score=vector_score,
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
        """Compact source type label for evidence rows.

        Routed through `_source_type_display_label` (task-7 PR-2 leftover)
        so the badge reads "Media"/"Notes" like the rest of the panel's
        vocabulary instead of the raw lowercase provenance identifier. An
        unrecognized/already-escaped value (e.g. a markup-escaping test's
        deliberately hostile `source_type`) falls through unchanged --
        `_source_type_display_label` only rewrites values it recognizes.
        """
        return _source_type_display_label(
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
    def scope_source_type(self) -> str:
        """Canonical Sources-toggle vocabulary for this row (D4/task-5).

        `LibraryRagPanelState.from_values` filters already-landed rows by
        scope using this -- a source toggled OFF must hide rows whose
        provenance says they came from it. Distinct from `open_source_type`
        just above: that property deliberately keeps "prompt" singular
        (`_open_library_item_by_id`'s dispatch key); this one lands in the
        exact "notes"/"media"/"conversations"/"prompts" vocabulary
        `LibraryRagScopeState.selected_source_types` speaks, or a
        Prompts-scope toggle-off would never catch a prompt row.

        Returns "" when `source_type`/`item_type`/`type` is missing or does
        not canonicalize -- the scope filter treats that as "cannot be
        attributed to any toggle" and never hides it, mirroring
        `_semantic_row_matches_scope`'s permissive default in
        `library_local_rag_search_service.py` (the retrieval-time analogue
        of this same filter).
        """
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
        return _SCOPE_SOURCE_TYPE_MAP.get(raw, "")

    @property
    def can_open(self) -> bool:
        """True when the row carries a resolvable parent id and known type."""
        return bool(self.open_source_type and self.source_id)


def library_rag_all_matches_weak(rows: Sequence[LibraryRagResultRow]) -> bool:
    """True when every row carrying a similarity bands weak (RAG-34/Task 8).

    Feeds Task 8's evidence-list coverage note, whose wording
    (`LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX`: "No strong semantic
    matches...") is a claim about SEMANTIC SIMILARITY. Only rows whose
    effective banding input is a similarity therefore participate --
    `library_rag_similarity_input`, the same seam
    `library_rag_score_suffix` bands with:

    * unscored rows (keyword-mode, `score is None`),
    * FTS-leg-only hybrid rows (no vector leg), and
    * reranked rows (unbounded scores)

    are ignored entirely -- neither counted toward "all" nor treated as
    weak. A hybrid row IS counted, on its preserved vector leg: the fused
    RRF number (a rank blend maxing out at `1/(rrf_k + 1)`) would otherwise
    make every hybrid result set read as uniformly weak (RAG-port P0/Task 6).

    True only when there is at least one such row and all of them fall
    below `LIBRARY_RAG_MATCH_MODERATE_THRESHOLD`; a result set with no
    similarity-bearing rows at all (e.g. pure keyword mode) returns
    `False`, not `True` -- "everything is weak" is a claim about actual
    scores, not about their absence.

    Args:
        rows: Evidence rows to inspect. Read by duck typing (`.score`, and
            optionally `.score_kind`/`.vector_score`) rather than by type:
            `mcp_inspector._ScoredRow` is a `__slots__ = ("score",)` shim
            that feeds this same canonical check, so the two newer
            attributes are read with `getattr` defaults.

    Returns:
        Whether every similarity-bearing row among `rows` bands weak.
    """
    scored = [
        similarity
        for similarity in (
            library_rag_similarity_input(
                getattr(row, "score", None),
                score_kind=getattr(
                    row, "score_kind", LIBRARY_RAG_SCORE_KIND_VECTOR_SIMILARITY
                ),
                vector_score=getattr(row, "vector_score", None),
            )
            for row in rows
        )
        if similarity is not None
    ]
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

# (RAG-port P0, Workstream A) Diagnostics slot carrying retrieval-ROUTING
# disclosures: one short phrase per way the retrieval that actually ran
# differs from the active RAG profile's configured search mode -- e.g. a
# hybrid profile forced onto the semantic path because no selected source
# has a keyword leg, or a plain (BM25) profile routed to the Library's own
# four-seam keyword path. (An active scope used to head that list; since
# TASK-15020/B1 the allowlist reaches both engine legs, so a scoped hybrid
# search runs hybrid and has nothing to disclose.)
# Distinct in MEANING from `semantic_scope_coverage` (which reports which
# requested source types a search that ran as configured actually touched),
# but deliberately rendered into the SAME single quiet line under the
# Evidence heading by `library_rag_coverage_note` -- one note channel on
# screen, never two competing ones.
LIBRARY_RAG_ROUTE_NOTES_KEY = "retrieval_route_notes"


def _route_note_sentence(note: str) -> str:
    """Render one service-supplied routing disclosure as a sentence.

    The service states these as lowercase fragments ("media excluded —
    semantic only") so they read correctly in logs and tests; here they
    become sentences that can sit after the coverage/weak sentences on one
    line. Escaped for the same reason the uncovered labels are (task-15
    finding M8): the text is service-supplied and reaches a `Static`.
    """
    text = escape_markup(str(note).strip())
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else f"{text}."


# (TASK-3502 note-(a)) The reranker's two disclosure tags, paired with the
# word the note uses for each. `enhanced_rag_service_v2.search()` stamps
# exactly one of them onto the FIRST result's metadata when a reranking
# attempt did not do what enabling it implies: `reranking_skipped` when the
# call raised at all (a dead credential, a provider outage), and
# `reranking_degraded` when it returned normally having silently failed to
# score some or all rows. Ordered: `skipped` is checked first so a row
# somehow carrying both (the service's two sites are mutually exclusive
# branches, but nothing here enforces that) produces ONE deterministic
# sentence.
LIBRARY_RAG_RERANKING_TAG_LABELS: tuple[tuple[str, str], ...] = (
    ("reranking_skipped", "skipped"),
    ("reranking_degraded", "degraded"),
)
#: What the failure actually means for what is on screen -- the reason this
#: is worth a line at all. A silently unreranked result list is
#: indistinguishable from a reranked one without it.
LIBRARY_RAG_RERANKING_CONSEQUENCE = (
    "these results are in their original retrieval order"
)
#: The tag detail is `str(exc)` off a provider call or a "N/M scorings
#: failed" counter -- unbounded, service-supplied text sharing one line with
#: the coverage/routing sentences.
LIBRARY_RAG_RERANKING_DETAIL_MAX_CHARS = 120


def library_rag_reranking_notice(rows: Sequence[LibraryRagResultRow]) -> str:
    """Return the reranking-disclosure sentence for `rows`, or `""`.

    TASK-3502 note-(a): the first UI consumer of the reranker's disclosure
    tags. They reach here on a row's `provenance` -- the engine writes them
    into the first `SearchResult`'s `metadata`
    (`enhanced_rag_service_v2._tag_first_result`), the Library service
    copies that metadata block into the row's provenance
    (`library_local_rag_search_service._semantic_row`), and
    `LibraryRagResultRow.from_result` copies the provenance mapping
    wholesale.

    Args:
        rows: The panel's current, already-normalized evidence rows. EVERY
            row is checked, not just the first: the engine tags position 0
            of ITS list, but scope post-filtering and the panel's own
            count-intersected filter both run afterwards, so the tagged row
            can land anywhere (or be dropped, in which case there is
            nothing to disclose and nothing claiming otherwise).

    Returns:
        One sentence naming which disclosure fired, its detail, and what
        that means for the order on screen -- or `""` when no row carries
        either tag (the overwhelmingly common case: reranking off, or on
        and working). The detail is collapsed, clamped and
        `escape_markup`-escaped, like every other service-supplied string
        this module renders.
    """
    for key, label in LIBRARY_RAG_RERANKING_TAG_LABELS:
        for row in rows:
            provenance = row.provenance
            if not isinstance(provenance, Mapping) or key not in provenance:
                continue
            detail = escape_markup(
                _clamp_display_text(
                    " ".join(str(provenance[key]).split()),
                    LIBRARY_RAG_RERANKING_DETAIL_MAX_CHARS,
                )
            )
            qualifier = f" ({detail})" if detail else ""
            return (
                f"Reranking was {label}{qualifier} — "
                f"{LIBRARY_RAG_RERANKING_CONSEQUENCE}."
            )
    return ""


def _coverage_labels(source_types: Sequence[str]) -> str:
    """Render coverage source types as one display-vocabulary, escaped list.

    `_source_type_display_label` falls back to the raw, unrecognized
    `source_type` verbatim when it isn't one of `LIBRARY_RAG_SOURCE_TYPES`
    -- and these come from the service's `semantic_scope_coverage`
    diagnostics mapping, a swappable attribute this module does not control
    the shape of. Every other user-visible string this module builds is
    `escape_markup`-escaped before reaching a `Static`; these labels were
    the one gap (task-15 finding M8).
    """
    return ", ".join(
        escape_markup(_source_type_display_label(source_type))
        for source_type in source_types
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
        `""` when `rows` is empty AND no routing disclosure is present
        (edge case: zero results overall is the no-match/empty state's
        territory, not a coverage note enumerating every requested source as
        "uncovered" -- but a routing disclosure under
        `LIBRARY_RAG_ROUTE_NOTES_KEY` is a statement about HOW the search
        ran, still true and still needed at zero rows, so it renders alone
        there), when every requested source
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

        A row carrying one of the reranker's disclosure tags appends
        `library_rag_reranking_notice`'s sentence LAST (TASK-3502
        note-(a)): those tags previously had no UI consumer at all, so a
        reranking-enabled profile with a dead credential returned
        normal-looking results in silently unreranked order. It joins this
        one note channel rather than opening a second, competing one.

        A hybrid profile can also report `"keyword_only"` types (TASK-14752):
        sources whose rows on screen came entirely from the engine's FTS leg
        with no semantic hit. Those get their own sentence, `"Keyword matches
        only from: <types>."`, appended after the uncovered one -- because
        the uncovered sentence said "found nothing" about a source the user
        can see rows from, which reads as the opposite of the screen. The
        key is absent for semantic and plain profiles, whose copy is
        therefore unchanged.
    """
    route_notes = (
        tuple(
            str(item)
            for item in (diagnostics.get(LIBRARY_RAG_ROUTE_NOTES_KEY) or ())
        )
        if isinstance(diagnostics, Mapping)
        else ()
    )
    if not rows:
        # Coverage claims stay suppressed with no rows (see the Returns
        # section) -- but a ROUTING disclosure is a different fact, and zero
        # rows is exactly when it matters most: a plain-profile query that
        # matched nothing must still say vectors were never consulted, or
        # the quiet no-match line reads as a verdict on an index this search
        # never touched (review finding I2).
        return " ".join(
            sentence
            for sentence in (_route_note_sentence(note) for note in route_notes)
            if sentence
        )
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
    keyword_only = (
        tuple(str(item) for item in coverage.get("keyword_only", ()) or ())
        if isinstance(coverage, Mapping)
        else ()
    )
    message = (
        f"Semantic search found nothing from: {_coverage_labels(uncovered)}."
        if uncovered
        else ""
    )
    keyword_only_message = (
        f"Keyword matches only from: {_coverage_labels(keyword_only)}."
        if keyword_only
        else ""
    )
    parts = [
        part
        for part in (
            LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX
            if library_rag_all_matches_weak(rows)
            else "",
            message,
            keyword_only_message,
            *(_route_note_sentence(note) for note in route_notes),
            # Last: routing describes how retrieval RAN, this describes what
            # a post-retrieval stage failed to do to the order below.
            library_rag_reranking_notice(rows),
        )
        if part
    ]
    return " ".join(parts)


def library_rag_results_count_line(
    results: Sequence[LibraryRagResultRow], searched_query: str
) -> str:
    """Return the Evidence region's "N results for 'query'" headline.

    task-2859 item 10: the Evidence region used to have no headline naming
    how many results actually landed or what query produced them -- only
    the mode/top-k-driven "Evidence · top 5" line (`results_heading_text`,
    in `library_search_rag_panel.py`), which is deliberately STABLE across
    a client-side scope toggle (Task 8: "the heading is mode/top_k-driven,
    not row-count-driven"). This is a separate, additive line that DOES
    track `results` -- it renders directly above the row cards it counts,
    so it must agree with what the user can actually see below it,
    including right after a scope toggle hides a row.

    Args:
        results: The panel's current, already-scope-filtered evidence rows
            (`LibraryRagPanelState.results` -- what is actually rendered).
        searched_query: The query that produced `results`
            (`LibraryRagPanelState.searched_query`, NOT the live query box
            text -- mirrors `library_rag_empty_state_quiet_copy`'s same
            distinction, RAG-33/task-15 finding I3).

    Returns:
        `""` when `results` is empty (the empty/searching/recovery states
        have their own copy -- this line is Evidence-row-count territory
        only). Otherwise `"N result(s) for 'query'."`, escaped and clamped
        the same way `library_rag_empty_state_quiet_copy` quotes a query.
    """
    if not results:
        return ""
    count = len(results)
    noun = "result" if count == 1 else "results"
    display_query = _clamp_display_text(
        searched_query, LIBRARY_RAG_EMPTY_QUERY_QUOTE_MAX_CHARS
    )
    escaped_query = escape_markup(display_query)
    return f"{count} {noun} for '{escaped_query}'."


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
    #: Evidence region "N results for 'query'" headline (task-2859 item
    #: 10), built by `library_rag_results_count_line` from `results` and
    #: `searched_query`. Empty string whenever `results` is empty.
    results_count_line: str = ""
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
    #: The provider `resolve_library_rag_answer_provider` resolved for the
    #: single grounded-answer call CURRENTLY IN FLIGHT (PR-3 Task 3) --
    #: distinct from `answer.provider`, which only exists once a call has
    #: SETTLED onto `answer`. Feeds `library_rag_answer_children`'s
    #: "Asking <provider>..." in-flight line: the one moment the answer
    #: region has something true to say about cost before the outcome
    #: (and therefore the token usage) is known at all -- which provider is
    #: about to be billed. `""` (the default -- every call site that
    #: predates this field) keeps the prior generic "Generating answer..."
    #: line, since there is no provider name to report.
    in_flight_answer_provider: str = ""

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
        provider_name: str | None = None,
        provider_credential_recovery: str = "",
        selected_source_types: Sequence[str] | None = None,
        history: Sequence[str] = (),
        history_collapsed: bool = False,
        diagnostics: Mapping[str, Any] | None = None,
        answer: LibraryRagAnswer | None = None,
        in_flight_answer_provider: str = "",
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
            results: Retrieval result rows or mappings. Filtered against
                `scope.selected_source_types` before use whenever
                `selected_source_types` is not `None` (D4/task-5): a row
                whose provenance canonicalizes to a source type NOT in the
                current (count-intersected) scope is dropped from the
                returned state's `results`/`selected_result`/
                `can_use_console`, so a scope toggle flipped off -- or a
                source's local count dropping to zero -- after retrieval
                already landed hides that source's rows in this exact
                snapshot rather than only affecting the next run.
                `selected_source_types=None` (every call site that
                predates this fix) skips the filter entirely.
            selected_result_id: Result ID selected for inspector/Console handoff.
            retrieval_status: Explicit retrieval status override.
            recovery_copy: Explicit retrieval recovery copy from a service outcome.
            recovery_selector: Stable selector used for explicit retrieval recovery.
            dependencies_ready: Whether Search/RAG optional dependencies are available.
            index_ready: Whether the selected source scope has an index.
            provider_name: The provider `resolve_library_rag_answer_provider`
                resolved, or `None`/blank when none is configured -- the
                SOLE source of RAG-mode provider readiness, forwarded
                unchanged to `LibraryRagQueryState.from_values` (PR-T2
                Task 4 review: collapsed from two independently settable
                parameters, `provider_ready: bool` and this, which could
                disagree -- see that method's docstring). `None` is the
                default, so `rag` mode is now BLOCKED by default (a
                behavior change from the pre-collapse `provider_ready:
                bool = True` default) -- every call site that wants a
                ready `rag`-mode state must now name a provider
                explicitly, which is the whole point: readiness can no
                longer be asserted without a name to back it up.
            provider_credential_recovery: Forwarded unchanged to
                `LibraryRagQueryState.from_values` -- the remedy shown when
                a provider IS named in config but cannot authenticate. See
                that method's docstring; it is display copy only and never
                affects readiness.
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
            in_flight_answer_provider: The provider resolved for the answer
                call currently in flight (PR-3 Task 3), or `""` (the
                default -- every call site that predates this parameter)
                when none is in flight or none was resolved yet.

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
            provider_name=provider_name,
            provider_credential_recovery=provider_credential_recovery,
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
        # D4/task-5: a scope toggle flipped OFF after retrieval already
        # landed used to leave that source's rows displayed, selectable,
        # and stageable into Console even though the Sources toggle read
        # "off" (the toggle only reset state for the NEXT run). Hide,
        # don't grey -- the scope line already claims the source is off,
        # so showing its rows would be the lie. This is a pure,
        # one-snapshot filter over the rows already built above (no
        # re-query): everything downstream -- coverage note, status
        # classification, selection resolution, `can_use_console` --
        # reads this same filtered `result_rows`, so a selection pointing
        # at a just-hidden row resolves to `None` for free below, and
        # toggling the source back ON restores its rows from this exact
        # `results` argument on the very next call. A row whose
        # provenance `source_type` cannot be attributed to any toggle
        # (`scope_source_type == ""`) is never hidden -- see
        # `LibraryRagResultRow.scope_source_type`.
        #
        # Hybrid basis (review round 2, I1): `selected_source_types is
        # None` is the "no explicit scope was ever supplied" sentinel --
        # every call site that predates this fix, and every gate16 fixture
        # that never exercises scope deselection -- so that case skips
        # filtering entirely, preserving prior behavior byte-for-byte
        # (`scope.selected_source_types` itself is NEVER `None`, so this
        # escape hatch has to key off the raw argument's None-ness, not
        # `scope`'s already-resolved tuple). The only caller that ever
        # passes a non-`None` value is the real screen
        # (`_library_rag_panel_state`), and for that case this filters
        # against `scope.selected_source_types` -- the count-intersected
        # local computed above, NOT the raw argument. An earlier draft
        # filtered against the raw argument directly (ignoring
        # availability) to avoid hiding rows under a test fixture with an
        # unrealistic zero-count source; that reopens D4's exact symptom
        # for a real, reachable case: land Notes evidence, delete the
        # backing note elsewhere in Library (count -> 0 via
        # `_refresh_local_source_snapshot`), return to Search without
        # re-querying -- the toggle strip's OWN marker
        # (`scope.selected_source_types`-driven) already reads "○ Notes
        # (0)", so evidence must follow that same count-intersected
        # signal or the toggle-vs-evidence lie just gets a different
        # trigger (count drift instead of a toggle press). The
        # unrealistic-fixture problem this traded away is fixed at its
        # actual source instead: the fixtures now seed non-zero counts for
        # every source their canned rows reference (see
        # `Tests/UI/test_library_shell.py`).
        result_rows = tuple(
            row
            for row in result_rows
            if not row.scope_source_type
            or selected_source_types is None
            or row.scope_source_type in scope.selected_source_types
        )
        coverage_note = library_rag_coverage_note(diagnostics, result_rows)
        results_count_line = library_rag_results_count_line(
            result_rows, normalized_searched_query
        )
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

        # "answering" counts as usable evidence (PR-3 Task 4 review): the
        # retrieval that produced `selected_result` has already settled and
        # its bundle is frozen -- generation cannot change what is stageable.
        # Disabling the action mid-generation greyed the button AND made the
        # `u` key answer "Run a query and select usable evidence before
        # sending to Console.", which is false in that state: a query HAS
        # run and evidence IS selected.
        can_use_console = (
            normalized_status in {"ready", "answering"} and selected_result is not None
        )
        return cls(
            scope=scope,
            query_state=query_state,
            results=result_rows,
            retrieval_status=normalized_status,
            next_action=next_action,
            use_in_console_action=LibraryRagActionState(
                label="Use in Console",
                enabled=can_use_console,
                # The panel mounts exactly one Console-handoff button per
                # evidence result -- the results-lane one below the
                # selected row (`library_rag_result_row_children`'s
                # sibling in `library_rag_results_body_children`), never a
                # separate inspector-column button. This must match that
                # mounted id, not an id no widget actually carries (task-7
                # PR-2 leftover: `LIBRARY_RAG_USE_IN_CONSOLE_ACTION_ID`
                # pointed at a retired 3-pane inspector button that was
                # never rebuilt for this canvas).
                widget_id="library-rag-use-selected-in-console",
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
            results_count_line=results_count_line,
            searched_query=normalized_searched_query,
            answer=answer,
            in_flight_answer_provider=str(in_flight_answer_provider or ""),
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
