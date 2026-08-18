"""Library-native Search/RAG display-state contracts."""

from __future__ import annotations

import pytest

from tldw_chatbook.Library import library_local_rag_search_service as _semantic_module
from tldw_chatbook.Library import library_rag_state as _rag_state_module
from tldw_chatbook.Library.library_rag_state import (
    LIBRARY_RAG_EMPTY_STATE_SELECTOR,
    LIBRARY_RAG_FALLBACK_TOP_K,
    LIBRARY_RAG_NO_SOURCES_GATE_COPY,
    LIBRARY_RAG_ROUTE_NOTES_KEY,
    LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY,
    LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
    LIBRARY_RAG_SNIPPET_DISPLAY_MAX_CHARS,
    LIBRARY_RAG_TOP_K_MAX,
    LibraryRagPanelState,
    LibraryRagQueryState,
    LibraryRagResultRow,
    LibraryRagScopeState,
    library_rag_all_matches_weak,
    library_rag_coverage_note,
    library_rag_empty_state_quiet_copy,
    library_rag_paid_mode_notice,
    library_rag_profile_top_k,
    library_rag_results_count_line,
    library_rag_score_suffix,
    library_rag_scope_summary,
    searching_status_line,
    update_search_history,
)


def test_scope_state_exposes_library_source_scope_and_empty_recovery() -> None:
    scope = LibraryRagScopeState.from_source_counts(
        notes=2,
        media=1,
        conversations=0,
        prompts=0,
        workspaces=0,
        collections=0,
        selected=("notes", "media"),
    )

    assert scope.heading == "Source Scope: All local sources"
    assert scope.total_count == 3
    assert scope.has_available_sources is True
    assert tuple(option.source_type for option in scope.options) == (
        "notes",
        "media",
        "conversations",
        "prompts",
        "workspaces",
        "collections",
    )
    assert scope.option_by_type("notes").label == "Notes"
    assert scope.option_by_type("notes").count_label == "2 sources"
    assert scope.option_by_type("notes").selected is True
    assert scope.option_by_type("conversations").available is False
    assert (
        "No conversations available" in scope.option_by_type("conversations").recovery
    )
    assert scope.option_by_type("prompts").label == "Prompts"
    assert scope.option_by_type("prompts").available is False

    empty_scope = LibraryRagScopeState.from_source_counts(
        notes=0,
        media=0,
        conversations=0,
        prompts=0,
        workspaces=0,
        collections=0,
    )

    assert empty_scope.has_available_sources is False
    assert empty_scope.status == "blocked"
    # (task-185) The no-sources state is ONE quiet gate line -- never the
    # retired Unavailable/Why/Next/Recovery/Owner dump or its checklist.
    assert empty_scope.recovery_copy == LIBRARY_RAG_NO_SOURCES_GATE_COPY
    assert "Owner:" not in empty_scope.recovery_copy
    assert "Recovery checklist" not in empty_scope.recovery_copy


class TestLibraryRagScopeSummary:
    """(RAG-32) `library_rag_scope_summary` is the ONE builder both the
    panel's compose() and the screen's incremental refresh path delegate
    to -- see `Tests/UI/test_library_shell.py::
    test_library_shell_search_scope_strip_refresh_path_uses_shared_copy`
    for the agreement test that pins the two seams stay in sync.

    Live UAT (critique RAG-32): the strip printed the hardcoded "all local
    sources" copy even when a user had switched a source off. The common
    case (every available source selected) keeps that exact copy verbatim;
    only a genuine subset gets the explicit list.
    """

    def test_all_selected_keeps_the_unchanged_common_case_copy(self):
        scope = LibraryRagScopeState.from_source_counts(
            notes=2, media=1, conversations=3, prompts=1
        )
        assert scope.selected_source_types == (
            "notes",
            "media",
            "conversations",
            "prompts",
        )
        assert library_rag_scope_summary(scope) == LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY

    def test_subset_lists_selected_and_parenthesizes_the_off_sources(self):
        scope = LibraryRagScopeState.from_source_counts(
            notes=2,
            media=1,
            conversations=3,
            prompts=1,
            selected=("notes", "conversations"),
        )
        assert library_rag_scope_summary(scope) == (
            "Scope: Notes, Conversations (Media, Prompts off)"
        )

    def test_subset_orders_selected_and_off_in_canonical_source_order(self):
        """Selected/off lists follow LIBRARY_RAG_SOURCE_TYPES order, not
        the order `selected` was passed in."""
        scope = LibraryRagScopeState.from_source_counts(
            notes=2,
            media=1,
            conversations=3,
            prompts=1,
            selected=("prompts", "media"),
        )
        assert library_rag_scope_summary(scope) == (
            "Scope: Media, Prompts (Notes, Conversations off)"
        )

    def test_single_source_off_reads_as_a_compact_callout(self):
        """The RAG-32 headline scenario: a user switches exactly one
        source off and the strip must say so, not claim "all local
        sources"."""
        scope = LibraryRagScopeState.from_source_counts(
            notes=2,
            media=1,
            conversations=3,
            prompts=1,
            selected=("notes", "conversations", "prompts"),
        )
        assert library_rag_scope_summary(scope) == (
            "Scope: Notes, Conversations, Prompts (Media off)"
        )

    def test_no_sources_selected_reads_as_none_selected_not_all_off_list(self):
        """Deselect-all is already surfaced by the run gate's own quiet
        line ("Select at least one Library source.") -- repeating every
        available source in a parenthetical "off" list here would just be
        noise restating the same fact."""
        scope = LibraryRagScopeState.from_source_counts(
            notes=2, media=1, selected=()
        )
        assert library_rag_scope_summary(scope) == "Scope: no sources selected"

    def test_none_available_keeps_the_unchanged_common_case_copy(self):
        """The empty-library edge is already owned by
        `LIBRARY_RAG_NO_SOURCES_GATE_COPY` elsewhere on screen -- this
        builder must not invent a second, conflicting message for it."""
        scope = LibraryRagScopeState.from_source_counts()
        assert scope.has_available_sources is False
        assert library_rag_scope_summary(scope) == LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY


def test_query_state_blocks_empty_query_and_runtime_blockers(monkeypatch) -> None:
    empty_query = LibraryRagQueryState.from_values(query="", mode="rag")

    assert empty_query.mode == "rag"
    assert empty_query.mode_label == "RAG Answer"
    assert empty_query.status == "blocked"
    assert empty_query.run_action.enabled is False
    assert empty_query.run_action.disabled_reason == "Enter a question or search query."
    assert "Owner: user." in empty_query.recovery_copy
    assert "Next: Type a query before running Search/RAG." in empty_query.recovery_copy

    missing_index = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="search",
        index_ready=False,
    )

    assert missing_index.mode == "search"
    assert missing_index.mode_label == "Search"
    assert missing_index.status == "blocked"
    assert missing_index.run_action.disabled_reason == (
        "Index selected Library sources before querying."
    )

    # An unparseable count resolves to the ACTIVE PROFILE's depth since
    # TASK-15020/B3 (it was the literal 5 before); pinned against a patched
    # profile so this stays about coercion, not about which profile ships.
    _patch_profile_depth(monkeypatch, 15)
    ready_query = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="unknown",
        top_k="bad",
        provider_name="openai",
    )

    assert ready_query.mode == "rag"
    assert ready_query.top_k == 15
    assert ready_query.status == "ready"
    assert ready_query.run_action.enabled is True


def test_query_state_provider_gate_is_rag_only() -> None:
    """(PR-3 task 2) `provider_name` feeds the ONE existing blocked branch
    at `Library/library_rag_state.py:893-897` -- and that branch is gated on
    `normalized_mode == "rag"`. A not-ready provider must therefore block
    `rag` mode with the pre-existing "Select a provider/model..." copy, and
    must leave `search` (keyword) mode completely unaffected: keyword mode
    never calls a provider at all, so gating it on one would block a query
    that could otherwise run.

    (PR-T2 Task 4 review) Originally written against a separate
    `provider_ready: bool` parameter, since collapsed into `provider_name`
    alone (readiness is now DERIVED from it -- see that method's
    docstring); updated to the new shape without changing what this test
    verifies.
    """
    blocked_rag = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="rag",
        provider_name=None,
    )

    assert blocked_rag.status == "blocked"
    assert blocked_rag.run_action.enabled is False
    assert blocked_rag.run_action.disabled_reason == (
        "Select a provider/model before asking for a RAG answer."
    )
    assert "Owner: LLM provider." in blocked_rag.recovery_copy
    assert (
        "Next: Select a provider and model before running a RAG answer."
        in blocked_rag.recovery_copy
    )

    unaffected_search = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="search",
        provider_name=None,
    )

    assert unaffected_search.status == "ready"
    assert unaffected_search.run_action.enabled is True
    assert unaffected_search.run_action.disabled_reason == ""

    ready_rag = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="rag",
        provider_name="openai",
    )

    assert ready_rag.status == "ready"
    assert ready_rag.run_action.enabled is True


def test_query_state_ready_answer_provider_names_the_paid_mode_provider() -> None:
    """(PR-T2 Task 4) `ready_answer_provider` feeds the quiet line's paid-
    mode notice -- non-empty ONLY when `rag` mode is actually ready to run,
    which (post-collapse) is itself derived from `provider_name`. Search
    mode and every blocked state leave it empty.
    """
    ready = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="rag",
        provider_name="openai",
    )
    assert ready.status == "ready"
    assert ready.ready_answer_provider == "openai"

    search_mode = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="search",
        provider_name="openai",
    )
    assert search_mode.status == "ready"
    assert search_mode.ready_answer_provider == ""

    blocked_empty_query = LibraryRagQueryState.from_values(
        query="",
        mode="rag",
        provider_name="openai",
    )
    assert blocked_empty_query.status == "blocked"
    assert blocked_empty_query.ready_answer_provider == ""

    blocked_no_provider = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="rag",
        provider_name=None,
    )
    assert blocked_no_provider.status == "blocked"
    assert blocked_no_provider.ready_answer_provider == ""

    blocked_default = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="rag",
    )
    assert blocked_default.status == "blocked"
    assert blocked_default.ready_answer_provider == ""


@pytest.mark.parametrize("name", ["openai", "anthropic", "local-vllm", "  ollama  "])
def test_ready_rag_mode_always_names_its_provider_invariant(name: str) -> None:
    """(PR-T2 Task 4 review) The footgun this test exists to catch:
    `provider_ready`/`provider_name` used to be two independently
    settable parameters that could disagree -- a caller could pass
    `provider_ready=True` with no name (or the reverse) and silently
    produce "the mode can spend money and the quiet line says nothing",
    the exact inversion Task 4 was written to fix, with no test able to
    catch it because the API *permitted* the disagreement. Collapsing to
    the single `provider_name` parameter makes that combination
    impossible to construct rather than merely untested: this asserts the
    INVARIANT itself, not just one happy-path example -- for ANY
    non-blank name (including one needing a strip), a ready `rag`-mode
    state's `ready_answer_provider` is always that same name, never
    empty, and never a different name than what was supplied.
    """
    state = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="rag",
        provider_name=name,
    )
    assert state.status == "ready"
    assert state.ready_answer_provider == name.strip()
    assert state.ready_answer_provider != ""


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_blank_provider_name_never_yields_a_ready_rag_mode_invariant(blank) -> None:
    """The other half of the same invariant (PR-T2 Task 4 review): `None`,
    `""`, and whitespace-only all mean "no provider" -- `rag` mode is
    blocked for all three, and there is nothing to derive a notice from.
    """
    state = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="rag",
        provider_name=blank,
    )
    assert state.status == "blocked"
    assert state.ready_answer_provider == ""
    assert state.run_action.disabled_reason == (
        "Select a provider/model before asking for a RAG answer."
    )


def test_provider_ready_parameter_no_longer_exists() -> None:
    """(PR-T2 Task 4 review) `provider_ready` is fully retired as a
    caller-facing parameter, not merely deprecated alongside `provider_
    name` -- passing it is a `TypeError`, proving the two-parameter
    footgun is now impossible to even attempt, not just discouraged.
    """
    with pytest.raises(TypeError):
        LibraryRagQueryState.from_values(
            query="summarize the policy",
            mode="rag",
            provider_ready=True,  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError):
        LibraryRagPanelState.from_values(
            query="summarize the policy",
            mode="rag",
            provider_ready=True,  # type: ignore[call-arg]
        )


def test_library_rag_paid_mode_notice_names_the_provider() -> None:
    assert library_rag_paid_mode_notice("openai") == (
        "RAG Answer sends your question and the evidence to openai. "
        "Search stays local."
    )


def test_panel_state_threads_provider_name_into_query_state() -> None:
    """(PR-T2 Task 4) `LibraryRagPanelState.from_values`'s `provider_name`
    reaches `query_state.ready_answer_provider` unchanged -- the screen
    resolves it once (`resolve_library_rag_answer_provider()[0]`) and
    passes it straight through, with readiness derived from it rather
    than asserted separately.
    """
    ready = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="summarize the policy",
        mode="rag",
        provider_name="anthropic",
    )
    assert ready.query_state.ready_answer_provider == "anthropic"

    default_state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="summarize the policy",
        mode="rag",
    )
    assert default_state.query_state.ready_answer_provider == ""


def test_query_state_blocked_is_empty_query_and_no_scope_properties() -> None:
    """A1: `blocked_is_empty_query`/`blocked_is_no_scope` key the Search
    canvas's single quiet line, distinct from real-failure blockers (which
    keep the full callout + recovery-copy presentation).
    """
    empty_query = LibraryRagQueryState.from_values(query="")
    assert empty_query.blocked_is_empty_query is True
    assert empty_query.blocked_is_no_scope is False

    no_scope = LibraryRagQueryState.from_values(query="find it", has_source_scope=False)
    assert no_scope.blocked_is_empty_query is False
    assert no_scope.blocked_is_no_scope is True

    unsafe_query = LibraryRagQueryState.from_values(query="<script>alert(1)</script>")
    assert unsafe_query.blocked_is_empty_query is False
    assert unsafe_query.blocked_is_no_scope is False

    missing_index = LibraryRagQueryState.from_values(query="find it", index_ready=False)
    assert missing_index.blocked_is_empty_query is False
    assert missing_index.blocked_is_no_scope is False

    ready = LibraryRagQueryState.from_values(query="find it")
    assert ready.blocked_is_empty_query is False
    assert ready.blocked_is_no_scope is False


def test_query_state_validates_and_sanitizes_external_values(monkeypatch) -> None:
    # 500 is out of `LIBRARY_RAG_TOP_K_MAX` range, so it falls back -- to the
    # active profile's depth since TASK-15020/B3, not the old literal 5.
    _patch_profile_depth(monkeypatch, 15)
    unsafe_query = LibraryRagQueryState.from_values(
        query="<script>alert('x')</script>",
        mode="<b>rag</b>",
        top_k=500,
    )

    assert unsafe_query.query == ""
    assert unsafe_query.status == "blocked"
    assert unsafe_query.run_action.disabled_reason == (
        "Enter a safe question or search query."
    )
    assert unsafe_query.mode == "rag"
    assert unsafe_query.top_k == 15

    bounded_query = LibraryRagQueryState.from_values(
        query="Find policy evidence",
        mode="search",
        top_k=50,
    )

    assert bounded_query.status == "ready"
    assert bounded_query.top_k == 50


def test_result_row_preserves_snippet_score_citations_and_provenance() -> None:
    row = LibraryRagResultRow.from_result(
        {
            "document_title": "Incident Review",
            "snippet": "Root cause was an expired credential.",
            "score": "0.91",
            "source_id": "note-42",
            "chunk_id": "chunk-7",
            "runtime_backend": "local-fts",
            "citations": [
                {"label": "Incident Review p.2", "url": "file:///incident.md"},
                "Ops note",
            ],
            "provenance": {"index": "library", "rank": 1},
        }
    )

    assert row.result_id == "note-42:chunk-7"
    assert row.title == "Incident Review"
    assert row.snippet == "Root cause was an expired credential."
    assert row.score == 0.91
    assert row.source_id == "note-42"
    assert row.chunk_id == "chunk-7"
    assert row.runtime_backend == "local-fts"
    assert row.citation_labels == ("Incident Review p.2", "Ops note")
    assert row.provenance["index"] == "library"
    assert row.provenance["rank"] == 1

    malformed = LibraryRagResultRow.from_result(
        {
            "title": "",
            "score": "not-a-number",
            "citations": [{"url": "https://example.test"}],
        }
    )

    assert malformed.title == "Untitled source"
    assert malformed.score is None
    assert malformed.citation_labels == ("https://example.test",)


def test_result_row_sanitizes_display_text_and_preserves_numeric_ids() -> None:
    """Title/citation-label assertions updated for the 2026-08-03 task-15
    finding-1 fix: `_sanitize_display_text` no longer HTML-entity-escapes
    for display (a Rich `Static` never decodes "&lt;"/"&gt;" back to literal
    characters, so the old "&lt;b&gt;Release&lt;/b&gt;" expectation pinned
    the same over-escaping bug finding 1 fixed for "&amp;" -- see
    `test_sanitize_display_text_decodes_html_entities_for_display`).
    `<b>`/`<i>` are not dangerous patterns (only `<script>`/`javascript:`/
    `onclick=`/`onerror=` are), so they now pass through as literal text."""
    row = LibraryRagResultRow.from_result(
        {
            "title": "<b>Release</b>",
            "snippet": "Line one\nLine two <script>alert(1)</script>",
            "source_id": 0,
            "chunk_id": 0,
            "citations": [
                {"label": "<i>Citation</i>", "url": "javascript:alert(1)"},
            ],
        }
    )

    assert row.result_id == "0:0"
    assert row.source_id == "0"
    assert row.chunk_id == "0"
    assert row.title == "<b>Release</b>"
    assert "Line one\nLine two" in row.snippet
    assert "<script" not in row.snippet
    assert row.citation_labels == ("<i>Citation</i>",)
    assert row.citations[0].url == ""


def test_result_row_display_snippet_strips_markdown_structure() -> None:
    """(RAG-30/31) Evidence rows carry raw Markdown from notes/media -- the
    on-screen `display_snippet` reads as plain prose, never literal
    `##`/`**`/`-` notation. The stored `snippet` is untouched (below)."""
    row = LibraryRagResultRow.from_result(
        {
            "title": "Project Doc",
            "snippet": "## Project Overview\n**Status:** Planning\n- item",
        }
    )

    assert row.display_snippet == "Project Overview Status: Planning item"
    # The full, structured snippet is preserved for Console handoff.
    assert row.snippet == "## Project Overview\n**Status:** Planning\n- item"


def test_result_row_display_snippet_strips_links_and_code_syntax() -> None:
    row = LibraryRagResultRow.from_result(
        {
            "title": "Doc",
            "snippet": (
                "See [the guide](https://example.test/guide) and run "
                "`make test` for details."
            ),
        }
    )

    assert row.display_snippet == "See the guide and run make test for details."


def test_result_row_display_snippet_preserves_technical_identifiers() -> None:
    """(RAG-30/31 review) The emphasis-marker strip must not delete `_`/`*`
    characters embedded in real content -- snippets exist so a user can
    judge relevance from quoted source text, and a snake_case identifier, a
    filename, or an env-var name silently losing its underscores defeats
    that. Pins the strip-vs-preserve boundary from the preserve side; the
    strip side is pinned above and by the `**bold**`/`_italic_` cases."""
    row = LibraryRagResultRow.from_result(
        {
            "title": "Config Notes",
            "snippet": (
                "Call chat_api_call() with top_k tuned via OPENAI_API_KEY; "
                "see my_notes_2026.md for user_id=42 details."
            ),
        }
    )

    assert row.display_snippet == row.snippet
    assert "chat_api_call()" in row.display_snippet
    assert "top_k" in row.display_snippet
    assert "OPENAI_API_KEY" in row.display_snippet
    assert "my_notes_2026.md" in row.display_snippet
    assert "user_id=42" in row.display_snippet


@pytest.mark.parametrize(
    ("snippet", "preserved"),
    [
        ("config [*/etc/hosts*]", ("etc", "hosts")),
        ("[*/tmp/*] is the scratch dir", ("tmp", "scratch")),
        ("[_TODO_] finish this", ("TODO", "finish")),
        ("[*bold*] emphasis in brackets", ("bold", "emphasis")),
    ],
)
def test_result_row_display_snippet_bracketed_emphasis_stays_inert(
    snippet: str, preserved: tuple[str, ...]
) -> None:
    """(final-review C1) Markdown stripping must run BEFORE the terminal
    markup escape, never after it.

    When the strip ran on already-escaped text, removing the `*`/`_`
    emphasis delimiters inside a bracket exposed a `[...]` that
    `escape_markup` had deliberately left alone (rich only escapes brackets
    that already look like tags). Ordinary technical note content then
    became LIVE Textual markup: `config [*/etc/hosts*]` turned into
    `config [/etc/hosts]`, which raises `MarkupError` and crashes the app,
    and `[_TODO_] finish this` turned into `[TODO] finish this`, whose text
    Textual silently swallowed as an unknown tag.

    Pins both halves of the contract: nothing parses as markup, and the
    words a user needs in order to judge relevance survive."""
    from rich.text import Text

    row = LibraryRagResultRow.from_result({"title": "Notes", "snippet": snippet})

    rendered = Text.from_markup(row.display_snippet)
    assert rendered.spans == []
    for word in preserved:
        assert word in rendered.plain


def test_result_row_display_snippet_clamps_long_text_at_word_boundary() -> None:
    words = [f"word{i}" for i in range(120)]
    long_text = " ".join(words)  # well over 320 plain-prose chars, no Markdown
    row = LibraryRagResultRow.from_result({"title": "Long", "snippet": long_text})

    # Stored snippet keeps the full text -- only the display projection clamps.
    assert row.snippet == long_text
    assert len(row.snippet) > LIBRARY_RAG_SNIPPET_DISPLAY_MAX_CHARS

    display = row.display_snippet
    assert len(display) <= LIBRARY_RAG_SNIPPET_DISPLAY_MAX_CHARS
    assert display.endswith("…")
    body = display[:-1].rstrip()
    assert body and not body.endswith(" ")
    # Clamp lands on a word boundary: every token in the clamped body is a
    # whole word from the source, never a mid-word cut.
    assert set(body.split(" ")) <= set(words)


def test_result_row_display_snippet_passes_through_short_snippet_unclamped() -> None:
    row = LibraryRagResultRow.from_result(
        {
            "title": "Short",
            "snippet": "Root cause was an expired credential.",
        }
    )

    assert row.display_snippet == row.snippet
    assert "…" not in row.display_snippet


def test_sanitize_display_text_decodes_html_entities_for_display() -> None:
    """(RAG-30/31, revised 2026-08-03 task-15 live-UAT finding 1) The
    display surface is a Textual/Rich `Static`, which renders Rich markup,
    not HTML -- it never decodes HTML entities back to literal characters.
    The original RAG-30/31 fix kept text HTML-escaped (re-escaping once,
    not twice) to avoid "R&amp;amp;D" on screen, but that still put the
    literal string "&amp;" on screen for a user who typed (or whose source
    stored) a plain "&" -- confirmed live: a Note containing "Alice & Bob"
    rendered as "Alice &amp; Bob" in the evidence card. Text arriving
    HTML-entity-escaped (e.g. an upstream source that ran html.escape on
    "R&D" and stored "R&amp;D Report") must now decode to the literal
    character for display instead."""
    row = LibraryRagResultRow.from_result(
        {
            "title": "R&amp;D Report",
            "snippet": "Budget covers R&amp;D spending this quarter.",
        }
    )

    assert row.title == "R&D Report"
    assert "&amp;" not in row.title

    assert row.snippet == "Budget covers R&D spending this quarter."
    assert "&amp;" not in row.snippet
    assert row.display_snippet == row.snippet
    assert "&amp;" not in row.display_snippet


def test_result_row_stays_inert_for_script_markup_and_encoded_payloads_round_trip() -> (
    None
):
    """Security invariant: the html.unescape pre-step must not open a
    bypass. A literal <script> block stays fully removed, live Rich markup
    ([bold]/[red]) stays neutralized behind a backslash.

    2026-08-03 task-15 finding-1 fix, ROUND-2 CORRECTION: this test was
    briefly (and wrongly) reported as "left unmodified" by the fix. It was
    NOT -- a scoping error in that edit orphaned this function's last 7
    assertions and 2 comments outside its body, and a later cleanup pass
    deleted them believing them to be stray leftovers, instead of
    recognizing they were this test's own tail. Reconstructed from git
    history (commit 3f65de5d4) and re-verified line by line against the
    fix: 5 of those 7 assertions still hold and are restored below
    unchanged. The other 2 pinned that an entity-encoded payload
    (`&lt;script&gt;...&lt;/script&gt;`) was a "fixed point" of
    unescape+escape -- it round-tripped back to the SAME single-escaped,
    visually-inert text, still literally readable as
    "&lt;script&gt;...". That is no longer true and is not restored: under
    the fix, `_sanitize_display_text` no longer re-escapes for display, and
    its dangerous-pattern scrubber now runs a SECOND time after unescaping
    (closing the sequencing gap the fix targets -- see the sequencing-gap
    test below), so an entity-encoded <script> payload is decoded and then
    stripped outright by that second pass, exactly like a literal one. The
    new, explicit contract: encoded and literal <script> payloads both end
    up fully removed -- neither survives anywhere, encoded, decoded, or
    escaped."""
    row = LibraryRagResultRow.from_result(
        {
            "title": "[bold]spoof[/] &lt;script&gt;alert(1)&lt;/script&gt;",
            "snippet": (
                "<script>alert(1)</script> [red]inject[/] "
                "&lt;script&gt;alert(2)&lt;/script&gt;"
            ),
        }
    )

    for text in (row.title, row.snippet, row.display_snippet):
        assert "<script" not in text.lower()
        assert "&amp;lt;" not in text
        assert "&amp;amp;" not in text

    # Restored verbatim from the original (commit 3f65de5d4) -- still hold
    # under the fix: Rich markup is escaped (backslash breaks the live
    # "[tag]...[/]" run) rather than merely present-but-inert-looking.
    assert "[bold]spoof[/]" not in row.title
    assert r"\[bold]spoof\[/]" in row.title
    assert "[red]inject[/]" not in row.snippet
    assert r"\[red]inject\[/]" in row.snippet
    assert "[red]inject[/]" not in row.display_snippet

    # NEW (round-2 correction): the already-encoded payload is no longer a
    # fixed point of unescape+escape -- it must not survive at all, encoded
    # or decoded, in any of the three text fields.
    for text in (row.title, row.snippet, row.display_snippet):
        assert "script" not in text.lower()
        assert "alert(" not in text


def test_sanitize_display_text_rescrubs_dangerous_patterns_unescaping_reveals() -> None:
    """(2026-08-03 task-15 finding-1 fix review) Pins the exact sequencing
    gap the reviewer flagged: `_remove_dangerous_display_patterns` runs once
    on the RAW/still-entity-encoded text, before any unescaping -- an
    entity-encoded `<script>` payload (`&lt;script&gt;...&lt;/script&gt;`)
    does not look dangerous at that point, so it passes through untouched.
    Naively deleting `html.escape` and returning `escape_markup(html.
    unescape(text))` would then decode that payload into a LIVE `<script>`
    tag that reaches the final `Static(...)` unescaped -- `escape_markup`
    only neutralizes Rich's own `[`/`]` markup syntax, never `<script`.
    The fix re-scrubs a second time, after unescaping, closing the gap."""
    row = LibraryRagResultRow.from_result(
        {
            "title": "&lt;script&gt;alert(1)&lt;/script&gt;",
            "snippet": (
                "safe text [danger] &lt;script&gt;alert(2)&lt;/script&gt; more text"
            ),
        }
    )

    for text in (row.title, row.snippet, row.display_snippet):
        assert "<script" not in text.lower()
        assert "&amp;" not in text

    # A bracket payload (Rich markup syntax, not HTML) must still come out
    # backslash-escaped rather than stripped or left live -- the re-scrub
    # pass only targets <script>/javascript:/onclick=/onerror=, never `[`/`]`.
    assert r"\[danger]" in row.snippet


def test_result_row_provenance_is_immutable_snapshot() -> None:
    row = LibraryRagResultRow.from_result(
        {
            "title": "Release Notes",
            "provenance": {"index": "library", "rank": 1},
        }
    )

    assert row.provenance["rank"] == 1
    with pytest.raises(TypeError):
        row.provenance["rank"] = 2


def test_row_badge_label_bare_source_type_when_no_signal() -> None:
    """UX wave M5: default workspace, zero citations, eligible -> just the
    source type, no "all workspaces"/"0 citations"/"eligible" filler.
    """
    row = LibraryRagResultRow.from_result(
        {
            "title": "Roadmap",
            "provenance": {"source_type": "media"},
        }
    )

    assert row.row_badge_label == "Media"


def test_row_badge_label_includes_citations_only_when_present() -> None:
    row = LibraryRagResultRow.from_result(
        {
            "title": "Roadmap",
            "provenance": {"source_type": "media"},
            "citations": [{"label": "Roadmap p.1"}, {"label": "Roadmap p.2"}],
        }
    )

    assert row.row_badge_label == "Media · 2 citations"


def test_row_badge_label_maps_blocked_eligibility_to_excluded_from_context() -> None:
    row = LibraryRagResultRow.from_result(
        {
            "title": "Roadmap",
            "provenance": {"source_type": "media", "active_context_eligible": False},
        }
    )

    assert row.row_badge_label == "Media · excluded from context"


def test_row_badge_label_includes_non_default_workspace() -> None:
    row = LibraryRagResultRow.from_result(
        {
            "title": "Roadmap",
            "provenance": {"source_type": "media", "workspace_id": "workspace-a"},
        }
    )

    assert row.row_badge_label == "Media · workspace-a"


def test_row_badge_label_joins_with_middle_dot_not_pipe() -> None:
    row = LibraryRagResultRow.from_result(
        {
            "title": "Roadmap",
            "provenance": {
                "source_type": "media",
                "workspace_id": "workspace-a",
                "active_context_eligible": False,
            },
            "citations": [{"label": "Roadmap p.1"}],
        }
    )

    assert (
        row.row_badge_label
        == "Media · workspace-a · 1 citation · excluded from context"
    )
    assert "|" not in row.row_badge_label


def test_panel_state_tracks_retrieval_status_and_console_action_readiness() -> None:
    blocked = LibraryRagPanelState.from_values(
        source_counts={
            "notes": 0,
            "media": 0,
            "conversations": 0,
            "workspaces": 0,
            "collections": 0,
        },
        query="What changed?",
    )

    assert blocked.retrieval_status == "blocked"
    assert blocked.use_in_console_action.enabled is False
    assert blocked.use_in_console_action.disabled_reason == (
        "Run a query and select usable evidence before sending to Console."
    )
    # (task-185) The panel's no-sources recovery copy is the single quiet
    # gate line, and the inspector's next action stays user-facing.
    assert blocked.recovery_copy == LIBRARY_RAG_NO_SOURCES_GATE_COPY
    assert blocked.next_action == "Import media or create notes, then search."

    result = LibraryRagResultRow.from_result(
        {
            "title": "Release Notes",
            "snippet": "Gate 1.6 adds Library-native Search/RAG.",
            "score": 0.88,
            "source_id": "note-release",
            "chunk_id": "chunk-1",
            "citations": ["Release Notes #1"],
        }
    )
    ready = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="What did Gate 1.6 add?",
        results=(result,),
        selected_result_id=result.result_id,
        provider_name="openai",
    )

    assert ready.retrieval_status == "ready"
    assert (
        ready.next_action
        == "Review cited evidence or send the selected result to Console."
    )
    assert ready.use_in_console_action.enabled is True
    assert ready.selected_result == result

    searching = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="What did Gate 1.6 add?",
        retrieval_status="searching",
        provider_name="openai",
    )

    assert searching.retrieval_status == "searching"
    assert searching.next_action == "Wait for retrieval results."
    # C2: the run action itself carries the in-flight state.
    assert searching.query_state.run_action.label == "Searching…"
    assert searching.query_state.run_action.enabled is False


def test_panel_state_searching_status_overrides_run_action_only_when_reached() -> None:
    """C2: "searching" only overrides an otherwise-open run gate -- a query
    that's ALSO blocked (e.g. no source scope) keeps its real blocked label,
    since the gate ladder never reaches the searching branch for it.
    """
    searching_ready = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Find policy evidence",
        retrieval_status="searching",
        provider_name="openai",
    )
    assert searching_ready.retrieval_status == "searching"
    assert searching_ready.query_state.run_action.label == "Searching…"
    assert searching_ready.query_state.run_action.enabled is False
    assert searching_ready.query_state.run_action.widget_id == "library-rag-run-query"

    searching_blocked = LibraryRagPanelState.from_values(
        source_counts={"notes": 0},
        query="Find policy evidence",
        retrieval_status="searching",
    )
    assert searching_blocked.retrieval_status == "blocked"
    assert searching_blocked.query_state.run_action.label == "Run"
    assert searching_blocked.query_state.run_action.enabled is False


def test_panel_state_answering_status_overrides_run_action_only_when_reached() -> None:
    """PR-3 Task 3: the new "answering" retrieval status (set while the RAG
    Answer worker's provider call is in flight) mirrors "searching" -- see
    `test_panel_state_searching_status_overrides_run_action_only_when_reached`
    above -- one more explicit-status branch in the same normalizer, not a
    forked copy of it. It overrides an otherwise-open run gate with a
    disabled, distinctly-labeled run action, but a query that's ALSO
    blocked (e.g. no source scope) keeps its real blocked label, since the
    gate ladder never reaches the answering branch for it.
    """
    answering_ready = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Find policy evidence",
        mode="rag",
        retrieval_status="answering",
        provider_name="openai",
    )
    assert answering_ready.retrieval_status == "answering"
    assert answering_ready.query_state.run_action.label == "Answering…"
    assert answering_ready.query_state.run_action.enabled is False
    assert answering_ready.query_state.run_action.widget_id == "library-rag-run-query"

    answering_blocked = LibraryRagPanelState.from_values(
        source_counts={"notes": 0},
        query="Find policy evidence",
        mode="rag",
        retrieval_status="answering",
    )
    assert answering_blocked.retrieval_status == "blocked"
    assert answering_blocked.query_state.run_action.label == "Run"
    assert answering_blocked.query_state.run_action.enabled is False


def test_panel_state_answering_keeps_selected_evidence_usable_in_console() -> None:
    """PR-3 Task 4 review: generation must not disable "Use in Console" for
    already-selected evidence. Retrieval has settled and its bundle is
    frozen by the time answering starts, so the answer cannot change what is
    stageable -- and the disabled copy ("Run a query and select usable
    evidence before sending to Console.") would be a plain falsehood in a
    state where a query HAS run and evidence IS selected. Only the run
    action is in-flight-disabled here.
    """
    result = LibraryRagResultRow.from_result(
        {
            "title": "Incident Review",
            "snippet": "Expired credential caused the incident.",
            "score": 0.93,
            "source_id": "note-42",
            "chunk_id": "chunk-7",
        }
    )
    answering = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Why did the incident happen?",
        mode="rag",
        results=(result,),
        selected_result_id=result.result_id,
        retrieval_status="answering",
        provider_name="openai",
    )

    assert answering.retrieval_status == "answering"
    assert answering.selected_result == result
    assert answering.use_in_console_action.enabled is True
    assert answering.use_in_console_action.disabled_reason == ""
    # ...while the run gate itself stays honestly in-flight.
    assert answering.query_state.run_action.enabled is False
    assert answering.query_state.run_action.label == "Answering…"

    # With nothing selected, "answering" is no more usable than "ready" is.
    nothing_selected = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Why did the incident happen?",
        mode="rag",
        results=(result,),
        retrieval_status="answering",
        provider_name="openai",
    )
    assert nothing_selected.use_in_console_action.enabled is False


def test_panel_state_carries_in_flight_answer_provider_for_the_asking_line() -> None:
    """PR-3 Task 3: the panel state must carry the provider resolved for an
    answer call CURRENTLY IN FLIGHT through to `library_rag_answer_children`'s
    "Asking <provider>..." line -- distinct from `state.answer.provider`,
    which is only set once a call has SETTLED onto `state.answer`. Defaults
    to `""` (every call site that predates this field), so a caller that
    never threads it through keeps the prior generic "Generating answer..."
    line rather than a broken "Asking ..." with nothing named.
    """
    with_provider = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Find policy evidence",
        mode="rag",
        retrieval_status="answering",
        in_flight_answer_provider="anthropic",
    )
    assert with_provider.in_flight_answer_provider == "anthropic"

    default_state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Find policy evidence",
        mode="rag",
        retrieval_status="answering",
    )
    assert default_state.in_flight_answer_provider == ""


def test_panel_state_answer_field_round_trips_provider_model_usage() -> None:
    """`from_values` must pass Task 2's `LibraryRagAnswer.provider`/`model`/
    `usage` through to `state.answer` completely untouched -- the Task 3
    footer's whole provenance depends on these three surviving the state
    build unchanged."""
    from tldw_chatbook.Chat.provider_usage import ProviderUsage
    from tldw_chatbook.Library.library_rag_answer_service import (
        ANSWER_STATUS_READY,
        LibraryRagAnswer,
    )

    usage = ProviderUsage(
        uncached_input=1000, output=240, provider="anthropic", model="claude-sonnet-4-6"
    )
    answer = LibraryRagAnswer(
        status=ANSWER_STATUS_READY,
        text="Expired credential caused the incident.",
        citation_status="validated",
        provider="anthropic",
        model="claude-sonnet-4-6",
        usage=usage,
    )
    state = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Why did the incident happen?",
        mode="rag",
        answer=answer,
    )
    assert state.answer is answer
    assert state.answer.provider == "anthropic"
    assert state.answer.model == "claude-sonnet-4-6"
    assert state.answer.usage == usage


def test_panel_state_defaults_stable_selectors_for_recovery_paths() -> None:
    failed = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Find policy evidence",
        retrieval_status="failed",
        provider_name="openai",
    )

    assert failed.recovery_selector == LIBRARY_RAG_SERVICE_ERROR_SELECTOR
    assert "Library retrieval could not complete" in failed.recovery_copy

    empty = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Find policy evidence",
        retrieval_status="empty",
        provider_name="openai",
    )

    assert empty.recovery_selector == LIBRARY_RAG_EMPTY_STATE_SELECTOR
    assert "No evidence matched the current query." in empty.recovery_copy


def test_panel_state_computes_coverage_note_from_diagnostics() -> None:
    """(Task 8) `LibraryRagPanelState.from_values` threads `diagnostics`
    into `library_rag_coverage_note`, keyed off the actual normalized
    `results` it was given -- not a bare pass-through of the raw dict."""
    result = LibraryRagResultRow.from_result(
        {
            "title": "Media doc",
            "score": 0.6,
            "source_id": "media-1",
            "provenance": {"source_type": "media"},
        }
    )
    ready = LibraryRagPanelState.from_values(
        source_counts={"notes": 1, "media": 1},
        query="cake",
        mode="rag",
        results=(result,),
        diagnostics={
            "semantic_scope_coverage": {
                "covered": ["media"],
                "uncovered": ["notes"],
            }
        },
    )

    assert ready.coverage_note == "Semantic search found nothing from: Notes."

    # Default: no `diagnostics` kwarg at all -> no coverage claim (keyword
    # mode, and every rag-mode call site that predates Task 8's diagnostics
    # plumbing, never pass one).
    silent = LibraryRagPanelState.from_values(
        source_counts={"notes": 1, "media": 1},
        query="cake",
        mode="rag",
        results=(result,),
    )
    assert silent.coverage_note == ""


def test_explicit_empty_scope_selection_is_not_defaulted_to_all_sources() -> None:
    scope = LibraryRagScopeState.from_source_counts(notes=2, media=1, selected=())

    assert scope.has_available_sources is True
    assert scope.has_selected_sources is False
    assert scope.selected_source_types == ()
    assert all(not option.selected for option in scope.options)

    panel = LibraryRagPanelState.from_values(
        source_counts={"notes": 2, "media": 1},
        selected_source_types=(),
        query="Find policy evidence",
    )

    assert panel.scope.has_selected_sources is False
    assert panel.retrieval_status == "blocked"
    assert (
        panel.query_state.run_action.disabled_reason
        == "Select at least one Library source."
    )


# --- Task 6: prompts as a Search source -----------------------------------


def test_prompts_source_toggle_label_and_gate_with_four_sources() -> None:
    """`Prompts (N)` toggle label composes from `label`/`count`, and the
    "select at least one source" gate keeps working once a 4th real source
    (prompts) exists alongside notes/media/conversations.
    """
    scope = LibraryRagScopeState.from_source_counts(
        notes=1,
        media=1,
        conversations=1,
        prompts=5,
        selected=("notes", "media", "conversations", "prompts"),
    )

    prompts_option = scope.option_by_type("prompts")
    assert prompts_option.label == "Prompts"
    assert prompts_option.count == 5
    assert prompts_option.available is True
    assert prompts_option.selected is True

    panel = LibraryRagPanelState.from_values(
        source_counts={"notes": 1, "media": 1, "conversations": 1, "prompts": 5},
        selected_source_types=(),
        query="Find policy evidence",
    )

    assert panel.scope.has_selected_sources is False
    assert panel.retrieval_status == "blocked"
    assert (
        panel.query_state.run_action.disabled_reason
        == "Select at least one Library source."
    )

    ready = LibraryRagPanelState.from_values(
        source_counts={"notes": 0, "media": 0, "conversations": 0, "prompts": 5},
        selected_source_types=("prompts",),
        query="Find policy evidence",
        provider_name="openai",
    )

    assert ready.scope.selected_source_types == ("prompts",)
    assert ready.retrieval_status == "ready"


class TestUpdateSearchHistory:
    def test_prepends_new_query(self):
        assert update_search_history(("b",), "a") == ("a", "b")

    def test_exact_match_dedupes_to_front(self):
        assert update_search_history(("a", "b", "c"), "b") == ("b", "a", "c")

    def test_caps_at_ten_entries(self):
        history = tuple(f"q{i}" for i in range(10))
        result = update_search_history(history, "new")
        assert len(result) == 10
        assert result[0] == "new"
        assert "q9" not in result

    def test_truncates_entries_to_200_chars(self):
        result = update_search_history((), "x" * 500)
        assert result == ("x" * 200,)

    def test_blank_query_is_ignored(self):
        assert update_search_history(("a",), "   ") == ("a",)


class TestSearchingStatusLine:
    def test_lists_selected_sources(self):
        assert searching_status_line(("notes", "media")) == "searching · Notes, Media…"

    def test_empty_scope_still_reads_searching(self):
        assert searching_status_line(()) == "searching…"


class TestResultRowOpenTarget:
    def test_note_result_opens_notes(self):
        row = LibraryRagResultRow.from_result(
            {
                "source_id": "note-42",
                "title": "T",
                "snippet": "s",
                "provenance": {"source_type": "note"},
            }
        )
        assert row.open_source_type == "notes"
        assert row.can_open is True

    def test_media_and_conversation_map(self):
        media = LibraryRagResultRow.from_result(
            {
                "source_id": "7",
                "title": "T",
                "snippet": "s",
                "provenance": {"source_type": "media"},
            }
        )
        convo = LibraryRagResultRow.from_result(
            {
                "source_id": "c1",
                "title": "T",
                "snippet": "s",
                "provenance": {"source_type": "conversation"},
            }
        )
        assert media.open_source_type == "media"
        assert convo.open_source_type == "conversations"

    def test_prompt_result_opens_prompt_singular_not_plural(self):
        """Task 6: prompts' open-target is the singular "prompt" -- distinct
        from the "prompts" scope-toggle/source key -- because
        `_open_library_item_by_id`'s dispatch key is "prompt" (singular).
        """
        row = LibraryRagResultRow.from_result(
            {
                "source_id": "5",
                "title": "T",
                "snippet": "s",
                "provenance": {"source_type": "prompt"},
            }
        )
        assert row.open_source_type == "prompt"
        assert row.can_open is True

    def test_unknown_type_or_missing_id_cannot_open(self):
        no_type = LibraryRagResultRow.from_result(
            {"source_id": "x", "title": "T", "snippet": "s"}
        )
        no_id = LibraryRagResultRow.from_result(
            {"title": "T", "snippet": "s", "provenance": {"source_type": "note"}}
        )
        assert no_type.can_open is False
        assert no_id.can_open is False


class TestPanelStateHistory:
    def test_from_values_carries_history(self):
        state = LibraryRagPanelState.from_values(history=("a", "b"))
        assert state.history == ("a", "b")

    def test_history_defaults_empty(self):
        assert LibraryRagPanelState.from_values().history == ()

    def test_history_collapsed_defaults_false_and_passes_through(self):
        """D1: `history_collapsed` is a plain passthrough -- the caller (the
        screen) owns when it changes; the pure layer just carries it.
        """
        assert LibraryRagPanelState.from_values().history_collapsed is False
        assert (
            LibraryRagPanelState.from_values(history_collapsed=True).history_collapsed
            is True
        )


class TestLibraryRagScoreSuffix:
    """(RAG-34) Evidence rows render an honest match band instead of a raw
    three-decimal cosine score -- keyword-mode rows carry score=None by
    deliberate service design (FTS relevance was dropped as misleading) and
    must render as an empty string, not "unknown"."""

    def test_none_renders_empty_string(self):
        assert library_rag_score_suffix(None) == ""

    def test_strong_band_above_threshold(self):
        assert library_rag_score_suffix(0.93) == " | match: strong"

    def test_strong_band_inclusive_at_exact_boundary(self):
        """0.5 lands in "strong", not "moderate" -- the strong band is
        `>= 0.5`, so the boundary value itself must not silently drift into
        the band below it under a future refactor."""
        assert library_rag_score_suffix(0.5) == " | match: strong"

    def test_moderate_band_between_thresholds(self):
        assert library_rag_score_suffix(0.35) == " | match: moderate"

    def test_moderate_band_inclusive_at_exact_boundary(self):
        """0.2 lands in "moderate", not "weak" -- the moderate band is
        `>= 0.2`, pinned from the low side the same way strong is pinned
        from its own boundary above."""
        assert library_rag_score_suffix(0.2) == " | match: moderate"

    def test_moderate_band_just_below_strong_boundary(self):
        assert library_rag_score_suffix(0.499) == " | match: moderate"

    def test_weak_band_keeps_raw_two_decimal_number_for_transparency(self):
        """Weak is "the best of a bad lot" -- unlike strong/moderate, the
        raw number stays visible so a user can tell how weak."""
        assert library_rag_score_suffix(0.091) == " | match: weak (0.09)"

    def test_weak_band_just_below_moderate_boundary(self):
        assert library_rag_score_suffix(0.199) == " | match: weak (0.20)"

    def test_weak_band_at_zero(self):
        assert library_rag_score_suffix(0.0) == " | match: weak (0.00)"


class TestLibraryRagAllMatchesWeak:
    """(RAG-34/Task 8) `library_rag_all_matches_weak` feeds Task 8's
    coverage note -- it must be True only when there is at least one scored
    row and every scored row bands weak; unscored (keyword) rows are
    ignored entirely, and a result set with no scored rows at all is False,
    not True."""

    def test_no_rows_is_false(self):
        assert library_rag_all_matches_weak(()) is False

    def test_all_scored_rows_weak_is_true(self):
        rows = (
            LibraryRagResultRow.from_result({"title": "A", "score": 0.09}),
            LibraryRagResultRow.from_result({"title": "B", "score": 0.15}),
        )
        assert library_rag_all_matches_weak(rows) is True

    def test_one_strong_or_moderate_row_makes_it_false(self):
        rows = (
            LibraryRagResultRow.from_result({"title": "A", "score": 0.09}),
            LibraryRagResultRow.from_result({"title": "B", "score": 0.35}),
        )
        assert library_rag_all_matches_weak(rows) is False

    def test_unscored_rows_are_ignored_not_counted_toward_all(self):
        """A keyword-mode row (score=None) sitting alongside a weak scored
        row does not flip the verdict either way -- it is simply excluded
        from the "every scored row" check."""
        rows = (
            LibraryRagResultRow.from_result({"title": "A", "score": 0.09}),
            LibraryRagResultRow.from_result({"title": "B"}),
        )
        assert library_rag_all_matches_weak(rows) is True

    def test_only_unscored_rows_is_false(self):
        """No scored rows at all (pure keyword mode) is False -- "everything
        is weak" is a claim about actual scores, not about their absence."""
        rows = (
            LibraryRagResultRow.from_result({"title": "A"}),
            LibraryRagResultRow.from_result({"title": "B"}),
        )
        assert library_rag_all_matches_weak(rows) is False

    def test_weak_boundary_row_counts_as_weak(self):
        rows = (LibraryRagResultRow.from_result({"title": "A", "score": 0.2 - 1e-9}),)
        assert library_rag_all_matches_weak(rows) is True

    def test_moderate_boundary_row_is_not_weak(self):
        rows = (LibraryRagResultRow.from_result({"title": "A", "score": 0.2}),)
        assert library_rag_all_matches_weak(rows) is False


class TestLibraryRagScoreKindAwareBands:
    """(RAG-port P0/Task 6) The match band is a claim about *cosine
    similarity*. Hybrid retrieval replaces every row's score with an
    RRF-fused number whose theoretical maximum is `1/(rrf_k + 1)` -- ~0.016
    at the `rrf_k=60` these fixtures were written against, ~0.167 at the
    shipped k of 5 (TASK-4110). Both are below the 0.2 weak boundary, but
    the disqualifier is the KIND, not the magnitude: banding a fused score
    on cosine thresholds renders a wall of "match: weak (0.02)" on results
    that are in fact excellent.
    Reranker scores are worse still: an LLM 0-10 scale or a raw
    cross-encoder logit is not on the unit interval at all.

    The band must therefore be computed from the score kind's own honest
    similarity input -- the preserved vector leg for hybrid rows (Task 2's
    `hybrid_fusion.vector_score`) -- and must disclose the kind instead of
    inventing a similarity when there is none.
    """

    def test_fused_score_never_bands_on_cosine_thresholds(self):
        """A fused RRF score (~0.016) with a strong vector leg bands strong."""
        assert (
            library_rag_score_suffix(
                0.016, score_kind="hybrid_fusion", vector_score=0.83
            )
            == " | match: strong"
        )

    def test_fts_only_hybrid_row_reads_keyword_match(self):
        """No vector leg means no similarity exists -- say "keyword match",
        never a fabricated band and never the fused 0.0x number."""
        assert (
            library_rag_score_suffix(
                0.0161, score_kind="hybrid_fusion", vector_score=None
            )
            == " | keyword match"
        )

    def test_reranker_scores_disclose_kind_not_band(self):
        """Reranker scores are unbounded (logits, 0-10 LLM scales): the kind
        is disclosed, the number is never banded as a cosine."""
        assert library_rag_score_suffix(-3.2, score_kind="reranker") == " | reranked"

    def test_hybrid_row_with_weak_vector_leg_still_bands_weak_on_the_leg(self):
        """The converse pin: hybrid banding reads the VECTOR leg, not the
        fused score -- a genuinely weak vector leg must still render weak,
        with the leg's own number, not the fused one."""
        assert (
            library_rag_score_suffix(
                0.0159, score_kind="hybrid_fusion", vector_score=0.09
            )
            == " | match: weak (0.09)"
        )

    def test_default_kind_preserves_the_legacy_similarity_contract(self):
        """Every pre-existing call site passes a cosine similarity and no
        kind -- the default must keep banding exactly as before."""
        assert library_rag_score_suffix(0.93) == " | match: strong"
        assert library_rag_score_suffix(None) == ""
        assert (
            library_rag_score_suffix(0.93, score_kind="vector_similarity")
            == " | match: strong"
        )

    def test_reranked_row_discloses_kind_even_without_a_score(self):
        """`None` means "unscored" for a similarity row, but a reranked row
        was scored by definition -- the kind stays disclosed."""
        assert library_rag_score_suffix(None, score_kind="reranker") == " | reranked"


class TestLibraryRagAllMatchesWeakScoreKinds:
    """(RAG-port P0/Task 6) The all-weak coverage note is a claim about
    semantic similarity ("No strong semantic matches"), so only rows whose
    effective banding input IS a similarity may participate. Keyword-only
    hybrid rows and reranked rows neither trigger it nor suppress it."""

    def test_all_matches_weak_ignores_non_similarity_kinds(self):
        keyword_only = LibraryRagResultRow.from_result(
            {
                "title": "Keyword only",
                "score": 0.0161,
                "provenance": {
                    "hybrid_fusion": {"fts_rank": 1, "vector_rank": None,
                                      "fts_score": 0.001, "vector_score": None},
                },
            }
        )
        weak_vector = LibraryRagResultRow.from_result({"title": "Weak", "score": 0.09})
        assert library_rag_all_matches_weak((keyword_only, weak_vector)) is True
        assert library_rag_all_matches_weak((keyword_only,)) is False

    def test_hybrid_rows_are_judged_on_their_vector_leg(self):
        """A fused 0.016 must not read as a weak similarity -- the row's
        strong vector leg makes the whole set non-weak."""
        strong_hybrid = LibraryRagResultRow.from_result(
            {
                "title": "Strong hybrid",
                "score": 0.0161,
                "provenance": {
                    "hybrid_fusion": {"fts_rank": 1, "vector_rank": 1,
                                      "fts_score": 0.001, "vector_score": 0.83},
                },
            }
        )
        assert library_rag_all_matches_weak((strong_hybrid,)) is False

    def test_reranked_rows_never_participate(self):
        reranked = LibraryRagResultRow.from_result(
            {
                "title": "Reranked",
                "score": 0.0,
                "provenance": {"_final_score_kind": "reranker"},
            }
        )
        # Alone: no similarity row exists at all, so no all-weak claim.
        assert library_rag_all_matches_weak((reranked,)) is False
        # Alongside a weak similarity row: does not suppress the claim.
        weak_vector = LibraryRagResultRow.from_result({"title": "Weak", "score": 0.09})
        assert library_rag_all_matches_weak((reranked, weak_vector)) is True
        # Alongside a strong similarity row: does not create one either.
        strong_vector = LibraryRagResultRow.from_result(
            {"title": "Strong", "score": 0.83}
        )
        assert library_rag_all_matches_weak((reranked, strong_vector)) is False

    def test_duck_typed_rows_without_the_new_fields_still_work(self):
        """`mcp_inspector._ScoredRow` is a `__slots__ = ("score",)` shim that
        feeds this same canonical check -- reading the new fields must be
        `getattr`-tolerant or the MCP Test Tool interpretation line breaks."""

        class _ScoreOnly:
            __slots__ = ("score",)

            def __init__(self, score):
                self.score = score

        assert library_rag_all_matches_weak((_ScoreOnly(0.09),)) is True
        assert library_rag_all_matches_weak((_ScoreOnly(0.83),)) is False


class TestLibraryRagResultRowScoreKind:
    """(RAG-port P0/Task 6) The row is where the service's score provenance
    becomes display state: `provenance["hybrid_fusion"]` (Task 2 preserves
    the per-leg scores there) and the `_final_score_kind` reranker channel
    are normalized once, on the row, so the band and the Console evidence
    bundle cannot disagree."""

    def test_plain_semantic_row_defaults_to_vector_similarity(self):
        row = LibraryRagResultRow.from_result({"title": "A", "score": 0.83})
        assert row.score_kind == "vector_similarity"
        assert row.vector_score is None

    def test_hybrid_row_carries_the_preserved_vector_leg(self):
        row = LibraryRagResultRow.from_result(
            {
                "title": "A",
                "score": 0.0161,
                "provenance": {
                    "hybrid_fusion": {"fts_rank": 1, "vector_rank": 1,
                                      "fts_score": 0.001, "vector_score": 0.83},
                },
            }
        )
        assert row.score_kind == "hybrid_fusion"
        assert row.vector_score == pytest.approx(0.83)
        assert row.score == pytest.approx(0.0161)

    def test_fts_only_hybrid_row_has_no_vector_leg(self):
        row = LibraryRagResultRow.from_result(
            {
                "title": "A",
                "score": 0.0161,
                "provenance": {
                    "hybrid_fusion": {"fts_rank": 1, "vector_rank": None,
                                      "fts_score": 0.001, "vector_score": None},
                },
            }
        )
        assert row.score_kind == "hybrid_fusion"
        assert row.vector_score is None

    def test_reranker_channel_is_read_from_provenance(self):
        row = LibraryRagResultRow.from_result(
            {"title": "A", "score": 7.5, "provenance": {"_final_score_kind": "reranker"}}
        )
        assert row.score_kind == "reranker"

    def test_the_rerank_score_stamp_is_the_production_marker(self):
        """`_final_score_kind` is READ by `local_citation_capture` but written
        by nothing in the app; what a real reranked row actually carries is
        `metadata["rerank_score"]`, stamped by
        `PointwiseReranker._apply_scores` as it replaces the score. Keying
        on that is what makes the reranked band reachable in production."""
        row = LibraryRagResultRow.from_result(
            {"title": "A", "score": 7.5, "provenance": {"rerank_score": 7.5}}
        )
        assert row.score_kind == "reranker"
        assert (
            library_rag_score_suffix(
                row.score, score_kind=row.score_kind, vector_score=row.vector_score
            )
            == " | reranked"
        )

    def test_a_reranked_score_inside_the_band_range_never_reads_as_strong(self):
        """The load-bearing case. `RerankingConfig.score_scale` defaults to
        (0.0, 1.0), so a default-configured pointwise reranker emits scores
        INSIDE the similarity band range -- 0.83 would have rendered
        "match: strong", a cosine claim about an LLM relevance score."""
        row = LibraryRagResultRow.from_result(
            {"title": "A", "score": 0.83, "provenance": {"rerank_score": 0.95}}
        )
        suffix = library_rag_score_suffix(
            row.score, score_kind=row.score_kind, vector_score=row.vector_score
        )
        assert suffix == " | reranked"
        assert "match:" not in suffix

    def test_reranking_wins_over_a_prior_fusion_block(self):
        """Reranking runs AFTER fusion, so a hybrid row that was then
        reranked carries both blocks -- the later stage owns what the final
        score means, and the row must not band on the stale vector leg."""
        row = LibraryRagResultRow.from_result(
            {
                "title": "A",
                "score": 0.95,
                "provenance": {
                    "rerank_score": 0.95,
                    "hybrid_fusion": {"fts_rank": 1, "vector_rank": 1,
                                      "fts_score": 0.001, "vector_score": 0.83},
                },
            }
        )
        assert row.score_kind == "reranker"
        assert row.vector_score is None
        assert (
            library_rag_score_suffix(
                row.score, score_kind=row.score_kind, vector_score=row.vector_score
            )
            == " | reranked"
        )

    def test_reranking_skipped_tag_is_not_a_score_kind_signal(self):
        """Task 4's `reranking_skipped`/`reranking_degraded` tags disclose
        that reranking FAILED -- the scores on those rows are the base
        retrieval scores, so treating the tag as a reranker score kind would
        hide a real similarity behind " | reranked"."""
        row = LibraryRagResultRow.from_result(
            {
                "title": "A",
                "score": 0.83,
                "provenance": {"reranking_skipped": "no credentials"},
            }
        )
        assert row.score_kind == "vector_similarity"

    def test_kind_also_resolves_from_engine_metadata(self):
        """Not every producer folds engine metadata into `provenance` --
        a top-level `metadata` block carrying the fusion provenance must
        resolve identically."""
        row = LibraryRagResultRow.from_result(
            {
                "title": "A",
                "score": 0.0161,
                "metadata": {
                    "hybrid_fusion": {"fts_rank": 1, "vector_rank": 1,
                                      "fts_score": 0.001, "vector_score": 0.51},
                },
            }
        )
        assert row.score_kind == "hybrid_fusion"
        assert row.vector_score == pytest.approx(0.51)


class TestLibraryRagCoverageNote:
    """(Task 8) `library_rag_coverage_note` builds the Evidence region's
    one-line semantic-coverage note from `_search_semantic`'s
    `semantic_scope_coverage` diagnostic plus Task 7's all-weak predicate.

    Live UAT (RAG-29): a "cake" query in rag mode returned unrelated media
    fixtures and no conversation, with nothing distinguishing "your notes
    have nothing relevant" from "semantic search never looked at your
    notes" -- this is the honesty note that closes that gap.
    """

    @staticmethod
    def _row(score: float | None = 0.6) -> LibraryRagResultRow:
        return LibraryRagResultRow.from_result({"title": "A", "score": score})

    def test_empty_when_diagnostics_carries_no_coverage_key(self):
        """Keyword mode's diagnostics only ever carry the scope-exclusion
        slot (conversations/prompts excluded) or are entirely empty -- no
        coverage claim is made either way."""
        rows = (self._row(score=None),)
        assert library_rag_coverage_note({}, rows) == ""
        assert (
            library_rag_coverage_note(
                {"scope": [{"status": "excluded", "reason": "conversations"}]}, rows
            )
            == ""
        )

    def test_none_diagnostics_is_treated_as_empty(self):
        assert library_rag_coverage_note(None, (self._row(),)) == ""

    def test_empty_when_everything_covered_and_not_weak(self):
        rows = (self._row(0.6),)
        diagnostics = {
            "semantic_scope_coverage": {"covered": ["notes", "media"], "uncovered": []}
        }
        assert library_rag_coverage_note(diagnostics, rows) == ""

    def test_uncovered_types_render_the_found_nothing_from_sentence(self):
        rows = (self._row(0.6),)
        diagnostics = {
            "semantic_scope_coverage": {
                "covered": ["media"],
                "uncovered": ["notes", "conversations"],
            }
        }
        assert (
            library_rag_coverage_note(diagnostics, rows)
            == "Semantic search found nothing from: Notes, Conversations."
        )

    def test_uncovered_types_route_through_the_display_label_table(self):
        """(controller amendment to Task 8) The Sources toggles two lines
        above render capitalized display labels (e.g. "✓ Notes") from
        `LIBRARY_RAG_SOURCE_TYPES` -- this note must speak the same
        vocabulary, not the raw lowercase source-type identifiers the
        service's diagnostics payload actually carries."""
        rows = (self._row(0.6),)
        diagnostics = {
            "semantic_scope_coverage": {
                "covered": [],
                "uncovered": ["media", "prompts"],
            }
        }
        assert (
            library_rag_coverage_note(diagnostics, rows)
            == "Semantic search found nothing from: Media, Prompts."
        )

    def test_unknown_uncovered_type_falls_back_to_the_raw_identifier(self):
        """Diagnostics are service-supplied, not a closed enum this module
        controls -- an unrecognized source type still renders (verbatim)
        rather than disappearing or raising."""
        rows = (self._row(0.6),)
        diagnostics = {
            "semantic_scope_coverage": {"covered": [], "uncovered": ["mystery_source"]}
        }
        assert (
            library_rag_coverage_note(diagnostics, rows)
            == "Semantic search found nothing from: mystery_source."
        )

    def test_all_weak_with_everything_covered_renders_only_the_weak_prefix(self):
        rows = (self._row(0.09),)
        diagnostics = {"semantic_scope_coverage": {"covered": ["notes"], "uncovered": []}}
        assert (
            library_rag_coverage_note(diagnostics, rows)
            == "No strong semantic matches — results below are weak."
        )

    def test_all_weak_and_uncovered_combine_weak_prefix_then_sentence(self):
        rows = (self._row(0.09),)
        diagnostics = {"semantic_scope_coverage": {"covered": [], "uncovered": ["notes"]}}
        assert library_rag_coverage_note(diagnostics, rows) == (
            "No strong semantic matches — results below are weak. "
            "Semantic search found nothing from: Notes."
        )

    # --- TASK-14752: keyword-sourced evidence is not "found nothing" -------

    def test_keyword_only_types_render_their_own_sentence(self):
        """(TASK-14752 AC#1/#2) A type whose rows came from the FTS leg has
        evidence ON SCREEN; the bare "Semantic search found nothing from:
        Notes." sentence, while literally true of the semantic leg, reads as
        "Notes produced nothing" to a user looking at note rows."""
        rows = (self._row(0.6),)
        diagnostics = {
            "semantic_scope_coverage": {
                "covered": ["media"],
                "uncovered": [],
                "keyword_only": ["notes"],
            }
        }
        assert (
            library_rag_coverage_note(diagnostics, rows)
            == "Keyword matches only from: Notes."
        )

    def test_keyword_only_and_absent_types_are_two_separate_sentences(self):
        """The mixed case is the whole point: one type matched on keywords
        alone, another produced nothing at all, and collapsing them into one
        list is what made the old sentence ambiguous."""
        rows = (self._row(0.6),)
        diagnostics = {
            "semantic_scope_coverage": {
                "covered": [],
                "uncovered": ["conversations"],
                "keyword_only": ["notes", "media"],
            }
        }
        assert library_rag_coverage_note(diagnostics, rows) == (
            "Semantic search found nothing from: Conversations. "
            "Keyword matches only from: Notes, Media."
        )

    def test_keyword_only_labels_route_through_the_display_label_table(self):
        """Same vocabulary rule as the uncovered sentence -- and the same
        escaping, since these labels are service-supplied and reach a
        `Static`."""
        rows = (self._row(0.6),)
        diagnostics = {
            "semantic_scope_coverage": {
                "covered": [],
                "uncovered": [],
                "keyword_only": ["media", "mystery_source"],
            }
        }
        assert (
            library_rag_coverage_note(diagnostics, rows)
            == "Keyword matches only from: Media, mystery_source."
        )

    def test_absent_keyword_only_key_leaves_the_note_exactly_as_before(self):
        """(TASK-14752 AC#3) The semantic and plain profiles never produce
        this key; their copy must be byte-identical to what it was."""
        rows = (self._row(0.6),)
        diagnostics = {
            "semantic_scope_coverage": {"covered": ["media"], "uncovered": ["notes"]}
        }
        assert (
            library_rag_coverage_note(diagnostics, rows)
            == "Semantic search found nothing from: Notes."
        )

    def test_keyword_only_claims_stay_suppressed_at_zero_rows(self):
        """A "Keyword matches only from: X" sentence is a claim about rows on
        screen; with no rows it would be self-contradicting, so it obeys the
        same zero-row suppression the uncovered sentence does."""
        diagnostics = {
            "semantic_scope_coverage": {
                "covered": [],
                "uncovered": ["media"],
                "keyword_only": ["notes"],
            }
        }
        assert library_rag_coverage_note(diagnostics, ()) == ""

    def test_empty_rows_never_render_a_coverage_note(self):
        """Edge case (c): zero results overall is the no-match state's
        territory (Task 11), not a coverage note listing every requested
        source -- even a diagnostics payload claiming every requested type
        uncovered must not render anything when there are no rows at all."""
        diagnostics = {
            "semantic_scope_coverage": {"covered": [], "uncovered": ["notes", "media"]}
        }
        assert library_rag_coverage_note(diagnostics, ()) == ""

    # (RAG-port P0, Workstream A) The service now also reports how the
    # retrieval was ROUTED when it could not run the active profile's
    # configured mode -- a hybrid profile diverted to semantic because no
    # selected source has a keyword leg, a plain profile routed to the
    # keyword seams. Those disclosures share this one quiet line rather than
    # opening a second note channel on the same screen. (The scope divert
    # that used to head this list retired with TASK-15020/B1: a scoped
    # hybrid search now runs hybrid, so there is nothing to disclose.)

    def test_route_note_renders_as_a_sentence_when_nothing_else_to_say(self):
        rows = (self._row(0.6),)
        diagnostics = {
            LIBRARY_RAG_ROUTE_NOTES_KEY: [
                "no keyword leg for the selected sources — semantic only"
            ]
        }
        assert (
            library_rag_coverage_note(diagnostics, rows)
            == "No keyword leg for the selected sources — semantic only."
        )

    def test_route_note_renders_without_any_coverage_diagnostic(self):
        """The plain-profile route returns the KEYWORD payload, which carries
        no `semantic_scope_coverage` at all -- the disclosure must still
        reach the line (this is the only thing telling the user their rag
        query ran as keyword search)."""
        rows = (self._row(score=None),)
        diagnostics = {
            LIBRARY_RAG_ROUTE_NOTES_KEY: [
                "Profile 'BM25 Only': keyword search (no vectors)"
            ]
        }
        assert (
            library_rag_coverage_note(diagnostics, rows)
            == "Profile 'BM25 Only': keyword search (no vectors)."
        )

    def test_route_notes_follow_the_weak_prefix_and_coverage_sentence(self):
        rows = (self._row(0.09),)
        diagnostics = {
            "semantic_scope_coverage": {"covered": [], "uncovered": ["notes"]},
            LIBRARY_RAG_ROUTE_NOTES_KEY: [
                "no keyword leg for the selected sources — semantic only"
            ],
        }
        assert library_rag_coverage_note(diagnostics, rows) == (
            "No strong semantic matches — results below are weak. "
            "Semantic search found nothing from: Notes. "
            "No keyword leg for the selected sources — semantic only."
        )

    def test_route_notes_survive_the_zero_row_outcome(self):
        """(RAG-port P0 review, I2) The empty-rows guard exists so a
        no-match search does not enumerate every requested source as
        "uncovered" -- a claim about coverage. A ROUTING disclosure is a
        different fact ("vectors were never consulted"), and zero rows is
        exactly when it is most diagnostic: a plain-profile query that
        matched nothing must still say the profile ran keyword-only, or the
        user reads the empty result as "the vector index has nothing"."""
        assert library_rag_coverage_note(
            {
                LIBRARY_RAG_ROUTE_NOTES_KEY: [
                    "Profile 'BM25 Only': keyword search (no vectors)"
                ]
            },
            (),
        ) == "Profile 'BM25 Only': keyword search (no vectors)."

    def test_zero_row_coverage_claims_stay_suppressed_alongside_a_route_note(self):
        """Only the routing fact survives zero rows -- the "found nothing
        from: …" coverage sentence stays suppressed exactly as before."""
        diagnostics = {
            "semantic_scope_coverage": {"covered": [], "uncovered": ["notes", "media"]},
            LIBRARY_RAG_ROUTE_NOTES_KEY: [
                "no keyword leg for the selected sources — semantic only"
            ],
        }
        assert library_rag_coverage_note(diagnostics, ()) == (
            "No keyword leg for the selected sources — semantic only."
        )

    def test_blank_route_notes_render_nothing(self):
        rows = (self._row(0.6),)
        diagnostics = {"semantic_scope_coverage": {"covered": ["notes"], "uncovered": []}}
        assert library_rag_coverage_note(diagnostics, rows) == ""
        assert (
            library_rag_coverage_note(
                {**diagnostics, LIBRARY_RAG_ROUTE_NOTES_KEY: ["", "   "]}, rows
            )
            == ""
        )


class TestLibraryRagResultsCountLine:
    """(task-2859 item 10) `library_rag_results_count_line` builds the
    Evidence region's "N results for 'query'" headline -- previously
    missing entirely, so the row cards had no line naming how many landed
    or what query produced them."""

    @staticmethod
    def _row(title: str = "A") -> LibraryRagResultRow:
        return LibraryRagResultRow.from_result({"title": title})

    def test_empty_results_render_no_line(self):
        assert library_rag_results_count_line((), "cats") == ""

    def test_singular_noun_for_exactly_one_result(self):
        rows = (self._row(),)
        assert library_rag_results_count_line(rows, "cats") == "1 result for 'cats'."

    def test_plural_noun_for_multiple_results(self):
        rows = (self._row("A"), self._row("B"), self._row("C"))
        assert (
            library_rag_results_count_line(rows, "cats")
            == "3 results for 'cats'."
        )

    def test_query_is_markup_escaped(self):
        rows = (self._row(),)
        line = library_rag_results_count_line(rows, "[bold]cats[/bold]")
        # Rich's escape_markup only needs to escape the opening bracket.
        assert "\\[bold]cats\\[/bold]" in line

    def test_long_query_is_clamped(self):
        rows = (self._row(),)
        long_query = "x" * 500
        line = library_rag_results_count_line(rows, long_query)
        # Same clamp budget `library_rag_empty_state_quiet_copy` uses.
        assert len(line) < len(long_query)


class TestLibraryRagEmptyStateQuietCopy:
    """(RAG-33/Task 11) `library_rag_empty_state_quiet_copy` builds the
    Evidence region's quiet two-line no-match copy, replacing the retired
    Unavailable/Why/Next/Recovery/Owner dump for the routine "your library
    has nothing matching this query" case."""

    def test_two_line_copy_quotes_the_query_with_no_dump_language(self) -> None:
        scope = LibraryRagScopeState.from_source_counts(notes=1, media=1)
        copy = library_rag_empty_state_quiet_copy("unicorn migration guide", scope)
        assert copy == (
            "No evidence matched 'unicorn migration guide'.\nTry broader terms."
        )
        for jargon in ("Owner:", "Unavailable:", "Why:", "Next:", "Recovery:", "No results"):
            assert jargon not in copy

    def test_escapes_rich_markup_in_the_query(self) -> None:
        """Precedent: the history-row builder escapes stored queries before
        handing them to a `Static`/`Button` (a raw `[bold]x[/]` would either
        inject markup or raise `MarkupError`); this is the other place the
        panel renders raw query text, so it must do the same. A single
        bracket pair like "[bold]x" round-trips as a substring of its own
        escaped form (the closing `]` is never escaped) -- a tag with an
        explicit close, as the provenance-label precedent test uses, is
        the form that actually proves escaping happened."""
        scope = LibraryRagScopeState.from_source_counts(notes=1)
        copy = library_rag_empty_state_quiet_copy("[bold]spoof[/]", scope)
        assert "[bold]spoof[/]" not in copy
        assert r"\[bold]spoof\[/]" in copy

    def test_second_line_offers_to_turn_on_sources_still_switched_off(self) -> None:
        scope = LibraryRagScopeState.from_source_counts(
            notes=1, media=1, selected=("notes",)
        )
        copy = library_rag_empty_state_quiet_copy("cake", scope)
        assert copy.endswith("Try broader terms or turn on more sources.")

    def test_second_line_stays_constant_when_no_more_sources_exist_to_enable(
        self,
    ) -> None:
        """Every available source is already selected -- offering to "turn
        on more sources" would be a false claim, so the line drops that
        clause instead of showing it regardless of whether it's true."""
        scope = LibraryRagScopeState.from_source_counts(notes=1, media=1)
        copy = library_rag_empty_state_quiet_copy("cake", scope)
        assert copy.endswith("Try broader terms.")
        assert "turn on" not in copy

    def test_clamps_a_very_long_query_at_a_word_boundary(self) -> None:
        scope = LibraryRagScopeState.from_source_counts(notes=1)
        long_query = "word " * 60
        copy = library_rag_empty_state_quiet_copy(long_query, scope)
        first_line = copy.splitlines()[0]
        assert first_line.startswith("No evidence matched 'word")
        assert first_line.endswith("…'.")
        assert len(first_line) < len(long_query)


# --- I2 (review round 2): canonicalization-map parity guard -----------------
#
# Three source-type canonicalization maps overlap almost entirely and were
# hand-synced by comments only, with no test tying them together:
#   - `_OPEN_SOURCE_TYPE_MAP` (library_rag_state.py) -- Library-canvas
#     "Open" dispatch keys.
#   - `_SCOPE_SOURCE_TYPE_MAP` (library_rag_state.py) -- D4/task-5's
#     already-landed-row scope filter (`LibraryRagPanelState.from_values`).
#   - `_SEMANTIC_SOURCE_TYPE_MAP` (library_local_rag_search_service.py) --
#     the retrieval-time analogue of the same filter, applied to rag mode's
#     semantic leg before rows land.
#
# A shared-base-dict refactor was considered and rejected in favor of this
# test: `_SEMANTIC_SOURCE_TYPE_MAP` lives in a different module, already
# imports FROM `library_rag_state` (no circularity blocker), but deriving
# it from a shared base would touch a heavily-commented, already-reviewed
# production file (`_SEMANTICALLY_COVERABLE_SOURCE_TYPES` is directly
# derived from `_SEMANTIC_SOURCE_TYPE_MAP.values()`) purely to add a drift
# guard -- a test achieves the identical goal (a future edit that breaks
# the documented relationship fails loudly, here) with zero production
# risk and zero behavior change. Smaller change, same protection.


def test_scope_source_type_map_is_a_superset_of_the_semantic_source_type_map() -> None:
    """`_SCOPE_SOURCE_TYPE_MAP` must agree with `_SEMANTIC_SOURCE_TYPE_MAP`
    on every key the semantic map defines -- the semantic map is
    deliberately NARROWER (it omits prompts/workspaces/collections, which
    have no semantic-index seam at all, per its own module comment), never
    disagreeing on a shared key."""
    scope_map = _rag_state_module._SCOPE_SOURCE_TYPE_MAP
    semantic_map = _semantic_module._SEMANTIC_SOURCE_TYPE_MAP

    for raw_source_type, canonical in semantic_map.items():
        assert raw_source_type in scope_map, (
            f"_SCOPE_SOURCE_TYPE_MAP is missing {raw_source_type!r}, which "
            "_SEMANTIC_SOURCE_TYPE_MAP defines."
        )
        assert scope_map[raw_source_type] == canonical, (
            f"_SCOPE_SOURCE_TYPE_MAP[{raw_source_type!r}] == "
            f"{scope_map[raw_source_type]!r} but "
            f"_SEMANTIC_SOURCE_TYPE_MAP[{raw_source_type!r}] == {canonical!r}."
        )


def test_scope_source_type_map_matches_open_source_type_map_except_prompt() -> None:
    """`_SCOPE_SOURCE_TYPE_MAP` and `_OPEN_SOURCE_TYPE_MAP` agree on every
    shared key EXCEPT "prompt": `_OPEN_SOURCE_TYPE_MAP` deliberately keeps
    it singular (`_open_library_item_by_id`'s dispatch key), while
    `_SCOPE_SOURCE_TYPE_MAP` canonicalizes it to the plural "prompts"
    scope-toggle key -- the one documented, intentional divergence. This
    asserts that ACTUAL delta rather than forcing false identity between
    the two maps."""
    scope_map = _rag_state_module._SCOPE_SOURCE_TYPE_MAP
    open_map = _rag_state_module._OPEN_SOURCE_TYPE_MAP

    shared_keys = set(open_map) & set(scope_map)
    assert "prompt" in shared_keys, (
        "Expected both maps to define 'prompt' -- the documented divergence "
        "this test pins no longer applies if either map dropped it."
    )
    for raw_source_type in shared_keys - {"prompt"}:
        assert open_map[raw_source_type] == scope_map[raw_source_type], (
            "_OPEN_SOURCE_TYPE_MAP and _SCOPE_SOURCE_TYPE_MAP disagree on "
            f"{raw_source_type!r} outside the documented 'prompt' divergence."
        )
    assert open_map["prompt"] == "prompt"  # _open_library_item_by_id's dispatch key
    assert scope_map["prompt"] == "prompts"  # the plural scope-toggle key


# --- PR-T2 review round 3, finding I1 -------------------------------------

_ANTHROPIC_CREDENTIAL_REMEDY = (
    "Set ANTHROPIC_API_KEY or add api_key under [api_settings.anthropic]."
)


def test_unselected_provider_keeps_the_select_a_provider_copy() -> None:
    """Half one of I1: the genuinely-unselected case is UNCHANGED.

    No provider named and no credential remedy to offer -- "select a
    provider/model", owner "LLM provider", recovery pointer "Console
    controls" is the right copy here and must survive the fix that gave
    the other half its own.
    """
    state = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="rag",
        provider_name=None,
        provider_credential_recovery="",
    )

    assert state.run_action.enabled is False
    assert state.run_action.disabled_reason == (
        "Select a provider/model before asking for a RAG answer."
    )
    assert "Owner: LLM provider." in state.recovery_copy
    assert "Recovery: Console controls." in state.recovery_copy


def test_named_but_uncredentialed_provider_shows_the_real_remedy() -> None:
    """Half two of I1: the case Task 7 newly routed into this branch.

    Task 7 widened the blocked branch to include "endpoint named,
    credential missing" -- which is now the ONLY way a user with a
    configured provider reaches it -- while Task 4's collapse to a single
    `provider_name` destroyed the reason at the call site. The result told
    a user to select the provider they had already selected and pointed at
    Console controls instead of at a credential. The remedy that
    `ProviderReadiness` already carried must reach the tooltip (`disabled_
    reason`), the Why line and the Recovery pointer, and the owner must
    name the credential rather than the provider.
    """
    state = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="rag",
        provider_name=None,
        provider_credential_recovery=_ANTHROPIC_CREDENTIAL_REMEDY,
    )

    assert state.run_action.enabled is False
    assert "Select a provider/model" not in state.run_action.disabled_reason
    assert "ANTHROPIC_API_KEY" in state.run_action.disabled_reason
    assert "api_settings.anthropic" in state.run_action.disabled_reason
    assert "Console controls" not in state.recovery_copy
    assert "Owner: LLM provider credential." in state.recovery_copy
    assert "ANTHROPIC_API_KEY" in state.recovery_copy


def test_credential_remedy_is_markup_escaped_for_its_rendering_sinks() -> None:
    """The remedy embeds a TOML table name in brackets, and both sinks --
    the run button's tooltip and the blocked callout / recovery `Static`s
    -- render Rich markup, which would swallow `[api_settings.anthropic]`
    and leave a sentence pointing at nothing.
    """
    state = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="rag",
        provider_name=None,
        provider_credential_recovery=_ANTHROPIC_CREDENTIAL_REMEDY,
    )

    assert r"\[api_settings.anthropic]" in state.run_action.disabled_reason


def test_credential_remedy_cannot_make_a_blocked_state_look_ready() -> None:
    """The new field is a MESSAGE, never a second readiness input (the
    invariant PR-T2 Task 4's review established): readiness stays derived
    from `provider_name` alone, so a remedy passed alongside a name is
    simply never read, and a remedy passed without one never unblocks.
    """
    blocked = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="rag",
        provider_name=None,
        provider_credential_recovery=_ANTHROPIC_CREDENTIAL_REMEDY,
    )
    assert blocked.status == "blocked"
    assert blocked.ready_answer_provider == ""

    ready = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="rag",
        provider_name="anthropic",
        provider_credential_recovery=_ANTHROPIC_CREDENTIAL_REMEDY,
    )
    assert ready.status == "ready"
    assert ready.run_action.enabled is True
    assert ready.ready_answer_provider == "anthropic"
    assert "ANTHROPIC_API_KEY" not in ready.run_action.disabled_reason


def test_panel_state_threads_the_credential_remedy_into_query_state() -> None:
    """The screen builds the panel state, not the query state directly --
    so the remedy has to survive that layer too, or the fix never reaches
    a rendered surface.
    """
    panel = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="summarize the policy",
        mode="rag",
        provider_name=None,
        provider_credential_recovery=_ANTHROPIC_CREDENTIAL_REMEDY,
    )

    assert "ANTHROPIC_API_KEY" in panel.query_state.run_action.disabled_reason


# --------------------------------------------------------------------------
# B3 (TASK-15020): the Search/RAG window's DEPTH follows the active profile.
#
# Before B3 the window's evidence depth was the literal
# `LIBRARY_RAG_DEFAULT_TOP_K = 5` -- a number nothing in Settings could move,
# while the Console's own Library RAG entry points had already been taught to
# read the active RAG profile's `search.default_top_k` (TASK-406/TASK-3170,
# `_console_library_rag_profile_top_k`). Two surfaces over the same retrieval
# stack disagreed about how deep "a search" goes, and only one of them could
# be configured. B3 makes the DEFAULT profile-resolved on both; an explicit
# caller-supplied count still wins, unchanged.
# --------------------------------------------------------------------------


class _CountingResolver:
    """A stand-in for `resolve_active_rag_top_k` that records its calls."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        return self.value


def _patch_profile_depth(monkeypatch, value: int) -> _CountingResolver:
    """Patch what `library_rag_profile_top_k` actually reads, not the seam.

    Mirrors the Console suite's discipline: patching the resolver the seam
    calls (imported lazily inside it, so the module attribute is what gets
    read at call time) exercises the real seam -- including its try/except
    and its own non-positive guard -- instead of mocking it away.
    """
    from tldw_chatbook.RAG_Search.simplified import active_config

    resolver = _CountingResolver(value)
    monkeypatch.setattr(active_config, "resolve_active_rag_top_k", resolver)
    return resolver


def test_query_state_depth_defaults_to_the_active_profile_top_k(monkeypatch) -> None:
    """Unset depth resolves to the profile, not to the literal 5."""
    _patch_profile_depth(monkeypatch, 15)

    state = LibraryRagQueryState.from_values(
        query="summarize the policy", provider_name="openai"
    )

    assert state.top_k == 15
    assert state.status == "ready"


def test_panel_state_depth_defaults_to_the_active_profile_top_k(monkeypatch) -> None:
    """The screen builds the PANEL state, so the default has to survive that
    layer too -- and the Evidence heading is where a user reads it.
    """
    from tldw_chatbook.Widgets.Library.library_search_rag_panel import (
        results_heading_text,
    )

    _patch_profile_depth(monkeypatch, 15)

    panel = LibraryRagPanelState.from_values(
        source_counts={"notes": 2},
        query="summarize the policy",
        mode="rag",
        provider_name="openai",
    )

    assert panel.query_state.top_k == 15
    assert results_heading_text(panel) == "Evidence · top 15"


def test_query_state_depth_keeps_an_explicit_caller_value(monkeypatch) -> None:
    """B3 changes the DEFAULT only. An explicit in-range count still wins --
    and the profile is not even consulted, so "the user wins" holds by
    construction rather than by the two numbers happening to match.
    """
    resolver = _patch_profile_depth(monkeypatch, 15)

    state = LibraryRagQueryState.from_values(
        query="summarize the policy", top_k=7, provider_name="openai"
    )

    assert state.top_k == 7
    assert resolver.calls == 0


@pytest.mark.parametrize("bad_value", ["bad", "", None, 0, -3, 51, 500])
def test_query_state_depth_falls_back_to_the_profile_for_invalid_values(
    monkeypatch, bad_value
) -> None:
    """Out-of-range/unparseable counts resolve to the profile, not to 5."""
    _patch_profile_depth(monkeypatch, 15)

    state = LibraryRagQueryState.from_values(
        query="summarize the policy", top_k=bad_value, provider_name="openai"
    )

    assert state.top_k == 15


def test_query_state_depth_falls_back_to_five_when_the_profile_is_unresolvable(
    monkeypatch,
) -> None:
    """A broken/absent profile must degrade to searching, never to raising
    inside a render -- the same contract the Console seam carries.
    """
    from tldw_chatbook.RAG_Search.simplified import active_config

    def _raise_profile_unavailable() -> int:
        raise RuntimeError("simulated: active RAG profile unresolvable")

    monkeypatch.setattr(
        active_config, "resolve_active_rag_top_k", _raise_profile_unavailable
    )

    state = LibraryRagQueryState.from_values(
        query="summarize the policy", provider_name="openai"
    )

    assert state.top_k == LIBRARY_RAG_FALLBACK_TOP_K == 5


@pytest.mark.parametrize("profile_value", [0, -1])
def test_profile_depth_seam_rejects_a_non_positive_profile_value(
    monkeypatch, profile_value
) -> None:
    """A profile that resolves to a useless count is treated as unresolvable."""
    _patch_profile_depth(monkeypatch, profile_value)

    assert library_rag_profile_top_k() == LIBRARY_RAG_FALLBACK_TOP_K


def test_query_state_clamps_a_profile_deeper_than_the_window_bound(
    monkeypatch,
) -> None:
    """Settings accepts a profile depth up to 100; this window's own bound is
    `LIBRARY_RAG_TOP_K_MAX` (50). Clamp rather than discard: a 100-deep
    profile means "as deep as you can", and falling back to 5 -- the pre-B3
    coercion's answer for an out-of-range value -- would invert it.
    """
    _patch_profile_depth(monkeypatch, 100)

    state = LibraryRagQueryState.from_values(
        query="summarize the policy", provider_name="openai"
    )

    assert state.top_k == LIBRARY_RAG_TOP_K_MAX == 50


def test_console_and_library_share_one_profile_depth_seam(monkeypatch) -> None:
    """The coupling pin: the Console chip and this window must never drift.

    `chat_screen._console_library_rag_profile_top_k` is a thin delegation to
    `library_rag_profile_top_k` (one definition, three call sites), so this
    asserts they agree on BOTH branches -- resolved and unresolvable -- and
    that the two fallback constants are the same number.
    """
    from tldw_chatbook.RAG_Search.simplified import active_config
    from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module

    _patch_profile_depth(monkeypatch, 23)
    assert (
        chat_screen_module._console_library_rag_profile_top_k()
        == library_rag_profile_top_k()
        == 23
    )

    def _raise_profile_unavailable() -> int:
        raise RuntimeError("simulated: active RAG profile unresolvable")

    monkeypatch.setattr(
        active_config, "resolve_active_rag_top_k", _raise_profile_unavailable
    )
    assert (
        chat_screen_module._console_library_rag_profile_top_k()
        == library_rag_profile_top_k()
        == LIBRARY_RAG_FALLBACK_TOP_K
    )
    assert (
        chat_screen_module.CONSOLE_LIBRARY_RAG_FALLBACK_TOP_K
        == LIBRARY_RAG_FALLBACK_TOP_K
    )


def test_clamp_divergence_is_pinned_as_a_pair(monkeypatch) -> None:
    """The window clamps a >50 profile; the Console seam deliberately does NOT.

    Task-8 review, minor 1: the clamp was only half-pinned. The window arm
    (profile 100 -> 50) had a test, but nothing exercised the UNCAPPED
    direction, so silently capping the SHARED seam -- `min(value,
    LIBRARY_RAG_TOP_K_MAX)` inside `library_rag_profile_top_k` -- left 199
    tests green while erasing a divergence that had been declared
    deliberate. A declared difference that no test can tell from its own
    removal is not a decision, it is a comment.

    Both arms live in ONE test on purpose: they are a pair, and the whole
    claim is that the same profile reads differently on the two surfaces.
    `LIBRARY_RAG_TOP_K_MAX` is this window's bound (the evidence list's own
    limit); the Console chip's request has no such list and honors whatever
    depth the user configured, up to Settings' own 100 ceiling.
    """
    from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module

    _patch_profile_depth(monkeypatch, 100)

    # Uncapped arm: the shared seam and its Console delegation report the
    # profile's real depth.
    assert library_rag_profile_top_k() == 100
    assert chat_screen_module._console_library_rag_profile_top_k() == 100

    # Capped arm: the Library window's own display state trims to its bound.
    window = LibraryRagQueryState.from_values(
        query="summarize the policy", provider_name="openai"
    )
    assert window.top_k == LIBRARY_RAG_TOP_K_MAX == 50


class TestLibraryRagRerankingNotice:
    """(TASK-3502 note-(a)) The reranker's `reranking_skipped` /
    `reranking_degraded` disclosure tags had ZERO UI consumers: a Hybrid
    Full user whose reranking credential was dead saw normal-looking,
    silently unreranked results. The tags ride the first result's
    provenance all the way to the panel (traced in
    `Tests/Library/test_library_local_rag_search_service.py`), so the
    Evidence region's existing one quiet note channel discloses them.
    """

    @staticmethod
    def _row(**provenance) -> LibraryRagResultRow:
        return LibraryRagResultRow.from_result(
            {"title": "A", "score": 0.6, "provenance": provenance}
        )

    def test_untagged_rows_say_nothing(self):
        rows = (self._row(source_type="note"), self._row(source_type="media"))
        assert library_rag_coverage_note({}, rows) == ""

    def test_skipped_tag_names_the_stage_and_the_detail(self):
        rows = (
            self._row(
                source_type="note",
                reranking_skipped="provider call failed (fake)",
            ),
        )
        assert library_rag_coverage_note({}, rows) == (
            "Reranking was skipped (provider call failed (fake)) "
            "— these results are in their original retrieval order."
        )

    def test_degraded_tag_names_the_stage_and_the_detail(self):
        rows = (self._row(source_type="note", reranking_degraded="3/5 scorings failed"),)
        assert library_rag_coverage_note({}, rows) == (
            "Reranking was degraded (3/5 scorings failed) — these results "
            "are in their original retrieval order."
        )

    def test_the_tag_is_found_wherever_the_tagged_row_landed(self):
        """The engine tags its FIRST result, but scope post-filtering and
        the panel's own count-intersected filter can move or drop rows --
        the disclosure keys off the tag being present at all, not off
        position 0."""
        rows = (self._row(source_type="media"), self._row(reranking_degraded="1/2"))
        assert "Reranking was degraded (1/2)" in library_rag_coverage_note({}, rows)

    def test_it_joins_the_existing_note_channel_rather_than_competing(self):
        weak = LibraryRagResultRow.from_result(
            {
                "title": "A",
                "score": 0.09,
                "provenance": {"reranking_degraded": "2/2 scorings failed"},
            }
        )
        note = library_rag_coverage_note(
            {"semantic_scope_coverage": {"covered": [], "uncovered": ["notes"]}},
            (weak,),
        )
        assert note == (
            "No strong semantic matches — results below are weak. "
            "Semantic search found nothing from: Notes. "
            "Reranking was degraded (2/2 scorings failed) — these results "
            "are in their original retrieval order."
        )

    def test_a_hostile_detail_string_is_escaped_and_clamped(self):
        """The detail is `str(exc)` from a provider call -- unsanitized text
        reaching a `Static`, exactly what every other service-supplied
        string in this module is escaped for."""
        rows = (self._row(reranking_skipped="[bold]boom[/] " + "x" * 400),)
        note = library_rag_coverage_note({}, rows)
        assert "\\[bold]" in note
        assert "…" in note
        assert len(note) < 300

    def test_skipped_wins_when_a_row_somehow_carries_both(self):
        """The service's two tag sites are mutually exclusive branches, but
        nothing enforces that here -- one sentence, deterministically."""
        rows = (self._row(reranking_skipped="dead credential", reranking_degraded="1/2"),)
        note = library_rag_coverage_note({}, rows)
        assert note.startswith("Reranking was skipped (dead credential)")
        assert "degraded" not in note

    def test_a_blank_detail_still_discloses_the_stage(self):
        rows = (self._row(reranking_skipped=""),)
        assert library_rag_coverage_note({}, rows) == (
            "Reranking was skipped — these results are in their original "
            "retrieval order."
        )
