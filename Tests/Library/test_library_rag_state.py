"""Library-native Search/RAG display-state contracts."""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_rag_state import (
    LIBRARY_RAG_EMPTY_STATE_SELECTOR,
    LIBRARY_RAG_NO_SOURCES_GATE_COPY,
    LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY,
    LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
    LIBRARY_RAG_SNIPPET_DISPLAY_MAX_CHARS,
    LibraryRagPanelState,
    LibraryRagQueryState,
    LibraryRagResultRow,
    LibraryRagScopeState,
    library_rag_all_matches_weak,
    library_rag_coverage_note,
    library_rag_empty_state_quiet_copy,
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


def test_query_state_blocks_empty_query_and_runtime_blockers() -> None:
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

    ready_query = LibraryRagQueryState.from_values(
        query="summarize the policy",
        mode="unknown",
        top_k="bad",
    )

    assert ready_query.mode == "rag"
    assert ready_query.top_k == 5
    assert ready_query.status == "ready"
    assert ready_query.run_action.enabled is True


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


def test_query_state_validates_and_sanitizes_external_values() -> None:
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
    assert unsafe_query.top_k == 5

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
    ([bold]/[red]) stays neutralized behind a backslash, and a payload that
    ARRIVES already HTML-entity-encoded (&lt;script&gt;...) round-trips
    through unescape+escape to a single, inert escape -- it never decodes
    back to a literal '<'/'>' and never double-escapes to '&amp;lt;'.

    Unchanged by the 2026-08-03 finding-1 fix (`_sanitize_display_text` no
    longer re-escapes with `html.escape` for display) -- this test must
    keep passing verbatim, because the dangerous-pattern scrubber now runs
    a second time AFTER unescaping (see the sequencing-gap test below) and
    still fully removes both the literal and the entity-encoded <script>
    blocks before the final markup-escape."""
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

    assert row.row_badge_label == "media"


def test_row_badge_label_includes_citations_only_when_present() -> None:
    row = LibraryRagResultRow.from_result(
        {
            "title": "Roadmap",
            "provenance": {"source_type": "media"},
            "citations": [{"label": "Roadmap p.1"}, {"label": "Roadmap p.2"}],
        }
    )

    assert row.row_badge_label == "media · 2 citations"


def test_row_badge_label_maps_blocked_eligibility_to_excluded_from_context() -> None:
    row = LibraryRagResultRow.from_result(
        {
            "title": "Roadmap",
            "provenance": {"source_type": "media", "active_context_eligible": False},
        }
    )

    assert row.row_badge_label == "media · excluded from context"


def test_row_badge_label_includes_non_default_workspace() -> None:
    row = LibraryRagResultRow.from_result(
        {
            "title": "Roadmap",
            "provenance": {"source_type": "media", "workspace_id": "workspace-a"},
        }
    )

    assert row.row_badge_label == "media · workspace-a"


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
        == "media · workspace-a · 1 citation · excluded from context"
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


def test_panel_state_defaults_stable_selectors_for_recovery_paths() -> None:
    failed = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Find policy evidence",
        retrieval_status="failed",
    )

    assert failed.recovery_selector == LIBRARY_RAG_SERVICE_ERROR_SELECTOR
    assert "Library retrieval could not complete" in failed.recovery_copy

    empty = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="Find policy evidence",
        retrieval_status="empty",
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
        assert searching_status_line(("notes", "media")) == "searching · notes, media…"

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

    def test_empty_rows_never_render_a_coverage_note(self):
        """Edge case (c): zero results overall is the no-match state's
        territory (Task 11), not a coverage note listing every requested
        source -- even a diagnostics payload claiming every requested type
        uncovered must not render anything when there are no rows at all."""
        diagnostics = {
            "semantic_scope_coverage": {"covered": [], "uncovered": ["notes", "media"]}
        }
        assert library_rag_coverage_note(diagnostics, ()) == ""


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
