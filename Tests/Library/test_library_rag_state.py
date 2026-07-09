"""Library-native Search/RAG display-state contracts."""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_rag_state import (
    LIBRARY_RAG_EMPTY_STATE_SELECTOR,
    LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
    LibraryRagPanelState,
    LibraryRagQueryState,
    LibraryRagResultRow,
    LibraryRagScopeState,
    searching_status_line,
    update_search_history,
)


def test_scope_state_exposes_library_source_scope_and_empty_recovery() -> None:
    scope = LibraryRagScopeState.from_source_counts(
        notes=2,
        media=1,
        conversations=0,
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
        "workspaces",
        "collections",
    )
    assert scope.option_by_type("notes").label == "Notes"
    assert scope.option_by_type("notes").count_label == "2 sources"
    assert scope.option_by_type("notes").selected is True
    assert scope.option_by_type("conversations").available is False
    assert "No conversations available" in scope.option_by_type("conversations").recovery

    empty_scope = LibraryRagScopeState.from_source_counts(
        notes=0,
        media=0,
        conversations=0,
        workspaces=0,
        collections=0,
    )

    assert empty_scope.has_available_sources is False
    assert empty_scope.status == "blocked"
    assert "Owner: Library source index." in empty_scope.recovery_copy
    assert "Next: Add or import Library sources before querying." in empty_scope.recovery_copy


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
    assert row.title == "&lt;b&gt;Release&lt;/b&gt;"
    assert "Line one\nLine two" in row.snippet
    assert "<script" not in row.snippet
    assert row.citation_labels == ("&lt;i&gt;Citation&lt;/i&gt;",)
    assert row.citations[0].url == ""


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
    assert "Owner: Library source index." in blocked.recovery_copy

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
    assert ready.next_action == "Review cited evidence or send the selected result to Console."
    assert ready.use_in_console_action.enabled is True
    assert ready.selected_result == result

    searching = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="What did Gate 1.6 add?",
        retrieval_status="searching",
    )

    assert searching.retrieval_status == "searching"
    assert searching.next_action == "Wait for retrieval results."


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
    assert panel.query_state.run_action.disabled_reason == "Select at least one Library source."


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
            {"source_id": "note-42", "title": "T", "snippet": "s",
             "provenance": {"source_type": "note"}}
        )
        assert row.open_source_type == "notes"
        assert row.can_open is True

    def test_media_and_conversation_map(self):
        media = LibraryRagResultRow.from_result(
            {"source_id": "7", "title": "T", "snippet": "s",
             "provenance": {"source_type": "media"}}
        )
        convo = LibraryRagResultRow.from_result(
            {"source_id": "c1", "title": "T", "snippet": "s",
             "provenance": {"source_type": "conversation"}}
        )
        assert media.open_source_type == "media"
        assert convo.open_source_type == "conversations"

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
