"""Library-native Search/RAG display-state contracts."""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_rag_state import (
    LibraryRagPanelState,
    LibraryRagQueryState,
    LibraryRagResultRow,
    LibraryRagScopeState,
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
