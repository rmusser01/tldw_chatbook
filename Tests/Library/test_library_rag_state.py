"""Pure Library Search/RAG display-state contract tests."""

from __future__ import annotations

from tldw_chatbook.Library.library_rag_state import (
    LibraryRagPanelState,
    LibraryRagQueryState,
    LibraryRagResultRow,
    LibraryRagScopeState,
)


def test_scope_state_exposes_source_counts_and_empty_recovery() -> None:
    state = LibraryRagScopeState.from_source_counts(
        notes=2,
        media=1,
        conversations=0,
        workspaces=0,
        collections=0,
    )

    assert state.summary_label == "Source Scope: All local sources"
    assert state.total_count == 3
    assert state.selectable_source_types == ("notes", "media")
    assert state.source_count_labels == (
        "Notes: 2",
        "Media: 1",
        "Conversations: 0",
        "Workspaces: 0",
        "Collections: 0",
    )
    assert not state.is_empty
    assert state.recovery_copy == ""

    empty_state = LibraryRagScopeState.from_source_counts()

    assert empty_state.is_empty
    assert empty_state.selectable_source_types == ()
    assert "No Library sources are available" in empty_state.recovery_copy
    assert "Owner: Library sources" in empty_state.recovery_copy
    assert "Next:" in empty_state.recovery_copy
    assert "Add notes, media, conversations, Workspaces, or Collections" in empty_state.next_action


def test_query_state_blocks_empty_queries_with_recovery_copy() -> None:
    state = LibraryRagQueryState.from_values(query="", mode="rag")

    assert state.query == ""
    assert state.mode == "rag"
    assert not state.can_run
    assert state.disabled_reason == "Enter a Search/RAG query before running retrieval."
    assert "Enter a question or search terms" in state.recovery_copy
    assert "Owner: Library Search/RAG" in state.recovery_copy
    assert "Next:" in state.recovery_copy

    ready_state = LibraryRagQueryState.from_values(query="summarize vector search", mode="search")

    assert ready_state.query == "summarize vector search"
    assert ready_state.mode == "search"
    assert ready_state.can_run
    assert ready_state.recovery_copy == ""

    unsafe_state = LibraryRagQueryState.from_values(
        query="<script>alert('x')</script>",
        mode="<script>",
    )

    assert unsafe_state.query == ""
    assert unsafe_state.mode == "rag"
    assert not unsafe_state.can_run


def test_result_row_preserves_citations_snippets_and_provenance() -> None:
    row = LibraryRagResultRow.from_result(
        {
            "document_title": "Vector Notes",
            "snippet": "FTS5 and embeddings can be combined.",
            "score": "0.87",
            "source_id": "note-42",
            "chunk_id": "chunk-7",
            "citations": [{"label": "Vector Notes p. 4"}, "Appendix A"],
            "metadata": {"backend": "local", "collection": "research"},
        }
    )

    assert row.title == "Vector Notes"
    assert row.snippet == "FTS5 and embeddings can be combined."
    assert row.score == 0.87
    assert row.source_id == "note-42"
    assert row.chunk_id == "chunk-7"
    assert row.citation_labels == ("Vector Notes p. 4", "Appendix A")
    assert row.provenance == {"backend": "local", "collection": "research"}
    assert row.source_authority_label == "Source: note-42"

    malformed = LibraryRagResultRow.from_result(
        {"title": "", "snippet": None, "score": "not-a-number"}
    )

    assert malformed.title == "Untitled result"
    assert malformed.snippet == ""
    assert malformed.score is None
    assert malformed.source_authority_label == "Source: unknown"

    unsafe = LibraryRagResultRow.from_result(
        {
            "title": "<script>alert('x')</script>",
            "snippet": "onclick=alert(1)",
            "source_id": "javascript:alert(1)",
        }
    )

    assert unsafe.title == "Untitled result"
    assert unsafe.snippet == ""
    assert unsafe.source_id == ""

    single_mapping_citation = LibraryRagResultRow.from_result(
        {"title": "Solo Citation", "citations": {"label": "Single mapping citation"}}
    )

    assert single_mapping_citation.citation_labels == ("Single mapping citation",)


def test_panel_state_marks_ready_searching_blocked_and_empty_states() -> None:
    scope = LibraryRagScopeState.from_source_counts(notes=1)
    query = LibraryRagQueryState.from_values(query="What is indexed?", mode="rag")
    result = LibraryRagResultRow.from_result(
        {"title": "Indexed Note", "snippet": "Index evidence.", "source_id": "note-1"}
    )

    ready = LibraryRagPanelState.from_values(scope=scope, query=query, results=[result])

    assert ready.status == "ready"
    assert ready.status_label == "Ready"
    assert ready.results == (result,)
    assert ready.run_action.enabled
    assert ready.console_action.enabled

    searching = LibraryRagPanelState.from_values(scope=scope, query=query, is_searching=True)

    assert searching.status == "searching"
    assert searching.status_label == "Searching"
    assert not searching.run_action.enabled
    assert searching.run_action.disabled_reason == "Search/RAG retrieval is already running."

    blocked = LibraryRagPanelState.from_values(scope=scope, query=LibraryRagQueryState.from_values())

    assert blocked.status == "blocked"
    assert blocked.status_label == "Blocked"
    assert not blocked.run_action.enabled
    assert "Enter a Search/RAG query" in blocked.next_action
    assert blocked.authority_owner == "Library Search/RAG"

    unavailable = LibraryRagPanelState.from_values(
        scope=scope,
        query=query,
        recovery_copy=(
            "Missing Library retrieval index. Next: Build or connect an index. "
            "Owner: embeddings."
        ),
    )

    assert unavailable.status == "blocked"
    assert "Missing Library retrieval index" in unavailable.recovery_copy
    assert unavailable.run_action.disabled_reason == unavailable.recovery_copy

    empty = LibraryRagPanelState.from_values(
        scope=LibraryRagScopeState.from_source_counts(),
        query=query,
    )

    assert empty.status == "empty"
    assert empty.status_label == "No sources"
    assert not empty.run_action.enabled
    assert "Add notes, media, conversations, Workspaces, or Collections" in empty.next_action
