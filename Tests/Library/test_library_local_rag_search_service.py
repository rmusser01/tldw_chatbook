"""Tests for the production local FTS-backed Library Search/RAG service."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Library.library_local_rag_search_service import LibraryLocalRagSearchService
from tldw_chatbook.Library.library_rag_service import (
    LibraryRagSearchOutcome,
    LibraryRagSearchRequest,
    run_library_rag_search,
)


class FakeNotesScopeService:
    """Mirrors NotesScopeService.search_notes's exact keyword-only signature."""

    def __init__(self, rows=None, error: Exception | None = None):
        self.rows = rows if rows is not None else []
        self.error = error
        self.calls: list[dict] = []

    async def search_notes(
        self,
        *,
        scope,
        query,
        limit=10,
        offset=0,
        user_id=None,
        workspace_id=None,
        workspace_notes=None,
    ):
        self.calls.append({"scope": scope, "query": query, "limit": limit, "user_id": user_id})
        if self.error is not None:
            raise self.error
        return self.rows


class FakeMediaReadingScopeService:
    """Mirrors MediaReadingScopeService.search_media's exact keyword-only signature."""

    def __init__(self, items=None, error: Exception | None = None):
        self.items = items if items is not None else []
        self.error = error
        self.calls: list[dict] = []

    async def search_media(self, *, mode=None, query=None, limit=20, offset=0, **filters):
        self.calls.append({"mode": mode, "query": query, "limit": limit, "offset": offset})
        if self.error is not None:
            raise self.error
        return {"items": self.items, "total": len(self.items), "offset": offset, "limit": limit}


class FakeRagService:
    """Mirrors RAGService.search's signature (RAG_Search/simplified/rag_service.py:476)."""

    def __init__(self, results=None, error: Exception | None = None):
        self.results = results if results is not None else []
        self.error = error
        self.calls: list[dict] = []

    async def search(
        self,
        query,
        top_k=None,
        search_type="semantic",
        filter_metadata=None,
        include_citations=None,
        score_threshold=None,
    ):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "search_type": search_type,
                "include_citations": include_citations,
            }
        )
        if self.error is not None:
            raise self.error
        return self.results


@pytest.fixture
def conversations_db():
    """Real in-memory CharactersRAGDB seeded with a conversation matched by FTS."""
    db = CharactersRAGDB(":memory:", client_id="test-client")
    conv_id = db.add_conversation({"title": "Incident retro"})
    db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": "The outage was caused by an expired credential.",
            "timestamp": "2026-01-01T00:00:00Z",
        }
    )
    try:
        yield db, conv_id
    finally:
        db.close_connection()


# (a) search mode returns note+media+conversation rows with correct source_id/provenance.
@pytest.mark.asyncio
async def test_search_mode_returns_rows_from_all_three_sources(conversations_db):
    db, conv_id = conversations_db
    notes_service = FakeNotesScopeService(
        rows=[{"id": "note-1", "title": "Runbook", "content": "Rotate the credential."}]
    )
    media_service = FakeMediaReadingScopeService(
        items=[{"id": 7, "source_id": "7", "title": "Postmortem video", "media_type": "video"}]
    )
    app = SimpleNamespace(
        notes_scope_service=notes_service,
        media_reading_scope_service=media_service,
        chachanotes_db=db,
        notes_user_id="tester",
    )
    service = LibraryLocalRagSearchService(app)

    result = await service.search("credential", ("notes", "media", "conversations"), "search", top_k=5)

    assert result["runtime_backend"] == "local-fts"
    rows = result["results"]
    by_type = {row["provenance"]["source_type"]: row for row in rows}
    assert by_type["note"]["source_id"] == "note-1"
    assert by_type["note"]["title"] == "Runbook"
    assert by_type["media"]["source_id"] == "7"
    assert by_type["conversation"]["source_id"] == conv_id
    # The fixture's one conversation has exactly one message, and it's the
    # one that matched the FTS query -- so message_count == 1.
    assert by_type["conversation"]["snippet"] == "Matched conversation · 1 messages"
    assert notes_service.calls[0]["user_id"] == "tester"
    assert notes_service.calls[0]["scope"] == "local_note"


# (b) a missing media service quietly yields notes/conversations rows only.
@pytest.mark.asyncio
async def test_missing_media_seam_yields_notes_and_conversations_only(conversations_db):
    db, _conv_id = conversations_db
    notes_service = FakeNotesScopeService(
        rows=[{"id": "note-1", "title": "Runbook", "content": "Rotate the credential."}]
    )
    app = SimpleNamespace(
        notes_scope_service=notes_service,
        media_reading_scope_service=None,
        chachanotes_db=db,
    )
    service = LibraryLocalRagSearchService(app)

    result = await service.search("credential", ("notes", "media", "conversations"), "search", top_k=5)

    types = {row["provenance"]["source_type"] for row in result["results"]}
    assert types == {"note", "conversation"}


# (c) all seams missing -> LibraryRagSearchOutcome with status == "blocked".
@pytest.mark.asyncio
async def test_all_seams_missing_returns_blocked_outcome():
    app = SimpleNamespace()
    service = LibraryLocalRagSearchService(app)

    result = await service.search("credential", ("notes", "media", "conversations"), "search", top_k=5)

    assert isinstance(result, LibraryRagSearchOutcome)
    assert result.status == "blocked"
    assert result.recovery_state is not None
    assert result.recovery_state.stable_selector == "library-rag-service-error"


# (d) rag mode with _rag_service=None -> blocked outcome; next_action mentions switching to Search.
@pytest.mark.asyncio
async def test_rag_mode_without_rag_service_returns_blocked_outcome():
    app = SimpleNamespace(_rag_service=None)
    service = LibraryLocalRagSearchService(app)

    result = await service.search("credential", ("notes",), "rag", top_k=5)

    assert isinstance(result, LibraryRagSearchOutcome)
    assert result.status == "blocked"
    assert result.recovery_state is not None
    assert result.recovery_state.status_label == "RAG unavailable"
    assert "switch mode to Search" in result.recovery_state.next_action
    assert result.recovery_state.recovery_action == "Settings > RAG"


# (e) scope filtering: scope=("notes",) never touches the media fake.
@pytest.mark.asyncio
async def test_scope_filtering_skips_unselected_media_seam(conversations_db):
    db, _conv_id = conversations_db
    notes_service = FakeNotesScopeService(rows=[])
    media_service = FakeMediaReadingScopeService(items=[])
    app = SimpleNamespace(
        notes_scope_service=notes_service,
        media_reading_scope_service=media_service,
        chachanotes_db=db,
    )
    service = LibraryLocalRagSearchService(app)

    await service.search("credential", ("notes",), "search", top_k=5)

    assert media_service.calls == []
    assert notes_service.calls


# (f) end-to-end through run_library_rag_search: status == "ready", rows normalized.
@pytest.mark.asyncio
async def test_end_to_end_through_run_library_rag_search(conversations_db):
    db, _conv_id = conversations_db
    notes_service = FakeNotesScopeService(
        rows=[{"id": "note-1", "title": "Runbook", "content": "Rotate the credential."}]
    )
    app = SimpleNamespace(
        notes_scope_service=notes_service,
        media_reading_scope_service=None,
        chachanotes_db=db,
    )
    app.library_rag_search_service = LibraryLocalRagSearchService(app)

    request = LibraryRagSearchRequest(
        query="credential",
        source_types=("notes", "conversations"),
        mode="search",
        top_k=5,
    )
    outcome = await run_library_rag_search(app, request)

    assert outcome.status == "ready"
    assert outcome.results
    assert outcome.runtime_backend == "local-fts"
    for row in outcome.results:
        assert row.result_id
        assert row.provenance.get("source_type") in {"note", "conversation"}


# Erroring seams contribute zero rows without raising, and do not force "blocked"
# as long as another seam is available.
@pytest.mark.asyncio
async def test_erroring_notes_seam_contributes_zero_rows_without_raising(conversations_db):
    db, _conv_id = conversations_db
    notes_service = FakeNotesScopeService(error=RuntimeError("notes index unavailable"))
    app = SimpleNamespace(
        notes_scope_service=notes_service,
        media_reading_scope_service=None,
        chachanotes_db=db,
    )
    service = LibraryLocalRagSearchService(app)

    result = await service.search("credential", ("notes", "conversations"), "search", top_k=5)

    assert not isinstance(result, LibraryRagSearchOutcome)
    assert result["results"]
    assert all(row["provenance"]["source_type"] != "note" for row in result["results"])


# An erroring seam that is the ONLY queried seam is still "available" (attribute
# present), so the result is empty results, not a blocked outcome.
@pytest.mark.asyncio
async def test_erroring_only_seam_returns_empty_results_not_blocked():
    notes_service = FakeNotesScopeService(error=RuntimeError("notes index unavailable"))
    app = SimpleNamespace(notes_scope_service=notes_service)
    service = LibraryLocalRagSearchService(app)

    result = await service.search("credential", ("notes",), "search", top_k=5)

    assert not isinstance(result, LibraryRagSearchOutcome)
    assert result["results"] == []


# Unknown scope types are ignored quietly rather than raising or matching a seam.
@pytest.mark.asyncio
async def test_unknown_scope_type_is_ignored_quietly():
    app = SimpleNamespace()
    service = LibraryLocalRagSearchService(app)

    result = await service.search("credential", ("workspaces", "collections"), "search", top_k=5)

    assert isinstance(result, LibraryRagSearchOutcome)
    assert result.status == "blocked"


# rag mode success path: delegates to _rag_service.search and maps results.
@pytest.mark.asyncio
async def test_rag_mode_delegates_and_maps_results():
    rag_service = FakeRagService(
        results=[
            {
                "id": "chunk-1",
                "score": 0.87,
                "document": "Rotate the credential immediately.",
                "metadata": {"title": "Runbook", "source_id": "note-1", "source_type": "note"},
            }
        ]
    )
    app = SimpleNamespace(_rag_service=rag_service)
    service = LibraryLocalRagSearchService(app)

    result = await service.search("credential", ("notes",), "rag", top_k=3, include_citations=True)

    assert rag_service.calls[0]["search_type"] == "semantic"
    assert rag_service.calls[0]["top_k"] == 3
    assert rag_service.calls[0]["include_citations"] is True
    assert result["runtime_backend"] == "rag-semantic"
    row = result["results"][0]
    assert row["title"] == "Runbook"
    assert row["source_id"] == "note-1"
    assert row["snippet"] == "Rotate the credential immediately."
    assert row["provenance"]["source_type"] == "note"
