"""Tests for the production local FTS-backed Library Search/RAG service."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Library.library_fts_query import build_fts_match_query, expand_keyword_term
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
    _prompt_row,
)
from tldw_chatbook.Library.library_rag_service import (
    LibraryRagSearchOutcome,
    LibraryRagSearchRequest,
    run_library_rag_search,
)
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptScopeService,
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
        fts_match_query=None,
    ):
        self.calls.append(
            {
                "scope": scope,
                "query": query,
                "limit": limit,
                "user_id": user_id,
                "fts_match_query": fts_match_query,
            }
        )
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


class FakePromptScopeService:
    """Mirrors PromptScopeService.search_prompts's exact keyword-only signature."""

    def __init__(self, rows=None, error: Exception | None = None):
        self.rows = rows if rows is not None else []
        self.error = error
        self.calls: list[dict] = []

    async def search_prompts(
        self,
        *,
        mode="local",
        query,
        limit=10,
        include_deleted=False,
        fts_match_query=None,
    ):
        self.calls.append(
            {
                "mode": mode,
                "query": query,
                "limit": limit,
                "fts_match_query": fts_match_query,
            }
        )
        if self.error is not None:
            raise self.error
        return self.rows


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


@pytest.fixture
def real_fts_app(tmp_path):
    """Real media and conversation FTS seams seeded with punctuation-rich text."""
    content = (
        "The report asks what caused the outage and says the foo-bar gateway recovered. "
        'The operator "said" hello now. Alpha appeared near the beginning, with many '
        "unrelated details in between, before omega closed the report."
    )
    conversations = CharactersRAGDB(":memory:", client_id="plain-text-fts")
    conversation_id = conversations.add_conversation({"title": "Punctuation incident"})
    conversations.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": content,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    )

    media_db = MediaDatabase(tmp_path / "library_plain_text_fts.db", client_id="plain-text-fts")
    media_id, _media_uuid, _message = media_db.add_media_with_keywords(
        title="Punctuation incident recording",
        media_type="document",
        content=content,
    )
    app = SimpleNamespace(
        media_reading_scope_service=MediaReadingScopeService(
            LocalMediaReadingService(media_db),
            None,
        ),
        chachanotes_db=conversations,
    )
    try:
        yield app, str(media_id), conversation_id
    finally:
        media_db.close_connection()
        conversations.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "what caused the outage?",
        "foo-bar",
        'operator "said',
        'operator "said" now',
        "alpha omega",
    ],
)
async def test_public_keyword_search_treats_fts_queries_as_plain_text(real_fts_app, query):
    app, media_id, conversation_id = real_fts_app

    result = await LibraryLocalRagSearchService(app).search(
        query,
        ("media", "conversations"),
        "search",
        top_k=5,
    )

    rows_by_type = {row["provenance"]["source_type"]: row for row in result["results"]}
    assert rows_by_type["media"]["source_id"] == media_id
    assert rows_by_type["conversation"]["source_id"] == conversation_id


@pytest.mark.asyncio
async def test_notes_keyword_search_keeps_the_plain_query_unchanged():
    notes_service = FakeNotesScopeService(rows=[])
    app = SimpleNamespace(notes_scope_service=notes_service)
    service = LibraryLocalRagSearchService(app)

    await service.search('operator "said', ("notes",), "search", top_k=5)

    call = notes_service.calls[0]
    assert call["query"] == 'operator "said'
    # (task-185) The seam also receives the pre-built widened MATCH string:
    # alphabetic terms become OR-groups; the quote-bearing term stays a
    # doubled-quote-escaped literal, never bare FTS5 syntax.
    assert call["fts_match_query"] == '("operator" OR "operators") AND """said"'


# --- Task 6: prompts as a Search source -----------------------------------


@pytest.mark.asyncio
async def test_prompts_keyword_search_sends_widened_match_query():
    prompt_service = FakePromptScopeService(rows=[])
    app = SimpleNamespace(prompt_scope_service=prompt_service)
    service = LibraryLocalRagSearchService(app)

    await service.search("feedback loop", ("prompts",), "search", top_k=5)

    call = prompt_service.calls[0]
    assert call["mode"] == "local"
    assert call["query"] == "feedback loop"
    assert call["fts_match_query"] == '("feedback" OR "feedbacks") AND ("loop" OR "loops")'


def test_prompt_row_uses_raw_local_id_not_composite_id():
    """Task 4 review trap: `search_prompts` normalizes each result via
    `normalize_prompt_record`, whose "id" is a composite "local:prompt:<n>"
    string -- `_prompt_row` must key off "local_id" (the raw int) instead,
    since `_open_library_item_by_id("prompt", ...)` expects the raw int.
    """
    row = _prompt_row(
        {
            "id": "local:prompt:5",
            "local_id": 5,
            "name": "Retro learnings",
            "user_prompt": "We keep creating feedback loops.",
        }
    )

    assert row["source_id"] == "5"
    assert row["title"] == "Retro learnings"
    assert row["provenance"]["source_type"] == "prompt"


@pytest.mark.asyncio
async def test_prompts_seam_missing_yields_unavailable_not_blocked():
    """A missing `prompt_scope_service` degrades like the other three seams
    (see `test_missing_media_seam_yields_notes_and_conversations_only`):
    when at least one OTHER seam is available the overall search stays
    "ready", just without any prompt rows.
    """
    notes_service = FakeNotesScopeService(rows=[])
    app = SimpleNamespace(notes_scope_service=notes_service, prompt_scope_service=None)
    service = LibraryLocalRagSearchService(app)

    result = await service.search("credential", ("notes", "prompts"), "search", top_k=5)

    assert not isinstance(result, LibraryRagSearchOutcome)
    assert all(row["provenance"]["source_type"] != "prompt" for row in result["results"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "<script>alert('unsafe')</script>",
        "control\x00character",
        "x" * 2_001,
    ],
)
async def test_public_search_rejects_unsafe_query_before_source_calls(query):
    notes_service = FakeNotesScopeService(rows=[])
    service = LibraryLocalRagSearchService(
        SimpleNamespace(notes_scope_service=notes_service)
    )

    with pytest.raises(ValueError, match="safe Library search query"):
        await service.search(query, ("notes",), "search", top_k=5)

    assert notes_service.calls == []


@pytest.mark.asyncio
async def test_public_search_rejects_unsafe_query_before_semantic_service_call():
    rag_service = FakeRagService()
    service = LibraryLocalRagSearchService(
        SimpleNamespace(_rag_service=rag_service)
    )

    with pytest.raises(ValueError, match="safe Library search query"):
        await service.search("<script>unsafe</script>", ("notes",), "rag", top_k=5)

    assert rag_service.calls == []


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
    # one that matched the FTS query -- so message_count == 1, and the
    # secondary line pluralizes correctly ("1 message", never "1 messages").
    assert by_type["conversation"]["snippet"] == "Matched conversation · 1 message"
    assert notes_service.calls[0]["user_id"] == "tester"
    assert notes_service.calls[0]["scope"] == "local_note"
    # C1: keyword-mode rows uniformly show no score. The conversation row is
    # the interesting case -- `search_conversations_by_content` populates a
    # `relevance_score` derived from FTS `best_rank`, which is a ranking
    # artifact, not a retrieval similarity score, so it must not surface
    # here even though the raw data has it available.
    assert by_type["note"]["score"] is None
    assert by_type["media"]["score"] is None
    assert by_type["conversation"]["score"] is None


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


# F3 (PR #590 review, Qodo): rag mode must post-filter semantic rows by the
# selected scope's canonical source type. Rows with an unknown/missing
# provenance source type are always kept (we cannot classify them).
@pytest.mark.asyncio
async def test_rag_mode_filters_semantic_rows_by_scope():
    rag_service = FakeRagService(
        results=[
            {
                "id": "note-chunk",
                "score": 0.9,
                "document": "Note evidence.",
                "metadata": {"title": "Note doc", "source_id": "note-1", "source_type": "note"},
            },
            {
                "id": "media-chunk",
                "score": 0.8,
                "document": "Media evidence.",
                "metadata": {"title": "Media doc", "source_id": "media-1", "source_type": "media_chunk"},
            },
            {
                "id": "conversation-chunk",
                "score": 0.7,
                "document": "Conversation evidence.",
                "metadata": {"title": "Chat doc", "source_id": "chat-1", "source_type": "chat"},
            },
            {
                "id": "unknown-chunk",
                "score": 0.6,
                "document": "Unattributed evidence.",
                "metadata": {"title": "Mystery doc", "source_id": "mystery-1"},
            },
        ]
    )
    app = SimpleNamespace(_rag_service=rag_service)
    service = LibraryLocalRagSearchService(app)

    result = await service.search("credential", ("notes",), "rag", top_k=5)

    source_ids = {row["source_id"] for row in result["results"]}
    # Only the note row (matches scope) and the unattributable row (unknown
    # provenance, always kept) survive; media and conversation are dropped.
    assert source_ids == {"note-1", "mystery-1"}


# Full scope selects every known source type, so nothing is dropped.
@pytest.mark.asyncio
async def test_rag_mode_full_scope_keeps_all_rows():
    rag_service = FakeRagService(
        results=[
            {
                "id": "note-chunk",
                "score": 0.9,
                "document": "Note evidence.",
                "metadata": {"title": "Note doc", "source_id": "note-1", "source_type": "note"},
            },
            {
                "id": "media-chunk",
                "score": 0.8,
                "document": "Media evidence.",
                "metadata": {"title": "Media doc", "source_id": "media-1", "source_type": "media_chunk"},
            },
            {
                "id": "conversation-chunk",
                "score": 0.7,
                "document": "Conversation evidence.",
                "metadata": {"title": "Chat doc", "source_id": "chat-1", "source_type": "chat"},
            },
        ]
    )
    app = SimpleNamespace(_rag_service=rag_service)
    service = LibraryLocalRagSearchService(app)

    result = await service.search(
        "credential", ("notes", "media", "conversations"), "rag", top_k=5
    )

    source_ids = {row["source_id"] for row in result["results"]}
    assert source_ids == {"note-1", "media-1", "chat-1"}


# Empty scope never reaches the service in practice (UI gate), but the
# service must guard defensively rather than dropping every row.
@pytest.mark.asyncio
async def test_rag_mode_empty_scope_does_not_filter():
    rag_service = FakeRagService(
        results=[
            {
                "id": "note-chunk",
                "score": 0.9,
                "document": "Note evidence.",
                "metadata": {"title": "Note doc", "source_id": "note-1", "source_type": "note"},
            },
        ]
    )
    app = SimpleNamespace(_rag_service=rag_service)
    service = LibraryLocalRagSearchService(app)

    result = await service.search("credential", (), "rag", top_k=5)

    assert [row["source_id"] for row in result["results"]] == ["note-1"]


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


# (task-185) The keyword-search conversation row's secondary line pluralizes
# its message count instead of the fixed "N messages" template.
def test_conversation_row_secondary_line_pluralizes_message_count():
    from tldw_chatbook.Library.library_local_rag_search_service import _conversation_row

    one = _conversation_row({"id": "c-1", "title": "Sync", "message_count": 1})
    many = _conversation_row({"id": "c-2", "title": "Sync", "message_count": 8})
    missing = _conversation_row({"id": "c-3", "title": "Sync"})
    malformed = _conversation_row({"id": "c-4", "title": "Sync", "message_count": "n/a"})

    assert one["snippet"] == "Matched conversation · 1 message"
    assert many["snippet"] == "Matched conversation · 8 messages"
    assert missing["snippet"] == "Matched conversation · 0 messages"
    assert malformed["snippet"] == "Matched conversation · 0 messages"


# --- task-185: keyword search plural/singular variant widening ------------
# FTS5 unicode61 has no stemming, so "feedback loop" never matched a note
# containing "feedback loops." (live UAT). `build_fts_match_query` widens each
# alphabetic term (length >= 3) into an OR-group of naive variants while
# keeping every user token a quoted FTS5 string literal.


def test_expand_keyword_term_singular_gains_plural():
    assert expand_keyword_term("loop") == ("loop", "loops")
    assert expand_keyword_term("feedback") == ("feedback", "feedbacks")


def test_expand_keyword_term_plural_loses_trailing_s():
    assert expand_keyword_term("loops") == ("loops", "loop")


def test_expand_keyword_term_ies_and_y_swap_both_directions():
    assert expand_keyword_term("stories") == ("stories", "story")
    assert "stories" in expand_keyword_term("story")


def test_expand_keyword_term_es_endings_both_directions():
    assert "box" in expand_keyword_term("boxes")
    assert "boxes" in expand_keyword_term("box")


def test_expand_keyword_term_short_numeric_and_nonalpha_terms_pass_through():
    assert expand_keyword_term("ab") == ("ab",)
    assert expand_keyword_term("42") == ("42",)
    assert expand_keyword_term("foo-bar") == ("foo-bar",)
    assert expand_keyword_term('say"s') == ('say"s',)


def test_build_fts_match_query_is_an_and_of_or_groups():
    assert (
        build_fts_match_query("feedback loop")
        == '("feedback" OR "feedbacks") AND ("loop" OR "loops")'
    )


def test_build_fts_match_query_quotes_short_and_nonalpha_terms_verbatim():
    assert build_fts_match_query("at foo-bar 42") == '"at" AND "foo-bar" AND "42"'


@pytest.mark.parametrize(
    "hostile",
    [
        'operator "said',
        "NEAR(a b)",
        "loop) OR (x",
        '" OR 1=1 --',
        "content:loop",
        "*wildcard ^caret",
        "a AND b OR c NOT d",
        "[bracket {brace",
    ],
)
def test_build_fts_match_query_output_is_safe_against_a_real_fts5_table(hostile):
    """FTS5-hostile input must never raise an FTS5 query-syntax error."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE probe USING fts5(content)")
        conn.execute("INSERT INTO probe(content) VALUES ('feedback loops everywhere')")
        match = build_fts_match_query(hostile)
        conn.execute("SELECT content FROM probe WHERE probe MATCH ?", (match,)).fetchall()
    finally:
        conn.close()


def test_build_fts_match_query_keeps_user_supplied_or_literal_not_an_operator():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE probe USING fts5(content)")
        conn.execute("INSERT INTO probe(content) VALUES ('feedback loops everywhere')")
        match = build_fts_match_query("feedback OR missingterm")
        rows = conn.execute(
            "SELECT content FROM probe WHERE probe MATCH ?", (match,)
        ).fetchall()
    finally:
        conn.close()
    # If "OR" leaked through as an operator this would match on "feedback"
    # alone; as a quoted literal the query demands the token "or" and the
    # absent "missingterm", so nothing matches.
    assert rows == []


@pytest.fixture
def real_notes_app(tmp_path):
    """Real notes seam (NotesScopeService -> NotesInteropService -> FTS DB).

    File-backed on purpose: NotesInteropService re-opens the DB by path per
    user, so an in-memory DB would hand it a blank schema.
    """
    notes_db = CharactersRAGDB(tmp_path / "library_notes_fts.db", client_id="notes-fts")
    interop = NotesInteropService(
        base_db_directory=tmp_path / "notes_user_dbs",
        api_client_id="library-notes-fts-test",
        global_db_to_use=notes_db,
    )
    plural_id = interop.add_note(
        "library-user", "Retro learnings", "We keep creating feedback loops."
    )
    singular_id = interop.add_note(
        "library-user", "Draft idea", "Sketch one feedback loop for onboarding."
    )
    app = SimpleNamespace(
        notes_scope_service=NotesScopeService(
            local_notes_service=interop,
            server_service=None,
        ),
        notes_user_id="library-user",
    )
    try:
        yield app, plural_id, singular_id
    finally:
        for db in interop._db_instances.values():
            db.close_connection()
        notes_db.close_connection()


# End-to-end UAT reproduction: a note containing "feedback loops." must be hit
# by the query "feedback loop" (and the singular note by the plural query).
@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["feedback loop", "feedback loops"])
async def test_notes_keyword_search_matches_plural_and_singular_variants(real_notes_app, query):
    app, plural_id, singular_id = real_notes_app

    result = await LibraryLocalRagSearchService(app).search(query, ("notes",), "search", top_k=5)

    note_ids = {row["source_id"] for row in result["results"]}
    assert {plural_id, singular_id} <= note_ids


# The media and conversations seams get the same widening: a plural query hits
# singular content ("reports" -> "report") and vice versa ("unrelated detail"
# -> "unrelated details") in the shared punctuation-rich fixture text.
@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["reports", "unrelated detail"])
async def test_media_and_conversation_keyword_search_match_plural_singular_variants(
    real_fts_app, query
):
    app, media_id, conversation_id = real_fts_app

    result = await LibraryLocalRagSearchService(app).search(
        query,
        ("media", "conversations"),
        "search",
        top_k=5,
    )

    rows_by_type = {row["provenance"]["source_type"]: row for row in result["results"]}
    assert rows_by_type["media"]["source_id"] == media_id
    assert rows_by_type["conversation"]["source_id"] == conversation_id


@pytest.fixture
def real_prompts_app(tmp_path):
    """Real prompts seam (PromptScopeService -> LocalPromptService -> Prompts FTS DB)."""
    prompts_db = PromptsDatabase(tmp_path / "library_prompts_fts.db", client_id="prompts-fts")
    prompt_id, _uuid, _message = prompts_db.add_prompt(
        name="Retro learnings",
        author="tester",
        details="",
        system_prompt="",
        user_prompt="We keep creating feedback loops.",
        keywords=None,
    )
    app = SimpleNamespace(
        prompt_scope_service=PromptScopeService(
            local_service=LocalPromptService(prompts_db),
            server_service=None,
        ),
    )
    try:
        yield app, prompt_id
    finally:
        prompts_db.close_connection()


# End-to-end UAT reproduction (task-185 pattern applied to prompts): a prompt
# whose user prompt contains "feedback loops." must be hit by the query
# "feedback loop" via the plural/singular expansion builder.
@pytest.mark.asyncio
async def test_prompts_keyword_search_matches_plural_variant(real_prompts_app):
    app, prompt_id = real_prompts_app

    result = await LibraryLocalRagSearchService(app).search(
        "feedback loop", ("prompts",), "search", top_k=5
    )

    rows = result["results"]
    assert len(rows) == 1
    assert rows[0]["provenance"]["source_type"] == "prompt"
    assert rows[0]["source_id"] == str(prompt_id)


# Deselecting the prompts source yields zero prompt rows even though the
# same query would otherwise match (mirrors (e), the media scope-filtering
# test above).
@pytest.mark.asyncio
async def test_prompts_deselected_yields_zero_prompt_rows(real_prompts_app):
    app, _prompt_id = real_prompts_app
    app.notes_scope_service = FakeNotesScopeService(rows=[])

    result = await LibraryLocalRagSearchService(app).search(
        "feedback loop", ("notes",), "search", top_k=5
    )

    assert not isinstance(result, LibraryRagSearchOutcome)
    assert all(row["provenance"]["source_type"] != "prompt" for row in result["results"])
