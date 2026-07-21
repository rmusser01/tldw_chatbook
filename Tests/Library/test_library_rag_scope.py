"""Tests for task-6: caller-passed RAG retrieval scope in
``LibraryLocalRagSearchService`` (Console "Run Library RAG").

Covers Backend B of the ``rag-scope-narrowing`` program: the Library
service accepts an optional, caller-resolved ``EffectiveScope`` and never
resolves one itself (spec decision D2 -- the Library screen's own Search
canvas call sites never pass this keyword, so they stay byte-identical);
scoped keyword search restricts the media/notes seams to the scope's id
allowlists and excludes the conversations seam entirely (spec D5), scoped
semantic search runs one store query per allowlisted source type merged by
score; an EMPTY effective scope must never reach the service at all; and
the Console call site (``UI/Screens/chat_screen.py``) resolves the active
conversation's effective scope the same way the task-5 chat entry point
does before calling the service.

Mirrors ``Tests/Library/test_library_local_rag_search_service.py``'s
fixture conventions (real file-backed ``MediaDatabase``/``CharactersRAGDB``
for the seams that are actually modified here, lightweight spies for
call-shape assertions) and ``Tests/RAG/test_scope_pipeline_enforcement.py``'s
real-``TldwCli``/real-``ChatScreen`` recipe for the Console call site.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tldw_chatbook
from tldw_chatbook.Chat.rag_scope import (
    EffectiveScope,
    RagScope,
    SCOPE_REASON_CONVERSATIONS_EXCLUDED,
    SCOPE_REASON_PROMPTS_EXCLUDED,
    SCOPE_STATUS_EXCLUDED,
    ScopeItem,
    SOURCE_TYPE_MEDIA,
    SOURCE_TYPE_NOTE,
    write_conversation_scope,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)
from tldw_chatbook.Library.library_rag_service import (
    LibraryRagSearchOutcome,
    LibraryRagSearchRequest,
)
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.RAG_Search.pipeline_functions_simple import SCOPE_DIAGNOSTICS_KEY
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

from Tests.UI.test_screen_navigation import _build_test_app

_SCOPED_CONTENT = "The quarterly roadmap review covered migration timelines."


# --- Spy doubles (mirrors test_library_local_rag_search_service.py's Fakes,
# extended to capture the new id_allowlist kwarg) ---------------------------


class _SpyNotesScopeService:
    """Mirrors NotesScopeService.search_notes's keyword-only signature."""

    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []
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
        id_allowlist=None,
    ):
        self.calls.append({"query": query, "id_allowlist": id_allowlist})
        if id_allowlist is not None:
            allowed = {str(i) for i in id_allowlist}
            return [row for row in self.rows if str(row["id"]) in allowed]
        return self.rows


class _SpyMediaReadingScopeService:
    """Mirrors MediaReadingScopeService.search_media's keyword-only signature."""

    def __init__(self, items=None):
        self.items = items if items is not None else []
        self.calls: list[dict] = []

    async def search_media(
        self, *, mode=None, query=None, limit=20, offset=0, id_allowlist=None, **filters
    ):
        self.calls.append({"mode": mode, "id_allowlist": id_allowlist})
        items = self.items
        if id_allowlist is not None:
            allowed = {str(i) for i in id_allowlist}
            items = [item for item in items if str(item.get("id")) in allowed]
        return {"items": items, "total": len(items), "offset": offset, "limit": limit}


class _SpyConversationsDB:
    """Mirrors the subset of CharactersRAGDB the conversations seam reads."""

    is_memory_db = True

    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []
        self.call_count = 0

    def search_conversations_by_content(self, query, limit):
        self.call_count += 1
        return self.rows


class _SpyPromptScopeService:
    """Mirrors PromptScopeService.search_prompts's keyword-only signature."""

    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []
        self.call_count = 0

    async def search_prompts(self, *, mode=None, query=None, limit=10, fts_match_query=None):
        self.call_count += 1
        return self.rows


def _sem_item(source_id: str, source_type: str, score: float) -> dict[str, Any]:
    return {
        "id": f"{source_type}-{source_id}",
        "score": score,
        "document": f"doc for {source_id}",
        "metadata": {"source_id": source_id, "source_type": source_type},
    }


class _SpyRagService:
    """Mirrors RAGService.search's signature, keyed on metadata_allowlist."""

    def __init__(self, results_by_source_type: dict[str, list[dict]] | None = None):
        self.results_by_source_type = results_by_source_type or {}
        self.calls: list[dict] = []

    async def search(
        self,
        query,
        top_k=None,
        search_type="semantic",
        filter_metadata=None,
        include_citations=None,
        score_threshold=None,
        *,
        metadata_allowlist=None,
    ):
        self.calls.append({"query": query, "metadata_allowlist": metadata_allowlist})
        if metadata_allowlist is None:
            merged: list[dict] = []
            for items in self.results_by_source_type.values():
                merged.extend(items)
            return merged
        source_type = next(iter(metadata_allowlist.get("source_type", ())), None)
        return list(self.results_by_source_type.get(source_type, []))


def _scoped(**allowlist: set) -> EffectiveScope:
    return EffectiveScope(
        state="scoped",
        allowlist={k: frozenset(v) for k, v in allowlist.items()},
        cause=None,
    )


# --- Real-seam fixtures (media/notes/conversations, file-backed) -----------


@pytest.fixture
def scoped_local_stack(tmp_path):
    """Real notes/media/conversations seams wired the way the app wires
    LibraryLocalRagSearchService's dependencies -- extends
    test_library_local_rag_search_service.py's real_fts_app fixture with a
    real notes seam (NotesInteropService needs a file-backed DB: its
    `_get_db` opens a NEW CharactersRAGDB connection against the same
    `db_path_str`, which would be a blank database for `:memory:`)."""
    chachanotes_db = CharactersRAGDB(tmp_path / "chacha.db", client_id="scope-stack")
    media_db = MediaDatabase(tmp_path / "media.db", client_id="scope-stack")
    notes_interop = NotesInteropService(
        base_db_directory=tmp_path / "notes_base",
        api_client_id="scope-stack",
        global_db_to_use=chachanotes_db,
    )
    app = SimpleNamespace(
        notes_scope_service=NotesScopeService(
            local_notes_service=notes_interop, server_service=None
        ),
        media_reading_scope_service=MediaReadingScopeService(
            LocalMediaReadingService(media_db), None
        ),
        chachanotes_db=chachanotes_db,
        notes_user_id="scope-stack-user",
    )
    try:
        yield app, chachanotes_db, media_db, notes_interop
    finally:
        media_db.close_connection()
        chachanotes_db.close_connection()


@pytest.mark.asyncio
async def test_scoped_keyword_search_real_seams_returns_only_allowlisted_ids(
    scoped_local_stack,
):
    app, chachanotes_db, media_db, notes_interop = scoped_local_stack
    media_in, _uuid_in, _msg_in = media_db.add_media_with_keywords(
        title="In-scope report", media_type="document", content=_SCOPED_CONTENT
    )
    media_db.add_media_with_keywords(
        title="Out-of-scope report", media_type="document", content=_SCOPED_CONTENT
    )
    note_in = notes_interop.add_note(
        user_id="scope-stack-user", title="In-scope note", content=_SCOPED_CONTENT
    )
    notes_interop.add_note(
        user_id="scope-stack-user", title="Out-of-scope note", content=_SCOPED_CONTENT
    )
    conv_id = chachanotes_db.add_conversation({"title": "Roadmap thread"})
    chachanotes_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": _SCOPED_CONTENT,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    )
    scope = _scoped(**{SOURCE_TYPE_MEDIA: {str(media_in)}, SOURCE_TYPE_NOTE: {note_in}})
    service = LibraryLocalRagSearchService(app)

    result = await service.search(
        "quarterly roadmap",
        ("notes", "media", "conversations"),
        "search",
        top_k=10,
        scope=scope,
    )

    rows_by_type: dict[str, list[str]] = {}
    for row in result["results"]:
        rows_by_type.setdefault(row["provenance"]["source_type"], []).append(
            row["source_id"]
        )
    assert rows_by_type.get("media") == [str(media_in)]
    assert rows_by_type.get("note") == [note_in]
    assert "conversation" not in rows_by_type
    assert result["diagnostics"][SCOPE_DIAGNOSTICS_KEY] == {
        "status": SCOPE_STATUS_EXCLUDED,
        "reason": SCOPE_REASON_CONVERSATIONS_EXCLUDED,
    }


@pytest.mark.asyncio
async def test_unscoped_keyword_search_real_seams_returns_everything(
    scoped_local_stack,
):
    """D2 guard companion: no `scope` kwarg at all -> today's unrestricted
    behavior across every real seam, no conversations exclusion recorded."""
    app, chachanotes_db, media_db, notes_interop = scoped_local_stack
    media_db.add_media_with_keywords(
        title="A", media_type="document", content=_SCOPED_CONTENT
    )
    notes_interop.add_note(user_id="scope-stack-user", title="A", content=_SCOPED_CONTENT)
    conv_id = chachanotes_db.add_conversation({"title": "Roadmap thread"})
    chachanotes_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": _SCOPED_CONTENT,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    )
    service = LibraryLocalRagSearchService(app)

    result = await service.search(
        "quarterly roadmap", ("notes", "media", "conversations"), "search", top_k=10
    )

    types_present = {row["provenance"]["source_type"] for row in result["results"]}
    assert types_present == {"media", "note", "conversation"}
    assert result["diagnostics"] == {}


@pytest.fixture
def real_media_stack(tmp_path):
    media_db = MediaDatabase(tmp_path / "media_scope.db", client_id="scope-media")
    service = MediaReadingScopeService(LocalMediaReadingService(media_db), None)
    try:
        yield media_db, service
    finally:
        media_db.close_connection()


@pytest.mark.asyncio
async def test_media_reading_scope_service_id_allowlist_restricts_real_query(
    real_media_stack,
):
    """Proves the new `id_allowlist` -> `media_ids_filter` translation added
    to `MediaReadingScopeService.search_media` actually restricts the SQL
    query, not just a caller-side filter."""
    media_db, service = real_media_stack
    id_a, _uuid_a, _msg_a = media_db.add_media_with_keywords(
        title="Alpha", media_type="document", content=_SCOPED_CONTENT
    )
    media_db.add_media_with_keywords(
        title="Beta", media_type="document", content=_SCOPED_CONTENT
    )

    payload = await service.search_media(
        mode="local", query="roadmap", limit=10, offset=0, id_allowlist=[str(id_a)]
    )

    returned_ids = {str(item["source_id"]) for item in payload["items"]}
    assert returned_ids == {str(id_a)}


@pytest.fixture
def real_notes_stack(tmp_path):
    chachanotes_db = CharactersRAGDB(tmp_path / "notes_scope.db", client_id="scope-notes")
    interop = NotesInteropService(
        base_db_directory=tmp_path / "notes_base",
        api_client_id="scope-notes",
        global_db_to_use=chachanotes_db,
    )
    service = NotesScopeService(local_notes_service=interop, server_service=None)
    try:
        yield chachanotes_db, interop, service
    finally:
        chachanotes_db.close_connection()


@pytest.mark.asyncio
async def test_notes_scope_service_id_allowlist_restricts_real_query(real_notes_stack):
    """Proves the new `id_allowlist` kwarg added to `NotesScopeService`
    (and forwarded through `NotesInteropService`) reaches
    `CharactersRAGDB.search_notes`'s existing `json_each` predicate."""
    _chachanotes_db, interop, service = real_notes_stack
    note_a = interop.add_note(
        user_id="scope-notes-user", title="Alpha", content="rotate the credential"
    )
    interop.add_note(
        user_id="scope-notes-user", title="Beta", content="rotate the credential"
    )

    rows = await service.search_notes(
        scope="local_note",
        query="rotate",
        limit=10,
        user_id="scope-notes-user",
        id_allowlist=[note_a],
    )

    returned_ids = {row["id"] for row in rows}
    assert returned_ids == {note_a}


# --- Spy-based coverage: conversations exclusion + diagnostics, D2 byte- --
# --- identical guard, EMPTY guard, zero-results marker --------------------


@pytest.mark.asyncio
async def test_scoped_keyword_search_excludes_conversations_and_diagnoses():
    conv_db = _SpyConversationsDB(rows=[{"id": "c1", "title": "Conv", "message_count": 2}])
    notes_service = _SpyNotesScopeService(rows=[{"id": "n1", "title": "N1", "content": "c"}])
    app = SimpleNamespace(
        chachanotes_db=conv_db,
        notes_scope_service=notes_service,
        media_reading_scope_service=None,
    )
    service = LibraryLocalRagSearchService(app)
    scope = _scoped(**{SOURCE_TYPE_NOTE: {"n1"}})

    result = await service.search(
        "q", ("notes", "conversations"), "search", top_k=5, scope=scope
    )

    assert conv_db.call_count == 0
    assert result["diagnostics"][SCOPE_DIAGNOSTICS_KEY] == {
        "status": SCOPE_STATUS_EXCLUDED,
        "reason": SCOPE_REASON_CONVERSATIONS_EXCLUDED,
    }
    assert {row["source_id"] for row in result["results"]} == {"n1"}


@pytest.mark.asyncio
async def test_unscoped_keyword_search_includes_conversations_no_diagnostics():
    conv_db = _SpyConversationsDB(rows=[{"id": "c1", "title": "Conv", "message_count": 2}])
    app = SimpleNamespace(
        chachanotes_db=conv_db, notes_scope_service=None, media_reading_scope_service=None
    )
    service = LibraryLocalRagSearchService(app)

    result = await service.search("q", ("conversations",), "search", top_k=5)

    assert conv_db.call_count == 1
    assert result["diagnostics"] == {}


@pytest.mark.asyncio
async def test_scoped_keyword_search_excludes_prompts_and_diagnoses():
    """Mirrors ``test_scoped_keyword_search_excludes_conversations_and_diagnoses``:
    prompts are not part of the scope vocabulary either (spec D5), so a
    scoped search must never call the prompts seam and must record the
    same-shaped exclusion diagnostic under a dedicated reason."""
    prompts_service = _SpyPromptScopeService(
        rows=[{"local_id": "p1", "name": "P1", "user_prompt": "c"}]
    )
    notes_service = _SpyNotesScopeService(rows=[{"id": "n1", "title": "N1", "content": "c"}])
    app = SimpleNamespace(
        prompt_scope_service=prompts_service,
        notes_scope_service=notes_service,
        media_reading_scope_service=None,
        chachanotes_db=None,
    )
    service = LibraryLocalRagSearchService(app)
    scope = _scoped(**{SOURCE_TYPE_NOTE: {"n1"}})

    result = await service.search(
        "q", ("notes", "prompts"), "search", top_k=5, scope=scope
    )

    assert prompts_service.call_count == 0
    assert result["diagnostics"][SCOPE_DIAGNOSTICS_KEY] == {
        "status": SCOPE_STATUS_EXCLUDED,
        "reason": SCOPE_REASON_PROMPTS_EXCLUDED,
    }
    assert {row["source_id"] for row in result["results"]} == {"n1"}


@pytest.mark.asyncio
async def test_unscoped_keyword_search_includes_prompts_no_diagnostics():
    prompts_service = _SpyPromptScopeService(
        rows=[{"local_id": "p1", "name": "P1", "user_prompt": "c"}]
    )
    app = SimpleNamespace(
        prompt_scope_service=prompts_service,
        notes_scope_service=None,
        media_reading_scope_service=None,
        chachanotes_db=None,
    )
    service = LibraryLocalRagSearchService(app)

    result = await service.search("q", ("prompts",), "search", top_k=5)

    assert prompts_service.call_count == 1
    assert result["diagnostics"] == {}


@pytest.mark.asyncio
async def test_unscoped_call_without_scope_kwarg_is_byte_identical_to_explicit_none():
    """D2 guard: the service called with NO `scope` kwarg behaves
    identically to an explicit `scope=None` -- both mean "unrestricted"."""
    notes_service = _SpyNotesScopeService(rows=[{"id": "n1", "title": "N1", "content": "c"}])
    app = SimpleNamespace(
        notes_scope_service=notes_service,
        media_reading_scope_service=None,
        chachanotes_db=None,
    )
    service = LibraryLocalRagSearchService(app)

    implicit = await service.search("q", ("notes",), "search", top_k=5)
    explicit = await service.search("q", ("notes",), "search", top_k=5, scope=None)

    assert implicit == explicit


def test_library_screen_call_sites_never_pass_scope_kwarg():
    """D2 guard: grep-assert (AST-based) that the Library screen's own
    Search canvas never passes `scope=` when building a
    `LibraryRagSearchRequest` -- it must stay unscoped by omission, the
    dataclass default doing all the work."""
    source_path = (
        Path(tldw_chatbook.__file__).parent / "UI" / "Screens" / "library_screen.py"
    )
    tree = ast.parse(source_path.read_text())
    request_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LibraryRagSearchRequest"
    ]
    assert request_calls, "expected at least one LibraryRagSearchRequest(...) call site"
    for call in request_calls:
        keyword_names = {kw.arg for kw in call.keywords}
        assert "scope" not in keyword_names


@pytest.mark.asyncio
async def test_empty_scope_raises_value_error():
    """state=='empty' must never reach the service -- callers must
    short-circuit first (mirrors task-5's EMPTY short-circuit contract)."""
    app = SimpleNamespace()
    service = LibraryLocalRagSearchService(app)
    scope = EffectiveScope(state="empty", allowlist={}, cause="deleted-items")

    with pytest.raises(ValueError):
        await service.search("q", ("notes",), "search", scope=scope)


@pytest.mark.asyncio
async def test_scoped_keyword_search_zero_results_marker():
    notes_service = _SpyNotesScopeService(rows=[])
    app = SimpleNamespace(
        notes_scope_service=notes_service,
        media_reading_scope_service=None,
        chachanotes_db=None,
    )
    service = LibraryLocalRagSearchService(app)
    scope = _scoped(**{SOURCE_TYPE_NOTE: {"n1", "n2"}})

    outcome = await service.search("q", ("notes",), "search", top_k=5, scope=scope)

    assert isinstance(outcome, LibraryRagSearchOutcome)
    assert outcome.status == "empty"
    assert "No results within scope (2 items searched)" in outcome.recovery_state.why


# --- Semantic (rag mode) delegate: per-type allowlists, merge, zero-marker -


@pytest.mark.asyncio
async def test_scoped_semantic_search_runs_one_query_per_type_and_merges_by_score():
    rag = _SpyRagService(
        {
            "media": [_sem_item("m1", "media", 0.5)],
            "note": [_sem_item("n1", "note", 0.9)],
        }
    )
    app = SimpleNamespace(_rag_service=rag)
    service = LibraryLocalRagSearchService(app)
    scope = _scoped(**{SOURCE_TYPE_MEDIA: {"m1"}, SOURCE_TYPE_NOTE: {"n1"}})

    result = await service.search("q", ("notes", "media"), "rag", top_k=5, scope=scope)

    assert len(rag.calls) == 2
    allowlists = [call["metadata_allowlist"] for call in rag.calls]
    assert {"source_type": {"media"}, "source_id": {"m1"}} in allowlists
    assert {"source_type": {"note"}, "source_id": {"n1"}} in allowlists
    ids_in_order = [row["source_id"] for row in result["results"]]
    assert ids_in_order == ["n1", "m1"]  # merged, sorted by score descending


@pytest.mark.asyncio
async def test_unscoped_semantic_search_issues_a_single_unrestricted_query():
    rag = _SpyRagService({"note": [_sem_item("n1", "note", 0.9)]})
    app = SimpleNamespace(_rag_service=rag)
    service = LibraryLocalRagSearchService(app)

    result = await service.search("q", ("notes",), "rag", top_k=5)

    assert len(rag.calls) == 1
    assert rag.calls[0]["metadata_allowlist"] is None
    assert [row["source_id"] for row in result["results"]] == ["n1"]


@pytest.mark.asyncio
async def test_scoped_semantic_search_zero_results_marker():
    rag = _SpyRagService({})
    app = SimpleNamespace(_rag_service=rag)
    service = LibraryLocalRagSearchService(app)
    scope = _scoped(**{SOURCE_TYPE_NOTE: {"n1", "n2", "n3"}})

    outcome = await service.search("q", ("notes",), "rag", top_k=5, scope=scope)

    assert isinstance(outcome, LibraryRagSearchOutcome)
    assert outcome.status == "empty"
    assert "No results within scope (3 items searched)" in outcome.recovery_state.why


# --- Console call site: scope resolution (task-6) --------------------------


@pytest.mark.asyncio
async def test_console_scope_resolution_scopes_request_for_scoped_conversation(tmp_path):
    # File-backed, not `:memory:`: scope resolution offloads the DB read to
    # a worker thread (`asyncio.to_thread`), and in-memory SQLite
    # connections are thread-local -- only the thread that created the `:memory:`
    # DB sees its schema (the same trap `_search_conversations` documents).
    app = _build_test_app()
    db = CharactersRAGDB(tmp_path / "console_scope.db", client_id="console-scope-test")
    app.chachanotes_db = db
    note_id = db.add_note(title="Runbook", content="Rotate credentials")
    conv_id = db.add_conversation({"title": "Scoped console convo"})
    write_conversation_scope(
        db,
        conv_id,
        RagScope(
            items=(ScopeItem(SOURCE_TYPE_NOTE, note_id),),
            updated_at="2026-01-01T00:00:00Z",
        ),
    )
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.create_session(title="Console session")
    session.persisted_conversation_id = conv_id
    app._screen_stacks[app._current_mode] = [screen]
    request = LibraryRagSearchRequest(
        query="q", source_types=("notes", "media", "conversations")
    )

    scoped_request, outcome = await screen._resolve_console_library_rag_scope(request)

    assert outcome is None
    assert scoped_request.scope is not None
    assert scoped_request.scope.state == "scoped"
    assert scoped_request.scope.allowlist[SOURCE_TYPE_NOTE] == frozenset({note_id})
    assert scoped_request.query == request.query
    assert scoped_request.source_types == request.source_types


@pytest.mark.asyncio
async def test_console_scope_resolution_short_circuits_on_empty_scope(tmp_path):
    app = _build_test_app()
    db = CharactersRAGDB(
        tmp_path / "console_scope_empty.db", client_id="console-scope-empty-test"
    )
    app.chachanotes_db = db
    conv_id = db.add_conversation({"title": "Scoped to deleted note"})
    write_conversation_scope(
        db,
        conv_id,
        RagScope(
            items=(ScopeItem(SOURCE_TYPE_NOTE, "note-never-created"),),
            updated_at="2026-01-01T00:00:00Z",
        ),
    )
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.create_session(title="Console session")
    session.persisted_conversation_id = conv_id
    app._screen_stacks[app._current_mode] = [screen]
    request = LibraryRagSearchRequest(query="q", source_types=("notes",))

    scoped_request, outcome = await screen._resolve_console_library_rag_scope(request)

    assert outcome is not None
    assert outcome.status == "empty"
    assert outcome.recovery_state is not None
    assert (
        "Retrieval scope is empty (deleted-items); no sources searched."
        in outcome.recovery_state.why
    )
    assert scoped_request.scope is None


@pytest.mark.asyncio
async def test_console_scope_resolution_unscoped_when_no_active_session():
    app = _build_test_app()
    screen = ChatScreen(app)
    app._screen_stacks[app._current_mode] = [screen]
    request = LibraryRagSearchRequest(query="q", source_types=("notes",))

    scoped_request, outcome = await screen._resolve_console_library_rag_scope(request)

    assert outcome is None
    assert scoped_request is request
