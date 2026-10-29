"""A note deleted for good never comes back through semantic search (I-1).

TASK-34000.29 (N-29, part 8b), review R0 finding I-1. Two layers, each
enough on its own:

- **Removal.** ``NotesScopeService.purge_note`` removes the note's
  ``note_<id>`` vectors from every store on this device whatever
  ``[AppRAGSearchConfig.rag.indexing].enabled`` says. That switch stops new
  indexing; Library semantic search still reads what an earlier backfill
  wrote. The reviewer's configuration: embeddings deps installed, indexing
  switched off, nothing built in this session.
- **Liveness.** Library semantic and hybrid search drop any note hit whose
  note no longer exists or sits in Recently deleted, in one batched lookup
  per page. A stale vector left by a purge -- or by a quit before the
  removal ran -- and a soft-deleted note's vectors (the older leak) yield
  no row; a live note keeps its row.

Base (780fe1da8b): the removal was skipped whenever indexing was switched
off (the Media guard), and search had no liveness check, so every pin here
fails there.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.RAG_Search import ingestion_indexing

#: The interop service opens its per-user database through the real
#: source-bound config readers (the Console library migration seed).
pytestmark = pytest.mark.bootstrap_profile

USER = "user-1"
#: One fixed query vector and one fixed chunk vector: the stub embedder.
_VECTOR = [0.25] * 8


class _Policy:
    def require_allowed(self, *, action_id: str) -> None:
        return None


class _Profile:
    """A live note, a note in Recently deleted, and one about to go for good."""

    def __init__(self, tmp_path: Path) -> None:
        self.db = CharactersRAGDB(tmp_path / "notes.sqlite3", client_id="t8b-i1")
        self.service = NotesScopeService(
            local_notes_service=NotesInteropService(
                base_db_directory=tmp_path,
                api_client_id="t8b-i1",
                global_db_to_use=self.db,
            ),
            server_service=None,
            policy_enforcer=_Policy(),
        )
        self.live_id = self.db.add_note("Live harbour log", "harbour tide table")
        self.trashed_id = self._trash("Trashed harbour log", "harbour gull count")
        self.gone_id = self._trash("Gone harbour log", "harbour secret ledger")
        self.gone_version = self._version(self.gone_id)

    def _trash(self, title: str, content: str) -> str:
        note_id = self.db.add_note(title, content)
        version = self.db.get_note_by_id(note_id)["version"]
        self.db.soft_delete_note(note_id, version)
        return note_id

    def _version(self, note_id: str) -> int:
        row = self.db.get_connection().execute(
            "SELECT version FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        return int(row[0])


def _hit(note_id: str, title: str, text: str) -> dict[str, Any]:
    """One chunk exactly as the backfill wrote it (``note_document``)."""
    return {
        "id": f"note_{note_id}_chunk_0",
        "score": 0.9,
        "document": text,
        "metadata": {
            "doc_id": f"note_{note_id}",
            "source_id": note_id,
            "title": title,
            "source_type": "note",
        },
    }


class _StaleIndex:
    """A semantic runtime whose vectors predate the purge and the delete."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.results)


def _rows(result: Any) -> list[dict[str, Any]]:
    return list(result.get("results") or ()) if isinstance(result, dict) else []


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ["semantic", "hybrid"])
async def test_semantic_search_drops_hits_for_notes_that_are_gone(
    tmp_path, arm
) -> None:
    """Liveness alone: no removal ran, the vectors are all still there."""
    profile = _Profile(tmp_path)
    assert profile.db.hard_delete_note(profile.gone_id) is True
    index = _StaleIndex(
        [
            _hit(profile.gone_id, "Gone harbour log", "harbour secret ledger"),
            _hit(profile.trashed_id, "Trashed harbour log", "harbour gull count"),
            _hit(profile.live_id, "Live harbour log", "harbour tide table"),
            {
                "id": "media_7_chunk_0",
                "score": 0.5,
                "document": "harbour film",
                "metadata": {"source_id": "7", "title": "Film", "source_type": "media"},
            },
        ]
    )
    app = SimpleNamespace(notes_scope_service=profile.service, notes_user_id=USER)
    search = LibraryLocalRagSearchService(app)
    run = search._search_semantic if arm == "semantic" else search._search_hybrid

    rows = _rows(
        await run("harbour", ("notes", "media"), 10, {}, rag_service=index)
    )

    painted = [(row["source_id"], row["title"], row["snippet"]) for row in rows]
    assert ("7", "Film", "harbour film") in painted, painted
    assert (profile.live_id, "Live harbour log", "harbour tide table") in painted, (
        "the control: a live note keeps its row"
    )
    assert not [row for row in painted if row[0] == profile.gone_id], (
        f"a purged note's stale vector came back: {painted}"
    )
    assert not [row for row in painted if row[0] == profile.trashed_id], (
        f"a note in Recently deleted came back through semantic search: {painted}"
    )


@pytest.fixture
def no_shared_service():
    """Nothing built in this session, and the singleton restored after."""
    before = ingestion_indexing.peek_shared_rag_service()
    ingestion_indexing.set_shared_rag_service(None)
    yield
    ingestion_indexing.set_shared_rag_service(before)


@pytest.fixture
def indexing_switched_off(monkeypatch):
    """``[AppRAGSearchConfig.rag.indexing] enabled = false``, as R0 probed."""
    monkeypatch.setattr(ingestion_indexing, "_indexing_enabled_in_config", lambda: False)
    assert ingestion_indexing.semantic_indexing_available() is False


def _chroma(persist: Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(persist),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def _seed_collection(persist: Path, name: str, profile: _Profile) -> None:
    """A backfill's note chunks: the live note and the one about to go."""
    client = _chroma(persist)
    try:
        collection = client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        hits = [
            _hit(profile.gone_id, "Gone harbour log", "harbour secret ledger"),
            _hit(profile.live_id, "Live harbour log", "harbour tide table"),
        ]
        collection.add(
            ids=[hit["id"] for hit in hits],
            embeddings=[_VECTOR for _ in hits],
            documents=[hit["document"] for hit in hits],
            metadatas=[hit["metadata"] for hit in hits],
        )
    finally:
        client.close()


def _found(persist: Path) -> dict[str, set[str]]:
    """What a search over each collection returns for the stub query."""
    client = _chroma(persist)
    try:
        found: dict[str, set[str]] = {}
        for collection in client.list_collections():
            handle = client.get_collection(
                collection.name, embedding_function=None, data_loader=None
            )
            answer = handle.query(
                query_embeddings=[_VECTOR],
                n_results=10,
                include=["metadatas", "documents"],
            )
            found[collection.name] = {
                f"{meta['doc_id']}:{doc}"
                for meta, doc in zip(answer["metadatas"][0], answer["documents"][0])
            }
        return found
    finally:
        client.close()


@pytest.mark.asyncio
async def test_delete_forever_removes_the_vectors_with_indexing_switched_off(
    tmp_path, monkeypatch, no_shared_service, indexing_switched_off
) -> None:
    """The reviewer's configuration: deps installed, indexing off, nothing built.

    The active profile's persisted Chroma store holds the note in its own
    collection AND in an older embedding fingerprint's collection (a profile
    switch makes that one searchable again). Both lose it.
    """
    pytest.importorskip("chromadb")
    from tldw_chatbook.RAG_Search.simplified.active_config import (
        resolve_active_rag_config,
    )
    from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import (
        fingerprinted_collection_name,
    )

    persist = tmp_path / "chromadb"
    monkeypatch.setenv("RAG_PERSIST_DIR", str(persist))
    config = resolve_active_rag_config()
    assert str(config.vector_store_type) == "chroma"
    assert Path(config.persist_directory) == persist
    active = fingerprinted_collection_name(config)
    profile = _Profile(tmp_path)
    _seed_collection(persist, active, profile)
    _seed_collection(persist, "older_fingerprint", profile)
    gone = f"note_{profile.gone_id}:harbour secret ledger"
    live = f"note_{profile.live_id}:harbour tide table"
    assert _found(persist) == {active: {gone, live}, "older_fingerprint": {gone, live}}

    purged = await profile.service.purge_note(
        scope=ScopeType.LOCAL_NOTE,
        note_id=profile.gone_id,
        version=profile.gone_version,
        user_id=USER,
    )

    assert purged is True
    assert _found(persist) == {active: {live}, "older_fingerprint": {live}}, (
        "the purged note's text is still searchable in the vector store"
    )


class _Cache:
    def __init__(self) -> None:
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1


@pytest.mark.asyncio
async def test_delete_forever_removes_the_vectors_from_a_running_service(
    tmp_path, monkeypatch, no_shared_service, indexing_switched_off
) -> None:
    """A runtime built earlier this session drops the note and its cached answers."""
    from tldw_chatbook.RAG_Search.simplified.vector_store import InMemoryVectorStore

    monkeypatch.setenv("RAG_PERSIST_DIR", str(tmp_path / "no-store-here"))
    profile = _Profile(tmp_path)
    store = InMemoryVectorStore()
    hits = [
        _hit(profile.gone_id, "Gone harbour log", "harbour secret ledger"),
        _hit(profile.live_id, "Live harbour log", "harbour tide table"),
    ]
    store.add(
        [hit["id"] for hit in hits],
        [_VECTOR for _ in hits],
        [hit["document"] for hit in hits],
        [hit["metadata"] for hit in hits],
    )
    cache = _Cache()
    ingestion_indexing.set_shared_rag_service(
        SimpleNamespace(vector_store=store, cache=cache)
    )

    purged = await profile.service.purge_note(
        scope=ScopeType.LOCAL_NOTE,
        note_id=profile.gone_id,
        version=profile.gone_version,
        user_id=USER,
    )

    assert purged is True
    found = {
        (result.metadata["doc_id"], result.document)
        for result in store.search(_VECTOR, top_k=10)
    }
    assert found == {(f"note_{profile.live_id}", "harbour tide table")}
    assert cache.cleared >= 1, "a cached answer could still show the purged note"
    assert not (tmp_path / "no-store-here").exists(), (
        "a removal must never create a store that was not there"
    )
