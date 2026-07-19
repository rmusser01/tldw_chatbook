"""
Tests for ingestion-time RAG indexing (task-247).

Covers:
- The media post-ingest hook seam on MediaDatabase.add_media_with_keywords
  (fires post-commit, never breaks ingestion, unregisterable).
- The deps/config availability gate: with embeddings deps missing no indexing
  is attempted and ingestion is unaffected (AC #5).
- Document builders producing the metadata contract consumed by
  library_local_rag_search_service._semantic_row (source_id / chunk_id /
  title / source_type).
- The IngestionIndexer background worker: indexes, skips unchanged items,
  re-indexes updated items (deleting stale chunks first), and survives +
  reports indexing failures (AC #1, #4).
- End-to-end: ingest a document with distinctive content -> background worker
  indexes it -> semantic search returns it (AC #2), using the deterministic
  mock embedding backend; a chromadb-gated variant proves the persistent
  round-trip.
- Bulk backfill of pre-existing media/notes/conversations, incremental on the
  second run (AC #3).

Real SQLite tmp-file databases are used throughout (in-memory media DBs are
thread-local and invisible to the worker thread by design).
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.DB import Client_Media_DB_v2 as media_db_module
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search import ingestion_indexing
from tldw_chatbook.RAG_Search.ingestion_indexing import (
    IndexEntry,
    IngestionIndexer,
    backfill_semantic_index,
    conversation_document,
    install_media_ingest_hook,
    media_document,
    media_index_entry,
    note_document,
    semantic_indexing_available,
    uninstall_media_ingest_hook,
)
from tldw_chatbook.RAG_Search.simplified.data_models import IndexingResult
from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed


DISTINCTIVE_CONTENT = (
    "The zanzibar quokka manifesto describes how quokkas organise snorkeling "
    "expeditions near flamingo lagoons. Zanzibar quokka snorkeling requires "
    "specialised flamingo-approved equipment and a manifesto committee."
)

DECOY_CONTENT = (
    "Corporate tax law amendments for fiscal year filings involve depreciation "
    "schedules, withholding obligations and statutory audit deadlines."
)


# === Fixtures ===

@pytest.fixture
def media_db(tmp_path):
    db = MediaDatabase(tmp_path / "media.db", client_id="test-ingest")
    yield db
    try:
        db.close_connection()
    except Exception:
        pass


@pytest.fixture
def indexing_db(tmp_path):
    return RAGIndexingDB(tmp_path / "rag_indexing.db")


@pytest.fixture(autouse=True)
def _clean_hook_registry():
    """Ensure no post-ingest callbacks or hook installs leak between tests."""
    yield
    uninstall_media_ingest_hook()
    media_db_module._MEDIA_POST_INGEST_CALLBACKS.clear()


def _add_media(db, *, title="Doc", content="hello world content", media_type="document", overwrite=True, url=None):
    return db.add_media_with_keywords(
        url=url,
        title=title,
        media_type=media_type,
        content=content,
        keywords=["test"],
        overwrite=overwrite,
    )


class FakeVectorStore:
    def __init__(self):
        self.deleted = []

    def delete_document(self, doc_id):
        self.deleted.append(doc_id)


class FakeRAGService:
    """Minimal stand-in exposing the real batch indexing API."""

    def __init__(self):
        self.vector_store = FakeVectorStore()
        self.indexed_docs = []
        self.fail = False

    async def index_batch_optimized(self, documents, show_progress=True, batch_size=32):
        if self.fail:
            raise RuntimeError("boom: embedding backend exploded")
        self.indexed_docs.extend(documents)
        return [
            IndexingResult(doc_id=d["id"], chunks_created=2, time_taken=0.0, success=True)
            for d in documents
        ]


@pytest.fixture
def fake_service():
    return FakeRAGService()


@pytest.fixture
def indexer(fake_service, indexing_db):
    idx = IngestionIndexer(rag_service=fake_service, indexing_db=indexing_db)
    yield idx
    idx.stop()


def _entry(item_id="1", *, content=DISTINCTIVE_CONTENT, title="Quokka Manifesto",
           last_modified=None, item_type="media"):
    last_modified = last_modified or datetime.now(timezone.utc)
    return IndexEntry(
        item_id=str(item_id),
        item_type=item_type,
        last_modified=last_modified,
        document={
            "id": f"{item_type}_{item_id}",
            "content": content,
            "title": title,
            "metadata": {
                "source_id": str(item_id),
                "title": title,
                "source_type": item_type,
            },
        },
    )


# === Post-ingest hook seam (DB layer) ===

@pytest.mark.unit
class TestMediaPostIngestHook:
    def test_callback_fires_post_commit_for_new_media(self, media_db):
        seen = []

        def callback(db, media_id, media_uuid):
            # The committed row must be visible from within the callback.
            row = db.get_media_by_id(media_id)
            seen.append((media_id, media_uuid, row["content"] if row else None))

        media_db_module.register_media_post_ingest_callback(callback)
        try:
            media_id, media_uuid, _msg = _add_media(media_db, content="post commit visible")
        finally:
            media_db_module.unregister_media_post_ingest_callback(callback)

        assert media_id is not None
        assert seen == [(media_id, media_uuid, "post commit visible")]

    def test_no_callback_for_duplicate_without_overwrite(self, media_db):
        calls = []
        _add_media(media_db, content="dup content", overwrite=False)

        media_db_module.register_media_post_ingest_callback(
            lambda db, mid, muuid: calls.append(mid)
        )
        try:
            media_id, _, message = _add_media(media_db, content="dup content", overwrite=False)
        finally:
            media_db_module._MEDIA_POST_INGEST_CALLBACKS.clear()

        assert media_id is None
        assert "already exists" in message
        assert calls == []

    def test_callback_exception_does_not_break_ingestion(self, media_db):
        def bad_callback(db, media_id, media_uuid):
            raise RuntimeError("hook exploded")

        media_db_module.register_media_post_ingest_callback(bad_callback)
        try:
            media_id, _, _ = _add_media(media_db, content="survives hook errors")
        finally:
            media_db_module._MEDIA_POST_INGEST_CALLBACKS.clear()

        assert media_id is not None
        assert media_db.get_media_by_id(media_id) is not None

    def test_unregister_stops_callbacks(self, media_db):
        calls = []
        cb = lambda db, mid, muuid: calls.append(mid)
        media_db_module.register_media_post_ingest_callback(cb)
        media_db_module.unregister_media_post_ingest_callback(cb)

        _add_media(media_db, content="nobody listening")
        assert calls == []

    def test_callback_fires_for_content_update(self, media_db):
        calls = []
        media_id, _, _ = _add_media(media_db, content="version one", url="https://example.com/x")

        media_db_module.register_media_post_ingest_callback(
            lambda db, mid, muuid: calls.append(mid)
        )
        try:
            updated_id, _, _ = _add_media(
                media_db, content="version two", url="https://example.com/x", overwrite=True
            )
        finally:
            media_db_module._MEDIA_POST_INGEST_CALLBACKS.clear()

        assert updated_id == media_id
        assert calls == [media_id]


# === Availability gate (AC #5) ===

@pytest.mark.unit
class TestAvailabilityGate:
    def test_no_indexing_attempted_when_deps_missing(self, media_db, monkeypatch):
        monkeypatch.setattr(ingestion_indexing, "embeddings_rag_deps_installed", lambda: False)

        def _fail_if_touched():
            raise AssertionError("indexer must not be touched when deps are missing")

        monkeypatch.setattr(ingestion_indexing, "get_ingestion_indexer", _fail_if_touched)

        install_media_ingest_hook()
        media_id, _, _ = _add_media(media_db, content="no deps, still ingests fine")

        assert media_id is not None
        assert media_db.get_media_by_id(media_id) is not None

    def test_semantic_indexing_available_false_when_deps_missing(self, monkeypatch):
        monkeypatch.setattr(ingestion_indexing, "embeddings_rag_deps_installed", lambda: False)
        assert semantic_indexing_available() is False

    def test_config_kill_switch_disables_indexing(self, media_db, monkeypatch):
        monkeypatch.setattr(ingestion_indexing, "embeddings_rag_deps_installed", lambda: True)
        monkeypatch.setattr(
            ingestion_indexing,
            "get_cli_setting",
            lambda section, key, default=None: {"indexing": {"enabled": False}}
            if (section, key) == ("AppRAGSearchConfig", "rag")
            else default,
        )

        def _fail_if_touched():
            raise AssertionError("indexer must not be touched when disabled by config")

        monkeypatch.setattr(ingestion_indexing, "get_ingestion_indexer", _fail_if_touched)

        install_media_ingest_hook()
        media_id, _, _ = _add_media(media_db, content="disabled by config")
        assert media_id is not None

    def test_install_hook_is_idempotent(self, media_db, monkeypatch):
        submitted = []

        class OneShotIndexer:
            def submit(self, entry):
                submitted.append(entry)
                return True

        monkeypatch.setattr(ingestion_indexing, "get_ingestion_indexer", lambda: OneShotIndexer())
        monkeypatch.setattr(ingestion_indexing, "embeddings_rag_deps_installed", lambda: True)

        install_media_ingest_hook()
        install_media_ingest_hook()
        _add_media(media_db, content="only one submission expected")

        assert len(submitted) == 1


# === Document builders (metadata contract for _semantic_row) ===

@pytest.mark.unit
class TestDocumentBuilders:
    def test_media_document_contract(self):
        media = {
            "id": 42,
            "uuid": "abc-123",
            "title": "Quokka Manifesto",
            "type": "document",
            "content": DISTINCTIVE_CONTENT,
            "last_modified": "2026-07-16T10:00:00.000Z",
        }
        doc = media_document(media)
        assert doc["id"] == "media_42"
        assert doc["content"] == DISTINCTIVE_CONTENT
        assert doc["title"] == "Quokka Manifesto"
        meta = doc["metadata"]
        assert meta["source_id"] == "42"
        assert meta["title"] == "Quokka Manifesto"
        assert meta["source_type"] == "media"

    def test_media_document_returns_none_without_content(self):
        assert media_document({"id": 1, "title": "empty", "content": ""}) is None
        assert media_document({"id": 1, "title": "none"}) is None

    def test_media_index_entry_parses_timestamp(self):
        media = {
            "id": 7,
            "title": "T",
            "content": "some content",
            "last_modified": "2026-07-16T10:00:00.123Z",
        }
        entry = media_index_entry(media)
        assert entry.item_id == "7"
        assert entry.item_type == "media"
        assert entry.last_modified.tzinfo is not None

    def test_note_document_contract(self):
        note = {
            "id": "note-uuid-1",
            "title": "My Note",
            "content": "note body",
            "last_modified": "2026-07-16T10:00:00Z",
        }
        doc = note_document(note)
        assert doc["id"] == "note_note-uuid-1"
        assert doc["metadata"]["source_type"] == "note"
        assert doc["metadata"]["source_id"] == "note-uuid-1"
        assert doc["metadata"]["title"] == "My Note"

    def test_conversation_document_contract(self):
        conv = {"id": "conv-1", "title": "Chat about quokkas", "last_modified": "2026-07-16T10:00:00Z"}
        messages = [
            {"sender": "user", "content": "Tell me about quokkas"},
            {"sender": "assistant", "content": "Quokkas are marsupials."},
        ]
        doc = conversation_document(conv, messages)
        assert doc["id"] == "conversation_conv-1"
        assert doc["metadata"]["source_type"] == "conversation"
        assert doc["metadata"]["source_id"] == "conv-1"
        assert "Tell me about quokkas" in doc["content"]
        assert "Quokkas are marsupials." in doc["content"]

    def test_conversation_document_none_without_messages(self):
        assert conversation_document({"id": "conv-2", "title": "empty"}, []) is None


# === Background worker (AC #1, #4) ===

@pytest.mark.unit
class TestIngestionIndexer:
    def test_submit_indexes_and_marks_item(self, indexer, fake_service, indexing_db):
        entry = _entry("11")
        assert indexer.submit(entry) is True
        assert indexer.wait_until_idle(timeout=10)

        assert [d["id"] for d in fake_service.indexed_docs] == ["media_11"]
        info = indexing_db.get_indexed_item_info("11", "media")
        assert info is not None
        assert info["chunk_count"] == 2
        stats = indexer.stats()
        assert stats["indexed"] == 1
        assert stats["failed"] == 0

    def test_unchanged_item_is_skipped(self, indexer, fake_service, indexing_db):
        ts = datetime.now(timezone.utc)
        indexing_db.mark_item_indexed("22", "media", last_modified=ts, chunk_count=2)

        indexer.submit(_entry("22", last_modified=ts))
        assert indexer.wait_until_idle(timeout=10)

        assert fake_service.indexed_docs == []
        assert indexer.stats()["skipped"] == 1

    def test_updated_item_is_reindexed_with_stale_chunk_delete(self, indexer, fake_service, indexing_db):
        old = datetime.now(timezone.utc) - timedelta(hours=1)
        indexing_db.mark_item_indexed("33", "media", last_modified=old, chunk_count=2)

        indexer.submit(_entry("33", last_modified=datetime.now(timezone.utc)))
        assert indexer.wait_until_idle(timeout=10)

        assert [d["id"] for d in fake_service.indexed_docs] == ["media_33"]
        assert "media_33" in fake_service.vector_store.deleted

    def test_failure_is_recorded_and_worker_survives(self, indexer, fake_service, indexing_db):
        failures = []
        indexer.set_failure_notifier(lambda msg: failures.append(msg))

        fake_service.fail = True
        indexer.submit(_entry("44"))
        assert indexer.wait_until_idle(timeout=10)

        stats = indexer.stats()
        assert stats["failed"] == 1
        assert "boom" in (stats["last_error"] or "")
        assert failures, "failure notifier should have been invoked"
        assert indexing_db.get_indexed_item_info("44", "media") is None

        # Worker must still be alive and able to process new work (AC #4).
        fake_service.fail = False
        indexer.submit(_entry("45"))
        assert indexer.wait_until_idle(timeout=10)
        assert indexer.stats()["indexed"] == 1
        assert indexing_db.get_indexed_item_info("45", "media") is not None

    def test_submit_does_not_block_caller(self, indexing_db):
        release = threading.Event()

        class SlowService(FakeRAGService):
            async def index_batch_optimized(self, documents, show_progress=True, batch_size=32):
                await asyncio.to_thread(release.wait, 10)
                return await super().index_batch_optimized(documents, show_progress, batch_size)

        slow = SlowService()
        idx = IngestionIndexer(rag_service=slow, indexing_db=indexing_db)
        try:
            start = time.monotonic()
            idx.submit(_entry("55"))
            elapsed = time.monotonic() - start
            assert elapsed < 1.0, "submit must not wait for indexing to complete"
            release.set()
            assert idx.wait_until_idle(timeout=10)
        finally:
            release.set()
            idx.stop()


# === End-to-end: ingest -> worker -> semantic search (AC #1, #2) ===

def _make_real_service(store_type="memory", persist_dir=None):
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

    cfg = RAGConfig()
    cfg.embedding.model = "mock"  # deterministic bag-of-words backend, offline
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = store_type
    cfg.vector_store.persist_directory = persist_dir
    cfg.chunking.chunk_size = 60
    cfg.chunking.chunk_overlap = 10
    cfg.search.enable_cache = False
    return RAGService(cfg)


@pytest.mark.integration
@pytest.mark.skipif(not embeddings_rag_deps_installed(), reason="embeddings_rag deps not installed")
class TestEndToEndSemanticSearch:
    def _ingest_and_search(self, media_db, tmp_path, monkeypatch, service):
        indexer = IngestionIndexer(
            rag_service=service,
            indexing_db=RAGIndexingDB(tmp_path / "rag_indexing.db"),
        )
        monkeypatch.setattr(ingestion_indexing, "get_ingestion_indexer", lambda: indexer)
        install_media_ingest_hook()
        try:
            _add_media(media_db, title="Tax Law Digest", content=DECOY_CONTENT)
            media_id, _, _ = _add_media(media_db, title="Quokka Manifesto", content=DISTINCTIVE_CONTENT)
            assert media_id is not None
            assert indexer.wait_until_idle(timeout=60), "background indexing did not finish"
            assert indexer.stats()["failed"] == 0, indexer.stats()["last_error"]

            results = asyncio.run(
                service.search(
                    "zanzibar quokka snorkeling manifesto",
                    top_k=3,
                    search_type="semantic",
                    include_citations=False,
                )
            )
            return media_id, results
        finally:
            uninstall_media_ingest_hook()
            indexer.stop()

    def test_semantic_search_returns_newly_ingested_document(self, media_db, tmp_path, monkeypatch):
        service = _make_real_service("memory")
        media_id, results = self._ingest_and_search(media_db, tmp_path, monkeypatch, service)

        assert results, "semantic search returned nothing for distinctive content"
        top = results[0]
        assert top.metadata["source_id"] == str(media_id)
        assert top.metadata["source_type"] == "media"
        assert top.metadata["title"] == "Quokka Manifesto"
        assert top.metadata.get("chunk_id"), "chunk_id must be present for _semantic_row"

    def test_v2_service_with_parallel_profile_indexes_and_searches(self, media_db, tmp_path, monkeypatch):
        """Regression: EnhancedRAGServiceV2 with parallel processing enabled
        used to crash on index_batch_optimized (imports of nonexistent
        enhanced_indexing_helpers functions); it must now index via the base
        optimized path."""
        from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
        from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2

        cfg = RAGConfig()
        cfg.embedding.model = "mock"
        cfg.embedding.device = "cpu"
        cfg.vector_store.type = "memory"
        cfg.vector_store.persist_directory = None
        cfg.chunking.chunk_size = 60
        cfg.chunking.chunk_overlap = 10
        cfg.search.enable_cache = False
        service = EnhancedRAGServiceV2(
            config=cfg,
            enable_parent_retrieval=False,
            enable_reranking=False,
            enable_parallel_processing=True,
        )
        media_id, results = self._ingest_and_search(media_db, tmp_path, monkeypatch, service)

        assert results
        assert results[0].metadata["source_id"] == str(media_id)

    def test_chroma_round_trip_persists_ingested_document(self, media_db, tmp_path, monkeypatch):
        pytest.importorskip("chromadb")
        persist_dir = tmp_path / "chromadb"
        service = _make_real_service("chroma", persist_dir)
        media_id, results = self._ingest_and_search(media_db, tmp_path, monkeypatch, service)

        assert results
        assert results[0].metadata["source_id"] == str(media_id)

        # A fresh service over the same persist dir must still find it.
        service2 = _make_real_service("chroma", persist_dir)
        results2 = asyncio.run(
            service2.search(
                "zanzibar quokka snorkeling manifesto",
                top_k=3,
                search_type="semantic",
                include_citations=False,
            )
        )
        assert results2
        assert results2[0].metadata["source_id"] == str(media_id)


# === Backfill (AC #3) ===

@pytest.mark.integration
@pytest.mark.skipif(not embeddings_rag_deps_installed(), reason="embeddings_rag deps not installed")
class TestBackfill:
    def test_backfill_media_notes_conversations_and_incremental_rerun(self, media_db, tmp_path):
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        # Pre-existing content, ingested with no hook installed.
        m1, _, _ = _add_media(media_db, title="Quokka Manifesto", content=DISTINCTIVE_CONTENT)
        m2, _, _ = _add_media(media_db, title="Tax Law Digest", content=DECOY_CONTENT)
        assert m1 and m2

        cha_db = CharactersRAGDB(tmp_path / "cha.db", "test-backfill")
        note_id = cha_db.add_note("Wombat Note", "Wombats dig unusually square burrows near eucalyptus groves.")
        conv_id = cha_db.add_conversation({"title": "Chat about pelicans"})
        cha_db.add_message({"conversation_id": conv_id, "sender": "user",
                            "content": "Do pelicans migrate across hemispheres?"})
        cha_db.add_message({"conversation_id": conv_id, "sender": "assistant",
                            "content": "Some pelican populations are migratory, yes."})

        service = _make_real_service("memory")
        indexing_db = RAGIndexingDB(tmp_path / "rag_indexing.db")

        summary = asyncio.run(
            backfill_semantic_index(
                media_db=media_db,
                chachanotes_db=cha_db,
                rag_service=service,
                indexing_db=indexing_db,
            )
        )
        assert summary["indexed"] == 4  # 2 media + 1 note + 1 conversation
        assert summary["failed"] == 0
        assert indexing_db.get_indexed_item_info(str(m1), "media") is not None
        assert indexing_db.get_indexed_item_info(note_id, "note") is not None
        assert indexing_db.get_indexed_item_info(conv_id, "conversation") is not None

        # Semantic search sees backfilled content from every source type.
        note_results = asyncio.run(
            service.search("square wombat burrows", top_k=3, search_type="semantic", include_citations=False)
        )
        assert note_results
        assert note_results[0].metadata["source_type"] == "note"

        conv_results = asyncio.run(
            service.search("pelicans migrate hemispheres", top_k=3, search_type="semantic", include_citations=False)
        )
        assert conv_results
        assert conv_results[0].metadata["source_type"] == "conversation"

        # Second run is incremental: nothing re-indexed.
        summary2 = asyncio.run(
            backfill_semantic_index(
                media_db=media_db,
                chachanotes_db=cha_db,
                rag_service=service,
                indexing_db=indexing_db,
            )
        )
        assert summary2["indexed"] == 0
        assert summary2["skipped"] == 4

    def test_backfill_unavailable_without_deps(self, media_db, tmp_path, monkeypatch):
        monkeypatch.setattr(ingestion_indexing, "embeddings_rag_deps_installed", lambda: False)
        summary = asyncio.run(backfill_semantic_index(media_db=media_db))
        assert summary["status"] == "unavailable"
        assert summary["indexed"] == 0


# === Shared service wiring ===

@pytest.mark.unit
class TestSharedRagService:
    def test_set_and_reset_shared_service(self):
        fake = FakeRAGService()
        ingestion_indexing.set_shared_rag_service(fake)
        try:
            assert ingestion_indexing.get_shared_rag_service() is fake
        finally:
            ingestion_indexing.reset_shared_rag_service()

    def test_chat_rag_events_uses_shared_service(self):
        from types import SimpleNamespace

        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
            get_or_initialize_rag_service,
        )

        fake = FakeRAGService()
        ingestion_indexing.set_shared_rag_service(fake)
        try:
            app = SimpleNamespace(_rag_service=None, config=None)
            service = asyncio.run(get_or_initialize_rag_service(app))
            assert service is fake
            assert app._rag_service is fake
        finally:
            ingestion_indexing.reset_shared_rag_service()
