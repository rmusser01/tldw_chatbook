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


def _add_media(
    db,
    *,
    title="Doc",
    content="hello world content",
    media_type="document",
    overwrite=True,
    url=None,
):
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
        self.close_calls = 0

    def close(self):
        """task-640 item 2: mirrors EnhancedRAGServiceV2's real close(), so
        tests can assert get_shared_rag_service() releases a build it
        discards due to a construction race."""
        self.close_calls += 1

    async def index_batch_optimized(self, documents, show_progress=True, batch_size=32):
        if self.fail:
            raise RuntimeError("boom: embedding backend exploded")
        self.indexed_docs.extend(documents)
        return [
            IndexingResult(
                doc_id=d["id"], chunks_created=2, time_taken=0.0, success=True
            )
            for d in documents
        ]

    async def search(self, *args, **kwargs):
        """Search seam: the shared-service resolver only accepts runtimes
        with a callable ``search`` (task-250 availability validation)."""
        return []


@pytest.fixture
def fake_service():
    return FakeRAGService()


@pytest.fixture
def indexer(fake_service, indexing_db):
    idx = IngestionIndexer(rag_service=fake_service, indexing_db=indexing_db)
    yield idx
    idx.stop()


def _entry(
    item_id="1",
    *,
    content=DISTINCTIVE_CONTENT,
    title="Quokka Manifesto",
    last_modified=None,
    item_type="media",
):
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
            media_id, media_uuid, _msg = _add_media(
                media_db, content="post commit visible"
            )
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
            media_id, _, message = _add_media(
                media_db, content="dup content", overwrite=False
            )
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
        def cb(db, mid, muuid):
            return calls.append(mid)
        media_db_module.register_media_post_ingest_callback(cb)
        media_db_module.unregister_media_post_ingest_callback(cb)

        _add_media(media_db, content="nobody listening")
        assert calls == []

    def test_callback_fires_for_content_update(self, media_db):
        calls = []
        media_id, _, _ = _add_media(
            media_db, content="version one", url="https://example.com/x"
        )

        media_db_module.register_media_post_ingest_callback(
            lambda db, mid, muuid: calls.append(mid)
        )
        try:
            updated_id, _, _ = _add_media(
                media_db,
                content="version two",
                url="https://example.com/x",
                overwrite=True,
            )
        finally:
            media_db_module._MEDIA_POST_INGEST_CALLBACKS.clear()

        assert updated_id == media_id
        assert calls == [media_id]


# === Availability gate (AC #5) ===


@pytest.mark.unit
class TestAvailabilityGate:
    def test_no_indexing_attempted_when_deps_missing(self, media_db, monkeypatch):
        monkeypatch.setattr(
            ingestion_indexing, "embeddings_rag_deps_installed", lambda: False
        )

        def _fail_if_touched():
            raise AssertionError("indexer must not be touched when deps are missing")

        monkeypatch.setattr(
            ingestion_indexing, "get_ingestion_indexer", _fail_if_touched
        )

        install_media_ingest_hook()
        media_id, _, _ = _add_media(media_db, content="no deps, still ingests fine")

        assert media_id is not None
        assert media_db.get_media_by_id(media_id) is not None

    def test_semantic_indexing_available_false_when_deps_missing(self, monkeypatch):
        monkeypatch.setattr(
            ingestion_indexing, "embeddings_rag_deps_installed", lambda: False
        )
        assert semantic_indexing_available() is False

    def test_config_kill_switch_disables_indexing(self, media_db, monkeypatch):
        monkeypatch.setattr(
            ingestion_indexing, "embeddings_rag_deps_installed", lambda: True
        )
        monkeypatch.setattr(
            ingestion_indexing,
            "get_cli_setting",
            lambda section, key, default=None: (
                {"indexing": {"enabled": False}}
                if (section, key) == ("AppRAGSearchConfig", "rag")
                else default
            ),
        )

        def _fail_if_touched():
            raise AssertionError("indexer must not be touched when disabled by config")

        monkeypatch.setattr(
            ingestion_indexing, "get_ingestion_indexer", _fail_if_touched
        )

        install_media_ingest_hook()
        media_id, _, _ = _add_media(media_db, content="disabled by config")
        assert media_id is not None

    def test_install_hook_is_idempotent(self, media_db, monkeypatch):
        submitted = []

        class OneShotIndexer:
            def submit(self, entry):
                submitted.append(entry)
                return True

        monkeypatch.setattr(
            ingestion_indexing, "get_ingestion_indexer", lambda: OneShotIndexer()
        )
        monkeypatch.setattr(
            ingestion_indexing, "embeddings_rag_deps_installed", lambda: True
        )

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
        conv = {
            "id": "conv-1",
            "title": "Chat about quokkas",
            "last_modified": "2026-07-16T10:00:00Z",
        }
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

    def test_updated_item_is_reindexed_with_stale_chunk_delete(
        self, indexer, fake_service, indexing_db
    ):
        old = datetime.now(timezone.utc) - timedelta(hours=1)
        indexing_db.mark_item_indexed("33", "media", last_modified=old, chunk_count=2)

        indexer.submit(_entry("33", last_modified=datetime.now(timezone.utc)))
        assert indexer.wait_until_idle(timeout=10)

        assert [d["id"] for d in fake_service.indexed_docs] == ["media_33"]
        assert "media_33" in fake_service.vector_store.deleted

    def test_failure_is_recorded_and_worker_survives(
        self, indexer, fake_service, indexing_db
    ):
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
            async def index_batch_optimized(
                self, documents, show_progress=True, batch_size=32
            ):
                await asyncio.to_thread(release.wait, 10)
                return await super().index_batch_optimized(
                    documents, show_progress, batch_size
                )

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
@pytest.mark.skipif(
    not embeddings_rag_deps_installed(), reason="embeddings_rag deps not installed"
)
class TestEndToEndSemanticSearch:
    def _ingest_and_search(self, media_db, tmp_path, monkeypatch, service):
        indexer = IngestionIndexer(
            rag_service=service,
            indexing_db=RAGIndexingDB(tmp_path / "rag_indexing.db"),
        )
        monkeypatch.setattr(
            ingestion_indexing, "get_ingestion_indexer", lambda: indexer
        )
        install_media_ingest_hook()
        try:
            _add_media(media_db, title="Tax Law Digest", content=DECOY_CONTENT)
            media_id, _, _ = _add_media(
                media_db, title="Quokka Manifesto", content=DISTINCTIVE_CONTENT
            )
            assert media_id is not None
            assert indexer.wait_until_idle(timeout=60), (
                "background indexing did not finish"
            )
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

    def test_semantic_search_returns_newly_ingested_document(
        self, media_db, tmp_path, monkeypatch
    ):
        service = _make_real_service("memory")
        media_id, results = self._ingest_and_search(
            media_db, tmp_path, monkeypatch, service
        )

        assert results, "semantic search returned nothing for distinctive content"
        top = results[0]
        assert top.metadata["source_id"] == str(media_id)
        assert top.metadata["source_type"] == "media"
        assert top.metadata["title"] == "Quokka Manifesto"
        assert top.metadata.get("chunk_id"), (
            "chunk_id must be present for _semantic_row"
        )

    def test_v2_service_with_parallel_profile_indexes_and_searches(
        self, media_db, tmp_path, monkeypatch
    ):
        """Regression: EnhancedRAGServiceV2 with parallel processing enabled
        used to crash on index_batch_optimized (imports of nonexistent
        enhanced_indexing_helpers functions); it must now index via the base
        optimized path."""
        from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
        from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import (
            EnhancedRAGServiceV2,
        )

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
        media_id, results = self._ingest_and_search(
            media_db, tmp_path, monkeypatch, service
        )

        assert results
        assert results[0].metadata["source_id"] == str(media_id)

    def test_chroma_round_trip_persists_ingested_document(
        self, media_db, tmp_path, monkeypatch
    ):
        pytest.importorskip("chromadb")
        persist_dir = tmp_path / "chromadb"
        service = _make_real_service("chroma", persist_dir)
        media_id, results = self._ingest_and_search(
            media_db, tmp_path, monkeypatch, service
        )

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
@pytest.mark.skipif(
    not embeddings_rag_deps_installed(), reason="embeddings_rag deps not installed"
)
class TestBackfill:
    def test_backfill_media_notes_conversations_and_incremental_rerun(
        self, media_db, tmp_path
    ):
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        # Pre-existing content, ingested with no hook installed.
        m1, _, _ = _add_media(
            media_db, title="Quokka Manifesto", content=DISTINCTIVE_CONTENT
        )
        m2, _, _ = _add_media(media_db, title="Tax Law Digest", content=DECOY_CONTENT)
        assert m1 and m2

        cha_db = CharactersRAGDB(tmp_path / "cha.db", "test-backfill")
        note_id = cha_db.add_note(
            "Wombat Note",
            "Wombats dig unusually square burrows near eucalyptus groves.",
        )
        conv_id = cha_db.add_conversation({"title": "Chat about pelicans"})
        cha_db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "user",
                "content": "Do pelicans migrate across hemispheres?",
            }
        )
        cha_db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "assistant",
                "content": "Some pelican populations are migratory, yes.",
            }
        )

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
            service.search(
                "square wombat burrows",
                top_k=3,
                search_type="semantic",
                include_citations=False,
            )
        )
        assert note_results
        assert note_results[0].metadata["source_type"] == "note"

        conv_results = asyncio.run(
            service.search(
                "pelicans migrate hemispheres",
                top_k=3,
                search_type="semantic",
                include_citations=False,
            )
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
        monkeypatch.setattr(
            ingestion_indexing, "embeddings_rag_deps_installed", lambda: False
        )
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

    def test_peek_shared_service_never_creates(self):
        """peek returns the existing service or None; it must not construct one (task-251)."""
        ingestion_indexing.reset_shared_rag_service()
        assert ingestion_indexing.peek_shared_rag_service() is None

        fake = FakeRAGService()
        ingestion_indexing.set_shared_rag_service(fake)
        try:
            assert ingestion_indexing.peek_shared_rag_service() is fake
        finally:
            ingestion_indexing.reset_shared_rag_service()
        assert ingestion_indexing.peek_shared_rag_service() is None

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


@pytest.mark.unit
class TestSharedRagServiceLockDeadlock:
    """task-641: get_shared_rag_service() must never hold
    ``_shared_service_lock`` across the blocking ``create_rag_service()``
    call (which can trigger real network I/O, e.g. a HuggingFace model
    download) -- otherwise any concurrent lock-taking caller
    (``reset_shared_rag_service`` / ``set_shared_rag_service``, both
    reachable from the main/UI thread via
    ``active_config.set_active_profile`` and the Settings-screen
    save/Backfill/Clone paths) blocks for the full duration of the stalled
    construction, which is exactly the 6+ minute total-app-freeze the UAT
    ``sample(1)`` stack capture showed (main thread parked in
    ``PyThread_acquire_lock_timed`` while a worker thread sat in a raw
    ``select()`` on a stalled socket).
    """

    def _patch_construction(self, monkeypatch, build_fn):
        import tldw_chatbook.RAG_Search.simplified as simplified_pkg
        import tldw_chatbook.RAG_Search.simplified.active_config as active_config

        monkeypatch.setattr(simplified_pkg, "create_rag_service", build_fn)
        # resolve_active_rag_config() is evaluated as part of resolving what
        # to build -- fake it so this test never touches a real
        # ConfigProfileManager / on-disk profiles dir (same rationale as
        # Tests/RAG/test_first_run_import.py's lock-ordering regression test).
        monkeypatch.setattr(active_config, "resolve_active_rag_config", lambda **kwargs: object())

    def test_reset_does_not_block_on_in_flight_construction(self, monkeypatch):
        """RED for task-641: a reset/set-active call concurrent with a slow
        (simulated stalled-network) construction must complete promptly
        instead of waiting on the lock construction currently holds."""
        ingestion_indexing.reset_shared_rag_service()
        entered_construction = threading.Event()
        release_construction = threading.Event()

        def _slow_create_rag_service(**kwargs):
            entered_construction.set()
            # Stand-in for the stalled HuggingFace CloudFront socket read
            # the UAT sample(1) capture showed.
            release_construction.wait(timeout=5)
            return FakeRAGService()

        self._patch_construction(monkeypatch, _slow_create_rag_service)

        builder = threading.Thread(
            target=ingestion_indexing.get_shared_rag_service, daemon=True
        )
        builder.start()
        try:
            assert entered_construction.wait(timeout=5), (
                "construction never started"
            )

            # While construction is still blocked "mid-download", a
            # concurrent reset (as fired from the main/UI thread by
            # Backfill's save path / Clone / Set active) must not queue
            # up behind it.
            start = time.monotonic()
            ingestion_indexing.reset_shared_rag_service()
            elapsed = time.monotonic() - start
            assert elapsed < 1.0, (
                f"reset_shared_rag_service() blocked for {elapsed:.2f}s on "
                "in-flight construction -- this is the task-641 deadlock"
            )
        finally:
            release_construction.set()
            builder.join(timeout=5)
            assert not builder.is_alive()
            ingestion_indexing.reset_shared_rag_service()

    def test_concurrent_callers_construct_exactly_once(self, monkeypatch):
        """Task-249 invariant preserved by the task-641 fix: two threads
        racing get_shared_rag_service() with no service yet installed must
        still trigger construction at most once -- the second caller waits
        for (and reuses) the first's result rather than paying the
        construction cost twice. This is what
        ``_shared_service_build_lock`` (separate from ``_shared_service_
        lock``, so it never blocks reset/set) is for."""
        ingestion_indexing.reset_shared_rag_service()
        calls = []
        calls_lock = threading.Lock()

        def _counting_create_rag_service(**kwargs):
            with calls_lock:
                calls.append(1)
            time.sleep(0.05)
            return FakeRAGService()

        self._patch_construction(monkeypatch, _counting_create_rag_service)

        results = []

        def _call():
            results.append(ingestion_indexing.get_shared_rag_service())

        t1 = threading.Thread(target=_call, daemon=True)
        t2 = threading.Thread(target=_call, daemon=True)
        t1.start()
        t2.start()
        try:
            t1.join(timeout=5)
            t2.join(timeout=5)
            assert not t1.is_alive() and not t2.is_alive()
            assert len(calls) == 1, f"construction ran {len(calls)} times, expected 1"
            assert len(results) == 2
            assert results[0] is not None and results[0] is results[1]
        finally:
            ingestion_indexing.reset_shared_rag_service()

    def test_reset_racing_in_flight_construction_discards_the_stale_build(
        self, monkeypatch
    ):
        """task-641 AC#3 (no double-construction leak on a construction/reset
        race): if reset_shared_rag_service() lands while a build is already
        past the lock and mid-``create_rag_service()``, that build must be
        discarded at swap time rather than quietly resurrecting a since-
        superseded profile immediately after the reset -- so at most one
        instance is EVER installed as the shared singleton, and it is never
        the stale, pre-reset one."""
        ingestion_indexing.reset_shared_rag_service()
        entered_construction = threading.Event()
        release_construction = threading.Event()

        def _slow_create_rag_service(**kwargs):
            entered_construction.set()
            release_construction.wait(timeout=5)
            return FakeRAGService()

        self._patch_construction(monkeypatch, _slow_create_rag_service)

        results = []
        builder = threading.Thread(
            target=lambda: results.append(ingestion_indexing.get_shared_rag_service()),
            daemon=True,
        )
        builder.start()
        try:
            assert entered_construction.wait(timeout=5), (
                "construction never started"
            )

            # Reset while the build above is still in flight.
            ingestion_indexing.reset_shared_rag_service()

            release_construction.set()
            builder.join(timeout=5)
            assert not builder.is_alive()

            # The in-flight build's own result was discarded (it reflects a
            # since-superseded generation): its caller sees None, and the
            # stale instance was NEVER installed as the shared singleton.
            assert results == [None]
            assert ingestion_indexing.peek_shared_rag_service() is None
        finally:
            release_construction.set()
            ingestion_indexing.reset_shared_rag_service()

    def test_reset_racing_in_flight_construction_closes_the_discarded_build(
        self, monkeypatch
    ):
        """task-640 item 2: a build discarded because a concurrent reset
        superseded its generation must have close() called on it (releasing
        whatever real resources -- thread pool, embeddings, vector store,
        DB connection pools -- EnhancedRAGServiceV2.close() frees) instead
        of just being dropped for GC."""
        ingestion_indexing.reset_shared_rag_service()
        entered_construction = threading.Event()
        release_construction = threading.Event()
        built_holder = []

        def _slow_create_rag_service(**kwargs):
            entered_construction.set()
            release_construction.wait(timeout=5)
            built = FakeRAGService()
            built_holder.append(built)
            return built

        self._patch_construction(monkeypatch, _slow_create_rag_service)

        builder = threading.Thread(
            target=ingestion_indexing.get_shared_rag_service, daemon=True
        )
        builder.start()
        try:
            assert entered_construction.wait(timeout=5), (
                "construction never started"
            )
            ingestion_indexing.reset_shared_rag_service()
            release_construction.set()
            builder.join(timeout=5)
            assert not builder.is_alive()

            assert len(built_holder) == 1
            assert built_holder[0].close_calls == 1, (
                "the build discarded by the concurrent reset was never "
                "closed -- its resources (thread pool/embeddings/vector "
                "store/DB connection pools) leak"
            )
        finally:
            release_construction.set()
            ingestion_indexing.reset_shared_rag_service()

    def test_concurrent_set_racing_in_flight_construction_closes_the_discarded_build(
        self, monkeypatch
    ):
        """task-640 item 2, the other discard branch: an injected
        set_shared_rag_service() winning while a build is in flight must
        also close the losing (discarded) build -- and must NOT close the
        winner it just installed."""
        ingestion_indexing.reset_shared_rag_service()
        entered_construction = threading.Event()
        release_construction = threading.Event()
        built_holder = []

        def _slow_create_rag_service(**kwargs):
            entered_construction.set()
            release_construction.wait(timeout=5)
            built = FakeRAGService()
            built_holder.append(built)
            return built

        self._patch_construction(monkeypatch, _slow_create_rag_service)

        results = []
        builder = threading.Thread(
            target=lambda: results.append(ingestion_indexing.get_shared_rag_service()),
            daemon=True,
        )
        builder.start()
        winner = FakeRAGService()
        try:
            assert entered_construction.wait(timeout=5), (
                "construction never started"
            )
            ingestion_indexing.set_shared_rag_service(winner)
            release_construction.set()
            builder.join(timeout=5)
            assert not builder.is_alive()

            assert results == [winner]
            assert len(built_holder) == 1
            assert built_holder[0].close_calls == 1, (
                "the build discarded because an injected "
                "set_shared_rag_service() already won was never closed"
            )
            assert winner.close_calls == 0, (
                "the WINNING (installed) service must never be closed just "
                "because a concurrent build lost the race"
            )
        finally:
            release_construction.set()
            ingestion_indexing.reset_shared_rag_service()

    def test_config_resolution_runs_before_the_build_lock_not_serialized_behind_it(
        self, monkeypatch
    ):
        """task-640 item 1: resolve_active_rag_config()/_configured_profile()
        must run BEFORE _shared_service_build_lock is acquired at all, so a
        second caller's config resolution is never serialized behind a slow
        in-flight build holding that lock."""
        import tldw_chatbook.RAG_Search.simplified as simplified_pkg
        import tldw_chatbook.RAG_Search.simplified.active_config as active_config

        ingestion_indexing.reset_shared_rag_service()
        entered_construction = threading.Event()
        release_construction = threading.Event()

        def _slow_create_rag_service(**kwargs):
            entered_construction.set()
            release_construction.wait(timeout=5)
            return FakeRAGService()

        monkeypatch.setattr(simplified_pkg, "create_rag_service", _slow_create_rag_service)

        resolve_calls = []
        resolve_calls_lock = threading.Lock()
        second_resolve_seen = threading.Event()

        def _tracking_resolve(**kwargs):
            with resolve_calls_lock:
                resolve_calls.append(1)
                count = len(resolve_calls)
            if count == 2:
                second_resolve_seen.set()
            return object()

        monkeypatch.setattr(active_config, "resolve_active_rag_config", _tracking_resolve)

        t1 = threading.Thread(
            target=ingestion_indexing.get_shared_rag_service, daemon=True
        )
        t1.start()
        try:
            assert entered_construction.wait(timeout=5), (
                "construction never started"
            )
            # t1 is now stuck mid-build, holding _shared_service_build_lock
            # (its own config resolution already happened before that).
            t2 = threading.Thread(
                target=ingestion_indexing.get_shared_rag_service, daemon=True
            )
            t2.start()
            try:
                assert second_resolve_seen.wait(timeout=2), (
                    "the second caller's config resolution was blocked "
                    "behind the in-flight build's _shared_service_build_"
                    "lock -- config resolution must run BEFORE that lock "
                    "is acquired (task-640 item 1)"
                )
            finally:
                release_construction.set()
                t2.join(timeout=5)
                assert not t2.is_alive()
        finally:
            release_construction.set()
            t1.join(timeout=5)
            assert not t1.is_alive()
            ingestion_indexing.reset_shared_rag_service()
