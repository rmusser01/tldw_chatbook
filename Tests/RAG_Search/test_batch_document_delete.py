"""Task 18 (F17): batched stale-chunk deletes for Chroma re-indexing.

``ChromaVectorStore.delete_documents`` replaces the one-``delete``-per-doc
stale-chunk sweep on re-index (Chroma's where-delete is a linear metadata
scan, so N changed docs cost N scans). Everything here is hermetic: a fake
collection records every ``delete()`` attempt, so no real Chroma client or
persisted server is ever required.

Covered behaviors (task brief, binding resolutions 1-4):

(a) 5,000 changed docs -> exactly ceil(5000/500) = 10 ``$in`` delete calls,
    each carrying at most 500 ids, all ids present across calls.
(b) Chroma build that rejects ``$in`` on delete -> per-doc loop (5,000
    scalar-wheres deletes), ONE warning logged at feature-detection time.
(c) A batch call failing after the probe succeeded -> per-doc fallback for
    that 500-id chunk only; later chunks stay batched; nothing raises.
(d) Empty input -> zero delete calls.
(e) Single doc -> exactly one call carrying that one id.

Plus caller wiring (``ingestion_indexing.index_entries``): one
``delete_documents`` call per ingestion batch, per-doc fallback for stores
without the batch API, and best-effort semantics (a raising batch delete
never aborts ingestion).
"""

import asyncio
from datetime import datetime, timezone

import pytest
from loguru import logger

from tldw_chatbook.RAG_Search.ingestion_indexing import IndexEntry
from tldw_chatbook.RAG_Search.simplified.data_models import IndexingResult
from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore

#: Maximum ids per ``$in`` delete, per the task's binding resolution.
MAX_IN_CHUNK = 500


# =============================================================================
# Hermetic Chroma collection double
# =============================================================================


class FakeChromaCollection:
    """Chroma collection double recording every ``delete()`` attempt.

    Failure knobs:

    - ``reject_in_where``: every ``$in``-shaped where raises (models a Chroma
      build without ``$in``-on-delete support -> feature-detect fallback).
    - ``poison_id``: any delete touching this id raises, batched or scalar
      (models one poison id inside an otherwise healthy batch).
    """

    def __init__(self, *, reject_in_where: bool = False, poison_id: str | None = None):
        self.attempts: list[dict] = []
        self.reject_in_where = reject_in_where
        self.poison_id = poison_id

    def _raises(self, where) -> bool:
        condition = (where or {}).get("doc_id")
        if isinstance(condition, dict) and "$in" in condition:
            return self.reject_in_where or self.poison_id in condition["$in"]
        return self.poison_id is not None and condition == self.poison_id

    def delete(self, where=None, ids=None):
        raised = self._raises(where)
        self.attempts.append({"where": where, "raised": raised})
        if raised:
            raise RuntimeError("fake chroma delete failure")

    # -- recording helpers -------------------------------------------------

    @property
    def in_attempts(self) -> list[dict]:
        """Successful ``$in``-shaped delete attempts."""
        return [a for a in self.attempts if self._in_ids(a) is not None and not a["raised"]]

    @property
    def scalar_attempts(self) -> list[dict]:
        """Attempts whose where was a bare ``{"doc_id": <id>}`` scalar."""
        return [a for a in self.attempts if self._in_ids(a) is None]

    @staticmethod
    def _in_ids(attempt) -> list[str] | None:
        condition = (attempt["where"] or {}).get("doc_id")
        if isinstance(condition, dict) and "$in" in condition:
            return condition["$in"]
        return None


def _store_with_collection(tmp_path, collection) -> ChromaVectorStore:
    """Build a ChromaVectorStore whose collection is the fake (no Chroma)."""
    store = ChromaVectorStore(persist_directory=tmp_path / "chroma")
    store._collection = collection
    return store


def _ids(count: int) -> list[str]:
    return [f"media_{i}" for i in range(count)]


# =============================================================================
# Store-level behavior
# =============================================================================


@pytest.mark.unit
class TestDeleteDocumentsBatching:
    def test_5000_docs_delete_in_exactly_10_batched_calls(self, tmp_path):
        """(a) N changed docs -> ceil(N/500) deletes, never N."""
        fake = FakeChromaCollection()
        store = _store_with_collection(tmp_path, fake)

        ids = _ids(5000)
        store.delete_documents(ids)

        in_attempts = fake.in_attempts
        assert len(fake.attempts) == 10, (
            "5,000 docs must produce exactly ceil(5000/500)=10 delete calls, "
            f"got {len(fake.attempts)}"
        )
        assert all(a in in_attempts for a in fake.attempts), (
            "every call must be a successful $in-shaped batch delete"
        )
        seen: list[str] = []
        for attempt in fake.attempts:
            in_list = FakeChromaCollection._in_ids(attempt)
            assert in_list is not None
            assert len(in_list) <= MAX_IN_CHUNK
            seen.extend(in_list)
        assert len(seen) == len(set(seen)) == 5000, "no id lost or duplicated"
        assert set(seen) == set(ids)

    def test_chunk_boundary_off_by_one(self, tmp_path):
        """501 docs -> two calls of 500 then 1 (exact chunking at 500)."""
        fake = FakeChromaCollection()
        store = _store_with_collection(tmp_path, fake)

        store.delete_documents(_ids(501))

        sizes = [
            len(FakeChromaCollection._in_ids(a)) for a in fake.attempts
        ]
        assert sizes == [500, 1]

    def test_empty_input_makes_zero_calls(self, tmp_path):
        """(d) Empty batch -> no delete call at all (not even a probe)."""
        fake = FakeChromaCollection()
        store = _store_with_collection(tmp_path, fake)

        store.delete_documents([])

        assert fake.attempts == []

    def test_single_doc_is_one_call_with_one_id(self, tmp_path):
        """(e) The common small case stays a single delete call."""
        fake = FakeChromaCollection()
        store = _store_with_collection(tmp_path, fake)

        store.delete_documents(["media_42"])

        assert len(fake.attempts) == 1
        assert fake.attempts[0]["where"] == {"doc_id": {"$in": ["media_42"]}}


@pytest.mark.unit
class TestDeleteDocumentsFeatureDetection:
    def test_in_rejection_falls_back_per_doc_with_one_time_warning(self, tmp_path):
        """(b) Chroma without $in-on-delete: per-doc loop, ONE warning."""
        fake = FakeChromaCollection(reject_in_where=True)
        store = _store_with_collection(tmp_path, fake)

        records: list = []
        sink = logger.add(lambda m: records.append(m), level="DEBUG")
        try:
            store.delete_documents(_ids(5000))
            # A second invocation must not warn again (detect ONCE, per store).
            store.delete_documents(_ids(3))
        finally:
            logger.remove(sink)

        # 5,000 scalar per-doc deletes for the first call + 3 for the second.
        assert len(fake.scalar_attempts) == 5003
        assert fake.in_attempts == [], "no $in-shaped delete may succeed"
        deleted = {a["where"]["doc_id"] for a in fake.scalar_attempts}
        assert deleted == set(_ids(5000)) | set(_ids(3))
        warnings = [
            m.record["message"]
            for m in records
            if m.record["level"].name == "WARNING" and "$in" in m.record["message"]
        ]
        assert len(warnings) == 1, (
            "feature-detection fallback must warn exactly once, not per call"
        )


@pytest.mark.unit
class TestDeleteDocumentsBatchFailureFallback:
    def test_poison_batch_falls_back_per_doc_within_that_batch_only(self, tmp_path):
        """(c) One failed batch retries only its own 500 ids one-by-one."""
        # 3 chunks of 500; the poison id sits in the SECOND chunk so the
        # feature-detect probe succeeds on chunk 1 before the failure.
        fake = FakeChromaCollection(poison_id="media_750")
        store = _store_with_collection(tmp_path, fake)

        records: list = []
        sink = logger.add(lambda m: records.append(m), level="DEBUG")
        try:
            store.delete_documents(_ids(1500))  # must not raise
        finally:
            logger.remove(sink)

        # Chunks 1 and 3: successful $in batch deletes of 500 ids each.
        in_attempts = fake.in_attempts
        assert len(in_attempts) == 2
        batched_ids = [
            i for a in in_attempts for i in FakeChromaCollection._in_ids(a)
        ]
        chunk3 = _ids(1500)[1000:]
        assert set(batched_ids) == set(_ids(500)) | set(chunk3), (
            "chunks before and after the failed one stay fully batched"
        )

        # The failed chunk was retried per-doc: 500 scalar attempts, of which
        # exactly the poison one failed.
        scalar = fake.scalar_attempts
        assert len(scalar) == 500
        second_chunk = _ids(1500)[500:1000]
        assert {a["where"]["doc_id"] for a in scalar} == set(second_chunk)
        assert sum(1 for a in scalar if a["raised"]) == 1

        # One failed batch call (the $in carrying the poison id) ...
        failed_in = [
            a
            for a in fake.attempts
            if FakeChromaCollection._in_ids(a) is not None and a["raised"]
        ]
        assert len(failed_in) == 1
        assert "media_750" in FakeChromaCollection._in_ids(failed_in[0])

        # ... one warning for the batch failure, one debug for the poison id.
        warnings = [
            m.record["message"]
            for m in records
            if m.record["level"].name == "WARNING"
        ]
        assert len(warnings) == 1
        debug_failures = [
            m.record["message"]
            for m in records
            if m.record["level"].name == "DEBUG"
            and "Stale-chunk delete failed" in m.record["message"]
            and "media_750" in m.record["message"]
        ]
        assert len(debug_failures) == 1

        # Feature detection is NOT reset by a transient batch failure: the
        # next call still leads with a batched $in delete.
        fake.attempts.clear()
        store.delete_documents(["media_9", "media_10"])
        assert len(fake.attempts) == 1
        assert fake.attempts[0]["where"] == {
            "doc_id": {"$in": ["media_9", "media_10"]}
        }


# =============================================================================
# Caller wiring (ingestion_indexing.index_entries)
# =============================================================================


class BatchAwareVectorStore:
    """Fake store exposing both the batch API and the legacy per-doc one."""

    def __init__(self):
        self.batch_calls: list[list[str]] = []
        self.per_doc_calls: list[str] = []

    def delete_documents(self, doc_ids):
        self.batch_calls.append(list(doc_ids))

    def delete_document(self, doc_id):
        self.per_doc_calls.append(doc_id)


class LegacyOnlyVectorStore:
    """Fake store without the batch API (InMemoryVectorStore shape)."""

    def __init__(self):
        self.per_doc_calls: list[str] = []

    def delete_document(self, doc_id):
        self.per_doc_calls.append(doc_id)


class ExplodingBatchVectorStore(BatchAwareVectorStore):
    """Batch API present but broken: ingestion must still proceed."""

    def delete_documents(self, doc_ids):
        super().delete_documents(doc_ids)
        raise RuntimeError("batch delete exploded")


class FakeService:
    def __init__(self, store):
        self.vector_store = store
        self.indexed: list[dict] = []

    async def index_batch_optimized(self, documents, show_progress=True, batch_size=32):
        self.indexed.extend(documents)
        return [
            IndexingResult(doc_id=d["id"], chunks_created=2, time_taken=0.0, success=True)
            for d in documents
        ]


def _entry(item_id: str) -> IndexEntry:
    return IndexEntry(
        item_id=item_id,
        item_type="media",
        last_modified=datetime.now(timezone.utc),
        document={
            "id": f"media_{item_id}",
            "content": "content",
            "title": "title",
            "metadata": {},
        },
    )


@pytest.mark.unit
class TestIngestionWiring:
    def test_one_batched_delete_call_per_ingestion_batch(self):
        from tldw_chatbook.RAG_Search.ingestion_indexing import index_entries

        store = BatchAwareVectorStore()
        service = FakeService(store)

        summary = asyncio.run(
            index_entries(service, None, [_entry(str(i)) for i in (1, 2, 3)])
        )

        assert summary["indexed"] == 3
        assert len(store.batch_calls) == 1, "one delete_documents call per batch"
        assert store.batch_calls[0] == ["media_1", "media_2", "media_3"]
        assert store.per_doc_calls == [], "per-doc API must not be used when batch exists"

    def test_batches_do_not_accumulate_across_index_entries_calls(self):
        from tldw_chatbook.RAG_Search.ingestion_indexing import index_entries

        store = BatchAwareVectorStore()
        service = FakeService(store)

        asyncio.run(index_entries(service, None, [_entry("1")]))
        asyncio.run(
            index_entries(service, None, [_entry(e) for e in ("2", "3")])
        )

        assert store.batch_calls == [["media_1"], ["media_2", "media_3"]]

    def test_store_without_batch_api_falls_back_to_per_doc_loop(self):
        from tldw_chatbook.RAG_Search.ingestion_indexing import index_entries

        store = LegacyOnlyVectorStore()
        service = FakeService(store)

        summary = asyncio.run(
            index_entries(service, None, [_entry(str(i)) for i in (1, 2)])
        )

        assert summary["indexed"] == 2
        assert store.per_doc_calls == ["media_1", "media_2"]

    def test_raising_batch_delete_never_aborts_ingestion(self):
        from tldw_chatbook.RAG_Search.ingestion_indexing import index_entries

        store = ExplodingBatchVectorStore()
        service = FakeService(store)

        summary = asyncio.run(
            index_entries(service, None, [_entry(str(i)) for i in (1, 2)])
        )

        assert summary["indexed"] == 2, "ingestion continues past a failed delete"
        assert summary["failed"] == 0
        assert [d["id"] for d in service.indexed] == ["media_1", "media_2"]
