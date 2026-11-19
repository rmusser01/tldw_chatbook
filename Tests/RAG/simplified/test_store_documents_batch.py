"""Batch-store tests for ``store_documents_batch`` (task-34398).

Before task-34398 the helper issued one ``_store_chunks`` call (executor hop
+ vector-store transaction) PER DOCUMENT. It now concatenates every
document's payload and issues ONE store add per batch. The original
per-document failure isolation is preserved: if the single batch add fails,
the payloads are retried document-by-document so only the documents that
actually fail to store are marked failed.

The fake RAG service records every ``_store_chunks`` payload so tests can
count store transactions and compare the final stored content against the
old per-document call shape.
"""

import numpy as np
import pytest

from tldw_chatbook.RAG_Search.simplified.indexing_helpers import (
    store_documents_batch,
)


class FakeRagService:
    """Records ``_store_chunks`` payloads; optionally fails some of them."""

    def __init__(self, fail_predicate=None):
        self.store_calls = []
        self._docs_indexed = 0
        self._total_chunks_created = 0
        self._fail_predicate = fail_predicate

    async def _store_chunks(self, ids, embeddings, documents, metadata):
        ids = list(ids)
        if self._fail_predicate is not None and self._fail_predicate(ids):
            raise RuntimeError(f"store exploded for {ids[0]}")
        self.store_calls.append(
            {
                "ids": ids,
                "embeddings": np.asarray(embeddings, dtype=np.float32),
                "documents": list(documents),
                "metadata": list(metadata),
            }
        )


def _three_doc_fixture(dim: int = 4):
    """3 documents x 2 chunks, with contiguous embeddings and matching
    ``doc_chunk_info`` — mirroring the Phase 1/2 contract."""
    documents = [
        {"id": f"doc_{c}", "title": f"Doc {c}", "content": f"content {c}", "metadata": {"source": "test"}}
        for c in "abc"
    ]
    doc_chunk_info = []
    chunk_texts = []
    for i, doc in enumerate(documents):
        chunks = [
            {"text": f"{doc['id']} chunk {j}", "start_char": j * 10, "end_char": j * 10 + 10, "word_count": 2}
            for j in range(2)
        ]
        chunk_texts.extend(c["text"] for c in chunks)
        doc_chunk_info.append(
            {
                "doc_idx": i,
                "chunk_start": i * 2,
                "chunk_count": 2,
                "chunks": chunks,
            }
        )
    all_embeddings = np.arange(len(chunk_texts) * dim, dtype=np.float32).reshape(
        len(chunk_texts), dim
    )
    return documents, doc_chunk_info, all_embeddings


@pytest.mark.unit
class TestStoreDocumentsBatchSingleAdd:
    async def test_n_documents_issue_one_store_add(self):
        """3 documents produce exactly ONE ``_store_chunks`` call whose
        payload is the concatenation of the per-document payloads."""
        service = FakeRagService()
        documents, doc_chunk_info, embeddings = _three_doc_fixture()

        results = await store_documents_batch(
            service, documents, doc_chunk_info, embeddings, batch_start_time=0.0
        )

        assert len(service.store_calls) == 1, (
            f"expected one batch store add, got {len(service.store_calls)}"
        )
        call = service.store_calls[0]
        assert call["ids"] == [
            "doc_a_chunk_0", "doc_a_chunk_1",
            "doc_b_chunk_0", "doc_b_chunk_1",
            "doc_c_chunk_0", "doc_c_chunk_1",
        ]
        assert call["documents"] == [
            "doc_a chunk 0", "doc_a chunk 1",
            "doc_b chunk 0", "doc_b chunk 1",
            "doc_c chunk 0", "doc_c chunk 1",
        ]
        assert call["embeddings"].shape == (6, 4)
        np.testing.assert_array_equal(call["embeddings"], embeddings)

        assert all(r.success for r in results), [r.error for r in results]
        assert len(results) == 3
        assert service._docs_indexed == 3
        assert service._total_chunks_created == 6
        # Metadata carries each row's doc identity, in order.
        assert [m["doc_id"] for m in call["metadata"]] == [
            "doc_a", "doc_a", "doc_b", "doc_b", "doc_c", "doc_c",
        ]

    async def test_stored_content_matches_per_document_calls(self):
        """The one batch add stores byte-identical content to what the old
        per-document loop would have stored (three separate calls)."""
        service = FakeRagService()
        documents, doc_chunk_info, embeddings = _three_doc_fixture()
        await store_documents_batch(
            service, documents, doc_chunk_info, embeddings, batch_start_time=0.0
        )
        batched = service.store_calls[0]

        per_doc = FakeRagService()
        for doc_info in doc_chunk_info:
            i = doc_info["doc_idx"]
            await per_doc._store_chunks(
                [f"{documents[i]['id']}_chunk_{j}" for j in range(2)],
                embeddings[i * 2 : i * 2 + 2],
                [f"{documents[i]['id']} chunk {j}" for j in range(2)],
                [{"doc_id": documents[i]["id"], "chunk_index": j} for j in range(2)],
            )

        assert batched["ids"] == [i for c in per_doc.store_calls for i in c["ids"]]
        assert batched["documents"] == [
            d for c in per_doc.store_calls for d in c["documents"]
        ]
        np.testing.assert_array_equal(
            batched["embeddings"],
            np.concatenate([c["embeddings"] for c in per_doc.store_calls]),
        )

    async def test_all_chunks_failed_document_is_skipped(self):
        """A document whose every chunk failed embedding produces a failed
        result and contributes nothing to the store add."""
        service = FakeRagService()
        documents, doc_chunk_info, embeddings = _three_doc_fixture()

        results = await store_documents_batch(
            service,
            documents,
            doc_chunk_info,
            embeddings,
            batch_start_time=0.0,
            failed_embedding_indices=[2, 3],  # all of doc_b's chunks
        )

        assert len(service.store_calls) == 1
        call = service.store_calls[0]
        assert "doc_b_chunk_0" not in call["ids"]
        assert "doc_b_chunk_1" not in call["ids"]
        assert call["embeddings"].shape == (4, 4)

        by_doc = {r.doc_id: r for r in results}
        assert by_doc["doc_b"].success is False
        assert "All chunks failed" in by_doc["doc_b"].error
        assert by_doc["doc_a"].success and by_doc["doc_c"].success
        assert service._docs_indexed == 2

    async def test_no_preparable_documents_short_circuits(self):
        """Nothing storable -> no store call, one failure per document."""
        service = FakeRagService()
        documents, doc_chunk_info, embeddings = _three_doc_fixture()

        results = await store_documents_batch(
            service,
            documents,
            doc_chunk_info,
            embeddings,
            batch_start_time=0.0,
            failed_embedding_indices=[0, 1, 2, 3, 4, 5],
        )

        assert service.store_calls == []
        assert len(results) == 3
        assert all(not r.success for r in results)


@pytest.mark.unit
class TestStoreDocumentsBatchFailureIsolation:
    async def test_batch_failure_falls_back_to_per_document(self):
        """A failing batch add is retried per document so the whole batch is
        not marked failed when only the transport hiccuped once."""
        service = FakeRagService()  # never fails on retry (predicate None)
        documents, doc_chunk_info, embeddings = _three_doc_fixture()
        # Fail only the FIRST (batch) call.
        real_store = service._store_chunks
        first_call = {"done": False}

        async def flaky_store(ids, embeddings, documents_, metadata):
            if not first_call["done"]:
                first_call["done"] = True
                raise RuntimeError("transient batch failure")
            await real_store(ids, embeddings, documents_, metadata)

        service._store_chunks = flaky_store

        results = await store_documents_batch(
            service, documents, doc_chunk_info, embeddings, batch_start_time=0.0
        )

        assert len(service.store_calls) == 3  # one per-document retry each
        assert all(r.success for r in results), [r.error for r in results]
        assert service._docs_indexed == 3
        assert service._total_chunks_created == 6

    async def test_single_bad_document_fails_alone(self):
        """A document whose store deterministically fails is the ONLY failure,
        matching the old per-document failure granularity."""
        service = FakeRagService(
            fail_predicate=lambda ids: any(i.startswith("doc_b_") for i in ids)
        )
        documents, doc_chunk_info, embeddings = _three_doc_fixture()

        results = await store_documents_batch(
            service, documents, doc_chunk_info, embeddings, batch_start_time=0.0
        )

        # 1 failed batch attempt (unrecorded) + 3 per-document retries, of
        # which doc_b fails again (unrecorded): a and c land in the store.
        assert len(service.store_calls) == 2
        assert [i for c in service.store_calls for i in c["ids"]] == [
            "doc_a_chunk_0", "doc_a_chunk_1", "doc_c_chunk_0", "doc_c_chunk_1",
        ]
        by_doc = {r.doc_id: r for r in results}
        assert by_doc["doc_b"].success is False
        assert "doc_b_chunk_0" in by_doc["doc_b"].error
        assert by_doc["doc_a"].success and by_doc["doc_c"].success
        assert service._docs_indexed == 2
        assert service._total_chunks_created == 4
