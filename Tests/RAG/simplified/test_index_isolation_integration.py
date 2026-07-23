# Tests/RAG/simplified/test_index_isolation_integration.py
import asyncio

import pytest
from tldw_chatbook.RAG_Search.simplified.config import (
    RAGConfig, EmbeddingConfig, ChunkingConfig, VectorStoreConfig, SearchConfig,
)
from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import (
    fingerprinted_collection_name,
)
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService


def _chroma_cfg(persist_dir, **over):
    vs = dict(type="chroma", persist_directory=persist_dir,
              collection_name="default", distance_metric="cosine")
    vs.update(over.pop("vector_store", {}))
    return RAGConfig(
        embedding=EmbeddingConfig(model="mock", device="cpu"),
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=100, chunking_method="words"),
        vector_store=VectorStoreConfig(**vs),
        search=SearchConfig(enable_cache=False),
    )


@pytest.mark.requires_chromadb
def test_service_opens_fingerprinted_collection(chroma_persist_dir):
    cfg = _chroma_cfg(chroma_persist_dir)
    svc = RAGService(cfg)
    expected = fingerprinted_collection_name(cfg)
    assert svc.vector_store.collection_name == expected
    assert expected != "default"


@pytest.mark.requires_chromadb
def test_query_only_diff_shares_collection(chroma_persist_dir):
    a = _chroma_cfg(chroma_persist_dir)
    b = _chroma_cfg(chroma_persist_dir)
    b.search.default_top_k = 42
    assert (RAGService(a).vector_store.collection_name
            == RAGService(b).vector_store.collection_name)


@pytest.mark.requires_chromadb
def test_metric_and_model_diffs_fork_collection(chroma_persist_dir):
    base = RAGService(_chroma_cfg(chroma_persist_dir)).vector_store.collection_name
    metric = _chroma_cfg(chroma_persist_dir, vector_store={"distance_metric": "l2"})
    model = _chroma_cfg(chroma_persist_dir)
    model.embedding.model = "other-model"
    assert RAGService(metric).vector_store.collection_name != base
    assert RAGService(model).vector_store.collection_name != base


@pytest.mark.requires_chromadb
def test_created_collection_carries_provenance(chroma_persist_dir):
    cfg = _chroma_cfg(chroma_persist_dir)
    svc = RAGService(cfg)
    meta = svc.vector_store.collection.metadata  # forces get_or_create
    assert meta.get("fp") is not None
    assert meta.get("embedding_model") == "mock"
    assert meta.get("hnsw:space") == "cosine"  # existing key preserved


@pytest.mark.requires_chromadb
def test_fresh_fingerprinted_collection_reads_as_empty(chroma_persist_dir):
    """A freshly created fingerprinted collection (no docs) must trip the
    existing honest-empty probe -- locks the Task-2 contract, not new
    behavior."""
    from tldw_chatbook.RAG_Search.semantic_availability import semantic_index_is_empty

    cfg = _chroma_cfg(chroma_persist_dir)
    svc = RAGService(cfg)
    svc.vector_store.collection  # create, no docs
    assert asyncio.run(semantic_index_is_empty(svc)) is True


@pytest.mark.requires_chromadb
def test_ingest_and_query_resolve_same_collection(chroma_persist_dir):
    # Two services built from the SAME config must open the SAME collection --
    # the embed/query parity guard against divergent-model index misses.
    cfg = _chroma_cfg(chroma_persist_dir)
    ingest_side = RAGService(cfg).vector_store.collection_name
    query_side = RAGService(cfg).vector_store.collection_name
    assert ingest_side == query_side == fingerprinted_collection_name(cfg)


@pytest.mark.requires_chromadb
def test_get_collection_stats_on_populated_collection(chroma_persist_dir):
    """Locks the numpy peek()["embeddings"] fix in
    ChromaVectorStore.get_collection_stats.

    Before the fix, a bare truthiness check on chromadb's numpy embeddings
    array (returned by collection.peek()) raised "the truth value of an
    array with more than one element is ambiguous", which was silently
    swallowed by get_collection_stats()'s broad except and made it report a
    masked error payload (count=0, "error": ...) for ANY collection --
    including a populated one -- degrading health_check()/get_chunk_count()
    downstream. This test seeds real documents so count > 0 and asserts the
    stats payload is trustworthy, not just that construction doesn't raise.
    """
    cfg = _chroma_cfg(chroma_persist_dir)
    svc = RAGService(cfg)
    svc.vector_store.add(
        ids=["id0", "id1"],
        embeddings=[[0.1] * 8, [0.2] * 8],
        documents=["doc 0", "doc 1"],
        metadata=[{"doc_id": "d0"}, {"doc_id": "d1"}],
    )

    stats = svc.vector_store.get_collection_stats()

    assert isinstance(stats["count"], int) and not isinstance(stats["count"], bool)
    assert stats["count"] > 0
    assert not stats.get("error")


@pytest.mark.requires_chromadb
def test_collection_metadata_cannot_override_hnsw_space(chroma_persist_dir):
    """A stray 'hnsw:space' key in collection_metadata must NOT override the
    real distance metric. hnsw:space is index-determining, so the merge keeps
    it LAST (Qodo #771 review) -- provenance never emits it today, but this
    locks the invariant structurally against future callers."""
    from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore

    store = ChromaVectorStore(
        persist_directory=chroma_persist_dir,
        collection_name="metric_guard__test",
        distance_metric="cosine",
        collection_metadata={"hnsw:space": "l2", "fp": "bogus"},
    )
    col = store.collection  # forces get_or_create with the merged metadata
    # The real metric (cosine) must win over the injected l2.
    assert col.configuration_json["hnsw"]["space"] == "cosine"
