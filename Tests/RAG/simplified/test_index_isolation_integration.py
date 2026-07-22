# Tests/RAG/simplified/test_index_isolation_integration.py
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
