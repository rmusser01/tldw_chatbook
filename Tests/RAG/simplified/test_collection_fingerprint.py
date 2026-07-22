# Tests/RAG/simplified/test_collection_fingerprint.py
import re
from dataclasses import replace

from tldw_chatbook.RAG_Search.simplified.config import (
    RAGConfig, EmbeddingConfig, ChunkingConfig, VectorStoreConfig,
)
from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import (
    FINGERPRINT_VERSION,
    fingerprint_collection,
    fingerprinted_collection_name,
    collection_provenance,
)

def _cfg(**vs):
    return RAGConfig(
        embedding=EmbeddingConfig(model="modelA", max_length=512),
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=100, chunking_method="words"),
        vector_store=VectorStoreConfig(type="memory", collection_name="default",
                                       distance_metric="cosine", **vs),
    )

def test_fingerprint_is_deterministic():
    assert fingerprint_collection(_cfg()) == fingerprint_collection(_cfg())

def test_str_and_int_chunk_size_hash_identically():
    a = _cfg()
    b = _cfg()
    b.chunking.chunk_size = "400"  # str from TOML
    assert fingerprint_collection(a) == fingerprint_collection(b)

def test_query_only_diff_shares_fingerprint():
    a = _cfg()
    b = _cfg()
    b.search.default_top_k = 999          # query-time, excluded
    b.search.enable_reranking = True      # query-time, excluded
    assert fingerprint_collection(a) == fingerprint_collection(b)

def test_index_field_diffs_change_fingerprint():
    base = fingerprint_collection(_cfg())
    m = _cfg(); m.embedding.model = "modelB"
    ch = _cfg(); ch.chunking.chunk_size = 401
    mx = _cfg(); mx.chunking.max_chunk_size = 1001
    metric = _cfg(); metric.vector_store.distance_metric = "l2"
    ml = _cfg(); ml.embedding.max_length = 256
    for c in (m, ch, mx, metric, ml):
        assert fingerprint_collection(c) != base

def test_name_is_valid_chroma_name():
    name = fingerprinted_collection_name(_cfg())
    assert name.startswith("default__")
    assert 3 <= len(name) <= 63
    assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]", name)
    assert ".." not in name

def test_long_base_truncated_but_suffix_preserved():
    c = _cfg()
    c.vector_store.collection_name = "x" * 200
    name = fingerprinted_collection_name(c)
    assert len(name) <= 63
    assert name.endswith("__" + fingerprint_collection(c))

def test_provenance_carries_version_and_fields():
    p = collection_provenance(_cfg(), source="legacy-adopted", verified=False)
    assert p["fp_version"] == FINGERPRINT_VERSION
    assert p["fp"] == fingerprint_collection(_cfg())
    assert p["embedding_model"] == "modelA"
    assert p["distance_metric"] == "cosine"
    assert p["source"] == "legacy-adopted"
    assert p["verified"] is False
