# Tests/RAG/simplified/test_collection_fingerprint.py
import re

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
    co = _cfg(); co.chunking.chunk_overlap = 101
    mx = _cfg(); mx.chunking.max_chunk_size = 1001
    metric = _cfg(); metric.vector_store.distance_metric = "l2"
    ml = _cfg(); ml.embedding.max_length = 256
    method = _cfg(); method.chunking.chunking_method = "sentences"
    minc = _cfg(); minc.chunking.min_chunk_size = 60
    epr = _cfg(); epr.chunking.enable_parent_retrieval = not epr.chunking.enable_parent_retrieval
    psm = _cfg(); psm.chunking.parent_size_multiplier = psm.chunking.parent_size_multiplier + 1
    pst = _cfg(); pst.chunking.preserve_structure = not pst.chunking.preserve_structure
    cla = _cfg(); cla.chunking.clean_artifacts = not cla.chunking.clean_artifacts
    ptb = _cfg(); ptb.chunking.preserve_tables = not ptb.chunking.preserve_tables
    for c in (m, ch, co, mx, metric, ml, method, minc, epr, psm, pst, cla, ptb):
        assert fingerprint_collection(c) != base

def test_query_time_field_diffs_do_not_change_fingerprint():
    base = fingerprint_collection(_cfg())
    bs = _cfg(); bs.embedding.batch_size = bs.embedding.batch_size + 1
    ha = _cfg(); ha.search.hybrid_alpha = 0.1 if ha.search.hybrid_alpha != 0.1 else 0.2
    cs = _cfg(); cs.search.citation_style = "footnote" if cs.search.citation_style != "footnote" else "none"
    for c in (bs, ha, cs):
        assert fingerprint_collection(c) == base

def test_non_ascii_collection_name_is_stripped_to_ascii():
    for raw in ("café-notes", "北京_docs"):
        c = _cfg()
        c.vector_store.collection_name = raw
        name = fingerprinted_collection_name(c)
        assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]", name)
        assert 3 <= len(name) <= 63

def test_internal_double_dot_is_collapsed():
    for raw in ("foo..bar", "a..b..c"):
        c = _cfg()
        c.vector_store.collection_name = raw
        name = fingerprinted_collection_name(c)
        assert ".." not in name

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

def test_double_negative_chunk_size_string_does_not_raise():
    # int("--50") raises; the tolerant path must not crash on it anywhere.
    c = _cfg()
    c.chunking.chunk_size = "--50"
    fingerprint_collection(c)  # must not raise
    collection_provenance(c)   # must not raise

def test_provenance_tolerates_non_numeric_chunk_fields():
    # collection_provenance() must use the same tolerant coercion as
    # fingerprint_collection()/_normalize() so it never raises on inputs
    # fingerprinting accepts, and the stamped value matches what was hashed.
    c = _cfg()
    c.chunking.chunk_size = "not-a-number"
    c.chunking.chunk_overlap = "also-not-a-number"
    p = collection_provenance(c)
    assert p["chunk_size"] == "not-a-number"
    assert p["chunk_overlap"] == "also-not-a-number"


def test_hybrid_basic_builtin_profile_fingerprint_is_stable_across_the_hf_model_id_fix(tmp_path):
    """task-640 item 4: the builtin "hybrid_basic" RAG profile's embedding
    model id ("all-MiniLM-L6-v2", missing its "sentence-transformers/" org
    prefix) 404s against the real HF Hub. The fix normalizes that id ONLY
    at the point EmbeddingsServiceWrapper hands a model_name_or_path to the
    HF loader (RAG_Search/simplified/embeddings_wrapper.py's _build_config)
    -- it must NEVER touch RAGConfig.embedding.model, the exact field this
    module's fingerprint_collection() hashes.

    This test proves that: fingerprint_collection() lives entirely in
    collection_fingerprint.py, with zero import/call dependency on
    embeddings_wrapper.py, so the pinned golden value below can only ever
    change if _index_fields' input set or embedding.model's stored value
    changes -- neither of which item 4 touches. The golden value was
    computed directly from this same (unmodified) builtin profile
    definition (RAG_Search/config_profiles.py) before the embeddings_
    wrapper.py loader fix landed, confirming byte-for-byte stability.
    """
    from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager

    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    profile = mgr.get_profile("hybrid_basic")
    assert profile is not None

    # The fingerprint-determining field itself must still be the literal,
    # unprefixed id -- proving item 4 did not "fix" it by editing the
    # builtin profile definition (option (b)/(c) from the task, explicitly
    # rejected as unsafe/heavy for existing installs).
    assert profile.rag_config.embedding.model == "all-MiniLM-L6-v2"

    assert fingerprint_collection(profile.rag_config) == "bf912f0fe11b"
