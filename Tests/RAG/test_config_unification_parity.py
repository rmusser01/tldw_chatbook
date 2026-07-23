import pytest
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ProfileConfig
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, EmbeddingConfig, VectorStoreConfig


@pytest.fixture(autouse=True)
def _reset_singleton():
    from tldw_chatbook.RAG_Search.ingestion_indexing import reset_shared_rag_service
    reset_shared_rag_service()
    yield
    reset_shared_rag_service()


def _wire(monkeypatch, tmp_path, active_rag):
    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    p = ProfileConfig(name="Active", description="d", profile_type="custom", rag_config=active_rag)
    mgr.save_profile(p)
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    monkeypatch.setattr(ac, "_manager", lambda: mgr, raising=False)
    monkeypatch.setattr(ac, "_active_profile_id", lambda: p.id, raising=False)
    return mgr, p


def test_ingest_and_query_config_are_identical_for_active_profile(monkeypatch, tmp_path):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig as RC
    rag = RAGConfig(embedding=EmbeddingConfig(model="mock"),
                    vector_store=VectorStoreConfig(type="memory"))
    _wire(monkeypatch, tmp_path, rag)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "env-wins-model")
    # Search path config:
    query_cfg = RC.from_settings()
    # Ingestion path config (what get_shared_rag_service will build from):
    ingest_cfg = resolve_active_rag_config()
    assert query_cfg.embedding.model == ingest_cfg.embedding.model == "env-wins-model"
    # And the fingerprint-determining fields match (anti dimension-crash):
    from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import fingerprint_collection
    assert fingerprint_collection(query_cfg) == fingerprint_collection(ingest_cfg)
