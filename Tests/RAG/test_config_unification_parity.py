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


def test_get_shared_rag_service_routes_active_pointer_through_resolver(monkeypatch):
    """Locks get_shared_rag_service's actual construction branch (task-2 review, Finding 2).

    The live consumer (chat_rag_events.get_or_initialize_rag_service) reads the
    active [rag.service].profile pointer and forwards it AS profile_name. So the
    "None" and "explicit == active pointer" cases must both route through
    resolve_active_rag_config() (env-applied); only a genuinely different explicit
    profile_name should take the legacy named-profile branch without the resolver.
    """
    import tldw_chatbook.RAG_Search.ingestion_indexing as ii
    import tldw_chatbook.RAG_Search.simplified as simplified_pkg
    import tldw_chatbook.RAG_Search.simplified.active_config as ac

    sentinel_config = RAGConfig(embedding=EmbeddingConfig(model="resolved-sentinel"),
                                 vector_store=VectorStoreConfig(type="memory"))
    calls = []

    def _fake_create_rag_service(**kwargs):
        calls.append(kwargs)
        return object()  # dummy sentinel service

    monkeypatch.setattr(simplified_pkg, "create_rag_service", _fake_create_rag_service)
    monkeypatch.setattr(ac, "resolve_active_rag_config", lambda: sentinel_config)
    monkeypatch.setattr(ii, "_configured_profile", lambda: "active_prof")

    # Case 1: profile_name=None -> resolver-routed.
    ii.get_shared_rag_service()
    assert len(calls) == 1
    assert calls[0].get("config") is not None
    assert calls[0].get("config") is sentinel_config
    assert calls[0].get("profile_name") == "active_prof"

    # Case 2: explicit profile_name == active pointer -> ALSO resolver-routed.
    ii.reset_shared_rag_service()
    ii.get_shared_rag_service("active_prof")
    assert len(calls) == 2
    assert calls[1].get("config") is not None
    assert calls[1].get("config") is sentinel_config
    assert calls[1].get("profile_name") == "active_prof"

    # Case 3: explicit profile_name != active pointer -> legacy named-profile branch.
    ii.reset_shared_rag_service()
    ii.get_shared_rag_service("some_other_profile")
    assert len(calls) == 3
    assert calls[2].get("config") is None
    assert calls[2].get("profile_name") == "some_other_profile"
