import pytest
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, EmbeddingConfig, ChunkingConfig, VectorStoreConfig
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ProfileConfig
from tldw_chatbook.config import get_cli_setting


@pytest.fixture
def active(tmp_path, monkeypatch):
    """A ConfigProfileManager over a temp dir + a helper to set the active pointer."""
    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    # Point resolve_active_rag_config at this manager + a chosen active id via monkeypatch.
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    monkeypatch.setattr(ac, "_manager", lambda: mgr, raising=False)
    state = {"active": "hybrid_basic"}
    monkeypatch.setattr(ac, "_active_profile_id", lambda: state["active"], raising=False)
    return mgr, state


def test_resolves_active_profiles_rag_config(active, monkeypatch):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    p = ProfileConfig(name="Custom A", description="d", profile_type="custom",
                      rag_config=RAGConfig(embedding=EmbeddingConfig(model="modelA"),
                                           chunking=ChunkingConfig(chunk_size=321),
                                           vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id
    cfg = resolve_active_rag_config()
    assert cfg.embedding.model == "modelA"
    assert cfg.chunking.chunk_size == 321


def test_env_overrides_win_over_profile(active, monkeypatch):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    p = ProfileConfig(name="Custom B", description="d", profile_type="custom",
                      rag_config=RAGConfig(embedding=EmbeddingConfig(model="profile-model"),
                                           vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "env-model")
    assert resolve_active_rag_config().embedding.model == "env-model"


def test_returns_deep_copy_not_the_profile_object(active):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    cfg = resolve_active_rag_config()
    cfg.chunking.chunk_size = 99999
    assert mgr.get_profile(state["active"]).rag_config.chunking.chunk_size != 99999


def test_env_overrides_chunking_and_search_and_pipeline_settings(active, monkeypatch):
    """These env reads exist in the legacy from_settings body (RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP, RAG_TOP_K, RAG_SEARCH_MODE, RAG_DEFAULT_PIPELINE) but were
    not covered in the task brief's _apply_env_overrides sketch. They must still
    be honored so no override behavior regresses."""
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    p = ProfileConfig(name="Custom C", description="d", profile_type="custom",
                      rag_config=RAGConfig(vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id

    monkeypatch.setenv("RAG_CHUNK_SIZE", "555")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "77")
    monkeypatch.setenv("RAG_TOP_K", "13")
    monkeypatch.setenv("RAG_SEARCH_MODE", "hybrid")
    monkeypatch.setenv("RAG_DEFAULT_PIPELINE", "custom_pipeline")

    cfg = resolve_active_rag_config()
    assert cfg.chunking.chunk_size == 555
    assert cfg.chunking.chunk_overlap == 77
    assert cfg.search.default_top_k == 13
    assert cfg.search.default_search_mode == "hybrid"
    assert cfg.pipeline.default_pipeline == "custom_pipeline"


def test_override_persist_dir_and_embedding_model_args_win_over_env(active, monkeypatch, tmp_path):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    p = ProfileConfig(name="Custom D", description="d", profile_type="custom",
                      rag_config=RAGConfig(vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id

    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "env-model")
    monkeypatch.setenv("RAG_PERSIST_DIR", str(tmp_path / "env-dir"))

    override_dir = tmp_path / "explicit-dir"
    cfg = resolve_active_rag_config(override_embedding_model="explicit-model",
                                     override_persist_dir=override_dir)
    assert cfg.embedding.model == "explicit-model"
    assert str(cfg.vector_store.persist_directory) == str(override_dir)


def test_hybrid_alpha_comes_from_active_profile(active, monkeypatch):
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, SearchConfig, VectorStoreConfig
    from tldw_chatbook.RAG_Search.fusion import resolve_hybrid_alpha
    mgr, state = active
    p = ProfileConfig(name="Alpha", description="d", profile_type="custom",
                      rag_config=RAGConfig(search=SearchConfig(hybrid_alpha=0.33),
                                           vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id
    assert resolve_hybrid_alpha() == pytest.approx(0.33)  # explicit=None -> active profile


def test_set_active_profile_writes_pointer_and_resets_service(active, monkeypatch):
    from tldw_chatbook.RAG_Search.simplified.active_config import set_active_profile
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    writes = {}
    monkeypatch.setattr(ac, "save_setting_to_cli_config",
                        lambda section, key, value: writes.update({(section, key): value}) or True,
                        raising=False)
    reset = {"called": False}
    monkeypatch.setattr(ac, "reset_shared_rag_service",
                        lambda: reset.update(called=True), raising=False)
    set_active_profile("my_profile")
    # CORRECTED assertion: save_setting_to_cli_config(section, key, value) nests
    # via the section arg, so the pointer write is section="rag.service",
    # key="profile" (NOT section="rag", key="service.profile" as the brief's
    # first draft had it) -- that shape is what the read path
    # (get_cli_setting("rag", "service", {}).get("profile")) actually resolves.
    assert writes.get(("rag.service", "profile")) == "my_profile"
    assert reset["called"] is True


def test_set_active_profile_round_trip_write_matches_read(monkeypatch, tmp_path):
    """Real (non-mock) proof: the pointer set_active_profile() writes is exactly
    what the resolver's read path (_active_profile_id() ->
    get_cli_setting("rag", "service", {}).get("profile")) reads back. Uses a
    real temp TOML file via TLDW_CONFIG_PATH so this exercises an actual
    write-then-read round trip, not mocks."""
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    from tldw_chatbook.RAG_Search.simplified.active_config import set_active_profile
    set_active_profile("some_profile_xyz")
    assert get_cli_setting("rag", "service", {}).get("profile") == "some_profile_xyz"
