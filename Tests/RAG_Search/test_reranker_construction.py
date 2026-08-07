"""Reranking profiles must construct and must never fail a search.

V2 called create_reranker(strategy=X, **config.__dict__) -- the dict also
contains 'strategy', so EVERY reranking profile raised TypeError at
construction; Hybrid Full additionally requested the unimplemented
'cross_encoder' strategy. Reranking has never executed on any profile.

Two further construction-seam bugs found while writing the above (both
fixed here too, task-3170 P0):
  - EnhancedRAGServiceV2.__init__'s `elif isinstance(config, ProfileConfig)`
    branch read `config.reranking_config` AFTER reassigning `config =
    config.rag_config`, raising AttributeError unconditionally on every
    from_profile()/create_rag_from_profile() call.
  - rag_factory.create_rag_service() -- the app's real RAG-service entry
    point -- computed `enable_reranking` from the profile but passed a bare
    RAGConfig into the constructor, which forces `reranking_config = None`
    regardless of that flag, so no reranker ever built in production.
"""

import asyncio

import pytest

from tldw_chatbook.RAG_Search.reranker import RerankingConfig, create_reranker_from_config
from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig, get_profile_manager
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2


def test_create_reranker_from_config_does_not_double_pass_strategy():
    cfg = RerankingConfig(strategy="pointwise", top_k_to_rerank=5)
    reranker = create_reranker_from_config(cfg)
    assert reranker.config.strategy == "pointwise"


def test_every_builtin_profile_reranking_config_constructs(tmp_path):
    # get_profile_manager(profiles_dir=...) always returns a FRESH manager
    # over that directory (never the cached singleton) -- the documented
    # isolation seam. tmp_path is empty, so list_profiles() is exactly the
    # built-in set; ``read_only`` marks a profile as a built-in seed (every
    # profile registered in _load_builtin_profiles sets it True; no custom
    # profile is ever loaded from an empty directory).
    manager = get_profile_manager(profiles_dir=tmp_path)
    builtin_profiles = [
        manager.get_profile(name)
        for name in manager.list_profiles()
        if manager.get_profile(name) and manager.get_profile(name).read_only
    ]
    assert builtin_profiles, "expected at least one built-in profile"

    for profile in builtin_profiles:
        rc = getattr(profile, "reranking_config", None)
        if rc is not None:
            create_reranker_from_config(rc)  # must not raise


def _make_v2_service_with_reranking(tmp_path):
    """EnhancedRAGServiceV2 with mock embeddings, in-memory store, and a
    real (pointwise) reranking config -- mirrors the mock-embeddings pattern
    used by Tests/RAG/test_ingestion_indexing.py's `_make_real_service`, but
    routed through a saved profile + the *profile name* (str) construction
    path so `self.reranking_config` (and thus `self.reranker`) actually gets
    populated. (The `elif isinstance(config, ProfileConfig)` branch used to
    be unconditionally broken here -- see
    `test_from_profile_construction_path_populates_reranker` below, which now
    covers that path directly since it was fixed.)
    """
    manager = get_profile_manager(profiles_dir=tmp_path)

    rag_cfg = RAGConfig()
    rag_cfg.embedding.model = "mock"  # deterministic bag-of-words backend, offline
    rag_cfg.embedding.device = "cpu"
    rag_cfg.vector_store.type = "memory"
    rag_cfg.vector_store.persist_directory = None
    rag_cfg.chunking.chunk_size = 60
    rag_cfg.chunking.chunk_overlap = 10
    rag_cfg.search.enable_cache = False

    profile = ProfileConfig(
        name="test_rerank_profile",
        description="test profile with reranking enabled",
        profile_type="balanced",
        rag_config=rag_cfg,
        reranking_config=RerankingConfig(strategy="pointwise", top_k_to_rerank=5),
    )
    manager.save_profile(profile)

    return EnhancedRAGServiceV2(
        config=profile.id,
        profile_manager=manager,
        enable_parent_retrieval=False,
        enable_reranking=True,
        enable_parallel_processing=False,
    )


@pytest.mark.asyncio
async def test_raising_reranker_degrades_to_unreranked_results(tmp_path):
    service = _make_v2_service_with_reranking(tmp_path)
    assert service.reranker is not None, "reranker must construct for this profile"

    await service.index_batch_optimized(
        [
            {
                "id": "doc-1",
                "content": "The quokka is a small marsupial found on Rottnest Island.",
                "title": "Quokka Facts",
            },
            {
                "id": "doc-2",
                "content": "Platypuses are egg-laying mammals native to eastern Australia.",
                "title": "Platypus Facts",
            },
        ]
    )

    async def _raise(*args, **kwargs):
        raise RuntimeError("boom: reranker backend unavailable")

    service.reranker.rerank = _raise

    results = await service.search(
        "quokka marsupial",
        top_k=5,
        search_type="semantic",
        include_citations=False,
    )

    assert results, "base (unreranked) results must still come back"
    assert results[0].metadata.get("reranking_skipped"), (
        "first result must be tagged to disclose that reranking was skipped"
    )


def test_from_profile_construction_path_populates_reranker(tmp_path):
    """The `elif isinstance(config, ProfileConfig)` branch in
    `EnhancedRAGServiceV2.__init__` -- the path `from_profile()` /
    `create_rag_from_profile()` use -- used to reassign the local `config`
    variable to `config.rag_config` and then read `config.reranking_config`
    off of THAT (a plain `RAGConfig`, which has no such attribute), instead
    of off the original profile. That raised `AttributeError` unconditionally
    on every call through this path, for every profile, reranking or not.
    Constructing directly with `config=<ProfileConfig instance>` (rather than
    a profile name string) must now both succeed and actually populate
    `self.reranker` when the profile has a reranking_config.
    """
    manager = get_profile_manager(profiles_dir=tmp_path)

    rag_cfg = RAGConfig()
    rag_cfg.embedding.model = "mock"
    rag_cfg.embedding.device = "cpu"
    rag_cfg.vector_store.type = "memory"
    rag_cfg.vector_store.persist_directory = None
    rag_cfg.chunking.chunk_size = 60
    rag_cfg.chunking.chunk_overlap = 10
    rag_cfg.search.enable_cache = False

    profile = ProfileConfig(
        name="from_profile_rerank_test",
        description="test profile with reranking enabled",
        profile_type="balanced",
        rag_config=rag_cfg,
        reranking_config=RerankingConfig(strategy="pointwise", top_k_to_rerank=5),
    )

    service = EnhancedRAGServiceV2(
        config=profile,  # a ProfileConfig instance, not a name string
        profile_manager=manager,
        enable_parent_retrieval=False,
        enable_reranking=True,
        enable_parallel_processing=False,
    )

    assert service.reranker is not None
    assert service.reranking_config is profile.reranking_config


def test_create_rag_service_threads_reranking_config_for_hybrid_full():
    """The app's real RAG-service entry point (`search_service.py` ->
    `rag_factory.create_rag_service()`) computed `enable_reranking =
    profile.reranking_config is not None` but passed a bare `RAGConfig` into
    `EnhancedRAGServiceV2` -- which forces `self.reranking_config = None`
    unconditionally in the constructor's `else` branch, regardless of the
    `enable_reranking` flag. So no reranker EVER constructed via this
    factory in production, for any profile, even after the reranker.py
    factory-seam fix. `create_rag_service` must thread the profile's actual
    `reranking_config` through so a reranker actually gets built.
    """
    from tldw_chatbook.RAG_Search.simplified.rag_factory import create_rag_service

    mock_cfg = RAGConfig()
    mock_cfg.embedding.model = "mock"
    mock_cfg.embedding.device = "cpu"
    mock_cfg.vector_store.type = "memory"
    mock_cfg.vector_store.persist_directory = None
    mock_cfg.chunking.chunk_size = 5
    mock_cfg.chunking.chunk_overlap = 1
    mock_cfg.search.enable_cache = False

    service = create_rag_service(profile_name="hybrid_full", config=mock_cfg)

    assert service.enable_reranking is True
    assert service.reranker is not None


def test_create_rag_service_no_reranker_for_profile_without_reranking_config():
    """Counterpart to the hybrid_full test: a profile with no
    reranking_config (hybrid_basic) must NOT get a reranker -- confirms the
    fix threads the profile's actual config through rather than switching
    reranking on unconditionally."""
    from tldw_chatbook.RAG_Search.simplified.rag_factory import create_rag_service

    mock_cfg = RAGConfig()
    mock_cfg.embedding.model = "mock"
    mock_cfg.embedding.device = "cpu"
    mock_cfg.vector_store.type = "memory"
    mock_cfg.vector_store.persist_directory = None
    mock_cfg.chunking.chunk_size = 60
    mock_cfg.chunking.chunk_overlap = 10
    mock_cfg.search.enable_cache = False

    service = create_rag_service(profile_name="hybrid_basic", config=mock_cfg)

    assert service.reranker is None
