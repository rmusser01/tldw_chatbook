"""Reranking profiles must construct and must never fail a search.

V2 called create_reranker(strategy=X, **config.__dict__) -- the dict also
contains 'strategy', so EVERY reranking profile raised TypeError at
construction; Hybrid Full additionally requested the unimplemented
'cross_encoder' strategy. Reranking has never executed on any profile.
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
    populated.

    Deliberately NOT `config=<ProfileConfig instance>` (the `elif isinstance
    (config, ProfileConfig)` branch in `EnhancedRAGServiceV2.__init__`):
    that branch reassigns the local `config` variable to `config.rag_config`
    and then reads `config.reranking_config` off of THAT (a plain
    `RAGConfig`, which has no such attribute) instead of off the original
    profile -- an unconditional `AttributeError` on every call, unrelated to
    the double-`strategy` TypeError this task fixes and out of this task's
    scoped files. Filed as a follow-up rather than silently expanded into
    this change. The `isinstance(config, str)` branch this test uses instead
    reads `profile.reranking_config` correctly.
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
