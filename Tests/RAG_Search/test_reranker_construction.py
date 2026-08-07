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

Two more review findings on this same task, both fixed here (task-3170 P0):
  - The `reranking_skipped` tag was written via `results[0].metadata[...] =
    ...`, an IN-PLACE mutation. RAGService.search()'s cache stores/returns
    SearchResult objects by reference, so this permanently poisoned the
    cached entry for up to cache_ttl seconds regardless of what a later
    search for the same query did. Fixed by tagging a COPY of the first
    result (`_tag_first_result` in enhanced_rag_service_v2.py) instead.
  - The dominant real LLM-reranking failure mode is silent, not an
    exception: PointwiseReranker swallows each per-result scoring failure
    and keeps the original score, so a total-failure rerank() call (e.g.
    every provider call failing under a missing credential) returns
    NORMALLY with an unchanged ordering and the reranking_skipped tag never
    fires. Fixed by having every reranker count per-result/per-comparison
    scoring failures (BaseReranker.last_rerank_failures/last_rerank_total)
    and tagging `reranking_degraded` when nonzero.
"""

import asyncio

import pytest

from tldw_chatbook.RAG_Search.reranker import (
    ListwiseReranker,
    PairwiseReranker,
    RerankingConfig,
    create_reranker_from_config,
)
from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig, get_profile_manager
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult


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


def _make_v2_service_with_reranking(tmp_path, enable_cache=False):
    """EnhancedRAGServiceV2 with mock embeddings, in-memory store, and a
    real (pointwise) reranking config -- mirrors the mock-embeddings pattern
    used by Tests/RAG/test_ingestion_indexing.py's `_make_real_service`, but
    routed through a saved profile + the *profile name* (str) construction
    path so `self.reranking_config` (and thus `self.reranker`) actually gets
    populated. (The `elif isinstance(config, ProfileConfig)` branch used to
    be unconditionally broken here -- see
    `test_from_profile_construction_path_populates_reranker` below, which now
    covers that path directly since it was fixed.)

    `enable_cache` defaults to False for the tests that don't care about
    caching; the cache-poisoning regression tests below pass True since the
    base search cache is exactly what let a mutated tag leak across queries.
    """
    manager = get_profile_manager(profiles_dir=tmp_path)

    rag_cfg = RAGConfig()
    rag_cfg.embedding.model = "mock"  # deterministic bag-of-words backend, offline
    rag_cfg.embedding.device = "cpu"
    rag_cfg.vector_store.type = "memory"
    rag_cfg.vector_store.persist_directory = None
    rag_cfg.chunking.chunk_size = 60
    rag_cfg.chunking.chunk_overlap = 10
    rag_cfg.search.enable_cache = enable_cache

    profile = ProfileConfig(
        name=f"test_rerank_profile_cache_{enable_cache}",
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


def _quokka_platypus_docs():
    return [
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


@pytest.mark.asyncio
async def test_reranking_skipped_tag_does_not_poison_the_cache(tmp_path):
    """Review finding on this task (task-3170 P0): RAGService.search()'s
    base-layer cache stores and returns SearchResult objects BY REFERENCE
    (rag_service.py's cache.get_async/put_async), not by copy. The old
    `results[0].metadata["reranking_skipped"] = str(exc)` in-place mutation
    therefore mutated the exact object sitting in the cache -- poisoning it
    for up to cache_ttl seconds (3600s default), regardless of what a LATER
    search for the same query does. Reproduces the reviewer's probe with
    cache ON: search once with a raising reranker (tags "boom"), then search
    the identical query again with reranking explicitly off -- the second
    call must NOT see the first call's tag."""
    service = _make_v2_service_with_reranking(tmp_path, enable_cache=True)
    assert service.reranker is not None

    await service.index_batch_optimized(_quokka_platypus_docs())

    query = "quokka marsupial"

    async def _raise(*args, **kwargs):
        raise RuntimeError("boom: reranker backend unavailable")

    service.reranker.rerank = _raise

    results1 = await service.search(
        query, top_k=5, search_type="semantic", include_citations=False
    )
    assert results1[0].metadata.get("reranking_skipped"), (
        "sanity check: the first search must still be tagged"
    )

    # Same query -- base search cache (enable_cache=True) now serves from
    # cache. Reranking explicitly OFF this time: if the first result object
    # had been mutated in place rather than copied, the stale tag would
    # still be sitting on it even though no reranking was attempted here.
    results2 = await service.search(
        query,
        top_k=5,
        search_type="semantic",
        include_citations=False,
        rerank=False,
    )
    assert not results2[0].metadata.get("reranking_skipped"), (
        "stale reranking_skipped tag leaked forward via a cached SearchResult "
        "object mutated in place by a previous failed rerank attempt"
    )


@pytest.mark.asyncio
async def test_reranking_degraded_tag_when_scoring_silently_fails_for_every_result(
    tmp_path,
):
    """Review finding on this task (task-3170 P0): the dominant real
    LLM-reranking failure mode is SILENT, not an exception.
    PointwiseReranker.rerank() catches per-result scoring failures
    internally and keeps the original score, so a total-failure run (e.g.
    every provider call failing under a missing credential) returns
    NORMALLY with an unchanged ordering -- indistinguishable from "nothing
    needed reranking" without an explicit disclosure. Asserts (a) ordering
    unchanged, (b) `reranking_degraded` tag present, (c) tag absent on a
    later clean (reranking-off) search with cache on -- same
    copy-not-mutate rule as the reranking_skipped test above."""
    service = _make_v2_service_with_reranking(tmp_path, enable_cache=True)
    assert service.reranker is not None

    await service.index_batch_optimized(_quokka_platypus_docs())

    query = "quokka marsupial"

    # Baseline: reranking off, primes the base-layer cache with the
    # PRE-rerank ordering.
    baseline = await service.search(
        query,
        top_k=5,
        search_type="semantic",
        include_citations=False,
        rerank=False,
    )
    baseline_order = [r.id for r in baseline]
    assert len(baseline_order) == 2

    async def _always_fail_scoring(query, result, original_rank):
        raise ValueError("No API key found for provider: openai")

    service.reranker._score_result = _always_fail_scoring

    # Same query -- base layer serves from cache; reranking is attempted
    # (enabled by default on this service) and returns NORMALLY (no
    # exception raised out of rerank()) but every per-result scoring call
    # failed.
    results1 = await service.search(
        query, top_k=5, search_type="semantic", include_citations=False
    )

    assert [r.id for r in results1] == baseline_order, (
        "ordering must be unchanged when every per-result scoring attempt fails"
    )
    assert results1[0].metadata.get("reranking_degraded") == "2/2 scorings failed"

    # Same query again, reranking explicitly OFF: the cached SearchResult
    # objects must be untouched by the degraded tag from the previous call.
    results2 = await service.search(
        query,
        top_k=5,
        search_type="semantic",
        include_citations=False,
        rerank=False,
    )
    assert not results2[0].metadata.get("reranking_degraded"), (
        "stale reranking_degraded tag leaked forward via a cached SearchResult "
        "object mutated in place by a previous degraded rerank attempt"
    )


@pytest.mark.asyncio
async def test_pairwise_reranker_counts_failed_comparisons():
    """The shared BaseReranker seam (last_rerank_failures/last_rerank_total)
    must actually work for PairwiseReranker too, not just Pointwise -- it's
    comparison-based rather than per-result, so a "failure" is a comparison
    whose LLM call raised and fell back to comparing original scores."""
    cfg = RerankingConfig(strategy="pairwise", top_k_to_rerank=5)
    reranker = PairwiseReranker(cfg)

    async def _always_raise(prompt, system_prompt=None):
        raise ValueError("No API key found for provider: openai")

    reranker._call_llm = _always_raise

    results = [
        SearchResult(id="a", score=0.9, document="doc a", metadata={}),
        SearchResult(id="b", score=0.5, document="doc b", metadata={}),
        SearchResult(id="c", score=0.2, document="doc c", metadata={}),
    ]

    reranked = await reranker.rerank("query", results)

    assert len(reranked) == 3
    assert reranker.last_rerank_total > 0
    assert reranker.last_rerank_failures == reranker.last_rerank_total


@pytest.mark.asyncio
async def test_listwise_reranker_counts_total_failure():
    """Same shared-seam check as the pairwise test, for ListwiseReranker --
    a single LLM call covers the whole batch, so a failure there means
    ALL of results_to_rerank failed, not a per-item count."""
    cfg = RerankingConfig(strategy="listwise", top_k_to_rerank=5)
    reranker = ListwiseReranker(cfg)

    async def _always_raise(prompt, system_prompt=None):
        raise ValueError("No API key found for provider: openai")

    reranker._call_llm = _always_raise

    results = [
        SearchResult(id="a", score=0.9, document="doc a", metadata={}),
        SearchResult(id="b", score=0.5, document="doc b", metadata={}),
    ]

    reranked = await reranker.rerank("query", results)

    assert [r.id for r in reranked] == ["a", "b"]
    assert reranker.last_rerank_failures == 2
    assert reranker.last_rerank_total == 2
