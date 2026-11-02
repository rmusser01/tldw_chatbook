"""TASK-38580: signature contract for the EnhancedRAGServiceV2 search chain.

``EnhancedRAGServiceV2.search`` forwards keyword arguments to
``super().search(...)`` -- i.e. to ``RAGService.search`` (no intermediate
class overrides ``search``; MRO: V2 -> EnhancedRAGService -> RAGService).
Twice now a kwarg was added to the override's forwarding without being added
to the base (``keyword_source_types`` first, then ``query_embedding``,
commit 859c76f104), and every unit test stayed green because their doubles
mirror ``RAGService.search`` instead of driving the REAL chain -- so every
shared-service search raised ``TypeError`` in production while the suite
passed.

These tests drive the REAL V2 -> RAGService -> InMemoryVectorStore chain
(end to end through ``super()``), doubling only the embedding service, so
any forwarding/signature drift fails here instead of in the app.
"""

import asyncio
import inspect
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from tldw_chatbook.RAG_Search.simplified.config import (
    ChunkingConfig,
    EmbeddingConfig,
    RAGConfig,
    SearchConfig,
    VectorStoreConfig,
)
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import (
    EnhancedRAGServiceV2,
)
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

_DIM = 4


class _CountingEmbeddings:
    """Deterministic embedding double that counts query embeds."""

    def __init__(self) -> None:
        self.async_calls = 0

    async def create_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        self.async_calls += 1
        # Text-independent vectors are fine: the tests that need specific
        # geometry pass `query_embedding` explicitly.
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def close(self) -> None:
        """No-op for `service.close()` parity with the real wrapper."""


def _build_real_v2_service(profiles_dir: Path) -> EnhancedRAGServiceV2:
    """A REAL V2 service over the real InMemoryVectorStore.

    Constructed directly (not through the profile factory) so the test does
    not depend on the user's on-disk RAG profiles: V2's ``__init__`` only
    needs a ``profile_manager`` with a ``profiles_dir`` for the activation
    ledger, and the plain-RAGConfig branch never touches it otherwise. The
    three ``*_db_path`` knobs are pinned to the same scratch dir for the
    same reason: left at None, the query-scope ledger would resolve them
    through the global CLI config, which couples this test to whatever
    config the surrounding test session loaded. The ``search``
    implementation chain under test is entirely real; only the embedding
    service is doubled (after construction).
    """
    config = RAGConfig(
        embedding=EmbeddingConfig(model="mock", device="cpu"),
        vector_store=VectorStoreConfig(
            type="memory",
            persist_directory=None,
            collection_name="search_signature_contract",
            distance_metric="cosine",
        ),
        chunking=ChunkingConfig(chunk_size=100, chunk_overlap=20),
        search=SearchConfig(
            default_top_k=5,
            score_threshold=0.0,
            enable_cache=True,
            cache_size=10,
            cache_ttl=300,
        ),
    )
    config.search.media_db_path = profiles_dir.parent / "media.db"
    config.search.chachanotes_db_path = profiles_dir.parent / "chachanotes.db"
    config.search.prompts_db_path = profiles_dir.parent / "prompts.db"
    service = EnhancedRAGServiceV2(
        config=config,
        enable_parent_retrieval=False,
        enable_reranking=False,
        enable_parallel_processing=False,
        profile_manager=SimpleNamespace(profiles_dir=profiles_dir),
    )
    service.embeddings = _CountingEmbeddings()
    service.vector_store.add(
        ids=["docA_chunk_0", "docB_chunk_0"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        documents=["alpha content", "beta content"],
        metadata=[{"doc_id": "docA"}, {"doc_id": "docB"}],
    )
    return service


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.mark.unit
class TestRealV2SearchChain:
    """Drive the real V2 -> RAGService chain (no search doubles)."""

    def test_default_search_succeeds_through_the_real_chain(self, tmp_path):
        """The P0 repro: even `query_embedding=None` used to raise TypeError."""
        service = _build_real_v2_service(tmp_path / "profiles")
        try:
            results = _run(
                service.search("alpha", include_citations=False)
            )
        finally:
            service.close()
        assert [r.id for r in results] == ["docA_chunk_0", "docB_chunk_0"]

    def test_query_embedding_passthrough_skips_embedding_and_reaches_store(
        self, tmp_path
    ):
        """A precomputed vector is threaded to the store, no embed call."""
        service = _build_real_v2_service(tmp_path / "profiles")
        try:
            results = _run(
                service.search(
                    "anything",
                    include_citations=False,
                    query_embedding=[1.0, 0.0, 0.0, 0.0],
                )
            )
            assert service.embeddings.async_calls == 0
            assert results, "store has two seeded documents"
            assert results[0].id == "docA_chunk_0"  # aligned with the query

            results_b = _run(
                service.search(
                    "anything else",
                    include_citations=False,
                    query_embedding=[0.0, 1.0, 0.0, 0.0],
                )
            )
            assert results_b[0].id == "docB_chunk_0"
            assert service.embeddings.async_calls == 0
        finally:
            service.close()

    def test_default_none_still_embeds_the_query(self, tmp_path):
        """`query_embedding=None` keeps the embed-per-call behavior."""
        service = _build_real_v2_service(tmp_path / "profiles")
        try:
            results = _run(
                service.search("needs embedding", include_citations=False)
            )
            assert service.embeddings.async_calls == 1
            assert results
        finally:
            service.close()

    def test_query_embedding_with_keyword_search_raises_value_error(self, tmp_path):
        """Non-semantic searches reject the kwarg loudly, per contract."""
        service = _build_real_v2_service(tmp_path / "profiles")
        try:
            with pytest.raises(ValueError, match="query_embedding"):
                _run(
                    service.search(
                        "alpha",
                        search_type="keyword",
                        query_embedding=[1.0, 0.0, 0.0, 0.0],
                    )
                )
        finally:
            service.close()


@pytest.mark.unit
class TestForwardedKwargsContract:
    """Every kwarg V2's call site forwards must exist on RAGService.search.

    Behavioral drift is caught by the chain tests above only for the kwargs
    they exercise; this structural test catches a NEW forwarded kwarg the
    moment it is added to the ``super().search(...)`` call, before any test
    is written for it.
    """

    def test_every_forwarded_kwarg_exists_on_base_search(self):
        unwrapped = inspect.unwrap(EnhancedRAGServiceV2.search)
        source = inspect.getsource(unwrapped)
        call_match = re.search(r"super\(\)\.search\((.*?)\)\s*\n", source, re.DOTALL)
        assert call_match is not None, (
            "Could not locate the super().search(...) forwarding call in "
            "EnhancedRAGServiceV2.search; update this contract test to the "
            "new call shape."
        )
        forwarded = set(re.findall(r"(\w+)=", call_match.group(1)))
        assert forwarded, "forwarding call parsed as empty; fix the parser"
        base_params = set(inspect.signature(RAGService.search).parameters)
        missing = forwarded - base_params
        assert not missing, (
            "EnhancedRAGServiceV2.search forwards kwargs that "
            f"RAGService.search does not accept: {sorted(missing)}. This is "
            "the TASK-34390/TASK-38580 TypeError regression shape -- add the "
            "parameter to RAGService.search (and thread it through), not "
            "just to the override."
        )

    def test_base_search_accepts_query_embedding_keyword(self):
        """The kwarg whose absence caused the P0, pinned structurally."""
        params = inspect.signature(RAGService.search).parameters
        assert "query_embedding" in params
        assert params["query_embedding"].default is None
