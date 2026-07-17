"""
Unit tests for hybrid retrieval fusion (task-256).

Covers the pure RRF + alpha-blend math in tldw_chatbook.RAG_Search.fusion
with hand-computed rankings, plus the two integration seams that consume it:
RAGService._fuse_hybrid_results and the pipeline rrf_merge parallel step.

Server-parity reference (tldw_server database_retrievers.py):
    leg_rrf(doc) = 1 / (k + rank)   # rank 1-based within the leg
    final(doc)   = (1 - alpha) * fts_rrf + alpha * vector_rrf
with k = 60 and alpha = 0.7 (vector-weighted) as defaults.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pytest

from tldw_chatbook.RAG_Search.fusion import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RRF_K,
    interleave_rankings,
    reciprocal_rank_fusion,
    resolve_hybrid_alpha,
)

pytestmark = pytest.mark.unit


@dataclass
class Doc:
    """Minimal ranked item for fusion tests."""
    id: str
    score: float = 1.0
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


def _ids(fused):
    return [entry.key for entry in fused]


def _scores(fused):
    return {entry.key: entry.score for entry in fused}


K = DEFAULT_RRF_K  # 60


class TestServerParityDefaults:
    def test_defaults_match_tldw_server(self):
        assert DEFAULT_RRF_K == 60
        assert DEFAULT_HYBRID_ALPHA == 0.7


class TestReciprocalRankFusion:
    """Hand-computed rankings: A rank1 FTS only, B rank1 vector only, C rank2 both."""

    @staticmethod
    def _fuse(alpha):
        fts = [Doc("A"), Doc("C")]
        vector = [Doc("B"), Doc("C")]
        return reciprocal_rank_fusion(
            fts, vector, key=lambda d: d.id, alpha=alpha
        )

    def test_alpha_zero_is_fts_only_ordering(self):
        fused = self._fuse(alpha=0.0)
        # A = 1/(K+1), C = 1/(K+2), B = 0 (vector-only doc contributes nothing)
        assert _ids(fused) == ["A", "C", "B"]
        scores = _scores(fused)
        assert scores["A"] == pytest.approx(1 / (K + 1))
        assert scores["C"] == pytest.approx(1 / (K + 2))
        assert scores["B"] == 0.0

    def test_alpha_one_is_vector_only_ordering(self):
        fused = self._fuse(alpha=1.0)
        assert _ids(fused) == ["B", "C", "A"]
        scores = _scores(fused)
        assert scores["B"] == pytest.approx(1 / (K + 1))
        assert scores["C"] == pytest.approx(1 / (K + 2))
        assert scores["A"] == 0.0

    def test_alpha_half_both_legs_beat_single_leg(self):
        fused = self._fuse(alpha=0.5)
        # C = 0.5/(K+2) + 0.5/(K+2) = 1/(K+2) beats A = B = 0.5/(K+1)
        assert _ids(fused) == ["C", "A", "B"]
        scores = _scores(fused)
        assert scores["C"] == pytest.approx(1 / (K + 2))
        assert scores["A"] == pytest.approx(0.5 / (K + 1))
        assert scores["B"] == pytest.approx(0.5 / (K + 1))

    def test_alpha_default_07_vector_weighted(self):
        fused = self._fuse(alpha=0.7)
        # C = 0.3/(K+2) + 0.7/(K+2) = 1/(K+2) ~ 0.01613
        # B = 0.7/(K+1) ~ 0.01148, A = 0.3/(K+1) ~ 0.00492
        assert _ids(fused) == ["C", "B", "A"]
        scores = _scores(fused)
        assert scores["C"] == pytest.approx(1 / (K + 2))
        assert scores["B"] == pytest.approx(0.7 / (K + 1))
        assert scores["A"] == pytest.approx(0.3 / (K + 1))

    def test_tie_between_legs_breaks_deterministically_fts_first(self):
        # A only in FTS at rank 1, B only in vector at rank 1, alpha=0.5:
        # identical scores; the FTS-side doc must sort first, stably.
        fused = reciprocal_rank_fusion(
            [Doc("A")], [Doc("B")], key=lambda d: d.id, alpha=0.5
        )
        assert _ids(fused) == ["A", "B"]
        assert fused[0].score == pytest.approx(fused[1].score)
        # Repeat runs give the same order (determinism, not dict luck)
        for _ in range(5):
            again = reciprocal_rank_fusion(
                [Doc("A")], [Doc("B")], key=lambda d: d.id, alpha=0.5
            )
            assert _ids(again) == ["A", "B"]

    def test_empty_vector_leg_preserves_fts_ordering(self):
        fts = [Doc("A"), Doc("B"), Doc("C")]
        fused = reciprocal_rank_fusion(fts, [], key=lambda d: d.id, alpha=0.7)
        assert _ids(fused) == ["A", "B", "C"]
        for rank, entry in enumerate(fused, start=1):
            assert entry.score == pytest.approx((1 - 0.7) * (1 / (K + rank)))
            assert entry.vector_rank is None
            assert entry.vector_rrf == 0.0

    def test_empty_fts_leg_preserves_vector_ordering(self):
        vector = [Doc("X"), Doc("Y")]
        fused = reciprocal_rank_fusion([], vector, key=lambda d: d.id, alpha=0.7)
        assert _ids(fused) == ["X", "Y"]
        assert fused[0].score == pytest.approx(0.7 / (K + 1))

    def test_alpha_one_with_empty_vector_leg_keeps_fts_order_at_zero_score(self):
        # Degenerate but must stay deterministic: all scores 0, FTS order kept.
        fused = reciprocal_rank_fusion(
            [Doc("A"), Doc("B")], [], key=lambda d: d.id, alpha=1.0
        )
        assert _ids(fused) == ["A", "B"]
        assert all(entry.score == 0.0 for entry in fused)

    def test_both_legs_empty(self):
        assert reciprocal_rank_fusion([], [], key=lambda d: d.id) == []

    def test_max_results_caps_output(self):
        fts = [Doc(f"f{i}") for i in range(10)]
        vector = [Doc(f"v{i}") for i in range(10)]
        fused = reciprocal_rank_fusion(
            fts, vector, key=lambda d: d.id, alpha=0.7, max_results=5
        )
        assert len(fused) == 5

    def test_custom_rrf_k(self):
        fused = reciprocal_rank_fusion(
            [Doc("A")], [], key=lambda d: d.id, alpha=0.0, rrf_k=0
        )
        assert fused[0].score == pytest.approx(1.0)  # 1/(0+1)

    def test_duplicate_key_within_leg_keeps_best_rank(self):
        fts = [Doc("A"), Doc("A"), Doc("B")]
        fused = reciprocal_rank_fusion(fts, [], key=lambda d: d.id, alpha=0.0)
        assert _ids(fused) == ["A", "B"]
        assert fused[0].fts_rank == 1
        assert fused[1].fts_rank == 3  # B keeps its position in the returned order

    def test_provenance_and_items_preserved(self):
        a_fts = Doc("A", metadata={"leg": "fts"})
        a_vec = Doc("A", metadata={"leg": "vector"})
        fused = reciprocal_rank_fusion(
            [a_fts], [a_vec], key=lambda d: d.id, alpha=0.7
        )
        entry = fused[0]
        assert entry.fts_item is a_fts
        assert entry.vector_item is a_vec
        assert entry.item is a_fts  # FTS item is primary, matching the server
        assert entry.fts_rank == 1 and entry.vector_rank == 1
        assert entry.fts_rrf == pytest.approx(1 / (K + 1))
        assert entry.vector_rrf == pytest.approx(1 / (K + 1))
        assert entry.provenance() == {
            "fts_rank": 1,
            "vector_rank": 1,
            "fts_rrf": entry.fts_rrf,
            "vector_rrf": entry.vector_rrf,
        }

    def test_inputs_not_mutated(self):
        doc = Doc("A", score=0.42)
        reciprocal_rank_fusion([doc], [doc], key=lambda d: d.id, alpha=0.7)
        assert doc.score == 0.42

    def test_rank_based_not_score_based(self):
        # Leg raw scores must not influence fusion: only order matters.
        fts_hi = [Doc("A", score=100.0), Doc("B", score=0.001)]
        fts_lo = [Doc("A", score=0.001), Doc("B", score=100.0)]
        fused_hi = reciprocal_rank_fusion(fts_hi, [], key=lambda d: d.id, alpha=0.0)
        fused_lo = reciprocal_rank_fusion(fts_lo, [], key=lambda d: d.id, alpha=0.0)
        assert _scores(fused_hi) == _scores(fused_lo)


class TestInterleaveRankings:
    def test_round_robin_by_rank(self):
        a = [Doc("a1"), Doc("a2")]
        b = [Doc("b1")]
        c = [Doc("c1"), Doc("c2"), Doc("c3")]
        merged = interleave_rankings([a, b, c], key=lambda d: d.id)
        assert [d.id for d in merged] == ["a1", "b1", "c1", "a2", "c2", "c3"]

    def test_deduplicates_keeping_earliest(self):
        a = [Doc("x"), Doc("a2")]
        b = [Doc("x"), Doc("b2")]
        merged = interleave_rankings([a, b], key=lambda d: d.id)
        assert [d.id for d in merged] == ["x", "a2", "b2"]
        assert merged[0] is a[0]

    def test_empty_inputs(self):
        assert interleave_rankings([], key=lambda d: d.id) == []
        assert interleave_rankings([[], []], key=lambda d: d.id) == []


class TestResolveHybridAlpha:
    def test_explicit_value_wins(self, monkeypatch):
        import tldw_chatbook.config as app_config
        monkeypatch.setattr(
            app_config, "get_cli_setting",
            lambda *a, **k: {"retriever": {"hybrid_alpha": 0.2}},
        )
        assert resolve_hybrid_alpha(0.4) == 0.4

    def test_reads_authoritative_config_knob(self, monkeypatch):
        import tldw_chatbook.config as app_config

        def fake_get_cli_setting(section, key=None, default=None):
            assert (section, key) == ("AppRAGSearchConfig", "rag")
            return {"retriever": {"hybrid_alpha": 0.25}}

        monkeypatch.setattr(app_config, "get_cli_setting", fake_get_cli_setting)
        assert resolve_hybrid_alpha() == 0.25

    def test_defaults_to_server_parity_when_unset(self, monkeypatch):
        import tldw_chatbook.config as app_config
        monkeypatch.setattr(app_config, "get_cli_setting", lambda *a, **k: {})
        assert resolve_hybrid_alpha() == DEFAULT_HYBRID_ALPHA

    @pytest.mark.parametrize("bad", [-0.1, 1.5, "not-a-number", object()])
    def test_invalid_values_fall_back_to_default(self, bad):
        assert resolve_hybrid_alpha(bad) == DEFAULT_HYBRID_ALPHA

    @pytest.mark.parametrize("edge", [0.0, 1.0])
    def test_bounds_are_valid(self, edge):
        assert resolve_hybrid_alpha(edge) == edge


class TestRAGServiceFusion:
    """RAGService._fuse_hybrid_results applies the shared fusion math."""

    @staticmethod
    def _make_results():
        from tldw_chatbook.RAG_Search.simplified.citations import (
            Citation, CitationType, SearchResultWithCitations,
        )

        def cite(doc_id, start, confidence, match_type):
            return Citation(
                document_id=doc_id, document_title=doc_id, chunk_id=f"{doc_id}_0",
                text="snippet", start_char=start, end_char=start + 7,
                confidence=confidence, match_type=match_type,
            )

        keyword = [
            SearchResultWithCitations(
                id="A", score=0.8, document="doc a", metadata={},
                citations=[cite("A", 0, 0.9, CitationType.KEYWORD)],
            ),
            SearchResultWithCitations(
                id="C", score=0.7, document="doc c", metadata={},
                citations=[cite("C", 0, 0.8, CitationType.KEYWORD)],
            ),
        ]
        semantic = [
            SearchResultWithCitations(
                id="B", score=0.95, document="doc b", metadata={},
                citations=[cite("B", 0, 0.9, CitationType.SEMANTIC)],
            ),
            SearchResultWithCitations(
                id="C", score=0.9, document="doc c", metadata={},
                citations=[cite("C", 100, 0.85, CitationType.SEMANTIC)],
            ),
        ]
        return keyword, semantic

    def test_rrf_scores_replace_leg_scores_and_citations_merge(self):
        from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

        keyword, semantic = self._make_results()
        results = RAGService._fuse_hybrid_results(
            keyword_results=keyword,
            semantic_results=semantic,
            top_k=10,
            alpha=0.7,
            include_citations=True,
        )

        assert [r.id for r in results] == ["C", "B", "A"]
        by_id = {r.id: r for r in results}
        assert by_id["C"].score == pytest.approx(1 / (K + 2))
        assert by_id["B"].score == pytest.approx(0.7 / (K + 1))
        assert by_id["A"].score == pytest.approx(0.3 / (K + 1))
        # C appeared in both legs: citations merged from both
        assert len(by_id["C"].citations) == 2
        # Leg provenance recorded
        fusion_meta = by_id["C"].metadata["hybrid_fusion"]
        assert fusion_meta["fts_rank"] == 2
        assert fusion_meta["vector_rank"] == 2
        assert fusion_meta["alpha"] == 0.7
        assert fusion_meta["rrf_k"] == K

    def test_top_k_cap_and_alpha_zero(self):
        from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

        keyword, semantic = self._make_results()
        results = RAGService._fuse_hybrid_results(
            keyword_results=keyword,
            semantic_results=semantic,
            top_k=2,
            alpha=0.0,
            include_citations=True,
        )
        # alpha=0: pure FTS ordering (A then C), capped at 2
        assert [r.id for r in results] == ["A", "C"]

    def test_basic_results_without_citations(self):
        from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
        from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult

        keyword = [SearchResult(id="A", score=0.5, document="a", metadata={})]
        semantic = [SearchResult(id="B", score=0.9, document="b", metadata={})]
        results = RAGService._fuse_hybrid_results(
            keyword_results=keyword,
            semantic_results=semantic,
            top_k=10,
            alpha=1.0,
            include_citations=False,
        )
        assert [r.id for r in results] == ["B", "A"]
        assert results[0].score == pytest.approx(1 / (K + 1))


class TestPipelineRrfMerge:
    """The pipeline 'rrf_merge' parallel step fuses FTS5 legs vs semantic leg."""

    @staticmethod
    def _pipeline_result(source, doc_id):
        from tldw_chatbook.RAG_Search.pipeline_types import SearchResult
        return SearchResult(
            source=source, id=doc_id, title=doc_id, content=f"content {doc_id}"
        )

    def test_builtin_hybrid_pipeline_uses_rrf_merge(self):
        from tldw_chatbook.RAG_Search.pipeline_builder_simple import BUILTIN_PIPELINES

        parallel_steps = [
            s for s in BUILTIN_PIPELINES['hybrid']['steps']
            if s.get('type') == 'parallel'
        ]
        assert len(parallel_steps) == 1
        assert parallel_steps[0].get('merge') == 'rrf_merge'

    def test_parallel_step_rrf_merge_fuses_legs(self, monkeypatch):
        from tldw_chatbook.RAG_Search import pipeline_builder_simple as pbs

        media = [self._pipeline_result("media", "m1"), self._pipeline_result("media", "shared")]
        conversations = [self._pipeline_result("conversation", "c1")]
        notes = []
        semantic = [self._pipeline_result("media", "shared"), self._pipeline_result("media", "s2")]

        async def fake_media(app, query, top_k, keyword_filter=None):
            return list(media)

        async def fake_conversations(app, query, top_k):
            return list(conversations)

        async def fake_notes(app, query, top_k):
            return list(notes)

        async def fake_semantic(app, query, sources, **config):
            return list(semantic)

        monkeypatch.setitem(pbs.RETRIEVAL_FUNCTIONS, 'search_media_fts5', fake_media)
        monkeypatch.setitem(pbs.RETRIEVAL_FUNCTIONS, 'search_conversations_fts5', fake_conversations)
        monkeypatch.setitem(pbs.RETRIEVAL_FUNCTIONS, 'search_notes_fts5', fake_notes)
        monkeypatch.setitem(pbs.RETRIEVAL_FUNCTIONS, 'search_semantic', fake_semantic)

        step_config = {
            'type': 'parallel',
            'functions': [
                {'function': 'search_media_fts5', 'config': {'top_k': 20}},
                {'function': 'search_conversations_fts5', 'config': {'top_k': 20}},
                {'function': 'search_notes_fts5', 'config': {'top_k': 20}},
                {'function': 'search_semantic', 'config': {'top_k': 20}},
            ],
            'merge': 'rrf_merge',
            'config': {'alpha': 0.7},
        }
        context = {
            'app': object(), 'query': 'q', 'sources': {}, 'params': {}, 'results': [],
        }

        results = asyncio.run(pbs._execute_parallel_step(step_config, context))

        # FTS leg (interleaved): m1, c1, shared -> ranks 1, 2, 3
        # Vector leg: shared, s2 -> ranks 1, 2
        expected = {
            ("media", "m1"): 0.3 / (K + 1),
            ("conversation", "c1"): 0.3 / (K + 2),
            ("media", "shared"): 0.3 / (K + 3) + 0.7 / (K + 1),
            ("media", "s2"): 0.7 / (K + 2),
        }
        got = {(r.source, r.id): r.score for r in results}
        assert set(got) == set(expected)
        for key, score in expected.items():
            assert got[key] == pytest.approx(score), key
        # Sorted by fused score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert [r.id for r in results][0] == "shared"
        # Provenance attached
        shared = next(r for r in results if r.id == "shared")
        assert shared.metadata['hybrid_fusion']['fts_rank'] == 3
        assert shared.metadata['hybrid_fusion']['vector_rank'] == 1
        assert shared.metadata['hybrid_fusion']['alpha'] == 0.7

    def test_vector_leg_live_with_real_search_semantic(self, monkeypatch):
        """Regression: the vector leg must survive the pipeline param splat.

        Before task-256 the parallel step called search_semantic(**config)
        with the full pipeline params; the duplicated top_k raised TypeError
        inside the leg, gather() swallowed it, and hybrid silently degraded
        to FTS-only. Uses the REAL search_semantic with a strict-signature
        stub RAG service (no **kwargs) so stray params fail loudly.
        """
        from types import SimpleNamespace
        from tldw_chatbook.RAG_Search import pipeline_builder_simple as pbs
        from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult as VSResult

        class StrictRagService:
            async def search(self, query, top_k=None, search_type="semantic",
                             filter_metadata=None, include_citations=None,
                             score_threshold=None):
                assert search_type == "semantic"
                return [VSResult(id="v1", score=0.9, document="vector doc", metadata={})]

        async def fake_media(app, query, top_k, keyword_filter=None):
            return [self._pipeline_result("media", "m1")]

        monkeypatch.setitem(pbs.RETRIEVAL_FUNCTIONS, 'search_media_fts5', fake_media)
        # search_semantic stays the REAL implementation.

        step_config = {
            'type': 'parallel',
            'functions': [
                {'function': 'search_media_fts5', 'config': {'top_k': 20}},
                {'function': 'search_semantic', 'config': {'top_k': 20, 'score_threshold': 0.0}},
            ],
            'merge': 'rrf_merge',
            'config': {'alpha': 1.0},
        }
        # The exact param soup perform_hybrid_rag_search feeds the pipeline
        context = {
            'app': SimpleNamespace(_rag_service=StrictRagService()),
            'query': 'q',
            'sources': {'media': True},
            'params': {
                'top_k': 5, 'max_context_length': 10000, 'chunk_size': 400,
                'chunk_overlap': 100, 'chunk_type': 'words', 'keyword_filter': None,
            },
            'results': [],
        }

        results = asyncio.run(pbs._execute_parallel_step(step_config, context))

        # alpha=1.0: the vector-leg doc must rank first with a real RRF score
        assert results[0].id == "v1"
        assert results[0].score == pytest.approx(1 / (K + 1))
        assert {r.id for r in results} == {"v1", "m1"}

    def test_perform_hybrid_rag_search_end_to_end_smoke(self):
        """perform_hybrid_rag_search runs the full RRF pipeline without mutating builtins."""
        from types import SimpleNamespace
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
            perform_hybrid_rag_search,
        )
        from tldw_chatbook.RAG_Search.pipeline_builder_simple import BUILTIN_PIPELINES
        from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult as VSResult

        class StrictRagService:
            async def search(self, query, top_k=None, search_type="semantic",
                             filter_metadata=None, include_citations=None,
                             score_threshold=None):
                return [VSResult(id="v1", score=0.9, document="vector doc", metadata={})]

        # FTS legs all guard out; the vector leg is live.
        app = SimpleNamespace(media_db=None, db_config={}, _rag_service=StrictRagService())

        results, context = asyncio.run(perform_hybrid_rag_search(
            app, "query", {"media": True},
            top_k=5, enable_rerank=False, hybrid_alpha=0.7,
        ))

        assert [r['id'] for r in results] == ["v1"]
        assert results[0]['score'] == pytest.approx(0.7 / (K + 1))
        assert results[0]['metadata']['hybrid_fusion']['vector_rank'] == 1
        assert isinstance(context, str)
        # The builtin definition must not have been mutated by the call
        parallel = next(
            s for s in BUILTIN_PIPELINES['hybrid']['steps'] if s.get('type') == 'parallel'
        )
        assert 'alpha' not in parallel.get('config', {})

    def test_vector_leg_failure_degrades_to_fts_ordering(self, monkeypatch):
        from tldw_chatbook.RAG_Search import pipeline_builder_simple as pbs

        async def fake_media(app, query, top_k, keyword_filter=None):
            return [self._pipeline_result("media", "m1"), self._pipeline_result("media", "m2")]

        async def fake_semantic(app, query, sources, **config):
            raise RuntimeError("vector store unavailable")

        monkeypatch.setitem(pbs.RETRIEVAL_FUNCTIONS, 'search_media_fts5', fake_media)
        monkeypatch.setitem(pbs.RETRIEVAL_FUNCTIONS, 'search_semantic', fake_semantic)

        step_config = {
            'type': 'parallel',
            'functions': [
                {'function': 'search_media_fts5', 'config': {'top_k': 20}},
                {'function': 'search_semantic', 'config': {'top_k': 20}},
            ],
            'merge': 'rrf_merge',
            'config': {'alpha': 0.7},
        }
        context = {
            'app': object(), 'query': 'q', 'sources': {}, 'params': {}, 'results': [],
        }

        results = asyncio.run(pbs._execute_parallel_step(step_config, context))

        assert [r.id for r in results] == ["m1", "m2"]
        assert results[0].score == pytest.approx(0.3 / (K + 1))
