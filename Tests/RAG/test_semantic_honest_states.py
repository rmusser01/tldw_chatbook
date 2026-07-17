"""
Honest semantic/hybrid availability states for pipeline search (task-250).

Covers the root fix in pipeline_functions_simple.search_semantic (lazy
initialization through the shared RAG-service factory + WHY-diagnostics
instead of a silent empty list), the retrieve-step params-splat cousin of the
task-256 parallel-step bug, the hybrid FTS-only indication, and the chat
sidebar honesty seams in chat_rag_events.get_rag_context_for_chat.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from tldw_chatbook.RAG_Search import semantic_availability as sa
from tldw_chatbook.RAG_Search import pipeline_builder_simple as pbs
from tldw_chatbook.RAG_Search import pipeline_functions_simple as pfs
from tldw_chatbook.RAG_Search.semantic_availability import (
    SEMANTIC_DIAGNOSTICS_KEY,
    SEMANTIC_EMPTY_INDEX_MESSAGE,
    SEMANTIC_REASON_DEPS_MISSING,
    SEMANTIC_REASON_INIT_FAILED,
    SEMANTIC_REASON_SEARCH_ERROR,
    SEMANTIC_STATUS_EMPTY_INDEX,
    SEMANTIC_STATUS_OK,
    SEMANTIC_STATUS_UNAVAILABLE,
    SEMANTIC_UNAVAILABLE_MESSAGES,
    resolve_semantic_rag_service,
    semantic_index_is_empty,
)

pytestmark = pytest.mark.unit


class _RagResult:
    """Minimal duck-typed RAG service result (id/score/document/metadata)."""

    def __init__(self, doc_id: str, score: float = 0.9, document: str = "vector doc"):
        self.id = doc_id
        self.score = score
        self.document = document
        self.metadata: Dict[str, Any] = {}


class StrictRagService:
    """Strict-signature stub (no **kwargs) so stray params fail loudly.

    Mirrors the task-256 regression-test contract for
    EnhancedRAGServiceV2.search: any pipeline param soup leaking through the
    call sites raises TypeError here instead of passing silently.
    """

    def __init__(self, results: Optional[List[_RagResult]] = None, stats: Any = None):
        self._results = [_RagResult("v1")] if results is None else results
        self._stats = stats
        self.search_calls: List[Dict[str, Any]] = []

    async def search(self, query, top_k=None, search_type="semantic",
                     filter_metadata=None, include_citations=None,
                     score_threshold=None):
        self.search_calls.append({"query": query, "top_k": top_k, "search_type": search_type})
        assert search_type == "semantic"
        return list(self._results)

    @property
    def vector_store(self):
        if self._stats is None:
            raise AttributeError("vector_store")
        stats = self._stats

        class _Store:
            @staticmethod
            def get_collection_stats():
                if isinstance(stats, Exception):
                    raise stats
                return stats

        return _Store()


def _deps(monkeypatch, installed: bool) -> None:
    monkeypatch.setattr(sa, "embeddings_rag_deps_installed", lambda: installed)


def _factory(monkeypatch, factory) -> None:
    monkeypatch.setattr(sa, "get_shared_rag_service", factory)


def _forbidden_factory(monkeypatch) -> None:
    def _boom(profile_name=None):
        raise AssertionError("get_shared_rag_service must not be called")
    _factory(monkeypatch, _boom)


class TestResolveSemanticRagService:
    """Shared resolver: existing service wins, deps gate first, factory off-loop."""

    def test_existing_app_service_wins_without_probes(self, monkeypatch):
        service = StrictRagService()
        app = SimpleNamespace(_rag_service=service)
        monkeypatch.setattr(
            sa, "embeddings_rag_deps_installed",
            lambda: (_ for _ in ()).throw(AssertionError("deps probe must not run")),
        )
        _forbidden_factory(monkeypatch)

        resolved, reason = asyncio.run(resolve_semantic_rag_service(app))
        assert resolved is service
        assert reason is None

    def test_deps_missing_short_circuits_before_factory(self, monkeypatch):
        app = SimpleNamespace()
        _deps(monkeypatch, False)
        _forbidden_factory(monkeypatch)

        resolved, reason = asyncio.run(resolve_semantic_rag_service(app))
        assert resolved is None
        assert reason == SEMANTIC_REASON_DEPS_MISSING

    def test_factory_raising_maps_to_init_failed(self, monkeypatch):
        app = SimpleNamespace()
        _deps(monkeypatch, True)
        _factory(monkeypatch, lambda profile_name=None: (_ for _ in ()).throw(RuntimeError("boom")))

        resolved, reason = asyncio.run(resolve_semantic_rag_service(app))
        assert resolved is None
        assert reason == SEMANTIC_REASON_INIT_FAILED

    def test_factory_returning_none_maps_to_init_failed(self, monkeypatch):
        app = SimpleNamespace()
        _deps(monkeypatch, True)
        _factory(monkeypatch, lambda profile_name=None: None)

        resolved, reason = asyncio.run(resolve_semantic_rag_service(app))
        assert resolved is None
        assert reason == SEMANTIC_REASON_INIT_FAILED

    def test_successful_init_caches_on_app(self, monkeypatch):
        service = StrictRagService()
        app = SimpleNamespace()
        _deps(monkeypatch, True)
        _factory(monkeypatch, lambda profile_name=None: service)

        resolved, reason = asyncio.run(resolve_semantic_rag_service(app))
        assert resolved is service
        assert reason is None
        assert app._rag_service is service

    def test_service_without_search_is_init_failed(self, monkeypatch):
        app = SimpleNamespace()
        _deps(monkeypatch, True)
        _factory(monkeypatch, lambda profile_name=None: object())

        resolved, reason = asyncio.run(resolve_semantic_rag_service(app))
        assert resolved is None
        assert reason == SEMANTIC_REASON_INIT_FAILED


class TestSemanticIndexIsEmpty:
    """Trustworthy-count probe: only an error-free integer 0 counts as empty."""

    def test_trustworthy_zero_is_empty(self):
        service = StrictRagService(stats={"count": 0})
        assert asyncio.run(semantic_index_is_empty(service)) is True

    @pytest.mark.parametrize("stats", [
        {"count": 0, "error": "stats failed"},
        {"count": None},
        {"count": "not-a-number"},
        "not-a-mapping",
        RuntimeError("stats probe failed"),
    ])
    def test_untrustworthy_stats_are_not_empty(self, stats):
        service = StrictRagService(stats=stats)
        assert asyncio.run(semantic_index_is_empty(service)) is False

    def test_missing_vector_store_is_not_empty(self):
        assert asyncio.run(semantic_index_is_empty(SimpleNamespace())) is False

    def test_nonzero_count_is_not_empty(self):
        service = StrictRagService(stats={"count": 3})
        assert asyncio.run(semantic_index_is_empty(service)) is False


class TestSearchSemanticHonestStates:
    """search_semantic initializes lazily and records WHY instead of silence."""

    def test_uninitialized_service_is_initialized_and_used(self, monkeypatch):
        service = StrictRagService()
        app = SimpleNamespace()
        _deps(monkeypatch, True)
        _factory(monkeypatch, lambda profile_name=None: service)

        diagnostics: Dict[str, Any] = {}
        results = asyncio.run(pfs.search_semantic(
            app, "query", {"media": True}, limit=5, diagnostics=diagnostics,
        ))

        assert [r.id for r in results] == ["v1"]
        assert app._rag_service is service
        assert service.search_calls and service.search_calls[0]["top_k"] == 5
        assert diagnostics[SEMANTIC_DIAGNOSTICS_KEY]["status"] == SEMANTIC_STATUS_OK

    def test_deps_missing_records_reason_and_does_no_heavy_work(self, monkeypatch):
        app = SimpleNamespace()
        _deps(monkeypatch, False)
        _forbidden_factory(monkeypatch)

        diagnostics: Dict[str, Any] = {}
        results = asyncio.run(pfs.search_semantic(
            app, "query", {}, diagnostics=diagnostics,
        ))

        assert results == []
        state = diagnostics[SEMANTIC_DIAGNOSTICS_KEY]
        assert state["status"] == SEMANTIC_STATUS_UNAVAILABLE
        assert state["reason"] == SEMANTIC_REASON_DEPS_MISSING
        assert state["message"] == SEMANTIC_UNAVAILABLE_MESSAGES[SEMANTIC_REASON_DEPS_MISSING]
        assert "embeddings" in state["message"]

    def test_init_failure_records_reason_instead_of_crashing(self, monkeypatch):
        app = SimpleNamespace()
        _deps(monkeypatch, True)
        _factory(monkeypatch, lambda profile_name=None: (_ for _ in ()).throw(RuntimeError("no model")))

        diagnostics: Dict[str, Any] = {}
        results = asyncio.run(pfs.search_semantic(app, "query", {}, diagnostics=diagnostics))

        assert results == []
        state = diagnostics[SEMANTIC_DIAGNOSTICS_KEY]
        assert state["status"] == SEMANTIC_STATUS_UNAVAILABLE
        assert state["reason"] == SEMANTIC_REASON_INIT_FAILED

    def test_zero_results_over_trustworthy_empty_index_is_distinct(self, monkeypatch):
        service = StrictRagService(results=[], stats={"count": 0})
        app = SimpleNamespace(_rag_service=service)

        diagnostics: Dict[str, Any] = {}
        results = asyncio.run(pfs.search_semantic(app, "query", {}, diagnostics=diagnostics))

        assert results == []
        state = diagnostics[SEMANTIC_DIAGNOSTICS_KEY]
        assert state["status"] == SEMANTIC_STATUS_EMPTY_INDEX
        assert state["message"] == SEMANTIC_EMPTY_INDEX_MESSAGE

    def test_zero_results_with_untrustworthy_stats_stays_generic(self, monkeypatch):
        service = StrictRagService(results=[], stats={"count": 0, "error": "broken"})
        app = SimpleNamespace(_rag_service=service)

        diagnostics: Dict[str, Any] = {}
        results = asyncio.run(pfs.search_semantic(app, "query", {}, diagnostics=diagnostics))

        assert results == []
        assert diagnostics[SEMANTIC_DIAGNOSTICS_KEY]["status"] == SEMANTIC_STATUS_OK

    def test_legacy_call_without_diagnostics_does_not_crash(self, monkeypatch):
        app = SimpleNamespace()
        _deps(monkeypatch, False)
        _forbidden_factory(monkeypatch)

        assert asyncio.run(pfs.search_semantic(app, "query", {})) == []


class TestRetrieveStepSplatRegression:
    """The retrieve-step cousin of the task-256 parallel-step params-splat bug.

    Before task-250 the retrieve step called search_semantic(**config) with
    the full pipeline params; the duplicated top_k (and chunk_* soup) raised
    TypeError inside the RAG service call, so the pure 'semantic' pipeline
    could never return vector results. Mirrors the #681 regression test:
    REAL search_semantic + strict-signature stub service.
    """

    # The exact param soup perform_full_rag_pipeline feeds the pipeline
    SEMANTIC_PARAM_SOUP = {
        'top_k': 5, 'max_context_length': 10000, 'chunk_size': 400,
        'chunk_overlap': 100, 'chunk_type': 'words',
        'include_metadata': True, 'include_citations': True,
    }

    def test_retrieve_step_forwards_only_supported_kwargs(self):
        service = StrictRagService()
        context = {
            'app': SimpleNamespace(_rag_service=service),
            'query': 'q',
            'sources': {'media': True},
            'params': dict(self.SEMANTIC_PARAM_SOUP),
            'results': [],
            'diagnostics': {},
        }
        step_config = {'type': 'retrieve', 'function': 'search_semantic',
                       'config': {'score_threshold': 0.0}}

        results = asyncio.run(pbs._execute_retrieve_step(step_config, context))

        assert [r.id for r in results] == ["v1"]
        assert service.search_calls[0]["top_k"] == 5
        assert context['diagnostics'][SEMANTIC_DIAGNOSTICS_KEY]["status"] == SEMANTIC_STATUS_OK

    def test_perform_full_rag_pipeline_vector_results_live(self):
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
            perform_full_rag_pipeline,
        )

        app = SimpleNamespace(media_db=None, db_config={}, _rag_service=StrictRagService())
        diagnostics: Dict[str, Any] = {}

        results, context = asyncio.run(perform_full_rag_pipeline(
            app, "query", {"media": True},
            top_k=5, enable_rerank=False, diagnostics=diagnostics,
        ))

        assert [r['id'] for r in results] == ["v1"]
        assert "vector doc" in context
        assert diagnostics[SEMANTIC_DIAGNOSTICS_KEY]["status"] == SEMANTIC_STATUS_OK

    def test_parallel_search_helper_vector_leg_live(self):
        """The registry parallel_search helper had the same splat bug."""
        service = StrictRagService()
        app = SimpleNamespace(media_db=None, db_config={}, _rag_service=service)

        diagnostics: Dict[str, Any] = {}
        results = asyncio.run(pfs.parallel_search(
            app, "q", {"media": True},
            [{'function': 'search_semantic', 'config': {'top_k': 3}}],
            diagnostics=diagnostics,
        ))

        assert [r.id for r in results] == ["v1"]
        assert service.search_calls[0]["top_k"] == 3


class TestHybridFtsOnlyIndication:
    """Hybrid must say when its results are FTS-only (vector leg unavailable)."""

    class _FakeMediaDB:
        @staticmethod
        def search_media_db(search_query, search_fields, page, results_per_page,
                            include_trash):
            return ([{'id': 1, 'title': 'Doc', 'content': 'keyword content'}], 1)

    def test_parallel_step_records_search_error_when_vector_leg_raises(self, monkeypatch):
        async def raising_semantic(app, query, sources, **config):
            raise RuntimeError("vector store exploded")

        monkeypatch.setitem(pbs.RETRIEVAL_FUNCTIONS, 'search_semantic', raising_semantic)

        step_config = {
            'type': 'parallel',
            'functions': [
                {'function': 'search_semantic', 'config': {'top_k': 5}},
            ],
        }
        context = {
            'app': SimpleNamespace(), 'query': 'q', 'sources': {},
            'params': {}, 'results': [], 'diagnostics': {},
        }

        results = asyncio.run(pbs._execute_parallel_step(step_config, context))

        assert results == []
        state = context['diagnostics'][SEMANTIC_DIAGNOSTICS_KEY]
        assert state["status"] == SEMANTIC_STATUS_UNAVAILABLE
        assert state["reason"] == SEMANTIC_REASON_SEARCH_ERROR

    def test_hybrid_search_records_deps_missing_but_keeps_fts_results(self, monkeypatch):
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
            perform_hybrid_rag_search,
        )

        _deps(monkeypatch, False)
        _forbidden_factory(monkeypatch)
        app = SimpleNamespace(media_db=self._FakeMediaDB(), db_config={})

        diagnostics: Dict[str, Any] = {}
        results, context = asyncio.run(perform_hybrid_rag_search(
            app, "keyword", {"media": True},
            top_k=5, enable_rerank=False, diagnostics=diagnostics,
        ))

        assert [r['id'] for r in results] == ["1"]
        state = diagnostics[SEMANTIC_DIAGNOSTICS_KEY]
        assert state["status"] == SEMANTIC_STATUS_UNAVAILABLE
        assert state["reason"] == SEMANTIC_REASON_DEPS_MISSING


class _MockWidget:
    def __init__(self, value):
        self.value = value


class _ChatMockApp:
    """Minimal query_one/notify surface for get_rag_context_for_chat."""

    def __init__(self, search_mode: str):
        self._widgets = {
            "#chat-rag-enable-checkbox": _MockWidget(True),
            "#chat-rag-plain-enable-checkbox": _MockWidget(False),
            "#chat-rag-search-mode": _MockWidget(search_mode),
            "#chat-rag-search-media-checkbox": _MockWidget(True),
            "#chat-rag-search-conversations-checkbox": _MockWidget(False),
            "#chat-rag-search-notes-checkbox": _MockWidget(False),
            "#chat-rag-keyword-filter": _MockWidget(""),
            "#chat-rag-top-k": _MockWidget("5"),
            "#chat-rag-max-context-length": _MockWidget("10000"),
            "#chat-rag-rerank-enable-checkbox": _MockWidget(False),
            "#chat-rag-reranker-model": _MockWidget("flashrank"),
            "#chat-rag-chunk-size": _MockWidget("400"),
            "#chat-rag-chunk-overlap": _MockWidget("100"),
            "#chat-rag-chunk-type": _MockWidget("words"),
            "#chat-rag-include-metadata-checkbox": _MockWidget(False),
        }
        self.notifications: List[tuple] = []

    def query_one(self, selector):
        return self._widgets[selector]

    def notify(self, message, severity="information", **kwargs):
        self.notifications.append((message, severity))


class TestChatSidebarHonesty:
    """get_rag_context_for_chat surfaces the semantic-leg state to the user."""

    def test_semantic_mode_fallback_to_plain_is_notified(self, monkeypatch):
        from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events as cre

        app = _ChatMockApp("semantic")

        async def fake_resolver(app_arg, profile_name=None):
            return None, SEMANTIC_REASON_DEPS_MISSING

        plain_calls = []

        async def fake_plain(app_arg, query, sources, **kwargs):
            plain_calls.append(query)
            return ([], "plain context")

        monkeypatch.setattr(cre, "resolve_semantic_rag_service", fake_resolver)
        monkeypatch.setattr(cre, "perform_plain_rag_search", fake_plain)

        context = asyncio.run(cre.get_rag_context_for_chat(app, "hello"))

        assert context is not None and "plain context" in context
        assert plain_calls == ["hello"]
        assert any(
            "keyword" in message.lower() and "embeddings" in message.lower()
            for message, _severity in app.notifications
        ), app.notifications

    def test_hybrid_mode_fts_only_is_notified(self, monkeypatch):
        from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events as cre

        app = _ChatMockApp("hybrid")

        async def fake_hybrid(app_arg, query, sources, diagnostics=None, **kwargs):
            if diagnostics is not None:
                diagnostics[SEMANTIC_DIAGNOSTICS_KEY] = {
                    "status": SEMANTIC_STATUS_UNAVAILABLE,
                    "reason": SEMANTIC_REASON_INIT_FAILED,
                    "message": SEMANTIC_UNAVAILABLE_MESSAGES[SEMANTIC_REASON_INIT_FAILED],
                }
            return ([{"id": "1"}], "fts context")

        monkeypatch.setattr(cre, "perform_hybrid_rag_search", fake_hybrid)

        context = asyncio.run(cre.get_rag_context_for_chat(app, "hello"))

        assert context is not None and "fts context" in context
        assert any(
            "keyword-only" in message.lower()
            for message, _severity in app.notifications
        ), app.notifications

    def test_semantic_mode_empty_index_is_notified(self, monkeypatch):
        from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events as cre

        app = _ChatMockApp("semantic")
        service = StrictRagService()

        async def fake_resolver(app_arg, profile_name=None):
            return service, None

        async def fake_full(app_arg, query, sources, diagnostics=None, **kwargs):
            if diagnostics is not None:
                diagnostics[SEMANTIC_DIAGNOSTICS_KEY] = {
                    "status": SEMANTIC_STATUS_EMPTY_INDEX,
                    "message": SEMANTIC_EMPTY_INDEX_MESSAGE,
                }
            return ([], "")

        monkeypatch.setattr(cre, "resolve_semantic_rag_service", fake_resolver)
        monkeypatch.setattr(cre, "perform_full_rag_pipeline", fake_full)

        context = asyncio.run(cre.get_rag_context_for_chat(app, "hello"))

        assert context is None
        assert any(
            "index has no content" in message.lower()
            for message, _severity in app.notifications
        ), app.notifications
