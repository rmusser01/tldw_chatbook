"""Tests for task-4/task-5: pipeline legs self-enforce RAG retrieval scope,
and the chat entry point wires resolution + the EMPTY short-circuit end to end.

Covers the ``rag-scope-narrowing`` program's pipeline-leg enforcement layer:
media/notes FTS legs restrict to the scope's id allowlist (or return ``[]``
without querying the DB when their source type is absent from an active
scope); the conversations leg is excluded outright under any active scope
(spec decision D5) and records a diagnostic; the semantic leg runs one store
query per source_type present in the scope and merges by score. All four
legs must also inherit enforcement identically when driven through
``execute_pipeline`` (builtin/parallel pipeline shape), proving
self-enforcement rather than caller-side leg-skipping (the task-250 lesson).

Task-5 adds the caller-side entry point: ``Event_Handlers.Chat_Events.
chat_rag_events.get_rag_context_for_chat`` resolves the effective scope
before any pipeline runs, seeds it into scoped searches, and short-circuits
entirely on an EMPTY scope (task-4's legs deliberately treat EMPTY the same
as unscoped, so the caller must never let one reach a leg call).

Real in-memory-adjacent (tmp_path file-backed) DBs are used throughout,
mirroring ``Tests/RAG_Search/test_pipeline_notes_search.py`` and
``Tests/RAG/test_semantic_honest_states.py``'s fixture patterns.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from tldw_chatbook.Chat.rag_scope import (
    EffectiveScope,
    RagScope,
    SCOPE_REASON_CONVERSATIONS_EXCLUDED,
    SCOPE_REASON_EMPTY,
    SCOPE_STATUS_EMPTY,
    SCOPE_STATUS_EXCLUDED,
    ScopeItem,
    SOURCE_TYPE_MEDIA,
    SOURCE_TYPE_NOTE,
    build_semantic_allowlists,
    media_id_params,
    note_id_params,
    write_conversation_scope,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events as cre
from tldw_chatbook.RAG_Search import pipeline_builder_simple as pbs
from tldw_chatbook.RAG_Search import pipeline_functions_simple as pfs
from tldw_chatbook.RAG_Search.pipeline_functions_simple import SCOPE_DIAGNOSTICS_KEY

pytestmark = pytest.mark.unit

_UNSCOPED = EffectiveScope(state="unscoped", allowlist={}, cause=None)


def _scoped(**allowlist: set) -> EffectiveScope:
    """Build a scoped EffectiveScope from ``source_type=ids`` kwargs."""
    return EffectiveScope(
        state="scoped",
        allowlist={k: frozenset(v) for k, v in allowlist.items()},
        cause=None,
    )


class _App:
    """App double exposing the seams the pipeline legs read."""

    def __init__(self, media_db=None, chachanotes_db=None, rag_service=None):
        self.media_db = media_db
        self.chachanotes_db = chachanotes_db
        if rag_service is not None:
            self._rag_service = rag_service


class _RagResult:
    """Minimal duck-typed RAG service result (id/score/document/metadata)."""

    def __init__(self, doc_id: str, score: float = 0.9, document: str = "doc"):
        self.id = doc_id
        self.score = score
        self.document = document
        self.metadata: Dict[str, Any] = {}


class _SpyRagService:
    """rag_service double: records every ``search()`` call's kwargs and
    returns one canned result list per call, consumed in call order."""

    def __init__(self, results_by_call: Optional[List[List[_RagResult]]] = None):
        self._results_by_call = list(results_by_call or [])
        self.search_calls: List[Dict[str, Any]] = []

    async def search(
        self,
        query,
        top_k=None,
        search_type="semantic",
        filter_metadata=None,
        include_citations=None,
        score_threshold=None,
        *,
        metadata_allowlist=None,
    ):
        self.search_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "search_type": search_type,
                "metadata_allowlist": metadata_allowlist,
            }
        )
        if self._results_by_call:
            return self._results_by_call.pop(0)
        return []


class _RefusingMediaDB:
    """Proves the media leg short-circuits without ever querying the DB."""

    def search_media_db(self, *args, **kwargs):
        raise AssertionError("search_media_db must not be called")


class _RefusingChaChaDB:
    """Proves the notes leg short-circuits without resolving/querying the DB."""

    def search_notes(self, *args, **kwargs):
        raise AssertionError("search_notes must not be called")


# === Fixtures ===


@pytest.fixture()
def media_db(tmp_path):
    db = MediaDatabase(tmp_path / "media.db", client_id="task4-test")
    yield db
    try:
        db.close_connection()
    except Exception:
        pass


@pytest.fixture()
def cha_db(tmp_path):
    return CharactersRAGDB(tmp_path / "cha.db", client_id="task4-test")


def _seed_media(db: MediaDatabase, n: int = 3) -> List[str]:
    ids = []
    for i in range(n):
        media_id, _uuid, _msg = db.add_media_with_keywords(
            title=f"Doc {i}",
            media_type="document",
            content=f"zanzibarite crystal sample {i}",
            keywords=["test"],
        )
        ids.append(str(media_id))
    return ids


def _seed_notes(db: CharactersRAGDB, n: int = 3) -> List[str]:
    ids = []
    for i in range(n):
        note_id = db.add_note(
            title=f"Note {i}", content=f"zanzibarite crystal sample {i}"
        )
        ids.append(note_id)
    return ids


# === rag_scope.py helpers ===


class TestScopeHelpers:
    def test_media_id_params_not_scoped_is_none(self):
        assert media_id_params(_UNSCOPED) is None
        empty = EffectiveScope(state="empty", allowlist={}, cause="deleted-items")
        assert media_id_params(empty) is None

    def test_media_id_params_type_absent_under_scope_is_none(self):
        eff = _scoped(note={"n1"})
        assert media_id_params(eff) is None

    def test_media_id_params_returns_sorted_ids(self):
        eff = _scoped(media={"9", "2", "42"})
        assert media_id_params(eff) == ["2", "42", "9"]  # lexicographic sort

    def test_note_id_params_mirrors_media(self):
        assert note_id_params(_UNSCOPED) is None
        assert note_id_params(_scoped(media={"1"})) is None
        assert note_id_params(_scoped(note={"b", "a"})) == ["a", "b"]

    def test_build_semantic_allowlists_not_scoped_is_none(self):
        assert build_semantic_allowlists(_UNSCOPED) is None

    def test_build_semantic_allowlists_one_entry_per_type(self):
        eff = _scoped(media={"1", "2"}, note={"n1"})
        allowlists = build_semantic_allowlists(eff)
        assert allowlists == [
            {"source_type": {SOURCE_TYPE_MEDIA}, "source_id": {"1", "2"}},
            {"source_type": {SOURCE_TYPE_NOTE}, "source_id": {"n1"}},
        ]

    def test_build_semantic_allowlists_single_type(self):
        eff = _scoped(media={"1"})
        assert build_semantic_allowlists(eff) == [
            {"source_type": {SOURCE_TYPE_MEDIA}, "source_id": {"1"}}
        ]


# === Media leg ===


class TestMediaLegScopeEnforcement:
    @pytest.mark.asyncio
    async def test_scoped_media_search_returns_only_allowlisted_ids(self, media_db):
        ids = _seed_media(media_db)
        app = _App(media_db=media_db)
        eff = _scoped(media={ids[1]})

        results = await pfs.search_media_fts5(
            app, "zanzibarite", limit=10, scope=eff
        )

        assert [r.id for r in results] == [ids[1]]

    @pytest.mark.asyncio
    async def test_media_absent_from_allowlist_short_circuits(self, media_db):
        _seed_media(media_db)
        # A refusing DB double proves no query is ever issued.
        app = _App(media_db=_RefusingMediaDB())
        eff = _scoped(note={"n1"})  # media type absent

        results = await pfs.search_media_fts5(
            app, "zanzibarite", limit=10, scope=eff
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_unscoped_media_search_is_zero_drift(self, media_db):
        ids = _seed_media(media_db)
        app = _App(media_db=media_db)

        no_kwarg = await pfs.search_media_fts5(app, "zanzibarite", limit=10)
        explicit_none = await pfs.search_media_fts5(
            app, "zanzibarite", limit=10, scope=None
        )
        explicit_unscoped = await pfs.search_media_fts5(
            app, "zanzibarite", limit=10, scope=_UNSCOPED
        )

        assert {r.id for r in no_kwarg} == set(ids)
        assert [r.id for r in no_kwarg] == [r.id for r in explicit_none]
        assert [r.id for r in no_kwarg] == [r.id for r in explicit_unscoped]


# === Notes leg ===


class TestNotesLegScopeEnforcement:
    @pytest.mark.asyncio
    async def test_scoped_notes_search_returns_only_allowlisted_ids(self, cha_db):
        ids = _seed_notes(cha_db)
        app = _App(chachanotes_db=cha_db)
        eff = _scoped(note={ids[2]})

        results = await pfs.search_notes_fts5(
            app, "zanzibarite", limit=10, scope=eff
        )

        assert [r.id for r in results] == [ids[2]]

    @pytest.mark.asyncio
    async def test_notes_absent_from_allowlist_short_circuits(self, cha_db):
        _seed_notes(cha_db)
        app = _App(chachanotes_db=_RefusingChaChaDB())
        eff = _scoped(media={"1"})  # note type absent

        results = await pfs.search_notes_fts5(
            app, "zanzibarite", limit=10, scope=eff
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_unscoped_notes_search_is_zero_drift(self, cha_db):
        ids = _seed_notes(cha_db)
        app = _App(chachanotes_db=cha_db)

        no_kwarg = await pfs.search_notes_fts5(app, "zanzibarite", limit=10)
        explicit_none = await pfs.search_notes_fts5(
            app, "zanzibarite", limit=10, scope=None
        )

        assert {r.id for r in no_kwarg} == set(ids)
        assert [r.id for r in no_kwarg] == [r.id for r in explicit_none]


# === Conversations leg ===


class TestConversationsLegExclusion:
    @pytest.mark.asyncio
    async def test_scoped_conversations_leg_excluded_and_diagnosed(self, cha_db):
        conv_id = cha_db.add_conversation({"title": "Conversation"})
        cha_db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "User",
                "content": "zanzibarite discussion",
            }
        )
        app = _App(chachanotes_db=cha_db)
        eff = _scoped(media={"1"})
        diagnostics: Dict[str, Any] = {}

        results = await pfs.search_conversations_fts5(
            app, "zanzibarite", limit=10, scope=eff, diagnostics=diagnostics
        )

        assert results == []
        assert diagnostics[SCOPE_DIAGNOSTICS_KEY] == {
            "status": SCOPE_STATUS_EXCLUDED,
            "reason": SCOPE_REASON_CONVERSATIONS_EXCLUDED,
        }

    @pytest.mark.asyncio
    async def test_unscoped_conversations_search_is_zero_drift(self, cha_db):
        conv_id = cha_db.add_conversation({"title": "Conversation"})
        cha_db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "User",
                "content": "zanzibarite discussion",
            }
        )
        app = _App(chachanotes_db=cha_db)
        diagnostics: Dict[str, Any] = {}

        no_kwarg = await pfs.search_conversations_fts5(app, "zanzibarite", limit=10)
        explicit_none = await pfs.search_conversations_fts5(
            app, "zanzibarite", limit=10, scope=None, diagnostics=diagnostics
        )

        assert [r.id for r in no_kwarg] == [conv_id]
        assert [r.id for r in no_kwarg] == [r.id for r in explicit_none]
        assert diagnostics == {}  # untouched when not scoped


# === Semantic leg ===


class TestSemanticLegAllowlists:
    @pytest.mark.asyncio
    async def test_scoped_semantic_runs_one_query_per_type_and_merges_by_score(self):
        spy = _SpyRagService(
            results_by_call=[
                [_RagResult("m1", score=0.4)],
                [_RagResult("n1", score=0.9)],
            ]
        )
        app = _App(rag_service=spy)
        eff = _scoped(media={"m1"}, note={"n1"})

        results = await pfs.search_semantic(
            app, "query", {"media": True}, limit=10, scope=eff
        )

        assert len(spy.search_calls) == 2
        assert spy.search_calls[0]["metadata_allowlist"] == {
            "source_type": {SOURCE_TYPE_MEDIA},
            "source_id": {"m1"},
        }
        assert spy.search_calls[1]["metadata_allowlist"] == {
            "source_type": {SOURCE_TYPE_NOTE},
            "source_id": {"n1"},
        }
        # Merged by score descending: n1 (0.9) before m1 (0.4).
        assert [r.id for r in results] == ["n1", "m1"]

    @pytest.mark.asyncio
    async def test_scoped_semantic_trims_merged_results_to_limit(self):
        spy = _SpyRagService(
            results_by_call=[
                [_RagResult("m1", score=0.4)],
                [_RagResult("n1", score=0.9)],
            ]
        )
        app = _App(rag_service=spy)
        eff = _scoped(media={"m1"}, note={"n1"})

        results = await pfs.search_semantic(
            app, "query", {"media": True}, limit=1, scope=eff
        )

        assert [r.id for r in results] == ["n1"]

    @pytest.mark.asyncio
    async def test_unscoped_semantic_is_zero_drift(self):
        spy = _SpyRagService(results_by_call=[[_RagResult("v1")]])
        app = _App(rag_service=spy)

        results = await pfs.search_semantic(app, "query", {"media": True}, limit=10)

        assert len(spy.search_calls) == 1
        assert spy.search_calls[0]["metadata_allowlist"] is None
        assert [r.id for r in results] == ["v1"]

    @pytest.mark.asyncio
    async def test_cross_type_id_collision_both_survive(self):
        """Scope contains media id "42" AND note id "42"; both results appear in output."""
        spy = _SpyRagService(
            results_by_call=[
                [_RagResult("42", score=0.5)],  # media type, id="42"
                [_RagResult("42", score=0.7)],  # note type, id="42"
            ]
        )
        app = _App(rag_service=spy)
        eff = _scoped(media={"42"}, note={"42"})

        results = await pfs.search_semantic(
            app, "query", {"media": True}, limit=10, scope=eff
        )

        # Both results should appear (no dedup by source_id alone).
        assert len(results) == 2
        result_ids = [r.id for r in results]
        assert result_ids.count("42") == 2
        # Note result (0.7) should come before media result (0.5) due to higher score.
        assert results[0].score == 0.7
        assert results[1].score == 0.5

    @pytest.mark.asyncio
    async def test_semantic_merge_tie_break_deterministic(self):
        """Two results with equal scores from different types; order is deterministic."""
        spy = _SpyRagService(
            results_by_call=[
                [_RagResult("m1", score=0.5)],
                [_RagResult("n1", score=0.5)],  # Same score as m1
            ]
        )
        app = _App(rag_service=spy)
        eff = _scoped(media={"m1"}, note={"n1"})

        # Run twice to ensure determinism
        results_1 = await pfs.search_semantic(
            app, "query", {"media": True}, limit=10, scope=eff
        )

        spy2 = _SpyRagService(
            results_by_call=[
                [_RagResult("m1", score=0.5)],
                [_RagResult("n1", score=0.5)],
            ]
        )
        app2 = _App(rag_service=spy2)
        results_2 = await pfs.search_semantic(
            app2, "query", {"media": True}, limit=10, scope=eff
        )

        # Order should be deterministic and match
        ids_1 = [r.id for r in results_1]
        ids_2 = [r.id for r in results_2]
        assert ids_1 == ids_2
        # Per alphabetical type ordering (media < note): media should come first
        assert ids_1 == ["m1", "n1"]

    @pytest.mark.asyncio
    async def test_single_type_scope_issues_one_search_call(self):
        """Scope with only media ids; spy receives exactly ONE search call."""
        spy = _SpyRagService(
            results_by_call=[
                [_RagResult("m1", score=0.5)],
            ]
        )
        app = _App(rag_service=spy)
        eff = _scoped(media={"m1"})  # Only media type

        results = await pfs.search_semantic(
            app, "query", {"media": True}, limit=10, scope=eff
        )

        # Only one search call should be issued
        assert len(spy.search_calls) == 1
        assert [r.id for r in results] == ["m1"]


# === Custom-pipeline inheritance (self-enforcement proof) ===


class TestCustomPipelineInheritance:
    """A hand-built pipeline (not one of BUILTIN_PIPELINES) still gets
    enforcement, because each leg reads context['scope'] itself rather than
    execute_pipeline special-casing scoped vs. unscoped runs."""

    @pytest.mark.asyncio
    async def test_custom_parallel_pipeline_inherits_enforcement(
        self, media_db, cha_db
    ):
        media_ids = _seed_media(media_db)
        note_ids = _seed_notes(cha_db)
        conv_id = cha_db.add_conversation({"title": "Conversation"})
        cha_db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "User",
                "content": "zanzibarite discussion",
            }
        )
        spy = _SpyRagService(
            results_by_call=[
                [_RagResult(media_ids[0], score=0.5)],
                [_RagResult(note_ids[0], score=0.7)],
            ]
        )
        app = SimpleNamespace(
            media_db=media_db, chachanotes_db=cha_db, _rag_service=spy
        )
        eff = _scoped(media={media_ids[0]}, note={note_ids[0]})
        diagnostics: Dict[str, Any] = {}

        config = {
            "name": "Custom Test Pipeline",
            "steps": [
                {
                    "type": "parallel",
                    "functions": [
                        {"function": "search_media_fts5", "config": {"top_k": 10}},
                        {"function": "search_notes_fts5", "config": {"top_k": 10}},
                        {
                            "function": "search_conversations_fts5",
                            "config": {"top_k": 10},
                        },
                        {"function": "search_semantic", "config": {"top_k": 10}},
                    ],
                },
                {"type": "format", "function": "format_as_context"},
            ],
        }

        results, _formatted = await pbs.execute_pipeline(
            config,
            app,
            "zanzibarite",
            {"media": True, "conversations": True, "notes": True},
            diagnostics=diagnostics,
            scope=eff,
        )

        by_source = {r["source"]: r["id"] for r in results}
        assert by_source["media"] == media_ids[0]
        assert by_source["note"] == note_ids[0]
        assert "conversation" not in by_source
        assert diagnostics[SCOPE_DIAGNOSTICS_KEY]["reason"] == (
            SCOPE_REASON_CONVERSATIONS_EXCLUDED
        )
        assert len(spy.search_calls) == 2  # one per source_type, not one flat call

    @pytest.mark.asyncio
    async def test_custom_pipeline_unscoped_is_zero_drift(self, media_db, cha_db):
        media_ids = _seed_media(media_db)
        note_ids = _seed_notes(cha_db)
        app = SimpleNamespace(media_db=media_db, chachanotes_db=cha_db)

        config = {
            "name": "Custom Test Pipeline",
            "steps": [
                {
                    "type": "parallel",
                    "functions": [
                        {"function": "search_media_fts5", "config": {"top_k": 10}},
                        {"function": "search_notes_fts5", "config": {"top_k": 10}},
                    ],
                },
                {"type": "format", "function": "format_as_context"},
            ],
        }

        with_scope_key, _ = await pbs.execute_pipeline(
            config, app, "zanzibarite", {"media": True, "notes": True}, scope=None
        )
        without_scope_key, _ = await pbs.execute_pipeline(
            config, app, "zanzibarite", {"media": True, "notes": True}
        )

        ids_with = sorted(r["id"] for r in with_scope_key)
        ids_without = sorted(r["id"] for r in without_scope_key)
        assert ids_with == ids_without == sorted(media_ids + note_ids)


# === Task-5: chat entry point (get_rag_context_for_chat) end to end ===


class _MockWidget:
    def __init__(self, value):
        self.value = value


class _ChatMockApp:
    """Minimal query_one/notify surface for ``get_rag_context_for_chat``,
    plus the real DB handles the scope-resolution wiring and pipeline legs
    read directly off the app (mirrors ``Tests/RAG/test_semantic_honest_states
    .py``'s ``_ChatMockApp`` fixture, extended with ``media_db``/
    ``chachanotes_db`` since this task drives real seeded DBs end to end)."""

    def __init__(
        self,
        search_mode: str = "plain",
        *,
        media_db: Any = None,
        chachanotes_db: Any = None,
        rag_service: Any = None,
    ):
        self._widgets = {
            "#chat-rag-enable-checkbox": _MockWidget(True),
            "#chat-rag-plain-enable-checkbox": _MockWidget(search_mode == "plain"),
            "#chat-rag-search-mode": _MockWidget(search_mode),
            "#chat-rag-search-media-checkbox": _MockWidget(True),
            "#chat-rag-search-conversations-checkbox": _MockWidget(False),
            "#chat-rag-search-notes-checkbox": _MockWidget(True),
            "#chat-rag-keyword-filter": _MockWidget(""),
            "#chat-rag-top-k": _MockWidget("10"),
            "#chat-rag-max-context-length": _MockWidget("10000"),
            "#chat-rag-rerank-enable-checkbox": _MockWidget(False),
            "#chat-rag-reranker-model": _MockWidget("flashrank"),
            "#chat-rag-chunk-size": _MockWidget("400"),
            "#chat-rag-chunk-overlap": _MockWidget("100"),
            "#chat-rag-chunk-type": _MockWidget("words"),
            "#chat-rag-include-metadata-checkbox": _MockWidget(False),
        }
        self.media_db = media_db
        self.chachanotes_db = chachanotes_db
        if rag_service is not None:
            self._rag_service = rag_service
        self.notifications: List[tuple] = []

    def query_one(self, selector):
        return self._widgets[selector]

    def notify(self, message, severity="information", **kwargs):
        self.notifications.append((message, severity))


class TestChatEntryPointScopedE2E:
    """Driving ``get_rag_context_for_chat`` with seeded DBs and a scoped,
    persisted conversation returns context built ONLY from in-scope content
    (design spec section 5's end-to-end requirement)."""

    @pytest.mark.asyncio
    async def test_scoped_send_context_contains_only_in_scope_content(
        self, media_db, cha_db, monkeypatch
    ):
        media_ids = _seed_media(media_db, n=3)
        note_ids = _seed_notes(cha_db, n=3)
        conv_id = cha_db.add_conversation({"title": "Scoped conversation"})

        scope = RagScope(
            items=(
                ScopeItem(SOURCE_TYPE_MEDIA, media_ids[1]),
                ScopeItem(SOURCE_TYPE_NOTE, note_ids[2]),
            ),
            updated_at="2026-07-21T00:00:00+00:00",
        )
        write_conversation_scope(cha_db, conv_id, scope)

        session = SimpleNamespace(persisted_conversation_id=conv_id)
        monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

        app = _ChatMockApp(
            search_mode="plain", media_db=media_db, chachanotes_db=cha_db
        )

        context = await cre.get_rag_context_for_chat(app, "zanzibarite")

        assert context is not None
        # In-scope content present.
        assert "Doc 1" in context
        assert "Note 2" in context
        # Out-of-scope content absent (both other media and both other notes).
        for excluded_title in ("Doc 0", "Doc 2", "Note 0", "Note 1"):
            assert excluded_title not in context, context

    @pytest.mark.asyncio
    async def test_unpersisted_session_with_no_holder_is_unscoped(
        self, media_db, cha_db, monkeypatch
    ):
        """A native-Console session that has not been persisted yet, and
        carries no ``SessionScopeHolder``, resolves unscoped -- there is
        nowhere yet for a scope-picker UI (not built in this task) to have
        put one."""
        # Media is seeded with n=1: search_media_db does not project the
        # `content` column (a pre-existing, scope-unrelated choice), so the
        # plain pipeline's deduplicate_results step -- keyed on content --
        # collapses multiple same-score media results with >1 seeded rows
        # regardless of scope. n=1 sidesteps that collision so this test
        # stays about scope, not about that orthogonal pipeline quirk.
        _seed_media(media_db, n=1)
        _seed_notes(cha_db, n=2)

        session = SimpleNamespace(persisted_conversation_id=None)
        monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

        app = _ChatMockApp(
            search_mode="plain", media_db=media_db, chachanotes_db=cha_db
        )

        context = await cre.get_rag_context_for_chat(app, "zanzibarite")

        assert context is not None
        assert "Doc 0" in context
        for i in range(2):
            assert f"Note {i}" in context


class TestChatEntryPointEmptyScopeShortCircuit:
    """An EMPTY effective scope (all scoped ids since deleted) short-circuits
    ``get_rag_context_for_chat`` before any pipeline/leg call, and records
    the cause into diagnostics via the existing notification pathway."""

    @pytest.mark.asyncio
    async def test_empty_scope_short_circuits_with_zero_leg_calls(
        self, media_db, cha_db, monkeypatch
    ):
        conv_id = cha_db.add_conversation({"title": "Deleted-item scope"})
        # References a media id that was never seeded -- the real
        # dangling-drop existence check (_existing_ids_sync) will find
        # nothing survives, landing on EMPTY/"deleted-items".
        scope = RagScope(
            items=(ScopeItem(SOURCE_TYPE_MEDIA, "999999"),), updated_at="t1"
        )
        write_conversation_scope(cha_db, conv_id, scope)

        session = SimpleNamespace(persisted_conversation_id=conv_id)
        monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

        for fn_name in (
            "perform_plain_rag_search",
            "perform_full_rag_pipeline",
            "perform_hybrid_rag_search",
            "perform_search_with_pipeline",
        ):
            def _refuse(*args, __name=fn_name, **kwargs):
                raise AssertionError(f"{__name} must not be called on EMPTY scope")

            monkeypatch.setattr(cre, fn_name, _refuse)

        captured: Dict[str, Any] = {}
        original_notify = cre._notify_semantic_leg_state

        def _spy_notify(app_arg, diagnostics, results):
            captured["diagnostics"] = dict(diagnostics)
            captured["results"] = results
            return original_notify(app_arg, diagnostics, results)

        monkeypatch.setattr(cre, "_notify_semantic_leg_state", _spy_notify)

        app = _ChatMockApp(
            search_mode="plain", media_db=media_db, chachanotes_db=cha_db
        )

        context = await cre.get_rag_context_for_chat(app, "zanzibarite")

        assert context is None
        assert captured["results"] is None
        assert captured["diagnostics"][SCOPE_DIAGNOSTICS_KEY] == {
            "status": SCOPE_STATUS_EMPTY,
            "reason": SCOPE_REASON_EMPTY,
            "cause": "deleted-items",
        }
        assert any(
            "retrieval scope is empty" in message.lower()
            and "deleted-items" in message.lower()
            for message, _severity in app.notifications
        ), app.notifications


class TestChatEntryPointUnscopedZeroDrift:
    """No active native-Console session (today's real default) must resolve
    unscoped and thread ``scope=None`` through to the pipeline exactly like
    before scope resolution existed -- zero drift through the full entry
    path."""

    @pytest.mark.asyncio
    async def test_no_active_session_searches_everything(
        self, media_db, cha_db, monkeypatch
    ):
        monkeypatch.setattr(cre, "_active_console_session", lambda app: None)
        # n=1 media: see the comment in
        # TestChatEntryPointScopedE2E.test_unpersisted_session_with_no_holder_is_unscoped
        # for why >1 same-score media rows collide in deduplicate_results
        # regardless of scope (content is never projected by search_media_db).
        _seed_media(media_db, n=1)
        _seed_notes(cha_db, n=3)

        app = _ChatMockApp(
            search_mode="plain", media_db=media_db, chachanotes_db=cha_db
        )

        context = await cre.get_rag_context_for_chat(app, "zanzibarite")

        assert context is not None
        assert "Doc 0" in context
        for i in range(3):
            assert f"Note {i}" in context

    @pytest.mark.asyncio
    async def test_unscoped_pipeline_call_receives_scope_none(self, monkeypatch):
        """Direct proof of the call shape: no active session -> the pipeline
        dispatch receives ``scope=None``, matching the pre-task call shape."""
        monkeypatch.setattr(cre, "_active_console_session", lambda app: None)

        captured: Dict[str, Any] = {}

        async def _spy_plain(app_arg, query, sources, **kwargs):
            captured.update(kwargs)
            return [], "some context"

        monkeypatch.setattr(cre, "perform_plain_rag_search", _spy_plain)

        app = _ChatMockApp(search_mode="plain")

        context = await cre.get_rag_context_for_chat(app, "hello")

        assert context == "some context"
        assert "scope" in captured
        assert captured["scope"] is None
