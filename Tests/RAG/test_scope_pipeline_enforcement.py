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

import asyncio
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
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events as cre
from tldw_chatbook.RAG_Search import pipeline_builder_simple as pbs
from tldw_chatbook.RAG_Search import pipeline_functions_simple as pfs
from tldw_chatbook.RAG_Search.pipeline_functions_simple import SCOPE_DIAGNOSTICS_KEY
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

# Reuses the proven real-``TldwCli``-with-heavy-init-patched recipe already
# shared across the Console UI test suites (e.g.
# Tests/UI/test_console_internals_decomposition.py, Tests/UI/
# test_destination_shells.py) rather than duplicating it here.
from Tests.UI.test_screen_navigation import _build_test_app

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


class _RefusingConversationsDB:
    """Proves the conversations leg short-circuits without querying the DB."""

    def search_conversations_by_content(self, *args, **kwargs):
        raise AssertionError("search_conversations_by_content must not be called")

    def get_messages_for_conversations_batch(self, *args, **kwargs):
        raise AssertionError(
            "get_messages_for_conversations_batch must not be called"
        )


class _RefusingRagService:
    """Proves the semantic leg short-circuits without ever calling search()."""

    async def search(self, *args, **kwargs):
        raise AssertionError("rag_service.search must not be called")


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
        assert diagnostics[SCOPE_DIAGNOSTICS_KEY] == [
            {
                "status": SCOPE_STATUS_EXCLUDED,
                "reason": SCOPE_REASON_CONVERSATIONS_EXCLUDED,
            }
        ]

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
        scope_entries = diagnostics[SCOPE_DIAGNOSTICS_KEY]
        assert [entry["reason"] for entry in scope_entries] == [
            SCOPE_REASON_CONVERSATIONS_EXCLUDED
        ]
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


# === EMPTY-scope fail-closed defense-in-depth (PR #734 review, id 3621197384) ===


class TestEmptyScopeFailsClosedAtEachLeg:
    """The caller-side EMPTY short-circuit (``chat_rag_events.
    get_rag_context_for_chat``) is expected to keep an EMPTY
    ``EffectiveScope`` from ever reaching a leg in normal operation -- but
    every leg only checked ``scope.state == "scoped"``, so an
    ``EffectiveScope(state="empty")`` reaching a leg directly (a caller that
    forgot to short-circuit, or a hand-built pipeline) fell through to an
    UNRESTRICTED search: fail-OPEN. Each leg must instead fail CLOSED --
    return ``[]`` without ever touching its DB/service -- and record
    ``SCOPE_REASON_EMPTY`` into diagnostics exactly like the caller-side
    short-circuit does.
    """

    _EMPTY = EffectiveScope(state="empty", allowlist={}, cause="deleted-items")

    @pytest.mark.asyncio
    async def test_media_leg_empty_scope_fails_closed(self):
        app = _App(media_db=_RefusingMediaDB())
        diagnostics: Dict[str, Any] = {}

        results = await pfs.search_media_fts5(
            app, "zanzibarite", limit=10, scope=self._EMPTY, diagnostics=diagnostics
        )

        assert results == []
        assert diagnostics[SCOPE_DIAGNOSTICS_KEY] == [
            {
                "status": SCOPE_STATUS_EMPTY,
                "reason": SCOPE_REASON_EMPTY,
                "cause": "deleted-items",
            }
        ]

    @pytest.mark.asyncio
    async def test_notes_leg_empty_scope_fails_closed(self):
        app = _App(chachanotes_db=_RefusingChaChaDB())
        diagnostics: Dict[str, Any] = {}

        results = await pfs.search_notes_fts5(
            app, "zanzibarite", limit=10, scope=self._EMPTY, diagnostics=diagnostics
        )

        assert results == []
        assert diagnostics[SCOPE_DIAGNOSTICS_KEY] == [
            {
                "status": SCOPE_STATUS_EMPTY,
                "reason": SCOPE_REASON_EMPTY,
                "cause": "deleted-items",
            }
        ]

    @pytest.mark.asyncio
    async def test_conversations_leg_empty_scope_fails_closed(self):
        app = _App(chachanotes_db=_RefusingConversationsDB())
        diagnostics: Dict[str, Any] = {}

        results = await pfs.search_conversations_fts5(
            app, "zanzibarite", limit=10, scope=self._EMPTY, diagnostics=diagnostics
        )

        assert results == []
        assert diagnostics[SCOPE_DIAGNOSTICS_KEY] == [
            {
                "status": SCOPE_STATUS_EMPTY,
                "reason": SCOPE_REASON_EMPTY,
                "cause": "deleted-items",
            }
        ]

    @pytest.mark.asyncio
    async def test_semantic_leg_empty_scope_fails_closed(self):
        app = _App(rag_service=_RefusingRagService())
        diagnostics: Dict[str, Any] = {}

        results = await pfs.search_semantic(
            app,
            "query",
            {"media": True},
            limit=10,
            scope=self._EMPTY,
            diagnostics=diagnostics,
        )

        assert results == []
        assert diagnostics[SCOPE_DIAGNOSTICS_KEY] == [
            {
                "status": SCOPE_STATUS_EMPTY,
                "reason": SCOPE_REASON_EMPTY,
                "cause": "deleted-items",
            }
        ]

    @pytest.mark.asyncio
    async def test_no_diagnostics_dict_is_a_safe_no_op(self):
        """``diagnostics=None`` (the default for legacy callers) must not
        raise -- mirrors every other scope/semantic recorder's None-safe
        contract."""
        app = _App(media_db=_RefusingMediaDB())

        results = await pfs.search_media_fts5(
            app, "zanzibarite", limit=10, scope=self._EMPTY
        )

        assert results == []


class TestEmptyScopeFailsClosedThroughPipeline:
    """Same invariant, proven end to end through ``execute_pipeline`` (the
    review's own focus: ``execute_pipeline`` forwards any non-None scope to
    every leg unconditionally) -- a custom parallel pipeline built from all
    four legs must come back with zero results and zero leg-service calls
    when driven with an EMPTY scope.
    """

    @pytest.mark.asyncio
    async def test_custom_parallel_pipeline_empty_scope_returns_nothing(
        self, media_db, cha_db
    ):
        _seed_media(media_db)
        _seed_notes(cha_db)
        conv_id = cha_db.add_conversation({"title": "Conversation"})
        cha_db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "User",
                "content": "zanzibarite discussion",
            }
        )
        spy = _SpyRagService()
        app = SimpleNamespace(
            media_db=media_db, chachanotes_db=cha_db, _rag_service=spy
        )
        eff = EffectiveScope(state="empty", allowlist={}, cause="deleted-items")
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

        assert results == []
        assert spy.search_calls == []  # semantic leg never called the service
        # All four legs independently fail closed on the same EMPTY scope and
        # each records its own entry (task-9 review finding 2: appended, not
        # assigned, so one leg's entry never clobbers another's).
        scope_entries = diagnostics[SCOPE_DIAGNOSTICS_KEY]
        assert scope_entries
        expected_entry = {
            "status": SCOPE_STATUS_EMPTY,
            "reason": SCOPE_REASON_EMPTY,
            "cause": "deleted-items",
        }
        assert all(entry == expected_entry for entry in scope_entries)
        assert len(scope_entries) == 4  # media, notes, conversations, semantic


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


class TestWorkspaceScopeIntersectionE2E:
    """Task-13 (Phase 3): ``resolve_effective_scope_for_chat`` now reads the
    active session's LINKED WORKSPACE scope (previously always passed as
    ``None`` -- see the removed "Phase 3" comment) via ``app.workspace_
    registry_service.get_workspace_scope`` and intersects it with the
    conversation scope end to end -- exactly the spec's "hunt X"/"sales
    reports" workflow: a workspace's in-scope set bounds retrieval for
    every conversation inside it."""

    @pytest.mark.asyncio
    async def test_conversation_and_workspace_scopes_intersect_end_to_end(
        self, media_db, cha_db, tmp_path, monkeypatch
    ):
        media_ids = _seed_media(media_db, n=4)  # Doc 0=A, Doc 1=B, Doc 2=C, Doc 3=D
        conv_id = cha_db.add_conversation({"title": "Sales reports"})
        conv_scope = RagScope(
            items=(
                ScopeItem(SOURCE_TYPE_MEDIA, media_ids[0]),  # A
                ScopeItem(SOURCE_TYPE_MEDIA, media_ids[1]),  # B
                ScopeItem(SOURCE_TYPE_MEDIA, media_ids[2]),  # C
            ),
            updated_at="t1",
        )
        write_conversation_scope(cha_db, conv_id, conv_scope)

        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="task13-test")
        )
        registry.create_workspace(workspace_id="ws-sales", name="Sales reports")
        ws_scope = RagScope(
            items=(
                ScopeItem(SOURCE_TYPE_MEDIA, media_ids[1]),  # B
                ScopeItem(SOURCE_TYPE_MEDIA, media_ids[2]),  # C
                ScopeItem(SOURCE_TYPE_MEDIA, media_ids[3]),  # D
            ),
            updated_at="t2",
        )
        registry.set_workspace_scope("ws-sales", ws_scope)

        session = SimpleNamespace(
            persisted_conversation_id=conv_id, workspace_id="ws-sales"
        )
        monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

        app = _App(media_db=media_db, chachanotes_db=cha_db)
        app.workspace_registry_service = registry

        # Resolution is checked directly against the allowlist (rather than
        # through a full `get_rag_context_for_chat` pipeline run and
        # scanning the generated context text): `search_media_db` doesn't
        # project the `content` column, so `deduplicate_results` collapses
        # multiple same-score media rows regardless of scope (a pre-
        # existing, scope-unrelated pipeline quirk documented elsewhere in
        # this file, e.g. `TestChatEntryPointScopedE2E.
        # test_unpersisted_session_with_no_holder_is_unscoped`'s n=1
        # workaround) -- `resolve_effective_scope_for_chat` IS the entry
        # point `get_rag_context_for_chat` calls, so this is genuinely
        # end-to-end for the scope-resolution layer task-13 changed.
        effective = await cre.resolve_effective_scope_for_chat(app)

        assert effective.state == "scoped"
        # Intersection {A, B, C} ∩ {B, C, D} = {B, C}.
        assert effective.allowlist == {
            SOURCE_TYPE_MEDIA: frozenset({media_ids[1], media_ids[2]})
        }
        assert effective.cause is None

    @pytest.mark.asyncio
    async def test_workspace_only_scope_narrows_an_unscoped_conversation(
        self, media_db, cha_db, tmp_path, monkeypatch
    ):
        """A conversation with no scope of its own, inside a scoped
        workspace, still has retrieval bounded by the workspace alone
        (single-level resolution -- spec section 2)."""
        media_ids = _seed_media(media_db, n=2)
        conv_id = cha_db.add_conversation({"title": "Unscoped conversation"})

        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="task13-test")
        )
        registry.create_workspace(workspace_id="ws-hunt", name="hunt X")
        registry.set_workspace_scope(
            "ws-hunt",
            RagScope(
                items=(ScopeItem(SOURCE_TYPE_MEDIA, media_ids[0]),), updated_at="t1"
            ),
        )

        session = SimpleNamespace(
            persisted_conversation_id=conv_id, workspace_id="ws-hunt"
        )
        monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

        app = _ChatMockApp(
            search_mode="plain", media_db=media_db, chachanotes_db=cha_db
        )
        app.workspace_registry_service = registry

        context = await cre.get_rag_context_for_chat(app, "zanzibarite")

        assert context is not None
        assert "Doc 0" in context
        assert "Doc 1" not in context, context

    @pytest.mark.asyncio
    async def test_no_workspace_overlap_short_circuits_empty_with_honest_notify(
        self, media_db, cha_db, tmp_path, monkeypatch
    ):
        media_ids = _seed_media(media_db, n=2)
        conv_id = cha_db.add_conversation({"title": "Disjoint"})
        write_conversation_scope(
            cha_db,
            conv_id,
            RagScope(
                items=(ScopeItem(SOURCE_TYPE_MEDIA, media_ids[0]),), updated_at="t1"
            ),
        )

        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="task13-test")
        )
        registry.create_workspace(workspace_id="ws-other", name="Other project")
        registry.set_workspace_scope(
            "ws-other",
            RagScope(
                items=(ScopeItem(SOURCE_TYPE_MEDIA, media_ids[1]),), updated_at="t2"
            ),
        )

        session = SimpleNamespace(
            persisted_conversation_id=conv_id, workspace_id="ws-other"
        )
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

        app = _ChatMockApp(
            search_mode="plain", media_db=media_db, chachanotes_db=cha_db
        )
        app.workspace_registry_service = registry

        context = await cre.get_rag_context_for_chat(app, "zanzibarite")

        assert context is None
        assert any(
            "retrieval scope is empty" in message.lower()
            and "no-workspace-overlap" in message.lower()
            for message, _severity in app.notifications
        ), app.notifications


class TestWorkspaceScopeMemoryDbGuard:
    """Task-13 (PR #747 discipline, extended to the workspace registry's
    own DB): the workspace-scope read must apply the identical
    ``is_memory_db`` guard the conversation-scope read already does -- a
    memory-backed registry is read inline, a file-backed one is offloaded
    via ``asyncio.to_thread``.

    Verified via a fake registry double (not a genuine ``WorkspaceDB(
    ":memory:")``): unlike ``CharactersRAGDB``'s thread-local *cached*
    connection, ``WorkspaceDB`` opens a brand-new connection on every
    ``.connection()``/``.transaction()`` call and caches nothing, so a real
    ``:memory:``-backed instance cannot survive past its own ``__init__``
    (each call opens an independent, empty in-memory database) -- exactly
    why ``WorkspaceDB`` is never constructed with ``:memory:`` anywhere in
    this codebase. This proves the guard's own branch selection instead.
    """

    @pytest.mark.asyncio
    async def test_memory_backed_registry_reads_inline(
        self, media_db, monkeypatch
    ):
        """Checks the WORKSPACE-scope read's own thread, not a blanket
        ``asyncio.to_thread`` spy: the overall ``resolve_effective_scope``
        call is offloaded independently based on ``chachanotes_db``/
        ``media_db``'s memory status (unrelated to the registry), so a
        blanket spy would see calls from that separate decision too."""
        import threading

        media_ids = _seed_media(media_db, n=1)
        ws_scope = RagScope(
            items=(ScopeItem(SOURCE_TYPE_MEDIA, media_ids[0]),), updated_at="t1"
        )
        main_thread = threading.current_thread()

        class _FakeRegistry:
            db = SimpleNamespace(is_memory_db=True)

            def __init__(self):
                self.call_thread = None

            def get_workspace_scope(self, workspace_id):
                self.call_thread = threading.current_thread()
                return ws_scope

        session = SimpleNamespace(persisted_conversation_id=None, workspace_id="ws-1")
        monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

        registry = _FakeRegistry()
        app = _App(media_db=media_db)
        app.workspace_registry_service = registry

        effective = await cre.resolve_effective_scope_for_chat(app)

        assert effective.state == "scoped"
        assert effective.allowlist == {SOURCE_TYPE_MEDIA: frozenset({media_ids[0]})}
        assert registry.call_thread is main_thread, (
            "a memory-backed registry's scope read must run inline, "
            "never offloaded to a worker thread"
        )

    @pytest.mark.asyncio
    async def test_file_backed_registry_still_offloaded(
        self, media_db, monkeypatch
    ):
        import threading

        media_ids = _seed_media(media_db, n=1)
        ws_scope = RagScope(
            items=(ScopeItem(SOURCE_TYPE_MEDIA, media_ids[0]),), updated_at="t1"
        )
        main_thread = threading.current_thread()

        class _FakeRegistry:
            db = SimpleNamespace(is_memory_db=False)

            def __init__(self):
                self.call_thread = None

            def get_workspace_scope(self, workspace_id):
                self.call_thread = threading.current_thread()
                return ws_scope

        session = SimpleNamespace(persisted_conversation_id=None, workspace_id="ws-1")
        monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

        registry = _FakeRegistry()
        app = _App(media_db=media_db)
        app.workspace_registry_service = registry

        effective = await cre.resolve_effective_scope_for_chat(app)

        assert effective.state == "scoped"
        assert registry.call_thread is not None
        assert registry.call_thread is not main_thread, (
            "a file-backed registry's scope read must still be offloaded "
            "via asyncio.to_thread"
        )


class TestScopeCacheWiring:
    """Task-13: ``resolve_effective_scope_for_chat`` now consults a per-app
    ``ScopeCache`` (``chat_rag_events._scope_cache_for``) keyed on the
    ``(conversation_id, workspace_id, conv_stamp, ws_stamp)`` 4-tuple
    before re-resolving -- repeat resolution against an unchanged scope
    skips the (conv ∩ ws) intersection and the per-item dangling-drop
    existence check entirely; a stamp change on either level invalidates
    correctly (design spec section 2)."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_recompute(self, media_db, cha_db, monkeypatch):
        media_ids = _seed_media(media_db, n=1)
        conv_id = cha_db.add_conversation({"title": "Cached"})
        write_conversation_scope(
            cha_db,
            conv_id,
            RagScope(
                items=(ScopeItem(SOURCE_TYPE_MEDIA, media_ids[0]),), updated_at="t1"
            ),
        )

        session = SimpleNamespace(persisted_conversation_id=conv_id)
        monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

        app = _App(media_db=media_db, chachanotes_db=cha_db)

        calls: list[int] = []
        real_resolve = cre.resolve_effective_scope

        def _spy_resolve(*args, **kwargs):
            calls.append(1)
            return real_resolve(*args, **kwargs)

        monkeypatch.setattr(cre, "resolve_effective_scope", _spy_resolve)

        first = await cre.resolve_effective_scope_for_chat(app)
        second = await cre.resolve_effective_scope_for_chat(app)

        assert first == second
        assert first.state == "scoped"
        assert len(calls) == 1, "the second call must be served from ScopeCache"

    @pytest.mark.asyncio
    async def test_cache_miss_on_conversation_scope_stamp_change(
        self, media_db, cha_db, monkeypatch
    ):
        media_ids = _seed_media(media_db, n=2)
        conv_id = cha_db.add_conversation({"title": "Restamped"})
        write_conversation_scope(
            cha_db,
            conv_id,
            RagScope(
                items=(ScopeItem(SOURCE_TYPE_MEDIA, media_ids[0]),), updated_at="t1"
            ),
        )

        session = SimpleNamespace(persisted_conversation_id=conv_id)
        monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

        app = _App(media_db=media_db, chachanotes_db=cha_db)

        calls: list[int] = []
        real_resolve = cre.resolve_effective_scope

        def _spy_resolve(*args, **kwargs):
            calls.append(1)
            return real_resolve(*args, **kwargs)

        monkeypatch.setattr(cre, "resolve_effective_scope", _spy_resolve)

        first = await cre.resolve_effective_scope_for_chat(app)
        assert first.allowlist == {SOURCE_TYPE_MEDIA: frozenset({media_ids[0]})}

        # A genuine scope edit: new stamp, different item.
        write_conversation_scope(
            cha_db,
            conv_id,
            RagScope(
                items=(ScopeItem(SOURCE_TYPE_MEDIA, media_ids[1]),), updated_at="t2"
            ),
        )

        second = await cre.resolve_effective_scope_for_chat(app)

        assert second.allowlist == {SOURCE_TYPE_MEDIA: frozenset({media_ids[1]})}
        assert len(calls) == 2, "a stamp change must invalidate the cached entry"

    @pytest.mark.asyncio
    async def test_cache_miss_on_workspace_scope_stamp_change(
        self, media_db, monkeypatch
    ):
        """Task-13 review finding 1: the symmetric workspace-side companion
        to ``test_cache_miss_on_conversation_scope_stamp_change`` -- editing
        the LINKED WORKSPACE's scope (a new ``ws_scope.updated_at`` stamp)
        must miss the cache and re-resolve, exactly like a conversation-side
        edit does. Driven through ``resolve_effective_scope_for_chat`` (the
        real enforcement entry point) with a fake ``workspace_registry_
        service`` double (the same style already used by
        ``TestResolveEffectiveScopeMemoryDbGuard``/the memory-backed-
        registry tests above), rather than a real ``WorkspaceDB``, so the
        test can freely swap the returned scope between calls."""
        media_ids = _seed_media(media_db, n=2)
        session = SimpleNamespace(persisted_conversation_id=None, workspace_id="ws-1")
        monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

        class _FakeRegistry:
            db = SimpleNamespace(is_memory_db=True)

            def __init__(self, scope):
                self.scope = scope

            def get_workspace_scope(self, workspace_id):
                return self.scope

        registry = _FakeRegistry(
            RagScope(
                items=(ScopeItem(SOURCE_TYPE_MEDIA, media_ids[0]),), updated_at="t1"
            )
        )
        app = _App(media_db=media_db)
        app.workspace_registry_service = registry

        calls: list[int] = []
        real_resolve = cre.resolve_effective_scope

        def _spy_resolve(*args, **kwargs):
            calls.append(1)
            return real_resolve(*args, **kwargs)

        monkeypatch.setattr(cre, "resolve_effective_scope", _spy_resolve)

        first = await cre.resolve_effective_scope_for_chat(app)
        assert first.allowlist == {SOURCE_TYPE_MEDIA: frozenset({media_ids[0]})}

        # A genuine workspace scope edit: new stamp, different item.
        registry.scope = RagScope(
            items=(ScopeItem(SOURCE_TYPE_MEDIA, media_ids[1]),), updated_at="t2"
        )

        second = await cre.resolve_effective_scope_for_chat(app)

        assert second.allowlist == {SOURCE_TYPE_MEDIA: frozenset({media_ids[1]})}
        assert len(calls) == 2, (
            "a workspace scope stamp change must invalidate the cached entry"
        )

    @pytest.mark.asyncio
    async def test_cache_miss_on_conversation_relinked_to_different_workspace(
        self, media_db, monkeypatch
    ):
        """Task-13 review finding 1: the cache key is the full
        ``(conversation_id/session_id, workspace_id, conv_stamp, ws_stamp)``
        4-tuple, so re-linking the SAME session to a DIFFERENT workspace --
        even when both workspaces' scopes happen to share the identical
        ``updated_at`` stamp -- must still miss the cache and re-resolve
        (the ``workspace_id`` component alone must be enough to
        invalidate)."""
        media_ids = _seed_media(media_db, n=2)
        session = SimpleNamespace(persisted_conversation_id=None, workspace_id="ws-a")
        monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

        same_stamp = "t1"

        class _MultiWorkspaceRegistry:
            db = SimpleNamespace(is_memory_db=True)

            def __init__(self, scopes_by_workspace):
                self.scopes_by_workspace = scopes_by_workspace

            def get_workspace_scope(self, workspace_id):
                return self.scopes_by_workspace[workspace_id]

        registry = _MultiWorkspaceRegistry(
            {
                "ws-a": RagScope(
                    items=(ScopeItem(SOURCE_TYPE_MEDIA, media_ids[0]),),
                    updated_at=same_stamp,
                ),
                "ws-b": RagScope(
                    items=(ScopeItem(SOURCE_TYPE_MEDIA, media_ids[1]),),
                    updated_at=same_stamp,
                ),
            }
        )
        app = _App(media_db=media_db)
        app.workspace_registry_service = registry

        calls: list[int] = []
        real_resolve = cre.resolve_effective_scope

        def _spy_resolve(*args, **kwargs):
            calls.append(1)
            return real_resolve(*args, **kwargs)

        monkeypatch.setattr(cre, "resolve_effective_scope", _spy_resolve)

        first = await cre.resolve_effective_scope_for_chat(app)
        assert first.allowlist == {SOURCE_TYPE_MEDIA: frozenset({media_ids[0]})}

        # Re-link the session to a DIFFERENT workspace; its scope carries
        # the SAME stamp as the previous workspace's, isolating the
        # assertion to the `workspace_id` component of the cache key.
        session.workspace_id = "ws-b"

        second = await cre.resolve_effective_scope_for_chat(app)

        assert second.allowlist == {SOURCE_TYPE_MEDIA: frozenset({media_ids[1]})}
        assert len(calls) == 2, (
            "a workspace_id change (even with an identical stamp) must "
            "invalidate the cached entry"
        )


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
        # The caller-side short-circuit (chat_rag_events._record_scope_empty)
        # records a single entry, same appended-list shape as every pipeline
        # leg's own writer (task-9 review finding 2).
        assert captured["diagnostics"][SCOPE_DIAGNOSTICS_KEY] == [
            {
                "status": SCOPE_STATUS_EMPTY,
                "reason": SCOPE_REASON_EMPTY,
                "cause": "deleted-items",
            }
        ]
        assert any(
            "retrieval scope is empty" in message.lower()
            and "deleted-items" in message.lower()
            for message, _severity in app.notifications
        ), app.notifications


class TestExistingIdsSyncDanglingDrop:
    """``_existing_ids_sync``'s dangling-drop must act per source_type
    independently: a mixed scope with one surviving media id and one
    already-deleted (dangling) note id keeps the media id and drops the
    note id, landing on ``state == "scoped"`` -- not ``"empty"``.

    ``TestChatEntryPointEmptyScopeShortCircuit
    .test_empty_scope_short_circuits_with_zero_leg_calls`` only exercises
    the media-table branch of ``_existing_ids_sync`` (a single dangling
    media id, landing EMPTY); this covers the notes-table branch with a
    dangling note id, mirroring that test's fixture shape.
    """

    @pytest.mark.asyncio
    async def test_surviving_media_id_kept_dangling_note_id_dropped(
        self, media_db, cha_db, monkeypatch
    ):
        media_ids = _seed_media(media_db, n=1)
        conv_id = cha_db.add_conversation({"title": "Mixed dangling scope"})
        scope = RagScope(
            items=(
                ScopeItem(SOURCE_TYPE_MEDIA, media_ids[0]),
                # Never created in cha_db -- a dangling reference.
                ScopeItem(SOURCE_TYPE_NOTE, "does-not-exist"),
            ),
            updated_at="t1",
        )
        write_conversation_scope(cha_db, conv_id, scope)

        session = SimpleNamespace(persisted_conversation_id=conv_id)
        monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

        app = _App(media_db=media_db, chachanotes_db=cha_db)

        effective = await cre._resolve_effective_scope_for_chat(app)

        assert effective.state == "scoped"
        assert effective.allowlist == {SOURCE_TYPE_MEDIA: frozenset({media_ids[0]})}
        assert effective.cause is None


class TestResolveEffectiveScopeMemoryDbGuard:
    """PR #747 review (qodo): in-memory SQLite connections are thread-local
    (``CharactersRAGDB.get_connection`` opens a brand-new ``:memory:``
    connection -- with no migrated schema -- per thread). Offloading
    ``read_conversation_scope``/``resolve_effective_scope`` to
    ``asyncio.to_thread`` therefore makes a worker thread see a blank
    connection: the scope read raises "no such table", is swallowed by
    ``read_conversation_scope``'s own try/except, and reads back as
    unscoped even though the conversation genuinely has a scope set. This
    mirrors the guard ``Library.library_local_rag_search_service.
    _LocalRagSearchService._search_conversations`` already applies
    (``getattr(db, "is_memory_db", False)`` -> run inline on the calling
    thread instead of ``asyncio.to_thread``); ``resolve_effective_scope_
    for_chat`` must apply the same guard to its own DB reads.
    """

    @pytest.mark.asyncio
    async def test_memory_db_scoped_conversation_resolves_scoped_not_unscoped(
        self, monkeypatch
    ):
        cha_db = CharactersRAGDB(":memory:", client_id="task-memdb-scope-test")
        try:
            conv_id = cha_db.add_conversation({"title": "Memory scoped"})
            note_id = cha_db.add_note(title="N1", content="body")
            scope = RagScope(
                items=(ScopeItem(SOURCE_TYPE_NOTE, note_id),),
                updated_at="t1",
            )
            write_conversation_scope(cha_db, conv_id, scope)

            session = SimpleNamespace(persisted_conversation_id=conv_id)
            monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

            app = _App(chachanotes_db=cha_db)

            effective = await cre.resolve_effective_scope_for_chat(app)

            # Pre-fix, a worker thread's blank `:memory:` connection makes
            # this read back as `state == "unscoped"` instead -- the exact
            # regression this test guards against.
            assert effective.state == "scoped"
            assert effective.allowlist == {SOURCE_TYPE_NOTE: frozenset({note_id})}
            assert effective.cause is None
        finally:
            cha_db.close_connection()

    @pytest.mark.asyncio
    async def test_file_backed_db_scope_read_still_offloaded_to_thread(
        self, monkeypatch, tmp_path
    ):
        """Zero-drift companion: a real (file-backed) DB must keep taking
        the ``asyncio.to_thread`` path -- the memory-db guard must not widen
        to always-inline and quietly remove the off-loop discipline for the
        common (non-memory) deployment."""
        cha_db = CharactersRAGDB(tmp_path / "cha.db", client_id="task-memdb-scope-test")
        try:
            conv_id = cha_db.add_conversation({"title": "File-backed scoped"})
            note_id = cha_db.add_note(title="N1", content="body")
            scope = RagScope(
                items=(ScopeItem(SOURCE_TYPE_NOTE, note_id),),
                updated_at="t1",
            )
            write_conversation_scope(cha_db, conv_id, scope)

            session = SimpleNamespace(persisted_conversation_id=conv_id)
            monkeypatch.setattr(cre, "_active_console_session", lambda app: session)

            calls: list[bool] = []
            real_to_thread = asyncio.to_thread

            async def _spy_to_thread(func, *args, **kwargs):
                calls.append(True)
                return await real_to_thread(func, *args, **kwargs)

            monkeypatch.setattr(cre.asyncio, "to_thread", _spy_to_thread)

            app = _App(chachanotes_db=cha_db)

            effective = await cre.resolve_effective_scope_for_chat(app)

            assert effective.state == "scoped"
            assert effective.allowlist == {SOURCE_TYPE_NOTE: frozenset({note_id})}
            assert calls, "file-backed DB reads must still be offloaded via asyncio.to_thread"
        finally:
            cha_db.close_connection()


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


class TestActiveConsoleSessionRealGlue:
    """Integration coverage for ``chat_rag_events._active_console_session``'s
    REAL chain: ``app.screen`` -> ``isinstance(screen, ChatScreen)`` ->
    ``screen._console_chat_store`` -> ``store.active_session_id`` ->
    matching session.

    Every ``TestChatEntryPoint*`` test above (and every other consumer of
    ``get_rag_context_for_chat``) monkeypatches ``_active_console_session``
    itself away, so none of them exercise this glue -- a real hazard in an
    area with a documented bug class (native Console never writes the
    legacy ``app.current_chat_conversation_id`` reactives; see
    ``UI/Screens/chat_screen.py``'s dictionary-summary comment around line
    1595). These tests build a real ``TldwCli`` (``_build_test_app``, the
    same heavy-init-patched recipe ``Tests/UI/test_screen_navigation.py``
    and every Console UI suite share) and a real ``ChatScreen`` instance
    constructed directly against it (unmounted -- the same
    ``ChatScreen(app)`` pattern ``Tests/UI/
    test_console_internals_decomposition.py`` uses to reach real
    ``ConsoleChatStore``-backed internals without paying for a full
    Textual mount), then place that real screen on the app's real screen
    stack (``app._screen_stacks[app._current_mode]``) so ``app.screen``
    itself -- not a stub -- resolves to it. Nothing here monkeypatches
    ``_active_console_session`` (the function under test); only
    unrelated heavy ``TldwCli.__init__`` collaborators are patched, via
    ``_build_test_app``.
    """

    def test_active_session_with_persisted_conversation_id_is_returned_by_identity(
        self,
    ):
        app = _build_test_app()
        screen = ChatScreen(app)
        store = screen._ensure_console_chat_store()
        session = store.create_session(title="Console session")
        session.persisted_conversation_id = "conv-real-glue-1"
        app._screen_stacks[app._current_mode] = [screen]

        result = cre._active_console_session(app)

        assert result is session
        assert result.persisted_conversation_id == "conv-real-glue-1"

    def test_non_chat_screen_resolves_to_none(self):
        app = _build_test_app()

        class _NotAChatScreen:
            """Stands in for any other real screen on the stack."""

        app._screen_stacks[app._current_mode] = [_NotAChatScreen()]

        assert cre._active_console_session(app) is None

    def test_store_with_no_active_session_resolves_to_none(self):
        app = _build_test_app()
        screen = ChatScreen(app)
        # Store created, but no session was ever created/activated on it.
        store = screen._ensure_console_chat_store()
        assert store.active_session_id is None
        app._screen_stacks[app._current_mode] = [screen]

        assert cre._active_console_session(app) is None

    def test_empty_screen_stack_never_raises(self):
        """FIX 2 guard: ``app.screen`` raises ``ScreenStackError`` with no
        screens on the stack; ``_active_console_session`` must degrade to
        ``None`` rather than propagate it (its own docstring's claim)."""
        app = _build_test_app()
        # A freshly constructed TldwCli has an empty default-mode screen
        # stack (never pushed/mounted here) -- real ``app.screen`` access
        # raises ScreenStackError, not a stubbed-out condition.
        assert app._screen_stacks[app._current_mode] == []

        assert cre._active_console_session(app) is None
