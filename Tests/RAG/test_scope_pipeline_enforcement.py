"""Tests for task-4: pipeline legs self-enforce RAG retrieval scope.

Covers the ``rag-scope-narrowing`` program's pipeline-leg enforcement layer:
media/notes FTS legs restrict to the scope's id allowlist (or return ``[]``
without querying the DB when their source type is absent from an active
scope); the conversations leg is excluded outright under any active scope
(spec decision D5) and records a diagnostic; the semantic leg runs one store
query per source_type present in the scope and merges by score. All four
legs must also inherit enforcement identically when driven through
``execute_pipeline`` (builtin/parallel pipeline shape), proving
self-enforcement rather than caller-side leg-skipping (the task-250 lesson).

Real in-memory-adjacent (tmp_path file-backed) DBs are used throughout,
mirroring ``Tests/RAG_Search/test_pipeline_notes_search.py`` and
``Tests/RAG/test_semantic_honest_states.py``'s fixture patterns.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from tldw_chatbook.Chat.rag_scope import (
    EffectiveScope,
    SCOPE_REASON_CONVERSATIONS_EXCLUDED,
    SOURCE_TYPE_MEDIA,
    SOURCE_TYPE_NOTE,
    build_semantic_allowlists,
    media_id_params,
    note_id_params,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
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
            "status": "excluded",
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
