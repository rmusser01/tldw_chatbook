"""rag mode must honor the active profile's default_search_mode.

The live path hardcoded search_type="semantic"; the engine's hybrid
(RRF k=60 + alpha, ADR-005 server parity) was unreachable, so a user who
picked "Hybrid Basic" or "BM25 Only" in Settings > RAG got vector-only
retrieval anyway, silently. Routing rules and disclosures per the P0 spec,
Workstream A.

Fixtures mirror `test_library_local_rag_search_service.py` (its
`FakeNotesScopeService`/`FakeRagService` are imported directly rather than
re-declared); the mode-aware double below adds the profile surface the
Library service now reads (`config.search.default_search_mode`,
`profile.name`) and reproduces `RAGService.search`'s real
`metadata_allowlist` contract -- accepted for semantic AND hybrid
(TASK-15020/B1), still refused for keyword, materialized once, empty
sequence entries rejected -- so the scoped-hybrid tests prove the engine's
actual guards are respected rather than assuming them.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.rag_scope import (
    EffectiveScope,
    SOURCE_TYPE_MEDIA,
    SOURCE_TYPE_NOTE,
)
from tldw_chatbook.Library import library_local_rag_search_service as rag_service_module
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)
from tldw_chatbook.Library.library_rag_service import LibraryRagSearchOutcome

from Tests.Library.test_library_local_rag_search_service import (
    FakeNotesScopeService,
    FakeRagService,
)

# The diagnostics slot the routing disclosures travel in, and the exact
# disclosure copy. Spelled literally here (not imported) so this file pins
# the contract rather than restating whatever the implementation happens to
# define.
_ROUTE_NOTES_KEY = "retrieval_route_notes"
_NOTE_HYBRID_NO_KEYWORD_SOURCES = (
    "no keyword leg for the selected sources — semantic only"
)
_NOTE_SEMANTIC_LEG_EMPTY = "semantic leg empty — keyword-only results"


class _FakeVectorStore:
    """Mirrors the VectorStore stats seam used for empty-index detection."""

    def __init__(self, count: int):
        self.count = count
        self.calls = 0

    def get_collection_stats(self):
        self.calls += 1
        return {"count": self.count}


class _ProfileRagService:
    """`RAGService.search`'s signature plus the profile surface (rag_service.py:542).

    `default_search_mode` is the profile knob the Library service resolves;
    `profile.name` is what the plain-profile disclosure names.
    """

    def __init__(
        self,
        *,
        mode: str = "semantic",
        profile_name: str | None = None,
        results=None,
        vector_count: int = 12,
    ):
        self.config = SimpleNamespace(
            search=SimpleNamespace(default_search_mode=mode)
        )
        self.profile = (
            SimpleNamespace(name=profile_name) if profile_name is not None else None
        )
        self.results = results if results is not None else []
        self.calls: list[dict] = []
        self.vector_store = _FakeVectorStore(vector_count)

    async def search(
        self,
        query,
        top_k=None,
        search_type="semantic",
        filter_metadata=None,
        include_citations=None,
        score_threshold=None,
        metadata_allowlist=None,
        keyword_source_types=None,
    ):
        # The real engine's allowlist contract (TASK-15020/B1, mirrored from
        # `RAGService.search` + `_allowlist_entries`): accepted for semantic
        # AND hybrid, still refused for keyword, materialized ONCE up front
        # (a one-shot iterable would otherwise reach the legs drained --
        # i.e. unscoped -- which fails OPEN), and a sequence carrying an
        # empty entry is a caller defect rather than a silent widening.
        # Reproduced here so a routing change that hands the engine a shape
        # it rejects fails loudly in this suite instead of passing against a
        # more permissive double.
        if metadata_allowlist is not None and not isinstance(
            metadata_allowlist, Mapping
        ):
            metadata_allowlist = tuple(metadata_allowlist)
            if any(not entry for entry in metadata_allowlist):
                raise ValueError(
                    "metadata_allowlist entries must each be a non-empty mapping"
                )
        if metadata_allowlist and search_type == "keyword":
            raise ValueError(
                "metadata_allowlist is not supported for search_type='keyword'"
            )
        # The mirror-image guard (TASK-14751): `keyword_source_types` scopes
        # the KEYWORD leg, so the engine refuses it for a semantic search
        # rather than silently ignoring it. Reproduced for the same reason.
        if keyword_source_types is not None and search_type == "semantic":
            raise ValueError(
                "keyword_source_types is not supported for "
                "search_type='semantic'"
            )
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "search_type": search_type,
                "include_citations": include_citations,
                "metadata_allowlist": metadata_allowlist,
                "keyword_source_types": keyword_source_types,
            }
        )
        return self.results


def _media_result(source_id: str = "media-1", score: float = 0.8, **metadata):
    return {
        "id": f"{source_id}-chunk",
        "score": score,
        "document": "Media evidence.",
        "metadata": {
            "title": "Media doc",
            "source_id": source_id,
            "source_type": "media",
            **metadata,
        },
    }


def _note_result(source_id: str = "note-1", score: float = 0.9, **metadata):
    return {
        "id": f"{source_id}-chunk",
        "score": score,
        "document": "Note evidence.",
        "metadata": {
            "title": "Note doc",
            "source_id": source_id,
            "source_type": "note",
            **metadata,
        },
    }


def _conversation_result(source_id: str = "conv-1", score: float = 0.85):
    return {
        "id": f"{source_id}-chunk",
        "score": score,
        "document": "Conversation evidence.",
        "metadata": {
            "title": "Conversation doc",
            "source_id": source_id,
            "source_type": "conversation",
        },
    }


def _prompt_result(source_id: str = "prompt-1", score: float = 0.8):
    """A hybrid row as the engine's PROMPTS sub-leg stamps it (TASK-15020/B2).

    `source_type` is the singular `prompt` -- the same string the keyword
    leg writes and the Library's own `_prompt_row` uses -- so this row is
    only kept by the source-type post-filter when `prompts` is selected.
    """
    return {
        "id": f"prompt_{source_id}",
        "score": score,
        "document": "Prompt evidence.",
        "metadata": {
            "title": "Prompt doc",
            "source_id": source_id,
            "source_type": "prompt",
        },
    }


def _scoped(**allowlist: set) -> EffectiveScope:
    return EffectiveScope(
        state="scoped",
        allowlist={key: frozenset(value) for key, value in allowlist.items()},
        cause=None,
    )


def _route_notes(result) -> tuple[str, ...]:
    diagnostics = (
        result.diagnostics
        if isinstance(result, LibraryRagSearchOutcome)
        else (result.get("diagnostics") or {})
    )
    return tuple(diagnostics.get(_ROUTE_NOTES_KEY, ()))


# --- The pure mapping -------------------------------------------------------


@pytest.mark.parametrize(
    "configured, expected",
    [
        ("plain", "plain"),
        ("semantic", "semantic"),
        ("hybrid", "hybrid"),
        ("nonsense", "semantic"),
        ("", "semantic"),
        (None, "semantic"),
    ],
)
def test_resolve_profile_search_mode_maps_known_modes_and_falls_back(
    configured, expected
):
    from tldw_chatbook.Library.library_local_rag_search_service import (
        _resolve_profile_search_mode,
    )

    service = _ProfileRagService(mode=configured)

    assert _resolve_profile_search_mode(service) == expected


def test_resolve_profile_search_mode_defaults_when_the_runtime_has_no_config():
    """Every pre-existing test fake (and any non-profile runtime) has no
    `config` at all -- it must keep today's semantic behavior, not crash."""
    from tldw_chatbook.Library.library_local_rag_search_service import (
        _resolve_profile_search_mode,
    )

    assert _resolve_profile_search_mode(SimpleNamespace()) == "semantic"
    assert _resolve_profile_search_mode(FakeRagService()) == "semantic"


def test_resolve_profile_search_mode_delegates_normalization(monkeypatch):
    received = []
    monkeypatch.setattr(
        rag_service_module,
        "normalize_rag_search_mode",
        lambda value: received.append(value) or "controlled",
        raising=False,
    )

    assert rag_service_module._resolve_profile_search_mode(
        _ProfileRagService(mode="runtime-mode")
    ) == "controlled"
    assert received == ["runtime-mode"]


# --- Dispatch arms ----------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_profile_unscoped_calls_engine_hybrid():
    """The whole point of the port: a hybrid profile actually reaches the
    engine's hybrid path (FTS leg + vector leg, RRF-fused)."""
    rag = _ProfileRagService(mode="hybrid", results=[_media_result()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))

    result = await service.search(
        "credential", ("notes", "media"), "rag", top_k=5, include_citations=True
    )

    assert [call["search_type"] for call in rag.calls] == ["hybrid"]
    assert rag.calls[0]["top_k"] == 5
    assert rag.calls[0]["include_citations"] is True
    assert result["runtime_backend"] == "rag-hybrid"
    assert [row["source_id"] for row in result["results"]] == ["media-1"]


@pytest.mark.asyncio
async def test_hybrid_profile_with_active_scope_runs_fused_hybrid():
    """(TASK-15020/B1) THE ROUTING FLIP.

    A scope used to force a hybrid profile onto the semantic path, because
    `RAGService.search` raised for a non-empty `metadata_allowlist` with any
    non-semantic search type. That guard is gone for hybrid: the allowlist
    now reaches BOTH legs, so a scoped query under a hybrid profile runs the
    profile's own fused search -- and has nothing left to disclose, because
    nothing was diverted.
    """
    rag = _ProfileRagService(mode="hybrid", results=[_media_result()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))
    scope = _scoped(**{SOURCE_TYPE_MEDIA: {"media-1"}})

    result = await service.search(
        "credential", ("media",), "rag", top_k=5, scope=scope
    )

    assert rag.calls, "the scoped query never reached the runtime"
    assert [call["search_type"] for call in rag.calls] == ["hybrid"]
    assert result["runtime_backend"] == "rag-hybrid"
    assert _route_notes(result) == (), (
        "a scoped hybrid search now runs exactly as the profile configures "
        "it, so there is no divert left to disclose"
    )
    # The scope reached the engine in the shape `build_semantic_allowlists`
    # produces (one AND-group per source type) -- not dropped, and not
    # flattened into a single dict that cannot express a union. Recorded as
    # the tuple the engine materializes it into, since that materialization
    # is itself part of the contract (a one-shot iterable would reach the
    # legs drained, i.e. unscoped).
    assert rag.calls[0]["metadata_allowlist"] == (
        {"source_type": {SOURCE_TYPE_MEDIA}, "source_id": {"media-1"}},
    )
    # ...alongside the keyword-leg selection, which scopes a different
    # dimension (which sub-legs run, not which ids they may return).
    assert rag.calls[0]["keyword_source_types"] == {"media"}


@pytest.mark.asyncio
async def test_scoped_hybrid_sends_one_allowlist_entry_per_scoped_source_type():
    """A two-type scope must arrive as a UNION of AND-groups.

    `{"source_type": {media, note}, "source_id": {m1, n1}}` would allow a
    note row carrying a media id; the engine's contract (and every FTS
    sub-leg's id filter) depends on the per-type split surviving the
    Library's call.
    """
    rag = _ProfileRagService(mode="hybrid", results=[_media_result()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))
    scope = _scoped(
        **{SOURCE_TYPE_MEDIA: {"media-1"}, SOURCE_TYPE_NOTE: {"note-1"}}
    )

    await service.search(
        "credential", ("media", "notes"), "rag", top_k=5, scope=scope
    )

    assert rag.calls[0]["metadata_allowlist"] == (
        {"source_type": {SOURCE_TYPE_MEDIA}, "source_id": {"media-1"}},
        {"source_type": {SOURCE_TYPE_NOTE}, "source_id": {"note-1"}},
    )
    assert rag.calls[0]["keyword_source_types"] == {"media", "note"}


@pytest.mark.asyncio
async def test_scoped_selection_with_no_servable_source_still_diverts():
    """The surviving divert: hybrid needs a source the FTS leg serves.

    DISCLOSED UPDATE (2026-08-11, TASK-15020/B2). This test used to make its
    point with a prompts-only selection, because prompts had no sub-leg;
    B2 gave them one, so all FOUR of the Search canvas's source types are
    now FTS-servable and the divert is reachable only through a selection
    carrying nothing the leg knows. The property under test is unchanged and
    still worth pinning: whatever diverts to semantic must still carry the
    scope's allowlists -- a divert that dropped the scope would silently
    widen retrieval.
    """
    rag = _ProfileRagService(mode="hybrid", results=[_note_result()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))
    scope = _scoped(**{SOURCE_TYPE_MEDIA: {"media-1"}})

    result = await service.search(
        "credential", ("workspaces",), "rag", top_k=5, scope=scope
    )

    assert [call["search_type"] for call in rag.calls] == ["semantic"]
    assert _NOTE_HYBRID_NO_KEYWORD_SOURCES in _route_notes(result)
    # `_search_semantic` predates the engine's multi-entry support and still
    # issues one store query per AND-group, so each call carries ONE mapping.
    assert rag.calls[0]["metadata_allowlist"] == {
        "source_type": {SOURCE_TYPE_MEDIA},
        "source_id": {"media-1"},
    }
    # An unknown selection drops every semantic row in the post-filter, so
    # this lands on the scoped path's own zero-results outcome -- labeled
    # semantic, which is what actually ran.
    assert isinstance(result, LibraryRagSearchOutcome)
    assert result.runtime_backend == "rag-semantic"


@pytest.mark.asyncio
async def test_scoped_prompts_only_selection_now_routes_hybrid_and_fails_closed():
    """TASK-15020/B2: prompts are FTS-servable, so they keep the hybrid path.

    The pair of facts this pins is the whole of B2's interaction with B1's
    scope. A prompts-only selection routes HYBRID now (it did not before),
    and the engine is handed both the translated selection (`{"prompt"}`)
    and the scope's allowlists -- which can never NAME prompts, because the
    scope vocabulary is media/note only (spec D5). The engine's own
    fail-closed rule then skips the prompts sub-leg entirely rather than
    running it unfiltered (pinned at the engine in
    `test_keyword_leg_prompts.test_a_scope_skips_the_prompts_sub_leg_
    entirely`). Scoped prompt search is therefore structurally out of the
    scope vocabulary today, and this is where that is written down.
    """
    rag = _ProfileRagService(mode="hybrid", results=[_note_result()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))
    scope = _scoped(**{SOURCE_TYPE_MEDIA: {"media-1"}})

    await service.search("credential", ("prompts",), "rag", top_k=5, scope=scope)

    assert [call["search_type"] for call in rag.calls] == ["hybrid"]
    assert rag.calls[0]["keyword_source_types"] == {"prompt"}
    allowlist = rag.calls[0]["metadata_allowlist"]
    assert all(
        "prompt" not in entry.get("source_type", ()) for entry in allowlist
    ), f"the scope named prompts, which spec D5 says it cannot: {allowlist}"


@pytest.mark.asyncio
async def test_scoped_hybrid_with_no_rows_keeps_the_scope_recovery_state():
    """Zero rows under a scope is a scope-shaped dead end, not a generic one.

    The scoped semantic path answers it with "nothing matched among the N
    items you scoped to -- broaden or clear the scope"; routing the same
    query to hybrid must not silently downgrade that to the generic no-match
    state, or the one actionable thing on screen disappears with the route
    change.
    """
    rag = _ProfileRagService(mode="hybrid", results=[])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))
    scope = _scoped(**{SOURCE_TYPE_MEDIA: {"media-1", "media-2"}})

    result = await service.search(
        "credential", ("media",), "rag", top_k=5, scope=scope
    )

    assert isinstance(result, LibraryRagSearchOutcome)
    assert result.status == "empty"
    assert result.runtime_backend == "rag-hybrid"
    assert result.recovery_state.why == "No results within scope (2 items searched)"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_types, result_row, expected_source_id",
    [
        (("notes",), _note_result(), "note-1"),
        (("conversations",), _conversation_result(), "conv-1"),
        (("notes", "conversations"), _note_result(), "note-1"),
    ],
)
async def test_hybrid_runs_for_any_fts_servable_source_without_media(
    source_types, result_row, expected_source_id
):
    """Media deselected must no longer disable the FTS leg (TASK-3996).

    The old gate skipped hybrid whenever `media` was absent, justified by
    "the FTS leg is media-only, so its rows would all be dropped by the
    source-type post-filter". That premise died with this task's
    notes/conversation sub-legs: the leg now serves media, notes AND
    conversations, so a user who turns Media off and keeps Notes on was
    getting semantic-only search precisely in the case the fix was for.
    """
    rag = _ProfileRagService(mode="hybrid", results=[result_row])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))

    result = await service.search("credential", source_types, "rag", top_k=5)

    assert [call["search_type"] for call in rag.calls] == ["hybrid"]
    assert result["runtime_backend"] == "rag-hybrid"
    assert [row["source_id"] for row in result["results"]] == [expected_source_id]
    assert _NOTE_HYBRID_NO_KEYWORD_SOURCES not in _route_notes(result)


@pytest.mark.asyncio
async def test_hybrid_profile_with_only_unservable_sources_stays_semantic():
    """A selection with nothing the FTS leg serves still stays semantic.

    DISCLOSED UPDATE (2026-08-11, TASK-15020/B2): this used to be spelled
    with `prompts`, "the one remaining case where the keyword leg could
    contribute nothing". B2 removed that case -- prompts have a sub-leg now
    -- so the gate is exercised with a selection the build does not know at
    all. The gate itself is unchanged and still load-bearing: without it,
    hybrid would run and every one of its rows would be dropped by the
    source-type post-filter, turning a search into a silent empty result.
    """
    rag = _ProfileRagService(mode="hybrid", results=[_note_result()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))

    result = await service.search("credential", ("workspaces",), "rag", top_k=5)

    assert [call["search_type"] for call in rag.calls] == ["semantic"]
    assert result["runtime_backend"] == "rag-semantic"
    assert _NOTE_HYBRID_NO_KEYWORD_SOURCES in _route_notes(result)


@pytest.mark.asyncio
async def test_hybrid_runs_for_a_prompts_only_selection():
    """TASK-15020/B2: prompts-only is a hybrid search now, not a divert.

    The rows a prompts-only hybrid returns can only have come from the FTS
    leg -- prompts have no vector index -- which is exactly why the divert
    had to go: sending this selection to the semantic leg searched the one
    index that structurally cannot answer it.
    """
    rag = _ProfileRagService(mode="hybrid", results=[_prompt_result()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))

    result = await service.search("credential", ("prompts",), "rag", top_k=5)

    assert [call["search_type"] for call in rag.calls] == ["hybrid"]
    assert rag.calls[0]["keyword_source_types"] == {"prompt"}
    assert result["runtime_backend"] == "rag-hybrid"
    assert _NOTE_HYBRID_NO_KEYWORD_SOURCES not in _route_notes(result)
    # And the row survives the source-type post-filter, which it only does
    # because `prompt` canonicalizes to the `prompts` toggle.
    assert [row["source_id"] for row in result["results"]] == ["prompt-1"]


@pytest.mark.asyncio
async def test_plain_profile_routes_to_four_seam_keyword_path():
    """A BM25 Only profile in rag mode must NOT get the engine's keyword
    leg (a strictly worse version of the Library's own Search mode: not
    scope-aware, and no prompts, even now that it spans media, notes and
    conversations); it routes through the four-seam, scope-aware keyword
    path, labeled."""
    rag = _ProfileRagService(mode="plain", profile_name="BM25 Only")
    notes = FakeNotesScopeService(
        rows=[{"id": "note-1", "title": "Runbook", "content": "Rotate the credential."}]
    )
    service = LibraryLocalRagSearchService(
        SimpleNamespace(_rag_service=rag, notes_scope_service=notes)
    )

    result = await service.search("credential", ("notes",), "rag", top_k=5)

    assert rag.calls == [], "plain profile must not query the vector engine"
    assert notes.calls, "plain profile must run the four-seam keyword path"
    assert result["runtime_backend"] == "local-fts"
    assert [row["source_id"] for row in result["results"]] == ["note-1"]
    assert "Profile 'BM25 Only': keyword search (no vectors)" in _route_notes(result)


@pytest.mark.asyncio
async def test_semantic_profile_unchanged():
    """The default profile keeps today's exact payload -- same search_type,
    same backend label, no routing note to explain (nothing was diverted)."""
    rag = _ProfileRagService(mode="semantic", results=[_note_result()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))

    result = await service.search("credential", ("notes",), "rag", top_k=5)

    assert [call["search_type"] for call in rag.calls] == ["semantic"]
    assert result["runtime_backend"] == "rag-semantic"
    assert _route_notes(result) == ()
    assert result["diagnostics"] == {
        "semantic_scope_coverage": {"covered": ["notes"], "uncovered": []}
    }


@pytest.mark.asyncio
async def test_index_empty_not_claimed_when_keyword_rows_present():
    """Zero-results honesty under hybrid: an empty vector store plus a
    hitting FTS leg is "semantic leg empty", never "nothing is indexed" --
    the user has evidence on screen while being told the index is empty."""
    rag = _ProfileRagService(
        mode="hybrid",
        vector_count=0,
        results=[
            _media_result(
                "media-1",
                hybrid_fusion={
                    "fts_rank": 1,
                    "vector_rank": None,
                    "fts_score": 0.001,
                    "vector_score": None,
                },
            ),
            _media_result(
                "media-2",
                hybrid_fusion={
                    "fts_rank": 2,
                    "vector_rank": None,
                    "fts_score": 0.0005,
                    "vector_score": None,
                },
            ),
        ],
    )
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))

    result = await service.search("credential", ("media",), "rag", top_k=5)

    assert not isinstance(result, LibraryRagSearchOutcome), (
        "hybrid returned keyword-leg rows -- 'Index empty' must not be claimed"
    )
    assert [row["source_id"] for row in result["results"]] == ["media-1", "media-2"]
    assert result["runtime_backend"] == "rag-hybrid"
    assert _NOTE_SEMANTIC_LEG_EMPTY in _route_notes(result)


@pytest.mark.asyncio
async def test_semantic_leg_empty_note_never_fires_when_the_vector_leg_scored():
    """The converse pin: a row carrying a real vector score proves the
    semantic leg contributed, so the keyword-only disclosure must stay
    silent (it would otherwise be a false claim on every hybrid search)."""
    rag = _ProfileRagService(
        mode="hybrid",
        vector_count=0,  # deliberately lying: a scored vector leg outranks it
        results=[
            _media_result(
                "media-1",
                hybrid_fusion={
                    "fts_rank": 1,
                    "vector_rank": 1,
                    "fts_score": 0.001,
                    "vector_score": 0.83,
                },
            )
        ],
    )
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))

    result = await service.search("credential", ("media",), "rag", top_k=5)

    assert _NOTE_SEMANTIC_LEG_EMPTY not in _route_notes(result)


# --- Coverage diagnostics under hybrid (TASK-14752) --------------------------


@pytest.mark.asyncio
async def test_hybrid_coverage_separates_keyword_only_types_from_absent_ones():
    """(TASK-14752) A type whose rows came only from the FTS leg is neither
    "covered" nor "found nothing".

    Before TASK-3996 the engine's keyword leg served media only, so under a
    hybrid profile a type with no semantic hits also had no evidence at all
    and one flat `uncovered` list said everything there was to say. With
    notes and conversation sub-legs, a type can now be ON SCREEN entirely
    from the keyword leg -- and calling that "covered" hides that the
    semantic leg never matched it, while calling it "uncovered" would
    contradict the rows the user is looking at. Three states, three lists.
    """
    rag = _ProfileRagService(
        mode="hybrid",
        results=[
            _media_result(
                "media-1",
                hybrid_fusion={
                    "fts_rank": 2,
                    "vector_rank": 1,
                    "fts_score": 0.001,
                    "vector_score": 0.81,
                },
            ),
            _note_result(
                "note-1",
                hybrid_fusion={
                    "fts_rank": 1,
                    "vector_rank": None,
                    "fts_score": 0.002,
                    "vector_score": None,
                },
            ),
        ],
    )
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))

    result = await service.search(
        "credential", ("notes", "media", "conversations"), "rag", top_k=5
    )

    assert result["diagnostics"]["semantic_scope_coverage"] == {
        "covered": ["media"],
        "uncovered": ["conversations"],
        "keyword_only": ["notes"],
    }


@pytest.mark.asyncio
async def test_hybrid_rows_without_a_fusion_block_are_never_called_keyword_only():
    """The un-judgeable case fails toward silence, not toward a claim.

    A row with no fusion provenance cannot prove which leg produced it (the
    same reason `_rows_are_keyword_only` refuses to judge such a set), so it
    keeps counting as covered -- inventing a "keyword matches only" claim
    from missing evidence would be the defect this fix exists to remove,
    pointed the other way.
    """
    rag = _ProfileRagService(mode="hybrid", results=[_media_result("media-1")])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))

    result = await service.search("credential", ("media",), "rag", top_k=5)

    assert result["diagnostics"]["semantic_scope_coverage"] == {
        "covered": ["media"],
        "uncovered": [],
    }


# --- Profile switching mid-session ------------------------------------------


@pytest.mark.asyncio
async def test_profile_switch_mid_session_reroutes_and_renames_the_disclosure(
    monkeypatch,
):
    """Review finding I1: routing is only honest if it follows the profile
    that is active NOW.

    `app._rag_service` is a per-app cache that a profile switch never
    cleared -- `set_active_profile()` / the Settings save path call
    `reset_shared_rag_service()`, which drops only the module singleton.
    Without invalidation the second query below still routes through profile
    A and the disclosure names a profile the user already switched away
    from: false attribution, the exact defect class this task exists to kill.
    """
    from tldw_chatbook.RAG_Search import ingestion_indexing
    from tldw_chatbook.RAG_Search import simplified as simplified_module

    monkeypatch.setattr(
        rag_service_module, "embeddings_rag_deps_installed", lambda: True
    )
    profile_a = _ProfileRagService(mode="plain", profile_name="BM25 Only")
    profile_b = _ProfileRagService(
        mode="hybrid", profile_name="Hybrid Basic", results=[_media_result()]
    )
    notes = FakeNotesScopeService(
        rows=[{"id": "note-1", "title": "Runbook", "content": "Rotate the credential."}]
    )
    ingestion_indexing.reset_shared_rag_service()
    try:
        # Profile A is the process-wide runtime; the app has no cache yet.
        ingestion_indexing.set_shared_rag_service(profile_a)
        app = SimpleNamespace(_rag_service=None, notes_scope_service=notes)
        service = LibraryLocalRagSearchService(app)

        first = await service.search("credential", ("notes", "media"), "rag", top_k=5)

        assert first["runtime_backend"] == "local-fts"
        assert "Profile 'BM25 Only': keyword search (no vectors)" in _route_notes(first)

        # The user switches profile in Settings: the pointer write resets the
        # shared service, and the next resolution rebuilds it (profile B).
        monkeypatch.setattr(
            simplified_module, "create_rag_service", lambda **kwargs: profile_b
        )
        ingestion_indexing.reset_shared_rag_service()

        second = await service.search("credential", ("notes", "media"), "rag", top_k=5)

        assert [call["search_type"] for call in profile_b.calls] == ["hybrid"]
        assert second["runtime_backend"] == "rag-hybrid"
        assert app._rag_service is profile_b
        assert not any("BM25 Only" in note for note in _route_notes(second))
    finally:
        ingestion_indexing.reset_shared_rag_service()


@pytest.mark.asyncio
async def test_injected_runtime_without_a_generation_stamp_still_wins(monkeypatch):
    """Guard on the staleness fix: a runtime injected straight onto the app
    (every pre-existing test fake, and any surface that predates the stamp)
    was never resolved through the shared seam, so it has no generation to
    compare -- it must keep winning outright, never be treated as stale and
    rebuilt."""
    from tldw_chatbook.RAG_Search import ingestion_indexing

    def _forbidden(*args, **kwargs):
        raise AssertionError("an injected runtime must not be re-resolved")

    monkeypatch.setattr(rag_service_module, "embeddings_rag_deps_installed", _forbidden)
    monkeypatch.setattr(rag_service_module, "get_shared_rag_service", _forbidden)
    injected = _ProfileRagService(mode="hybrid", results=[_media_result()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=injected))
    # Bump the generation so a naive "generation moved -> stale" rule would
    # wrongly discard the injected runtime.
    ingestion_indexing.set_shared_rag_service(None)

    result = await service.search("credential", ("media",), "rag", top_k=5)

    assert result["runtime_backend"] == "rag-hybrid"
    assert injected.calls
