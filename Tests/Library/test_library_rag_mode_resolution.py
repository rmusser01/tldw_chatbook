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
`metadata_allowlist`-only-with-semantic ValueError, so the scoped-hybrid
test proves the guard is respected rather than assuming it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.rag_scope import (
    EffectiveScope,
    SOURCE_TYPE_MEDIA,
)
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
_NOTE_HYBRID_SCOPED = "scope active — semantic only until scope-aware hybrid lands"
_NOTE_HYBRID_MEDIA_EXCLUDED = "media excluded — semantic only"
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
    ):
        # The real engine raises here (rag_service.py:580) -- allowlists are
        # a semantic-only pushdown. Reproduced so a routing bug that sends a
        # scoped query to hybrid fails loudly in this suite too.
        if metadata_allowlist and search_type != "semantic":
            raise ValueError(
                "metadata_allowlist is only supported for search_type='semantic'"
            )
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "search_type": search_type,
                "include_citations": include_citations,
                "metadata_allowlist": metadata_allowlist,
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


def _note_result(source_id: str = "note-1", score: float = 0.9):
    return {
        "id": f"{source_id}-chunk",
        "score": score,
        "document": "Note evidence.",
        "metadata": {
            "title": "Note doc",
            "source_id": source_id,
            "source_type": "note",
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
async def test_hybrid_profile_with_active_scope_stays_semantic_and_discloses():
    """Scope allowlists are a semantic-only pushdown (the engine raises for
    hybrid + allowlist), so a scoped query under a hybrid profile runs
    semantic -- and says so rather than pretending it ran hybrid."""
    rag = _ProfileRagService(mode="hybrid", results=[_media_result()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))
    scope = _scoped(**{SOURCE_TYPE_MEDIA: {"media-1"}})

    result = await service.search(
        "credential", ("media",), "rag", top_k=5, scope=scope
    )

    assert rag.calls, "the scoped query never reached the runtime"
    assert {call["search_type"] for call in rag.calls} == {"semantic"}
    assert result["runtime_backend"] == "rag-semantic"
    assert _NOTE_HYBRID_SCOPED in _route_notes(result)


@pytest.mark.asyncio
async def test_hybrid_profile_with_media_deselected_stays_semantic_and_discloses():
    """The engine's FTS leg is media-only in P0: with media deselected it
    could only contribute rows the Library post-filter drops, so hybrid is
    skipped -- disclosed, not silent."""
    rag = _ProfileRagService(mode="hybrid", results=[_note_result()])
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=rag))

    result = await service.search("credential", ("notes",), "rag", top_k=5)

    assert [call["search_type"] for call in rag.calls] == ["semantic"]
    assert result["runtime_backend"] == "rag-semantic"
    assert _NOTE_HYBRID_MEDIA_EXCLUDED in _route_notes(result)


@pytest.mark.asyncio
async def test_plain_profile_routes_to_four_seam_keyword_path():
    """A BM25 Only profile in rag mode must NOT get the engine's media-only
    keyword leg (a strictly worse version of the Library's own Search mode);
    it routes through the four-seam, scope-aware keyword path, labeled."""
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
