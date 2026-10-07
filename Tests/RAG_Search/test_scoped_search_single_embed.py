"""B2: a scoped semantic search performs exactly ONE engine search call --
one query-embedding round -- no matter how many allowlist entries the scope
carries.

Review-B finding: ``Library/library_local_rag_search_service._search_semantic``
and ``RAG_Search.pipeline_functions_simple.search_semantic`` both looped over
``build_semantic_allowlists``' per-source-type entries and called
``rag_service.search(..., metadata_allowlist=allowlist)`` once PER entry, so a
scope over k source types re-embedded the identical query and re-ran the
search machinery k times. The engine's ``_semantic_search_scoped`` already
accepts the multi-entry union (that is what ``_search_hybrid`` passes it), so
both callers now pass the whole union list in ONE call and let the engine run
the per-entry store queries and the score merge.

The fake runtime below mirrors the engine's union contract (one store query
per entry inside the single call, merged by score, trimmed to top_k) so each
test can also verify the results are equivalent to the old loop's merged
output.
"""

from collections.abc import Mapping as AbcMapping
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from tldw_chatbook.Chat.rag_scope import EffectiveScope, SOURCE_TYPE_MEDIA, SOURCE_TYPE_NOTE
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)
from tldw_chatbook.RAG_Search import pipeline_functions_simple as pfs


def _scoped(**allowlist: set) -> EffectiveScope:
    return EffectiveScope(
        state="scoped",
        allowlist={k: frozenset(v) for k, v in allowlist.items()},
        cause=None,
    )


class _RagResult:
    """Minimal duck-typed engine result (id/score/document/metadata)."""

    def __init__(self, doc_id: str, score: float, source_type: str):
        self.id = doc_id
        self.score = score
        self.document = f"doc {doc_id}"
        self.metadata = {"source_type": source_type, "source_id": doc_id}


class _UnionAwareRagService:
    """Fake runtime mirroring ``RAGService.search``'s union contract.

    Caller-visible accounting: ONE ``search()`` call is ONE query-embedding
    round (the engine embeds once per search call it receives from a caller).
    Inside one call, a multi-entry allowlist is served per entry -- one store
    query each, merged by score descending, trimmed to top_k -- exactly what
    ``_semantic_search_scoped`` does, so assertions can compare the single
    call's output against the old loop's merged output.
    """

    def __init__(self, results_by_source_type: Dict[str, List[_RagResult]]):
        self.results_by_source_type = results_by_source_type
        self.search_calls = 0
        self.embed_calls = 0
        self.last_metadata_allowlist: Any = None
        self.calls: List[Dict[str, Any]] = []

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
        **kwargs,
    ):
        self.search_calls += 1
        self.embed_calls += 1
        if metadata_allowlist is None:
            entries = None
        elif isinstance(metadata_allowlist, AbcMapping):
            entries = (metadata_allowlist,)
        else:
            entries = tuple(metadata_allowlist)
        self.last_metadata_allowlist = entries
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "search_type": search_type,
                "metadata_allowlist": entries,
            }
        )

        if entries is None:
            merged: List[_RagResult] = [
                row for rows in self.results_by_source_type.values() for row in rows
            ]
        else:
            merged = []
            for entry in entries:
                source_type = next(iter(entry.get("source_type", ())), None)
                merged.extend(self.results_by_source_type.get(source_type, []))
        merged.sort(key=lambda row: row.score, reverse=True)
        return merged[:top_k] if top_k else merged


def _expected_union(scope: EffectiveScope):
    return tuple(
        {"source_type": {source_type}, "source_id": set(ids)}
        for source_type, ids in sorted(
            (k, set(v)) for k, v in scope.allowlist.items()
        )
    )


def _reference_loop_merge(fake: _UnionAwareRagService, union, top_k):
    """What the OLD per-entry loop produced, computed from the fake's per-type
    result sets (the equivalence oracle for the single union call)."""
    merged: List[_RagResult] = []
    for entry in union:
        source_type = next(iter(entry["source_type"]), None)
        merged.extend(fake.results_by_source_type.get(source_type, []))
    merged.sort(key=lambda row: row.score, reverse=True)
    return merged[:top_k]


@pytest.mark.asyncio
async def test_library_scoped_semantic_single_embed_union_allowlist():
    fake = _UnionAwareRagService(
        {
            SOURCE_TYPE_MEDIA: [_RagResult("m1", 0.5, SOURCE_TYPE_MEDIA)],
            SOURCE_TYPE_NOTE: [_RagResult("n1", 0.9, SOURCE_TYPE_NOTE)],
        }
    )
    service = LibraryLocalRagSearchService(SimpleNamespace(_rag_service=fake))
    scope = _scoped(**{SOURCE_TYPE_MEDIA: {"m1"}, SOURCE_TYPE_NOTE: {"n1"}})

    result = await service.search(
        "test query", ("notes", "media"), "rag", top_k=5, scope=scope
    )

    assert fake.embed_calls == 1, (
        f"k allowlist entries must not re-embed the query k times "
        f"(embed rounds: {fake.embed_calls})"
    )
    assert fake.search_calls == 1
    assert fake.last_metadata_allowlist == _expected_union(scope)

    # Result equivalence with the old per-entry loop's merged output.
    ids = [row["source_id"] for row in result["results"]]
    expected_ids = [row.id for row in _reference_loop_merge(fake, _expected_union(scope), 5)]
    assert ids == expected_ids


@pytest.mark.asyncio
async def test_pipeline_scoped_semantic_single_embed_union_allowlist():
    fake = _UnionAwareRagService(
        {
            SOURCE_TYPE_MEDIA: [_RagResult("m1", 0.5, SOURCE_TYPE_MEDIA)],
            SOURCE_TYPE_NOTE: [_RagResult("n1", 0.9, SOURCE_TYPE_NOTE)],
        }
    )
    app = SimpleNamespace(_rag_service=fake)
    scope = _scoped(**{SOURCE_TYPE_MEDIA: {"m1"}, SOURCE_TYPE_NOTE: {"n1"}})

    results = await pfs.search_semantic(
        app, "test query", {"media": True, "notes": True}, limit=10, scope=scope
    )

    assert fake.embed_calls == 1, (
        f"k allowlist entries must not re-embed the query k times "
        f"(embed rounds: {fake.embed_calls})"
    )
    assert fake.search_calls == 1
    assert fake.last_metadata_allowlist == _expected_union(scope)

    ids = [row.id for row in results]
    expected_ids = [
        row.id for row in _reference_loop_merge(fake, _expected_union(scope), 10)
    ]
    assert ids == expected_ids
