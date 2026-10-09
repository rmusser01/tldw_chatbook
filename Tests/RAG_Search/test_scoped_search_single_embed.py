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

The caller tests below verify union forwarding and result equivalence. The
real-engine tests use the actual in-memory store and count embedding calls
at the external model seam for both basic and citation-bearing searches.
"""

from collections.abc import Mapping as AbcMapping
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

import pytest

from tldw_chatbook.Chat.rag_scope import (
    EffectiveScope,
    SOURCE_TYPE_MEDIA,
    SOURCE_TYPE_NOTE,
)
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)
from tldw_chatbook.RAG_Search import pipeline_functions_simple as pfs
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search.simplified.vector_store import InMemoryVectorStore


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

    Models only the caller-visible result union; embedding accounting is
    covered separately against the real engine.
    """

    def __init__(self, results_by_source_type: Dict[str, List[_RagResult]]):
        self.results_by_source_type = results_by_source_type
        self.search_calls = 0
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
        for source_type, ids in sorted((k, set(v)) for k, v in scope.allowlist.items())
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
async def test_library_scoped_semantic_single_search_union_allowlist():
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

    assert fake.search_calls == 1
    assert fake.last_metadata_allowlist == _expected_union(scope)

    # Result equivalence with the old per-entry loop's merged output.
    ids = [row["source_id"] for row in result["results"]]
    expected_ids = [
        row.id for row in _reference_loop_merge(fake, _expected_union(scope), 5)
    ]
    assert ids == expected_ids


@pytest.mark.asyncio
async def test_pipeline_scoped_semantic_single_search_union_allowlist():
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

    assert fake.search_calls == 1
    assert fake.last_metadata_allowlist == _expected_union(scope)

    ids = [row.id for row in results]
    expected_ids = [
        row.id for row in _reference_loop_merge(fake, _expected_union(scope), 10)
    ]
    assert ids == expected_ids


@pytest.mark.parametrize("include_citations", [False, True])
@pytest.mark.parametrize("source_types", [("media",), ("media", "note")])
async def test_real_engine_embeds_once_and_preserves_scoped_results(
    monkeypatch, include_citations, source_types
):
    service = RAGService.__new__(RAGService)
    service.vector_store = InMemoryVectorStore()
    service.embeddings = SimpleNamespace()
    embedding_calls = []

    async def embed(texts):
        embedding_calls.append(texts)
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        service.embeddings, "create_embeddings_async", embed, raising=False
    )
    service.vector_store.add(
        ids=["m1", "n1", "outside"],
        embeddings=[[0.8, 0.6], [1.0, 0.0], [1.0, 0.0]],
        documents=["media query", "note query", "outside query"],
        metadata=[
            {"source_type": "media", "source_id": "m1"},
            {"source_type": "note", "source_id": "n1"},
            {"source_type": "note", "source_id": "outside"},
        ],
    )
    allowlists = [
        {"source_type": {source_type}, "source_id": {"m1", "n1"}}
        for source_type in source_types
    ]
    results = await service._semantic_search_scoped(
        "query",
        top_k=3,
        include_citations=include_citations,
        metadata_allowlist=allowlists,
    )

    assert [result.id for result in results] == (
        ["m1"] if len(source_types) == 1 else ["n1", "m1"]
    )
    assert embedding_calls == [["query"]]
