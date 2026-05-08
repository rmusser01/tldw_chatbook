"""Library-native Search/RAG retrieval adapter contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Library.library_rag_service import (
    LibraryRagSearchRequest,
    run_library_rag_search,
)
from tldw_chatbook.runtime_policy.types import PolicyDeniedError


class StaticLibraryRagSearchService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def search(self, query, scope, mode, **kwargs):
        self.calls.append(
            {
                "query": query,
                "scope": scope,
                "mode": mode,
                **kwargs,
            }
        )
        return self.result


class RaisingLibraryRagSearchService:
    async def search(self, query, scope, mode, **kwargs):
        raise RuntimeError("index unavailable")


class PolicyDeniedLibraryRagSearchService:
    async def search(self, query, scope, mode, **kwargs):
        raise PolicyDeniedError(
            action_id="library.rag.search",
            reason_code="wrong_source",
            user_message="Server Library RAG requires server mode.",
            effective_source="local",
            authority_owner="active server",
        )


@pytest.mark.asyncio
async def test_run_library_rag_search_normalizes_results_and_preserves_metadata() -> None:
    service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Incident Review",
                    "snippet": "Expired credential caused the incident.",
                    "score": "0.93",
                    "source_id": "note-42",
                    "chunk_id": "chunk-7",
                    "runtime_backend": "local-fts",
                    "citations": [{"label": "Incident Review p.2"}],
                    "provenance": {"rank": 1, "index": "library"},
                }
            ],
            "runtime_backend": "local-fts",
        }
    )
    app = SimpleNamespace(library_rag_search_service=service)
    request = LibraryRagSearchRequest(
        query="why did the incident happen?",
        source_types=("notes", "media"),
        mode="rag",
        top_k=3,
    )

    outcome = await run_library_rag_search(app, request)

    assert service.calls == [
        {
            "query": "why did the incident happen?",
            "scope": ("notes", "media"),
            "mode": "rag",
            "top_k": 3,
            "include_citations": True,
        }
    ]
    assert outcome.status == "ready"
    assert outcome.recovery_state is None
    assert len(outcome.results) == 1
    row = outcome.results[0]
    assert row.title == "Incident Review"
    assert row.snippet == "Expired credential caused the incident."
    assert row.score == 0.93
    assert row.source_id == "note-42"
    assert row.chunk_id == "chunk-7"
    assert row.runtime_backend == "local-fts"
    assert row.citation_labels == ("Incident Review p.2",)
    assert row.provenance["rank"] == 1


@pytest.mark.asyncio
async def test_run_library_rag_search_returns_unavailable_recovery_without_service() -> None:
    outcome = await run_library_rag_search(
        SimpleNamespace(),
        LibraryRagSearchRequest(query="policy", source_types=("notes",), mode="search"),
    )

    assert outcome.status == "blocked"
    assert outcome.results == ()
    assert outcome.recovery_state is not None
    assert outcome.recovery_state.stable_selector == "library-rag-service-error"
    assert "Unavailable: Library Search/RAG retrieval." in outcome.recovery_state.visible_copy
    assert "Owner: Library retrieval service." in outcome.recovery_state.visible_copy


@pytest.mark.asyncio
async def test_run_library_rag_search_maps_policy_denied_to_persistent_recovery() -> None:
    outcome = await run_library_rag_search(
        SimpleNamespace(library_rag_search_service=PolicyDeniedLibraryRagSearchService()),
        LibraryRagSearchRequest(query="policy", source_types=("notes",), mode="search"),
    )

    assert outcome.status == "blocked"
    assert outcome.recovery_state is not None
    assert outcome.recovery_state.status_label == "Wrong source"
    assert "Server Library RAG requires server mode." in outcome.recovery_state.visible_copy
    assert "Owner: active server." in outcome.recovery_state.visible_copy


@pytest.mark.asyncio
async def test_run_library_rag_search_maps_empty_and_failed_results_to_recovery() -> None:
    empty = await run_library_rag_search(
        SimpleNamespace(library_rag_search_service=StaticLibraryRagSearchService([])),
        LibraryRagSearchRequest(query="missing", source_types=("notes",), mode="search"),
    )

    assert empty.status == "empty"
    assert empty.results == ()
    assert empty.recovery_state is not None
    assert "No evidence matched the current query." in empty.recovery_state.visible_copy

    failed = await run_library_rag_search(
        SimpleNamespace(library_rag_search_service=RaisingLibraryRagSearchService()),
        LibraryRagSearchRequest(query="policy", source_types=("notes",), mode="search"),
    )

    assert failed.status == "failed"
    assert failed.results == ()
    assert failed.recovery_state is not None
    assert "Library Search/RAG retrieval failed." in failed.recovery_state.visible_copy
    assert "index unavailable" not in failed.recovery_state.visible_copy
