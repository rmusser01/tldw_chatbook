"""A BM25 Only Automatic send never builds the embedding runtime (TASK-33620.15.41).

Live on dev 29592ecc9b (BM25 Only, Library Auto: Automatic), the first send
after launch spent 7.7 s of its 10.1 s Enter-to-provider wait in the Library
search service's `prewarm` at host load 11, and 26.5-29.2 s of 33.6-36.4 s at
load 98-110. That prewarm is the shared RAG runtime build: mostly importing
sentence-transformers, torch, transformers and chromadb, plus the embedding
model load. A BM25 Only search reads no vectors, so none of it was needed.

These tests drive the real Console controller's Automatic preparation over
the real `LibraryLocalRagSearchService`, with the active RAG profile chosen
through the real profile manager. Only the shared-runtime factory is
replaced, by one that records each build, and the notes seam is a double.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_automatic_library_preparation import (
    _controller_for_preparation,
    _preparation,
)
from Tests.Library.test_library_local_rag_search_service import FakeNotesScopeService
from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState
from tldw_chatbook.Library import library_local_rag_search_service as service_module
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def active_profile(tmp_path, monkeypatch):
    """Point the active-profile pointer at a builtin, over a temp profiles dir."""
    from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager
    import tldw_chatbook.RAG_Search.simplified.active_config as active_config

    manager = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    state = {"active": "hybrid_basic"}
    monkeypatch.setattr(active_config, "_manager", lambda: manager)
    monkeypatch.setattr(active_config, "_active_profile_id", lambda: state["active"])
    monkeypatch.delenv("RAG_SEARCH_MODE", raising=False)

    def choose(profile_id: str) -> None:
        state["active"] = profile_id

    return choose


class _Runtime:
    """The shared runtime the factory would build for the active profile."""

    def __init__(self, mode: str, name: str) -> None:
        self.config = SimpleNamespace(search=SimpleNamespace(default_search_mode=mode))
        self.profile = SimpleNamespace(name=name)
        self.searches: list[str] = []

    async def search(self, query, **_kwargs):
        self.searches.append(query)
        return [
            {
                "id": "media_m1",
                "score": 0.9,
                "document": "Fresnel lens lighthouse keepers.",
                "metadata": {
                    "title": "Lighthouse",
                    "source_id": "m1",
                    "source_type": "media",
                },
            }
        ]


def _library(monkeypatch, runtime: _Runtime):
    """The real search service, a notes seam with one note, and a build log."""
    builds: list[str] = []

    def get_shared_rag_service():
        builds.append("build")
        return runtime

    monkeypatch.setattr(service_module, "embeddings_rag_deps_installed", lambda: True)
    monkeypatch.setattr(service_module, "get_shared_rag_service", get_shared_rag_service)
    notes = FakeNotesScopeService(
        rows=[
            {
                "id": "note-1",
                "title": "Lighthouse keeping notes",
                "content": "Fresnel lens lighthouse keepers trimmed the wicks.",
            }
        ]
    )
    service = LibraryLocalRagSearchService(SimpleNamespace(notes_scope_service=notes))
    return service, builds, notes


@pytest.mark.asyncio
async def test_a_bm25_only_automatic_send_never_builds_the_embedding_runtime(
    monkeypatch, active_profile
):
    active_profile("bm25_only")
    runtime = _Runtime("plain", "BM25 Only")
    service, builds, notes = _library(monkeypatch, runtime)
    controller, store = _controller_for_preparation(
        _preparation(draft="Fresnel lens lighthouse keepers"), service
    )

    outcome = await asyncio.wait_for(
        controller.prepare_library_for_turn("preparation-1"), timeout=10
    )

    assert builds == [], "a BM25 Only Automatic send built the embedding runtime"
    assert runtime.searches == []
    assert outcome.error_code is None, outcome.error_code
    assert outcome.state is ConsoleTurnPreparationState.READY
    # The send still carries the keyword evidence.
    assert notes.calls
    assert [ref.source_id for ref in outcome.evidence_bundle.references] == ["note-1"]
    assert store.preparation_for_session("session-1").pause_kind is None


@pytest.mark.asyncio
async def test_an_embedding_profile_automatic_send_still_builds_and_searches_it(
    monkeypatch, active_profile
):
    """Control: a hybrid profile genuinely needs the runtime; it is built once."""
    active_profile("hybrid_basic")
    runtime = _Runtime("hybrid", "Hybrid Basic")
    service, builds, notes = _library(monkeypatch, runtime)
    controller, _store = _controller_for_preparation(
        _preparation(draft="Fresnel lens lighthouse keepers"), service
    )

    outcome = await asyncio.wait_for(
        controller.prepare_library_for_turn("preparation-1"), timeout=10
    )

    assert builds == ["build"]
    assert runtime.searches == ["Fresnel lens lighthouse keepers"]
    assert not notes.calls
    assert outcome.state is ConsoleTurnPreparationState.READY
    assert [ref.source_id for ref in outcome.evidence_bundle.references] == ["m1"]
