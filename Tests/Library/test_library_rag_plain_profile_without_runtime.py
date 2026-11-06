"""A BM25 Only profile's `rag` search needs no RAG runtime (TASK-3601, TASK-33620.15.41).

`_search_rag` used to build the shared RAG runtime -- torch, transformers,
sentence-transformers, chromadb and the embedding model -- only to read the
active profile's `default_search_mode` off it. For a BM25 Only ("plain")
profile that read is all the runtime was used for: the route is the
Library's own four-seam keyword search. Live on dev 29592ecc9b the build took
7.7 s of a first Automatic Console send's 10.1 s at host load 11, and 26-29 s
under load (TASK-33620.15.41). With the `embeddings_rag` extras absent, the
same read turned a keyword search into "RAG unavailable -- install
embeddings" (TASK-3601).

These tests run the real `LibraryLocalRagSearchService` with the active
profile chosen through the real profile manager (a temporary profiles dir,
the real builtins), and replace only the shared-runtime factory, by one that
records each build.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Library import library_local_rag_search_service as service_module
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)
from tldw_chatbook.Library.library_rag_service import LibraryRagSearchOutcome

from Tests.Library.test_library_local_rag_search_service import FakeNotesScopeService

#: The builtin profiles are real `RAGConfig`s, whose defaults read the user
#: config through the config-participant admission; the per-test sandbox
#: redirect fails that closed (`raw_source_selection_changed`).
pytestmark = pytest.mark.bootstrap_profile

#: The disclosure the plain route attaches, spelled out (not imported) so this
#: file pins the copy rather than restating the implementation's constant.
_BM25_NOTE = "Profile 'BM25 Only': keyword search (no vectors)"
_ROUTE_NOTES_KEY = "retrieval_route_notes"


@pytest.fixture
def active_profile(tmp_path, monkeypatch):
    """Point the active-profile pointer at a builtin, over a temp profiles dir.

    Returns:
        A one-argument function that sets the active profile id.
    """
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


class _BuiltRuntime:
    """What the shared factory returns for the active profile, mode included."""

    def __init__(self, mode: str, name: str) -> None:
        self.config = SimpleNamespace(search=SimpleNamespace(default_search_mode=mode))
        self.profile = SimpleNamespace(name=name)
        self.searches: list[str] = []

    async def search(self, query, **_kwargs):
        self.searches.append(query)
        return []


def _recording_factory(monkeypatch, runtime: _BuiltRuntime) -> list[str]:
    """Replace the shared-runtime build with one that records each call."""
    builds: list[str] = []

    def get_shared_rag_service():
        builds.append("build")
        return runtime

    monkeypatch.setattr(service_module, "get_shared_rag_service", get_shared_rag_service)
    return builds


def _service_with_one_note() -> tuple[LibraryLocalRagSearchService, FakeNotesScopeService]:
    notes = FakeNotesScopeService(
        rows=[{"id": "note-1", "title": "Lighthouse", "content": "Fresnel lens."}]
    )
    return LibraryLocalRagSearchService(SimpleNamespace(notes_scope_service=notes)), notes


def _route_notes(result) -> tuple[str, ...]:
    diagnostics = (
        result.diagnostics
        if isinstance(result, LibraryRagSearchOutcome)
        else (result.get("diagnostics") or {})
    )
    return tuple(diagnostics.get(_ROUTE_NOTES_KEY, ()))


@pytest.mark.asyncio
async def test_bm25_only_rag_search_never_builds_the_runtime(monkeypatch, active_profile):
    """TASK-33620.15.41 AC#2: the plain route reads the profile, not a runtime."""
    active_profile("bm25_only")
    monkeypatch.setattr(service_module, "embeddings_rag_deps_installed", lambda: True)
    builds = _recording_factory(monkeypatch, _BuiltRuntime("plain", "BM25 Only"))
    service, notes = _service_with_one_note()

    assert await service.prewarm() is True
    result = await service.search("Fresnel lens", ("notes",), "rag", top_k=5)

    assert builds == [], "a BM25 Only search built the embedding runtime"
    assert notes.calls, "the plain route must run the four-seam keyword search"
    assert result["runtime_backend"] == "local-fts"
    assert [row["source_id"] for row in result["results"]] == ["note-1"]
    assert _BM25_NOTE in _route_notes(result)


@pytest.mark.asyncio
async def test_bm25_only_rag_search_works_without_the_embeddings_extras(
    monkeypatch, active_profile
):
    """TASK-3601 AC#1: no 'RAG unavailable' for a profile that needs no vectors."""
    active_profile("bm25_only")
    monkeypatch.setattr(service_module, "embeddings_rag_deps_installed", lambda: False)
    builds = _recording_factory(monkeypatch, _BuiltRuntime("plain", "BM25 Only"))
    service, notes = _service_with_one_note()

    assert await service.prewarm() is True
    result = await service.search("Fresnel lens", ("notes",), "rag", top_k=5)

    assert not isinstance(result, LibraryRagSearchOutcome), result
    assert builds == []
    assert notes.calls
    assert [row["source_id"] for row in result["results"]] == ["note-1"]
    assert _BM25_NOTE in _route_notes(result)


@pytest.mark.asyncio
async def test_an_embedding_profile_still_builds_the_runtime_and_routes_by_it(
    monkeypatch, active_profile
):
    """Control: a hybrid profile needs the runtime, and it is still built once."""
    active_profile("hybrid_basic")
    monkeypatch.setattr(service_module, "embeddings_rag_deps_installed", lambda: True)
    runtime = _BuiltRuntime("hybrid", "Hybrid Basic")
    builds = _recording_factory(monkeypatch, runtime)
    service, notes = _service_with_one_note()

    assert await service.prewarm() is True
    result = await service.search("Fresnel lens", ("notes",), "rag", top_k=5)

    assert builds == ["build"]
    assert runtime.searches == ["Fresnel lens"]
    assert not notes.calls, "a hybrid profile runs the engine, not the four seams"
    assert not any("BM25" in note for note in _route_notes(result))


@pytest.mark.asyncio
async def test_an_embedding_profile_without_the_extras_is_still_unavailable(
    monkeypatch, active_profile
):
    """Control: only the plain route stopped needing the extras."""
    active_profile("hybrid_basic")
    monkeypatch.setattr(service_module, "embeddings_rag_deps_installed", lambda: False)
    builds = _recording_factory(monkeypatch, _BuiltRuntime("hybrid", "Hybrid Basic"))
    service, _notes = _service_with_one_note()

    assert await service.prewarm() is False
    result = await service.search("Fresnel lens", ("notes",), "rag", top_k=5)

    assert builds == []
    assert isinstance(result, LibraryRagSearchOutcome)
    assert result.status == "blocked"
    assert result.recovery_state.status_label == "RAG unavailable"


@pytest.mark.asyncio
async def test_an_embedding_profile_searched_cold_builds_the_runtime_at_search_time(
    monkeypatch, active_profile
):
    """Control: no prewarm and no cached runtime -- `search` itself builds it.

    The control above goes through `prewarm`, which caches the runtime before
    the search runs, so the search never consults the profile route. This one
    reaches the route read inside `_search_rag`: a Hybrid read must still
    lead to one build and the engine's search, never the keyword seams.
    """
    active_profile("hybrid_basic")
    monkeypatch.setattr(service_module, "embeddings_rag_deps_installed", lambda: True)
    runtime = _BuiltRuntime("hybrid", "Hybrid Basic")
    builds = _recording_factory(monkeypatch, runtime)
    service, notes = _service_with_one_note()

    result = await service.search("Fresnel lens", ("notes",), "rag", top_k=5)

    assert builds == ["build"]
    assert runtime.searches == ["Fresnel lens"]
    assert not notes.calls, "a hybrid profile runs the engine, not the four seams"
    assert not any("BM25" in note for note in _route_notes(result))


@pytest.mark.asyncio
async def test_a_switch_from_bm25_to_hybrid_rereads_the_profile(
    monkeypatch, active_profile
):
    """The route read is per shared-service generation, not per service.

    A profile switch resets the shared RAG service, which bumps its
    generation. A read kept across that would go on answering "plain" for a
    user who has switched to Hybrid: a keyword search under a stale
    "Profile 'BM25 Only'" disclosure, with no runtime ever built.
    """
    from tldw_chatbook.RAG_Search.ingestion_indexing import reset_shared_rag_service

    active_profile("bm25_only")
    monkeypatch.setattr(service_module, "embeddings_rag_deps_installed", lambda: True)
    runtime = _BuiltRuntime("hybrid", "Hybrid Basic")
    builds = _recording_factory(monkeypatch, runtime)
    service, notes = _service_with_one_note()

    first = await service.search("Fresnel lens", ("notes",), "rag", top_k=5)
    assert builds == []
    assert _BM25_NOTE in _route_notes(first)

    # What `active_config.set_active_profile()` does on a switch.
    active_profile("hybrid_basic")
    reset_shared_rag_service()
    notes.calls.clear()
    second = await service.search("Fresnel lens", ("notes",), "rag", top_k=5)

    assert builds == ["build"], "the switch to Hybrid kept the stale BM25 route"
    assert runtime.searches == ["Fresnel lens"]
    assert not notes.calls
    assert not any("BM25" in note for note in _route_notes(second))


@pytest.mark.asyncio
async def test_an_unreadable_profile_falls_back_to_the_runtime(
    monkeypatch, active_profile
):
    """A profile read that fails must not guess "plain": the runtime decides."""
    import tldw_chatbook.RAG_Search.simplified.active_config as active_config

    def broken():
        raise OSError("profiles dir unreadable")

    monkeypatch.setattr(active_config, "_active_profile_id", broken)
    monkeypatch.setattr(service_module, "embeddings_rag_deps_installed", lambda: True)
    builds = _recording_factory(monkeypatch, _BuiltRuntime("plain", "BM25 Only"))
    service, notes = _service_with_one_note()

    result = await service.search("Fresnel lens", ("notes",), "rag", top_k=5)

    assert builds == ["build"]
    assert notes.calls
    assert _BM25_NOTE in _route_notes(result)


@pytest.mark.asyncio
async def test_an_unreadable_profile_warns_once_per_generation(
    monkeypatch, active_profile
):
    """Each search retries the read, but the warning is not repeated per search.

    A profile that stays unreadable (in the per-test sandbox, every search)
    used to log one warning per search. The read is still retried each time,
    so a profile that becomes readable is picked up; the warning comes back
    after a profile switch or save, which is a new generation.
    """
    import tldw_chatbook.RAG_Search.simplified.active_config as active_config
    from tldw_chatbook.RAG_Search.ingestion_indexing import reset_shared_rag_service

    reads: list[str] = []

    def broken():
        reads.append("read")
        raise OSError("profiles dir unreadable")

    monkeypatch.setattr(active_config, "_active_profile_id", broken)
    monkeypatch.setattr(service_module, "embeddings_rag_deps_installed", lambda: False)
    _recording_factory(monkeypatch, _BuiltRuntime("plain", "BM25 Only"))
    service, _notes = _service_with_one_note()
    warnings: list[str] = []
    sink = service_module.logger.add(
        lambda message: warnings.append(str(message)),
        level="WARNING",
        filter=lambda record: "route unreadable" in record["message"],
    )
    try:
        for _ in range(3):
            await service.search("Fresnel lens", ("notes",), "rag", top_k=5)
        assert len(reads) == 3, "every search must retry the profile read"
        assert len(warnings) == 1, warnings

        reset_shared_rag_service()
        await service.search("Fresnel lens", ("notes",), "rag", top_k=5)
        assert len(reads) == 4
        assert len(warnings) == 2, "a new generation warns again"
    finally:
        service_module.logger.remove(sink)


_PLAIN_ROUTE_IMPORTS_SNIPPET = """
import asyncio
import sys
from types import SimpleNamespace

#: What the shared RAG runtime build imports (TASK-33620.15.41's live probe on
#: dev 29592ecc9b: 6.5 s of a 7.7 s build at load 11, 23-26 s under load).
EMBEDDING_STACK = ("torch", "transformers", "sentence_transformers", "chromadb")

import tldw_chatbook.RAG_Search.simplified.active_config as active_config

active_config._active_profile_id = lambda: "bm25_only"

from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)


class Notes:
    async def search_notes(self, **kwargs):
        return [{"id": "note-1", "title": "Lighthouse", "content": "Fresnel lens."}]


async def main():
    service = LibraryLocalRagSearchService(SimpleNamespace(notes_scope_service=Notes()))
    built = await service.prewarm()
    result = await service.search("Fresnel lens", ("notes",), "rag", top_k=5)
    return built, result


built, result = asyncio.run(main())
loaded = [name for name in EMBEDDING_STACK if name in sys.modules]
print("PREWARM:" + repr(built))
print("BACKEND:" + str(getattr(result, "get", lambda _k: result)("runtime_backend")))
print("EMBEDDING_STACK:" + ",".join(loaded))
"""


def test_a_bm25_only_prewarm_and_search_import_no_embedding_stack(tmp_path, monkeypatch):
    """AC#3, end to end in a fresh interpreter, with the REAL runtime factory.

    The in-process tests above replace the factory; this one does not, so it
    also goes red if the plain route ever reaches torch another way (for
    example through `resolve_active_rag_config`'s `device == "auto"` probe).
    Offline, so a regression fails here instead of downloading a model.
    """
    from Tests.Packaging.test_rag_boot_import_closure import _run_isolated_python

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    result = _run_isolated_python(tmp_path, _PLAIN_ROUTE_IMPORTS_SNIPPET)

    assert result.returncode == 0, result.stderr[-4000:]
    lines = result.stdout.splitlines()
    assert "PREWARM:True" in lines, result.stdout
    assert "BACKEND:local-fts" in lines, result.stdout
    assert "EMBEDDING_STACK:" in lines, (
        "a BM25 Only prewarm/search imported the embedding stack: "
        f"{[line for line in lines if line.startswith('EMBEDDING_STACK')]}"
    )
