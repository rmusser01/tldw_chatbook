"""TASK-18903: a seam that FAILED must not read as a seam that found nothing.

Every keyword seam used to end `except Exception: return True, []` -- `True`
meaning AVAILABLE. The merge site gates on "any seam available", so a total
backend failure passed the gate, produced zero rows, and normalized to
`status="empty"` -- which is in `LIBRARY_RAG_ANSWERABLE_RETRIEVAL_STATUSES`,
so the RAG answer path then generated an answer with NO retrieved context and
presented it as Library-grounded.

These pins cover the three states and, most importantly, that a total failure
is not answerable.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Library.library_local_rag_search_service import (
    KEYWORD_SEAM_DIAGNOSTICS_KEY,
    SEAM_STATUS_FAILED,
    LibraryLocalRagSearchService,
    SeamState,
)

ALL_TYPES = ("notes", "media", "conversations", "prompts")


class _Exploding:
    """Any awaited method raises -- a configured backend that is broken."""

    def __init__(self, label: str) -> None:
        self._label = label

    def __getattr__(self, name):
        async def boom(*_a, **_k):
            raise RuntimeError(f"{self._label} backend down ({name})")

        return boom


class _ExplodingSync:
    """Sync-method backend (chachanotes_db) that raises."""

    def __init__(self, label: str) -> None:
        self._label = label

    def __getattr__(self, name):
        def boom(*_a, **_k):
            raise RuntimeError(f"{self._label} backend down ({name})")

        return boom


def _all_broken_app() -> SimpleNamespace:
    return SimpleNamespace(
        notes_scope_service=_Exploding("notes"),
        media_reading_scope_service=_Exploding("media"),
        chachanotes_db=_ExplodingSync("chachanotes"),
        prompt_scope_service=_Exploding("prompts"),
        notes_user_id="u",
    )


class TestSeamStates:
    @pytest.mark.asyncio
    async def test_unconfigured_seam_is_UNAVAILABLE(self):
        service = LibraryLocalRagSearchService(SimpleNamespace())
        state, rows = await service._search_prompts("q", 5)
        assert state is SeamState.UNAVAILABLE
        assert rows == []

    @pytest.mark.asyncio
    async def test_throwing_seam_is_FAILED_not_available(self):
        """THE BUG. A configured backend that raises used to return
        `(True, [])` -- claiming health while broken."""
        service = LibraryLocalRagSearchService(_all_broken_app())
        for coro in (
            service._search_notes("q", 5, "u"),
            service._search_media("q", 5),
            service._search_conversations("q", 5),
            service._search_prompts("q", 5),
        ):
            state, rows = await coro
            assert state is SeamState.FAILED, "a thrown seam must not report available"
            assert rows == []


class TestTotalFailure:
    @pytest.mark.asyncio
    async def test_total_failure_is_not_a_zero_row_success(self):
        """The headline. Before: a plain dict with results=[] -- a successful
        search that matched nothing."""
        service = LibraryLocalRagSearchService(_all_broken_app())
        outcome = await service.search("q", ALL_TYPES, "search", top_k=5)
        assert not isinstance(outcome, dict), (
            "a total backend failure must not present as a successful search"
        )
        assert outcome.status == "failed"

    @pytest.mark.asyncio
    async def test_total_failure_is_NOT_answerable(self):
        """R1: `empty` is in LIBRARY_RAG_ANSWERABLE_RETRIEVAL_STATUSES, so the
        old behaviour let a total outage reach the answer path and generate an
        answer from no context at all."""
        from tldw_chatbook.UI.Screens.library_screen import (
            LIBRARY_RAG_ANSWERABLE_RETRIEVAL_STATUSES,
        )

        service = LibraryLocalRagSearchService(_all_broken_app())
        outcome = await service.search("q", ALL_TYPES, "search", top_k=5)
        assert outcome.status not in LIBRARY_RAG_ANSWERABLE_RETRIEVAL_STATUSES

    @pytest.mark.asyncio
    async def test_total_failure_carries_a_recovery_state_naming_the_seams(self):
        service = LibraryLocalRagSearchService(_all_broken_app())
        outcome = await service.search("q", ALL_TYPES, "search", top_k=5)
        assert outcome.recovery_state is not None
        assert outcome.recovery_state.status_label


class TestAllUnavailableUnchanged:
    @pytest.mark.asyncio
    async def test_no_configured_seam_still_blocks(self):
        """Byte-identical to today: nothing configured is `blocked`, not
        `failed`. This pin also catches the enum-truthiness trap -- an `Enum`
        member is truthy, so a gate left as `if not any(state ...)` would go
        silently inert and stop blocking."""
        service = LibraryLocalRagSearchService(SimpleNamespace())
        outcome = await service.search("q", ALL_TYPES, "search", top_k=5)
        assert outcome.status == "blocked"


class TestPartialFailure:
    @pytest.mark.asyncio
    async def test_partial_failure_returns_survivors_and_records_the_failures(self):
        """Silence must not read as absence: the caller gets what worked AND
        is told what did not run."""

        class _Notes:
            async def search_notes(self, **_k):
                return [
                    {
                        "id": 1,
                        "title": "t",
                        "content": "c",
                        "last_modified": "2026-01-01",
                    }
                ]

        app = SimpleNamespace(
            notes_scope_service=_Notes(),
            prompt_scope_service=_Exploding("prompts"),
            notes_user_id="u",
        )
        outcome = await LibraryLocalRagSearchService(app).search(
            "c", ("notes", "prompts"), "search", top_k=5
        )
        # survivors still returned
        rows = outcome["results"] if isinstance(outcome, dict) else outcome.results
        assert rows, "a surviving seam's rows must still be returned"
        diagnostics = (
            outcome["diagnostics"] if isinstance(outcome, dict) else outcome.diagnostics
        )
        entries = diagnostics.get(KEYWORD_SEAM_DIAGNOSTICS_KEY) or []
        assert any(
            e.get("status") == SEAM_STATUS_FAILED and e.get("seam") == "prompts"
            for e in entries
        ), f"the failed seam must be recorded; got {entries!r}"

    @pytest.mark.asyncio
    async def test_healthy_search_records_no_failure_entries(self):
        class _Notes:
            async def search_notes(self, **_k):
                return []

        app = SimpleNamespace(notes_scope_service=_Notes(), notes_user_id="u")
        outcome = await LibraryLocalRagSearchService(app).search(
            "q", ("notes",), "search", top_k=5
        )
        diagnostics = (
            outcome["diagnostics"] if isinstance(outcome, dict) else outcome.diagnostics
        )
        assert not (diagnostics.get(KEYWORD_SEAM_DIAGNOSTICS_KEY) or [])


class TestUserVisibleNotice:
    @pytest.mark.asyncio
    async def test_partial_failure_emits_a_rendered_route_note(self):
        """The structured slot is for machines; this sentence is what the user
        reads. Without it we would ship a diagnostics key nothing renders --
        the 'declared but inert' trap this repo has hit before."""
        from tldw_chatbook.Library.library_rag_state import (
            LIBRARY_RAG_ROUTE_NOTES_KEY,
        )

        class _Notes:
            async def search_notes(self, **_k):
                return [
                    {"id": 1, "title": "t", "content": "c",
                     "last_modified": "2026-01-01"}
                ]

        app = SimpleNamespace(
            notes_scope_service=_Notes(),
            prompt_scope_service=_Exploding("prompts"),
            notes_user_id="u",
        )
        outcome = await LibraryLocalRagSearchService(app).search(
            "c", ("notes", "prompts"), "search", top_k=5
        )
        diagnostics = (
            outcome["diagnostics"] if isinstance(outcome, dict) else outcome.diagnostics
        )
        notes = diagnostics.get(LIBRARY_RAG_ROUTE_NOTES_KEY) or []
        assert any("prompts" in str(n) and "failed" in str(n) for n in notes), (
            f"the user must be told which seam failed; got {notes!r}"
        )

    @pytest.mark.asyncio
    async def test_healthy_search_emits_no_failure_route_note(self):
        from tldw_chatbook.Library.library_rag_state import (
            LIBRARY_RAG_ROUTE_NOTES_KEY,
        )

        class _Notes:
            async def search_notes(self, **_k):
                return []

        app = SimpleNamespace(notes_scope_service=_Notes(), notes_user_id="u")
        outcome = await LibraryLocalRagSearchService(app).search(
            "q", ("notes",), "search", top_k=5
        )
        diagnostics = (
            outcome["diagnostics"] if isinstance(outcome, dict) else outcome.diagnostics
        )
        notes = diagnostics.get(LIBRARY_RAG_ROUTE_NOTES_KEY) or []
        assert not any("failed" in str(n) for n in notes)
