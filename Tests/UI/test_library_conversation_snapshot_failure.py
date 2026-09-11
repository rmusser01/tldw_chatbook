"""Independent conversation paging survives unrelated source-snapshot failures."""

import asyncio

import pytest
from textual.widgets import Input

from Tests.UI.test_library_conversation_recovery_flow import wait_until
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
)
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_MODE
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


@pytest.mark.asyncio
async def test_cold_conversation_entry_reconciles_successful_page_after_source_failure(
    monkeypatch,
):
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    release = asyncio.Event()

    async def failed_snapshot(self):
        await release.wait()
        return (
            {"notes": (), "media": (), "conversations": ()},
            {"notes": 0, "media": 0, "conversations": 0},
            {"notes": False, "media": False, "conversations": False},
            "Unrelated Library source service unavailable.",
            None,
            {"study_decks": None, "flashcards_due": None, "quizzes": None},
        )

    monkeypatch.setattr(LibraryScreen, "_list_local_source_snapshot", failed_snapshot)
    screen = LibraryScreen(app)
    screen.apply_navigation_context(
        {LIBRARY_NAV_CONTEXT_MODE: "conversations", "conversation_archive_scope": "all"}
    )
    host = LibraryHarness(app, screen=screen)
    try:
        async with host.run_test(size=(120, 40)) as pilot:
            await wait_until(pilot, lambda: screen._conversations_state.page_loaded)
            assert screen._conversations_state.total == 2
            assert screen.query("#library-canvas-loading")
            release.set()
            await wait_until(
                pilot,
                lambda: (
                    screen._library_loaded
                    and screen._library_snapshot_rendered_generation
                    == screen._library_snapshot_state_generation
                ),
            )
            assert screen._library_lookup_error
            assert screen.query("#library-conversations-canvas")
            assert not screen.query("#library-canvas-error")
            assert screen.query_one("#library-conversations-filter", Input).value == ""
            await wait_until(
                pilot,
                lambda: (
                    screen._conversations_state.reader_state.loaded_actions_eligible
                ),
            )
            assert screen._conversations_state.reader_state.loaded_id == "chat-1"
    finally:
        release.set()
