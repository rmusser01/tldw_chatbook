"""The Conversations reader re-checks its loaded transcript on each list read.

TASK-33628.10. A return visit to the reused Library re-reads the list; the
reader now re-reads the saved transcript's ``message_epoch``/``message_total``
once per list read and reloads only when either moved. The full Console
Delete/Undo journey lives in ``test_library_reader_console_delete_refresh``;
these cases pin the re-check's own rules against a scripted detail service.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from textual.screen import Screen
from textual.widgets import Button

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _StaticLibraryConversationDetailService,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
)
from tldw_chatbook.UI.Library_Modules.library_conversation_reader_freshness import (
    RECHECK_WORKER_GROUP,
)
from tldw_chatbook.UI.Screens.library_screen import (
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LibraryScreen,
)

# Building the app goes through config-participant admission, which the
# per-test sandbox refuses (RecoveryRequired); keep the collection-time profile.
pytestmark = pytest.mark.bootstrap_profile


def _records() -> list[dict[str, Any]]:
    return [
        {"id": "chat-a", "title": "Alpha planning", "version": 4, "message_count": 2},
        {"id": "chat-b", "title": "Beta review", "version": 7, "message_count": 1},
    ]


class _ScriptedDetailService(_StaticLibraryConversationDetailService):
    """The static detail fixture, with a movable saved transcript per id.

    ``recheck_gate``, when set, holds every one-message, one-character read
    (the re-check's shape) on a worker thread until the test releases it.
    """

    def __init__(self, conversations: list[dict[str, Any]]) -> None:
        super().__init__(conversations)
        self.calls: list[dict[str, Any]] = []
        self.epochs: dict[str, str] = {}
        self.recheck_gate: threading.Event | None = None
        self.recheck_started = threading.Event()
        self.recheck_error: Exception | None = None

    def set_saved_count(self, conversation_id: str, count: int) -> None:
        self._records[conversation_id] = {
            **self._records[conversation_id],
            "message_count": count,
        }
        self.epochs[conversation_id] = f"epoch-{conversation_id}-{count}"

    def get_library_conversation_messages(self, conversation_id, **kwargs):
        self.calls.append({"conversation_id": conversation_id, **kwargs})
        if _is_recheck(kwargs):
            self.recheck_started.set()
            if self.recheck_gate is not None:
                assert self.recheck_gate.wait(timeout=10)
            if self.recheck_error is not None:
                raise self.recheck_error
        response = super().get_library_conversation_messages(conversation_id, **kwargs)
        if response is not None and conversation_id in self.epochs:
            response["message_epoch"] = self.epochs[conversation_id]
        return response


def _is_recheck(call: dict[str, Any]) -> bool:
    return call.get("message_limit") == 1 and call.get("max_chars") == 1


def _rechecks(service: _ScriptedDetailService) -> list[dict[str, Any]]:
    return [call for call in service.calls if _is_recheck(call)]


def _page_reads(service: _ScriptedDetailService, conversation_id: str) -> int:
    return sum(
        1
        for call in service.calls
        if call["conversation_id"] == conversation_id and not _is_recheck(call)
    )


async def _open_alpha(
    pilot: Any, host: LibraryHarness
) -> tuple[LibraryScreen, _ScriptedDetailService]:
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await _wait_for_condition(
        pilot,
        lambda: (
            screen._conversations_state.reader_state.loaded_id == "chat-a"
            and screen._conversations_state.reader_state.loaded_actions_eligible
        ),
        message=lambda: screen._conversations_state.reader_state,
    )
    return screen, host.app_instance.local_chat_conversation_service


def _harness(service: _ScriptedDetailService | None = None) -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(app, _records())
    app.local_chat_conversation_service = service or _ScriptedDetailService(_records())
    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS})
    return LibraryHarness(app, screen=screen)


async def _leave_and_return(host: LibraryHarness, pilot: Any) -> None:
    """Cover Library with another screen and come back, as navigation does."""
    library = _active_library_screen(host)
    await host.push_screen(Screen())
    await pilot.pause()
    host.pop_screen()
    await _wait_for_condition(
        pilot,
        lambda: host.screen is library,
        message="Library did not resume",
    )


@pytest.mark.asyncio
async def test_first_list_read_loads_without_a_recheck() -> None:
    """A load this list read started is current; later ensures skip the re-check.

    Compose, the source-snapshot reconcile and inspection admission all call
    ``_ensure_library_conversation_reader_selection`` again within one list
    read; none of them may re-read a transcript that read just loaded.
    """
    host = _harness()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, service = await _open_alpha(pilot, host)
        await screen.workers.wait_for_complete()
        screen._ensure_library_conversation_reader_selection()
        await pilot.pause()
        await screen.workers.wait_for_complete()
        assert _rechecks(service) == []
        assert screen._conversations_state.reader_state.message_total == 2


@pytest.mark.asyncio
async def test_return_visit_reloads_a_transcript_whose_saved_epoch_moved() -> None:
    host = _harness()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, service = await _open_alpha(pilot, host)
        before = screen._conversations_state.reader_state
        service.set_saved_count("chat-a", 3)

        await _leave_and_return(host, pilot)

        await _wait_for_condition(
            pilot,
            lambda: (
                screen._conversations_state.reader_state.message_total == 3
                and screen._conversations_state.reader_state.loaded_actions_eligible
            ),
            message=lambda: screen._conversations_state.reader_state,
        )
        after = screen._conversations_state.reader_state
        assert after.loaded_id == "chat-a"
        assert after.loaded_generation != before.loaded_generation
        assert after.mode == before.mode
        assert len(_rechecks(service)) == 1
        await _wait_for_condition(
            pilot,
            lambda: len(screen.query(".library-conversation-reader-message")) == 3,
            message="the reloaded transcript was not painted",
        )


@pytest.mark.asyncio
async def test_return_visit_reloads_an_edited_transcript_with_the_same_count() -> None:
    """An in-place edit moves only the epoch (a row version), never the count."""
    host = _harness()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, service = await _open_alpha(pilot, host)
        before = screen._conversations_state.reader_state
        service.epochs["chat-a"] = "epoch-chat-a-after-edit"

        await _leave_and_return(host, pilot)

        await _wait_for_condition(
            pilot,
            lambda: (
                screen._conversations_state.reader_state.message_epoch
                == "epoch-chat-a-after-edit"
                and screen._conversations_state.reader_state.loaded_actions_eligible
            ),
            message=lambda: screen._conversations_state.reader_state,
        )
        after = screen._conversations_state.reader_state
        assert after.loaded_generation != before.loaded_generation
        assert after.message_total == before.message_total == 2


@pytest.mark.asyncio
async def test_return_visit_keeps_an_unchanged_transcript_without_reloading() -> None:
    host = _harness()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, service = await _open_alpha(pilot, host)
        before = screen._conversations_state.reader_state
        page_reads = len(service.calls)

        await _leave_and_return(host, pilot)
        await _wait_for_condition(
            pilot,
            lambda: len(_rechecks(service)) == 1,
            message=lambda: service.calls,
        )
        await screen.workers.wait_for_complete()

        assert screen._conversations_state.reader_state is before or (
            screen._conversations_state.reader_state.loaded_generation
            == before.loaded_generation
        )
        # The one re-check is the only read: nothing was reloaded.
        assert len(service.calls) == page_reads + 1


@pytest.mark.asyncio
async def test_recheck_superseded_by_another_selection_never_reloads_it() -> None:
    service = _ScriptedDetailService(_records())
    host = _harness(service)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, _ = await _open_alpha(pilot, host)
        service.recheck_gate = threading.Event()
        service.set_saved_count("chat-a", 3)
        try:
            await _leave_and_return(host, pilot)
            await asyncio.to_thread(service.recheck_started.wait, 10)
            screen.query_one("#library-conversation-row-1", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._conversations_state.reader_state.loaded_id == "chat-b"
                    and screen._conversations_state.reader_state.loaded_actions_eligible
                ),
                message=lambda: screen._conversations_state.reader_state,
            )
        finally:
            alpha_reads = _page_reads(service, "chat-a")
            service.recheck_gate.set()
        await screen.workers.wait_for_complete()

        # The released re-check saw Alpha's moved epoch, but Beta is open now.
        state = screen._conversations_state.reader_state
        assert (state.selected_id, state.loaded_id) == ("chat-b", "chat-b")
        assert state.loaded_actions_eligible
        assert _page_reads(service, "chat-a") == alpha_reads


@pytest.mark.asyncio
async def test_failed_recheck_keeps_the_loaded_transcript() -> None:
    service = _ScriptedDetailService(_records())
    host = _harness(service)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, _ = await _open_alpha(pilot, host)
        before = screen._conversations_state.reader_state
        service.recheck_error = RuntimeError("storage busy")

        await _leave_and_return(host, pilot)
        await _wait_for_condition(
            pilot, lambda: len(_rechecks(service)) == 1, message=lambda: service.calls
        )
        await screen.workers.wait_for_complete()

        after = screen._conversations_state.reader_state
        assert after.loaded_generation == before.loaded_generation
        assert after.loaded_actions_eligible and after.error is None
        assert [
            worker
            for worker in screen.workers
            if worker.group == RECHECK_WORKER_GROUP and worker.error is not None
        ] == []
