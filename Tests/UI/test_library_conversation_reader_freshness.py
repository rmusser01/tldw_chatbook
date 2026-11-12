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
from loguru import logger
from textual.screen import Screen
from textual.widgets import Button, Input

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
from tldw_chatbook.Widgets.Library.library_conversation_reader import (
    LibraryConversationReader,
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
    ``page_gate``, when set, holds every other read the same way, but only
    after the response is built: a released page carries the transcript as
    saved when it was read, as a slow storage read does.
    """

    def __init__(self, conversations: list[dict[str, Any]]) -> None:
        super().__init__(conversations)
        self.calls: list[dict[str, Any]] = []
        self.epochs: dict[str, str] = {}
        self.recheck_gate: threading.Event | None = None
        self.recheck_started = threading.Event()
        self.recheck_error: Exception | None = None
        self.page_gate: threading.Event | None = None
        self.page_started = threading.Event()

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
        if not _is_recheck(kwargs) and self.page_gate is not None:
            self.page_started.set()
            assert self.page_gate.wait(timeout=10)
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


def _harness(
    service: _ScriptedDetailService | None = None,
    records: list[dict[str, Any]] | None = None,
) -> LibraryHarness:
    records = records or _records()
    app = _build_test_app()
    _seed_conversations(app, records)
    app.local_chat_conversation_service = service or _ScriptedDetailService(records)
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
    """A saved count that moved while Library was covered reloads the reader.

    One re-check read finds the new epoch and total; the existing reader
    pipeline then reloads the same conversation in place and paints all
    three messages, without changing Read/Info.
    """
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
async def test_reload_keeps_info_mode_and_the_find_query() -> None:
    """The User Guide promises Read/Info and the Find text survive a reload.

    The query only matches the message the reload adds, so the kept query is
    also proven to have been re-run against the reloaded transcript.
    """
    host = _harness()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, service = await _open_alpha(pilot, host)
        reader = screen.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        find = reader.query_one("#library-conversation-reader-find", Input)
        find.value = "message 3"
        find.focus()
        await pilot.press("enter")
        reader.query_one("#library-conversation-reader-info", Button).press()
        await pilot.pause()
        before = screen._conversations_state.reader_state
        assert (before.mode, before.find_query) == ("info", "message 3")
        assert before.find_complete and before.find_matches == ()
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
        assert after.loaded_generation != before.loaded_generation
        assert (after.mode, after.find_query) == ("info", "message 3")
        assert [match.message_id for match in after.find_matches] == [
            "chat-a-message-3"
        ]
        await pilot.pause()
        assert reader.query_one("#library-conversation-reader-info-body").display
        assert find.value == "message 3"


@pytest.mark.asyncio
async def test_row_press_load_is_not_rechecked_in_the_same_list_read() -> None:
    """A row press loads the transcript fresh; a later ensure must not re-read it.

    The row press goes straight to the reader pipeline, and compose, the
    source-snapshot reconcile, inspection admission and leaving select mode
    can all call ``_ensure_library_conversation_reader_selection`` again
    within the same list read.
    """
    host = _harness()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, service = await _open_alpha(pilot, host)
        await screen.workers.wait_for_complete()
        screen.query_one("#library-conversation-row-1", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._conversations_state.reader_state.loaded_id == "chat-b"
                and screen._conversations_state.reader_state.loaded_actions_eligible
            ),
            message=lambda: screen._conversations_state.reader_state,
        )
        await screen.workers.wait_for_complete()

        screen._ensure_library_conversation_reader_selection()
        await pilot.pause()
        await screen.workers.wait_for_complete()

        assert _rechecks(service) == []
        assert screen._conversations_state.reader_state.loaded_id == "chat-b"


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
    """An unchanged saved transcript costs one re-check read and nothing more.

    The reader keeps the generation it loaded, and the re-check is the only
    detail read the return visit makes.
    """
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
@pytest.mark.parametrize(
    "beta_versioned", [True, False], ids=["versioned", "bootstrap"]
)
async def test_load_pending_across_a_list_read_is_rechecked_once_it_settles(
    monkeypatch: pytest.MonkeyPatch, beta_versioned: bool
) -> None:
    """A load in flight when the list re-reads may settle a pre-write page.

    Beta's page is read before the write but lands after the return visit's
    list read, which found the reader busy and could not re-check it then.
    The re-check runs once that load settles and reloads the moved transcript.
    A list record without a version loads through the version bootstrap,
    which selects once more before it settles the same pre-write page.
    """
    records = _records()
    if not beta_versioned:
        del records[1]["version"]
    service = _ScriptedDetailService(records)
    host = _harness(service, records)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen, _ = await _open_alpha(pilot, host)
        await screen.workers.wait_for_complete()
        controller = screen._conversation_reader_controller
        ensure = controller._ensure_library_conversation_reader_selection
        ensured_after_apply: list[int] = []

        def _recording_ensure() -> None:
            ensure()
            if not screen._conversations_state.loading:
                ensured_after_apply.append(
                    screen._conversations_state.request_generation
                )

        monkeypatch.setattr(
            controller,
            "_ensure_library_conversation_reader_selection",
            _recording_ensure,
        )
        service.page_gate = threading.Event()
        try:
            screen.query_one("#library-conversation-row-1", Button).press()
            await asyncio.to_thread(service.page_started.wait, 10)
            service.set_saved_count("chat-b", 2)
            list_read = screen._conversations_state.request_generation

            await _leave_and_return(host, pilot)
            # The return visit's list read has applied and asked the reader.
            await _wait_for_condition(
                pilot,
                lambda: any(read > list_read for read in ensured_after_apply),
                message=lambda: (ensured_after_apply, screen._conversations_state),
            )
            pending = screen._conversations_state.reader_state
            assert (pending.selected_id, pending.loading) == ("chat-b", True)
        finally:
            service.page_gate.set()
            service.page_gate = None

        await _wait_for_condition(
            pilot,
            lambda: (
                screen._conversations_state.reader_state.message_total == 2
                and screen._conversations_state.reader_state.loaded_actions_eligible
            ),
            message=lambda: screen._conversations_state.reader_state,
        )
        assert screen._conversations_state.reader_state.loaded_id == "chat-b"
        assert len(_rechecks(service)) == 1


@pytest.mark.asyncio
async def test_failed_recheck_keeps_the_loaded_transcript() -> None:
    """A failed re-check read keeps the transcript and logs only its type.

    The warning names the exception type and nothing the storage layer put
    in the exception's text, which can quote saved message content; no
    traceback is attached, so no frame locals reach the log either.
    """
    secret = "Saved message 1 quoted by storage"
    service = _ScriptedDetailService(_records())
    host = _harness(service)
    records: list[tuple[str, Any]] = []
    sink = logger.add(
        lambda message: records.append((str(message), message.record)),
        level="WARNING",
        filter=lambda record: record["name"].endswith(
            "library_conversation_reader_freshness"
        ),
    )
    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen, _ = await _open_alpha(pilot, host)
            before = screen._conversations_state.reader_state
            service.recheck_error = RuntimeError(secret)

            await _leave_and_return(host, pilot)
            await _wait_for_condition(
                pilot,
                lambda: len(_rechecks(service)) == 1,
                message=lambda: service.calls,
            )
            await screen.workers.wait_for_complete()
            after = screen._conversations_state.reader_state
            failed_workers = [
                worker
                for worker in screen.workers
                if worker.group == RECHECK_WORKER_GROUP and worker.error is not None
            ]
    finally:
        logger.remove(sink)

    assert after.loaded_generation == before.loaded_generation
    assert after.loaded_actions_eligible and after.error is None
    assert failed_workers == []
    assert len(records) == 1
    rendered, record = records[0]
    assert "exception_type=RuntimeError" in record["message"]
    assert secret not in rendered
    assert record["exception"] is None
