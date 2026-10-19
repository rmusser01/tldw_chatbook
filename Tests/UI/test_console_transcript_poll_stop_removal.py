"""TASK-34982: stopping a paint's timer mid-removal must not kill the removed rows.

The Console transcript poll paints inside its timer's own task, and Textual
cancels that task whenever the timer stops (the poll stopping itself, or the
screen closing). Textual's ``AwaitRemove`` gathers the removed widgets' own
message-pump tasks, so that cancel used to reach them mid-exit: the rows stayed
in the DOM with cancelled tasks, and the next prune of their parent (app
shutdown included) raised a bare ``CancelledError``.
"""

import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.UI.Navigation.surface_swap_guard import finish_removal
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


def _msg(i: int) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content=f"message {i}",
        id=f"m{i}",
        status="completed",
    )


def _slow_unmount(
    monkeypatch, row, entered: asyncio.Event, release: asyncio.Event
) -> None:
    """Hold ``row``'s Unmount until ``release``, so its removal is in flight."""

    async def on_unmount(self) -> None:
        if self is row:
            entered.set()
            await release.wait()

    monkeypatch.setattr(type(row), "on_unmount", on_unmount, raising=False)


class _HeldRow(Static):
    """A row of its own class, so patching its Unmount touches no other widget."""


class _SelfRemovingRow(Static):
    async def remove_self(self) -> None:
        await finish_removal(self.remove())


class _Rows(App):
    def __init__(self, row: Static) -> None:
        super().__init__()
        self._row = row

    def compose(self) -> ComposeResult:
        with Vertical(id="rows"):
            yield self._row


@pytest.mark.asyncio
async def test_finish_removal_outlives_the_cancelled_caller(monkeypatch) -> None:
    app = _Rows(_HeldRow("row", id="row"))
    entered, release = asyncio.Event(), asyncio.Event()
    async with app.run_test():
        rows, row = app.query_one("#rows"), app.query_one("#row")
        _slow_unmount(monkeypatch, row, entered, release)
        caller = asyncio.ensure_future(finish_removal(rows.remove_children()))
        await asyncio.wait_for(entered.wait(), 5)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        release.set()
        for _ in range(100):
            if row not in rows.children:
                break
            await asyncio.sleep(0.02)
        assert row not in rows.children, "the removal stopped with its caller"


@pytest.mark.asyncio
async def test_finish_removal_from_the_removed_widgets_own_pump_completes() -> None:
    # AwaitRemove leaves the awaiting pump out of what it waits on; a shield's
    # new task would wait on that pump, which is waiting on the shield.
    app = _Rows(_SelfRemovingRow("row", id="row"))
    async with app.run_test():
        rows, row = app.query_one("#rows"), app.query_one("#row")
        row.call_later(row.remove_self)
        for _ in range(100):
            if row not in rows.children:
                break
            await asyncio.sleep(0.02)
        assert row not in rows.children, (
            "a widget awaiting its own removal never finished"
        )


@pytest.mark.asyncio
async def test_a_removal_that_fails_after_its_caller_is_cancelled_is_logged() -> None:
    from loguru import logger

    seen: list[str] = []
    sink = logger.add(lambda line: seen.append(str(line)), level="WARNING")
    release = asyncio.Event()

    async def failing_removal() -> None:
        await release.wait()
        raise RuntimeError("prune failed")

    try:
        caller = asyncio.ensure_future(finish_removal(failing_removal()))
        await asyncio.sleep(0)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        release.set()
        for _ in range(50):
            if seen:
                break
            await asyncio.sleep(0.01)
    finally:
        logger.remove(sink)
    assert any("exception_type=RuntimeError" in line for line in seen), seen


class _TranscriptHost(App):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


@pytest.mark.asyncio
async def test_stopping_the_paint_timer_mid_removal_still_removes_the_row(
    monkeypatch,
) -> None:
    app = _TranscriptHost()
    entered, release = asyncio.Event(), asyncio.Event()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([_msg(0), _msg(1)])
        await transcript.refresh_messages()
        await pilot.pause()
        doomed = next(
            row
            for row in transcript.query(".console-transcript-message")
            if row.message_id == "m0"
        )
        _slow_unmount(monkeypatch, doomed, entered, release)

        painted = []

        async def paint() -> None:
            # Like the Console poll's sync: inside the timer's own task, push
            # the messages and await the refresh before the queued one runs.
            if painted:
                return
            painted.append(True)
            transcript.set_messages([_msg(1)])
            await transcript.refresh_messages()

        # set_interval, as the poll uses: it runs the callback in the timer's
        # own task (set_timer hands it to the target's pump via call_next).
        timer = app.set_interval(0.01, paint)
        await asyncio.wait_for(entered.wait(), 5)
        timer.stop()
        await asyncio.sleep(0.05)
        release.set()
        # Polled with sleep, not pilot.pause(): Pilot waits on every mounted
        # pump, and the held (or, unfixed, dead) row's pump does not answer.
        for _ in range(100):
            if doomed not in transcript.children:
                break
            await asyncio.sleep(0.02)
        assert doomed not in transcript.children, (
            "the stopped paint left its removed row mounted"
        )
        assert doomed._task is None or not doomed._task.cancelled()
        # The next paint (the poll restarts when the user returns) finds a
        # consistent transcript: exactly the current rows, none resurrected.
        await transcript.refresh_messages()
        assert [
            row.message_id for row in transcript.query(".console-transcript-message")
        ] == ["m1"]
