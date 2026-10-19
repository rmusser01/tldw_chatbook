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


class _Rows(App):
    def compose(self) -> ComposeResult:
        with Vertical(id="rows"):
            yield Static("row", id="row")


@pytest.mark.asyncio
async def test_finish_removal_outlives_the_cancelled_caller(monkeypatch) -> None:
    app = _Rows()
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
        # Not pilot.pause(): it waits on every mounted pump, the held row's too.
        await asyncio.sleep(0.05)
        release.set()
        # Polled with sleep, not pilot.pause(): a dead row's pump never answers.
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
