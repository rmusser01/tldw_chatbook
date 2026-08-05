"""TASK-1366: expandable inline diff rows for file-write TOOL markers.

A TOOL marker carrying ``tool_diff`` (raw before/after contents, captured
live at the provider's strip seam) renders a ``DiffView`` row directly
under its message row when expanded via the existing full-output toggle.
The diff is prepared off the UI thread before mounting (AC2), non-diff
markers render exactly as before (AC4), and the row is part of its
message's group so view-window pruning drops it with the message.
"""

import asyncio
import time
from collections.abc import Callable

import pytest
from textual.app import App, ComposeResult
from textual_diff_view import DiffView

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console import console_transcript as transcript_module
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleToolDiffRow,
    ConsoleTranscript,
    ConsoleTranscriptMessage,
)


async def wait_for_condition(
    predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.02
) -> bool:
    """Poll ``predicate`` until true or ``timeout`` seconds elapse."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(interval)


class DiffHarness(App):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _tool_message(**overrides) -> ConsoleChatMessage:
    kwargs = {
        "role": ConsoleMessageRole.TOOL,
        "content": "write_file → /tmp/a.py",
        "id": "m1",
        "tool_output_full": "action: overwritten\nlines_written: 1",
        "tool_diff": ("/tmp/a.py", "def f():\n    return 1\n", "def f():\n    return 2\n"),
    }
    kwargs.update(overrides)
    return ConsoleChatMessage(**kwargs)


@pytest.mark.asyncio
async def test_expanded_write_marker_mounts_prepared_diff(monkeypatch):
    """AC1+AC2: expanding a diff-carrying marker mounts a DiffView whose
    diff was computed off the UI thread (prepare ran before mount)."""
    prepared = []
    real_make_diff = transcript_module.make_diff

    def tracking_make_diff(path, old, new, **kwargs):
        diff_view = real_make_diff(path, old, new, **kwargs)
        original_prepare = diff_view.prepare

        async def tracked_prepare():
            prepared.append(path)
            await original_prepare()

        diff_view.prepare = tracked_prepare
        return diff_view

    monkeypatch.setattr(transcript_module, "make_diff", tracking_make_diff)

    app = DiffHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([_tool_message()])
        await transcript.refresh_messages()
        await pilot.pause()

        # Collapsed: no diff row.
        assert not transcript.query(ConsoleToolDiffRow)

        transcript.toggle_tool_output("m1")
        found = await wait_for_condition(
            lambda: bool(transcript.query(ConsoleToolDiffRow))
            and bool(transcript.query(DiffView))
        )
        assert found, "DiffView was not mounted within the timeout"
        await pilot.pause()

        # prepare() ran before the DiffView mounted (off-thread prepare).
        assert prepared == ["/tmp/a.py"]
        diff_view = transcript.query(DiffView).first()
        assert diff_view.is_mounted
        assert diff_view.path_modified == "/tmp/a.py"
        assert diff_view.code_original == "def f():\n    return 1\n"
        assert diff_view.code_modified == "def f():\n    return 2\n"

        # The diff row sits directly after its message row, inside the
        # same message group.
        children = list(transcript.children)
        message_index = next(
            i for i, w in enumerate(children) if isinstance(w, ConsoleTranscriptMessage)
        )
        diff_index = next(
            i for i, w in enumerate(children) if isinstance(w, ConsoleToolDiffRow)
        )
        assert diff_index == message_index + 1

        # Collapsing removes the diff row again.
        transcript.toggle_tool_output("m1")
        removed = await wait_for_condition(
            lambda: not transcript.query(ConsoleToolDiffRow)
        )
        assert removed, "diff row was not removed on collapse"


@pytest.mark.asyncio
async def test_non_diff_tool_message_expands_as_text_only():
    """AC4: a marker without tool_diff expands to plain text, no diff row."""
    app = DiffHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([_tool_message(tool_diff=None)])
        await transcript.refresh_messages()
        await pilot.pause()

        transcript.toggle_tool_output("m1")
        await pilot.pause()
        await pilot.pause()

        # Bounded negative wait: no diff row should ever appear.
        appeared = await wait_for_condition(
            lambda: bool(transcript.query(ConsoleToolDiffRow)), timeout=0.5
        )
        assert not appeared
        # The text expansion still happens (pre-diff behavior, unchanged).
        rows = transcript._transcript_rows()
        message_row = next(row for row in rows if row.kind == "message")
        assert "lines_written" in str(message_row.signature)


@pytest.mark.asyncio
async def test_pruned_message_drops_its_diff_row():
    """Pruning invariant: the diff row is inside its message's group, so
    the view window drops it together with the message (never orphaned)."""
    app = DiffHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([_tool_message()])
        await transcript.refresh_messages()
        await pilot.pause()

        transcript.toggle_tool_output("m1")
        await pilot.pause()
        keys = {row.key for row in transcript._transcript_rows()}
        assert "diff:m1" in keys

        transcript._pruned_message_ids.add("m1")
        keys = {row.key for row in transcript._transcript_rows()}
        assert "message:m1" not in keys
        assert "diff:m1" not in keys
