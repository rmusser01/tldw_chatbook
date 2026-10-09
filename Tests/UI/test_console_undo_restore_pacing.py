"""Undo of a large Console Delete restores its window a screenful at a time.

TASK-33628.5.1 AC#3. Measured on dev 46c3959526 (ConsoleHarness 160x48,
file-backed database, gc frozen, every asyncio handle timed, a 1 ms stack
sampler): the worst event-loop block of an Undo from the first message was
196-316 ms with 3,000 messages and 211-229 ms with 60, so it did not depend
on the chat's length. It was one screen update: the Undo mounted its whole
restored window (64 rows, about 520 widgets, at this size) in one batch, and
Textual laid that batch out three times -- its first layout, the relayout
every new widget's virtual size triggers, and the relayout after the
transcript's scrollbar came back -- at 68-175 ms a pass, then painted the
screen. The Undo's own sync step (100-133 ms, a third of it building those
rows) could land in the same loop iteration. With the window cut to two
viewports in the probe, the worst block was 127 ms; a relayout with nothing
new costs 6-7 ms.

What these tests pin is that work, not wall-clock time: every transcript
mount during select, Delete and Undo adds about a screenful of rows, each
later batch of a window mounts only once the layout of the rows before it
has settled, and the window still ends up whole. The timings above come
from the scratch probe recorded in the task notes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import pytest
from textual.widgets import Button

from Tests.UI.test_console_long_chat_bounds import (
    _chain,
    _Heartbeat,
    _open,
    _seed,
    _settled_rows,
    _until,
    _until_painted,
    _window_rows,
)
from Tests.UI.test_console_native_chat_flow import _wait_for_selector
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.message_delete import (
    handle_console_delete_action,
)
from tldw_chatbook.Widgets.Console import ConsoleTranscript

# The real ChatScreen/store goes through config-participant admission, which
# the per-test sandbox refuses (RecoveryRequired); keep the collection-time
# profile.
pytestmark = pytest.mark.bootstrap_profile

_SIZE = (160, 48)


def _batch_rows(transcript: ConsoleTranscript) -> int:
    """The most messages one mount may add: about a screenful.

    One viewport of the transcript's own estimated lines over the smallest
    per-message estimate, plus two for the turn a batch is aligned to: 13 of
    these short messages at 160x48. Dev mounted the whole window, 64 rows,
    in one batch.
    """
    lines = transcript._window_viewport_height()
    smallest = min(map(transcript._estimated_message_lines, transcript._messages))
    return -(-lines // smallest) + 2


def _one_window(transcript: ConsoleTranscript) -> int:
    """Nine tenths of the rows one load-shaped window holds.

    The window's line budget follows the viewport's height when it is read,
    which moves by a line or two as the selection's action row comes and
    goes (64 to 66 rows here). A window that stopped after its first batch or
    two holds 22 or 44.
    """
    budget = transcript._initial_window_line_budget()
    largest = max(map(transcript._estimated_message_lines, transcript._messages))
    return budget // largest * 9 // 10


@contextmanager
def _mounts(transcript: ConsoleTranscript) -> Iterator[list[dict[str, Any]]]:
    """Record each mount on the transcript: its messages, and what had a layout.

    ``laid_out`` is False when a row mounted before this batch had no size
    yet, or a relayout was still pending for the transcript or its screen
    (Textual 8's ``_layout_required``): this batch's first layout would then
    land in the same pass as the rows before it.
    """
    batches: list[dict[str, Any]] = []
    real = transcript.mount

    def mount(*widgets: Any, **kwargs: Any) -> Any:
        rows = set(map(id, widgets))
        known = {message.id for message in transcript._messages}
        # Row keys are "<kind>:<message id>[:<n>]"; the empty state has none.
        message_ids = {
            key.split(":", 2)[1]
            for key, widget in transcript._row_widgets.items()
            if id(widget) in rows and ":" in key
        } & known
        unsized = [
            key
            for key, widget in transcript._row_widgets.items()
            if id(widget) not in rows
            and widget.parent is transcript
            and widget.display
            and not widget.size.height
        ]
        pending = [
            type(node).__name__
            for node in transcript.walk_children(with_self=True)
            if node._layout_required
        ]
        if transcript.screen._layout_required:
            pending.append("screen")
        batches.append(
            {
                "messages": message_ids,
                "laid_out": not unsized and not pending,
                "unsized": unsized[:4] + pending[:4],
            }
        )
        return real(*widgets, **kwargs)

    transcript.mount = mount
    try:
        yield batches
    finally:
        del transcript.mount


@pytest.mark.parametrize("count", [60, 3_000])
@pytest.mark.asyncio
async def test_select_delete_and_undo_from_the_first_message_mount_a_screenful_at_a_time(
    tmp_path, count
):
    """5.1 AC#3: no select, Delete or Undo mount lays out a whole window at once."""
    db = CharactersRAGDB(tmp_path / "long-chat.db", "long-chat")
    rows = _chain(count)
    conversation_id = _seed(db, rows)
    app = _build_test_app()
    app.chachanotes_db = db
    notices: list[str] = []
    app.notify = lambda message, **kwargs: notices.append(str(message))
    host = ConsoleHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console, store, native, transcript = await _open(
            host, pilot, db, conversation_id, rows[-1][0]
        )
        first, last = native[rows[0][0]], native[rows[-1][0]]
        batch_rows = _batch_rows(transcript)
        whole = min(count, _one_window(transcript))
        # The bound must sit well inside the window, or it pins nothing.
        assert batch_rows < whole // 2, (batch_rows, whole)
        heartbeat = _Heartbeat()
        heartbeat.start()
        try:
            await _settled_rows(transcript, heartbeat)
            with _mounts(transcript) as batches:
                transcript.select_message(first)
                await _until(
                    lambda: first in transcript.mounted_message_content_ids(),
                    "the selected first message to mount",
                )
                await _settled_rows(transcript, heartbeat)

                await handle_console_delete_action(console._message, "delete", first)
                confirm = f"#console-message-action-delete-confirm-{first}"
                await _wait_for_selector(console, pilot, confirm, timeout=30.0)
                console.query_one(confirm, Button).press()
                await _until(
                    lambda: bool(host.screen.query("#console-delete-receipt-undo")),
                    "Undo to be offered",
                )
                await heartbeat.quiet()
                assert not transcript.mounted_message_content_ids()
                restored_from = len(batches)

                host.screen.query_one("#console-delete-receipt-undo", Button).press()
                await _until(
                    lambda: transcript.selected_message_id == first,
                    "Undo to reselect the restored first message",
                )
                await _until(
                    lambda: not host.screen.query("#console-delete-receipt-box"),
                    "the receipt to close after Undo",
                )
                mounted = await _settled_rows(transcript, heartbeat)

            sizes = [len(batch["messages"]) for batch in batches]
            assert sizes and max(sizes) <= batch_rows, (
                f"one mount added {max(sizes)} rows "
                f"(batches {sizes}, bound {batch_rows})"
            )
            # Within the select and within the Undo, each batch of the window
            # after the first waits for the one before it. (A one-message
            # mount is the selection's action row, not part of a window.)
            early = [
                (index, len(batch["messages"]), batch["unsized"])
                for phase in (batches[:restored_from], batches[restored_from:])
                for index, batch in enumerate(
                    [batch for batch in phase if len(batch["messages"]) > 1]
                )
                if index and not batch["laid_out"]
            ]
            assert not early, (
                f"a batch mounted before the rows ahead of it had a layout: {early}"
            )
            undo_batches = batches[restored_from:]
            assert first in undo_batches[0]["messages"], (
                "Undo must mount the restored root first"
            )
            # The window still ends up whole, and the selection painted.
            assert first in mounted
            assert whole <= len(mounted) <= _window_rows(transcript)
            if count > _window_rows(transcript):
                assert last not in mounted
            else:
                assert len(mounted) == count
            assert transcript.selected_message_id == first
            assert len(store.messages_for_session(store.active_session_id)) == count
            await _until_painted(host, "m0000 text")
            assert any(f"Restored {count} messages" in n for n in notices), notices
        finally:
            await heartbeat.stop()


@pytest.mark.asyncio
async def test_undo_of_the_last_turns_keeps_following_and_mounts_a_screenful_at_a_time(
    tmp_path,
):
    """5.1 AC#3: an Undo under a reader who follows the tail is paced too.

    Deleting the last 50 of 3,000 messages leaves the reader at the bottom,
    so Textual re-attaches the tail-follow. Fewer than a window come back,
    so the Undo does not jump; on dev it mounted all 50 rows in one batch.
    The reader still ends at the newest message, following it.
    """
    db = CharactersRAGDB(tmp_path / "long-chat.db", "long-chat")
    rows = _chain(3_000)
    conversation_id = _seed(db, rows)
    app = _build_test_app()
    app.chachanotes_db = db
    app.notify = lambda *args, **kwargs: None
    host = ConsoleHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console, store, native, transcript = await _open(
            host, pilot, db, conversation_id, rows[-1][0]
        )
        root, last = native[rows[-50][0]], native[rows[-1][0]]
        batch_rows = _batch_rows(transcript)
        heartbeat = _Heartbeat()
        heartbeat.start()
        try:
            await _settled_rows(transcript, heartbeat)
            transcript.select_message(root)
            await _settled_rows(transcript, heartbeat)
            await handle_console_delete_action(console._message, "delete", root)
            confirm = f"#console-message-action-delete-confirm-{root}"
            await _wait_for_selector(console, pilot, confirm, timeout=30.0)
            console.query_one(confirm, Button).press()
            await _until(
                lambda: bool(host.screen.query("#console-delete-receipt-undo")),
                "Undo to be offered",
            )
            await heartbeat.quiet()
            # The precondition this case exists for.
            assert transcript._raw_anchor_engaged(), "the Delete left a follower"

            with _mounts(transcript) as batches:
                host.screen.query_one("#console-delete-receipt-undo", Button).press()
                await _until(
                    lambda: not host.screen.query("#console-delete-receipt-box"),
                    "the receipt to close after Undo",
                )
                mounted = await _settled_rows(transcript, heartbeat)

            sizes = [len(batch["messages"]) for batch in batches]
            assert sizes and max(sizes) <= batch_rows, (
                f"one mount added {max(sizes)} rows "
                f"(batches {sizes}, bound {batch_rows})"
            )
            windows = [batch for batch in batches if len(batch["messages"]) > 1]
            early = [
                (index, batch["unsized"])
                for index, batch in enumerate(windows)
                if index and not batch["laid_out"]
            ]
            assert not early, early
            assert root in mounted and last in mounted
            assert transcript.selected_message_id == root
            assert transcript._raw_anchor_engaged(), "the reader stopped following"
            await _until_painted(host, f"{rows[-1][0]} text")
        finally:
            await heartbeat.stop()
