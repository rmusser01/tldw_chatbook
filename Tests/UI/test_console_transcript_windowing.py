"""TASK-15455: tail-first Console transcript mounting and lazy scrollback.

The store-facing message list remains complete.  Only a contiguous tail window
is mounted initially; reaching its upper boundary prepends an earlier chunk and
keeps the same message under the reader.  Reconciliation uses Textual's batch
DOM APIs so a long resume does not pay one awaited mount/remove per row.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


class WindowHarness(App):
    """Compact transcript host with deterministic pruning and Markdown settings."""

    CSS = "ConsoleTranscript { height: 24; width: 100; }"

    def __init__(self, *, low: int = 12_000, high: int = 0) -> None:
        super().__init__()
        self.app_config = {
            "chat_defaults": {
                "assistant_markdown": False,
                "prune_low_watermark": low,
                "prune_high_watermark": high,
            }
        }

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _messages(count: int, *, prefix: str = "m") -> list[ConsoleChatMessage]:
    return [
        ConsoleChatMessage(
            id=f"{prefix}{index}",
            role=(
                ConsoleMessageRole.USER
                if index % 2 == 0
                else ConsoleMessageRole.ASSISTANT
            ),
            content="\n".join(f"message {index} line {line}" for line in range(4)),
        )
        for index in range(count)
    ]


def _mounted_message_ids(transcript: ConsoleTranscript) -> list[str]:
    return [
        row.message_id for row in transcript.query(".console-transcript-message")
    ]


async def _wait_for(pilot, predicate, *, attempts: int = 80) -> bool:
    for _ in range(attempts):
        await pilot.pause()
        if predicate():
            return True
    return False


def _top_visible_message_id(transcript: ConsoleTranscript) -> str | None:
    viewport_top = transcript.content_region.y
    viewport_bottom = viewport_top + transcript.content_region.height
    for row in transcript.query(".console-transcript-message"):
        if row.region.y + row.region.height > viewport_top and row.region.y < viewport_bottom:
            return row.message_id
    return None


@pytest.mark.asyncio
async def test_long_resume_mounts_only_a_contiguous_tail_window() -> None:
    app = WindowHarness()
    history = _messages(500)

    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await pilot.pause()

        mounted = _mounted_message_ids(transcript)

        assert 0 < len(mounted) < len(history) // 2
        assert mounted == [message.id for message in history[-len(mounted) :]]
        assert transcript._pruned_message_ids == {
            message.id for message in history[: -len(mounted)]
        }
        assert transcript._messages == history


@pytest.mark.asyncio
async def test_long_session_swap_batches_row_mounts_and_removals(monkeypatch) -> None:
    app = WindowHarness()

    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(180, prefix="old-"))
        await transcript.refresh_messages()
        await pilot.pause()

        mount_batch_sizes: list[int] = []
        remove_batch_sizes: list[int] = []
        original_mount = transcript.mount
        original_remove_children = transcript.remove_children

        def recording_mount(*widgets, **kwargs):
            mount_batch_sizes.append(len(widgets))
            return original_mount(*widgets, **kwargs)

        def recording_remove_children(widgets="*"):
            materialized = list(widgets) if not isinstance(widgets, str) else widgets
            if isinstance(materialized, list):
                remove_batch_sizes.append(len(materialized))
            return original_remove_children(materialized)

        monkeypatch.setattr(transcript, "mount", recording_mount)
        monkeypatch.setattr(transcript, "remove_children", recording_remove_children)

        transcript.set_messages(_messages(180, prefix="new-"))
        await transcript.refresh_messages()
        await pilot.pause()

        assert mount_batch_sizes and max(mount_batch_sizes) > 1
        assert remove_batch_sizes and max(remove_batch_sizes) > 1
        assert len(mount_batch_sizes) <= 2
        assert len(remove_batch_sizes) == 1


@pytest.mark.asyncio
async def test_scroll_boundary_hydrates_earlier_rows_and_preserves_reader_state() -> None:
    app = WindowHarness()
    history = _messages(180)

    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await pilot.pause()

        hidden_before = len(transcript._pruned_message_ids)
        mounted_before = _mounted_message_ids(transcript)
        top_before = mounted_before[0]
        selected = mounted_before[2]
        transcript.select_message(selected)
        await transcript.refresh_messages()
        transcript.release_anchor()
        transcript.scroll_to(y=0, animate=False)
        await pilot.pause()

        assert await _wait_for(
            pilot,
            lambda: len(transcript._pruned_message_ids) < hidden_before,
        ), "reaching the scrollback boundary did not hydrate an earlier chunk"
        assert await _wait_for(
            pilot,
            lambda: _top_visible_message_id(transcript) == top_before,
        ), "prepending scrollback moved the reader to different content"
        assert transcript.selected_message_id == selected
        assert not transcript._is_following_tail()
        assert _mounted_message_ids(transcript)[-1] == history[-1].id

        # A sibling swipe replaces the selected native id at the same tree
        # position.  The pending handoff must stay mounted even when that row is
        # the first row of the current lazy window.
        visible_ids = _mounted_message_ids(transcript)
        boundary_id = visible_ids[0]
        boundary_index = next(
            index for index, message in enumerate(history) if message.id == boundary_id
        )
        sibling = replace(history[boundary_index], id="branch-sibling")
        branched = list(history)
        branched[boundary_index] = sibling
        transcript.pending_selection_id = sibling.id
        transcript.set_messages(branched)
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.selected_message_id == sibling.id
        assert sibling.id in _mounted_message_ids(transcript)


@pytest.mark.asyncio
async def test_hydration_still_obeys_mounted_height_watermarks() -> None:
    app = WindowHarness(low=45, high=0)
    history = _messages(180)

    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await pilot.pause()
        hidden_before_hydration = len(transcript._pruned_message_ids)
        transcript.release_anchor()
        await transcript._hydrate_scrollback()
        assert await _wait_for(
            pilot,
            lambda: len(transcript._pruned_message_ids) < hidden_before_hydration,
        ), "the watermark check never observed a genuinely hydrated chunk"
        hidden_after_hydration = len(transcript._pruned_message_ids)

        app.app_config["chat_defaults"]["prune_high_watermark"] = 70
        await transcript.refresh_messages()
        assert await _wait_for(
            pilot, lambda: transcript.virtual_size.height <= 70
        ), "hydrated rows escaped the configured height watermark"

        assert transcript._messages == history
        assert len(transcript._pruned_message_ids) > hidden_after_hydration
        assert len(_mounted_message_ids(transcript)) < len(history)
