"""TASK-15455: tail-first windowed mount + scrollback hydration.

Session resume used to mount EVERY persisted row (one awaited ``mount()`` per
row, one Textual Markdown widget per assistant message) before the height
watermarks could trim anything. The transcript now mounts only the newest
window on a fresh load and hydrates older messages when the reader scrolls
back.

The first half of this module is the PIN section: invariants that existed
before windowing and must survive it, exercised on a history large enough
(60+ messages) that the window path is actually in play — the pre-existing
suites top out at 30 messages, so none of them would have noticed.
"""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleVariantSet,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


class WindowHarness(App):
    """Small viewport so a handful of messages overflow it."""

    CSS = "ConsoleTranscript { height: 10; }"

    def __init__(
        self,
        *,
        low: int = 12000,
        high: int = 20000,
        window_messages: int | None = None,
        window_lines: int | None = None,
        hydrate_messages: int | None = None,
    ) -> None:
        super().__init__()
        chat_defaults: dict[str, object] = {
            "prune_high_watermark": high,
            "prune_low_watermark": low,
        }
        if window_messages is not None:
            chat_defaults["transcript_window_messages"] = window_messages
        if window_lines is not None:
            chat_defaults["transcript_window_lines"] = window_lines
        if hydrate_messages is not None:
            chat_defaults["transcript_hydrate_messages"] = hydrate_messages
        self.app_config = {"chat_defaults": chat_defaults}

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _msg(
    i: int | str,
    role: ConsoleMessageRole = ConsoleMessageRole.ASSISTANT,
    *,
    status: str = "completed",
    lines: int = 4,
) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=role,
        content="\n".join(f"line {i}.{j}" for j in range(lines)),
        id=f"m{i}",
        status=status,
    )


def _messages(n: int, *, lines: int = 4) -> list[ConsoleChatMessage]:
    return [
        _msg(
            i,
            ConsoleMessageRole.USER if i % 2 == 0 else ConsoleMessageRole.ASSISTANT,
            lines=lines,
        )
        for i in range(n)
    ]


def _mounted_message_ids(transcript: ConsoleTranscript) -> list[str]:
    # Class query, not a type query: assistant rows may render as
    # ConsoleMarkdownMessage (TASK-1990) while other roles stay plain.
    return [
        widget.message_id for widget in transcript.query(".console-transcript-message")
    ]


async def _wait_for(pilot, predicate, *, attempts: int = 50) -> bool:
    for _ in range(attempts):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _settle(pilot, *, times: int = 8) -> None:
    for _ in range(times):
        await pilot.pause()


# ---------------------------------------------------------------------------
# PIN section — behavior that predates windowing and must survive it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pin_large_history_load_leaves_reader_at_the_tail():
    """A 60-message load lands anchored at the newest content."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        assert transcript._is_following_tail(), "load must keep tail-follow engaged"
        assert transcript.max_scroll_y > 0, "fixture must overflow the viewport"
        assert transcript.scroll_y == transcript.max_scroll_y
        mounted = _mounted_message_ids(transcript)
        assert mounted, "the newest rows must be mounted"
        assert mounted[-1] == "m59", "the newest message must be the last row"


@pytest.mark.asyncio
async def test_pin_selection_and_action_row_on_a_mounted_row():
    """Clicking/selecting a mounted row shows its contextual action row."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        target = _mounted_message_ids(transcript)[-2]
        transcript.select_message(target)
        assert await _wait_for(
            pilot,
            lambda: bool(transcript.query(".console-transcript-action-row")),
        ), "selecting a mounted row must mount its action row"
        assert transcript.selected_message_id == target
        assert target in _mounted_message_ids(transcript)


@pytest.mark.asyncio
async def test_pin_prune_watermarks_bound_mounted_height():
    """The height watermarks still bound the mounted transcript."""
    app = WindowHarness(low=25, high=40)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        assert transcript.virtual_size.height <= 40, (
            "mounted height must settle at or under the high watermark"
        )
        assert transcript._is_following_tail()
        assert len(transcript._messages) == 60, "the store snapshot is untouched"


@pytest.mark.asyncio
async def test_pin_branch_swap_keeps_shared_prefix_rows():
    """A branch/variant swap replaces the suffix without rebuilding the prefix."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = _messages(60)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        kept_key = f"message:{history[-3].id}"
        builds_before = transcript.row_build_counts().get(kept_key)
        assert builds_before is not None, "the pinned row must be mounted"

        swapped = list(history[:-1]) + [
            _msg("branch-b", ConsoleMessageRole.ASSISTANT)
        ]
        transcript.set_messages(swapped)
        await transcript.refresh_messages()
        await _settle(pilot)

        mounted = _mounted_message_ids(transcript)
        assert mounted[-1] == "mbranch-b", "the swapped-in branch tail must mount"
        assert history[-1].id not in mounted, "the replaced tail must be gone"
        assert transcript.row_build_counts().get(kept_key) == builds_before, (
            "a branch swap must not rebuild the shared prefix rows"
        )


@pytest.mark.asyncio
async def test_pin_variant_navigation_on_a_mounted_row():
    """`>`/`<` variant navigation still swaps the rendered variant."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = _messages(60)
        history[-1] = ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="variant one",
            id="mvariants",
            variants=ConsoleVariantSet.from_contents(
                turn_id="turn-variants",
                contents=["variant one", "variant two"],
                selected_index=0,
            ),
        )
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        transcript.select_next_variant("mvariants")
        await _settle(pilot)
        assert "variant two" in transcript.to_plain_text(width=60)


@pytest.mark.asyncio
async def test_pin_session_switch_replaces_every_row():
    """Switching sessions drops the old rows and renders the new tail."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        other = [
            _msg(f"other{i}", ConsoleMessageRole.ASSISTANT) for i in range(60)
        ]
        transcript.set_messages(other)
        await transcript.refresh_messages()
        await _settle(pilot)

        mounted = _mounted_message_ids(transcript)
        assert mounted, "the new session must render rows"
        assert all(message_id.startswith("mother") for message_id in mounted), (
            "no row from the previous session may survive a switch"
        )
        assert mounted[-1] == "mother59"
        assert transcript._is_following_tail()


@pytest.mark.asyncio
async def test_pin_jump_pill_and_jump_to_latest_after_scrolling_up():
    """Detaching shows the pill; jump_to_latest re-anchors and hides it."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        transcript.release_anchor()
        transcript.scroll_to(y=0, animate=False)
        await _settle(pilot)
        transcript.sync_jump_indicator("streaming")
        pill = transcript.query_one("#console-transcript-jump-pill")
        assert pill.display is True, "a detached reader mid-run sees the pill"

        transcript.jump_to_latest()
        await _settle(pilot)
        assert pill.display is False
        assert transcript._is_following_tail()
        assert transcript.scroll_y == transcript.max_scroll_y


@pytest.mark.asyncio
async def test_pin_exports_render_full_history_regardless_of_mounting():
    """`to_plain_text` renders every message, mounted or not."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        plain = transcript.to_plain_text(width=40)
        assert "line 0.0" in plain
        assert "line 59.3" in plain


# ---------------------------------------------------------------------------
# Batched DOM churn (reconciler).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_rows_mount_in_one_batched_call():
    """Contiguous new rows are mounted with a single `mount()` call."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        calls: list[int] = []
        original_mount = transcript.mount

        def _counting_mount(*widgets, **kwargs):
            calls.append(len(widgets))
            return original_mount(*widgets, **kwargs)

        transcript.mount = _counting_mount  # type: ignore[method-assign]
        transcript.set_messages(_messages(20))
        await transcript.refresh_messages()
        await _settle(pilot)
        transcript.mount = original_mount  # type: ignore[method-assign]

        mounted_rows = sum(calls)
        assert mounted_rows > 20, "the load must mount at least one row per message"
        assert len(calls) <= 2, (
            "a fresh load must mount its rows in one batch "
            f"(saw {len(calls)} mount calls for {mounted_rows} rows)"
        )


@pytest.mark.asyncio
async def test_removed_rows_are_pruned_in_one_batched_call():
    """A session switch removes the old rows with one `remove_children()`."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(20))
        await transcript.refresh_messages()
        await _settle(pilot)

        removals: list[int] = []
        original_remove_children = transcript.remove_children

        def _counting_remove_children(selector="*"):
            widgets = list(selector) if not isinstance(selector, str) else []
            removals.append(len(widgets))
            return original_remove_children(widgets or selector)

        transcript.remove_children = _counting_remove_children  # type: ignore[method-assign]
        transcript.set_messages(
            [_msg(f"other{i}", ConsoleMessageRole.ASSISTANT) for i in range(4)]
        )
        await transcript.refresh_messages()
        await _settle(pilot)
        transcript.remove_children = original_remove_children  # type: ignore[method-assign]

        assert removals, "the switch must remove the previous session's rows"
        assert len(removals) == 1, (
            f"stale rows must be removed in one batch (saw {len(removals)} calls)"
        )
        assert removals[0] > 20, "every stale row belongs to that one batch"
        assert _mounted_message_ids(transcript) == [f"mother{i}" for i in range(4)]
