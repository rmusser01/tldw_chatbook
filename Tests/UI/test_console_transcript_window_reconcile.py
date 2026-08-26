"""TASK-15455 reconciliation: the review-hardened delta on top of the merged window.

Two sessions implemented this task independently. The merged-first
implementation (PR #1538) owns the design: ONE contiguous hidden prefix
(`_pruned_message_ids`), a viewport-derived line budget, turn-aligned
boundaries, and wheel/page-up/scroll-boundary hydration. Its four tests
(`test_console_transcript_windowing.py`) stay green and unmodified.

This file carries what the second implementation's review round found and the
merged one does not have: the prune/hydration fixed point, reading-state
restore into a windowed-out selection, a config surface with a kill switch, and
the load-shape pins that were written before either implementation existed.
"""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    DEFAULT_INITIAL_WINDOW_LINES,
    DEFAULT_SCROLLBACK_CHUNK_LINES,
    ConsoleTranscript,
    get_console_transcript_window_lines,
)


class ReconcileHarness(App):
    """Transcript host with explicit watermarks and window settings."""

    CSS = "ConsoleTranscript { height: 24; width: 100; }"

    def __init__(
        self,
        *,
        low: int = 12_000,
        high: int = 0,
        window_lines: int | None = None,
        scrollback_lines: int | None = None,
    ) -> None:
        super().__init__()
        chat_defaults: dict[str, object] = {
            "assistant_markdown": False,
            "prune_low_watermark": low,
            "prune_high_watermark": high,
        }
        if window_lines is not None:
            chat_defaults["transcript_window_lines"] = window_lines
        if scrollback_lines is not None:
            chat_defaults["transcript_scrollback_lines"] = scrollback_lines
        self.app_config = {"chat_defaults": chat_defaults}

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
    return [row.message_id for row in transcript.query(".console-transcript-message")]


async def _wait_for(pilot, predicate, *, attempts: int = 80) -> bool:
    for _ in range(attempts):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _settle(pilot, *, times: int = 8) -> None:
    for _ in range(times):
        await pilot.pause()


# ---------------------------------------------------------------------------
# Load-shape pins (written before either implementation; adapted to this API).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pin_long_load_is_bounded_and_leaves_the_reader_at_the_tail():
    """A 500-message load mounts a bounded tail and stays anchored to it."""
    app = ReconcileHarness()
    history = _messages(500)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        mounted = _mounted_message_ids(transcript)
        assert 0 < len(mounted) <= 60, f"the load window must be bounded: {len(mounted)}"
        assert mounted[-1] == "m499"
        assert transcript._is_following_tail(), "load must keep tail-follow engaged"
        assert transcript.scroll_y == transcript.max_scroll_y
        assert len(transcript._messages) == 500, "the store snapshot is untouched"
        # Exports still render the whole history, mounted or not.
        plain = transcript.to_plain_text(width=40)
        assert "message 0 line 0" in plain
        assert "message 499 line 3" in plain


@pytest.mark.asyncio
async def test_pin_session_switch_rewindows_on_the_new_history():
    """Switching sessions drops every old row and re-windows on the new tail."""
    app = ReconcileHarness()
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(200))
        await transcript.refresh_messages()
        await _settle(pilot)

        transcript.set_messages(_messages(200, prefix="other"))
        await transcript.refresh_messages()
        await _settle(pilot)

        mounted = _mounted_message_ids(transcript)
        assert mounted, "the new session must render rows"
        assert all(message_id.startswith("other") for message_id in mounted)
        assert mounted[-1] == "other199"
        assert len(mounted) < 200, "the new session gets a bounded window too"
        assert transcript._is_following_tail()


@pytest.mark.asyncio
async def test_pin_watermarks_still_bound_the_mounted_height():
    """The height watermarks keep bounding the transcript under windowing."""
    app = ReconcileHarness(low=45, high=70)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(200))
        await transcript.refresh_messages()
        assert await _wait_for(pilot, lambda: transcript.virtual_size.height <= 70), (
            f"height {transcript.virtual_size.height} escaped the high watermark"
        )
        assert len(transcript._messages) == 200


# ---------------------------------------------------------------------------
# Selection into windowed-out history (the merged implementation already does
# this; nothing pinned it).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selecting_a_windowed_out_message_reveals_it_contiguously():
    """`select_message` reveals through the target, keeping ONE tail stretch."""
    app = ReconcileHarness()
    history = _messages(300)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)
        assert "m10" not in _mounted_message_ids(transcript), "target must start hidden"

        transcript.select_message("m10")
        assert await _wait_for(
            pilot, lambda: "m10" in _mounted_message_ids(transcript)
        ), "selecting a windowed-out message never mounted it"
        assert transcript.selected_message_id == "m10"
        assert transcript.query(".console-transcript-action-row"), (
            "the revealed selection must show its action row"
        )

        mounted = _mounted_message_ids(transcript)
        expected = [message.id for message in history[history.index(history[10]) :]]
        assert mounted == expected[: len(mounted)], "the reveal must stay contiguous"
        assert mounted[-1] == "m299"


@pytest.mark.asyncio
async def test_mounted_rows_stay_one_contiguous_suffix_after_prune_and_jump():
    """Why no gap marker is needed here: the hidden set is always a PREFIX.

    A jump above a pruned stretch cannot produce an island in this design —
    both the watermark walk and every reveal path move the same single
    boundary. (The second implementation, which tracked pruning and windowing
    as two sets, could produce islands and needed a seam row.)
    """
    app = ReconcileHarness(low=45, high=70)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = _messages(200)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        assert await _wait_for(pilot, lambda: bool(transcript._pruned_message_ids))
        await _settle(pilot)

        transcript.select_message("m20")
        assert await _wait_for(
            pilot, lambda: "m20" in _mounted_message_ids(transcript)
        )
        await _settle(pilot)

        mounted = _mounted_message_ids(transcript)
        first = int(mounted[0][1:])
        assert mounted == [f"m{index}" for index in range(first, 200)], (
            f"mounted rows must be one contiguous suffix, got {mounted[:5]}…"
        )
        hidden = transcript._pruned_message_ids
        assert hidden == {f"m{index}" for index in range(first)}, (
            "the hidden set must stay a contiguous prefix"
        )


# ---------------------------------------------------------------------------
# The prune/hydration fixed point (the merged implementation churns forever).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrollback_hydration_and_pruning_reach_a_fixed_point():
    """Idle frames must not churn rows in and out.

    Measured on the merged implementation before this fix: with the reader at
    the scroll boundary and the transcript over its high watermark, the hidden
    prefix oscillated between 169 and 152 messages (height 47 <-> 115) forever,
    with no user input — the prune's own scroll restoration re-triggers the
    boundary hydration that produced it.
    """
    app = ReconcileHarness(low=45, high=70)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(180))
        await transcript.refresh_messages()
        await _settle(pilot, times=5)

        transcript.release_anchor()
        transcript.scroll_to(y=0, animate=False)
        samples: list[int] = []
        for _ in range(40):
            await pilot.pause()
            samples.append(len(transcript._pruned_message_ids))

        assert len(set(samples[-12:])) == 1, (
            f"the window must settle, not churn: {samples[-12:]}"
        )
        assert transcript.virtual_size.height <= 70, (
            "the settled window must still respect the high watermark"
        )


@pytest.mark.asyncio
async def test_explicit_hydration_still_works_below_the_low_watermark():
    """The fixed point must not cost ordinary scrollback its hydration.

    Watermarks are ENABLED here (the shipped defaults) so the new gate is
    actually evaluated: a normal transcript sits far below the low mark, and
    hydration must be allowed. With `high = 0` the gate short-circuits and this
    test would prove nothing about it.
    """
    app = ReconcileHarness(low=12_000, high=20_000)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(180))
        await transcript.refresh_messages()
        await _settle(pilot)

        hidden_before = len(transcript._pruned_message_ids)
        transcript.release_anchor()
        transcript.scroll_to(y=0, animate=False)
        assert await _wait_for(
            pilot, lambda: len(transcript._pruned_message_ids) < hidden_before
        ), "reaching the boundary must still hydrate earlier history"


# ---------------------------------------------------------------------------
# Reading-state restore into a windowed-out selection.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restored_reading_state_reveals_its_selected_message():
    """`restore_reading_state` assigns the id directly — it must reveal it first."""
    from tldw_chatbook.UI.Console_Modules.transcript import (
        ConsoleTranscriptRegion,
        _ConsoleTranscriptReadingState,
    )

    app = ReconcileHarness()
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(300))
        await transcript.refresh_messages()
        await _settle(pilot)
        assert "m12" not in _mounted_message_ids(transcript)

        region = ConsoleTranscriptRegion.__new__(ConsoleTranscriptRegion)
        region._transcript_or_none = lambda: transcript  # type: ignore[method-assign]
        ConsoleTranscriptRegion.restore_reading_state(
            region,
            _ConsoleTranscriptReadingState(
                anchored=False, scroll_y=6.0, selected_message_id="m12"
            ),
        )
        assert await _wait_for(
            pilot, lambda: "m12" in _mounted_message_ids(transcript)
        ), "the restored selection was never revealed"
        assert transcript.selected_message_id == "m12"


@pytest.mark.asyncio
async def test_restored_offset_is_applied_against_the_revealed_window():
    """The offset must be clamped against the window the restore produces.

    Read order matters: applying `scroll_y` before the revealed rows mount
    clamps it against the SMALLER pre-reveal `max_scroll_y`, silently dropping
    the reader somewhere else. (The second implementation shipped the mirror
    image of this bug — it evaluated tail-following against the pre-restore
    anchor — and its re-review caught it.)
    """
    from tldw_chatbook.UI.Console_Modules.transcript import (
        ConsoleTranscriptRegion,
        _ConsoleTranscriptReadingState,
    )

    app = ReconcileHarness()
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(300))
        await transcript.refresh_messages()
        await _settle(pilot)
        pre_reveal_max = transcript.max_scroll_y
        target_offset = float(pre_reveal_max) + 40.0

        region = ConsoleTranscriptRegion.__new__(ConsoleTranscriptRegion)
        region._transcript_or_none = lambda: transcript  # type: ignore[method-assign]
        ConsoleTranscriptRegion.restore_reading_state(
            region,
            _ConsoleTranscriptReadingState(
                anchored=False,
                scroll_y=target_offset,
                selected_message_id="m12",
            ),
        )
        assert await _wait_for(
            pilot, lambda: "m12" in _mounted_message_ids(transcript)
        )
        await _settle(pilot)

        assert transcript.max_scroll_y > pre_reveal_max, (
            "the reveal must have grown the scrollable region"
        )
        assert transcript.scroll_y == pytest.approx(target_offset, abs=1.0), (
            "the restored offset must survive the reveal, not be clamped to the "
            f"pre-reveal maximum ({transcript.scroll_y} vs {target_offset})"
        )


# ---------------------------------------------------------------------------
# Config surface + kill switch.
# ---------------------------------------------------------------------------


def test_window_line_settings_resolve_from_config():
    """Defaults, invalid values, and the kill switch."""
    assert get_console_transcript_window_lines(None) == (
        DEFAULT_INITIAL_WINDOW_LINES,
        DEFAULT_SCROLLBACK_CHUNK_LINES,
    )
    assert get_console_transcript_window_lines({"chat_defaults": "nope"}) == (
        DEFAULT_INITIAL_WINDOW_LINES,
        DEFAULT_SCROLLBACK_CHUNK_LINES,
    )
    assert get_console_transcript_window_lines(
        {"chat_defaults": {"transcript_window_lines": "x"}}
    ) == (DEFAULT_INITIAL_WINDOW_LINES, DEFAULT_SCROLLBACK_CHUNK_LINES)
    assert get_console_transcript_window_lines(
        {
            "chat_defaults": {
                "transcript_window_lines": 500,
                "transcript_scrollback_lines": 250,
            }
        }
    ) == (500, 250)
    # Kill switch is preserved verbatim; the chunk floor is clamped to >= 1.
    assert get_console_transcript_window_lines(
        {
            "chat_defaults": {
                "transcript_window_lines": 0,
                "transcript_scrollback_lines": 0,
            }
        }
    ) == (0, 1)


@pytest.mark.asyncio
async def test_kill_switch_mounts_the_whole_history():
    """`transcript_window_lines = 0` restores the pre-task behavior."""
    app = ReconcileHarness(window_lines=0)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(120))
        await transcript.refresh_messages()
        await _settle(pilot)

        assert transcript._pruned_message_ids == set()
        assert len(_mounted_message_ids(transcript)) == 120


@pytest.mark.asyncio
async def test_configured_window_lines_change_the_load_window():
    """A larger configured floor mounts more of the tail."""
    small = ReconcileHarness(window_lines=1)
    async with small.run_test(size=(100, 30)) as pilot:
        transcript = small.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(300))
        await transcript.refresh_messages()
        await _settle(pilot)
        default_window = len(_mounted_message_ids(transcript))

    big = ReconcileHarness(window_lines=800)
    async with big.run_test(size=(100, 30)) as pilot:
        transcript = big.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(300))
        await transcript.refresh_messages()
        await _settle(pilot)
        big_window = len(_mounted_message_ids(transcript))

    assert big_window > default_window, (
        f"the configured floor must widen the window ({big_window} vs {default_window})"
    )
    assert big_window < 300, "a configured floor is still a window"


@pytest.mark.asyncio
async def test_kill_switch_does_not_resurrect_watermark_pruned_rows():
    """The kill switch disables WINDOWING, not the height watermarks.

    Measured before the fix: forcing the window start to 0 on every ingest
    cleared the pruned prefix, so an over-watermark session re-mounted its
    whole history on each 0.2s sync tick and pruned it back down again (180
    rows remounted, settled to 11, every tick). The default-watermark kill
    switch test cannot see this — nothing is ever pruned there.
    """
    app = ReconcileHarness(low=45, high=70, window_lines=0)
    history = _messages(180)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        assert await _wait_for(pilot, lambda: bool(transcript._pruned_message_ids)), (
            "the watermarks must still prune with windowing disabled"
        )
        await _settle(pilot)
        pruned = set(transcript._pruned_message_ids)
        settled_mounted = len(_mounted_message_ids(transcript))
        assert settled_mounted < 30, "the fixture must settle well under the history"

        mounted_per_tick: list[int] = []
        original_mount = transcript.mount

        def _recording_mount(*widgets, **kwargs):
            mounted_per_tick.append(len(widgets))
            return original_mount(*widgets, **kwargs)

        transcript.mount = _recording_mount  # type: ignore[method-assign]
        try:
            for _ in range(3):
                transcript.set_messages(list(history))
                assert transcript._pruned_message_ids >= pruned, (
                    "an ingest must not resurrect watermark-pruned rows"
                )
                await transcript.refresh_messages()
                assert len(_mounted_message_ids(transcript)) <= settled_mounted, (
                    "the tick must not re-mount the full history"
                )
                await _settle(pilot, times=4)
        finally:
            transcript.mount = original_mount  # type: ignore[method-assign]

        assert sum(mounted_per_tick) < 60, (
            f"steady-state ticks must mount almost nothing, saw {mounted_per_tick}"
        )
        assert len(transcript._messages) == 180
