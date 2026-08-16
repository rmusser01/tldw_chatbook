"""TASK-15777: two-sided Console transcript windowing.

TASK-15455 shipped a ONE-sided window: a contiguous hidden prefix whose
boundary every hydration and reveal path moves. That design left two
residuals its own Implementation Notes recorded:

1. *Scroll-back reachability ceiling* — once the mounted view reaches the LOW
   prune watermark, boundary hydration is refused forever (the refusal was
   the prune/hydration fixed-point loop-breaker), so older history can never
   be scrolled to (probe: 400 messages at 600/900 marks froze at ``m248``).
2. *Unbounded reveal on a far jump* — revealing a message near the start
   mounts everything from it to the tail in one pass (probe: 490 rows for
   ``m10`` of 500).

This task adds a second contiguous hidden boundary — a hidden TAIL — so the
mounted rows stay ONE contiguous slice (both boundaries are indices over the
same message list; islands are impossible by construction, so the no-gap-seam
property of the merged design is preserved). Two-sided behavior engages only
when windowing is on, pruning is on, and the watermarks can hold at least a
chunk plus a couple of viewports; degenerate configurations (the 45/70
fixed-point fixtures) and the kill switch keep the one-sided behavior
byte-for-byte, which is why the TASK-15455 suites pass unmodified.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


class TwoSidedHarness(App):
    """Transcript host with sane, scaled watermarks (two-sided regime)."""

    CSS = "ConsoleTranscript { height: 24; width: 100; }"

    def __init__(
        self,
        *,
        low: int = 600,
        high: int = 900,
        window_lines: int | None = None,
    ) -> None:
        super().__init__()
        chat_defaults: dict[str, object] = {
            "assistant_markdown": False,
            "prune_low_watermark": low,
            "prune_high_watermark": high,
        }
        if window_lines is not None:
            chat_defaults["transcript_window_lines"] = window_lines
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


def _is_contiguous(mounted: list[str], *, prefix: str = "m") -> bool:
    indices = [int(message_id[len(prefix) :]) for message_id in mounted]
    return indices == list(range(indices[0], indices[0] + len(indices)))


async def _settle(pilot, *, times: int = 6) -> None:
    for _ in range(times):
        await pilot.pause()


async def _wait_for(pilot, predicate, *, attempts: int = 80) -> bool:
    for _ in range(attempts):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _scroll_back_until(
    pilot,
    transcript: ConsoleTranscript,
    predicate,
    *,
    rounds: int = 80,
    bound: int | None = None,
) -> bool:
    """Drive repeated top-boundary hits, optionally asserting a DOM bound."""
    transcript.release_anchor()
    for _ in range(rounds):
        transcript.scroll_to(y=0, animate=False)
        await _settle(pilot)
        if bound is not None:
            mounted = len(_mounted_message_ids(transcript))
            assert mounted <= bound, (
                f"scroll-back must keep the DOM bounded: {mounted} > {bound}"
            )
        if predicate():
            return True
    return False


# ---------------------------------------------------------------------------
# Symptom 2: the scroll-back reachability ceiling (born red on TASK-15455).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sustained_scrollback_reaches_the_oldest_message_and_stays_bounded():
    """Scroll-back must reach m0 of 400 while the mounted DOM stays bounded.

    Born red on the one-sided design: hydration froze at ``first=m248`` the
    moment the mounted height crossed the low watermark (probe, 2026-08-15).
    """
    app = TwoSidedHarness()
    history = _messages(400)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        reached = await _scroll_back_until(
            pilot,
            transcript,
            lambda: "m0" in _mounted_message_ids(transcript),
            bound=170,
        )
        assert reached, (
            "sustained scroll-back never reached the oldest message: "
            f"stuck at {_mounted_message_ids(transcript)[:1]}"
        )
        assert transcript.virtual_size.height <= 900, (
            "the mounted view escaped the high watermark during scroll-back"
        )
        mounted = _mounted_message_ids(transcript)
        assert _is_contiguous(mounted), "mounted rows must stay one contiguous slice"
        assert len(transcript._messages) == 400, "the store keeps the full history"


@pytest.mark.asyncio
async def test_deep_scrollback_trims_the_tail_and_scrolling_down_recovers_it():
    """The tail leaves the DOM during deep scroll-back and rehydrates downward.

    Born red on the one-sided design: the tail row was ALWAYS mounted, which
    is exactly why the DOM could not stay bounded once the ceiling moved.
    """
    app = TwoSidedHarness()
    history = _messages(400)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        await _scroll_back_until(
            pilot,
            transcript,
            lambda: "m0" in _mounted_message_ids(transcript),
        )
        assert "m399" not in _mounted_message_ids(transcript), (
            "deep scroll-back must trim the tail out of the DOM"
        )

        # Scrolling down at the bottom boundary walks back toward the tail.
        for _ in range(80):
            transcript.scroll_to(y=transcript.max_scroll_y, animate=False)
            await _settle(pilot)
            mounted = len(_mounted_message_ids(transcript))
            assert mounted <= 260, (
                f"the downward walk must keep the DOM bounded: {mounted}"
            )
            if "m399" in _mounted_message_ids(transcript):
                break
        assert "m399" in _mounted_message_ids(transcript), (
            "scrolling down never recovered the trimmed tail"
        )
        assert _is_contiguous(_mounted_message_ids(transcript))


# ---------------------------------------------------------------------------
# Symptom 1: unbounded reveal on a far jump (born red on TASK-15455).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_far_jump_mounts_a_bounded_recentered_window():
    """Selecting a message near the start mounts a window, not 490 rows.

    Born red on the one-sided design: revealing ``m10`` of 500 mounted every
    row from it to the tail in one pass (probe: 490 rows).
    """
    app = TwoSidedHarness()
    history = _messages(500)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)
        assert "m10" not in _mounted_message_ids(transcript)

        transcript.select_message("m10")
        assert await _wait_for(
            pilot, lambda: "m10" in _mounted_message_ids(transcript)
        ), "selecting a windowed-out message never mounted it"
        await _settle(pilot)

        mounted = _mounted_message_ids(transcript)
        assert len(mounted) <= 80, (
            f"a far jump must mount a bounded window, got {len(mounted)} rows"
        )
        assert "m499" not in mounted, "the far window must not drag the tail along"
        assert _is_contiguous(mounted), "the re-centered window must be contiguous"
        assert transcript.selected_message_id == "m10"
        assert transcript.query(".console-transcript-action-row"), (
            "the revealed selection must show its action row"
        )
        assert not transcript._is_following_tail(), (
            "a far jump detaches the reader from the tail"
        )


@pytest.mark.asyncio
async def test_far_jump_then_scrolling_down_walks_back_to_the_tail():
    """After a far jump the tail stays reachable by scrolling down."""
    app = TwoSidedHarness()
    history = _messages(500)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        transcript.select_message("m10")
        assert await _wait_for(
            pilot, lambda: "m10" in _mounted_message_ids(transcript)
        )
        await _settle(pilot)
        assert "m499" not in _mounted_message_ids(transcript)

        for _ in range(120):
            transcript.scroll_to(y=transcript.max_scroll_y, animate=False)
            await _settle(pilot, times=4)
            if "m499" in _mounted_message_ids(transcript):
                break
        assert "m499" in _mounted_message_ids(transcript), (
            "the tail must be reachable by scrolling down after a far jump"
        )


# ---------------------------------------------------------------------------
# Interactions the 15455 notes named: tail-follow, streaming, the jump pill.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jump_to_latest_from_deep_scrollback_restores_a_bounded_tail():
    """The jump pill lands on a fresh, bounded, tail-following window."""
    app = TwoSidedHarness()
    history = _messages(400)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        await _scroll_back_until(
            pilot,
            transcript,
            lambda: "m0" in _mounted_message_ids(transcript),
        )
        assert "m399" not in _mounted_message_ids(transcript)

        transcript.jump_to_latest()
        assert await _wait_for(
            pilot, lambda: "m399" in _mounted_message_ids(transcript)
        ), "jump-to-latest never remounted the tail"
        await _settle(pilot)
        assert transcript._is_following_tail(), "the jump re-engages tail-follow"
        assert transcript.scroll_y == transcript.max_scroll_y
        assert len(_mounted_message_ids(transcript)) <= 170, (
            "the jump must land on a bounded tail window"
        )


@pytest.mark.asyncio
async def test_streaming_tick_keeps_the_hidden_tail_sticky():
    """A 0.2s-style ingest must not remount the trimmed tail under the reader."""
    app = TwoSidedHarness()
    history = _messages(400)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        await _scroll_back_until(
            pilot,
            transcript,
            lambda: "m0" in _mounted_message_ids(transcript),
        )
        mounted_before = _mounted_message_ids(transcript)
        assert "m399" not in mounted_before
        reader_y = float(transcript.scroll_y)

        live = ConsoleChatMessage(
            id="live-reply",
            role=ConsoleMessageRole.ASSISTANT,
            content="streamed so far",
            status="streaming",
        )
        for _ in range(3):
            transcript.set_messages([*history, live])
            await transcript.refresh_messages()
            await _settle(pilot, times=3)

        mounted_after = _mounted_message_ids(transcript)
        assert "live-reply" not in mounted_after, (
            "a streamed append while scrolled back must stay in the hidden tail"
        )
        assert mounted_after == mounted_before, (
            "the tick must not churn the reader's mounted slice"
        )
        assert float(transcript.scroll_y) == pytest.approx(reader_y, abs=1.0), (
            "the tick must not move the reader"
        )


@pytest.mark.asyncio
async def test_new_user_send_from_deep_scrollback_rewindows_the_tail():
    """A send yanks to a fresh tail window: suffix cleared, bounded, following."""
    app = TwoSidedHarness()
    history = _messages(400)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        await _scroll_back_until(
            pilot,
            transcript,
            lambda: "m0" in _mounted_message_ids(transcript),
        )
        assert "m399" not in _mounted_message_ids(transcript)

        transcript.note_follow_intent()
        sent = ConsoleChatMessage(
            id="sent-user", role=ConsoleMessageRole.USER, content="a fresh send"
        )
        placeholder = ConsoleChatMessage(
            id="pending-reply",
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            status="streaming",
        )
        transcript.set_messages([*history, sent, placeholder])
        await transcript.refresh_messages()
        assert await _wait_for(
            pilot, lambda: "sent-user" in _mounted_message_ids(transcript)
        ), "the send must land the reader on the tail"
        await _settle(pilot)
        assert transcript._is_following_tail(), "a send re-engages tail-follow"
        assert len(_mounted_message_ids(transcript)) <= 170, (
            "the send must re-window the tail, not remount the whole slice"
        )


# ---------------------------------------------------------------------------
# The gates: kill switch and degenerate watermarks keep one-sided behavior.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kill_switch_keeps_the_one_sided_ceiling_and_full_reveals():
    """`transcript_window_lines = 0` keeps the exact pre-task behavior."""
    app = TwoSidedHarness(window_lines=0)
    history = _messages(400)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        assert await _wait_for(
            pilot, lambda: bool(transcript._pruned_message_ids)
        ), "the watermarks must still prune with windowing disabled"
        await _settle(pilot)

        first_before = _mounted_message_ids(transcript)[0]
        transcript.release_anchor()
        for _ in range(6):
            transcript.scroll_to(y=0, animate=False)
            await _settle(pilot)
        assert _mounted_message_ids(transcript)[0] == first_before, (
            "with windowing disabled, the pruned prefix must stay unreachable "
            "by scroll (the pre-task contract the kill switch restores)"
        )
        assert _mounted_message_ids(transcript)[-1] == "m399", (
            "with windowing disabled the tail must never be trimmed"
        )
