"""TASK-16851: head-pinned selection must not let tailward hydration outrun the prune.

Deferred out of TASK-15777's merge gate (PR #1733): a far jump both SELECTS its
target and lands it at the HEAD of the re-centered window. The prune protects
the selected message and stops its walk at the first protected group, so with
the target selected at the head it can never trim anything — while the tailward
hydration chain keeps revealing. The round-3 review measured 490 mounted rows /
virtual height 1966 against a high watermark of 900 (2.18x), growing with
session length; Esc was the only recovery.

The fix restores the invariant the other three boundary regimes already hold —
mounted height stays bounded by ~high regardless of selection position — by
refusing ``_hydrate_tailward`` while the measured height is at/over the high
mark AND the prune walk is blocked: hydration must not outrun a prune that
cannot make room. The selection is never evicted (contiguity + a mounted
selection + a mounted far tail are mutually exclusive, so with the selection
HELD the downward walk stalls bounded); clearing the selection (Esc) or the
jump pill restores full downward reachability.

Also pinned here: the frame-wide End-during-prune race the same review filed —
an End pressed between the prune's entry-capture and its restore had its anchor
cancelled by the restore's quiet release (entry state won), truncating the
drain to one chunk until a second End.
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

    def __init__(self, *, low: int = 300, high: int = 450) -> None:
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


def _messages(count: int) -> list[ConsoleChatMessage]:
    return [
        ConsoleChatMessage(
            id=f"m{index}",
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


async def _settle(pilot, *, times: int = 6) -> None:
    for _ in range(times):
        await pilot.pause()


async def _wait_for(pilot, predicate, *, attempts: int = 80) -> bool:
    for _ in range(attempts):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _far_jump_to_m10(pilot, transcript: ConsoleTranscript) -> None:
    """Select a windowed-out early message: re-center + head-pinned selection."""
    transcript.select_message("m10")
    assert await _wait_for(
        pilot, lambda: "m10" in _mounted_message_ids(transcript)
    ), "selecting a windowed-out message never mounted it"
    await _settle(pilot)
    assert _mounted_message_ids(transcript)[0] == "m10", (
        "precondition: the jump target must be head-pinned (first mounted row)"
    )


# ---------------------------------------------------------------------------
# Finding 1: the head-pinned walk-down (born red at ecbcd5cd8: the reviewer's
# paced repro grew to 490 rows / height 1966 vs a high mark of 900).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_head_pinned_selection_walkdown_stays_bounded_near_high_mark():
    """Jump -> paced walk down: height stays ~high, the selection stays mounted.

    Born red at HEAD (`ecbcd5cd8`): with the selection pinned at the window
    head, ``_compute_prunable_prefix`` stops at the first (protected) group
    on every check, so nothing ever trims while ``_hydrate_tailward`` keeps
    revealing — the mounted height escaped the high watermark and kept
    growing with every gesture, bounded only by session length.
    """
    app = TwoSidedHarness()
    history = _messages(500)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)
        await _far_jump_to_m10(pilot, transcript)

        # The reviewer's paced walk: repeated bottom-boundary gestures with
        # frames between each, exactly how a reader walks down from a jump.
        for _ in range(40):
            transcript.scroll_to(y=transcript.max_scroll_y, animate=False)
            await _settle(pilot)
            height = transcript.virtual_size.height
            assert height <= 1000, (
                "a head-pinned selection must not disable the height bound: "
                f"virtual height {height} escaped the 900 high watermark"
            )
            # ~4 measured rows per message in this harness: the 1100-line
            # height budget holds ~275 messages. HEAD mounted 490.
            assert len(_mounted_message_ids(transcript)) <= 300, (
                "the mounted DOM must stay bounded during the walk-down"
            )

        # The bound is a fixed point, not a lucky frame: idle does not grow it.
        await _settle(pilot, times=20)
        assert transcript.virtual_size.height <= 1000

        # Selection UX survives the bounding mechanism: the jumped-to row is
        # still mounted, still first, still selected, action row and all.
        mounted = _mounted_message_ids(transcript)
        assert mounted[0] == "m10", "the head-pinned selection must stay mounted"
        assert transcript.selected_message_id == "m10", "no selection loss"
        assert transcript.query(".console-transcript-action-row"), (
            "the selection must keep its action row while it blocks the prune"
        )


@pytest.mark.asyncio
async def test_head_pinned_walkdown_stalls_then_esc_restores_reachability():
    """With the selection HELD the walk stalls bounded; Esc resumes it.

    Born red at HEAD (`ecbcd5cd8`) in the stall half: the walk reached the
    true tail by mounting everything between the selection and m499 in one
    contiguous, unbounded slice. Contiguity + a mounted selection + a mounted
    far tail are mutually exclusive, so under the height bound the walk MUST
    stall while the head-pinned selection is held — and clearing it (Esc)
    unblocks the prune, which makes room, which lets hydration resume all the
    way to the tail. That recovery is the second half of this pin.
    """
    app = TwoSidedHarness()
    history = _messages(500)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)
        await _far_jump_to_m10(pilot, transcript)

        for _ in range(40):
            transcript.scroll_to(y=transcript.max_scroll_y, animate=False)
            await _settle(pilot)
        assert "m499" not in _mounted_message_ids(transcript), (
            "reaching the tail with the head-pinned selection still mounted "
            "would mean the slice grew unbounded again"
        )
        assert transcript._hidden_tail_ids, (
            "the stall must be a paused hidden tail, not a dropped one"
        )

        # Esc: the documented recovery. Clearing the selection unblocks the
        # prune (which trims the head and makes room), and the next REAL
        # boundary gestures — PageDown here; wheel-down and End share the
        # same hook — slide the window to the tail. A bare scroll_to(max) is
        # not a user path: once the reader sits exactly at max it produces
        # no scroll_y change, so nothing re-fires the boundary watcher.
        transcript.action_clear_selection()
        await _settle(pilot)
        reached = False
        for _ in range(80):
            transcript.scroll_to(y=transcript.max_scroll_y, animate=False)
            transcript.action_page_down()
            await _settle(pilot)
            assert len(_mounted_message_ids(transcript)) <= 260, (
                "the post-Esc walk must slide (bounded), not grow"
            )
            if "m499" in _mounted_message_ids(transcript):
                reached = True
                break
        assert reached, "clearing the selection must restore tail reachability"


# ---------------------------------------------------------------------------
# Finding 2: the frame-wide End-during-prune race (round-3 residual).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_pressed_inside_the_prune_window_still_drains_to_the_tail():
    """An End that lands between prune entry-capture and restore must win.

    Born red at HEAD (`ecbcd5cd8`): the prune's restore compares the anchor
    state against its ENTRY capture, so an End pressed inside the entry ->
    restore window — engaging the raw anchor AFTER the capture — was read as
    the shrink-clamp's spurious re-attach and quietly released, and the
    entry-offset compensation then pulled the reader off the bottom. The
    drain truncated after ~one chunk; the pill was up and a SECOND End
    resumed it (annoying, not stranding). The fix stamps ``scroll_end`` with
    an intent time: a stamp newer than prune entry is the user's End, so the
    restore keeps the anchor, skips the stale entry-offset compensation, and
    re-arms the drain.
    """
    app = TwoSidedHarness()
    history = _messages(400)
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        # Deep scroll-back: build a hidden tail, then park DETACHED mid-window
        # (not at either boundary) — the entry state the race needs.
        transcript.release_anchor()
        for _ in range(80):
            transcript.scroll_to(y=0, animate=False)
            await _settle(pilot)
            if "m0" in _mounted_message_ids(transcript):
                break
        assert transcript._hidden_tail_ids, "precondition: a tail must be hidden"
        transcript.scroll_to(y=min(300, transcript.max_scroll_y // 2), animate=False)
        await _settle(pilot)
        assert not transcript._raw_anchor_engaged(), (
            "precondition: the reader must be detached at prune entry"
        )

        # Shrink the marks so the settled ~low-height view is now over high:
        # the next prune check fires deterministically, entry-detached.
        high_mark = transcript.virtual_size.height - 25
        app.app_config["chat_defaults"]["prune_low_watermark"] = high_mark - 100
        app.app_config["chat_defaults"]["prune_high_watermark"] = high_mark
        assert transcript.virtual_size.height > high_mark, (
            "precondition: the mounted view must sit over the shrunken high mark"
        )

        # Inject the End INSIDE the race window. `Widget.scroll_end` engages
        # the raw anchor synchronously but DEFERS its scroll via
        # call_after_refresh; pressing End during the prune's reconcile (after
        # entry-capture, before `call_after_refresh(_restore_scroll)` is even
        # enqueued) puts the End's deferred scroll AHEAD of the restore in
        # the after-refresh queue — so the restore has the last word on the
        # anchor, exactly the frame-wide interleaving the round-3 review
        # measured. (An End that lands after the restore is enqueued has its
        # deferred scroll run last and self-heals; that ordering is not the
        # race.)
        fired: dict[str, object] = {
            "end": False,
            "pruned_len": len(transcript._pruned_message_ids),
        }
        original_reconcile = transcript._reconcile_rows

        async def reconcile_then_press_end(rows) -> None:
            grew = len(transcript._pruned_message_ids) > fired["pruned_len"]
            await original_reconcile(rows)
            fired["pruned_len"] = len(transcript._pruned_message_ids)
            if not fired["end"] and grew and transcript._hidden_tail_ids:
                fired["end"] = True
                transcript.scroll_end(animate=False)

        transcript._reconcile_rows = reconcile_then_press_end  # type: ignore[method-assign]
        try:
            transcript._schedule_prune_check()
            assert await _wait_for(pilot, lambda: bool(fired["end"]), attempts=40), (
                "the harness never landed an End inside the prune window"
            )
        finally:
            transcript._reconcile_rows = original_reconcile  # type: ignore[method-assign]

        # No further input: the End alone must drain the hidden tail to the
        # true tail. At HEAD the restore cancelled the anchor and the drain
        # stopped after one chunk.
        assert await _wait_for(
            pilot, lambda: not transcript._hidden_tail_ids, attempts=300
        ), (
            "the End-during-prune race truncated the drain: "
            f"{len(transcript._hidden_tail_ids)} messages still hidden at "
            f"last={_mounted_message_ids(transcript)[-1]}"
        )
        assert _mounted_message_ids(transcript)[-1] == "m399"
        assert transcript._is_following_tail(), (
            "the honored End must leave the reader following the tail"
        )
