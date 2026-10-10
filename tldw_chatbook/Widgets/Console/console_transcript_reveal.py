"""Bounded reveals for the Console transcript's two-sided window.

Every "show a message the window does not hold" path -- selection, the
task-501 swipe handoff and the Delete/Undo reselection that reuses it, and
reading-state restore -- lands here. A NEAR reveal (a j/k step over the
window's edge, a nearby restore) extends the window's one contiguous slice.
A FAR reveal re-centres a load-shaped window on the target instead, with
everything past it in the hidden tail (TASK-15777).

TASK-33628.5.1: "far" used to mean "more than the prune LOW watermark" --
12,000 estimated lines by default, about 4,000 short messages -- and the
handoff path had no bound at all. So selecting the first message of a
3,000-message chat, or undoing a Delete of it, mounted every later row:
24,085 widgets and 183 s before the event loop went quiet (dev 7d155170dc).
A reveal is now far once it would mount more than one load-shaped window
(the initial window budget, never more than the low watermark). A handed-off
selection the kept window already holds is bounded the same way when more
than one window lands below it -- the rows an Undo restores under its root.

The bound needs two-sided windowing (``ConsoleTranscript._two_sided_active``:
windowing and pruning both on, with sane watermarks). With the windowing
kill switch on, the whole history mounts anyway. With pruning off or
degenerate watermarks, every reveal is near: it mounts each row between the
window and the target, as before this task.

A window built this way, and the stretch an Undo restores below a kept
window, mounts a screenful at a time (``console_transcript_fill``).

The functions take the transcript and use only its window primitives, so
``console_transcript.py`` imports this module lazily and boot pays nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .console_transcript_fill import pace, paceable

if TYPE_CHECKING:
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


def far_reveal_line_budget(transcript: ConsoleTranscript) -> int:
    """Return the most estimated lines a reveal may add to the window."""
    return min(
        transcript._prune_watermarks()[0],
        transcript._initial_window_line_budget(),
    )


def estimated_window_lines(
    transcript: ConsoleTranscript, start: int, end: int, *, limit: int | None = None
) -> int:
    """Return the estimated rendered lines of ``_messages[start:end]``.

    Args:
        transcript: The transcript whose messages are measured.
        start: First index of the slice.
        end: Exclusive end of the slice.
        limit: Stop counting once the total exceeds this (the caller only
            needs to know the slice is over a budget), so a far reveal over
            3,000 messages costs one window's worth of estimates.
    """
    total = 0
    for message in transcript._messages[start:end]:
        total += transcript._estimated_message_lines(message)
        if limit is not None and total > limit:
            break
    return total


def _is_far(transcript: ConsoleTranscript, start: int, end: int) -> bool:
    if not transcript._two_sided_active():
        return False
    budget = far_reveal_line_budget(transcript)
    return estimated_window_lines(transcript, start, end, limit=budget) > budget


def reveal_message(transcript: ConsoleTranscript, message_id: str) -> bool:
    """Bring ``message_id`` into the window; see ``ConsoleTranscript.reveal_message``."""
    messages = transcript._messages
    for requested_index, message in enumerate(messages):
        if message.id != message_id:
            continue
        index, unit_end, owner_id, _owned_ids = transcript._unit_span_at(
            messages, requested_index
        )
        first_visible = transcript._first_visible_message_index()
        tail_start = transcript._hidden_tail_start_index()
        if first_visible <= index and unit_end <= tail_start:
            return False
        if index < first_visible:
            revealed_start = transcript._turn_aligned_start(messages, index)
            revealed_end = first_visible
        else:
            revealed_start = tail_start
            revealed_end = unit_end
        if _is_far(transcript, revealed_start, revealed_end):
            recenter_window_on(transcript, index, owner_id)
        elif index < first_visible:
            transcript._set_hidden_prefix(revealed_start)
        else:
            transcript._reveal_hidden_tail_through(revealed_end)
        return True
    return False


def hand_off_selection(
    transcript: ConsoleTranscript, message_id: str, pending_index: int, start: int
) -> int:
    """Bring a handed-off selection into the window an ingest is building.

    ``set_messages`` calls this for a pending selection (a sibling swipe, or
    the restored root an Undo reselects) once its id is in the ingested set.

    Args:
        transcript: The ingesting transcript.
        message_id: The handed-off selection.
        pending_index: Its index in the new message list.
        start: The window start the ingest computed so far.

    Returns:
        The window start to use. The hidden tail is updated in place.
    """
    messages = transcript._messages
    if pending_index < start:
        # Branch sibling handoff: the replacement id may sit exactly where
        # the old window boundary id disappeared. Keep the new selected row
        # in the window rather than mounting a disjoint orphan.
        revealed_start = transcript._turn_aligned_start(messages, pending_index)
        if not _is_far(transcript, revealed_start, start):
            return revealed_start
        return recenter_window_on(transcript, pending_index, message_id)
    tail_start = transcript._hidden_tail_start_index()
    if pending_index >= tail_start:
        # TASK-15777: same contract on the other boundary -- a handed-off
        # selection inside the hidden tail extends the slice down through it.
        _unit_start, unit_end, _owner, _ids = transcript._unit_span_at(
            messages, pending_index
        )
        if _is_far(transcript, tail_start, unit_end):
            return recenter_window_on(transcript, pending_index, message_id)
        transcript._reveal_hidden_tail_through(pending_index + 1)
    elif _is_far(transcript, pending_index, tail_start):
        # Inside the kept window, but what lands below it is no window: the
        # rows an Undo restores (or a swipe's new branch) under a window
        # kept from before the ingest. Undo of a Delete from an early row
        # mounted all 2,970 later rows this way.
        return recenter_window_on(transcript, pending_index, message_id)
    elif paceable(transcript, pending_index, tail_start):
        # Up to a window of new rows below it (the 60 an Undo of a whole
        # short chat restores): a screenful at a time.
        pace(transcript, pending_index, tail_start, keep_following=True)
    return start


def recenter_window_on(
    transcript: ConsoleTranscript, index: int, message_id: str
) -> int:
    """Mount a bounded, load-shaped window with the target's turn on top.

    The far jump detaches the reader from the tail (it IS a user navigation
    away from it -- a later send's follow intent still outranks it, per
    TASK-336's ordering), and the target row is scrolled to the top of the
    viewport once mounted, because the old scroll offset is meaningless in
    the new window.

    Returns:
        The new window start (the first mounted message index).
    """
    messages = transcript._messages
    requested_index = next(
        (
            candidate_index
            for candidate_index, message in enumerate(messages)
            if message.id == message_id
        ),
        index,
    )
    unit_start, unit_end, owner_id, _owned_ids = transcript._unit_span_at(
        messages, requested_index
    )
    start = transcript._turn_aligned_start(messages, unit_start)
    budget = transcript._initial_window_line_budget()
    used = 0
    end = unit_start
    while end < len(messages) and used < budget:
        used += transcript._estimated_message_lines(messages[end])
        end += 1
    if end < unit_end:
        end = unit_end
    elif 0 < end < len(messages):
        _included_start, included_end, _included_owner, _included_ids = (
            transcript._unit_span_at(messages, end - 1)
        )
        end = included_end
    transcript._set_hidden_prefix(start)
    # The target's turn and a screen below it first; the rest a batch later.
    pace(transcript, unit_start, end)
    transcript.release_anchor()
    transcript._reveal_scroll_target = owner_id
    # Review E: the reconcile that realizes this window transits an emptied
    # arrangement, and the placement parks the target near y=0 -- both read
    # as top-boundary hits and hydrated one spurious chunk ABOVE the jump
    # target. Suppressed until the placement lands.
    transcript._suppress_boundary_hydration = True
    return start
