"""Mount a Console transcript window a screenful at a time.

TASK-33628.5.1 AC#3. A re-centred window (a far selection, or the restored
root an Undo reselects) and the stretch an Undo restores below a kept window
were mounted in one batch: 64 short rows, about 520 widgets, at 160x48.
Textual lays a batch of new widgets out three times -- its first layout, the
relayout their new virtual sizes ask for, and another once the transcript's
scrollbar changes its width -- and each pass re-arranged every new widget.
One screen update then held the event loop for 196-316 ms with 3,000
messages and 211-229 ms with 60 (dev 46c3959526, ConsoleHarness 160x48).

``pace`` now mounts the target's turn and the screen below it first
(``fill_line_budget``: one viewport of estimated lines, which runs short of
real row heights, so the screen fills) and remembers where the planned
window ends. Everything past the batch waits in the hidden tail, and
``fill_window`` reveals it one batch at a time, each only once the layout of
the batch before it has settled; mounted any earlier, its first layout lands
in the same pass as that batch's relayout (measured: a 158 ms pass
re-arranging 44 rows' widgets). The window ends up the same shape it always
had. Interleaved with dev under the same load (n=5 each), the worst Undo
block fell from 191-308 ms to 122-158 ms at 3,000 messages; two-viewport
batches left it at 143-205 ms.

Pacing needs two-sided windowing (windowing and pruning on, sane
watermarks), like the far-reveal bound it extends. A reader following the
tail when an Undo brings rows back below a kept window (the usual state
after deleting the last turns) is detached quietly while the batches land
and follows the tail again once the window is whole, unless they scrolled
meanwhile; a follower would otherwise ride each batch down, and the
next sync's ghost-follow heal would mount the rest at once. A fill stops as
soon as anything else moves the window: a send, End, the jump pill, scroll
hydration past its end, or a session switch.

``console_transcript_reveal`` imports this module, and the transcript
imports that one lazily, so boot pays nothing (ADR-097).
"""

from __future__ import annotations

from functools import partial
from time import monotonic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

#: Viewports of estimated lines one batch mounts: the screen the reader sees.
FILL_VIEWPORTS = 1
#: Seconds between checks that the batch before has settled.
_POLL_S = 0.03
#: Seconds a fill waits for one batch to settle before it leaves the rest of
#: the window to scroll hydration (a screen nobody shows never lays out).
_PATIENCE_S = 5.0


def fill_line_budget(transcript: ConsoleTranscript) -> int:
    """Return the estimated lines one batch of a paced window may mount."""
    return min(
        FILL_VIEWPORTS * transcript._window_viewport_height(),
        transcript._initial_window_line_budget(),
    )


def _batch_end(transcript: ConsoleTranscript, start: int, end: int) -> int:
    """Return the exclusive end of one batch of ``_messages[start:end]``."""
    budget = fill_line_budget(transcript)
    used = 0
    stop = start
    while stop < end and used < budget:
        used += transcript._estimated_message_lines(transcript._messages[stop])
        stop += 1
    return stop


def paceable(transcript: ConsoleTranscript, first: int, end: int) -> bool:
    """Return True when ``_messages[first:end]`` should mount a batch at a time.

    Only a stretch longer than one batch that nothing has mounted yet is
    held back (hiding mounted rows would unmount them), and only with
    two-sided windowing on: the one-sided regimes never hold a hidden tail.
    """
    if not transcript._two_sided_active():
        return False
    if _batch_end(transcript, first, end) >= end:
        return False
    mounted = {key.split(":", 2)[1] for key in transcript._row_widgets if ":" in key}
    return not any(message.id in mounted for message in transcript._messages[first:end])


def pace(
    transcript: ConsoleTranscript, first: int, end: int, *, keep_following: bool = False
) -> None:
    """Reveal the window through one batch from ``first``; fill the rest later.

    The window's start is already set. ``_messages[:end]`` is the planned
    window; its end is remembered by its last message's id, so an ingest
    that shifts indices cannot move it.

    Args:
        transcript: The transcript whose window is being built.
        first: Index the first batch starts at (the selection's turn).
        end: Exclusive end of the planned window.
        keep_following: A reader following the tail is detached while the
            batches land and follows it again once the window is whole.
    """
    transcript._reveal_hidden_tail_through(_batch_end(transcript, first, end))
    if transcript._hidden_tail_start_index() >= end:
        transcript._window_fill = None
        return
    refollow = None
    if keep_following and transcript._raw_anchor_engaged():
        transcript._release_anchor_quietly()
        refollow = monotonic()
    transcript._window_fill = (transcript._messages[end - 1].id, refollow)
    _schedule(transcript)


def _schedule(transcript: ConsoleTranscript) -> None:
    fill = transcript._window_fill
    if fill is not None:
        _poll(transcript, fill, monotonic() + _PATIENCE_S, settled=False)


def _finish(transcript: ConsoleTranscript, fill: tuple[str, float | None]) -> None:
    """End a fill; a reader it detached follows the tail again unless they scrolled."""
    transcript._window_fill = None
    refollow = fill[1]
    if refollow is not None and transcript._user_scroll_time <= refollow:
        transcript.anchor()


def _poll(
    transcript: ConsoleTranscript,
    fill: tuple[str, float | None],
    deadline: float,
    settled: bool,
) -> None:
    """Start the next batch once two checks a poll apart find the last settled.

    Two checks close the race with Textual's own hand-off: a widget clears
    its pending-relayout flag only as it posts the request to the screen,
    and by the next poll the screen has taken that request (its flag) or run
    the pass.
    """
    if transcript._window_fill is not fill:
        return
    if not transcript.is_mounted:
        transcript._window_fill = None
        return
    ready = _ready(transcript)
    if ready and settled:
        transcript.call_later(fill_window, transcript, fill)
        return
    if monotonic() > deadline:
        _finish(transcript, fill)
        return
    transcript.set_timer(_POLL_S, partial(_poll, transcript, fill, deadline, ready))


def _ready(transcript: ConsoleTranscript) -> bool:
    """Return True when nothing is mounting, placing or laying out the window.

    Textual 8 flags a widget whose relayout is on its way with
    ``_layout_required`` until its idle handler sends it to the screen, and
    the screen until its next update runs it.
    """
    if (
        transcript._refresh_lock.locked()
        or transcript._hydrating_scrollback
        or transcript._suppress_boundary_hydration
        or transcript._reveal_scroll_target is not None
    ):
        return False
    tail_start = transcript._hidden_tail_start_index()
    if tail_start > 0:
        _start, _end, owner, _owned = transcript._unit_span_at(
            transcript._messages, tail_start - 1
        )
        rows = transcript._row_widgets
        last = rows.get(f"assistant-turn:{owner}") or rows.get(f"message:{owner}")
        if last is None or last.parent is not transcript or not last.size.height:
            return False
    if getattr(transcript.screen, "_layout_required", False):
        return False
    return not any(
        getattr(node, "_layout_required", False)
        for node in transcript.walk_children(with_self=True)
    )


async def fill_window(
    transcript: ConsoleTranscript, fill: tuple[str, float | None]
) -> None:
    """Reveal the next batch of a paced window, then wait for it to settle.

    Stops when the planned window is mounted, when its last message left the
    transcript or the window moved past it, or when the reader follows the
    tail again.
    """
    if transcript._window_fill is not fill:
        return
    end_id = fill[0]
    if not transcript.is_mounted or transcript._raw_anchor_engaged():
        transcript._window_fill = None
        return
    if not _ready(transcript):
        _schedule(transcript)
        return
    messages = transcript._messages
    tail_start = transcript._hidden_tail_start_index()
    target = next(
        (
            index
            for index in range(tail_start, len(messages))
            if messages[index].id == end_id
        ),
        None,
    )
    if target is None or tail_start < transcript._first_visible_message_index():
        transcript._window_fill = None
        return
    async with transcript._refresh_lock:
        # The hydration latch: boundary hydration waits while the batch lands.
        transcript._hydrating_scrollback = True
        transcript._reveal_hidden_tail_through(
            _batch_end(transcript, tail_start, target + 1)
        )
        try:
            await transcript._reconcile_rows(transcript._transcript_rows())
        finally:
            transcript._hydrating_scrollback = False
    transcript._schedule_prune_check()
    if transcript._window_fill is not fill:
        return
    if transcript._hidden_tail_start_index() <= target:
        _schedule(transcript)
    else:
        _finish(transcript, fill)
