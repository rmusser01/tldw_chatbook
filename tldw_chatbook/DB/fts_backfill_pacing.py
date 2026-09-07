"""Pacing primitives shared by the background FTS backfills.

Extracted in TASK-22215 from ``DB/chachanotes_fts_backfill.py``, where
TASK-22200 first built them, so the *other* boot-time backfill
(``Subscriptions/fts_backfill.py``, task-688) yields to foreground work by the
same rules rather than by a second, drifting copy of them.

Both drivers have the same shape: a resumable loop of ``BEGIN IMMEDIATE``
chunks whose frontier lives in the database, running on a boot-time thread
worker that nothing waits on. Two properties matter and both live here:

* **Yield between chunks.** Back-to-back chunks convoy against every
  foreground UI write (also ``BEGIN IMMEDIATE``), so a foreground writer waits
  out chunk after chunk. A pause after each chunk that did work bounds the
  backfill's share of the write lock to a few percent and lets a foreground
  writer take it inside the gap.
* **Abort-checked waits.** A cancel that only sets a flag cannot cut a plain
  ``time.sleep``, so shutdown would wait out every remaining pause. Waits are
  sliced at :data:`ABORT_POLL_SECONDS` and re-check the caller's abort flag
  between slices; stopping is always safe because the frontier is in the
  database, not in the loop.
"""

from __future__ import annotations

from typing import Callable, Optional

__all__ = [
    "INTER_CHUNK_PAUSE_SECONDS",
    "ABORT_POLL_SECONDS",
    "interruptible_sleep",
]

#: Pause between chunks. Chosen against measurement, not taste (TASK-22200): a
#: default 500-row chunk of typical chat text holds the write lock for
#: single-digit milliseconds, so 0.1 s caps the backfill's lock duty cycle at a
#: few percent while still finishing a 100k-row history in tens of seconds of
#: added wall time (background thread; nobody is waiting on it).
INTER_CHUNK_PAUSE_SECONDS = 0.1

#: Slice size for abort-checked sleeps. Also the worst-case extra shutdown
#: latency a sleeping backfill adds once its worker is cancelled.
ABORT_POLL_SECONDS = 0.05


def interruptible_sleep(
    seconds: float,
    should_abort: Optional[Callable[[], bool]],
    sleep: Callable[[float], None],
) -> bool:
    """Sleep ``seconds``, abort-checked. Returns True if aborted.

    Without ``should_abort`` this is a single ``sleep`` call (keeps injected
    recorders 1:1 with pauses in tests). With it, the wait is sliced into
    :data:`ABORT_POLL_SECONDS` steps so a cancelled worker stops within one
    slice instead of finishing the whole pause.

    Args:
        seconds: How long to wait; ``<= 0`` only polls the abort flag.
        should_abort: Optional predicate polled between slices.
        sleep: The sleep implementation (injected by tests).

    Returns:
        True if ``should_abort`` asked to stop, False otherwise.
    """
    if seconds <= 0:
        return bool(should_abort and should_abort())
    if should_abort is None:
        sleep(seconds)
        return False
    remaining = seconds
    while remaining > 0:
        if should_abort():
            return True
        step = min(ABORT_POLL_SECONDS, remaining)
        sleep(step)
        remaining -= step
    return should_abort()
