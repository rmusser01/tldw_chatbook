"""TASK-26003: content-stall watchdog for streamed provider responses.

A provider -- or a proxy/gateway in front of one -- can emit keep-alive or
heartbeat frames without ever producing new content, holding a run open until
the wall budget expires: the transport read timeout never fires because bytes
keep arriving. Keep-alive frames are filtered upstream (a decoded stream item is
dropped before it reaches the consumer), so a CONTENT-idle watchdog at the
consumption boundary is sufficient and does not need to see raw bytes:

- Only real items (content, thinking, tool-call deltas) reach the consumer, so
  only they reset the clock -- keep-alives inherently cannot (AC#2).
- It terminates a contentless stream regardless of transport bytes (AC#1),
  freeing the run. NOTE: it closes the item source, which unwinds the
  consumer -- but a sync provider whose worker thread is blocked inside a
  single wedged read is not aborted by that close (the read ends only when
  the connection drops). Bounding the RUN is the guarantee here; truly
  aborting a blocked provider read is TASK-30015.
- A slow-but-productive stream keeps yielding items, so it never trips (AC#5).

The stall is reported as a distinct exception so callers can tell it apart from
a network error and from a user cancel (AC#3).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator, Optional, TypeVar

_T = TypeVar("_T")

#: Default content-idle ceiling. Long enough that legitimate slow generation
#: (large responses, quiet thinking that still emits deltas) does not trip;
#: short enough that a wedged stream does not ride the wall budget.
DEFAULT_STALL_TIMEOUT_SECONDS = 90.0

#: Default number of stalls against one provider in a session before a warning
#: is surfaced instead of silently continuing (AC#4).
DEFAULT_STALL_WARN_THRESHOLD = 2


class StreamStallError(RuntimeError):
    """A stream produced no new content within the stall timeout (AC#3).

    Distinct from transport/network errors and from ``CancelledError`` so the
    caller can report a stall honestly rather than as a generic failure.
    """

    def __init__(self, timeout_seconds: float, provider: Optional[str] = None) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.provider = provider
        detail = f" (provider={provider})" if provider else ""
        super().__init__(
            f"stream produced no content for {self.timeout_seconds:g}s{detail}"
        )


async def watch_content_stalls(
    source: AsyncIterator[_T],
    timeout_seconds: Optional[float],
    *,
    provider: Optional[str] = None,
) -> AsyncIterator[_T]:
    """Yield items from ``source``, tripping on a content stall.

    Args:
        source: The async iterator of decoded stream items to guard.
        timeout_seconds: Maximum time to wait for the next item. ``None`` or a
            non-positive value disables the watchdog (pass-through).
        provider: Optional provider label carried on a raised
            :class:`StreamStallError`.

    Yields:
        Each item from ``source`` unchanged; every item resets the clock.

    Raises:
        StreamStallError: When no item arrives within ``timeout_seconds`` while
            the stream is still open. The source is closed first so the
            underlying stream/worker is cancelled.
    """
    it = source.__aiter__()
    if timeout_seconds is None or timeout_seconds <= 0:
        async for item in it:
            yield item
        return
    try:
        while True:
            try:
                item = await asyncio.wait_for(it.__anext__(), timeout_seconds)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                # No content for the whole window while the stream is still
                # open -> stall. Report it distinctly; the `finally` closes the
                # source, which unwinds an async-generator consumer. (A sync
                # provider blocked inside a wedged read is not aborted by that
                # close -- see TASK-30015; the run is freed regardless.)
                raise StreamStallError(timeout_seconds, provider)
            yield item
    finally:
        # A consumer that breaks/cancels out of the loop must not leak the
        # underlying stream; aclose() is idempotent and safe if already closed.
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()


class StallTracker:
    """Per-provider stall counter for one session (AC#4).

    Repeated stalls against the same provider surface a warning rather than
    silently continuing; a productive turn resets that provider's count.
    """

    def __init__(self, warn_threshold: int = DEFAULT_STALL_WARN_THRESHOLD) -> None:
        self._counts: dict[str, int] = {}
        self._warn_threshold = max(1, int(warn_threshold))

    def record_stall(self, provider: Optional[str]) -> bool:
        """Count one stall for ``provider``; return True at/over the threshold.

        Args:
            provider: The provider that stalled.

        Returns:
            True when this provider's stall count has reached the warn
            threshold, so the caller should surface a warning.
        """
        key = str(provider or "")
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        return count >= self._warn_threshold

    def reset(self, provider: Optional[str]) -> None:
        """Clear the stall count for ``provider`` after a productive turn."""
        self._counts.pop(str(provider or ""), None)

    def count(self, provider: Optional[str]) -> int:
        """Return the current stall count for ``provider``."""
        return self._counts.get(str(provider or ""), 0)


# --- Session-scoped stall tracking (AC#4) -------------------------------------
# The streaming adapter that catches a stall is per-turn, but "repeated stalls
# within a session" is cross-turn state. Keyed by session id here rather than
# threaded through the per-turn object; a productive turn prunes its entry, so
# only actively-stalling sessions hold one (small) tracker.

_SESSION_TRACKERS: dict[str, StallTracker] = {}
#: Bound on tracked sessions. A stalled run never reaches the reset path, so
#: without a cap a very long-lived process could accumulate one small tracker
#: per distinct stall-then-die session. Evict the oldest on overflow.
_MAX_TRACKED_SESSIONS = 512


def record_session_stall(
    session_id: Optional[str],
    provider: Optional[str],
    *,
    warn_threshold: int = DEFAULT_STALL_WARN_THRESHOLD,
) -> bool:
    """Record a stall for ``provider`` in ``session_id``; warn at threshold.

    Args:
        session_id: The owning session; ``None`` collapses to a shared bucket.
        provider: The provider that stalled.
        warn_threshold: Stalls before a warning is due (used on first sight of
            the session).

    Returns:
        True when this provider has stalled enough times in the session that a
        warning should be surfaced (AC#4).
    """
    key = str(session_id or "")
    tracker = _SESSION_TRACKERS.get(key)
    if tracker is None:
        if len(_SESSION_TRACKERS) >= _MAX_TRACKED_SESSIONS:
            # dict preserves insertion order; drop the oldest tracked session.
            oldest = next(iter(_SESSION_TRACKERS), None)
            if oldest is not None:
                _SESSION_TRACKERS.pop(oldest, None)
        tracker = StallTracker(warn_threshold)
        _SESSION_TRACKERS[key] = tracker
    return tracker.record_stall(provider)


def reset_session_stalls(
    session_id: Optional[str], provider: Optional[str] = None
) -> None:
    """Clear stall state after a productive turn.

    Args:
        session_id: The owning session.
        provider: Clear just this provider; ``None`` drops the whole session
            entry (a fully productive turn).
    """
    key = str(session_id or "")
    tracker = _SESSION_TRACKERS.get(key)
    if tracker is None:
        return
    if provider is None:
        _SESSION_TRACKERS.pop(key, None)
    else:
        tracker.reset(provider)
        if tracker.count(provider) == 0 and not tracker._counts:  # fully clear
            _SESSION_TRACKERS.pop(key, None)
