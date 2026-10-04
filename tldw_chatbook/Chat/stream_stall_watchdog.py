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
import threading
from typing import AsyncIterator, Optional, TypeVar

_T = TypeVar("_T")

#: Default content-idle ceiling. Long enough that legitimate slow generation
#: (large responses, quiet thinking that still emits deltas) does not trip;
#: short enough that a wedged stream does not ride the wall budget.
DEFAULT_STALL_TIMEOUT_SECONDS = 90.0

#: Default number of stalls against one provider in a session before a warning
#: is surfaced instead of silently continuing (AC#4).
DEFAULT_STALL_WARN_THRESHOLD = 2

#: TASK-34100.5 AC#5: a self-hosted model's FIRST token can take minutes (a
#: cold load plus CPU prompt processing). Gaps between tokens keep the stall
#: window; only the wait for the first one is longer.
DEFAULT_LOCAL_FIRST_TOKEN_TIMEOUT_SECONDS = 300.0


class StreamStallError(RuntimeError):
    """A stream produced no new content within the stall timeout (AC#3).

    Distinct from transport/network errors and from ``CancelledError`` so the
    caller can report a stall honestly rather than as a generic failure.

    Args:
        timeout_seconds: The content-idle window that elapsed with no new item.
        provider: Optional provider label for the message.
    """

    def __init__(
        self,
        timeout_seconds: float,
        provider: Optional[str] = None,
        *,
        first_token: bool = False,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.provider = provider
        #: True when NO content ever arrived (the first-token window expired).
        self.first_token = bool(first_token)
        detail = f" (provider={provider})" if provider else ""
        what = "no first token" if self.first_token else "no content"
        super().__init__(
            f"stream produced {what} for {self.timeout_seconds:g}s{detail}"
        )


def first_token_timeout_seconds(
    provider: Optional[str], *, stall_timeout: float
) -> Optional[float]:
    """The window for a stream's FIRST item (TASK-34100.5 AC#5).

    Precedence follows the project rule env -> config.toml -> default:
    ``TLDW_FIRST_TOKEN_TIMEOUT_SECONDS``, then ``[chat_defaults]
    first_token_timeout_seconds``, then
    :data:`DEFAULT_LOCAL_FIRST_TOKEN_TIMEOUT_SECONDS` for self-hosted
    (keyless) providers and the stall window for everything else. Never
    shorter than the stall window; ``None`` when the watchdog is disabled.

    Args:
        provider: Provider key of the call.
        stall_timeout: The between-items stall window in force.

    Returns:
        Seconds to wait for the first item, or ``None`` (watchdog off).
    """
    import math
    import os

    if stall_timeout is None or stall_timeout <= 0:
        return None
    raw: object = os.environ.get("TLDW_FIRST_TOKEN_TIMEOUT_SECONDS")
    if raw is None or not str(raw).strip():
        from tldw_chatbook.config import get_cli_setting

        raw = get_cli_setting("chat_defaults", "first_token_timeout_seconds", None)
    value: Optional[float] = None
    if raw is not None and str(raw).strip():
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            value = None
        if value is not None and not math.isfinite(value):
            value = None
    if value is None:
        from tldw_chatbook.Chat.provider_readiness import is_self_hosted_provider

        local = is_self_hosted_provider(provider)
        value = DEFAULT_LOCAL_FIRST_TOKEN_TIMEOUT_SECONDS if local else stall_timeout
    return max(float(value), float(stall_timeout))


def _configured_stall_window() -> float:
    """The gap window in force: env, then config, then the default.

    The same precedence as ``console_agent_bridge._stall_timeout_seconds``,
    read here so the transport layer need not import the bridge.
    """
    import math
    import os

    raw: object = os.environ.get("TLDW_STREAM_STALL_TIMEOUT_SECONDS")
    if raw is None or not str(raw).strip():
        from tldw_chatbook.config import get_cli_setting

        raw = get_cli_setting(
            "chat_defaults", "stream_stall_timeout_seconds", DEFAULT_STALL_TIMEOUT_SECONDS
        )
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_STALL_TIMEOUT_SECONDS
    return value if math.isfinite(value) else DEFAULT_STALL_TIMEOUT_SECONDS


#: How much longer than the first-token window a self-hosted server's HTTP
#: read timeout runs, so the watchdog -- not the transport -- decides.
FIRST_TOKEN_READ_TIMEOUT_MARGIN_SECONDS = 30.0


def self_hosted_read_timeout(configured: float) -> float:
    """A self-hosted chat request's HTTP read timeout (review round 1, B-F1).

    Live: a Custom OpenAI-compatible endpoint's 120 s read timeout fired
    before the 300 s first-token window, so a cold model's slow first token
    surfaced as a retryable network error instead of the first-token copy.
    The read timeout is never shorter than the first-token window plus
    :data:`FIRST_TOKEN_READ_TIMEOUT_MARGIN_SECONDS`.

    Args:
        configured: The handler's configured read timeout in seconds.

    Returns:
        ``configured`` raised to the floor (unchanged when the floor cannot
        be read).
    """
    try:
        window = first_token_timeout_seconds(
            "llama_cpp", stall_timeout=_configured_stall_window()
        )
    except Exception:  # noqa: BLE001 -- the floor is best effort
        return configured
    if window is None:
        return configured
    return max(configured, window + FIRST_TOKEN_READ_TIMEOUT_MARGIN_SECONDS)


async def watch_content_stalls(
    source: AsyncIterator[_T],
    timeout_seconds: Optional[float],
    *,
    provider: Optional[str] = None,
    first_item_timeout_seconds: Optional[float] = None,
) -> AsyncIterator[_T]:
    """Yield items from ``source``, tripping on a content stall.

    Args:
        source: The async iterator of decoded stream items to guard.
        timeout_seconds: Maximum time to wait for the next item. ``None`` or a
            non-positive value disables the watchdog (pass-through).
        provider: Optional provider label carried on a raised
            :class:`StreamStallError`.
        first_item_timeout_seconds: A longer window for the FIRST item only
            (a cold local model); ``None`` uses ``timeout_seconds`` for it.
            A whitespace-only text item does not end it.

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
    first = True
    try:
        while True:
            window = (
                first_item_timeout_seconds
                if first and first_item_timeout_seconds
                else timeout_seconds
            )
            try:
                item = await asyncio.wait_for(it.__anext__(), window)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                # No content for the whole window while the stream is still
                # open -> stall. Report it distinctly; the `finally` closes the
                # source, which unwinds an async-generator consumer. (A sync
                # provider blocked inside a wedged read is not aborted by that
                # close -- see TASK-30015; the run is freed regardless.)
                raise StreamStallError(window, provider, first_token=first)
            if not (isinstance(item, str) and not item.strip()):
                # Review round 1 (A-F5): a blank delta from a cold server
                # still reading the prompt is not the answer starting.
                first = False
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

    Args:
        warn_threshold: Stalls against one provider before a warning is due
            (coerced to at least 1).
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
        """Clear the stall count for ``provider`` after a productive turn.

        Args:
            provider: The provider whose count to clear.
        """
        self._counts.pop(str(provider or ""), None)

    def count(self, provider: Optional[str]) -> int:
        """Return the current stall count for ``provider``.

        Args:
            provider: The provider to look up.

        Returns:
            The number of stalls recorded for ``provider`` (0 if none).
        """
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
#: Guards the process-global registry: the bridge runs the primary turn and each
#: fleet child on separate threads, all calling the record/reset helpers.
_REGISTRY_LOCK = threading.Lock()


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
    with _REGISTRY_LOCK:
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
    with _REGISTRY_LOCK:
        tracker = _SESSION_TRACKERS.get(key)
        if tracker is None:
            return
        if provider is None:
            _SESSION_TRACKERS.pop(key, None)
        else:
            tracker.reset(provider)
            if not tracker._counts:  # fully clear
                _SESSION_TRACKERS.pop(key, None)
