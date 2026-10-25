"""Bounded served-child projection publisher (ADR-244).

The App polls retained runtime revision witnesses at the existing cadence and
sends body-free invalidations through one bounded thread-safe queue and drainer.
Projection pressure cannot own command responses or stop native work. Legacy
approval/card methods remain compatibility helpers until the mobile cutover;
served App wiring no longer uses mounted callbacks as its state source.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_PENDING_RELAY_EVENTS = 64
DEFAULT_RUN_STATUS_POLL_INTERVAL_SECONDS = 0.5

_RunStateSource = Callable[[], Mapping[str, Any] | None]


def _timeout_seconds(raw: Any) -> int:
    """Coerce the console payload's timeout to the protocol's int form.

    The console payload carries a float (``0.0`` = no timeout, ADR-067);
    ``relay.approval.raised`` requires a non-bool ``int >= 0``.
    """

    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0
    try:
        return max(0, int(raw))
    except (OverflowError, ValueError):
        return 0


def _relay_calls(raw_calls: Any) -> list[dict[str, Any]]:
    """Map console approval rows down to the four relay-safe call fields.

    The protocol's payload registry rejects unknown top-level fields, so the
    publisher also keeps each row minimal: only what the phone's approval
    card renders (``llm_name``, ``call_id``, ``options``, ``arguments``).
    """

    if not isinstance(raw_calls, list):
        return []
    calls: list[dict[str, Any]] = []
    for row in raw_calls:
        if not isinstance(row, Mapping):
            continue
        calls.append(
            {
                "llm_name": str(row.get("llm_name") or ""),
                "call_id": row.get("call_id") if isinstance(row.get("call_id"), str) else None,
                "options": list(row.get("options") or []),
                "arguments": dict(row.get("arguments") or {}),
            }
        )
    return calls


class ServedRelayPublisher:
    """Forwards served-child Console approval rounds + run status to the parent broker."""

    def __init__(
        self,
        client: Any,
        *,
        max_pending: int = DEFAULT_MAX_PENDING_RELAY_EVENTS,
        poll_interval: float = DEFAULT_RUN_STATUS_POLL_INTERVAL_SECONDS,
    ) -> None:
        """Bind the publisher to the child control client (``None`` = no-op).

        Args:
            client: The child's ``CanvasControlClient`` (or ``None`` when
                the spawn environment carried no transport, making the
                relay inert).
            max_pending: Bound on queued relay events; excess is dropped.
            poll_interval: Run-status poll cadence once a run-state source
                is attached.
        """

        if type(max_pending) is not int or not 1 <= max_pending <= 1024:
            raise ValueError("invalid pending-relay limit")
        self._client = client
        self._max_pending = max_pending
        self._poll_interval = poll_interval
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: deque[tuple[str, dict[str, Any]]] = deque()
        self._queue_lock = threading.Lock()
        self._wake = asyncio.Event()
        self._wake_scheduled = False
        self._reset_required = False
        self._sequence = 0
        self._drainer: asyncio.Task[None] | None = None
        self._poller: asyncio.Task[None] | None = None
        self._busy = False
        # round ids of currently mounted approval rounds, newest last —
        # mirrors the TUI card slot (one deep) and the parent's capture.
        self._mounted_rounds: deque[str] = deque(maxlen=max_pending)
        self._last_run_status: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def attach(
        self,
        *,
        run_state_source: _RunStateSource | None = None,
        projection_source: Callable[[], Any] | None = None,
    ) -> None:
        """Start the drainer (and optional run-status poller) on this loop.

        Must be called from the event loop the client sends on (the app's
        ``on_mount``); idempotent. Publishing before ``attach`` is dropped,
        never raised — the relay is advisory to the child.

        Args:
            run_state_source: Returns the app-owned controller's current
                ``{session_id: run-state}`` mapping (or ``None`` while the
                lazy controller does not exist yet). This is the same
                run-state read seam the TUI polls for terminal transitions;
                the poller publishes only per-session status DELTAS.
        """

        if self._drainer is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._loop = loop
        self._drainer = asyncio.create_task(
            self._drain(), name="served-relay-publisher"
        )
        if projection_source is not None:
            self._poller = asyncio.create_task(
                self._poll_projection(projection_source),
                name="served-console-projection",
            )
        elif run_state_source is not None:
            self._poller = asyncio.create_task(
                self._poll_run_states(run_state_source),
                name="served-relay-run-status",
            )

    async def aclose(self) -> None:
        """Stop the drainer/poller tasks; safe to call more than once."""

        tasks = [task for task in (self._drainer, self._poller) if task is not None]
        self._drainer = None
        self._poller = None
        self._loop = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_idle(self, *, timeout: float = 1.0) -> None:
        """Wait until every queued event has been sent (test surface)."""

        queue = self._queue
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            # Always yield first so a just-scheduled put (call_soon_threadsafe)
            # lands before the emptiness check.
            await asyncio.sleep(0)
            if queue or self._reset_required or self._busy:
                if asyncio.get_running_loop().time() >= deadline:
                    return
                continue
            return

    # ------------------------------------------------------------------ #
    # Publish surface (fire-and-forget, never blocking, never raising)     #
    # ------------------------------------------------------------------ #

    def approval_raised(self, payload: Mapping[str, Any]) -> None:
        """Queue ``relay.approval.raised`` for one mounted console round."""

        round_id = payload.get("round_id")
        round_id = round_id if isinstance(round_id, str) else ""
        if not round_id:
            logger.debug("served relay drop approval raise without round_id")
            return
        self._enqueue(
            "relay.approval.raised",
            {
                "round_id": round_id,
                "session_id": (
                    payload.get("session_id")
                    if isinstance(payload.get("session_id"), str)
                    else ""
                ),
                "calls": _relay_calls(payload.get("calls")),
                "timeout_seconds": _timeout_seconds(payload.get("timeout_seconds")),
            },
        )
        with self._queue_lock:
            if round_id in self._mounted_rounds:
                self._mounted_rounds.remove(round_id)
            self._mounted_rounds.append(round_id)

    def approval_cleared(
        self, round_id: str | None = None, resolution: str = "cleared"
    ) -> None:
        """Queue ``relay.approval.cleared`` for one cleared console round.

        ``round_id=None`` clears the newest mounted round (the card slot's
        clear payload carries no id).
        """

        with self._queue_lock:
            resolved_round = round_id
            if resolved_round is None:
                resolved_round = (
                    self._mounted_rounds.pop() if self._mounted_rounds else ""
                )
            elif resolved_round in self._mounted_rounds:
                self._mounted_rounds.remove(resolved_round)
        if not resolved_round:
            return
        self._enqueue(
            "relay.approval.cleared",
            {"round_id": resolved_round, "resolution": resolution},
        )

    def run_status(self, session_id: str, status: str) -> None:
        """Queue ``relay.run.status`` for one session transition."""

        self._enqueue(
            "relay.run.status",
            {"session_id": str(session_id or ""), "status": str(status or "")},
        )

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _enqueue(self, message_type: str, payload: dict[str, Any]) -> None:
        if self._client is None:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with self._queue_lock:
            self._sequence += 1
            if len(self._queue) >= self._max_pending:
                self._reset_required = True
            else:
                self._queue.append((message_type, payload))
            if self._wake_scheduled:
                return
            self._wake_scheduled = True
        try:
            loop.call_soon_threadsafe(self._notify)
        except RuntimeError:
            with self._queue_lock:
                self._wake_scheduled = False

    def _notify(self) -> None:
        with self._queue_lock:
            self._wake_scheduled = False
        self._wake.set()

    async def _drain(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while True:
                with self._queue_lock:
                    if self._reset_required:
                        message_type = "console.changed"
                        payload = {"sequence": self._sequence, "reset_required": True}
                        self._reset_required = False
                    elif self._queue:
                        message_type, payload = self._queue.popleft()
                    else:
                        break
                self._busy = True
                try:
                    await self._client.send_relay_event(message_type, payload)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("served relay send failed type=%s", message_type)
                finally:
                    self._busy = False

    async def _poll_projection(self, source: Callable[[], Any]) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                projection = source()
                if projection is not None:
                    sequence = projection.poll_changes()
                    if sequence is not None:
                        self._enqueue("console.changed", {"sequence": sequence})
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - projection failures cannot stop control
                logger.debug("served Console projection poll unavailable")

    async def _poll_run_states(self, source: _RunStateSource) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                states = source()
                if not states:
                    continue
                for session_id, state in list(states.items()):
                    status = getattr(state, "status", None)
                    value = getattr(status, "value", status)
                    if not isinstance(value, str) or not isinstance(session_id, str):
                        continue
                    previous = self._last_run_status.get(session_id)
                    self._last_run_status[session_id] = value
                    if previous is None or previous == value:
                        continue
                    self.run_status(session_id, value)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("served relay run-status poll failed", exc_info=True)


def wrap_pending_approval_hook(
    publisher: ServedRelayPublisher | None, original: Callable[..., Any]
) -> Callable[..., Any]:
    """Wrap one ``set_pending_approval`` binding call-through-first.

    The wrapper exists ONLY where served mode was detected (the app returns
    the original untouched otherwise, so native runs never wrap). It always
    runs the original binding first — TUI card behavior stays
    byte-identical, return value included — then mirrors the card
    transition onto the relay: a payload whose original returned TRUTHY
    (the round mounted a card) raises the round; a falsy return (the
    round parked on a background session, or nothing mounted) publishes
    nothing — the relay mirrors the mounted card, not the card request; a
    ``None`` payload clears the newest mount. Relay failures cannot reach
    the caller: the publisher never raises.
    """

    if publisher is None:
        return original

    def _relay_pending_approval(payload: Any) -> Any:
        result = original(payload)
        if payload is None:
            publisher.approval_cleared(None, resolution="cleared")
        elif result:
            publisher.approval_raised(payload)
        return result

    return _relay_pending_approval


__all__ = [
    "DEFAULT_MAX_PENDING_RELAY_EVENTS",
    "DEFAULT_RUN_STATUS_POLL_INTERVAL_SECONDS",
    "ServedRelayPublisher",
    "wrap_pending_approval_hook",
]
