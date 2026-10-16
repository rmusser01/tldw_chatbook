"""Served-child publisher for the desktop→phone approval relay (ADR-236).

A textual-serve child App forwards its Console approval rounds and
run-status transitions to the parent broker through the child's
:class:`~tldw_chatbook.Canvas.control_protocol.CanvasControlClient`.
The module is importable without a Textual app context: the app wires it,
everything else talks to it through three fire-and-forget methods.

Hard invariants (plan 2026-10-15-served-mobile-phase2c, Global
Constraints):

- Publishing must NEVER block or raise into a caller. Approval callbacks
  run on the agent worker thread; run-status observation runs on the event
  loop. Every method enqueues onto a bounded internal queue through
  ``call_soon_threadsafe`` (dropping on overflow or before attach) and a
  single drainer task performs the async client send, swallowing and
  logging (ids only) every exception.
- The relay mirrors the TUI card lifecycle: ``approval_raised`` records the
  mounted round, a ``None``-payload clear pops the newest mount and
  publishes ``resolution="cleared"`` — the same one-deep card-slot
  semantics the parent's mobile capture uses.
"""

from __future__ import annotations

import asyncio
import logging
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

        self._client = client
        self._max_pending = max_pending
        self._poll_interval = poll_interval
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] | None = None
        self._drainer: asyncio.Task[None] | None = None
        self._poller: asyncio.Task[None] | None = None
        self._busy = False
        # round ids of currently mounted approval rounds, newest last —
        # mirrors the TUI card slot (one deep) and the parent's capture.
        self._mounted_rounds: list[str] = []
        self._last_run_status: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def attach(
        self, *, run_state_source: _RunStateSource | None = None
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
        self._queue = asyncio.Queue(maxsize=self._max_pending)
        self._drainer = asyncio.create_task(
            self._drain(), name="served-relay-publisher"
        )
        if run_state_source is not None:
            self._poller = asyncio.create_task(
                self._poll_run_states(run_state_source),
                name="served-relay-run-status",
            )

    async def aclose(self) -> None:
        """Stop the drainer/poller tasks; safe to call more than once."""

        tasks = [task for task in (self._drainer, self._poller) if task is not None]
        self._drainer = None
        self._poller = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_idle(self, *, timeout: float = 1.0) -> None:
        """Wait until every queued event has been sent (test surface)."""

        queue = self._queue
        if queue is None:
            return
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            # Always yield first so a just-scheduled put (call_soon_threadsafe)
            # lands before the emptiness check.
            await asyncio.sleep(0)
            if not queue.empty() or self._busy:
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

        resolved_round = round_id
        if resolved_round is None:
            resolved_round = self._mounted_rounds.pop() if self._mounted_rounds else ""
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
        queue = self._queue
        if loop is None or queue is None or loop.is_closed():
            logger.debug(
                "served relay drop type=%s reason=not_attached", message_type
            )
            return

        def _put() -> None:
            try:
                queue.put_nowait((message_type, payload))
            except asyncio.QueueFull:
                logger.warning(
                    "served relay drop type=%s reason=queue_full", message_type
                )

        try:
            loop.call_soon_threadsafe(_put)
        except RuntimeError:
            logger.debug(
                "served relay drop type=%s reason=closed_loop", message_type
            )

    async def _drain(self) -> None:
        queue = self._queue
        assert queue is not None
        while True:
            message_type, payload = await queue.get()
            self._busy = True
            try:
                await self._client.send_relay_event(message_type, payload)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "served relay send failed type=%s round_or_session=%s",
                    message_type,
                    payload.get("round_id") or payload.get("session_id") or "",
                )
            finally:
                self._busy = False

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
    transition onto the relay (a payload raises the round; ``None`` clears
    the newest mount). Relay failures cannot reach the caller: the
    publisher never raises.
    """

    if publisher is None:
        return original

    def _relay_pending_approval(payload: Any) -> Any:
        result = original(payload)
        if payload is None:
            publisher.approval_cleared(None, resolution="cleared")
        else:
            publisher.approval_raised(payload)
        return result

    return _relay_pending_approval


__all__ = [
    "DEFAULT_MAX_PENDING_RELAY_EVENTS",
    "DEFAULT_RUN_STATUS_POLL_INTERVAL_SECONDS",
    "ServedRelayPublisher",
    "wrap_pending_approval_hook",
]
