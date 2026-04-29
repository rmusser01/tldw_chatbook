"""Transport-pluggable observer for normalized realtime events."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from tldw_chatbook.Notifications.event_cursor_store import (
    CursorAdvanceResult,
    CursorAdvanceStatus,
    EventCursorStore,
)
from tldw_chatbook.runtime_policy.server_parity_models import (
    EventCursor,
    NormalizedEventRecord,
    SourceAuthority,
)


class EventStreamTransport(Protocol):
    def stream(self, cursor: EventCursor) -> Any:
        ...


class StaleCursorError(RuntimeError):
    """Raised by a transport when a cursor is too old and must be requeried."""


class UnsupportedCursorError(RuntimeError):
    """Raised by a transport when a stream cannot resume from a cursor."""


Handler = Callable[[NormalizedEventRecord], bool | Awaitable[bool]]
Backoff = Callable[[int], None | Awaitable[None]]


@dataclass(frozen=True, slots=True)
class EventObserverResult:
    handled_events: int
    reset: CursorAdvanceResult | None = None
    cancelled: bool = False


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def bounded_exponential_backoff(
    attempt: int,
    *,
    base_delay: float = 0.25,
    max_delay: float = 5.0,
) -> None:
    await asyncio.sleep(min(max_delay, base_delay * (2 ** max(0, attempt - 1))))


class EventObserver:
    def __init__(
        self,
        *,
        store: EventCursorStore,
        transport: EventStreamTransport,
        backoff: Backoff | None = None,
    ) -> None:
        self.store = store
        self.transport = transport
        self.backoff = backoff or bounded_exponential_backoff

    async def run(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        handler: Handler,
        cancel_event: asyncio.Event | None = None,
        max_events: int | None = None,
        max_reconnects: int = 0,
    ) -> EventObserverResult:
        handled_events = 0
        reconnects = 0
        last_reset: CursorAdvanceResult | None = None

        while True:
            if cancel_event is not None and cancel_event.is_set():
                return EventObserverResult(handled_events=handled_events, reset=last_reset, cancelled=True)

            cursor = self.store.get_cursor(
                source_authority=source_authority,
                server_profile_id=server_profile_id,
                stream_name=stream_name,
                stream_instance_id=stream_instance_id,
            )

            try:
                async for event in self.transport.stream(cursor):
                    if cancel_event is not None and cancel_event.is_set():
                        return EventObserverResult(
                            handled_events=handled_events,
                            reset=last_reset,
                            cancelled=True,
                        )
                    if self.store.is_duplicate_event(event):
                        continue

                    should_ack = bool(await _maybe_await(handler(event)))
                    handled_events += 1
                    if should_ack:
                        advance = self.store.acknowledge_event(event)
                        if advance.status is not CursorAdvanceStatus.STALE_RESET:
                            self.store.remember_event(event)

                    if max_events is not None and handled_events >= max_events:
                        return EventObserverResult(handled_events=handled_events, reset=last_reset)

                return EventObserverResult(handled_events=handled_events, reset=last_reset)
            except StaleCursorError:
                last_reset = self.store.reset_cursor(cursor, reason="stale_cursor")
            except UnsupportedCursorError:
                last_reset = self.store.reset_cursor(cursor, reason="unsupported_cursor")
            except Exception:
                if reconnects >= max_reconnects:
                    raise

            reconnects += 1
            if reconnects > max_reconnects:
                return EventObserverResult(handled_events=handled_events, reset=last_reset)
            await _maybe_await(self.backoff(reconnects))
