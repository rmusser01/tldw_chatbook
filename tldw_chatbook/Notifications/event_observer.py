"""Transport-pluggable observer for normalized realtime events."""

from __future__ import annotations

import asyncio
import contextlib
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


class _CancelledByEvent(Exception):
    pass


async def _await_with_cancel(awaitable: Awaitable[Any], cancel_event: asyncio.Event | None) -> Any:
    if cancel_event is None:
        return await awaitable
    if cancel_event.is_set():
        raise _CancelledByEvent

    task = asyncio.ensure_future(awaitable)
    cancel_task = asyncio.create_task(cancel_event.wait())
    done, _ = await asyncio.wait({task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
    if task in done:
        cancel_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancel_task
        return task.result()

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    raise _CancelledByEvent


async def _maybe_await_with_cancel(value: Any, cancel_event: asyncio.Event | None) -> Any:
    if cancel_event is not None and cancel_event.is_set():
        raise _CancelledByEvent
    if inspect.isawaitable(value):
        return await _await_with_cancel(value, cancel_event)
    return value


async def _close_iterator(iterator: Any) -> None:
    aclose = getattr(iterator, "aclose", None)
    if callable(aclose):
        await _maybe_await(aclose())


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
            expected_ack_cursor = cursor.cursor

            try:
                stream = self.transport.stream(cursor)
                iterator = stream.__aiter__()
                try:
                    while True:
                        try:
                            event = await _await_with_cancel(anext(iterator), cancel_event)
                        except StopAsyncIteration:
                            return EventObserverResult(handled_events=handled_events, reset=last_reset)

                        if cancel_event is not None and cancel_event.is_set():
                            return EventObserverResult(
                                handled_events=handled_events,
                                reset=last_reset,
                                cancelled=True,
                            )
                        if self.store.is_duplicate_event(event):
                            continue

                        should_ack = bool(await _maybe_await_with_cancel(handler(event), cancel_event))
                        handled_events += 1
                        if should_ack:
                            advance = self.store.acknowledge_event(event, expected_cursor=expected_ack_cursor)
                            if advance.status is CursorAdvanceStatus.ADVANCED:
                                self.store.remember_event(event)
                                expected_ack_cursor = advance.cursor.cursor
                            elif advance.status is not CursorAdvanceStatus.STALE_RESET:
                                self.store.remember_event(event)
                            else:
                                last_reset = advance

                        if max_events is not None and handled_events >= max_events:
                            return EventObserverResult(handled_events=handled_events, reset=last_reset)
                finally:
                    await _close_iterator(iterator)

            except _CancelledByEvent:
                return EventObserverResult(handled_events=handled_events, reset=last_reset, cancelled=True)
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
            try:
                await _maybe_await_with_cancel(self.backoff(reconnects), cancel_event)
            except _CancelledByEvent:
                return EventObserverResult(handled_events=handled_events, reset=last_reset, cancelled=True)
