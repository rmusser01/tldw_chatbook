import asyncio

import pytest

from tldw_chatbook.Notifications.event_cursor_store import (
    CursorAdvanceStatus,
    EventCursorStore,
)
from tldw_chatbook.Notifications.event_observer import (
    EventObserver,
    StaleCursorError,
    UnsupportedCursorError,
)
from tldw_chatbook.runtime_policy.server_parity_models import EventCursor, NormalizedEventRecord

_USE_CURSOR = object()


def _event(
    cursor: str | None,
    *,
    server_profile_id: str = "server-a",
    stream_name: str = "notifications",
    stream_instance_id: str = "workspace-1",
    entity_id: str | None = None,
    payload_hash: str | None = None,
    server_cursor: str | None | object = _USE_CURSOR,
) -> NormalizedEventRecord:
    entity_id = entity_id or str(cursor)
    payload_hash = payload_hash or f"hash-{cursor}"
    return NormalizedEventRecord(
        source_authority="server",
        server_profile_id=server_profile_id,
        stream_name=stream_name,
        stream_instance_id=stream_instance_id,
        event_kind="notification.created",
        entity_ref={"id": entity_id},
        payload_hash=payload_hash,
        event_id=f"{entity_id}:{payload_hash}",
        server_cursor=cursor if server_cursor is _USE_CURSOR else server_cursor,
        transport_type="sse",
    )


class FakeTransport:
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.cursors: list[EventCursor] = []

    async def stream(self, cursor):
        self.cursors.append(cursor)
        script = self.scripts.pop(0)
        if isinstance(script, BaseException):
            raise script
        for item in script:
            if isinstance(item, BaseException):
                raise item
            await asyncio.sleep(0)
            yield item


class RecordingBackoff:
    def __init__(self):
        self.calls = []

    async def __call__(self, attempt):
        self.calls.append(attempt)


class IdleTransport:
    def __init__(self):
        self.started = asyncio.Event()

    async def stream(self, cursor):
        self.started.set()
        await asyncio.Event().wait()
        if False:
            yield _event("never")


class SlowBackoff:
    def __init__(self):
        self.started = asyncio.Event()

    async def __call__(self, attempt):
        self.started.set()
        await asyncio.Event().wait()


class CloseableAsyncIterator:
    def __init__(self, items):
        self.items = list(items)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        await asyncio.sleep(0)
        return self.items.pop(0)

    async def aclose(self):
        self.closed = True


class CloseableIdleIterator:
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.started.set()
        await asyncio.Event().wait()
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


class CloseableTransport:
    def __init__(self, iterator):
        self.iterator = iterator

    def stream(self, cursor):
        return self.iterator


@pytest.mark.asyncio
async def test_observer_reconnects_with_injectable_backoff_and_resumes_cursor():
    store = EventCursorStore()
    backoff = RecordingBackoff()
    transport = FakeTransport([
        [_event("cursor-1"), RuntimeError("disconnect")],
        [_event("cursor-2")],
    ])
    observed = []
    observer = EventObserver(store=store, transport=transport, backoff=backoff)

    await observer.run(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        handler=lambda event: observed.append(event) or True,
        max_events=2,
        max_reconnects=1,
    )

    assert [event.server_cursor for event in observed] == ["cursor-1", "cursor-2"]
    assert [cursor.cursor for cursor in transport.cursors] == [None, "cursor-1"]
    assert backoff.calls == [1]


@pytest.mark.asyncio
async def test_observer_dedupes_duplicate_reconnect_event():
    store = EventCursorStore()
    transport = FakeTransport([
        [_event("cursor-1"), RuntimeError("disconnect")],
        [_event("cursor-1"), _event("cursor-2")],
    ])
    observed = []

    await EventObserver(store=store, transport=transport, backoff=lambda attempt: None).run(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        handler=lambda event: observed.append(event) or True,
        max_events=2,
        max_reconnects=1,
    )

    assert [event.server_cursor for event in observed] == ["cursor-1", "cursor-2"]


@pytest.mark.asyncio
async def test_unacknowledged_duplicate_after_reconnect_is_retried_and_can_advance_cursor():
    store = EventCursorStore()
    duplicate = _event("cursor-1")
    transport = FakeTransport([
        [duplicate, RuntimeError("disconnect")],
        [duplicate],
    ])
    observed = []
    acknowledgements = iter([False, True])

    await EventObserver(store=store, transport=transport, backoff=lambda attempt: None).run(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        handler=lambda event: observed.append(event) or next(acknowledgements),
        max_events=2,
        max_reconnects=1,
    )

    cursor = store.get_cursor(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )
    assert [event.server_cursor for event in observed] == ["cursor-1", "cursor-1"]
    assert cursor.cursor == "cursor-1"


@pytest.mark.asyncio
async def test_no_cursor_event_after_existing_cursor_does_not_clear_expected_cursor():
    store = EventCursorStore()
    store.acknowledge_event(_event("cursor-start"))
    transport = FakeTransport([[
        _event("no-cursor", server_cursor=None),
        _event("cursor-next"),
    ]])
    observed = []

    result = await EventObserver(store=store, transport=transport).run(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        handler=lambda event: observed.append(event) or True,
        max_events=2,
    )

    cursor = store.get_cursor(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )
    assert result.reset is None
    assert [event.server_cursor for event in observed] == [None, "cursor-next"]
    assert cursor.cursor == "cursor-next"


@pytest.mark.asyncio
async def test_observer_uses_stream_start_cursor_to_reject_stale_acknowledgement():
    store = EventCursorStore()
    older_event = _event("cursor-old")
    newer_event = _event("cursor-new")
    transport = FakeTransport([[older_event]])
    observed = []

    def handler(event):
        observed.append(event)
        store.acknowledge_event(newer_event)
        return True

    result = await EventObserver(store=store, transport=transport).run(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        handler=handler,
        max_events=1,
    )

    cursor = store.get_cursor(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )
    assert [event.server_cursor for event in observed] == ["cursor-old"]
    assert result.reset is not None
    assert result.reset.status is CursorAdvanceStatus.STALE_RESET
    assert cursor.cursor is None


@pytest.mark.asyncio
async def test_observer_does_not_advance_cursor_for_unacknowledged_events():
    store = EventCursorStore()
    transport = FakeTransport([[_event("cursor-1")]])

    await EventObserver(store=store, transport=transport).run(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        handler=lambda event: False,
        max_events=1,
    )

    cursor = store.get_cursor(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )
    assert cursor.cursor is None


@pytest.mark.asyncio
async def test_observer_returns_typed_reset_result_for_stale_cursor_and_requeries():
    store = EventCursorStore()
    store.acknowledge_event(_event("cursor-old"))
    transport = FakeTransport([
        StaleCursorError("expired"),
        [_event("cursor-new")],
    ])
    observed = []

    result = await EventObserver(store=store, transport=transport, backoff=lambda attempt: None).run(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        handler=lambda event: observed.append(event) or True,
        max_events=1,
        max_reconnects=1,
    )

    assert result.reset is not None
    assert result.reset.status is CursorAdvanceStatus.STALE_RESET
    assert [cursor.cursor for cursor in transport.cursors] == ["cursor-old", None]
    assert [event.server_cursor for event in observed] == ["cursor-new"]


@pytest.mark.asyncio
async def test_unsupported_cursor_stream_does_not_reuse_another_stream_cursor():
    store = EventCursorStore()
    store.acknowledge_event(_event("cursor-notifications", stream_name="notifications"))
    transport = FakeTransport([
        UnsupportedCursorError("cursor unsupported"),
        [_event("cursor-jobs", stream_name="jobs")],
    ])
    observed = []

    await EventObserver(store=store, transport=transport, backoff=lambda attempt: None).run(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="jobs",
        stream_instance_id="workspace-1",
        handler=lambda event: observed.append(event) or True,
        max_events=1,
        max_reconnects=1,
    )

    assert [cursor.cursor for cursor in transport.cursors] == [None, None]
    assert [event.stream_name for event in observed] == ["jobs"]


@pytest.mark.asyncio
async def test_active_server_switch_and_credential_clear_cancel_observation():
    store = EventCursorStore()
    event = _event("cursor-1")
    cancel = asyncio.Event()

    async def handler(seen):
        cancel.set()
        return True

    await EventObserver(store=store, transport=FakeTransport([[event]])).run(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        handler=handler,
        cancel_event=cancel,
        max_events=10,
    )

    assert cancel.is_set()
    assert store.get_cursor(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "cursor-1"


@pytest.mark.asyncio
async def test_cancel_event_wakes_idle_stream_wait():
    cancel = asyncio.Event()
    transport = IdleTransport()
    task = asyncio.create_task(
        EventObserver(store=EventCursorStore(), transport=transport).run(
            source_authority="server",
            server_profile_id="server-a",
            stream_name="notifications",
            stream_instance_id="workspace-1",
            handler=lambda event: True,
            cancel_event=cancel,
        )
    )
    await asyncio.wait_for(transport.started.wait(), timeout=0.2)

    cancel.set()
    result = await asyncio.wait_for(task, timeout=0.2)

    assert result.cancelled is True


@pytest.mark.asyncio
async def test_cancel_event_wakes_pending_handler_without_acknowledging_event():
    cancel = asyncio.Event()
    handler_started = asyncio.Event()
    store = EventCursorStore()

    async def handler(event):
        handler_started.set()
        await asyncio.Event().wait()
        return True

    task = asyncio.create_task(
        EventObserver(store=store, transport=FakeTransport([[_event("cursor-1")]])).run(
            source_authority="server",
            server_profile_id="server-a",
            stream_name="notifications",
            stream_instance_id="workspace-1",
            handler=handler,
            cancel_event=cancel,
        )
    )
    await asyncio.wait_for(handler_started.wait(), timeout=0.2)

    cancel.set()
    result = await asyncio.wait_for(task, timeout=0.2)

    cursor = store.get_cursor(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )
    assert result.cancelled is True
    assert cursor.cursor is None


@pytest.mark.asyncio
async def test_stream_iterator_is_closed_on_max_events_return():
    iterator = CloseableAsyncIterator([_event("cursor-1"), _event("cursor-2")])

    await EventObserver(store=EventCursorStore(), transport=CloseableTransport(iterator)).run(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        handler=lambda event: True,
        max_events=1,
    )

    assert iterator.closed is True


@pytest.mark.asyncio
async def test_stream_iterator_is_closed_on_cancelled_wait():
    cancel = asyncio.Event()
    iterator = CloseableIdleIterator()
    task = asyncio.create_task(
        EventObserver(store=EventCursorStore(), transport=CloseableTransport(iterator)).run(
            source_authority="server",
            server_profile_id="server-a",
            stream_name="notifications",
            stream_instance_id="workspace-1",
            handler=lambda event: True,
            cancel_event=cancel,
        )
    )
    await asyncio.wait_for(iterator.started.wait(), timeout=0.2)

    cancel.set()
    result = await asyncio.wait_for(task, timeout=0.2)

    assert result.cancelled is True
    assert iterator.closed is True


@pytest.mark.asyncio
async def test_cancel_event_wakes_backoff_wait():
    cancel = asyncio.Event()
    backoff = SlowBackoff()
    task = asyncio.create_task(
        EventObserver(
            store=EventCursorStore(),
            transport=FakeTransport([RuntimeError("disconnect"), []]),
            backoff=backoff,
        ).run(
            source_authority="server",
            server_profile_id="server-a",
            stream_name="notifications",
            stream_instance_id="workspace-1",
            handler=lambda event: True,
            cancel_event=cancel,
            max_reconnects=1,
        )
    )
    await asyncio.wait_for(backoff.started.wait(), timeout=0.2)

    cancel.set()
    result = await asyncio.wait_for(task, timeout=0.2)

    assert result.cancelled is True
