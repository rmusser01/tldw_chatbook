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


def _event(
    cursor: str,
    *,
    server_profile_id: str = "server-a",
    stream_name: str = "notifications",
    stream_instance_id: str = "workspace-1",
    entity_id: str | None = None,
    payload_hash: str | None = None,
) -> NormalizedEventRecord:
    entity_id = entity_id or cursor
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
        server_cursor=cursor,
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
