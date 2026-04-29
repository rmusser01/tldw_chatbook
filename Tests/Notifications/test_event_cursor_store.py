from tldw_chatbook.Notifications.event_cursor_store import (
    CursorAdvanceStatus,
    EventCursorStore,
)
from tldw_chatbook.runtime_policy.server_parity_models import NormalizedEventRecord


def _event(
    *,
    server_profile_id: str = "server-a",
    stream_name: str = "notifications",
    stream_instance_id: str = "workspace-1",
    event_kind: str = "notification.created",
    entity_id: str = "n1",
    payload_hash: str = "hash-1",
    server_cursor: str | None = "cursor-1",
    event_id: str | None = None,
    include_event_id: bool = True,
) -> NormalizedEventRecord:
    resolved_event_id = event_id if event_id is not None else f"{entity_id}:{payload_hash}"
    return NormalizedEventRecord(
        source_authority="server",
        server_profile_id=server_profile_id,
        stream_name=stream_name,
        stream_instance_id=stream_instance_id,
        event_kind=event_kind,
        entity_ref={"id": entity_id},
        payload_hash=payload_hash,
        event_id=resolved_event_id if include_event_id else None,
        server_cursor=server_cursor,
        transport_type="sse",
    )


def test_cursor_store_isolates_cursors_per_server_profile():
    store = EventCursorStore()
    event_a = _event(server_profile_id="server-a", server_cursor="cursor-a")
    event_b = _event(server_profile_id="server-b", server_cursor="cursor-b")

    store.acknowledge_event(event_a)
    store.acknowledge_event(event_b)

    cursor_a = store.get_cursor(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )
    cursor_b = store.get_cursor(
        source_authority="server",
        server_profile_id="server-b",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )

    assert cursor_a.cursor == "cursor-a"
    assert cursor_b.cursor == "cursor-b"


def test_cursor_only_advances_after_acknowledgement():
    store = EventCursorStore()
    event = _event(server_cursor="cursor-2")

    assert store.remember_event(event).is_duplicate is False
    cursor_before_ack = store.get_cursor(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )
    advance = store.acknowledge_event(event)

    assert cursor_before_ack.cursor is None
    assert advance.status is CursorAdvanceStatus.ADVANCED
    assert advance.cursor.cursor == "cursor-2"


def test_acknowledgement_reports_stale_reset_when_expected_cursor_no_longer_matches():
    store = EventCursorStore()
    store.acknowledge_event(_event(server_cursor="cursor-current"))

    result = store.acknowledge_event(_event(server_cursor="cursor-next"), expected_cursor="cursor-old")

    assert result.status is CursorAdvanceStatus.STALE_RESET
    assert result.cursor.cursor is None
    assert result.reason == "cursor_mismatch"


def test_dedupe_retention_is_bounded_and_evicted_entries_can_be_seen_again():
    store = EventCursorStore(dedupe_retention=2)
    first = _event(entity_id="n1", payload_hash="hash-1")
    second = _event(entity_id="n2", payload_hash="hash-2")
    third = _event(entity_id="n3", payload_hash="hash-3")

    assert store.remember_event(first).is_duplicate is False
    assert store.remember_event(second).is_duplicate is False
    assert store.remember_event(third).is_duplicate is False

    assert store.dedupe_size == 2
    assert store.remember_event(first).is_duplicate is False


def test_dedupe_prefers_stable_event_id_over_fallback_fields():
    store = EventCursorStore()
    first = _event(entity_id="same", payload_hash="same", event_id="event-a")
    second = _event(entity_id="same", payload_hash="same", event_id="event-b")

    assert store.remember_event(first).is_duplicate is False
    assert store.remember_event(second).is_duplicate is False
    assert store.dedupe_size == 2


def test_dedupe_prefers_server_cursor_when_event_id_is_absent():
    store = EventCursorStore()
    first = _event(
        entity_id="same",
        payload_hash="same",
        server_cursor="cursor-a",
        include_event_id=False,
    )
    second = _event(
        entity_id="same",
        payload_hash="same",
        server_cursor="cursor-b",
        include_event_id=False,
    )

    assert store.remember_event(first).is_duplicate is False
    assert store.remember_event(second).is_duplicate is False
    assert store.dedupe_size == 2
