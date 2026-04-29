import pytest

from tldw_chatbook.Notifications.local_event_producer import LocalEventProducer


def test_local_event_producer_emits_source_scoped_offline_events_without_server_profile():
    producer = LocalEventProducer(source_name="local-notifications", stream_instance_id="device-1")

    event = producer.emit(
        event_kind="notification.delivered",
        entity_ref={"id": "n1"},
        payload={"title": "Local"},
        payload_hash="payload-1",
        emitted_at="2026-04-28T12:00:00Z",
    )

    assert event.source_authority == "local"
    assert event.server_profile_id is None
    assert event.stream_name == "local-notifications"
    assert event.stream_instance_id == "device-1"
    assert event.transport_type == "local_producer"
    assert event.payload["title"] == "Local"


def test_local_event_producer_rejects_server_profile_id():
    producer = LocalEventProducer(source_name="local-notifications")

    with pytest.raises(ValueError, match="server_profile_id"):
        producer.emit(
            event_kind="notification.delivered",
            entity_ref={"id": "n1"},
            payload_hash="payload-1",
            server_profile_id="server-a",
        )
