from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.Notifications import EventStateRepository
from tldw_chatbook.Notifications.event_observer import EventObserver
from tldw_chatbook.runtime_policy.server_parity_models import NormalizedEventRecord


def _event(
    *,
    server_profile_id: str = "server-a",
    authenticated_principal_id: str | None = "user-a",
    stream_name: str = "notifications",
    stream_instance_id: str = "workspace-1",
    event_kind: str = "notification.created",
    entity_id: str = "n1",
    payload_hash: str = "hash-1",
    server_cursor: str | None = "cursor-1",
    event_id: str | None = "event-1",
) -> NormalizedEventRecord:
    return NormalizedEventRecord(
        source_authority="server",
        server_profile_id=server_profile_id,
        authenticated_principal_id=authenticated_principal_id,
        stream_name=stream_name,
        stream_instance_id=stream_instance_id,
        event_kind=event_kind,
        entity_ref={"type": "notification", "id": entity_id},
        payload_hash=payload_hash,
        event_id=event_id,
        server_cursor=server_cursor,
        emitted_at="2026-04-29T01:02:03Z",
        received_at="2026-04-29T01:02:04Z",
        transport_type="sse",
        payload_kind="notification",
        payload={"title": "Notice"},
    )


def _local_event() -> NormalizedEventRecord:
    return NormalizedEventRecord(
        source_authority="local",
        server_profile_id=None,
        authenticated_principal_id=None,
        stream_name="local_notifications",
        stream_instance_id="local",
        event_kind="notification.created",
        entity_ref={"type": "notification", "id": "local-1"},
        payload_hash="local-hash",
        event_id="local-event-1",
        server_cursor="local-cursor-1",
        transport_type="local_producer",
    )


class _Transport:
    def __init__(self, events: list[NormalizedEventRecord]) -> None:
        self.events = events

    async def stream(self, cursor):
        for event in self.events:
            await asyncio.sleep(0)
            yield event


def test_repository_persists_events_dedupe_and_processed_cursor(tmp_path):
    db_path = tmp_path / "events.db"
    repo = EventStateRepository(db_path)
    event = _event()

    result = repo.record_event_and_advance_processed_cursor(event)
    repo.close()

    reopened = EventStateRepository(db_path)
    cursor = reopened.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )

    assert result.is_duplicate is False
    assert result.cursor.cursor == "cursor-1"
    assert reopened.is_duplicate_event(event) is True
    assert cursor.cursor == "cursor-1"
    assert [row["event_id"] for row in reopened.list_events(limit=10)] == ["event-1"]


def test_repository_scopes_processed_cursor_and_dedupe_by_authenticated_principal(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    first = _event(authenticated_principal_id="user-a", server_cursor="cursor-a", event_id="event-1")
    second = _event(authenticated_principal_id="user-b", server_cursor="cursor-b", event_id="event-1")

    first_result = repo.record_event_and_advance_processed_cursor(first)
    second_result = repo.record_event_and_advance_processed_cursor(second)

    assert first_result.is_duplicate is False
    assert second_result.is_duplicate is False
    assert repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "cursor-a"
    assert repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-b",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "cursor-b"
    assert len(repo.list_events(limit=10)) == 2


def test_duplicate_event_does_not_insert_second_record_or_advance_cursor(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    original = _event(event_id="event-1", server_cursor="cursor-1")
    duplicate = _event(event_id="event-1", server_cursor="cursor-2")

    first = repo.record_event_and_advance_processed_cursor(original)
    second = repo.record_event_and_advance_processed_cursor(duplicate)

    cursor = repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )

    assert first.is_duplicate is False
    assert second.is_duplicate is True
    assert cursor.cursor == "cursor-1"
    assert [row["server_cursor"] for row in repo.list_events(limit=10)] == ["cursor-1"]


def test_presented_high_water_is_separate_from_processed_cursor(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    event = _event()

    record = repo.record_event_and_advance_processed_cursor(event)

    assert repo.get_presented_high_water(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor is None

    presentation = repo.mark_event_presented_and_advance_high_water(
        event_key=record.event_key,
        cursor="cursor-1",
        presented_at="2026-04-29T01:02:05Z",
    )

    assert presentation.event_key == record.event_key
    assert presentation.local_delivery_state == "delivered"
    assert repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "cursor-1"
    assert repo.get_presented_high_water(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "cursor-1"


def test_cursor_upsert_handles_null_server_and_principal_scope(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    repo.record_event_and_advance_processed_cursor(_local_event())
    repo.record_event_and_advance_processed_cursor(
        NormalizedEventRecord(
            source_authority="local",
            server_profile_id=None,
            authenticated_principal_id=None,
            stream_name="local_notifications",
            stream_instance_id="local",
            event_kind="notification.updated",
            entity_ref={"type": "notification", "id": "local-2"},
            payload_hash="local-hash-2",
            event_id="local-event-2",
            server_cursor="local-cursor-2",
            transport_type="local_producer",
        )
    )

    assert repo.get_processed_cursor(
        source_authority="local",
        server_profile_id=None,
        authenticated_principal_id=None,
        stream_name="local_notifications",
        stream_instance_id="local",
    ).cursor == "local-cursor-2"


@pytest.mark.asyncio
async def test_repository_can_back_event_observer_without_in_memory_cursor_store(tmp_path):
    db_path = tmp_path / "events.db"
    repo = EventStateRepository(db_path)
    event = _event()

    await EventObserver(store=repo, transport=_Transport([event])).run(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        handler=lambda handled: handled.event_id == "event-1",
        max_events=1,
    )
    repo.close()

    reopened = EventStateRepository(db_path)
    assert reopened.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "cursor-1"
    assert reopened.is_duplicate_event(event) is True


def test_prune_stream_state_removes_oldest_events_but_preserves_cursors(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    first = _event(entity_id="n1", event_id="event-1", server_cursor="cursor-1")
    second = _event(entity_id="n2", event_id="event-2", server_cursor="cursor-2")
    third = _event(entity_id="n3", event_id="event-3", server_cursor="cursor-3")

    repo.record_event_and_advance_processed_cursor(first)
    repo.record_event_and_advance_processed_cursor(second)
    third_record = repo.record_event_and_advance_processed_cursor(third)
    repo.mark_event_presented_and_advance_high_water(
        event_key=third_record.event_key,
        cursor="cursor-3",
        presented_at="2026-04-29T01:02:06Z",
    )

    pruned = repo.prune_stream_state(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        max_count=2,
    )

    assert pruned == 1
    assert [row["event_id"] for row in repo.list_events(limit=10)] == ["event-2", "event-3"]
    assert repo.is_duplicate_event(first) is False
    assert repo.is_duplicate_event(third) is True
    assert repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "cursor-3"
    assert repo.get_presented_high_water(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "cursor-3"


def test_prune_stream_state_supports_age_cutoff_and_preserves_cursors(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    first = _event(entity_id="n1", event_id="event-1", server_cursor="cursor-1")
    second = _event(entity_id="n2", event_id="event-2", server_cursor="cursor-2")

    repo.record_event_and_advance_processed_cursor(first)
    repo.record_event_and_advance_processed_cursor(second)

    pruned = repo.prune_stream_state(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        older_than="9999-01-01T00:00:00Z",
    )

    assert pruned == 2
    assert repo.list_events(limit=10) == []
    assert repo.is_duplicate_event(first) is False
    assert repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "cursor-2"


def test_observer_status_records_survive_restart_and_reset_cursor_records_status(tmp_path):
    db_path = tmp_path / "events.db"
    repo = EventStateRepository(db_path)
    cursor = repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )

    repo.reset_cursor(cursor, reason="unsupported_cursor")
    repo.close()

    reopened = EventStateRepository(db_path)
    status = reopened.get_observer_status(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )

    assert status["status"] == "cursor_reset"
    assert status["reason"] == "unsupported_cursor"
    assert status["details"] == {}


def test_observer_status_can_record_degraded_dedupe_only_mode(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")

    repo.record_observer_status(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        status="degraded_dedupe_only",
        reason="stable_cursor_missing",
        details={"transport": "sse"},
    )

    status = repo.get_observer_status(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )
    assert status["status"] == "degraded_dedupe_only"
    assert status["reason"] == "stable_cursor_missing"
    assert status["details"] == {"transport": "sse"}


def test_reset_stream_cursor_is_scoped_by_principal_and_records_status(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    repo.record_event_and_advance_processed_cursor(
        _event(authenticated_principal_id="user-a", event_id="event-a", server_cursor="cursor-a")
    )
    repo.record_event_and_advance_processed_cursor(
        _event(authenticated_principal_id="user-b", event_id="event-b", server_cursor="cursor-b")
    )

    result = repo.reset_stream_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        reason="manual_replay",
    )

    assert result.cursor.cursor is None
    assert repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor is None
    assert repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-b",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "cursor-b"
    assert repo.get_observer_status(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )["reason"] == "manual_replay"


def test_clear_server_profile_state_removes_scoped_event_state_only(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    server_a = _event(event_id="event-a", server_cursor="cursor-a")
    server_b = _event(server_profile_id="server-b", event_id="event-b", server_cursor="cursor-b")
    server_a_record = repo.record_event_and_advance_processed_cursor(server_a)
    repo.record_event_and_advance_processed_cursor(server_b)
    repo.mark_event_presented_and_advance_high_water(
        event_key=server_a_record.event_key,
        cursor="cursor-a",
    )
    repo.record_observer_status(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        status="degraded_dedupe_only",
        reason="stable_cursor_missing",
    )

    cleared = repo.clear_server_profile_state(server_profile_id="server-a", authenticated_principal_id="user-a")

    assert cleared["events"] == 1
    assert cleared["dedupe_records"] == 1
    assert cleared["processed_cursors"] == 1
    assert cleared["presented_high_water"] == 1
    assert cleared["observer_status"] == 1
    assert repo.is_duplicate_event(server_a) is False
    assert repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor is None
    assert repo.get_presented_high_water(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor is None
    assert repo.get_observer_status(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ) is None
    assert repo.is_duplicate_event(server_b) is True
    assert repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-b",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "cursor-b"


def test_retention_policy_has_durable_default_and_stream_override(tmp_path):
    db_path = tmp_path / "events.db"
    repo = EventStateRepository(db_path)

    default_policy = repo.get_retention_policy(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )
    repo.set_retention_policy(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        max_age_days=7,
        max_count=250,
    )
    repo.close()

    reopened = EventStateRepository(db_path)
    overridden_policy = reopened.get_retention_policy(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )

    assert default_policy.max_age_days == 30
    assert default_policy.max_count == 10_000
    assert overridden_policy.max_age_days == 7
    assert overridden_policy.max_count == 250
