from __future__ import annotations

import pytest

from tldw_chatbook.runtime_policy.server_parity_models import (
    EventCursor,
    EventDedupeKey,
    NormalizedEventRecord,
    NotificationPresentationRecord,
    ProviderMigrationStatus,
    SyncIdentityMapEntry,
    SyncReadinessReport,
)


def test_event_cursor_key_is_scoped_by_source_server_stream_and_instance() -> None:
    server_cursor = EventCursor(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="default",
        cursor="abc",
    )
    local_cursor = EventCursor(
        source_authority="local",
        server_profile_id=None,
        stream_name="notifications",
        stream_instance_id="default",
        cursor="abc",
    )

    assert server_cursor.storage_key() == "server:server-a:notifications:default"
    assert local_cursor.storage_key() == "local:none:notifications:default"
    assert server_cursor.storage_key() != local_cursor.storage_key()


def test_server_normalized_event_requires_server_profile_id() -> None:
    with pytest.raises(ValueError, match="server_profile_id"):
        NormalizedEventRecord(
            source_authority="server",
            server_profile_id=None,
            stream_name="notifications",
            stream_instance_id="default",
            event_kind="notification.created",
            entity_ref={"type": "notification", "id": "n-1"},
            payload_hash="hash",
        )


def test_local_normalized_event_does_not_require_server_profile_id() -> None:
    record = NormalizedEventRecord(
        source_authority="local",
        server_profile_id=None,
        stream_name="notifications",
        stream_instance_id="default",
        event_kind="notification.created",
        entity_ref={"type": "notification", "id": "n-1"},
        payload_hash="hash",
    )

    assert record.source_authority == "local"
    assert record.server_profile_id is None


def test_event_dedupe_key_falls_back_to_normalized_event_identity() -> None:
    record = NormalizedEventRecord(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="default",
        event_kind="notification.created",
        entity_ref={"type": "notification", "id": "n-1"},
        payload_hash="payload-sha",
        emitted_at="2026-04-29T01:02:03Z",
    )

    assert EventDedupeKey.from_event(record) == EventDedupeKey(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="default",
        event_kind="notification.created",
        entity_id="n-1",
        timestamp="2026-04-29T01:02:03Z",
        payload_hash="payload-sha",
    )


def test_sync_readiness_defaults_to_not_registered_and_not_write_enabled() -> None:
    report = SyncReadinessReport(domain="chat")

    assert report.sync_eligible is False
    assert report.write_enabled is False
    assert report.reason_codes == ("not_registered",)


def test_sync_identity_map_entry_preserves_source_scope_server_and_workspace_ids() -> None:
    entry = SyncIdentityMapEntry(
        domain="notes",
        source_authority="server",
        source_scope="workspace",
        local_entity_id="local-1",
        remote_entity_id="remote-1",
        server_profile_id="server-a",
        workspace_id="workspace-a",
    )

    assert entry.source_authority == "server"
    assert entry.source_scope == "workspace"
    assert entry.server_profile_id == "server-a"
    assert entry.workspace_id == "workspace-a"


def test_provider_migration_status_represents_migrated_and_compatibility_mode() -> None:
    migrated = ProviderMigrationStatus(service_name="chat", provider_backed=True)
    compatibility = ProviderMigrationStatus(
        service_name="notes",
        provider_backed=False,
        compatibility_mode=True,
        reason_code="legacy_config_factory",
    )

    assert migrated.provider_backed is True
    assert migrated.compatibility_mode is False
    assert compatibility.provider_backed is False
    assert compatibility.compatibility_mode is True
    assert compatibility.reason_code == "legacy_config_factory"


def test_notification_presentation_keeps_local_delivery_state_separate_from_server_state() -> None:
    presentation = NotificationPresentationRecord(
        event_key="server:server-a:notifications:default:n-1",
        local_delivery_state="delivered",
        server_read_state="unread",
        server_dismiss_state="active",
    )

    assert presentation.local_delivery_state == "delivered"
    assert presentation.server_read_state == "unread"
    assert presentation.server_dismiss_state == "active"
