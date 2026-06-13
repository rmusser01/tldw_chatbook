from __future__ import annotations

import pytest

from tldw_chatbook.Notifications import ClientNotificationsDB, EventStateRepository
from tldw_chatbook.Sync_Interop import SyncStateRepository
from tldw_chatbook.runtime_policy.server_parity_state import (
    ServerParityStateRepositories,
    build_server_parity_state_repositories,
)


def test_server_parity_state_bundle_reuses_client_notifications_db(tmp_path):
    local_notifications = ClientNotificationsDB(":memory:")

    bundle = build_server_parity_state_repositories(
        data_dir=tmp_path,
        client_id="test-client",
        local_notifications_db=local_notifications,
    )

    assert isinstance(bundle, ServerParityStateRepositories)
    assert bundle.local_notifications_db is local_notifications
    assert isinstance(bundle.event_state_repository, EventStateRepository)
    assert isinstance(bundle.sync_state_repository, SyncStateRepository)
    assert bundle.event_state_repository.db_path == tmp_path / "tldw_chatbook_event_state.db"
    assert bundle.sync_state_repository.db_path == tmp_path / "tldw_chatbook_sync_state.db"


def test_server_parity_state_builder_requires_existing_local_notifications_db(tmp_path):
    with pytest.raises(TypeError):
        build_server_parity_state_repositories(data_dir=tmp_path, client_id="test-client")


def test_server_parity_state_bundle_clears_scoped_server_profile_state(tmp_path):
    local_notifications = ClientNotificationsDB(":memory:")
    bundle = build_server_parity_state_repositories(
        data_dir=tmp_path,
        client_id="test-client",
        local_notifications_db=local_notifications,
    )
    event = _event(server_profile_id="server-a", authenticated_principal_id="user-a")
    other_event = _event(server_profile_id="server-a", authenticated_principal_id="user-b")
    bundle.event_state_repository.record_event_and_advance_processed_cursor(event)
    bundle.event_state_repository.record_event_and_advance_processed_cursor(other_event)
    bundle.sync_state_repository.record_identity_mapping(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        entity_type="note",
        local_entity_id="local-note-1",
        remote_entity_id="remote-note-1",
        mapping_status="confirmed",
    )
    bundle.sync_state_repository.record_identity_mapping(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-b",
        workspace_scope="workspace-1",
        domain="notes",
        entity_type="note",
        local_entity_id="local-note-2",
        remote_entity_id="remote-note-2",
        mapping_status="confirmed",
    )

    bundle.clear_server_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
    )

    assert bundle.event_state_repository.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor is None
    assert bundle.event_state_repository.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-b",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "cursor-user-b"
    assert bundle.sync_state_repository.list_identity_mappings(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
    ) == []
    assert len(
        bundle.sync_state_repository.list_identity_mappings(
            server_profile_id="server-a",
            authenticated_principal_id="user-b",
        )
    ) == 1


def _event(*, server_profile_id: str, authenticated_principal_id: str):
    from tldw_chatbook.runtime_policy.server_parity_models import NormalizedEventRecord

    return NormalizedEventRecord(
        source_authority="server",
        server_profile_id=server_profile_id,
        authenticated_principal_id=authenticated_principal_id,
        stream_name="notifications",
        stream_instance_id="workspace-1",
        event_kind="notification.created",
        entity_ref={"type": "notification", "id": f"notification-{authenticated_principal_id}"},
        payload_hash=f"hash-{authenticated_principal_id}",
        event_id=f"event-{authenticated_principal_id}",
        server_cursor=f"cursor-{authenticated_principal_id}",
        transport_type="sse",
    )
