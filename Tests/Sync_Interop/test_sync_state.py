from __future__ import annotations

import pytest

from tldw_chatbook.Sync_Interop.sync_state import (
    ConflictPolicy,
    ConflictStrategy,
    LocalOutboxEntry,
    RemotePullCursor,
    SyncProfileState,
    SyncProfileStateStore,
    SyncV2ProfileMode,
)


def test_sync_profile_state_store_keys_state_by_server_profile_id() -> None:
    store = SyncProfileStateStore()

    server_a = store.get_or_create("server-a")
    server_b = store.get_or_create("server-b")
    server_a.enabled_domains.add("notes")

    assert server_a.server_profile_id == "server-a"
    assert server_a.profile_mode == SyncV2ProfileMode.LOCAL_ONLY
    assert server_b.server_profile_id == "server-b"
    assert store.get_or_create("server-a") is server_a
    assert server_b.enabled_domains == set()


def test_sync_profile_state_tracks_sync_v2_device_dataset_and_cursors() -> None:
    state = SyncProfileState(
        server_profile_id="server-a",
        profile_mode=SyncV2ProfileMode.LOCAL_FIRST,
        workspace_id="workspace-a",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"notes": "cursor-1"},
    )

    state.dataset_cursors["chat"] = "cursor-2"

    assert state.profile_mode == SyncV2ProfileMode.LOCAL_FIRST
    assert state.device_id == "device-1"
    assert state.dataset_id == "dataset-1"
    assert state.dataset_cursors == {"notes": "cursor-1", "chat": "cursor-2"}


def test_sync_profile_state_store_isolates_same_server_by_workspace_id() -> None:
    store = SyncProfileStateStore()

    workspace_a = store.get_or_create("server-a", workspace_id="workspace-a")
    workspace_b = store.get_or_create("server-a", workspace_id="workspace-b")
    workspace_a.enabled_domains.add("notes")

    assert workspace_a is store.get_or_create("server-a", workspace_id="workspace-a")
    assert workspace_a is not workspace_b
    assert workspace_a.workspace_id == "workspace-a"
    assert workspace_b.workspace_id == "workspace-b"
    assert workspace_b.enabled_domains == set()


def test_sync_profile_state_requires_server_profile_id() -> None:
    with pytest.raises(ValueError, match="server_profile_id"):
        SyncProfileState(server_profile_id="")


def test_remote_pull_cursor_storage_key_is_scoped_by_server_domain_and_collection() -> None:
    notes_cursor = RemotePullCursor(
        source_authority="server",
        server_profile_id="server-a",
        workspace_id="workspace-1",
        domain="notes",
        remote_collection="collection-1",
        cursor="remote-cursor-1",
    )
    chat_cursor = RemotePullCursor(
        source_authority="server",
        server_profile_id="server-a",
        workspace_id="workspace-1",
        domain="chat",
        remote_collection="collection-1",
        cursor="remote-cursor-1",
    )
    other_server_cursor = RemotePullCursor(
        source_authority="server",
        server_profile_id="server-b",
        workspace_id="workspace-1",
        domain="notes",
        remote_collection="collection-1",
        cursor="remote-cursor-1",
    )

    assert notes_cursor.storage_key() == "server:server-a:workspace-1:notes:collection-1"
    assert notes_cursor.storage_key() != chat_cursor.storage_key()
    assert notes_cursor.storage_key() != other_server_cursor.storage_key()


def test_remote_pull_cursor_storage_key_differs_by_workspace_and_source_authority() -> None:
    server_cursor = RemotePullCursor(
        source_authority="server",
        server_profile_id="server-a",
        workspace_id="workspace-a",
        domain="notes",
        remote_collection="collection-1",
    )
    other_workspace = RemotePullCursor(
        source_authority="server",
        server_profile_id="server-a",
        workspace_id="workspace-b",
        domain="notes",
        remote_collection="collection-1",
    )
    local_cursor = RemotePullCursor(
        source_authority="local",
        server_profile_id=None,
        workspace_id="workspace-a",
        domain="notes",
        remote_collection="collection-1",
    )

    assert server_cursor.storage_key() != other_workspace.storage_key()
    assert server_cursor.storage_key() != local_cursor.storage_key()


def test_default_conflict_policy_is_read_only_and_preserves_local() -> None:
    policy = ConflictPolicy.default()

    assert policy.strategy == ConflictStrategy.PRESERVE_LOCAL
    assert policy.allow_remote_overwrite is False


def test_local_outbox_entry_is_shape_only_without_dispatch_metadata() -> None:
    entry = LocalOutboxEntry(
        entry_id="outbox-1",
        server_profile_id="server-a",
        domain="notes",
        workspace_id="workspace-1",
        local_entity_id="note-1",
        operation="update",
        payload_hash="sha256:abc",
        payload={"title": "Draft"},
    )

    assert entry.storage_key() == "server-a:notes:workspace-1:outbox-1"
    assert entry.payload == {"title": "Draft"}
    assert not hasattr(entry, "dispatch")
    assert not hasattr(entry, "replay")


def test_local_outbox_entry_payload_is_immutable_and_stable_after_input_mutation() -> None:
    payload = {"title": "Draft", "metadata": {"tags": ["sync"]}}
    entry = LocalOutboxEntry(
        entry_id="outbox-1",
        server_profile_id="server-a",
        domain="notes",
        workspace_id="workspace-1",
        local_entity_id="note-1",
        operation="update",
        payload_hash="sha256:abc",
        payload=payload,
    )

    payload["metadata"]["tags"].append("mutated")

    assert entry.payload == {"title": "Draft", "metadata": {"tags": ("sync",)}}
    with pytest.raises(TypeError, match="immutable"):
        entry.payload["title"] = "Changed"
    with pytest.raises(TypeError, match="immutable"):
        entry.payload["metadata"]["new"] = "value"
