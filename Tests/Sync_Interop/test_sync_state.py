from __future__ import annotations

import pytest

from tldw_chatbook.Sync_Interop.sync_state import (
    ConflictPolicy,
    ConflictStrategy,
    LocalOutboxEntry,
    RemotePullCursor,
    SyncProfileState,
    SyncProfileStateStore,
)


def test_sync_profile_state_store_keys_state_by_server_profile_id() -> None:
    store = SyncProfileStateStore()

    server_a = store.get_or_create("server-a")
    server_b = store.get_or_create("server-b")
    server_a.enabled_domains.add("notes")

    assert server_a.server_profile_id == "server-a"
    assert server_b.server_profile_id == "server-b"
    assert store.get_or_create("server-a") is server_a
    assert server_b.enabled_domains == set()


def test_sync_profile_state_requires_server_profile_id() -> None:
    with pytest.raises(ValueError, match="server_profile_id"):
        SyncProfileState(server_profile_id="")


def test_remote_pull_cursor_storage_key_is_scoped_by_server_domain_and_collection() -> None:
    notes_cursor = RemotePullCursor(
        server_profile_id="server-a",
        domain="notes",
        remote_collection="workspace-1",
        cursor="remote-cursor-1",
    )
    chat_cursor = RemotePullCursor(
        server_profile_id="server-a",
        domain="chat",
        remote_collection="workspace-1",
        cursor="remote-cursor-1",
    )
    other_server_cursor = RemotePullCursor(
        server_profile_id="server-b",
        domain="notes",
        remote_collection="workspace-1",
        cursor="remote-cursor-1",
    )

    assert notes_cursor.storage_key() == "server-a:notes:workspace-1"
    assert notes_cursor.storage_key() != chat_cursor.storage_key()
    assert notes_cursor.storage_key() != other_server_cursor.storage_key()


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

