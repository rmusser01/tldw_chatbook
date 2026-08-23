from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.Sync_Interop import (
    sync_state_repository as sync_state_repository_module,
)


def test_identity_mapping_persists_scope_and_side_keys(tmp_path):
    db_path = tmp_path / "sync_state.db"
    repo = SyncStateRepository(db_path)

    mapping = repo.record_identity_mapping(
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
    repo.close()

    reopened = SyncStateRepository(db_path)
    rows = reopened.list_identity_mappings(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        entity_type="note",
    )

    assert mapping.source_scope_key == "server:server-a:user-a:workspace-1:notes:note"
    assert (
        mapping.local_side_key
        == "server:server-a:user-a:workspace-1:notes:note:local:local-note-1"
    )
    assert (
        mapping.remote_side_key
        == "server:server-a:user-a:workspace-1:notes:note:remote:remote-note-1"
    )
    assert [row.local_entity_id for row in rows] == ["local-note-1"]
    assert rows[0].remote_entity_id == "remote-note-1"


def test_duplicate_local_side_mapping_creates_conflict_without_overwrite(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.record_identity_mapping(
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

    duplicate = repo.record_identity_mapping(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        entity_type="note",
        local_entity_id="local-note-1",
        remote_entity_id="remote-note-2",
        mapping_status="confirmed",
    )
    conflicts = repo.list_conflict_reports(domain="notes")

    assert duplicate.mapping_status == "conflict"
    assert len(repo.list_identity_mappings(domain="notes")) == 2
    assert conflicts[0]["conflict_type"] == "duplicate_local_side"
    assert (
        conflicts[0]["source_scope_key"]
        == "server:server-a:user-a:workspace-1:notes:note"
    )


def test_identity_mapping_validation_allows_orphans_but_not_confirmed_missing_side(
    tmp_path,
):
    repo = SyncStateRepository(tmp_path / "sync_state.db")

    orphan = repo.record_identity_mapping(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        entity_type="note",
        local_entity_id="local-note-1",
        remote_entity_id=None,
        mapping_status="orphaned_local",
    )

    assert orphan.remote_side_key is None
    with pytest.raises(
        ValueError, match="confirmed mapping requires local and remote entity IDs"
    ):
        repo.record_identity_mapping(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="notes",
            entity_type="note",
            local_entity_id="local-note-2",
            remote_entity_id=None,
            mapping_status="confirmed",
        )


def test_pull_cursor_and_mirror_report_persist_by_principal(tmp_path):
    db_path = tmp_path / "sync_state.db"
    repo = SyncStateRepository(db_path)
    repo.set_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        remote_collection="notes",
        cursor="cursor-a",
    )
    repo.set_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-b",
        workspace_scope="workspace-1",
        domain="notes",
        remote_collection="notes",
        cursor="cursor-b",
    )
    report = repo.record_mirror_report(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        report={"dry_run": True, "write_enabled": False, "mapped_count": 1},
    )
    repo.close()

    reopened = SyncStateRepository(db_path)

    assert (
        reopened.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="notes",
            remote_collection="notes",
        ).cursor
        == "cursor-a"
    )
    assert (
        reopened.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-b",
            workspace_scope="workspace-1",
            domain="notes",
            remote_collection="notes",
        ).cursor
        == "cursor-b"
    )
    assert (
        reopened.list_mirror_reports(domain="notes")[0]["report_id"]
        == report["report_id"]
    )
    assert (
        reopened.list_mirror_reports(domain="notes")[0]["report"]["write_enabled"]
        is False
    )


def test_latest_mirror_report_fetches_newest_without_full_history(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    first = repo.record_mirror_report(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="library_collections",
        report={"dry_run": True, "write_enabled": False, "mapped_count": 1},
    )
    latest = repo.record_mirror_report(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="library_collections",
        report={"dry_run": True, "write_enabled": False, "mapped_count": 2},
    )

    report = repo.get_latest_mirror_report(domain="library_collections")

    assert report is not None
    assert report["report_id"] == latest["report_id"]
    assert report["report_id"] != first["report_id"]
    assert report["report"]["mapped_count"] == 2


def test_latest_mirror_report_can_be_scoped_to_active_profile(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    active = repo.record_mirror_report(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="library_collections",
        report={"dry_run": True, "write_enabled": False, "mapped_count": 1},
    )
    repo.record_mirror_report(
        source_authority="server",
        server_profile_id="server-b",
        authenticated_principal_id="user-b",
        workspace_scope="workspace-2",
        domain="library_collections",
        report={"dry_run": True, "write_enabled": False, "mapped_count": 99},
    )

    report = repo.get_latest_mirror_report(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="library_collections",
    )

    assert report is not None
    assert report["report_id"] == active["report_id"]
    assert report["report"]["mapped_count"] == 1


def test_conflict_report_listing_supports_bounded_reads(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    for index in range(3):
        repo.record_identity_mapping(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="library_collections",
            entity_type="collection",
            local_entity_id="collection-1",
            remote_entity_id=f"remote-{index}",
            mapping_status="confirmed",
        )

    conflicts = repo.list_conflict_reports(domain="library_collections", limit=1)

    assert len(conflicts) == 1
    assert conflicts[0]["local_side_key"].endswith(":local:collection-1")
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        repo.list_conflict_reports(domain="library_collections", limit=0)


def test_conflict_report_listing_and_count_can_be_scoped_to_active_profile(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    for server_profile_id, user_id, workspace_id, collection_id in (
        ("server-a", "user-a", "workspace-1", "collection-active"),
        ("server-b", "user-b", "workspace-2", "collection-other"),
    ):
        repo.record_identity_mapping(
            source_authority="server",
            server_profile_id=server_profile_id,
            authenticated_principal_id=user_id,
            workspace_scope=workspace_id,
            domain="library_collections",
            entity_type="collection",
            local_entity_id=collection_id,
            remote_entity_id="remote-a",
            mapping_status="confirmed",
        )
        repo.record_identity_mapping(
            source_authority="server",
            server_profile_id=server_profile_id,
            authenticated_principal_id=user_id,
            workspace_scope=workspace_id,
            domain="library_collections",
            entity_type="collection",
            local_entity_id=collection_id,
            remote_entity_id="remote-b",
            mapping_status="confirmed",
        )

    conflicts = repo.list_conflict_reports(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="library_collections",
    )
    conflict_count = repo.count_conflict_reports(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="library_collections",
    )

    assert len(conflicts) == 1
    assert conflicts[0]["local_side_key"].endswith(":local:collection-active")
    assert conflict_count == 1


def test_sync_profile_state_persists_last_report_and_error_by_principal(tmp_path):
    db_path = tmp_path / "sync_state.db"
    repo = SyncStateRepository(db_path)
    report = repo.record_mirror_report(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        report={"dry_run": True, "write_enabled": False},
    )

    repo.set_sync_profile_state(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        last_mirror_report_id=report["report_id"],
        last_error="remote_unavailable",
    )
    repo.close()

    reopened = SyncStateRepository(db_path)

    assert (
        reopened.get_sync_profile_state(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
        )["last_mirror_report_id"]
        == report["report_id"]
    )
    assert (
        reopened.get_sync_profile_state(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
        )["last_error"]
        == "remote_unavailable"
    )
    assert (
        reopened.get_sync_profile_state(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-b",
            workspace_scope="workspace-1",
        )
        is None
    )


def test_sync_v2_profile_state_persists_device_dataset_cursors_and_metadata(tmp_path):
    db_path = tmp_path / "sync_state.db"
    repo = SyncStateRepository(db_path)

    profile = repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"notes": "cursor-1"},
        capabilities={"max_batch_size": 100},
        dry_run_metadata={"pulled_envelopes": 0},
    )
    repo.close()

    reopened = SyncStateRepository(db_path)
    stored = reopened.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert profile["profile_mode"] == "local_first"
    assert stored["device_id"] == "device-1"
    assert stored["dataset_id"] == "dataset-1"
    assert stored["dataset_cursors"] == {"notes": "cursor-1"}
    assert stored["capabilities"] == {"max_batch_size": 100}
    assert stored["dry_run_metadata"] == {"pulled_envelopes": 0}


def test_sync_v2_profile_state_persists_canonical_local_first_sync_mode(tmp_path):
    db_path = tmp_path / "sync_state.db"
    repo = SyncStateRepository(db_path)

    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"notes": "cursor-1"},
    )
    repo.close()

    reopened = SyncStateRepository(db_path)
    stored = reopened.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert stored["profile_mode"] == "local_first_sync"


def test_sync_v2_schema_migration_updates_v3_without_losing_existing_rows(tmp_path):
    db_path = tmp_path / "sync_state.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY NOT NULL
            );
            INSERT INTO schema_version (version) VALUES (3);

            CREATE TABLE sync_profile_state (
                source_authority TEXT NOT NULL,
                server_profile_id TEXT NOT NULL,
                authenticated_principal_id TEXT NOT NULL,
                workspace_scope TEXT NOT NULL,
                last_error TEXT,
                last_mirror_report_id INTEGER,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope
                )
            );
            INSERT INTO sync_profile_state (
                source_authority, server_profile_id,
                authenticated_principal_id, workspace_scope,
                last_error, updated_at
            ) VALUES (
                'server', 'server-a', 'user-a', 'workspace-1',
                'preserved-profile', '2026-08-12T00:00:00Z'
            );

            CREATE TABLE remote_pull_cursors (
                source_scope_key TEXT NOT NULL,
                remote_collection TEXT NOT NULL,
                source_authority TEXT NOT NULL,
                server_profile_id TEXT,
                authenticated_principal_id TEXT,
                workspace_scope TEXT,
                domain TEXT NOT NULL,
                cursor TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (source_scope_key, remote_collection)
            );
            INSERT INTO remote_pull_cursors VALUES (
                'server:server-a:user-a:workspace-1:chat:message',
                'messages', 'server', 'server-a', 'user-a', 'workspace-1',
                'chat', 'preserved-cursor', '2026-08-12T00:00:00Z'
            );

            CREATE TABLE sync_v2_local_outbox (
                outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_scope_key TEXT NOT NULL,
                server_profile_id TEXT NOT NULL,
                authenticated_principal_id TEXT NOT NULL,
                workspace_scope TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                client_envelope_id TEXT NOT NULL,
                envelope TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                dispatched_at TEXT,
                UNIQUE(source_scope_key, dataset_id, client_envelope_id)
            );
            INSERT INTO sync_v2_local_outbox (
                source_scope_key, server_profile_id,
                authenticated_principal_id, workspace_scope, dataset_id,
                domain, client_envelope_id, envelope, status,
                created_at, updated_at
            ) VALUES (
                'server:server-a:user-a:workspace-1:sync_v2:outbox',
                'server-a', 'user-a', 'workspace-1', 'dataset-1', 'chat',
                'device-1:chat:message-1:sha256:old', '{}', 'pending',
                '2026-08-12T00:00:00Z', '2026-08-12T00:00:00Z'
            );

            CREATE TABLE sync_v2_conflict_reviews (
                conflict_review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_scope_key TEXT NOT NULL,
                server_profile_id TEXT NOT NULL,
                authenticated_principal_id TEXT NOT NULL,
                workspace_scope TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                source_conflict_key TEXT NOT NULL,
                conflict_kind TEXT NOT NULL,
                item_label TEXT NOT NULL,
                cause TEXT NOT NULL,
                local_summary TEXT NOT NULL,
                remote_summary TEXT NOT NULL,
                recovery_options TEXT NOT NULL,
                resolution_status TEXT NOT NULL,
                details TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_at TEXT,
                UNIQUE(source_scope_key, dataset_id, source_conflict_key)
            );
            INSERT INTO sync_v2_conflict_reviews (
                source_scope_key, server_profile_id,
                authenticated_principal_id, workspace_scope, dataset_id,
                domain, source_conflict_key, conflict_kind, item_label,
                cause, local_summary, remote_summary, recovery_options,
                resolution_status, details, created_at, updated_at
            ) VALUES (
                'server:server-a:user-a:workspace-1:sync_v2:outbox',
                'server-a', 'user-a', 'workspace-1', 'dataset-1', 'chat',
                'preserved-conflict', 'stale_base', 'Message', 'Concurrent edit',
                'Local', 'Remote', '{}', 'open', '{}',
                '2026-08-12T00:00:00Z', '2026-08-12T00:00:00Z'
            );
            """
        )

    repo = SyncStateRepository(db_path)
    with repo._get_connection() as conn:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(sync_profile_state)").fetchall()
        }
        outbox = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sync_v2_local_outbox'"
        ).fetchone()
        conflict_reviews = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sync_v2_conflict_reviews'"
        ).fetchone()
        receipts = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'sync_v2_source_projection_receipts'"
        ).fetchone()
        preserved_outbox = conn.execute(
            "SELECT client_envelope_id FROM sync_v2_local_outbox"
        ).fetchone()[0]
        preserved_profile = conn.execute(
            "SELECT last_error FROM sync_profile_state"
        ).fetchone()[0]
        preserved_cursor = conn.execute(
            "SELECT cursor FROM remote_pull_cursors"
        ).fetchone()[0]
        preserved_conflict = conn.execute(
            "SELECT source_conflict_key FROM sync_v2_conflict_reviews"
        ).fetchone()[0]
        schema_version = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0]
        schema_versions = [
            row[0]
            for row in conn.execute(
                "SELECT version FROM schema_version ORDER BY version"
            ).fetchall()
        ]

    assert {
        "profile_mode",
        "device_id",
        "dataset_id",
        "dataset_cursors",
        "capabilities",
        "dry_run_metadata",
    }.issubset(columns)
    assert outbox is not None
    assert conflict_reviews is not None
    assert receipts is not None
    assert preserved_outbox == "device-1:chat:message-1:sha256:old"
    assert preserved_profile == "preserved-profile"
    assert preserved_cursor == "preserved-cursor"
    assert preserved_conflict == "preserved-conflict"
    assert schema_version == 4
    assert schema_versions == [4]


def test_sync_state_repository_exposes_explicit_durability(tmp_path):
    assert SyncStateRepository(tmp_path / "sync_state.db").is_durable is True
    assert SyncStateRepository(":memory:").is_durable is False


def test_source_projection_receipt_is_atomic_idempotent_and_versioned(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=generate_dataset_key(),
    )
    scope = {
        "server_profile_id": "server-a",
        "authenticated_principal_id": "user-a",
        "workspace_scope": "workspace-1",
        "dataset_id": "dataset-1",
    }
    first = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="first",
        entity_version=1,
    )

    projected = repo.enqueue_sync_v2_outbox_envelope_with_source_receipt(
        **scope,
        envelope=first,
        source_entity_id="message-1",
        source_version=1,
        source_payload_hash=first.payload_hash,
    )
    repeated = repo.enqueue_sync_v2_outbox_envelope_with_source_receipt(
        **scope,
        envelope=first,
        source_entity_id="message-1",
        source_version=1,
        source_payload_hash=first.payload_hash,
    )

    assert (
        repeated["outbox_entry"]["outbox_id"] == projected["outbox_entry"]["outbox_id"]
    )
    assert repeated["receipt"] == projected["receipt"]
    assert len(repo.list_sync_v2_outbox_entries(**scope)) == 1
    assert projected["receipt"]["client_envelope_id"] == first.client_envelope_id
    assert "first" not in repr(projected["receipt"])

    second = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="second",
        entity_version=2,
    )
    repo.enqueue_sync_v2_outbox_envelope_with_source_receipt(
        **scope,
        envelope=second,
        source_entity_id="message-1",
        source_version=2,
        source_payload_hash=second.payload_hash,
    )
    assert len(repo.list_sync_v2_outbox_entries(**scope)) == 2
    with repo._get_connection() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM sync_v2_source_projection_receipts"
            ).fetchone()[0]
            == 2
        )


def test_source_projection_receipt_readback_failure_rolls_back_both_rows(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "sync_state.db"
    repo = SyncStateRepository(db_path)
    envelope = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=generate_dataset_key(),
    ).build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="private",
        entity_version=1,
    )

    def refuse_readback(_row):
        raise RuntimeError("injected receipt readback failure")

    monkeypatch.setattr(
        SyncStateRepository,
        "_source_projection_receipt_from_row",
        staticmethod(refuse_readback),
    )
    with pytest.raises(RuntimeError, match="injected receipt readback failure"):
        repo.enqueue_sync_v2_outbox_envelope_with_source_receipt(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            dataset_id="dataset-1",
            envelope=envelope,
            source_entity_id="message-1",
            source_version=1,
            source_payload_hash=envelope.payload_hash,
        )

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM sync_v2_local_outbox").fetchone()[0] == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM sync_v2_source_projection_receipts"
            ).fetchone()[0]
            == 0
        )


def test_source_projection_receipt_rejects_orphan_and_wrong_scope(tmp_path):
    db_path = tmp_path / "sync_state.db"
    repo = SyncStateRepository(db_path)
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=generate_dataset_key(),
    )
    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="private",
        entity_version=1,
    )
    scope = {
        "server_profile_id": "server-a",
        "authenticated_principal_id": "user-a",
        "workspace_scope": "workspace-1",
        "dataset_id": "dataset-1",
    }
    repo.enqueue_sync_v2_outbox_envelope_with_source_receipt(
        **scope,
        envelope=envelope,
        source_entity_id="message-1",
        source_version=1,
        source_payload_hash=envelope.payload_hash,
    )

    assert (
        repo.get_sync_v2_source_projection_receipt(
            **scope,
            domain="chat",
            source_entity_id="message-1",
            source_version=1,
            source_payload_hash=envelope.payload_hash,
        )
        is not None
    )
    assert (
        repo.get_sync_v2_source_projection_receipt(
            **{**scope, "authenticated_principal_id": "user-b"},
            domain="chat",
            source_entity_id="message-1",
            source_version=1,
            source_payload_hash=envelope.payload_hash,
        )
        is None
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM sync_v2_local_outbox")
    assert (
        repo.get_sync_v2_source_projection_receipt(
            **scope,
            domain="chat",
            source_entity_id="message-1",
            source_version=1,
            source_payload_hash=envelope.payload_hash,
        )
        is None
    )


def test_sync_v2_profile_column_migration_validates_column_identifiers(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "sync_state.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY NOT NULL
            );
            INSERT INTO schema_version (version) VALUES (1);

            CREATE TABLE sync_profile_state (
                source_authority TEXT NOT NULL,
                server_profile_id TEXT NOT NULL,
                authenticated_principal_id TEXT NOT NULL,
                workspace_scope TEXT NOT NULL,
                last_error TEXT,
                last_mirror_report_id INTEGER,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope
                )
            );
            """
        )
    calls: list[tuple[str, str | None]] = []

    def record_validated_column(
        column_name: str, table_name: str | None = None
    ) -> bool:
        calls.append((column_name, table_name))
        return True

    monkeypatch.setattr(
        sync_state_repository_module,
        "validate_column_name",
        record_validated_column,
    )

    # TASK-21105: the store opens (and migrates) on FIRST USE, not at
    # construction; one operation is what runs the column migration now.
    SyncStateRepository(db_path).list_identity_mappings()

    assert ("profile_mode", "sync_profile_state") in calls
    assert ("dry_run_metadata", "sync_profile_state") in calls


def test_sync_v2_outbox_persists_pending_entries_and_push_results(tmp_path):
    db_path = tmp_path / "sync_state.db"
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    )
    accepted = builder.build_note_metadata_update(note_id="note-1", status="archived")
    rejected = builder.build_note_metadata_update(note_id="note-2", status="active")
    conflicted = builder.build_note_metadata_update(note_id="note-3", status="draft")
    repo = SyncStateRepository(db_path)
    accepted_entry = repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=accepted,
    )
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=rejected,
    )
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=conflicted,
    )
    repo.close()

    reopened = SyncStateRepository(db_path)
    pending = reopened.list_pending_sync_v2_outbox_envelopes(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
    )

    assert accepted_entry["status"] == "pending"
    assert accepted_entry["attempt_count"] == 0
    assert pending[0]["client_envelope_id"] == accepted.client_envelope_id
    assert pending[0]["envelope"]["payload_clear"] == {"status": "archived"}
    assert [entry["client_envelope_id"] for entry in pending] == [
        accepted.client_envelope_id,
        rejected.client_envelope_id,
        conflicted.client_envelope_id,
    ]

    result = reopened.mark_sync_v2_outbox_push_results(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        accepted=[{"client_envelope_id": accepted.client_envelope_id}],
        rejected=[
            {
                "client_envelope_id": rejected.client_envelope_id,
                "error_code": "stale_base",
                "message": "Local base is stale.",
            }
        ],
        conflicts=[
            {
                "client_envelope_id": conflicted.client_envelope_id,
                "conflict_id": "conflict-1",
                "message": "Needs manual review.",
            }
        ],
    )

    pending_after = reopened.list_pending_sync_v2_outbox_envelopes(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
    )
    dispatched = reopened.list_sync_v2_outbox_entries(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        status="dispatched",
    )

    assert result == {"dispatched": 1, "retained": 2}
    assert [entry["client_envelope_id"] for entry in dispatched] == [
        accepted.client_envelope_id
    ]
    assert dispatched[0]["attempt_count"] == 1
    assert [entry["client_envelope_id"] for entry in pending_after] == [
        rejected.client_envelope_id,
        conflicted.client_envelope_id,
    ]
    assert [entry["attempt_count"] for entry in pending_after] == [1, 1]
    assert pending_after[0]["last_error"]["error_code"] == "stale_base"
    assert pending_after[1]["last_error"]["error_code"] == "conflict"


def test_sync_v2_outbox_readback_failure_rolls_back_uncommitted_insert(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "sync_state.db"
    dataset_key = generate_dataset_key()
    envelope = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="active")
    repo = SyncStateRepository(db_path)

    def refuse_readback(_row):
        raise RuntimeError("injected outbox readback failure")

    monkeypatch.setattr(
        SyncStateRepository,
        "_outbox_from_row",
        staticmethod(refuse_readback),
    )

    with pytest.raises(RuntimeError, match="injected outbox readback failure"):
        repo.enqueue_sync_v2_outbox_envelope(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            dataset_id="dataset-1",
            envelope=envelope,
        )

    with sqlite3.connect(db_path) as verification:
        committed = verification.execute(
            "SELECT COUNT(*) FROM sync_v2_local_outbox"
        ).fetchone()[0]
    assert committed == 0


def test_sync_v2_outbox_atomic_enqueue_returns_committed_payload_hash(tmp_path):
    db_path = tmp_path / "sync_state.db"
    dataset_key = generate_dataset_key()
    envelope = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    ).build_note_metadata_update(note_id="note-1", status="active")
    repo = SyncStateRepository(db_path)

    entry = repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=envelope,
    )

    assert entry["envelope"]["payload_hash"] == envelope.payload_hash
    assert (
        repo.list_pending_sync_v2_outbox_envelopes(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            dataset_id="dataset-1",
        )[0]["envelope"]["payload_hash"]
        == envelope.payload_hash
    )


def test_sync_v2_identical_reenqueue_preserves_dispatched_outbox_state(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=generate_dataset_key(),
    )
    dispatched_envelope = builder.build_note_upsert(
        note_id="note-1", title="Title", body="Body"
    )
    scope = {
        "server_profile_id": "server-a",
        "authenticated_principal_id": "user-a",
        "workspace_scope": "workspace-1",
        "dataset_id": "dataset-1",
    }
    repo.enqueue_sync_v2_outbox_envelope(**scope, envelope=dispatched_envelope)
    assert repo.mark_sync_v2_outbox_push_results(
        **scope,
        accepted=[{"client_envelope_id": dispatched_envelope.client_envelope_id}],
        rejected=[],
        conflicts=[],
    ) == {"dispatched": 1, "retained": 0}
    dispatched_before = repo.list_sync_v2_outbox_entries(**scope, status="dispatched")[
        0
    ]

    same_payload_envelope = builder.build_note_upsert(
        note_id="note-1", title="Title", body="Body"
    )
    assert (
        same_payload_envelope.client_envelope_id
        == dispatched_envelope.client_envelope_id
    )
    assert same_payload_envelope.payload_hash == dispatched_envelope.payload_hash

    same_entry = repo.enqueue_sync_v2_outbox_envelope(
        **scope, envelope=same_payload_envelope
    )

    assert same_entry["status"] == "dispatched"
    assert same_entry["dispatched_at"] == dispatched_before["dispatched_at"]
    assert same_entry["attempt_count"] == 1
    assert repo.list_pending_sync_v2_outbox_envelopes(**scope) == []
    changed_envelope = builder.build_note_upsert(
        note_id="note-1", title="Title", body="Changed body"
    )
    assert changed_envelope.payload_hash != dispatched_envelope.payload_hash
    changed_entry = repo.enqueue_sync_v2_outbox_envelope(
        **scope, envelope=changed_envelope
    )
    assert changed_entry["status"] == "pending"
    assert [
        entry["client_envelope_id"]
        for entry in repo.list_pending_sync_v2_outbox_envelopes(**scope)
    ] == [changed_envelope.client_envelope_id]


def test_sync_v2_profile_summary_aggregates_state_counts_and_status(tmp_path):
    db_path = tmp_path / "sync_state.db"
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=dataset_key,
    )
    pending = builder.build_note_metadata_update(note_id="note-1", status="active")
    accepted = builder.build_note_metadata_update(note_id="note-2", status="archived")
    conflicted = builder.build_note_metadata_update(note_id="note-3", status="draft")
    repo = SyncStateRepository(db_path)
    report = repo.record_mirror_report(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        report={"dry_run": True, "write_enabled": False, "mapped_count": 1},
    )
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"sync_v2": "cursor-profile"},
        capabilities={"max_batch_size": 25},
        dry_run_metadata={"domains": ["notes", "chat"]},
        last_error="push_conflicts: 1",
        last_mirror_report_id=report["report_id"],
    )
    repo.set_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="sync_v2",
        remote_collection="dataset-1",
        cursor="cursor-remote",
    )
    repo.record_identity_mapping(
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
    repo.record_identity_mapping(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        entity_type="note",
        local_entity_id="local-note-1",
        remote_entity_id="remote-note-2",
        mapping_status="confirmed",
    )
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=pending,
    )
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=accepted,
    )
    repo.enqueue_sync_v2_outbox_envelope(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        envelope=conflicted,
    )
    repo.mark_sync_v2_outbox_push_results(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        accepted=[{"client_envelope_id": accepted.client_envelope_id}],
        rejected=[],
        conflicts=[
            {
                "client_envelope_id": conflicted.client_envelope_id,
                "conflict_id": "conflict-1",
            }
        ],
    )

    summary = repo.get_sync_v2_profile_summary(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert summary["status"] == "attention_required"
    assert summary["profile"]["profile_mode"] == "local_first_sync"
    assert summary["profile"]["device_id"] == "device-1"
    assert summary["profile"]["dataset_id"] == "dataset-1"
    assert summary["profile"]["last_error"] == "push_conflicts: 1"
    assert summary["cursor"]["remote_cursor"] == "cursor-remote"
    assert summary["cursor"]["profile_cursor"] == "cursor-profile"
    assert summary["outbox"] == {
        "pending": 2,
        "dispatched": 1,
        "by_domain": {"notes": {"pending": 2, "dispatched": 1}},
    }
    assert summary["identity_map"] == {
        "total": 2,
        "confirmed": 1,
        "conflict": 1,
        "by_domain": {"notes": {"confirmed": 1, "conflict": 1}},
    }
    assert summary["conflicts"]["count"] == 1
    assert summary["last_mirror_report"]["report_id"] == report["report_id"]
    assert summary["last_mirror_report"]["domain"] == "notes"


def test_sync_v2_profile_summary_combines_legacy_and_v2_conflicts_before_limit(
    tmp_path,
):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
    )
    for index in range(6):
        repo.record_identity_mapping(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="notes",
            entity_type="note",
            local_entity_id=f"legacy-note-{index}",
            remote_entity_id=f"legacy-remote-{index}-a",
            mapping_status="confirmed",
        )
        repo.record_identity_mapping(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="notes",
            entity_type="note",
            local_entity_id=f"legacy-note-{index}",
            remote_entity_id=f"legacy-remote-{index}-b",
            mapping_status="confirmed",
        )
    repo.record_sync_v2_conflict_review(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        domain="notes",
        item_label="Newest v2 conflict",
        cause="Remote edit conflicts with local edit.",
        local_summary="Local note changed.",
        remote_summary="Remote note changed.",
        source_conflict_key="v2-conflict-1",
        conflict_kind="encrypted_content_edit",
        recovery_options={"retry": "available"},
    )

    summary = repo.get_sync_v2_profile_summary(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert summary["conflicts"]["count"] == 7
    assert len(summary["conflicts"]["latest"]) == 5
    assert summary["conflicts"]["latest"][0]["item_label"] == "Newest v2 conflict"


def test_sync_v2_conflict_review_listing_is_bounded_by_default_and_newest_first(
    tmp_path,
):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    for index in range(101):
        repo.record_sync_v2_conflict_review(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            dataset_id="dataset-1",
            domain="notes",
            item_label=f"Conflict {index}",
            cause="Remote edit conflicts with local edit.",
            local_summary="Local note changed.",
            remote_summary="Remote note changed.",
            source_conflict_key=f"review-{index}",
            conflict_kind="encrypted_content_edit",
            recovery_options={"retry": "available"},
        )

    reviews = repo.list_sync_v2_conflict_reviews(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
    )

    assert len(reviews) == 100
    assert reviews[0]["source_conflict_key"] == "review-100"
    assert reviews[-1]["source_conflict_key"] == "review-1"


def test_sync_v2_conflict_review_listing_filters_domain_before_limit(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    for domain in ("chat", "notes"):
        for index in range(3):
            repo.record_sync_v2_conflict_review(
                server_profile_id="server-a",
                authenticated_principal_id="user-a",
                workspace_scope="workspace-1",
                dataset_id="dataset-1",
                domain=domain,
                item_label=f"{domain} conflict {index}",
                cause="Remote edit conflicts with local edit.",
                local_summary="Local item changed.",
                remote_summary="Remote item changed.",
                source_conflict_key=f"{domain}-{index}",
                conflict_kind="encrypted_content_edit",
                recovery_options={"retry": "available"},
            )

    reviews = repo.list_sync_v2_conflict_reviews(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        dataset_id="dataset-1",
        domains=["notes"],
        limit=2,
    )

    assert [review["source_conflict_key"] for review in reviews] == [
        "notes-2",
        "notes-1",
    ]
    assert {review["domain"] for review in reviews} == {"notes"}


def test_sync_v2_profile_summary_reports_missing_profile(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")

    summary = repo.get_sync_v2_profile_summary(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert summary == {
        "status": "not_configured",
        "profile": None,
        "cursor": None,
        "outbox": {"pending": 0, "dispatched": 0, "by_domain": {}},
        "identity_map": {"total": 0, "by_domain": {}},
        "conflicts": {"count": 0, "latest": []},
        "last_mirror_report": None,
    }


def test_sync_v2_profile_summary_scopes_none_principal_and_workspace_exactly(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
    )
    repo.record_identity_mapping(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
        domain="notes",
        entity_type="note",
        local_entity_id="local-note-1",
        remote_entity_id="remote-note-1",
        mapping_status="confirmed",
    )
    repo.record_identity_mapping(
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

    summary = repo.get_sync_v2_profile_summary(
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
    )

    assert summary["identity_map"] == {
        "total": 1,
        "confirmed": 1,
        "by_domain": {"notes": {"confirmed": 1}},
    }


def test_domain_eligibility_defaults_to_not_eligible_and_persists_override(tmp_path):
    db_path = tmp_path / "sync_state.db"
    repo = SyncStateRepository(db_path)

    default = repo.get_domain_eligibility("writing")
    repo.set_domain_eligibility(
        domain="notes",
        sync_eligible=True,
        write_enabled=False,
        reason_codes=("dry_run_only", "identity_ready"),
        details={"mode": "read_only_mirror"},
    )
    repo.close()

    reopened = SyncStateRepository(db_path)
    notes = reopened.get_domain_eligibility("notes")

    assert default["domain"] == "writing"
    assert default["sync_eligible"] is False
    assert default["write_enabled"] is False
    assert default["reason_codes"] == ("not_eligible",)
    assert notes["sync_eligible"] is True
    assert notes["write_enabled"] is False
    assert notes["reason_codes"] == ("dry_run_only", "identity_ready")
    assert notes["details"] == {"mode": "read_only_mirror"}


def test_clear_server_profile_state_removes_only_scoped_sync_rows(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.record_identity_mapping(
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
    repo.record_identity_mapping(
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
    report = repo.record_mirror_report(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        report={"dry_run": True, "write_enabled": False},
    )
    repo.set_sync_profile_state(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        last_mirror_report_id=report["report_id"],
    )
    repo.set_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        remote_collection="notes",
        cursor="cursor-user-a",
    )
    repo.set_domain_eligibility(
        domain="notes",
        sync_eligible=True,
        write_enabled=False,
        reason_codes=("dry_run_only",),
    )

    repo.clear_server_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
    )

    assert (
        repo.list_identity_mappings(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
        )
        == []
    )
    assert (
        len(
            repo.list_identity_mappings(
                server_profile_id="server-a",
                authenticated_principal_id="user-b",
            )
        )
        == 1
    )
    assert (
        repo.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            domain="notes",
            remote_collection="notes",
        ).cursor
        is None
    )
    assert (
        repo.get_sync_profile_state(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
        )
        is None
    )
    assert repo.list_mirror_reports(domain="notes") == []
    assert repo.get_domain_eligibility("notes")["sync_eligible"] is True
