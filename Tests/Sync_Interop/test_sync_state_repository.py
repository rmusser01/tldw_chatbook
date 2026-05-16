from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.Sync_Interop import sync_state_repository as sync_state_repository_module


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
    assert mapping.local_side_key == "server:server-a:user-a:workspace-1:notes:note:local:local-note-1"
    assert mapping.remote_side_key == "server:server-a:user-a:workspace-1:notes:note:remote:remote-note-1"
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
    assert conflicts[0]["source_scope_key"] == "server:server-a:user-a:workspace-1:notes:note"


def test_identity_mapping_validation_allows_orphans_but_not_confirmed_missing_side(tmp_path):
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
    with pytest.raises(ValueError, match="confirmed mapping requires local and remote entity IDs"):
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

    assert reopened.get_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        remote_collection="notes",
    ).cursor == "cursor-a"
    assert reopened.get_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-b",
        workspace_scope="workspace-1",
        domain="notes",
        remote_collection="notes",
    ).cursor == "cursor-b"
    assert reopened.list_mirror_reports(domain="notes")[0]["report_id"] == report["report_id"]
    assert reopened.list_mirror_reports(domain="notes")[0]["report"]["write_enabled"] is False


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

    assert reopened.get_sync_profile_state(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )["last_mirror_report_id"] == report["report_id"]
    assert reopened.get_sync_profile_state(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )["last_error"] == "remote_unavailable"
    assert reopened.get_sync_profile_state(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-b",
        workspace_scope="workspace-1",
    ) is None


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


def test_sync_v2_schema_migration_updates_legacy_schema_version(tmp_path):
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

    repo = SyncStateRepository(db_path)
    with repo._get_connection() as conn:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(sync_profile_state)").fetchall()
        }
        outbox = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sync_v2_local_outbox'"
        ).fetchone()
        schema_version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        schema_versions = [
            row[0]
            for row in conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
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
    assert schema_version == 2
    assert schema_versions == [2]


def test_sync_v2_profile_column_migration_validates_column_identifiers(tmp_path, monkeypatch):
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

    def record_validated_column(column_name: str, table_name: str | None = None) -> bool:
        calls.append((column_name, table_name))
        return True

    monkeypatch.setattr(
        sync_state_repository_module,
        "validate_column_name",
        record_validated_column,
    )

    SyncStateRepository(db_path)

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
    assert [entry["client_envelope_id"] for entry in dispatched] == [accepted.client_envelope_id]
    assert dispatched[0]["attempt_count"] == 1
    assert [entry["client_envelope_id"] for entry in pending_after] == [
        rejected.client_envelope_id,
        conflicted.client_envelope_id,
    ]
    assert [entry["attempt_count"] for entry in pending_after] == [1, 1]
    assert pending_after[0]["last_error"]["error_code"] == "stale_base"
    assert pending_after[1]["last_error"]["error_code"] == "conflict"


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
        "failed": 0,
        "by_domain": {"notes": {"pending": 2, "dispatched": 1, "failed": 0}},
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
        "outbox": {"pending": 0, "dispatched": 0, "failed": 0, "by_domain": {}},
        "identity_map": {"total": 0, "by_domain": {}},
        "conflicts": {"count": 0, "latest": []},
        "last_mirror_report": None,
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

    assert repo.list_identity_mappings(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
    ) == []
    assert len(
        repo.list_identity_mappings(
            server_profile_id="server-a",
            authenticated_principal_id="user-b",
        )
    ) == 1
    assert repo.get_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        remote_collection="notes",
    ).cursor is None
    assert repo.get_sync_profile_state(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    ) is None
    assert repo.list_mirror_reports(domain="notes") == []
    assert repo.get_domain_eligibility("notes")["sync_eligible"] is True
