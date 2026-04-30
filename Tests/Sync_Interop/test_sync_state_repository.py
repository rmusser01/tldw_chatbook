from __future__ import annotations

import pytest

from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


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
