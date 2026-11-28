import sqlite3

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService


@pytest.mark.parametrize("upgrade", [False, True])
def test_output_provenance_schema_preserves_existing_workspace_source_links(
    tmp_path, monkeypatch, upgrade
):
    path = tmp_path / "workspaces.sqlite"
    if upgrade:
        with monkeypatch.context() as historical:
            historical.setattr(
                WorkspaceDB, "_migrate_v10_to_v11", lambda self: None, raising=False
            )
            historical.setattr(
                WorkspaceDB, "_migrate_v11_to_v12", lambda self: None, raising=False
            )
            legacy = WorkspaceDB(path)
            rows = LocalWorkspaceRegistryService(legacy)
            rows.create_workspace(workspace_id="reading", name="Reading")
            with legacy.transaction() as conn:
                conn.execute(
                    """INSERT INTO workspace_memberships
                       (membership_id, workspace_id, item_type, item_id, role, transfer_policy, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        "legacy-link",
                        "reading",
                        "media",
                        "42",
                        "source",
                        "reference",
                        "2026-11-24T00:00:00+00:00",
                    ),
                )
            assert legacy.get_schema_version() == 10
            legacy.close()

    db = WorkspaceDB(path)
    try:
        assert db.get_schema_version() == WorkspaceDB._CURRENT_SCHEMA_VERSION
        with db.connection() as conn:
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(workspace_memberships)")
            }
        assert {"metadata_json", "updated_at"} <= columns
        if upgrade:
            [source] = LocalWorkspaceRegistryService(db).list_workspace_memberships(
                "reading"
            )
            assert source.membership_id == "legacy-link"
            assert source.metadata == {}
            assert source.updated_at is None
            with pytest.raises(sqlite3.IntegrityError):
                with db.transaction() as conn:
                    conn.execute(
                        "UPDATE workspace_memberships SET metadata_json = ? WHERE membership_id = ?",
                        ("{}", source.membership_id),
                    )
    finally:
        db.close()


def test_output_provenance_migration_rolls_back_both_columns_if_stamp_fails(tmp_path):
    path = tmp_path / "workspaces.sqlite"
    db = WorkspaceDB(path)
    with db.transaction() as conn:
        conn.execute("ALTER TABLE workspace_memberships DROP COLUMN metadata_json")
        conn.execute("ALTER TABLE workspace_memberships DROP COLUMN updated_at")
        conn.execute("DELETE FROM schema_version WHERE version >= 11")
        conn.execute("""CREATE TRIGGER refuse_v11 BEFORE INSERT ON schema_version
            WHEN NEW.version = 11 BEGIN SELECT RAISE(ABORT, 'version failure'); END""")
    db.close()

    with pytest.raises(sqlite3.IntegrityError, match="version failure"):
        WorkspaceDB(path)
    conn = sqlite3.connect(path)
    try:
        assert (
            conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 10
        )
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(workspace_memberships)")
        }
        assert "metadata_json" not in columns
        assert "updated_at" not in columns
    finally:
        conn.close()
