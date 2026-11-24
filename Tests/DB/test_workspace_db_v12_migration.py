"""Copy receipts upgrade with the WorkspaceDB stamp in one transaction."""

import sqlite3

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService


def make_v11(path, monkeypatch):
    with monkeypatch.context() as historical:
        historical.setattr(
            WorkspaceDB, "_migrate_v11_to_v12", lambda self: None, raising=False
        )
        db = WorkspaceDB(path)
        registry = LocalWorkspaceRegistryService(db)
        registry.create_workspace(workspace_id="study", name="Study")
        registry.link_membership("study", item_type="media", item_id="42")
        assert db.get_schema_version() == 11
        db.close()


def test_copy_receipt_upgrade_preserves_existing_workspace_memberships(
    tmp_path, monkeypatch
):
    path = tmp_path / "workspace.sqlite"
    make_v11(path, monkeypatch)
    db = WorkspaceDB(path)
    try:
        assert db.get_schema_version() == 12
        with db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM research_copy_operations"
                ).fetchone()[0]
                == 0
            )
            assert (
                conn.execute("SELECT COUNT(*) FROM research_copy_items").fetchone()[0]
                == 0
            )
        assert (
            LocalWorkspaceRegistryService(db)
            .list_workspace_memberships("study")[0]
            .item_id
            == "42"
        )
    finally:
        db.close()


def test_copy_receipt_migration_rolls_back_if_stamp_fails(tmp_path, monkeypatch):
    path = tmp_path / "workspace.sqlite"
    make_v11(path, monkeypatch)
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TRIGGER refuse_v12 BEFORE INSERT ON schema_version
            WHEN NEW.version = 12 BEGIN SELECT RAISE(ABORT, 'version failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="version failure"):
        WorkspaceDB(path)
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 11
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'research_copy_%'"
            ).fetchone()[0]
            == 0
        )
