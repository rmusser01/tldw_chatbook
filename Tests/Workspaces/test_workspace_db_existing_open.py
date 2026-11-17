"""Workspace opens that leave schema creation to the normal owner."""

import sqlite3

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Utils.private_paths import PrivatePathError


def test_workspace_existing_open_reads_initialized_schema_without_ddl(
    tmp_path, monkeypatch
):
    path = tmp_path / "workspaces.db"
    normal = WorkspaceDB(path, "seed")
    normal.close()
    assert path.is_file()
    statements = []
    original = WorkspaceDB._configure_connection

    def configure(self, conn, **kwargs):
        conn.set_trace_callback(statements.append)
        return original(self, conn, **kwargs)

    monkeypatch.setattr(WorkspaceDB, "_configure_connection", configure)
    existing = WorkspaceDB(path, "reader", initialize_schema=False, must_exist=True)
    try:
        assert statements == []
        with existing.connection() as conn:
            assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[
                0
            ] == (WorkspaceDB._CURRENT_SCHEMA_VERSION)
            assert (
                conn.execute("SELECT count(*) FROM workspace_records").fetchone()[0]
                == 0
            )
        assert any(
            "SELECT MAX(version) FROM schema_version" in sql for sql in statements
        )
        assert not any(
            sql.lstrip().upper().startswith(("CREATE ", "ALTER ", "DROP ", "INSERT "))
            for sql in statements
        )
    finally:
        existing.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        sqlite3.Connection.in_transaction.__get__(conn)


@pytest.mark.parametrize("must_exist", [False, True])
def test_schema_suppression_defers_open_and_must_exist_refuses_creation(
    tmp_path, must_exist
):
    path = tmp_path / "missing.db"
    options = {"must_exist": True} if must_exist else {}
    db = WorkspaceDB(path, initialize_schema=False, **options)
    try:
        assert not path.exists()
        if must_exist:
            with pytest.raises(PrivatePathError, match="missing_sqlite_artifact"):
                with db.connection():
                    pytest.fail("missing Workspace registry admitted")
            assert not path.exists()
        else:
            with db.connection() as conn:
                assert conn.execute("SELECT name FROM sqlite_schema").fetchall() == []
            assert path.is_file()
    finally:
        db.close()
