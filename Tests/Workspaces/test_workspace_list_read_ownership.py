"""Finite workspace listings preserve the lifetime of their actual SQL owner."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Tool_Packs.service import _WorkspaceReferences
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
)

pytestmark = pytest.mark.bootstrap_profile


def _read(service, route):
    return (
        _WorkspaceReferences(service).capture()
        if route == "tool_pack"
        else service.list_workspaces(include_archived=True)
    )


@pytest.mark.parametrize("route", ["registry", "tool_pack"])
@pytest.mark.parametrize("sql_failure", [False, True])
def test_workspace_listing_retires_new_worker_handle(tmp_path, route, sql_failure):
    db = WorkspaceDB(tmp_path / "finite-list.sqlite", client_id="finite-list")
    service = LocalWorkspaceRegistryService(db)
    try:
        service.create_workspace(workspace_id="listed", name="Listed")
        caller = db._held_connection()
        if sql_failure:
            with db.transaction() as connection:
                connection.execute("ALTER TABLE workspace_records RENAME TO hidden_records")
        baseline = tuple(db._maintenance_participant.connections)

        def read():
            assert getattr(db._thread_local, "conn", None) is None
            try:
                if sql_failure:
                    with pytest.raises(WorkspaceRegistryServiceError):
                        _read(service, route)
                else:
                    rows = _read(service, route)
                    assert any(row.workspace_id == "listed" for row in rows)
                assert getattr(db._thread_local, "conn", None) is None
                assert tuple(db._maintenance_participant.connections) == baseline
            finally:
                # Retire the original worker cache even on the causal RED.
                db.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(read).result(timeout=10)
        assert db._held_connection() is caller
        assert caller.execute("SELECT 1").fetchone()[0] == 1
    finally:
        db.close()


@pytest.mark.parametrize("route", ["registry", "tool_pack"])
def test_workspace_listing_preserves_borrowed_transaction(tmp_path, route):
    db = WorkspaceDB(tmp_path / "borrowed-list.sqlite", client_id="borrowed-list")
    service = LocalWorkspaceRegistryService(db)
    try:
        service.create_workspace(workspace_id="listed", name="Original")
        with pytest.raises(RuntimeError, match="rollback caller"):
            with db.transaction() as connection:
                connection.execute(
                    "UPDATE workspace_records SET name = ? WHERE workspace_id = ?",
                    ("Uncommitted", "listed"),
                )
                rows = _read(service, route)
                assert next(row for row in rows if row.workspace_id == "listed").name == "Uncommitted"
                assert db._held_connection() is connection
                assert connection.in_transaction
                raise RuntimeError("rollback caller")
        assert service.get_workspace("listed").name == "Original"
    finally:
        db.close()


@pytest.mark.parametrize("owner", ["memory", "custom"])
def test_workspace_listing_preserves_excluded_connection_owner(tmp_path, owner):
    class CustomWorkspaceDB(WorkspaceDB):
        pass

    db = (
        WorkspaceDB(":memory:", client_id="memory-list")
        if owner == "memory"
        else CustomWorkspaceDB(tmp_path / "custom-list.sqlite", client_id="custom-list")
    )
    service = LocalWorkspaceRegistryService(db)
    try:
        service.create_workspace(workspace_id="listed", name="Retained")
        connection = db._held_connection()
        for route in ("registry", "tool_pack"):
            assert any(row.workspace_id == "listed" for row in _read(service, route))
            assert db._held_connection() is connection
            assert connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        db.close()
