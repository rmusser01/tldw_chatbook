"""Empty frozen authority must not admit an unnecessary registry read."""

from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Tools import workspace_file_roots
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService


@pytest.mark.parametrize("as_iterator", [False, True])
def test_empty_frozen_authority_does_not_construct_default_database(
    tmp_path, monkeypatch, as_iterator
) -> None:
    """No authority means no default database creation, even for an iterator."""
    database_path = tmp_path / "unneeded.sqlite"
    monkeypatch.setattr(workspace_file_roots, "_default_registry_instance", None)
    monkeypatch.setattr(
        "tldw_chatbook.config.get_workspaces_db_path", lambda: database_path
    )
    authority = iter(()) if as_iterator else ()
    try:
        assert (
            workspace_file_roots.frozen_workspace_roots("workspace-a", authority) == ()
        )
        assert not database_path.exists()
    finally:
        owned = workspace_file_roots._default_registry_instance
        if owned is not None:
            assert Path(owned.db.db_path).resolve() == database_path.resolve()
            owned.db.close()


@pytest.mark.parametrize("as_iterator", [False, True])
def test_empty_frozen_authority_does_not_reopen_supplied_database(
    tmp_path, as_iterator
) -> None:
    """An empty captured maximum cannot reopen a supplied registry handle."""
    database = WorkspaceDB(tmp_path / "provided.sqlite", client_id="empty-authority")
    registry = LocalWorkspaceRegistryService(database)
    database.close()
    assert database._thread_local.conn is None
    authority = iter(()) if as_iterator else ()
    try:
        assert (
            workspace_file_roots.frozen_workspace_roots(
                "workspace-a", authority, registry=registry
            )
            == ()
        )
        assert database._thread_local.conn is None
    finally:
        database.close()
