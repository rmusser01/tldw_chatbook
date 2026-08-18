from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
    validate_folder_binding_path,
)


def test_valid_directory_resolves(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    assert validate_folder_binding_path(project) == project.resolve()


def test_missing_directory_rejected(tmp_path):
    with pytest.raises(WorkspaceRegistryServiceError, match="does not exist"):
        validate_folder_binding_path(tmp_path / "nope")


def test_filesystem_root_rejected():
    with pytest.raises(WorkspaceRegistryServiceError, match="filesystem root"):
        validate_folder_binding_path(Path("/"))


def test_home_directory_rejected():
    with pytest.raises(WorkspaceRegistryServiceError, match="home directory"):
        validate_folder_binding_path(Path.home())


def test_duplicate_locator_rejected(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(WorkspaceRegistryServiceError, match="already bound"):
        validate_folder_binding_path(project, [str(project.resolve())])


def test_nested_inside_existing_rejected(tmp_path):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    with pytest.raises(WorkspaceRegistryServiceError, match="inside the already-bound"):
        validate_folder_binding_path(child, [str(parent.resolve())])


def test_existing_inside_candidate_rejected(tmp_path):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    with pytest.raises(WorkspaceRegistryServiceError, match="remove it first"):
        validate_folder_binding_path(parent, [str(child.resolve())])


def test_sensitive_conflict_rejected(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    import tldw_chatbook.Workspaces.registry_service as rs

    monkeypatch.setattr(
        rs, "find_root_binding_conflict", lambda p: Path("/protected")
    )
    with pytest.raises(WorkspaceRegistryServiceError, match="protected path"):
        validate_folder_binding_path(project)


def test_add_folder_binding_still_enforces(tmp_path):
    db = WorkspaceDB(tmp_path / "ws.sqlite", client_id="validator-tests")
    service = LocalWorkspaceRegistryService(db)
    service.create_workspace(workspace_id="workspace-local-1", name="Workspace 1")
    project = tmp_path / "project"
    project.mkdir()
    binding = service.add_folder_binding("workspace-local-1", project)
    assert binding.locator == str(project.resolve())
    with pytest.raises(WorkspaceRegistryServiceError, match="already bound"):
        service.add_folder_binding("workspace-local-1", project)
