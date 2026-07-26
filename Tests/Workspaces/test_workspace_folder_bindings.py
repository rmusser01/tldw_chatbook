"""Folder-binding service methods (spec 2026-07-26 settings-workspaces §2)."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID, LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.registry_service import (
    BindingNotFound,
    WorkspaceNotFound,
    WorkspaceRegistryServiceError,
)


def build_registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    return LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="folder-tests")
    )


@pytest.fixture()
def service(tmp_path: Path) -> LocalWorkspaceRegistryService:
    registry = build_registry(tmp_path)
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="Client A")
    return registry


def test_add_folder_binding_stores_canonical_ro_binding(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    folder = tmp_path / "project"
    folder.mkdir()

    binding = service.add_folder_binding("ws-a", folder)

    assert binding.locator == str(folder.resolve())
    assert binding.label == "project"
    assert str(binding.binding_kind) in ("local-filesystem", "RuntimeBindingKind.LOCAL_FILESYSTEM")
    assert binding.metadata["access"] == "ro"
    assert str(binding.status) in ("ready", "RuntimeBindingStatus.READY")


def test_add_folder_binding_allow_write(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    folder = tmp_path / "writable"
    folder.mkdir()
    binding = service.add_folder_binding("ws-a", folder, allow_write=True)
    assert binding.metadata["access"] == "rw"


def test_add_folder_binding_validation_matrix(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    missing = tmp_path / "nope"
    a_file = tmp_path / "file.txt"
    a_file.write_text("x")
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", missing)
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", a_file)
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", Path("/"))
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", Path.home())
    with pytest.raises(WorkspaceNotFound):
        service.add_folder_binding("ws-missing", tmp_path)


def test_add_folder_binding_rejects_default_workspace(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    folder = tmp_path / "any"
    folder.mkdir()
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding(DEFAULT_WORKSPACE_ID, folder)


def test_add_folder_binding_rejects_duplicates_and_nesting(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    service.add_folder_binding("ws-a", parent)

    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", parent)  # duplicate
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", child)  # nested under existing

    # And the reverse direction: existing child blocks a new parent root.
    other = build_registry(tmp_path / "second-db")
    other.create_workspace(workspace_id="ws-b", name="Client B")
    other.add_folder_binding("ws-b", child)
    with pytest.raises(WorkspaceRegistryServiceError):
        other.add_folder_binding("ws-b", parent)


def test_list_folder_bindings_recomputes_status(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    folder = tmp_path / "ephemeral"
    folder.mkdir()
    service.add_folder_binding("ws-a", folder)
    folder.rmdir()

    bindings = service.list_folder_bindings("ws-a")
    assert len(bindings) == 1
    assert str(bindings[0].status) in ("missing", "RuntimeBindingStatus.MISSING")


def test_list_folder_bindings_reports_symlink_swap_as_missing(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    """Item F: a bound folder later replaced by a symlink must report MISSING,
    since trusting it would silently widen the root at enforcement time."""
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    service.add_folder_binding("ws-a", real_root)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    shutil.rmtree(real_root)
    real_root.symlink_to(elsewhere)

    bindings = service.list_folder_bindings("ws-a")
    assert len(bindings) == 1
    assert str(bindings[0].status) in ("missing", "RuntimeBindingStatus.MISSING")


def test_remove_and_toggle_access(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    folder = tmp_path / "toggled"
    folder.mkdir()
    binding = service.add_folder_binding("ws-a", folder)

    updated = service.set_folder_binding_access(binding.binding_id, allow_write=True)
    assert updated.metadata["access"] == "rw"

    service.remove_runtime_binding(binding.binding_id)
    assert service.list_folder_bindings("ws-a") == ()
    with pytest.raises(BindingNotFound):
        service.remove_runtime_binding(binding.binding_id)
    with pytest.raises(BindingNotFound):
        service.set_folder_binding_access(binding.binding_id, allow_write=False)
