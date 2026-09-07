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
    """Build a registry backed by a fresh ``WorkspaceDB`` under ``tmp_path``.

    Real callers always place ``WorkspaceDB`` under ``get_user_data_dir()``,
    which is created as a side effect before the path is ever used. The
    private-paths hardening removed ``BaseDB``'s own parent-directory
    auto-creation (see ``P06`` in the SQLite private-owner inventory), so
    opening a database whose containing directory does not yet exist now
    raises ``PrivatePathError`` instead of silently creating it. Callers of
    this helper may pass a subdirectory that has not been created yet (e.g.
    a second, isolated registry root), so create it here the same way a real
    caller's directory is guaranteed to exist.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    # TASK-857: an ordinary subdirectory, not `tmp_path` itself -- this
    # suite's own autouse HOME-redirection fixture (Tests/conftest.py)
    # nests the test's effective config directory under `tmp_path`
    # (`tmp_path/test_data/config`), so `tmp_path` is now correctly
    # rejected as a folder-binding root in its own right; that is not what
    # this assertion is testing (unknown-workspace precedence), so use an
    # ordinary carved-out folder instead.
    ok_folder = tmp_path / "ok"
    ok_folder.mkdir()
    with pytest.raises(WorkspaceNotFound):
        service.add_folder_binding("ws-missing", ok_folder)


def test_add_folder_binding_rejects_sensitive_paths(
    service: LocalWorkspaceRegistryService,
) -> None:
    """TASK-857: folder binding must consult the sensitive-path denylist,
    not just the filesystem root and home directory. Every candidate here
    is derived from the app's own accessors (never a re-spelled literal),
    per the AC, so this can't silently drift the way the read-time
    denylist's own literals once did.
    """
    from tldw_chatbook import config as app_config

    # Case 1: root IS a protected directory (this app's own config dir).
    config_dir = app_config._get_effective_config_path().parent
    config_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", config_dir)

    # Case 2: root IS a protected directory (this app's own data dir).
    user_data_dir = app_config.get_user_data_dir()
    user_data_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", user_data_dir)

    # Case 3: root is NESTED INSIDE a protected directory -- a subdirectory
    # of get_user_data_dir() must be refused too, even though it doesn't
    # look sensitive by name alone.
    nested = user_data_dir / "some_subdir"
    nested.mkdir(parents=True, exist_ok=True)
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", nested)

    # Case 4: root is coarse enough to CONTAIN a protected directory --
    # the reverse direction, where the root itself doesn't look sensitive.
    ancestor_of_user_data_dir = user_data_dir.parent
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", ancestor_of_user_data_dir)


def test_add_folder_binding_rejection_names_the_protected_path(
    service: LocalWorkspaceRegistryService,
) -> None:
    """The rejection must be actionable -- name what was protected -- not a
    silent failure or a bare/opaque exception."""
    from tldw_chatbook import config as app_config

    user_data_dir = app_config.get_user_data_dir()
    user_data_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(WorkspaceRegistryServiceError) as excinfo:
        service.add_folder_binding("ws-a", user_data_dir)

    assert str(user_data_dir) in str(excinfo.value)


def test_add_folder_binding_still_binds_an_ordinary_project_folder(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    """TASK-857 AC#3: no regression on the common case -- an ordinary
    project folder unrelated to any of this app's own state still binds."""
    folder = tmp_path / "ordinary-project"
    folder.mkdir()

    binding = service.add_folder_binding("ws-a", folder)

    assert binding.locator == str(folder.resolve())


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
