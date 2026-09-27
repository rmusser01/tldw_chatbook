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


def test_add_folder_binding_path_validation_before_db_lookup(tmp_path):
    """Verify path validation happens before list_folder_bindings is called.

    Regression test (evaluation-order guard): with broken ordering, the call
    list_folder_bindings("") would raise ValueError("workspace_id is required")
    before path validation could reject the bad path. With correct ordering,
    _validate_folder_path_rules runs first and raises WorkspaceRegistryServiceError
    about the missing directory.
    """
    db = WorkspaceDB(tmp_path / "ws.sqlite", client_id="validator-tests")
    service = LocalWorkspaceRegistryService(db)
    bad_path = tmp_path / "does_not_exist"
    # Empty workspace_id would fail in list_folder_bindings if called first
    with pytest.raises(WorkspaceRegistryServiceError, match="does not exist"):
        service.add_folder_binding("", bad_path)


# ---------------------------------------------------------------------------
# Phase 3c (task 17): exclusions on ssh-filesystem bindings
# ---------------------------------------------------------------------------


def _ssh_service_and_binding(tmp_path):
    from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

    db = WorkspaceDB(tmp_path / "ws.sqlite", client_id="validator-tests")
    service = LocalWorkspaceRegistryService(db)
    service.create_workspace(workspace_id="workspace-ssh-1", name="SSH WS")
    binding = service.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="workspace-ssh-1",
            binding_id="ssh-binding-1",
            binding_kind="ssh-filesystem",
            label="devbox",
            locator="ssh://devbox/srv/www",
            status="ready",
            metadata={"access": "ro"},
            created_at="2026-09-24T00:00:00Z",
            updated_at="2026-09-24T00:00:00Z",
        )
    )
    return service, binding


def test_add_binding_exclusion_accepts_ssh_binding_storing_raw_relative(tmp_path):
    """SSH exclusions store RAW relative strings: no laptop resolve, no
    laptop stat (the remote host owns the filesystem). Kind defaults to
    "directory" -- subtree semantics cover not-yet-existing targets and
    files alike; real validation is worker-side at apply time (documented
    choice: no ping-style validation call, keeping the edit offline-safe
    exactly like add-binding's advisory probe)."""
    from tldw_chatbook.Workspaces.registry_service import binding_exclusion_entries

    service, binding = _ssh_service_and_binding(tmp_path)
    updated = service.add_binding_exclusion(
        "workspace-ssh-1", "ssh-binding-1", "secrets"
    )

    entries = binding_exclusion_entries(updated)
    assert [entry.path for entry in entries] == ["secrets"]
    assert entries[0].kind == "directory"


def test_add_binding_exclusion_ssh_rejects_escaping_paths(tmp_path):
    service, binding = _ssh_service_and_binding(tmp_path)
    for bad in ("../escape", "/abs", "~", "a/../../b", ""):
        with pytest.raises(WorkspaceRegistryServiceError):
            service.add_binding_exclusion("workspace-ssh-1", "ssh-binding-1", bad)


def test_add_binding_exclusion_ssh_rejects_root_itself(tmp_path):
    """The locator's own path is the binding root: not excludable (remove
    the binding instead), lexically decidable without any stat."""
    service, binding = _ssh_service_and_binding(tmp_path)
    with pytest.raises(WorkspaceRegistryServiceError, match="whole root"):
        service.add_binding_exclusion("workspace-ssh-1", "ssh-binding-1", ".")


def test_add_binding_exclusion_ssh_rejects_duplicates(tmp_path):
    service, binding = _ssh_service_and_binding(tmp_path)
    service.add_binding_exclusion("workspace-ssh-1", "ssh-binding-1", "secrets")
    with pytest.raises(WorkspaceRegistryServiceError, match="already excluded"):
        service.add_binding_exclusion("workspace-ssh-1", "ssh-binding-1", "SECRETS")


def test_remove_binding_exclusion_works_for_ssh_bindings(tmp_path):
    from tldw_chatbook.Workspaces.registry_service import binding_exclusion_entries

    service, binding = _ssh_service_and_binding(tmp_path)
    service.add_binding_exclusion("workspace-ssh-1", "ssh-binding-1", "secrets")

    updated = service.remove_binding_exclusion(
        "workspace-ssh-1", "ssh-binding-1", "secrets"
    )

    assert binding_exclusion_entries(updated) == ()


def test_add_binding_exclusion_local_binding_still_stats_the_laptop(tmp_path):
    """Local bindings keep the exact pre-Phase-3c behavior: resolve(strict)
    against the laptop root and file/dir inference from the laptop disk."""
    from tldw_chatbook.Workspaces.registry_service import binding_exclusion_entries

    db = WorkspaceDB(tmp_path / "ws.sqlite", client_id="validator-tests")
    service = LocalWorkspaceRegistryService(db)
    service.create_workspace(workspace_id="workspace-local-2", name="Local 2")
    project = tmp_path / "project"
    project.mkdir()
    note = project / "notes.txt"
    note.write_text("x", encoding="utf-8")
    binding = service.add_folder_binding("workspace-local-2", project)

    with_file = service.add_binding_exclusion(
        "workspace-local-2", binding.binding_id, "notes.txt"
    )
    with_dir = service.add_binding_exclusion(
        "workspace-local-2", binding.binding_id, "not-yet-created"
    )

    file_kinds = {
        entry.path: entry.kind for entry in binding_exclusion_entries(with_file)
    }
    dir_kinds = {
        entry.path: entry.kind for entry in binding_exclusion_entries(with_dir)
    }
    assert file_kinds["notes.txt"] == "file"
    assert dir_kinds["not-yet-created"] == "directory"
