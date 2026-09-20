"""Registry CRUD for per-binding exclusions (spec §1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.models import BindingExclusion
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    binding_exclusion_entries,
)
from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError


@pytest.fixture()
def registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    db = WorkspaceDB(tmp_path / "workspaces.db")
    try:
        service = LocalWorkspaceRegistryService(db)
        service.create_workspace(workspace_id="ws-excl", name="Exclusions WS")
        yield service
    finally:
        db.close()


@pytest.fixture()
def binding(registry: LocalWorkspaceRegistryService, tmp_path: Path) -> str:
    root = tmp_path / "repo"
    (root / "secrets").mkdir(parents=True)
    (root / "secrets" / "key.pem").write_text("k")
    (root / ".env.prod").write_text("v")
    binding = registry.add_folder_binding("ws-excl", root, allow_write=True)
    return binding.binding_id


def test_add_and_list_roundtrip(registry, binding):
    updated = registry.add_binding_exclusion("ws-excl", binding, "secrets")
    entries = registry.list_binding_exclusions(binding)
    assert entries == (BindingExclusion(path="secrets", kind="directory", added_at=entries[0].added_at),)
    assert binding_exclusion_entries(updated)[0].path == "secrets"


def test_kind_inferred_from_disk(registry, binding):
    registry.add_binding_exclusion("ws-excl", binding, ".env.prod")
    assert registry.list_binding_exclusions(binding)[0].kind == "file"


def test_nonexistent_path_defaults_to_directory(registry, binding):
    registry.add_binding_exclusion("ws-excl", binding, "build/output")
    assert registry.list_binding_exclusions(binding)[0].kind == "directory"
    assert registry.list_binding_exclusions(binding)[0].path == "build/output"


def test_remove(registry, binding):
    registry.add_binding_exclusion("ws-excl", binding, "secrets")
    registry.remove_binding_exclusion("ws-excl", binding, "secrets")
    assert registry.list_binding_exclusions(binding) == ()


@pytest.mark.parametrize("bad", ["", ".", "./", "/etc/passwd", "~/.ssh", "../outside", "a/../../outside"])
def test_invalid_paths_rejected(registry, binding, bad):
    with pytest.raises(WorkspaceRegistryServiceError):
        registry.add_binding_exclusion("ws-excl", binding, bad)


def test_symlink_escape_rejected(registry, binding, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "repo" / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceRegistryServiceError):
        registry.add_binding_exclusion("ws-excl", binding, "link/inner")


def test_duplicate_casefold_rejected(registry, binding):
    registry.add_binding_exclusion("ws-excl", binding, "secrets")
    with pytest.raises(WorkspaceRegistryServiceError):
        registry.add_binding_exclusion("ws-excl", binding, "Secrets/")


def test_cap_enforced(registry, binding):
    for i in range(200):
        registry.add_binding_exclusion("ws-excl", binding, f"e{i}")
    with pytest.raises(WorkspaceRegistryServiceError):
        registry.add_binding_exclusion("ws-excl", binding, "e200")


def test_wrong_workspace_rejected(registry, binding):
    registry.create_workspace(workspace_id="ws-other", name="Other")
    with pytest.raises(WorkspaceRegistryServiceError):
        registry.add_binding_exclusion("ws-other", binding, "secrets")
