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

# Targeted-run guard (known environmental failure): the first real
# ``add_folder_binding`` call lazily imports the RAG config-profile chain,
# whose module-level internal-prompt reads run the config bootstrap under
# the PER-TEST sandboxed ``TLDW_CONFIG_PATH`` while the config raw
# participant was opened against the collection-time bootstrap path --
# aborting with ``RecoveryRequired: raw_source_selection_changed``. A full
# suite run imports this chain during collection (before the per-test
# sandbox diverges), which is why only single-file runs hit it. Importing
# it here, under the same collection-time env, keeps targeted runs as
# green as full runs without touching any behavior under test.
import tldw_chatbook.RAG_Search.config_profiles  # noqa: F401


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


@pytest.mark.parametrize(
    "bad",
    ["", ".", "./", "/etc/passwd", "~/.ssh", "../outside", "a/../../outside", "a\x00b"],
)
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


def test_concurrent_adds_do_not_lose_exclusions(registry, binding):
    """Finding C (PR #2767): the exclusion read-modify-write is serialized.

    ``add_binding_exclusion`` reads the binding, mutates the entry list,
    and saves; without a lock, two concurrent adds each read the stale
    entry list and the last write silently drops the other's entry. A
    barrier releases both writers together so the race is not left to the
    scheduler, then BOTH exclusions must survive.
    """
    import threading

    paths = ("secrets", ".env.prod", "build", "dist")
    barrier = threading.Barrier(len(paths))
    failures: list[Exception] = []

    def add(path: str) -> None:
        try:
            barrier.wait(timeout=10)
            registry.add_binding_exclusion("ws-excl", binding, path)
        except Exception as exc:  # noqa: BLE001 -- reported via assertion below
            failures.append(exc)

    threads = [threading.Thread(target=add, args=(p,)) for p in paths]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)
    assert failures == []
    stored = {entry.path for entry in registry.list_binding_exclusions(binding)}
    assert stored == set(paths), sorted(stored)
