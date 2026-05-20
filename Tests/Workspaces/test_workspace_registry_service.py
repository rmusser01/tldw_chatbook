"""Local workspace registry persistence tests."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import (
    LocalWorkspaceRegistryService,
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceRuntimeBinding,
)


def build_test_registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    return LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
    )


def test_registry_persists_active_workspace(tmp_path: Path) -> None:
    service = build_test_registry(tmp_path)

    service.create_workspace(workspace_id="ws-a", name="Local Research")
    service.set_active_workspace("ws-a")

    reloaded = build_test_registry(tmp_path)
    active = reloaded.get_active_workspace()

    assert active is not None
    assert active.workspace_id == "ws-a"
    assert active.active is True


def test_registry_links_note_without_hiding_other_workspaces(tmp_path: Path) -> None:
    service = build_test_registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")

    service.link_membership("ws-a", item_type="note", item_id="note-1", role="source")
    service.link_membership("ws-b", item_type="note", item_id="note-1", role="reference")

    memberships = service.get_item_memberships("note", "note-1")

    assert {membership.workspace_id for membership in memberships} == {"ws-a", "ws-b"}


def test_registry_persists_runtime_bindings_without_secrets(tmp_path: Path) -> None:
    service = build_test_registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Workspace A")

    service.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="ws-a",
            binding_id="binding-1",
            binding_kind=RuntimeBindingKind.GIT_WORKTREE,
            label="Repo",
            locator="/tmp/repo",
            status=RuntimeBindingStatus.INSPECT_ONLY,
            metadata={"branch": "dev", "token": "secret"},
        )
    )

    bindings = service.list_runtime_bindings("ws-a")

    assert len(bindings) == 1
    assert bindings[0].metadata == {"branch": "dev"}


def test_registry_lists_workspaces_deterministically(tmp_path: Path) -> None:
    service = build_test_registry(tmp_path)

    service.create_workspace(workspace_id="ws-b", name="B")
    service.create_workspace(workspace_id="ws-a", name="A")

    assert [workspace.workspace_id for workspace in service.list_workspaces()] == [
        "ws-b",
        "ws-a",
    ]
