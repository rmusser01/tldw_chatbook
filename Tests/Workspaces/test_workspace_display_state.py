"""Console workspace context display-state tests."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import (
    LocalWorkspaceRegistryService,
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceAuthority,
    WorkspaceRuntimeBinding,
    WorkspaceSyncStatus,
)
from tldw_chatbook.Workspaces.display_state import (
    ConsoleWorkspaceConversationRow,
    build_console_workspace_state,
)


def _registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    return LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
    )


def test_console_workspace_state_explains_missing_service() -> None:
    state = build_console_workspace_state(
        registry_service=None,
        current_conversation=None,
    )

    assert state.heading == "Convos & Workspaces"
    assert state.workspace_label == "No workspace selected"
    assert state.change_workspace_enabled is False
    assert state.new_conversation_enabled is False
    assert "service not ready" in state.recovery_copy.lower()


def test_console_workspace_state_explains_no_active_workspace(tmp_path: Path) -> None:
    service = _registry(tmp_path)

    state = build_console_workspace_state(
        registry_service=service,
        current_conversation=None,
    )

    assert state.workspace_label == "No workspace selected"
    assert state.authority_label == "Authority: local registry ready"
    assert state.change_workspace_enabled is False
    assert state.new_conversation_enabled is False
    assert "create or select a workspace" in state.recovery_copy.lower()


def test_console_workspace_state_reports_active_workspace_and_runtime(tmp_path: Path) -> None:
    service = _registry(tmp_path)
    service.create_workspace(
        workspace_id="ws-a",
        name="Research Sprint",
        authority=WorkspaceAuthority.LOCAL_ONLY,
        sync_status=WorkspaceSyncStatus.READY,
    )
    service.set_active_workspace("ws-a")
    service.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="ws-a",
            binding_id="binding-1",
            binding_kind=RuntimeBindingKind.GIT_WORKTREE,
            label="Repo worktree",
            locator="/tmp/repo",
            status=RuntimeBindingStatus.INSPECT_ONLY,
        )
    )

    state = build_console_workspace_state(
        registry_service=service,
        current_conversation="conv-1",
        conversations=(
            ConsoleWorkspaceConversationRow(
                conversation_id="conv-1",
                title="Planning thread",
                status="active",
            ),
        ),
    )

    assert state.workspace_label == "Workspace: Research Sprint"
    assert state.authority_label == "Authority: local-only"
    assert state.sync_label == "Sync: ready"
    assert state.runtime_label == "Runtime: 1 binding, 0 ready"
    assert state.conversation_rows[0].title == "Planning thread"
    assert state.conversation_rows[0].selected is True
    assert state.change_workspace_enabled is False
    assert state.new_conversation_enabled is False
    assert "later slice" in state.new_conversation_recovery.lower()


def test_console_workspace_state_derives_conversation_rows_from_memberships(
    tmp_path: Path,
) -> None:
    service = _registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Research Sprint")
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-1",
        role="workspace-thread",
        title="Planning thread",
    )

    state = build_console_workspace_state(
        registry_service=service,
        current_conversation="conv-1",
    )

    assert len(state.conversation_rows) == 1
    assert state.conversation_rows[0].conversation_id == "conv-1"
    assert state.conversation_rows[0].title == "Planning thread"
    assert state.conversation_rows[0].selected is True
