"""Local workspace registry persistence tests."""

from __future__ import annotations

from contextlib import contextmanager
import inspect
from pathlib import Path
from typing import Iterator

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import (
    DEFAULT_WORKSPACE_ID,
    DEFAULT_WORKSPACE_NAME,
    LocalWorkspaceRegistryService,
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceAuthority,
    WorkspaceRuntimeBinding,
    WorkspaceSyncStatus,
)
from tldw_chatbook.Workspaces.registry_service import (
    WorkspaceNotFound,
    WorkspaceRegistryServiceError,
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


def test_registry_ensures_default_workspace_when_none_is_configured(
    tmp_path: Path,
) -> None:
    service = build_test_registry(tmp_path)

    active = service.ensure_default_workspace()

    assert active.workspace_id == DEFAULT_WORKSPACE_ID
    assert active.name == DEFAULT_WORKSPACE_NAME
    assert active.active is True
    assert active.authority is WorkspaceAuthority.LOCAL_ONLY
    assert active.sync_status is WorkspaceSyncStatus.NOT_CONFIGURED
    assert service.list_runtime_bindings(DEFAULT_WORKSPACE_ID) == ()


def test_registry_does_not_replace_user_active_workspace_with_default(
    tmp_path: Path,
) -> None:
    service = build_test_registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.set_active_workspace("ws-a")

    active = service.ensure_default_workspace()

    assert active.workspace_id == "ws-a"
    assert service.get_workspace(DEFAULT_WORKSPACE_ID) is None


def test_registry_rejects_runtime_binding_for_default_workspace(
    tmp_path: Path,
) -> None:
    service = build_test_registry(tmp_path)
    service.ensure_default_workspace()

    with pytest.raises(WorkspaceRegistryServiceError, match="Default workspace"):
        service.save_runtime_binding(
            WorkspaceRuntimeBinding(
                workspace_id=DEFAULT_WORKSPACE_ID,
                binding_id="binding-1",
                binding_kind=RuntimeBindingKind.LOCAL_FILESYSTEM,
                label="Local files",
                locator="/tmp",
                status=RuntimeBindingStatus.READY,
            )
        )


def test_registry_never_exposes_runtime_bindings_for_default_workspace(
    tmp_path: Path,
) -> None:
    db = WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
    service = LocalWorkspaceRegistryService(db)
    service.ensure_default_workspace()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO workspace_runtime_bindings (
                binding_id,
                workspace_id,
                binding_kind,
                label,
                locator,
                status,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "stale-default-binding",
                DEFAULT_WORKSPACE_ID,
                RuntimeBindingKind.LOCAL_FILESYSTEM.value,
                "Unsafe local files",
                "/tmp",
                RuntimeBindingStatus.READY.value,
                "{}",
                "2026-06-08T00:00:00Z",
                "2026-06-08T00:00:00Z",
            ),
        )

    service.ensure_default_workspace()

    assert service.list_runtime_bindings(DEFAULT_WORKSPACE_ID) == ()
    assert service.get_runtime_binding("stale-default-binding") is None


def test_registry_skips_default_runtime_binding_write_when_none_exist(
    tmp_path: Path,
) -> None:
    service = build_test_registry(tmp_path)
    service.ensure_default_workspace()

    @contextmanager
    def fail_on_write_transaction() -> Iterator[None]:
        raise AssertionError("Default runtime binding reads should not write.")
        yield

    service.db.transaction = fail_on_write_transaction  # type: ignore[method-assign]

    assert service.list_runtime_bindings(DEFAULT_WORKSPACE_ID) == ()


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


def test_registry_lists_workspace_conversations_only(tmp_path: Path) -> None:
    service = build_test_registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Workspace A")

    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-1",
        role="workspace-thread",
        title="Planning thread",
    )
    service.link_membership(
        "ws-a",
        item_type="note",
        item_id="note-1",
        role="source",
        title="Research note",
    )

    conversations = service.list_workspace_conversations("ws-a")

    assert [conversation.item_id for conversation in conversations] == ["conv-1"]
    assert conversations[0].title == "Planning thread"


def test_registry_list_workspace_conversations_uses_transaction_context() -> None:
    source = inspect.getsource(LocalWorkspaceRegistryService.list_workspace_conversations)

    assert "self.db.transaction()" in source
    assert "self.db.connection()" not in source


def test_registry_list_workspaces_uses_constant_sql() -> None:
    source = inspect.getsource(LocalWorkspaceRegistryService.list_workspaces)

    assert "where_clause" not in source
    assert 'f"""' not in source


def test_registry_normalizes_workspace_ids_at_service_boundary(tmp_path: Path) -> None:
    service = build_test_registry(tmp_path)
    service.create_workspace(workspace_id=" ws-a ", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")

    service.set_active_workspace(" ws-a ")
    service.link_membership(" ws-a ", item_type=" note ", item_id=" note-1 ", role=" source ")

    active = service.get_active_workspace()

    assert active is not None
    assert active.workspace_id == "ws-a"
    assert service.get_workspace(" ws-a ") is not None
    memberships = service.get_item_memberships(" note ", " note-1 ")
    assert len(memberships) == 1
    assert memberships[0].workspace_id == "ws-a"


def test_registry_rejects_archived_active_workspace_without_clearing_current(
    tmp_path: Path,
) -> None:
    db = WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
    service = LocalWorkspaceRegistryService(db)
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    with db.transaction() as conn:
        conn.execute(
            """
            UPDATE workspace_records
            SET archived = ?
            WHERE workspace_id = ?
            """,
            (1, "ws-b"),
        )

    with pytest.raises(WorkspaceNotFound):
        service.set_active_workspace("ws-b")

    active = service.get_active_workspace()

    assert active is not None
    assert active.workspace_id == "ws-a"


def test_link_membership_uses_unique_membership_lookup() -> None:
    source = inspect.getsource(LocalWorkspaceRegistryService.link_membership)

    assert "get_item_memberships" not in source
    assert "workspace_id = ?" in source
    assert "item_type = ?" in source
    assert "item_id = ?" in source
    assert "role = ?" in source


def test_registry_wraps_non_json_runtime_binding_metadata(tmp_path: Path) -> None:
    service = build_test_registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Workspace A")

    with pytest.raises(WorkspaceRegistryServiceError, match="JSON-serializable"):
        service.save_runtime_binding(
            WorkspaceRuntimeBinding(
                workspace_id="ws-a",
                binding_id="binding-1",
                binding_kind=RuntimeBindingKind.GIT_WORKTREE,
                label="Repo",
                locator="/tmp/repo",
                status=RuntimeBindingStatus.INSPECT_ONLY,
                metadata={"bad": object()},
            )
        )


def test_registry_tolerates_corrupt_runtime_binding_metadata(tmp_path: Path) -> None:
    db = WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
    service = LocalWorkspaceRegistryService(db)
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="ws-a",
            binding_id="binding-1",
            binding_kind=RuntimeBindingKind.GIT_WORKTREE,
            label="Repo",
            locator="/tmp/repo",
            status=RuntimeBindingStatus.INSPECT_ONLY,
            metadata={"branch": "dev"},
        )
    )
    with db.transaction() as conn:
        conn.execute(
            """
            UPDATE workspace_runtime_bindings
            SET metadata_json = ?
            WHERE binding_id = ?
            """,
            ("{not-json", "binding-1"),
        )

    bindings = service.list_runtime_bindings(" ws-a ")

    assert len(bindings) == 1
    assert bindings[0].metadata == {}
