"""Console workspace context display-state tests."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import (
    ConsoleWorkspaceACPHandoffState,
    DEFAULT_WORKSPACE_ID,
    LocalWorkspaceRegistryService,
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceAuthority,
    WorkspaceRecord,
    WorkspaceRuntimeBinding,
    WorkspaceSyncStatus,
    WorkspaceTransferPolicy,
)
from tldw_chatbook.Workspaces import display_state
from tldw_chatbook.Workspaces.display_state import (
    ConsoleWorkspaceConversationRow,
    ConsoleWorkspaceServerAdapterState,
    build_library_workspace_depth_state,
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
    service.ensure_default_workspace()

    state = build_console_workspace_state(
        registry_service=service,
        current_conversation=None,
    )

    assert state.workspace_label == "Workspace: Default"
    assert state.authority_label == "Authority: local-only"
    assert state.change_workspace_enabled is False
    assert state.new_conversation_enabled is False
    assert state.runtime_label == "Runtime: none, file tools disabled"
    assert state.recovery_copy == ""
    assert state.server_readiness_label == "Server: local fallback"
    assert service.list_runtime_bindings(DEFAULT_WORKSPACE_ID) == ()


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
    assert state.sync_label == "Sync: dry-run only"
    assert state.runtime_label == "Runtime: 1 binding, 0 ready"
    assert state.conversation_rows[0].title == "Planning thread"
    assert state.conversation_rows[0].selected is True
    assert state.change_workspace_enabled is False
    assert state.new_conversation_enabled is False
    assert "later slice" in state.new_conversation_recovery.lower()


def test_console_workspace_state_enables_switching_with_multiple_workspaces(
    tmp_path: Path,
) -> None:
    service = _registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")

    state = build_console_workspace_state(
        registry_service=service,
        current_conversation=None,
    )

    assert state.change_workspace_enabled is True
    assert state.change_workspace_recovery == ""
    assert state.recovery_copy == ""


def test_safe_workspaces_treats_missing_registry_as_empty_without_warning(monkeypatch) -> None:
    warnings: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeLogger:
        def warning(self, *args, **kwargs) -> None:
            warnings.append((args, kwargs))

    monkeypatch.setattr(display_state, "logger", FakeLogger(), raising=False)

    assert display_state._safe_workspaces(None) == ()
    assert warnings == []


def test_console_workspace_state_maps_workspace_sync_status_before_promotion(tmp_path: Path) -> None:
    expected_labels = {
        WorkspaceSyncStatus.NOT_CONFIGURED: "Sync: not configured",
        WorkspaceSyncStatus.SYNCING: "Sync: syncing",
        WorkspaceSyncStatus.BLOCKED: "Sync: blocked",
        WorkspaceSyncStatus.CONFLICT: "Sync: conflict review required",
    }

    for sync_status, expected_label in expected_labels.items():
        service = _registry(tmp_path / sync_status.value)
        service.create_workspace(
            workspace_id=f"ws-{sync_status.value}",
            name=f"Workspace {sync_status.value}",
            sync_status=sync_status,
        )
        service.set_active_workspace(f"ws-{sync_status.value}")

        state = build_console_workspace_state(
            registry_service=service,
            current_conversation=None,
        )

        assert state.sync_label == expected_label


def test_console_workspace_state_distinguishes_server_and_authority_readiness(
    tmp_path: Path,
) -> None:
    cases = (
        (
            WorkspaceAuthority.LOCAL_ONLY,
            None,
            "Server: local fallback",
            "Local registry is authoritative",
        ),
        (
            WorkspaceAuthority.REMOTE_ONLY,
            None,
            "Server: remote-only",
            "not materialized locally",
        ),
        (
            WorkspaceAuthority.CONFLICT,
            None,
            "Server: conflict",
            "review required",
        ),
        (
            WorkspaceAuthority.RUNTIME_MISSING,
            None,
            "Server: runtime missing",
            "runtime binding cannot be restored",
        ),
        (
            WorkspaceAuthority.SERVER_BACKED,
            ConsoleWorkspaceServerAdapterState(
                available=False,
                detail="No tldw_server workspace API configured.",
            ),
            "Server: unavailable",
            "adapter boundary",
        ),
    )

    for authority, adapter_state, expected_label, expected_detail in cases:
        service = _registry(tmp_path / authority.value)
        service.create_workspace(
            workspace_id=f"ws-{authority.value}",
            name=f"Workspace {authority.value}",
            authority=authority,
            sync_status=WorkspaceSyncStatus.BLOCKED,
        )
        service.set_active_workspace(f"ws-{authority.value}")

        state = build_console_workspace_state(
            registry_service=service,
            current_conversation=None,
            server_adapter_state=adapter_state,
        )

        assert state.server_readiness_label == expected_label
        assert expected_detail in state.server_readiness_detail
        assert "No background sync" in state.server_readiness_detail


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


def test_console_workspace_state_disambiguates_duplicate_conversation_titles(
    tmp_path: Path,
) -> None:
    service = _registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Research Sprint")
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-alpha-1234",
        role="workspace-thread",
        title="Chat 1",
        transfer_policy=WorkspaceTransferPolicy.REFERENCE,
    )
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-beta-5678",
        role="workspace-thread",
        title="Chat 1",
        transfer_policy=WorkspaceTransferPolicy.REFERENCE,
    )

    state = build_console_workspace_state(
        registry_service=service,
        current_conversation="conv-beta-5678",
    )

    conversation_titles = [row.title for row in state.conversation_rows]
    assert conversation_titles == [
        "Chat 1 [conv-alp]",
        "Chat 1 [conv-bet]",
    ]
    assert [row.selected for row in state.conversation_rows] == [False, True]

    handoff_titles_by_id = {row.item_id: row.title for row in state.handoff_rows}
    assert handoff_titles_by_id["conv-alpha-1234"] == "Chat 1 [conv-alp]"
    assert handoff_titles_by_id["conv-beta-5678"] == "Chat 1 [conv-bet]"


def test_console_workspace_state_exposes_handoff_transfer_policy_rows(
    tmp_path: Path,
) -> None:
    service = _registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Research Sprint")
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-a",
        item_type="note",
        item_id="note-1",
        role="source",
        title="Copied note",
        transfer_policy=WorkspaceTransferPolicy.COPY,
    )
    service.link_membership(
        "ws-a",
        item_type="media",
        item_id="media-1",
        role="source",
        title="Referenced media",
        transfer_policy=WorkspaceTransferPolicy.REFERENCE,
    )
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-1",
        role="workspace-thread",
        title="Metadata thread",
        transfer_policy=WorkspaceTransferPolicy.METADATA_ONLY,
    )
    service.link_membership(
        "ws-a",
        item_type="artifact",
        item_id="artifact-1",
        role="source",
        title="Local secret draft",
        transfer_policy=WorkspaceTransferPolicy.LOCAL_ONLY,
    )

    state = build_console_workspace_state(
        registry_service=service,
        current_conversation="conv-1",
    )

    rows_by_id = {row.item_id: row for row in state.handoff_rows}
    assert rows_by_id["note-1"].handoff_label == "Handoff: copy"
    assert rows_by_id["media-1"].handoff_label == "Handoff: reference"
    assert rows_by_id["conv-1"].handoff_label == "Handoff: metadata-only"
    assert rows_by_id["artifact-1"].handoff_label == "Handoff: local-only"
    assert rows_by_id["artifact-1"].portable is False
    assert "not portable" in rows_by_id["artifact-1"].detail


def test_console_workspace_state_exposes_acp_task_run_handoff_readiness_and_audit(
    tmp_path: Path,
) -> None:
    service = _registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Research Sprint")
    service.set_active_workspace("ws-a")

    state = build_console_workspace_state(
        registry_service=service,
        current_conversation=None,
        acp_handoff_state=ConsoleWorkspaceACPHandoffState(
            status="failed",
            detail="ACP runtime package failed preflight.",
            audit_detail="Audit: no secrets copied; source references only.",
        ),
    )

    assert state.acp_handoff_label == "ACP task/run: failed"
    assert state.acp_handoff_detail == "ACP runtime package failed preflight."
    assert state.acp_handoff_audit == "Audit: no secrets copied; source references only."


def test_console_workspace_state_normalizes_acp_handoff_ready_and_blocked_states(
    tmp_path: Path,
) -> None:
    service = _registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Research Sprint")
    service.set_active_workspace("ws-a")

    for status in ("ready", "blocked"):
        state = build_console_workspace_state(
            registry_service=service,
            current_conversation=None,
            acp_handoff_state=ConsoleWorkspaceACPHandoffState(
                status=status,
                detail=f"ACP handoff is {status}.",
                audit_detail=f"Audit: {status} preflight visible.",
            ),
        )

        assert state.acp_handoff_label == f"ACP task/run: {status}"
        assert state.acp_handoff_detail == f"ACP handoff is {status}."
        assert state.acp_handoff_audit == f"Audit: {status} preflight visible."


def test_console_workspace_state_treats_none_memberships_as_empty() -> None:
    class NullMembershipService:
        def get_active_workspace(self) -> WorkspaceRecord:
            return WorkspaceRecord(workspace_id="ws-a", name="Research Sprint")

        def list_runtime_bindings(self, _workspace_id: str):
            return ()

        def list_workspace_memberships(self, _workspace_id: str):
            return None

    state = build_console_workspace_state(
        registry_service=NullMembershipService(),
        current_conversation="conv-1",
    )

    assert state.workspace_label == "Workspace: Research Sprint"
    assert state.conversation_rows == ()
    assert state.conversation_empty_copy == "No conversations in this workspace yet."


def test_console_workspace_state_logs_registry_failures(monkeypatch) -> None:
    class FailingRegistryService:
        def get_active_workspace(self) -> WorkspaceRecord:
            raise RuntimeError("workspace db unavailable")

    warnings: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeLogger:
        def warning(self, *args, **kwargs) -> None:
            warnings.append((args, kwargs))

    monkeypatch.setattr(display_state, "logger", FakeLogger(), raising=False)

    state = build_console_workspace_state(
        registry_service=FailingRegistryService(),
        current_conversation=None,
    )

    assert state.workspace_label == "No workspace selected"
    assert warnings
    assert warnings[0][1].get("exc_info") is True


def test_library_workspace_depth_state_preserves_visibility_but_blocks_cross_workspace_context(
    tmp_path: Path,
) -> None:
    service = _registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-b",
        item_type="note",
        item_id="note-1",
        title="Research Note",
    )
    service.link_membership(
        "ws-a",
        item_type="media",
        item_id="media-1",
        title="Transcript A",
    )

    state = build_library_workspace_depth_state(
        registry_service=service,
        source_records={
            "notes": ({"id": "note-1", "title": "Research Note"},),
            "media": ({"id": "media-1", "title": "Transcript A"},),
            "conversations": (),
        },
    )

    assert state.workspace_label == "Workspace: Workspace A"
    assert "all Library and Notes items remain visible" in state.visibility_label
    assert state.context_handoff_enabled is False
    assert "1 blocked" in state.handoff_label
    assert "Copy or link" in state.context_handoff_tooltip
    assert [row.title for row in state.source_rows] == ["Research Note", "Transcript A"]
    assert [row.workspace_label for row in state.source_rows] == ["Workspace B", "Workspace A"]
    assert [row.visible for row in state.source_rows] == [True, True]
    assert state.source_rows[0].active_context_eligible is False
    assert state.source_rows[0].context_label == "Console/RAG: blocked"
    assert state.source_rows[1].active_context_eligible is True
    assert state.source_rows[1].context_label == "Console/RAG: eligible"


def test_library_workspace_depth_state_recognizes_media_id_and_ignores_idless_rows(
    tmp_path: Path,
) -> None:
    service = _registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-a",
        item_type="note",
        item_id="note-1",
        title="Workspace note",
    )
    service.link_membership(
        "ws-a",
        item_type="media",
        item_id="media-1",
        title="Workspace transcript",
    )

    state = build_library_workspace_depth_state(
        registry_service=service,
        source_records={
            "notes": (
                {"id": "note-1", "title": "Workspace note"},
                {"title": "Malformed source without an id"},
            ),
            "media": ({"media_id": "media-1", "title": "Workspace transcript"},),
            "conversations": (),
        },
    )

    assert state.context_handoff_enabled is True
    assert state.handoff_label == "Console/RAG handoff: 2 eligible"
    assert [row.item_id for row in state.source_rows] == ["note-1", "media-1"]
    assert all(row.active_context_eligible for row in state.source_rows)
