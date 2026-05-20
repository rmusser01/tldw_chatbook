"""Pure display state for workspace-aware Console surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from loguru import logger

from .models import (
    RuntimeBindingStatus,
    WorkspaceRecord,
    WorkspaceRuntimeBinding,
)

logger = logger.bind(module="WorkspaceDisplayState")


@dataclass(frozen=True)
class ConsoleWorkspaceConversationRow:
    """One conversation row shown inside the Console workspace context tray."""

    conversation_id: str
    title: str
    status: str = ""
    selected: bool = False


@dataclass(frozen=True)
class ConsoleWorkspaceContextState:
    """Renderable Console workspace context snapshot."""

    heading: str
    workspace_label: str
    authority_label: str
    sync_label: str
    runtime_label: str
    conversation_rows: tuple[ConsoleWorkspaceConversationRow, ...]
    conversation_empty_copy: str
    change_workspace_enabled: bool
    change_workspace_recovery: str
    new_conversation_enabled: bool
    new_conversation_recovery: str
    recovery_copy: str


def build_console_workspace_state(
    *,
    registry_service: Any,
    current_conversation: str | None,
    conversations: Iterable[ConsoleWorkspaceConversationRow] | None = None,
) -> ConsoleWorkspaceContextState:
    """Build Console workspace display state from the local registry seam.

    Args:
        registry_service: Local workspace registry service used to read the
            active workspace, runtime bindings, and workspace memberships. A
            missing or failing service produces a safe degraded state.
        current_conversation: Active conversation id used to mark the selected
            conversation row, when it belongs to the active workspace.
        conversations: Optional prebuilt conversation rows. When omitted, rows
            are derived from `conversation` workspace memberships.

    Returns:
        Renderable Console workspace context state.
    """

    if registry_service is None:
        return ConsoleWorkspaceContextState(
            heading="Convos & Workspaces",
            workspace_label="No workspace selected",
            authority_label="Authority: unavailable",
            sync_label="Sync: unavailable",
            runtime_label="Runtime: unavailable",
            conversation_rows=(),
            conversation_empty_copy="Workspace conversations are unavailable.",
            change_workspace_enabled=False,
            change_workspace_recovery="Workspace service not ready.",
            new_conversation_enabled=False,
            new_conversation_recovery="Workspace service not ready.",
            recovery_copy="Workspace service not ready. Restart or open Settings if this persists.",
        )

    try:
        active_workspace = registry_service.get_active_workspace()
    except Exception:
        logger.warning(
            "Failed to read active workspace for Console context rail",
            exc_info=True,
        )
        return ConsoleWorkspaceContextState(
            heading="Convos & Workspaces",
            workspace_label="No workspace selected",
            authority_label="Authority: unavailable",
            sync_label="Sync: unavailable",
            runtime_label="Runtime: unavailable",
            conversation_rows=(),
            conversation_empty_copy="Workspace conversations are unavailable.",
            change_workspace_enabled=False,
            change_workspace_recovery="Workspace registry could not be read.",
            new_conversation_enabled=False,
            new_conversation_recovery="Workspace registry could not be read.",
            recovery_copy="Workspace registry could not be read. Check local workspace storage.",
        )

    if active_workspace is None:
        return ConsoleWorkspaceContextState(
            heading="Convos & Workspaces",
            workspace_label="No workspace selected",
            authority_label="Authority: local registry ready",
            sync_label="Sync: not configured",
            runtime_label="Runtime: none",
            conversation_rows=(),
            conversation_empty_copy="No active workspace conversations.",
            change_workspace_enabled=False,
            change_workspace_recovery="Workspace switching is not wired yet.",
            new_conversation_enabled=False,
            new_conversation_recovery="Select a workspace before creating a conversation.",
            recovery_copy="Create or select a workspace before using workspace-scoped context.",
        )

    runtime_bindings = _safe_runtime_bindings(registry_service, active_workspace)
    source_rows = (
        tuple(conversations)
        if conversations is not None
        else _conversation_rows_from_memberships(registry_service, active_workspace)
    )
    rows = tuple(_select_conversation(row, current_conversation) for row in source_rows)
    return ConsoleWorkspaceContextState(
        heading="Convos & Workspaces",
        workspace_label=f"Workspace: {active_workspace.name}",
        authority_label=f"Authority: {active_workspace.authority.value}",
        sync_label=f"Sync: {active_workspace.sync_status.value}",
        runtime_label=_runtime_label(runtime_bindings),
        conversation_rows=rows,
        conversation_empty_copy="No conversations in this workspace yet.",
        change_workspace_enabled=False,
        change_workspace_recovery="Workspace switching is not wired yet.",
        new_conversation_enabled=False,
        new_conversation_recovery="Workspace conversation creation lands in a later slice.",
        recovery_copy="Workspace switching is read-only in this slice.",
    )


def _safe_runtime_bindings(
    registry_service: Any,
    active_workspace: WorkspaceRecord,
) -> tuple[WorkspaceRuntimeBinding, ...]:
    try:
        runtime_bindings = registry_service.list_runtime_bindings(active_workspace.workspace_id)
        if not runtime_bindings:
            return ()
        return tuple(runtime_bindings)
    except Exception:
        logger.warning(
            "Failed to read workspace runtime bindings for Console context rail",
            exc_info=True,
        )
        return ()


def _conversation_rows_from_memberships(
    registry_service: Any,
    active_workspace: WorkspaceRecord,
) -> tuple[ConsoleWorkspaceConversationRow, ...]:
    try:
        memberships = registry_service.list_workspace_memberships(active_workspace.workspace_id)
    except Exception:
        logger.warning(
            "Failed to read workspace memberships for Console context rail",
            exc_info=True,
        )
        return ()
    if not memberships:
        return ()
    rows: list[ConsoleWorkspaceConversationRow] = []
    for membership in memberships:
        if membership.item_type != "conversation":
            continue
        rows.append(
            ConsoleWorkspaceConversationRow(
                conversation_id=membership.item_id,
                title=membership.title or membership.item_id,
                status=membership.role,
            )
        )
    return tuple(rows)


def _select_conversation(
    row: ConsoleWorkspaceConversationRow,
    current_conversation: str | None,
) -> ConsoleWorkspaceConversationRow:
    return ConsoleWorkspaceConversationRow(
        conversation_id=row.conversation_id,
        title=row.title,
        status=row.status,
        selected=bool(current_conversation) and row.conversation_id == current_conversation,
    )


def _runtime_label(bindings: tuple[WorkspaceRuntimeBinding, ...]) -> str:
    if not bindings:
        return "Runtime: none"
    ready_count = sum(binding.status == RuntimeBindingStatus.READY for binding in bindings)
    return (
        f"Runtime: {len(bindings)} {_plural('binding', len(bindings))}, "
        f"{ready_count} ready"
    )


def _plural(label: str, count: int) -> str:
    return label if count == 1 else f"{label}s"
