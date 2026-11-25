"""Land a qualified Research workspace after Console finishes attaching."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..Screens.chat_screen import ChatScreen


def queue_workspace_handoff(
    screen: ChatScreen,
    ref: object,
    *,
    create_chat: bool,
) -> None:
    from ...Research_Workspace.contracts import (
        QualifiedWorkspaceRef,
        WorkspaceDataSource,
    )

    if (
        not isinstance(ref, QualifiedWorkspaceRef)
        or ref.data_source is not WorkspaceDataSource.LOCAL
    ):
        return
    screen._pending_research_console_workspace = (ref.workspace_id, create_chat)
    consume_workspace_handoff(screen)


def consume_workspace_handoff(screen: ChatScreen) -> None:
    pending = screen._pending_research_console_workspace
    if (
        pending is None
        or not screen._console_attach_reconciled
        or not screen._is_active_console_screen()
    ):
        return
    screen._pending_research_console_workspace = None
    workspace_id, create_chat = pending
    registry = getattr(screen.app_instance, "workspace_registry_service", None)
    record = registry.get_workspace(workspace_id) if registry is not None else None
    if record is None or record.archived:
        screen.app_instance.notify(
            "Choose an active Local workspace in Research.", severity="warning"
        )
        return
    if create_chat:
        screen.run_worker(
            screen._create_console_chat_in_workspace(workspace_id),
            exclusive=True,
            group="console-workspace-new-chat",
        )
    elif not record.active:
        screen._workspace.activate_workspace_id(workspace_id)
