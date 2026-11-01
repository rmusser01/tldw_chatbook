"""Settings archive Undo; receipt and completion ownership stay screen-local."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.widgets import Button

from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError

if TYPE_CHECKING:
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


def undo_workspace_archive(screen: SettingsScreen, event: Button.Pressed) -> None:
    """Restore a retained archive receipt without blocking the UI.

    Args:
        screen: Settings screen owning the receipt and worker.
        event: The Undo archive button event.
    """
    from tldw_chatbook.Chat.conversation_archive_actions import storage_call

    event.stop()
    receipt = getattr(screen, "_settings_workspace_archive_receipt", None)
    registry = getattr(screen.app_instance, "workspace_registry_service", None)
    if receipt is None or registry is None:
        return
    request = object()
    screen._settings_workspace_undo_request = request
    selected_id = screen._settings_selected_workspace_id

    def _owns_receipt() -> bool:
        return (
            screen._settings_workspace_undo_request is request
            and screen._settings_workspace_archive_receipt is receipt
        )

    def _visible() -> bool:
        return screen.is_mounted and screen.app.screen is screen

    async def _undo() -> None:
        try:
            current = await storage_call(
                registry, "get_workspace", receipt.workspace_id
            )
            if not _owns_receipt() or not _visible():
                return
            if current != receipt:
                screen._set_settings_workspaces_result(
                    "Workspace changed since archive. Use View archived to review before restoring."
                )
                return
        except WorkspaceRegistryServiceError as exc:
            if _owns_receipt() and _visible():
                screen._set_settings_workspaces_result(
                    f"Could not restore workspace: {exc}. Retry Undo; use View archived "
                    "and Restore as if the name is in use."
                )
            return
        operation = asyncio.create_task(_complete_undo())
        operations = getattr(
            screen.app_instance, "_workspace_lifecycle_operations", None
        )
        if operations is None:
            operations = screen.app_instance._workspace_lifecycle_operations = set()
        operations.add(operation)
        operation.add_done_callback(operations.discard)
        operation.add_done_callback(
            lambda task: None if task.cancelled() else task.exception()
        )
        await asyncio.shield(operation)

    async def _complete_undo() -> None:
        # Keep receipt completion coupled to a write even if a newer
        # exclusive worker cancels its predecessor's await.
        try:
            restored = await storage_call(
                registry,
                "unarchive_workspace",
                receipt.workspace_id,
                expected_archive_revision=receipt.archive_revision,
                expected_name=receipt.name,
            )
        except WorkspaceRegistryServiceError as exc:
            if _owns_receipt() and _visible():
                screen._set_settings_workspaces_result(
                    f"Could not restore workspace: {exc}. Retry Undo; use View archived "
                    "and Restore as if the name is in use."
                )
            return
        if screen._settings_workspace_archive_receipt is not receipt:
            return
        screen._settings_workspace_archive_receipt = None
        screen._settings_workspaces_result = (
            f"Restored {restored.name}. Active workspace unchanged."
        )
        if _visible():
            if screen._settings_selected_workspace_id == selected_id:
                screen._settings_selected_workspace_id = restored.workspace_id
                screen._set_settings_workspaces_result(
                    screen._settings_workspaces_result,
                    control_id="settings-workspace-set-active",
                )
            screen._refresh_settings_workspaces_pane()

    screen.app.run_worker(_undo(), group="settings-workspace-undo", exclusive=True)
