"""Archive and exact-identity resume workflows shared by Console and Library."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from ...Chat.conversation_archive_actions import (
    archive_failure_copy,
    change_conversation_archive,
    local_conversation_service,
    storage_call,
)
from ...UI.Navigation.pending_handoff_store import (
    ConsoleConversationResumeIntent,
    HandoffChannel,
)
from ...Widgets.confirmation_dialog import ConfirmationDialog


async def request_conversation_resume(app: Any, conversation_id: str) -> None:
    """Restore with explicit scope disclosure, then navigate using a typed ID."""
    intent = None
    try:
        intent = ConsoleConversationResumeIntent(conversation_id)
        service = local_conversation_service(app)
        row = await storage_call(service, "get_conversation_metadata", conversation_id)
    except Exception:  # noqa: BLE001 - a failed read must not crash the application worker
        logger.bind(
            conversation_id=intent.conversation_id if intent else None
        ).exception("Conversation metadata unavailable")
        app.notify(
            "Could not read this conversation. Refresh Library and try Resume again.",
            severity="error",
        )
        return
    if not row:
        app.notify("This conversation is no longer available.", severity="warning")
        return
    registry = getattr(app, "workspace_registry_service", None)
    workspace_id = row.get("workspace_id")
    try:
        workspace = (
            await storage_call(registry, "get_workspace", workspace_id)
            if registry and workspace_id
            else None
        )
    except Exception:  # noqa: BLE001 - retain recovery state when storage is unavailable
        logger.bind(
            conversation_id=conversation_id, workspace_id=workspace_id
        ).exception("Conversation workspace unavailable")
        app.notify(
            "Could not read this workspace. Refresh Library and try Resume again.",
            severity="error",
        )
        return

    async def proceed(replacement_name: str | None = None) -> None:
        workspace_restored = False
        conversation_restored = False
        try:
            # A confirmation (including Restore as) may outlive its metadata.
            # Refuse a changed identity/scope/version before either store writes.
            current_row = await storage_call(
                service, "get_conversation_metadata", conversation_id
            )
            if (
                not current_row
                or current_row.get("version") != row.get("version")
                or current_row.get("workspace_id") != workspace_id
                or bool(current_row.get("archived")) != bool(row.get("archived"))
            ):
                app.notify(
                    "This conversation changed or is no longer available. Refresh Library and try Resume again.",
                    severity="warning",
                )
                return
            if workspace and workspace.archived:
                current_workspace = await storage_call(
                    registry, "get_workspace", workspace_id
                )
                if current_workspace is None:
                    app.notify(
                        "This workspace is no longer available. Refresh Library and try Resume again.",
                        severity="warning",
                    )
                    return
                if current_workspace.archived:
                    active_workspaces = await storage_call(registry, "list_workspaces")
                    target_name = replacement_name or current_workspace.name
                    if any(
                        item.name.casefold() == target_name.casefold()
                        for item in active_workspaces
                    ):
                        from ...Widgets.Console.console_workspace_switcher_modal import (
                            ConsoleWorkspaceRenameModal,
                        )

                        app.notify(
                            "That workspace name is in use. Choose a name to restore this workspace.",
                            severity="warning",
                        )
                        app.push_screen(
                            ConsoleWorkspaceRenameModal(
                                current_name=target_name, restoring=True
                            ),
                            callback=lambda name: (
                                app.run_worker(
                                    proceed(name),
                                    group="resume-saved-conversation",
                                    exclusive=True,
                                )
                                if name
                                else None
                            ),
                        )
                        return
                    await storage_call(
                        registry,
                        "unarchive_workspace",
                        workspace_id,
                        name=replacement_name,
                    )
                    workspace_restored = True
            if row.get("archived"):
                result = await change_conversation_archive(
                    app,
                    [conversation_id],
                    archived=False,
                    expected_versions={conversation_id: row["version"]},
                )
                if conversation_id not in result["changed"]:
                    app.notify(
                        ("Workspace restored. " if workspace_restored else "")
                        + "The conversation could not be restored. Try Resume again from Library.",
                        severity="warning",
                    )
                    return
                conversation_restored = True
            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_CONVERSATION_RESUME, intent
            )
            from ...Constants import TAB_CHAT
            from ...UI.Navigation.main_navigation import NavigateToScreen

            app.post_message(NavigateToScreen(TAB_CHAT))
        except Exception:  # noqa: BLE001 - retain completed changes across separate stores
            logger.bind(
                conversation_id=conversation_id, workspace_id=workspace_id
            ).exception("Conversation recovery failed")
            completed = "Workspace restored. " if workspace_restored else ""
            if conversation_restored:
                completed += "Conversation restored. "
            app.notify(
                completed + "Resume did not complete. Try Resume again from Library.",
                severity="error",
            )

    def confirmed(accepted: bool) -> None:
        if accepted:
            app.run_worker(proceed(), group="resume-saved-conversation", exclusive=True)

    if workspace and workspace.archived:
        from rich.markup import escape

        app.push_screen(
            ConfirmationDialog(
                title="Restore workspace and resume?",
                message=f"This restores the whole workspace ‘{escape(workspace.name)}’ and makes its other conversations available again. This conversation will then open in Console.",
                confirm_label="Restore & resume",
            ),
            callback=confirmed,
        )
    elif row.get("archived"):
        app.push_screen(
            ConfirmationDialog(
                title="Restore conversation?",
                message="Return this conversation to active history and resume the original chat in Console.",
                confirm_label="Restore & resume",
            ),
            callback=confirmed,
        )
    else:
        await proceed()


async def consume_conversation_resume(screen: Any) -> None:
    """Reuse an open session or hydrate the original persisted conversation."""
    handoffs = screen.app_instance.pending_handoffs
    channel = HandoffChannel.CONSOLE_CONVERSATION_RESUME
    while (
        screen.app.screen is screen and (claim := handoffs.claim(channel)) is not None
    ):
        workspace_id = None
        try:
            conversation_id = claim.value.conversation_id
            store = screen._ensure_console_chat_store()
            existing = next(
                (
                    item
                    for item in store.sessions()
                    if item.persisted_conversation_id == conversation_id
                ),
                None,
            )

            def resume_is_current() -> bool:
                return screen.app.screen is screen and handoffs.is_current_claim(claim)

            if existing is not None:
                row = await storage_call(
                    local_conversation_service(screen.app_instance),
                    "get_conversation_metadata",
                    conversation_id,
                )
                if not resume_is_current():
                    handoffs.release(claim)
                    return
                if not row:
                    handoffs.release(claim)
                    screen.app_instance.notify(
                        "This conversation is no longer available. Refresh Library.",
                        severity="warning",
                    )
                    return
                workspace_id = row.get("workspace_id")
                registry = getattr(
                    screen.app_instance, "workspace_registry_service", None
                )
                workspace = (
                    await storage_call(registry, "get_workspace", workspace_id)
                    if workspace_id and registry is not None
                    else None
                )
                if not resume_is_current():
                    handoffs.release(claim)
                    return
                if row.get("archived") or (workspace and workspace.archived):
                    handoffs.release(claim)
                    await request_conversation_resume(
                        screen.app_instance, conversation_id
                    )
                    return
                if workspace_id and workspace is None:
                    handoffs.release(claim)
                    screen.app_instance.notify(
                        "This workspace is no longer available. Refresh Library.",
                        severity="warning",
                    )
                    return
                await screen._session._activate_native_console_session(
                    existing.id, activate_if=resume_is_current
                )
                result = True if resume_is_current() else None
            else:
                result = await screen._workspace._resume_console_workspace_conversation(
                    conversation_id,
                    preserve_persisted_scope=True,
                    resume_if=resume_is_current,
                )
            superseded = not handoffs.is_current_claim(claim)
            if result is not None or superseded:
                handoffs.acknowledge(claim)
                if result is False and not superseded:
                    screen.app_instance.notify(
                        "This conversation is no longer available.", severity="warning"
                    )
            else:
                handoffs.release(claim)
                return  # Retain transient failures for an explicit retry.
        except asyncio.CancelledError:
            handoffs.release(claim)
            raise
        except Exception:  # noqa: BLE001 - notify and retain recovery state at the UI boundary
            logger.bind(
                conversation_id=claim.value.conversation_id, workspace_id=workspace_id
            ).exception("Conversation resume failed")
            superseded = not handoffs.is_current_claim(claim)
            handoffs.release(claim)
            if not superseded:
                screen.app_instance.notify(
                    "Could not resume this conversation. Try Resume again from Library.",
                    severity="error",
                )
                return


async def archive_current_conversation(screen: Any) -> None:
    """Archive one saved idle conversation and offer version-checked Undo."""
    app = screen.app_instance
    conversation_id = screen._current_console_conversation_id()
    if not conversation_id:
        app.notify(
            "Send and save this conversation before archiving it.",
            severity="information",
        )
        return
    row = None
    try:
        row = await storage_call(
            local_conversation_service(app),
            "get_conversation_metadata",
            conversation_id,
        )
        if not row:
            app.notify("This conversation is no longer available.", severity="warning")
            return
        result = await change_conversation_archive(
            app,
            [conversation_id],
            archived=True,
            expected_versions={conversation_id: row["version"]},
        )
        if not result["changed"]:
            app.notify(
                archive_failure_copy(
                    result["failures"].get(
                        conversation_id, "Archive could not complete."
                    )
                ),
                severity="warning",
            )
            return
        from ...Widgets.Console.console_workspace_switcher_modal import (
            WorkspaceArchiveReceiptModal,
        )

        async def recover(action: str | None) -> None:
            if action == "view":
                app.open_conversation_archive()
            elif action == "undo":
                try:
                    restored = await change_conversation_archive(
                        app,
                        [conversation_id],
                        archived=False,
                        expected_versions=result["changed"],
                    )
                    app.notify(
                        "Conversation restored."
                        if restored["changed"]
                        else "Conversation changed; refresh Archived chats to restore it."
                    )
                    screen._workspace._invalidate_console_persisted_rows_cache()
                    await screen._sync_native_console_chat_ui()
                except Exception:  # noqa: BLE001 - notify and retain recovery state at the UI boundary
                    logger.bind(
                        conversation_id=conversation_id,
                        workspace_id=row.get("workspace_id"),
                    ).exception("Conversation Undo failed")
                    app.notify(
                        "Undo could not complete. Open Archived chats to retry restoration.",
                        severity="error",
                    )

        await screen.app.push_screen(
            WorkspaceArchiveReceiptModal(
                name=str(row.get("title") or "Conversation"),
                kind="Conversation",
                description="Saved history is kept. Open Archived chats to search, review, restore and resume. This open tab can resume after restoration.",
            ),
            callback=recover,
        )
        screen._workspace._invalidate_console_persisted_rows_cache()
        await screen._sync_native_console_chat_ui()
    except Exception as exc:  # noqa: BLE001 - notify and retain recovery state at the UI boundary
        logger.bind(conversation_id=conversation_id).exception(
            "Conversation archive failed"
        )
        app.notify(f"Could not archive conversation: {exc}", severity="error")
