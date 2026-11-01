"""Coordinator-only, loss-aware local archive restoration effects (ADR-166)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from tldw_chatbook.Chat.conversation_archive_actions import (
    change_conversation_archive,
    local_conversation_service,
    storage_call,
)
from tldw_chatbook.DB.conversation_archive_queries import normalized_archive_time
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces.registry_service import (
    WorkspaceArchiveConflict,
    WorkspaceNotFound,
    WorkspaceRegistryServiceError,
)

from .contracts import (
    ArchiveRecoveryError,
    ArchiveTarget,
    ArchiveWorkspaceSnapshot,
    RecoveryOutcome,
    RecoveryPhase,
    RestoreScope,
)


def allowed_restore_scopes(
    chat_archived: bool, workspace_archived: bool
) -> tuple[RestoreScope, ...]:
    """Return only meaningful choices from the approved lifecycle matrix."""
    if chat_archived and workspace_archived:
        return ("chat_only", "chat_and_workspace")
    if chat_archived:
        return ("chat_only",)
    if workspace_archived:
        return ("workspace_only",)
    return ()


async def read_recovery_target(app: Any, conversation_id: str) -> ArchiveTarget:
    """Read a saved non-deleted local target, rejecting unresolved ownership."""
    row = await storage_call(
        local_conversation_service(app), "get_conversation_metadata", conversation_id
    )
    if not row or row.get("deleted"):
        raise ArchiveRecoveryError(
            "not_found", "The selected chat is no longer available."
        )
    workspace_id = row.get("workspace_id")
    if workspace_id == DEFAULT_WORKSPACE_ID:
        workspace_id = None
    if row.get("scope_type") not in ("global", "workspace") or (
        row.get("scope_type") == "workspace" and not row.get("workspace_id")
    ):
        raise ArchiveRecoveryError(
            "unavailable", "Chat ownership could not be resolved."
        )
    workspace = None
    if workspace_id:
        record = await storage_call(
            app.workspace_registry_service, "get_workspace", workspace_id
        )
        if record is None:
            raise ArchiveRecoveryError(
                "unavailable", "The owning workspace is unavailable."
            )
        workspace = ArchiveWorkspaceSnapshot(
            record.workspace_id,
            record.name,
            record.archived,
            normalized_archive_time(record.archived_at) if record.archived else None,
            record.archive_revision,
        )
    return ArchiveTarget(
        conversation_id,
        int(row["version"]),
        row["title"],
        bool(row["archived"]),
        workspace_id,
        workspace,
    )


def recovery_outcome(
    status: str,
    target: ArchiveTarget | None = None,
    *,
    chat_changed: bool = False,
    workspace_changed: bool = False,
    chat_available: bool | None = None,
    next_action: str | None = None,
) -> RecoveryOutcome:
    """Build sanitized receipt text without storage errors or transcript bodies."""
    messages = {
        "restored": "The selected scope was restored.",
        "already_available": "The chat is already available.",
        "declined": "Restoration was declined; nothing was changed.",
        "expired": "This recovery request expired. Refresh and try again.",
        "busy": "Another recovery request is active in this chat.",
        "cancelled": "Restoration was cancelled before any change.",
        "stale": "The selected chat or workspace changed. Refresh and try again.",
        "not_found": "The selected chat is no longer available.",
        "name_conflict": "The workspace name is in use. Choose Restore as.",
        "unavailable": "An answerable recovery confirmation is unavailable.",
        "partial": "Recovery did not complete as requested. Review the resource outcomes before retrying.",
        "storage_error": "Recovery state could not be read or changed. Refresh before retrying.",
    }
    message = messages[status]
    if status == "restored" and chat_available is False:
        message += " The workspace remains archived; restore it to open this chat."
    if status == "partial":
        message += f" Chat changed: {'yes' if chat_changed else 'no'}. Workspace changed: {'yes' if workspace_changed else 'no'}."
    if chat_available is None:
        message += " Current chat availability is unknown."
    return RecoveryOutcome(
        status,
        target.conversation_id if target else "",
        target.workspace_id if target else None,
        chat_changed,
        workspace_changed,
        chat_available,
        next_action,
        message,
    )


async def execute_restore(
    app: Any,
    target: ArchiveTarget,
    scope: RestoreScope,
    *,
    replacement_name: str | None,
    cancel_event: threading.Event,
    on_phase: Callable[[RecoveryPhase], None] | None = None,
) -> RecoveryOutcome:
    """Apply confirmed CAS writes, stopping unstarted writes after cancellation.

    The coordinator retains and shields this coroutine's task. Storage is never
    cancelled by a UI/tool waiter. No navigation, resume or send occurs here.
    """
    chat_changed = workspace_changed = False
    status = "restored"
    registry = getattr(app, "workspace_registry_service", None)
    phase = on_phase or (lambda _phase: None)
    try:
        current = await read_recovery_target(app, target.conversation_id)
        if current != target:
            raise ArchiveRecoveryError("stale", "Target changed")
        choices = allowed_restore_scopes(
            target.conversation_archived,
            bool(target.workspace and target.workspace.archived),
        )
        if not choices:
            return recovery_outcome(
                "already_available", target, chat_available=True, next_action="open"
            )
        if scope not in choices:
            raise ArchiveRecoveryError("unavailable", "Invalid recovery scope")
        if cancel_event.is_set():
            status = "cancelled"
        elif scope in ("workspace_only", "chat_and_workspace"):
            phase("workspace_write")
            await storage_call(
                registry,
                "unarchive_workspace",
                target.workspace_id,
                name=replacement_name,
                expected_archive_revision=target.workspace.archive_revision,
                expected_name=target.workspace.name,
            )
            workspace_changed = True
        if status == "restored" and scope in ("chat_only", "chat_and_workspace"):
            if cancel_event.is_set():
                status = "cancelled"
            else:
                phase("chat_write")
                result = await change_conversation_archive(
                    app,
                    [target.conversation_id],
                    archived=False,
                    expected_versions={
                        target.conversation_id: target.conversation_version
                    },
                )
                chat_changed = target.conversation_id in result["changed"]
                if not chat_changed:
                    reason = result["failures"].get(target.conversation_id)
                    status = "not_found" if reason == "not_found" else "stale"
    except ArchiveRecoveryError as exc:
        status = exc.code
    except (WorkspaceArchiveConflict, WorkspaceNotFound):
        status = "stale"
    except WorkspaceRegistryServiceError:
        status = "storage_error"
        # Existing registry exposes duplicate-name errors as generic exceptions;
        # classify using durable active records, never exception text.
        try:
            records = await storage_call(registry, "list_workspaces")
            name = (
                replacement_name
                if replacement_name is not None
                else target.workspace.name
            )
            if any(
                record.workspace_id != target.workspace_id
                and record.name.strip().casefold() == name.strip().casefold()
                for record in records
            ):
                status = "name_conflict"
        except Exception:  # noqa: BLE001 - sanitized app/storage boundary
            status = "storage_error"
    except Exception:  # noqa: BLE001 - sanitized app/storage boundary
        status = "storage_error"

    available = None
    final = None
    try:
        final = await read_recovery_target(app, target.conversation_id)
        ownership_matches = final.workspace_id == target.workspace_id
        available = (
            ownership_matches
            and not final.conversation_archived
            and not bool(final.workspace and final.workspace.archived)
        )
        if (
            not ownership_matches
            or (
                (chat_changed or not target.conversation_archived)
                and final.conversation_archived
            )
            or (workspace_changed and final.workspace and final.workspace.archived)
        ):
            status = "stale"
    except ArchiveRecoveryError as exc:
        available = False
        status = exc.code
    except Exception:  # noqa: BLE001 - sanitized app/storage boundary
        status = "storage_error"
    phase("observed")
    if status != "restored" and (chat_changed or workspace_changed):
        status = "partial"
    next_action = (
        "restore_as"
        if status == "name_conflict"
        else "open"
        if available and status == "restored"
        else "restore_workspace"
        if status == "restored"
        and final
        and final.workspace
        and final.workspace.archived
        else "refresh"
    )
    outcome = recovery_outcome(
        status,
        target,
        chat_changed=chat_changed,
        workspace_changed=workspace_changed,
        chat_available=available,
        next_action=next_action,
    )
    if (
        status == "partial"
        and final
        and final.workspace_id == target.workspace_id
        and final.conversation_archived
        and not bool(final.workspace and final.workspace.archived)
    ):
        outcome = replace(
            outcome,
            message=outcome.message
            + " The selected chat is archived; its workspace is active."
            + " Refresh and confirm restoration of the chat.",
        )
    return outcome
