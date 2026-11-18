"""Run and prepare Settings Manual Sync after server activation."""

from __future__ import annotations

import asyncio
import platform
from typing import TYPE_CHECKING

from loguru import logger

from ...Sync_Interop.sync_state import SERVER_TRUSTED_SYNC_POLICY, SyncV2EnrollmentError
from ...Workspaces.registry_service import DEFAULT_WORKSPACE_ID

if TYPE_CHECKING:
    from ..Screens.settings_screen import SettingsScreen


async def prepare_after_activation(
    screen: SettingsScreen, server_id: str, *, approved: bool
) -> None:
    """Enroll the approved account and workspace, then refresh preview."""
    if not approved:
        screen.app.notify(
            "Server activated. To set up Manual Sync, opt in under Switch Source / Server.",
            severity="information",
        )
        screen._refresh_manual_sync_rows()
        return

    sync_scope = await asyncio.to_thread(screen._active_sync_scope)
    sync_scope_service = getattr(screen.app_instance, "sync_scope_service", None)
    prepare = getattr(sync_scope_service, "prepare_sync_v2_profile_mode", None)
    try:
        if (
            sync_scope["server_profile_id"] != server_id
            or not sync_scope["authenticated_principal_id"]
        ):
            raise SyncV2EnrollmentError(
                "Manual Sync setup requires an authenticated server account."
            )
        if sync_scope["workspace_scope"] not in (None, DEFAULT_WORKSPACE_ID):
            raise SyncV2EnrollmentError(
                "Switch to the Default workspace to set up personal Manual Sync."
            )
        if not callable(prepare):
            raise SyncV2EnrollmentError(
                "Manual Sync setup is unavailable in this runtime."
            )

        await prepare(
            **sync_scope,
            profile_mode="local_first_sync",
            display_name=platform.node() or "tldw_chatbook",
            domains=["notes.note", "chat.conversation", "chat.message"],
            encryption_policy=SERVER_TRUSTED_SYNC_POLICY,
            enrollment_only=True,
        )
    except SyncV2EnrollmentError as exc:
        screen._manual_sync_setup_issue = (sync_scope, str(exc))
        screen.app.notify(str(exc), severity="warning")
    except Exception as exc:
        logger.warning(
            "Sync v2 profile preparation failed "
            "(mode=local_first_sync, exception_category=%s).",
            type(exc).__name__,
        )
        message = "Server activated, but Sync v2 setup could not be completed."
        screen._manual_sync_setup_issue = (sync_scope, message)
        screen.app.notify(message, severity="warning")
    else:
        screen._manual_sync_setup_issue = None
        screen.app.notify("Server activated. Manual Sync preview is available.")
    screen._refresh_manual_sync_rows()


async def run_once(
    screen: SettingsScreen, *, prepare_notes_organization_only: bool = False
) -> None:
    control = getattr(screen.app_instance, "manual_sync_control_service", None)
    if control is None:
        screen._apply_manual_sync_rows(
            (
                ("Manual sync status", "blocked"),
                ("Manual sync result", "Manual Sync control is not available."),
                ("Pending outgoing", "unknown"),
            ),
        )
        return
    active_workspace = screen._active_workspace_record()
    # TASK-32926: the principal scope reads the auth token, which can
    # be an OS-keyring read -- never on the event loop.
    sync_scope = await asyncio.to_thread(screen._active_sync_scope, active_workspace)
    server_profile_id = sync_scope["server_profile_id"]
    if not server_profile_id:
        screen._apply_manual_sync_rows(
            (
                ("Manual sync status", "blocked"),
                (
                    "Manual sync result",
                    "Manual Sync requires an active server profile.",
                ),
                ("Pending outgoing", "none"),
            ),
        )
        return
    try:
        issue = screen._manual_sync_setup_issue
        if issue is not None and issue[0] == sync_scope:
            screen._apply_manual_sync_rows(
                (
                    ("Manual sync status", "blocked"),
                    ("Manual sync result", issue[1]),
                    ("Pending outgoing", "unknown"),
                )
            )
            return
        preview = await asyncio.to_thread(control.preview, **sync_scope)
        if not preview.can_run and not (
            prepare_notes_organization_only and preview.status == "setup"
        ):
            screen._apply_manual_sync_rows(
                screen._manual_sync_rows_from_preview(preview)
            )
            return
        options = (
            {"prepare_notes_organization_only": True}
            if prepare_notes_organization_only
            else {}
        )
        if getattr(control, "notes_organization_sync_service", None) is not None:
            options["require_approved_notes_organization"] = True
        result = await control.run_once(**sync_scope, **options)
    except Exception as exc:
        logger.warning("Settings manual sync run failed.", exc_info=True)
        screen._apply_manual_sync_rows(
            (
                ("Manual sync status", "failed"),
                ("Manual sync result", f"Manual Sync failed: {type(exc).__name__}"),
                ("Pending outgoing", "unknown"),
            ),
        )
        return
    screen._apply_manual_sync_result(result)
