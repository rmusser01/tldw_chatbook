"""Review a frozen Local Research bundle and publish it to a chosen file."""

import asyncio
from os.path import lexists
from pathlib import Path

from ...Research_Workspace.workspace_export import prepare_workspace_export
from ...Utils.atomic_file_ops import atomic_write_bytes
from ...Utils.path_validation import validate_path_simple
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.enhanced_file_picker import EnhancedFileSave


async def export_workspace_bundle(screen, ref, current) -> None:
    app = screen.app_instance
    registry = app.workspace_registry_service
    if registry is None:
        raise ValueError("Local workspace export is unavailable.")
    port = screen.controller.port_for_data_source(ref.data_source)
    prepared = await prepare_workspace_export(
        ref, port, registry, chatbooks=app.local_chatbook_service
    )
    if not current():
        return
    approved = await screen.app.push_screen_wait(
        ConfirmationDialog(
            title="Export this Local workspace?",
            message=(
                f"{prepared.content_count} content files · {prepared.content_bytes:,} bytes. "
                "The bundle includes a manifest of included, omitted, and unsupported items. "
                "Device-only items and preferences are excluded."
            ),
            confirm_label="Choose destination",
            cancel_label="Cancel",
        )
    )
    if approved is not True or not current():
        return
    picked = await screen.app.push_screen_wait(
        EnhancedFileSave(
            location=Path.home(),
            title="Save Research workspace bundle",
            default_filename="research-workspace.zip",
            context="research_workspace_export",
        )
    )
    if picked is None or not current():
        return
    target = Path(
        validate_path_simple(picked, require_exists=False, probe_existing=False)
    )
    if target.suffix.lower() != ".zip":
        raise ValueError("Choose a .zip filename for the workspace bundle.")
    if await asyncio.to_thread(lexists, target):
        overwrite = await screen.app.push_screen_wait(
            ConfirmationDialog(
                title="Replace the existing bundle?",
                message=target.name,
                confirm_label="Replace bundle",
                cancel_label="Cancel",
            )
        )
        if overwrite is not True or not current():
            return
    await prepared.confirm_current_policy(registry)
    if not current():
        return
    data = await asyncio.to_thread(prepared.bundle_bytes)
    if current():
        await asyncio.to_thread(atomic_write_bytes, target, data, private=True)
        if current():
            screen.notify("Workspace bundle exported.")
