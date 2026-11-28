"""Review a frozen Local Research bundle and publish it to a chosen file."""

import asyncio
from os.path import lexists
from pathlib import Path

from ...Research_Workspace.bibtex_export import prepare_workspace_bibtex
from ...Research_Workspace.workspace_export import prepare_workspace_export
from ...Research_Workspace.workspace_import import (
    publish_workspace_bundle,
    read_workspace_bundle,
)
from ...Utils.atomic_file_ops import atomic_write_bytes
from ...Utils.path_validation import validate_path_simple
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.enhanced_file_picker import EnhancedFileOpen, EnhancedFileSave
from .workspace_lifecycle_actions import ResearchWorkspaceName


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
    target = await _export_destination(
        screen,
        current,
        filename="research-workspace.zip",
        suffix=".zip",
        title="Save Research workspace bundle",
        context="research_workspace_export",
    )
    if target is None:
        return
    await prepared.confirm_current_policy(registry)
    if not current():
        return
    data = await asyncio.to_thread(prepared.bundle_bytes)
    if current():
        await asyncio.to_thread(atomic_write_bytes, target, data, private=True)
        if current():
            screen.notify("Workspace bundle exported.")


async def export_workspace_bibtex(screen, ref, current) -> None:
    registry = screen.app_instance.workspace_registry_service
    port = screen.controller.port_for_data_source(ref.data_source)
    prepared = await prepare_workspace_bibtex(ref, port, registry)
    if not current():
        return
    if not prepared.text:
        screen.notify("No sources are eligible for this bibliography.")
        return
    target = await _export_destination(
        screen,
        current,
        filename="research-sources.bib",
        suffix=".bib",
        title="Save Research bibliography",
        context="research_workspace_bibtex",
    )
    if target is None:
        return
    await prepared.confirm_current_policy(registry)
    if current():
        await asyncio.to_thread(
            atomic_write_bytes, target, prepared.text.encode("utf-8"), private=True
        )
        if current():
            screen.notify("BibTeX bibliography exported.")


async def import_local_workspace(screen, current) -> None:
    app = screen.app_instance
    registry = app.workspace_registry_service
    port = screen.controller.port_for_data_source("local")
    if registry is None or port is None:
        raise ValueError("Local workspace import is unavailable.")
    picked = await screen.app.push_screen_wait(
        EnhancedFileOpen(
            location=Path.home(),
            title="Open Research workspace bundle",
            filters=["*.zip"],
            context="research_workspace_import",
            select_button="Review bundle",
        )
    )
    if picked is None or not current():
        return
    path = Path(validate_path_simple(picked, require_exists=True))
    prepared = await asyncio.to_thread(read_workspace_bundle, path)
    if not current():
        return
    if not prepared.items:
        raise ValueError("This bundle contains no text entries to import.")
    notes = sum(item.kind == "note" for item in prepared.items)
    title = await screen.app.push_screen_wait(
        ResearchWorkspaceName(
            "Import Local workspace",
            path.stem[:256],
            description=f"Restore {notes} notes and {len(prepared.items) - notes} artifacts through the Local Library. Use the same bundle to resume an interrupted import.",
        )
    )
    if (
        title is None
        or not current()
        or not await screen.flush_pending_work()
        or not current()
    ):
        return
    ref = await publish_workspace_bundle(
        prepared, title, port, registry, app.local_chatbook_service
    )
    if current():
        await screen._refresh_workspace_catalog(ref)
        await screen.remember_workspace_visit(ref)
        screen.notify("Workspace bundle imported into Local Library.")


async def _export_destination(
    screen, current, *, filename: str, suffix: str, title: str, context: str
) -> Path | None:
    picked = await screen.app.push_screen_wait(
        EnhancedFileSave(
            location=Path.home(),
            title=title,
            default_filename=filename,
            context=context,
        )
    )
    if picked is None or not current():
        return
    target = Path(
        validate_path_simple(picked, require_exists=False, probe_existing=False)
    )
    if target.suffix.lower() != suffix:
        raise ValueError(f"Choose a {suffix} filename.")
    if await asyncio.to_thread(lexists, target):
        overwrite = await screen.app.push_screen_wait(
            ConfirmationDialog(
                title="Replace the existing export?",
                message=target.name,
                confirm_label="Replace file",
                cancel_label="Cancel",
            )
        )
        if overwrite is not True or not current():
            return
    return target
