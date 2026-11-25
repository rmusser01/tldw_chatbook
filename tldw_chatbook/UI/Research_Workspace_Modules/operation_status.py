"""Read the scoped owners of Research operations for the Help status view."""

import asyncio
import re
from collections.abc import Mapping

from ...Research_Workspace.deep_research_bridge import DeepResearchBridge

from .chat_region import ResearchChatRegion
from .studio_region import ResearchStudioRegion


async def recent_workspace_runs(screen, ref) -> tuple[str, ...]:
    scope = getattr(screen.app, "research_scope_service", None)
    if scope is None or screen.overlay_store is None:
        return ()
    overlay = await asyncio.to_thread(screen.overlay_store.load, ref)
    if overlay is None:
        return ()
    launches = sorted(
        overlay.deep_research_launches.items(), key=lambda entry: entry[1].created_at, reverse=True,
    )[:4]
    states = await asyncio.gather(
        *(scope.get_run(run_id, mode=ref.data_source.value) for run_id, _ in launches),
        return_exceptions=True,
    )
    rows = []
    for (run_id, origin), state in zip(launches, states, strict=True):
        if origin.origin != ref:
            continue
        if isinstance(state, Exception):
            rows.append("Runs · Status unavailable. Open Runs to refresh.")
            continue
        if DeepResearchBridge.target_from_run(state, server_origin=ref) != origin.target(run_id):
            continue
        status = state.get("status") if isinstance(state, Mapping) else None
        if isinstance(status, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", status):
            rows.append(f"Runs · {status.replace('_', ' ').replace('-', ' ').capitalize()}")
    return tuple(rows)


async def workspace_operation_status(screen) -> str:
    ref = screen.controller.selected_ref
    authority = screen.controller.selected_data_source.value.title()
    workspace = screen.controller.visible_workspace
    owner = workspace.name if workspace else "No workspace selected"
    lines = [f"{authority} · {owner}"]
    if ref is None:
        return "\n".join((*lines, "Select a workspace to view its operations."))
    lines.append(
        "Workspace and sources are stored in Local Workspace and Library."
        if ref.data_source.value == "local"
        else "Workspace and Media are stored on the selected Server."
    )
    capture = screen.controller.capture_request()
    active = []
    if screen.query_one(ResearchChatRegion).is_busy:
        active.append("Grounded Chat responding")
    if screen.query_one(ResearchStudioRegion).is_busy:
        active.append("Studio generation running")
    if active:
        lines.extend(active)
    else:
        lines.append("No active operations in Chat or Studio")
    try:
        operations = await screen._recent_operations(ref)
    except (OSError, ValueError, RuntimeError):
        operations = ()
        lines.append("Source receipts unavailable. Open Sources to refresh.")
    for operation in operations[:12]:
        catalog = operation.catalog_status.value.replace("_", " ")
        association = operation.association_status.value.replace("_", " ")
        readiness = operation.readiness_status.value.replace("_", " ")
        lines.append(
            f"Media intake · Catalog {catalog} · Workspace {association} · Readiness {readiness}"
        )
        if operation.error_stage is not None:
            lines.append("Open the source receipt in Sources for recovery.")
    if screen.copy_workspace is not None:
        try:
            receipts = await screen.copy_workspace.list_history(ref)
        except (OSError, ValueError, RuntimeError):
            receipts = ()
            lines.append(
                "Copy receipts unavailable. Reopen Workspace actions to retry."
            )
        for receipt in receipts[:12]:
            source = receipt.manifest.source.data_source.value.title()
            target = receipt.manifest.destination.data_source.value.title()
            lines.append(
                f"Copy {source} to {target} · {receipt.status.value.replace('_', ' ')}"
            )
    try:
        lines.extend(await recent_workspace_runs(screen, ref))
    except (OSError, ValueError, RuntimeError):
        lines.append("Recent Runs are unavailable. Open Runs to refresh.")
    if not screen.controller.is_current_request(capture):
        return "Workspace changed. Reopen status to refresh."
    lines.append("For durable Research runs, open Runs.")
    return "\n\n".join(lines)
