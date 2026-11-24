"""Read the scoped owners of Research operations for the Help status view."""

from .chat_region import ResearchChatRegion
from .studio_region import ResearchStudioRegion


async def workspace_operation_status(screen) -> str:
    ref = screen.controller.selected_ref
    authority = screen.controller.selected_data_source.value.title()
    workspace = screen.controller.visible_workspace
    owner = workspace.name if workspace else "No workspace selected"
    lines = [f"{authority} · {owner}"]
    if ref is None:
        return "\n".join((*lines, "Select a workspace to view its operations."))
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
    if not screen.controller.is_current_request(capture):
        return "Workspace changed. Reopen status to refresh."
    lines.append("For durable Research runs, open Runs.")
    return "\n\n".join(lines)
