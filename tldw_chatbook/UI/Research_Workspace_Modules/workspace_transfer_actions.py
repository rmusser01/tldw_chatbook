"""Drive confirmed Copy actions against the captured Workspace selection."""

from __future__ import annotations

import asyncio

from loguru import logger

from ...Research_Workspace.contracts import CapabilityUnavailableError, WorkspaceDataSource
from ...Research_Workspace.copy_manifest import ResearchCopyCandidate
from ...Research_Workspace.owner_links import RESEARCH_OWNER_LINKS
from ...Research_Workspace.copy_models import (
    CopyReason,
    ResearchCopyConfirmation,
    ResearchCopyStatus,
)
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Workspaces.models import WorkspaceTransferPolicy
from ..Navigation.main_navigation import NavigateToScreen
from .copy_dialog import (
    ResearchCopyPreview,
    ResearchCopyTargetDialog,
    ResearchWorkspaceActions,
)
from .copy_receipt_view import ResearchCopyProgress, ResearchCopyReceiptView
from .workspace_lifecycle_actions import LIFECYCLE_ACTIONS, run_workspace_lifecycle
from .workspace_presentation_actions import PRESENTATION_ACTIONS, run_workspace_presentation_action


def _current(screen, capture) -> bool:
    return (
        screen.is_attached
        and screen.app.is_running
        and screen.controller.is_current_request(capture)
    )


async def show_workspace_actions(screen, capture) -> None:
    workspace = screen.copy_workspace
    authority = screen.controller.selected_data_source
    revision = screen.controller.context_revision
    ref = capture.ref if capture else None

    def current():
        return (
            screen.is_attached
            and screen.app.is_running
            and screen.controller.selected_data_source is authority
            and screen.controller.context_revision == revision
        )

    try:
        state = screen.controller.catalog_state
        port = screen.controller.port_for_data_source(authority)
        catalog_ready = bool(state and state.recovery is None and port is not None)
        capabilities = await port.capabilities(ref) if ref and catalog_ready else {}
        presentation_available = catalog_ready and screen.overlay_store is not None
        overlay = None
        if presentation_available and ref is not None:
            try:
                overlay = await asyncio.to_thread(screen.overlay_store.load, ref)
            except (OSError, ValueError):
                presentation_available = False
        if not current():
            return
        action = await screen.app.push_screen_wait(
            ResearchWorkspaceActions(
                ref,
                authority=authority,
                catalog_ready=catalog_ready,
                capabilities=capabilities,
                can_copy=workspace is not None,
                can_share=getattr(screen.app_instance, "sharing_scope_service", None)
                is not None,
                presentation_available=presentation_available,
                pinned=bool(overlay and overlay.workspace_presentation.pinned),
            )
        )
        if action is None or not current():
            return
        if action in LIFECYCLE_ACTIONS:
            await run_workspace_lifecycle(screen, action, authority, current)
        elif action in PRESENTATION_ACTIONS:
            await run_workspace_presentation_action(screen, action, authority, current)
        elif action == "import" and authority is WorkspaceDataSource.LOCAL:
            from .workspace_export import import_local_workspace

            await import_local_workspace(screen, current)
        elif capture is None:
            return
        elif action == "share":
            await screen._open_research_sharing(ref)
        elif action == "export":
            from .workspace_export import export_workspace_bundle

            await export_workspace_bundle(screen, ref, current)
        elif action == "collections":
            screen.post_message(NavigateToScreen("library", {"mode": "collections"}))
        elif action == "console":
            link = RESEARCH_OWNER_LINKS[action]
            screen.post_message(NavigateToScreen(link.route, dict(link.context)))
        elif action == "bibtex":
            from .workspace_export import export_workspace_bibtex

            await export_workspace_bibtex(screen, ref, current)
        elif workspace is not None and action == "receipts":
            receipts = await workspace.list_history(ref)
            if current():
                selected = await screen.app.push_screen_wait(
                    ResearchCopyReceiptView(open_receipts=receipts)
                )
                if selected and current():
                    receipt = await workspace.service.get_copy_receipt(
                        selected, workspace=ref
                    )
                    if receipt is not None and current():
                        retry = await screen.app.push_screen_wait(
                            ResearchCopyReceiptView(receipt)
                        )
                        if retry and current():
                            await _run_copy(screen, capture, resume=retry)
        elif workspace is not None and action in {"copy-local", "copy-server", "copy-server-share"}:
            await _prepare_copy(screen, capture, share=action == "copy-server-share")
    except Exception as exc:
        logger.warning(
            "Research workspace action failed (exception_category={})",
            type(exc).__name__,
        )
        if current():
            if isinstance(exc, CapabilityUnavailableError):
                message = f"{exc.capability.user_message} {exc.capability.recovery_action}".strip()
            else:
                message = (
                    str(exc) if type(exc) is ValueError
                    else "The workspace action could not finish. Review the selected owners and retry."
                )
            screen.notify(message, severity="warning")


async def _prepare_copy(screen, capture, *, share: bool) -> None:
    ref, workspace = capture.ref, screen.copy_workspace
    current = lambda: _current(screen, capture)
    if ref.data_source is WorkspaceDataSource.LOCAL:
        visible = screen.controller.visible_workspace
        name = visible.name if visible is not None else ref.workspace_id
        dialog = ResearchCopyTargetDialog(
            server_ref=workspace.new_server_target(),
            default_name=f"{name} copy"[:256],
            allow_sharing_draft=share,
        )
    else:
        targets = await workspace.local_targets()
        dialog = ResearchCopyTargetDialog(local_workspaces=targets)
    if not current():
        return
    target = await screen.app.push_screen_wait(dialog)
    if target is None or not current():
        return
    manifest = await workspace.preflight_copy(ref, target.ref, target_name=target.name)
    while current():
        reviewed = await screen.app.push_screen_wait(ResearchCopyPreview(manifest))
        if reviewed is None or not current():
            return
        if reviewed is True:
            break
        candidates = tuple(
            ResearchCopyCandidate(
                item.source,
                transfer_policy=WorkspaceTransferPolicy.LOCAL_ONLY
                if item.reason is CopyReason.LOCAL_ONLY
                else WorkspaceTransferPolicy.COPY,
                destination=item.destination,
                conflict=reviewed.get(index, item.conflict),
                membership_id=item.membership_id,
                estimated_bytes=item.estimated_bytes,
            )
            for index, item in enumerate(manifest.items)
        )
        manifest = await workspace.preflight_copy(
            ref, target.ref, candidates, target_name=target.name
        )
    if not current():
        return
    replacement = False
    if manifest.requires_replacement_confirmation:
        replacement = (
            await screen.app.push_screen_wait(
                ConfirmationDialog(
                    title="Replace the captured destination version?",
                    message="Copy will replace the specific versions shown in the preview.",
                    confirm_label="Replace selected versions",
                    cancel_label="Cancel",
                )
            )
            is True
        )
        if not replacement or not current():
            return
    confirmation = ResearchCopyConfirmation(manifest.manifest_hash, True, replacement)
    await _run_copy(
        screen,
        capture,
        manifest=manifest,
        confirmation=confirmation,
        share=share,
        share_draft=target.share_draft,
    )


async def _run_copy(
    screen,
    capture,
    *,
    manifest=None,
    confirmation=None,
    resume="",
    share=False,
    share_draft=None,
) -> None:
    current = lambda: _current(screen, capture)
    service = screen.copy_workspace.service
    stop = asyncio.Event()
    progress = ResearchCopyProgress(stop)
    await screen.app.push_screen(progress)
    try:
        if resume:
            receipt = await service.resume_copy(
                resume,
                workspace=capture.ref,
                cancel_event=stop,
                on_progress=progress.update_receipt,
            )
        else:
            receipt = await service.execute_copy(
                manifest,
                confirmation,
                cancel_event=stop,
                on_progress=progress.update_receipt,
            )
    finally:
        if screen.app.screen is progress:
            await screen.app.pop_screen()
    if not current():
        return
    if (
        share
        and receipt.status is ResearchCopyStatus.COMPLETED
        and receipt.server_migration_finalized
    ):
        await screen._open_research_sharing(
            receipt.manifest.destination, draft=share_draft
        )
        return
    next_transfer = await screen.app.push_screen_wait(ResearchCopyReceiptView(receipt))
    if next_transfer and current():
        await _run_copy(
            screen, capture, resume=next_transfer, share=share, share_draft=share_draft
        )
