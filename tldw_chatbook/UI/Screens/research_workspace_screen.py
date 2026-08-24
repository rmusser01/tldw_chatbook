"""Responsive, compose-once Research Workspace foundation screen."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Awaitable, Callable, Literal
from uuid import uuid4

import httpx
from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Resize
from textual.widgets import Button, Static

from ...Research_Workspace import (
    CapabilityUnavailableError,
    QualifiedWorkspaceRef,
    ResearchPresentationOverlayStore,
    ResearchWorkspaceCatalogState,
    ResearchWorkspaceController,
    WorkspaceDataSource,
)
from ...Research_Workspace.overlay_store import (
    OverlayConflictError,
    OverlayLimitError,
    OverlayValidationError,
    ResearchSourceAnnotation,
    ResearchSourceFolder,
)
from ...Research_Workspace.source_operation_store import SourceOperationConflictError
from ...Research_Workspace.source_operations import (
    CanonicalItemType,
    ResearchSourceOperation,
    SourceOperationValidationError,
    SourceOperationStage,
    SourceOperationStatus,
)
from ...tldw_api.exceptions import TLDWAPIError
from ...Research_Workspace.layout_state import (
    ResearchPaneLayout,
    ResearchPanePreferences,
    derive_research_pane_layout,
    toggle_research_pane,
)
from ..Navigation.base_app_screen import BaseAppScreen
from ..Research_Workspace_Modules import (
    ResearchChatRegion,
    ResearchHeaderRegion,
    ResearchPaneHandle,
    ResearchPaneModeStrip,
    ResearchOverlayConflictModal,
    ResearchSourcesRegion,
    ResearchAddSourceModal,
    ResearchSourceIntakeRequest,
    ResearchSourceList,
    ResearchSourceReceiptList,
    ResearchSourceAnnotationDraft,
    ResearchSourceInspectorModal,
    ResearchStudioRegion,
)
from ..Research_Workspace_Modules.pane_handle import ResearchSidePane
from ...Widgets.confirmation_dialog import ConfirmationDialog


ResearchPaneName = Literal["sources", "chat", "studio"]


class ResearchWorkspaceScreen(BaseAppScreen):
    """Sources, Grounded Chat, and Studio shell with in-place responsive layout."""

    CSS_PATH = None
    BINDINGS = []

    def __init__(
        self,
        app_instance: Any,
        *,
        controller: ResearchWorkspaceController | None = None,
        overlay_store: ResearchPresentationOverlayStore | None = None,
        operation_store: Any | None = None,
        association_scheduler: Any | None = None,
        paste_staging_store: Any | None = None,
        operation_id_factory: Callable[[], str] | None = None,
        now_factory: Callable[[], str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(app_instance, "research_workspace", **kwargs)
        self.controller = controller or ResearchWorkspaceController({})
        self.overlay_store = overlay_store
        self.operation_store = operation_store
        self.association_scheduler = association_scheduler
        self.paste_staging_store = paste_staging_store
        self._operation_id_factory = operation_id_factory or (
            lambda: f"research-source-operation-{uuid4().hex}"
        )
        self._now_factory = now_factory or (
            lambda: (
                datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )
        )
        self._source_folders: tuple[ResearchSourceFolder, ...] = ()
        self._source_annotations: tuple[ResearchSourceAnnotation, ...] = ()
        self._source_page_offset = 0
        self._focused_folder_id = ""
        self.pane_preferences = ResearchPanePreferences()
        self.active_pane: ResearchPaneName = "chat"
        self._pane_layout: ResearchPaneLayout | None = None
        self._overlay_revision = 0
        self._overlay_ref: QualifiedWorkspaceRef | None = None
        self._pane_preferences_ref: QualifiedWorkspaceRef | None = None
        self._overlay_generation = 0
        self._overlay_owner_generation = 0
        self._overlay_save_lock = asyncio.Lock()
        self._overlay_save_requested = False
        self._overlay_save_running = False
        self._overlay_committed_revisions: dict[QualifiedWorkspaceRef, int] = {}
        self._overlay_conflict_open = False
        self._overlay_fork_draft: tuple[object, ...] | None = None

    def save_state(self) -> dict[str, object]:
        """Save Phase-1 authority, workspace intent, and responsive view state."""

        state = dict(super().save_state() or {})
        selected_ref = self.controller.selected_ref
        state.update(
            {
                "research_workspace_data_source": (
                    self.controller.selected_data_source.value
                ),
                "research_workspace_ref": (
                    {
                        "data_source": selected_ref.data_source.value,
                        "workspace_id": selected_ref.workspace_id,
                        "server_profile_id": selected_ref.server_profile_id,
                        "principal_id": selected_ref.principal_id,
                    }
                    if selected_ref is not None
                    else None
                ),
                "research_workspace_active_pane": self.active_pane,
                "research_workspace_pane_preferences": {
                    "sources_open": self.pane_preferences.sources_open,
                    "studio_open": self.pane_preferences.studio_open,
                    "preferred_companion": (self.pane_preferences.preferred_companion),
                },
            }
        )
        return state

    def restore_state(self, state: dict[str, object]) -> None:
        """Restore safe view intent before catalog and overlay reconciliation."""

        super().restore_state(state)
        try:
            data_source = WorkspaceDataSource(
                state.get("research_workspace_data_source", "local")
            )
        except (TypeError, ValueError):
            data_source = WorkspaceDataSource.LOCAL

        active_pane = state.get("research_workspace_active_pane", "chat")
        if active_pane in {"sources", "chat", "studio"}:
            self.active_pane = active_pane

        raw_preferences = state.get("research_workspace_pane_preferences")
        if isinstance(raw_preferences, Mapping):
            try:
                self.pane_preferences = ResearchPanePreferences(
                    sources_open=raw_preferences.get("sources_open", True),
                    studio_open=raw_preferences.get("studio_open", True),
                    preferred_companion=raw_preferences.get(
                        "preferred_companion", "sources"
                    ),
                )
            except (TypeError, ValueError):
                self.pane_preferences = ResearchPanePreferences()

        selected_ref = None
        raw_ref = state.get("research_workspace_ref")
        if isinstance(raw_ref, Mapping):
            try:
                selected_ref = QualifiedWorkspaceRef(
                    raw_ref.get("data_source", data_source.value),
                    raw_ref.get("workspace_id", ""),
                    server_profile_id=raw_ref.get("server_profile_id", ""),
                    principal_id=raw_ref.get("principal_id", ""),
                )
            except (TypeError, ValueError):
                selected_ref = None
        if selected_ref is not None and selected_ref.data_source is not data_source:
            selected_ref = None

        self.controller.select_data_source(data_source)
        if selected_ref is not None:
            self.controller.select_workspace(selected_ref)
        self._pane_preferences_ref = selected_ref

    def compose_content(self) -> ComposeResult:
        with Vertical(id="research-workspace-shell"):
            yield ResearchHeaderRegion(id="research-workspace-header")
            yield ResearchPaneModeStrip(id="research-pane-mode-strip")
            with Horizontal(id="research-workspace-grid"):
                yield ResearchSourcesRegion(
                    id="research-sources-pane", classes="research-workspace-pane"
                )
                yield ResearchPaneHandle("sources", id="research-sources-handle")
                yield ResearchChatRegion(
                    id="research-chat-pane", classes="research-workspace-pane"
                )
                yield ResearchPaneHandle("studio", id="research-studio-handle")
                yield ResearchStudioRegion(
                    id="research-studio-pane", classes="research-workspace-pane"
                )
            yield Static(
                "Foundation ready · Workspace service setup required · No operation active",
                id="research-workspace-status",
                markup=False,
            )

    def on_mount(self) -> None:
        """Derive initial layout after every compose-once child is mounted."""
        self._apply_pane_layout(max(1, self.size.width))
        self.query_one(
            "#research-workspace-header", ResearchHeaderRegion
        ).sync_data_source(self.controller.selected_data_source)
        self._start_catalog_refresh()

    def on_resize(self, event: Resize) -> None:
        """Patch effective visibility and focus from the pure width reducer."""
        self._apply_pane_layout(max(1, event.size.width), relocate_hidden_focus=True)

    @on(ResearchPaneHandle.Toggled)
    def toggle_side_pane(self, message: ResearchPaneHandle.Toggled) -> None:
        """Apply an explicit wide/medium side-pane preference change."""
        width = max(1, self.size.width)
        if width < 100:
            self.active_pane = message.pane
            self._apply_pane_layout(width)
            self._focus_pane(message.pane)
            return

        self.pane_preferences = toggle_research_pane(
            self.pane_preferences, message.pane, width=width
        )
        self._apply_pane_layout(width)
        self._start_overlay_save()
        if message.reveal:
            self._focus_pane(message.pane)
        else:
            self._focus_reveal(message.pane)

    @on(ResearchHeaderRegion.DataSourceSelected)
    def select_data_source(
        self, message: ResearchHeaderRegion.DataSourceSelected
    ) -> None:
        """Switch the whole catalog owner without reading the other authority."""

        self.controller.select_data_source(message.data_source)
        self._source_page_offset = 0
        self._source_folders = ()
        self._source_annotations = ()
        self._overlay_generation += 1
        self._set_overlay_ref(None)
        self._pane_preferences_ref = None
        self.pane_preferences = ResearchPanePreferences()
        self._apply_pane_layout(max(1, self.size.width), relocate_hidden_focus=True)
        header = self.query_one("#research-workspace-header", ResearchHeaderRegion)
        header.sync_data_source(message.data_source)
        self.query_one("#research-workspace-selection", Static).update(
            f"Loading {message.data_source.value.title()} research workspaces..."
        )
        self.query_one("#research-sources-pane", ResearchSourcesRegion).clear_workspace(
            authority=message.data_source.value.title(),
            reason=f"Loading {message.data_source.value.title()} workspace sources...",
        )
        self._start_catalog_refresh()

    @on(ResearchPaneModeStrip.Selected)
    def select_pane_mode(self, message: ResearchPaneModeStrip.Selected) -> None:
        """Reveal the selected pane according to the current responsive band."""
        width = max(1, self.size.width)
        preferences_before = self.pane_preferences
        if width < 100:
            self.active_pane = message.pane
        elif (
            message.pane != "chat"
            and message.pane
            not in derive_research_pane_layout(
                width, self.pane_preferences
            ).visible_panes
        ):
            self.pane_preferences = toggle_research_pane(
                self.pane_preferences, message.pane, width=width
            )
        self._apply_pane_layout(width)
        if self.pane_preferences != preferences_before:
            self._start_overlay_save()
        self._focus_pane(message.pane)

    def _apply_pane_layout(
        self, width: int, *, relocate_hidden_focus: bool = False
    ) -> None:
        focused_pane = self._pane_for_widget(self.app.focused)
        layout = derive_research_pane_layout(
            width, self.pane_preferences, active_pane=self.active_pane
        )
        self._pane_layout = layout

        grid = self.query_one("#research-workspace-grid")
        for mode in ("wide", "medium", "narrow"):
            grid.set_class(layout.mode == mode, f"layout-{mode}")
        shell = self.query_one("#research-workspace-shell")
        shell.set_class(self.size.height < 30, "height-compact")

        for pane in ("sources", "chat", "studio"):
            self.query_one(f"#research-{pane}-pane").display = (
                pane in layout.visible_panes
            )

        handles_visible = layout.mode != "narrow"
        for pane in ("sources", "studio"):
            handle = self.query_one(f"#research-{pane}-handle", ResearchPaneHandle)
            handle.sync_expanded(
                pane in layout.visible_panes, handle_visible=handles_visible
            )
            handle.display = handles_visible

        mode_strip = self.query_one("#research-pane-mode-strip", ResearchPaneModeStrip)
        mode_strip.display = layout.mode != "wide"
        mode_strip.sync_visible_panes(layout.visible_panes)

        if (
            relocate_hidden_focus
            and focused_pane is not None
            and focused_pane not in layout.visible_panes
        ):
            button = self.query_one(f"#research-pane-mode-{focused_pane}", Button)
            if button.display:
                button.focus()
                self.notify(
                    f"Layout changed; {focused_pane.title()} pane is hidden. "
                    "Use the pane mode controls to restore it."
                )

    def _pane_for_widget(self, widget: object | None) -> ResearchPaneName | None:
        current = widget
        while current is not None:
            widget_id = getattr(current, "id", None)
            for pane in ("sources", "chat", "studio"):
                if widget_id in {
                    f"research-{pane}-pane",
                    f"research-{pane}-handle",
                    f"research-{pane}-collapse",
                    f"research-{pane}-reveal",
                }:
                    return pane
            current = getattr(current, "parent", None)
        return None

    def _focus_pane(self, pane: ResearchPaneName) -> None:
        target = self.query_one(f"#research-{pane}-pane")
        if target.display:
            target.focus()

    def _focus_reveal(self, pane: ResearchSidePane) -> None:
        handle = self.query_one(f"#research-{pane}-handle", ResearchPaneHandle)
        button = handle.query_one(f"#research-{pane}-reveal", Button)
        if handle.display and button.display:
            button.focus()

    def _start_catalog_refresh(self) -> None:
        self.run_worker(
            self._refresh_workspace_catalog(),
            group="research-workspace-catalog",
            exclusive=True,
        )

    async def _refresh_workspace_catalog(self) -> None:
        state = await self.controller.refresh_workspace_catalog()
        if not self.controller.is_current_catalog_state(state):
            return
        await self._apply_catalog_state(state)

    async def _apply_catalog_state(self, state: ResearchWorkspaceCatalogState) -> None:
        if not self.controller.is_current_catalog_state(state):
            return
        header = self.query_one("#research-workspace-header", ResearchHeaderRegion)
        header.sync_catalog_state(state)
        selection = self.query_one("#research-workspace-selection", Static)
        status = self.query_one("#research-workspace-status", Static)
        if state.recovery is not None:
            selection.update(
                f"{state.data_source.value.title()} workspace catalog unavailable."
            )
            status.update(
                f"Foundation ready · {state.data_source.value.title()} selected · "
                "Recovery required · No operation active"
            )
            self._set_overlay_ref(None)
            self.query_one(
                "#research-sources-pane", ResearchSourcesRegion
            ).clear_workspace(
                authority=state.data_source.value.title(),
                reason=state.recovery.user_message,
            )
            return
        if not state.workspaces:
            selection.update(
                f"No {state.data_source.value.title()} research workspaces."
            )
            status.update(
                f"Foundation ready · {state.data_source.value.title()} catalog ready · "
                "0 workspaces · Foundation only"
            )
            self._set_overlay_ref(None)
            self.query_one(
                "#research-sources-pane", ResearchSourcesRegion
            ).clear_workspace(
                authority=state.data_source.value.title(),
            )
            return

        intended_ref = self.controller.selected_ref
        workspace = next(
            (
                candidate
                for candidate in state.workspaces
                if intended_ref is not None and candidate.ref == intended_ref
            ),
            state.workspaces[0],
        )
        self.controller.select_workspace(workspace.ref)
        self._source_page_offset = 0
        self._source_folders = ()
        self._source_annotations = ()
        selection.update(
            f"{workspace.name} · {workspace.ref.data_source.value.title()}"
        )
        status.update(
            f"Foundation ready · {state.data_source.value.title()} catalog ready · "
            f"{len(state.workspaces)} workspace(s) · Foundation only"
        )
        if self._pane_preferences_ref != workspace.ref:
            self.pane_preferences = ResearchPanePreferences()
            self._pane_preferences_ref = workspace.ref
            self._apply_pane_layout(max(1, self.size.width), relocate_hidden_focus=True)
        self._overlay_generation += 1
        overlay_generation = self._overlay_generation
        overlay_capture = self.controller.capture_request()
        self._set_overlay_ref(workspace.ref)
        self._start_sources_refresh()
        if self.overlay_store is None:
            return
        try:
            overlay = await asyncio.to_thread(self.overlay_store.load, workspace.ref)
        except (OSError, ValueError):
            self.notify(
                "Device-only pane preferences are unavailable; workspace data is unchanged.",
                severity="warning",
            )
            return
        if overlay is None:
            return
        if (
            getattr(overlay, "ref", None) != workspace.ref
            or overlay_generation != self._overlay_generation
            or workspace.ref != self._overlay_ref
            or not self.controller.is_current_request(overlay_capture)
        ):
            return
        self._overlay_revision = overlay.revision
        self.pane_preferences = overlay.preferences
        self._source_folders = overlay.source_folders
        self._source_annotations = overlay.source_annotations
        self._pane_preferences_ref = workspace.ref
        self._apply_pane_layout(max(1, self.size.width), relocate_hidden_focus=True)
        self._start_sources_refresh()

    def _start_sources_refresh(self) -> None:
        if self.controller.selected_ref is None or not self.is_mounted:
            return
        self.run_worker(
            self._refresh_source_workbench(),
            group="research-workspace-sources",
            exclusive=True,
        )

    def _run_source_action(
        self,
        action: Awaitable[Any],
        *,
        group: str,
        exclusive: bool = False,
        recovery: str = "Source action could not be completed. Refresh Sources and retry.",
    ) -> None:
        """Contain expected source failures without making them app-fatal."""

        self.run_worker(
            self._guard_source_action(action, recovery=recovery),
            group=group,
            exclusive=exclusive,
            exit_on_error=False,
        )

    async def _guard_source_action(
        self, action: Awaitable[Any], *, recovery: str
    ) -> None:
        """Show bounded recovery for expected failures and log unexpected defects."""

        try:
            await action
        except asyncio.CancelledError:
            raise
        except CapabilityUnavailableError as exc:
            self._show_source_action_recovery(
                f"{exc.capability.user_message} "
                f"{exc.capability.recovery_action}".strip()
            )
        except (
            OverlayConflictError,
            OverlayLimitError,
            OverlayValidationError,
            SourceOperationConflictError,
            SourceOperationValidationError,
            TLDWAPIError,
            httpx.HTTPError,
            OSError,
        ):
            self._show_source_action_recovery(recovery)
        except Exception:
            logger.exception("Unexpected Research source action failure")
            self._show_source_action_recovery(
                "Unexpected source action failure. Refresh Sources; details were logged."
            )

    def _show_source_action_recovery(self, message: str) -> None:
        """Publish one sanitized recovery message in the pane and notification log."""

        if self.is_mounted:
            self.query_one("#research-source-recovery", Static).update(message)
        self.notify(message, severity="warning")

    async def _refresh_source_workbench(self) -> None:
        capture = self.controller.capture_request()
        region = self.query_one("#research-sources-pane", ResearchSourcesRegion)
        region.query_one("#research-source-recovery", Static).update(
            f"Loading {capture.ref.data_source.value.title()} sources..."
        )
        try:
            operations = await self._recent_operations(capture.ref)
        except (OSError, ValueError, RuntimeError):
            operations = ()
            if self.controller.is_current_request(capture):
                self.notify(
                    "Recent source receipts could not be loaded; owner sources can still refresh.",
                    severity="warning",
                )
        if not self.controller.is_current_request(capture):
            return
        region.sync_receipts(operations, incomplete=len(operations) == 20)
        try:
            capabilities_current = await self.controller.refresh_selected_capabilities()
            sources_current = await self.controller.refresh_selected_sources(
                limit=25, offset=self._source_page_offset
            )
            page = self.controller.visible_source_page
            if not capabilities_current or not sources_current or page is None:
                return
            readiness_current = await self.controller.refresh_selected_readiness(
                source_ids=tuple(source.source_id for source in page.items)
            )
            if not readiness_current or not self.controller.is_current_request(capture):
                return
        except CapabilityUnavailableError as exc:
            if self.controller.is_current_request(capture):
                region.clear_source_projection(
                    authority=capture.ref.data_source.value.title(),
                    reason=f"{exc.capability.user_message} {exc.capability.recovery_action}".strip(),
                )
            return
        except Exception:
            if self.controller.is_current_request(capture):
                region.clear_source_projection(
                    authority=capture.ref.data_source.value.title(),
                    reason=(
                        "Sources could not be loaded from the selected owner. "
                        "Refresh or verify that authority's service."
                    ),
                )
            return
        if not self.controller.is_current_request(capture):
            return
        region.sync_workspace(
            page,
            readiness=self.controller.visible_readiness,
            capabilities=self.controller.visible_capabilities,
            folders=self._source_folders,
            operations=operations,
            focused_folder_id=self._focused_folder_id,
            receipts_incomplete=len(operations) == 20,
        )

    async def _recent_operations(
        self, ref: QualifiedWorkspaceRef
    ) -> tuple[ResearchSourceOperation, ...]:
        if self.operation_store is None:
            return ()
        return await asyncio.to_thread(
            self.operation_store.list_recent,
            data_source=ref.data_source,
            server_profile_id=ref.server_profile_id,
            principal_id=ref.principal_id,
            workspace_id=ref.workspace_id,
            limit=20,
        )

    @on(ResearchSourcesRegion.AddRequested)
    def open_add_sources(self) -> None:
        ref = self.controller.selected_ref
        if ref is None:
            self.notify("Select a Research workspace first.", severity="warning")
            return

        async def catalog_search(**kwargs):
            if ref != self.controller.selected_ref:
                return None
            accepted = await self.controller.search_selected_catalog(**kwargs)
            return self.controller.visible_catalog_page if accepted else None

        def submitted(request: ResearchSourceIntakeRequest | None) -> None:
            if request is not None:
                self._run_source_action(
                    self._submit_intake_request(ref, request),
                    group="research-source-intake",
                )

        self.app.push_screen(
            ResearchAddSourceModal(ref.data_source, catalog_search=catalog_search),
            callback=submitted,
        )

    @on(ResearchSourcesRegion.QuickUrlRequested)
    def quick_add_url(self, message: ResearchSourcesRegion.QuickUrlRequested) -> None:
        ref = self.controller.selected_ref
        if ref is not None:
            self._run_source_action(
                self._submit_intake_request(
                    ref, ResearchSourceIntakeRequest("url", (message.url,))
                ),
                group="research-source-intake",
            )

    @on(ResearchSourcesRegion.RefreshRequested)
    def refresh_sources(self) -> None:
        self._start_sources_refresh()

    @on(ResearchSourcesRegion.PageRequested)
    def change_source_page(self, message: ResearchSourcesRegion.PageRequested) -> None:
        self._source_page_offset = max(0, self._source_page_offset + message.delta * 25)
        self._start_sources_refresh()

    @on(ResearchSourcesRegion.SelectionScopeRequested)
    def change_selection_scope(
        self, message: ResearchSourcesRegion.SelectionScopeRequested
    ) -> None:
        async def apply() -> None:
            if message.mode == "all":
                await self.controller.select_all_sources()
            elif message.mode == "clear":
                await self.controller.set_selected_scope(())
            else:
                region = self.query_one("#research-sources-pane", ResearchSourcesRegion)
                desired = tuple(
                    dict.fromkeys(
                        (
                            *self.controller.desired_source_ids,
                            *region.visible_owner_ids(),
                        )
                    )
                )
                await self.controller.set_selected_scope(desired)
            self._start_sources_refresh()

        self._run_source_action(
            apply(), group="research-source-selection", exclusive=True
        )

    @on(ResearchSourceList.SelectionToggled)
    def toggle_source_selection(
        self, message: ResearchSourceList.SelectionToggled
    ) -> None:
        desired = list(self.controller.desired_source_ids)
        if message.selected and message.desired_owner_id not in desired:
            desired.append(message.desired_owner_id)
        elif not message.selected:
            desired = [item for item in desired if item != message.desired_owner_id]

        async def apply() -> None:
            await self.controller.set_selected_scope(tuple(desired))
            self._start_sources_refresh()

        self._run_source_action(
            apply(), group="research-source-selection", exclusive=True
        )

    @on(ResearchSourceList.ReorderRequested)
    def reorder_source(self, message: ResearchSourceList.ReorderRequested) -> None:
        async def apply() -> None:
            await self.controller.move_selected_source(
                message.source_id, message.delta
            )
            self._start_sources_refresh()

        self._run_source_action(
            apply(), group="research-source-reorder", exclusive=True
        )

    @on(ResearchSourceList.ActionRequested)
    def source_action(self, message: ResearchSourceList.ActionRequested) -> None:
        if message.action == "remove":
            def confirmed(accepted: bool | None) -> None:
                if accepted:
                    self._run_source_action(
                        self._remove_source(message.source_id),
                        group="research-source-remove",
                    )

            self.app.push_screen(
                ConfirmationDialog(
                    title="Remove source association?",
                    message=(
                        "This removes from this workspace; "
                        "Library/Media item is retained."
                    ),
                    confirm_label="Remove association",
                ),
                callback=confirmed,
            )
        elif message.action in {"details", "preview"}:
            self._run_source_action(
                self._show_source_inspector(
                    message.source_id, load_preview=message.action == "preview"
                ),
                group="research-source-preview",
                exclusive=True,
            )
        elif message.action == "folders":
            self._toggle_source_folder(message.source_id)
        elif message.action == "copy":
            self.notify(
                "Move / Copy is unavailable because the selected owner exposes no canonical action.",
                severity="warning",
            )

    @on(ResearchSourcesRegion.BatchRequested)
    def batch_source_action(
        self, message: ResearchSourcesRegion.BatchRequested
    ) -> None:
        region = self.query_one("#research-sources-pane", ResearchSourcesRegion)
        source_ids = region.selected_source_ids()
        if message.action == "preview-selected" and len(source_ids) == 1:
            self._run_source_action(
                self._show_source_inspector(source_ids[0], load_preview=True),
                group="research-source-preview",
                exclusive=True,
            )
        elif message.action == "remove-selected" and source_ids:

            async def remove_all() -> None:
                for source_id in source_ids:
                    await self.controller.remove_selected_source(source_id)
                removed = frozenset(source_ids)
                self._source_folders = tuple(
                    ResearchSourceFolder(
                        folder.folder_id,
                        folder.name,
                        tuple(
                            item for item in folder.source_ids if item not in removed
                        ),
                        folder.parent_folder_id,
                    )
                    for folder in self._source_folders
                )
                self._start_overlay_save()
                self._start_sources_refresh()

            def confirmed(accepted: bool | None) -> None:
                if accepted:
                    self._run_source_action(
                        remove_all(),
                        group="research-source-remove",
                        exclusive=True,
                    )

            self.app.push_screen(
                ConfirmationDialog(
                    title="Remove visible source associations?",
                    message=(
                        "This removes from this workspace; Library/Media item is retained. "
                        "Only selected associations on this visible page are removed."
                    ),
                    confirm_label="Remove associations",
                ),
                callback=confirmed,
            )
        else:
            self.notify(
                "Move / Copy is unavailable because the selected owner exposes no canonical action.",
                severity="warning",
            )

    async def _remove_source(self, source_id: str) -> None:
        await self.controller.remove_selected_source(source_id)
        self._source_folders = tuple(
            ResearchSourceFolder(
                folder.folder_id,
                folder.name,
                tuple(item for item in folder.source_ids if item != source_id),
                folder.parent_folder_id,
            )
            for folder in self._source_folders
        )
        self._start_overlay_save()
        self._start_sources_refresh()

    @on(ResearchSourcesRegion.FolderRequested)
    def folder_action(self, message: ResearchSourcesRegion.FolderRequested) -> None:
        if message.action == "new":
            if not message.name:
                self.notify("Enter a device-only folder name.", severity="warning")
                return
            self._source_folders = (
                *self._source_folders,
                ResearchSourceFolder(
                    f"folder-{uuid4().hex}",
                    message.name,
                    parent_folder_id=message.parent_folder_id,
                ),
            )
        elif message.action == "rename":
            if not message.folder_id or not message.name:
                self.notify(
                    "Choose a folder and enter its new name.", severity="warning"
                )
                return
            self._source_folders = tuple(
                ResearchSourceFolder(
                    folder.folder_id,
                    message.name
                    if folder.folder_id == message.folder_id
                    else folder.name,
                    folder.source_ids,
                    folder.parent_folder_id,
                )
                for folder in self._source_folders
            )
        elif message.action == "focus":
            self._focused_folder_id = (
                ""
                if self._focused_folder_id == message.folder_id
                else message.folder_id
            )
            self.notify("Folder focus is device-only; retrieval scope is unchanged.")
        elif message.action == "select-folder":
            self._select_folder_sources(message.folder_id)
            return
        self._start_overlay_save()
        self._start_sources_refresh()

    def _toggle_source_folder(self, source_id: str) -> None:
        region = self.query_one("#research-sources-pane", ResearchSourcesRegion)
        folder_id = str(region.query_one("#research-source-folder-tree").value or "")
        if not folder_id:
            self.notify(
                "Choose or create a device-only folder first.", severity="warning"
            )
            return
        self._source_folders = tuple(
            ResearchSourceFolder(
                folder.folder_id,
                folder.name,
                (
                    tuple(item for item in folder.source_ids if item != source_id)
                    if source_id in folder.source_ids
                    else (*folder.source_ids, source_id)
                )
                if folder.folder_id == folder_id
                else folder.source_ids,
                folder.parent_folder_id,
            )
            for folder in self._source_folders
        )
        self._start_overlay_save()
        self._start_sources_refresh()

    def _select_folder_sources(self, folder_id: str) -> None:
        folder = next(
            (item for item in self._source_folders if item.folder_id == folder_id),
            None,
        )
        page = self.controller.visible_source_page
        if folder is None or page is None:
            return
        by_source = {source.source_id: source for source in page.items}
        if any(source_id not in by_source for source_id in folder.source_ids):
            self.notify(
                "Some folder sources are outside this page; load them before changing retrieval scope.",
                severity="warning",
            )
            return
        desired = tuple(
            by_source[source_id].catalog_item_id
            if by_source[source_id].ref.data_source is WorkspaceDataSource.LOCAL
            else source_id
            for source_id in folder.source_ids
        )

        async def apply() -> None:
            await self.controller.set_selected_scope(desired)
            self._start_sources_refresh()

        self._run_source_action(
            apply(), group="research-source-selection", exclusive=True
        )

    async def _show_source_inspector(
        self, source_id: str, *, load_preview: bool
    ) -> None:
        ref = self.controller.selected_ref
        page = self.controller.visible_source_page
        if ref is None or page is None:
            return
        source = next(
            (item for item in page.items if item.source_id == source_id), None
        )
        if source is None:
            return
        if load_preview:
            await self.controller.preview_selected_source(source_id)
        if ref != self.controller.selected_ref:
            return
        readiness = self.controller.canonical_source_readiness(ref, source_id)
        preview = self.controller.canonical_source_preview(ref, source_id)

        def save_annotation(draft: ResearchSourceAnnotationDraft | None) -> None:
            if draft is None:
                return
            if draft.action == "recheck":
                self._start_sources_refresh()
                return
            if draft.source_id != source_id or ref != self.controller.selected_ref:
                self.notify(
                    "Annotation owner changed; reopen source details.",
                    severity="warning",
                )
                return
            now = self._now_factory()
            existing = next(
                (
                    item
                    for item in self._source_annotations
                    if item.annotation_id == draft.annotation_id
                    and item.source_id == source_id
                ),
                None,
            )
            if draft.action == "delete":
                if existing is None:
                    return
                self._source_annotations = tuple(
                    item
                    for item in self._source_annotations
                    if item.annotation_id != existing.annotation_id
                )
            elif draft.action == "update":
                if existing is None:
                    self.notify(
                        "Annotation changed on this device; reopen source details.",
                        severity="warning",
                    )
                    return
                self._source_annotations = tuple(
                    ResearchSourceAnnotation(
                        annotation_id=item.annotation_id,
                        source_id=item.source_id,
                        quote=draft.quote,
                        note=draft.note,
                        created_at=item.created_at,
                        updated_at=now,
                    )
                    if item.annotation_id == existing.annotation_id
                    else item
                    for item in self._source_annotations
                )
            else:
                self._source_annotations = (
                    *self._source_annotations,
                    ResearchSourceAnnotation(
                        annotation_id=f"annotation-{uuid4().hex}",
                        source_id=draft.source_id,
                        quote=draft.quote,
                        note=draft.note,
                        created_at=now,
                        updated_at=now,
                    ),
                )
            self._start_overlay_save()

        self.app.push_screen(
            ResearchSourceInspectorModal(
                source,
                readiness=readiness,
                preview=preview,
                annotations=tuple(
                    annotation
                    for annotation in self._source_annotations
                    if annotation.source_id == source_id
                ),
            ),
            callback=save_annotation,
        )

    @on(ResearchSourceReceiptList.RetryRequested)
    def retry_source_stage(
        self, message: ResearchSourceReceiptList.RetryRequested
    ) -> None:
        scheduler = self.association_scheduler
        if scheduler is None:
            self.notify("Source operation retry is unavailable.", severity="warning")
            return

        async def retry() -> None:
            operation = await scheduler.retry(
                message.operation_id, stage=message.stage
            )
            if (
                operation is not None
                and operation.catalog_status is SourceOperationStatus.SUCCEEDED
                and self.paste_staging_store is not None
            ):
                await asyncio.to_thread(
                    self.paste_staging_store.delete, message.operation_id
                )
            self._start_sources_refresh()

        self._run_source_action(retry(), group="research-source-retry")

    def _start_overlay_save(self) -> None:
        if self.overlay_store is None or self._overlay_ref is None:
            return
        self._overlay_generation += 1
        self._overlay_save_requested = True
        if self._overlay_save_running:
            return
        self._overlay_save_running = True
        self.run_worker(
            self._drain_overlay_saves(),
            group="research-workspace-overlay",
            exclusive=False,
        )

    async def _drain_overlay_saves(self) -> None:
        """Serialize and coalesce pane writes without cancelling committed work."""

        try:
            while self._overlay_save_requested:
                self._overlay_save_requested = False
                async with self._overlay_save_lock:
                    await self._save_overlay_preferences()
        finally:
            self._overlay_save_running = False

    async def _save_overlay_preferences(self) -> None:
        store = self.overlay_store
        ref = self._overlay_ref
        if store is None or ref is None:
            return
        owner_generation = self._overlay_owner_generation
        preferences = self.pane_preferences
        source_folders = self._source_folders
        source_annotations = self._source_annotations
        expected_revision = max(
            self._overlay_revision,
            self._overlay_committed_revisions.get(ref, 0),
        )
        try:
            saved = await asyncio.to_thread(
                store.save,
                ref,
                preferences,
                expected_revision=expected_revision,
                source_folders=source_folders,
                source_annotations=source_annotations,
            )
        except OverlayConflictError:
            self._open_overlay_conflict_recovery(ref)
            return
        except (OSError, ValueError, RuntimeError):
            self.notify(
                "Device-only pane preference was not saved; retry the pane action.",
                severity="warning",
            )
            return
        self._overlay_committed_revisions[ref] = max(
            saved.revision,
            self._overlay_committed_revisions.get(ref, 0),
        )
        if (
            ref == self._overlay_ref
            and owner_generation == self._overlay_owner_generation
        ):
            self._overlay_revision = saved.revision

    def _open_overlay_conflict_recovery(self, ref: QualifiedWorkspaceRef) -> None:
        """Expose explicit device-only recovery without changing the draft."""

        if self._overlay_conflict_open:
            return
        if not self.is_mounted:
            self.notify(
                "Device overlay changed; reopen Research to choose recovery.",
                severity="warning",
            )
            return
        self._overlay_conflict_open = True

        def chosen(action: str | None) -> None:
            self._overlay_conflict_open = False
            if action == "reload":
                self._run_source_action(
                    self._reload_overlay_after_conflict(ref),
                    group="research-overlay-conflict",
                    exclusive=True,
                    recovery="Device overlay could not be reloaded; local draft remains.",
                )
            elif action == "export":
                self.app.copy_to_clipboard(self._device_overlay_recovery_export(ref))
                self.notify(
                    "Private-free device overlay recovery metadata copied.",
                )
            elif action == "fork":
                self._overlay_fork_draft = (
                    self.pane_preferences,
                    self._source_folders,
                    self._source_annotations,
                )
                self.notify(
                    "Device layout copy retained in memory; no owner data was overwritten."
                )

        self.app.push_screen(ResearchOverlayConflictModal(), callback=chosen)

    async def _reload_overlay_after_conflict(
        self, ref: QualifiedWorkspaceRef
    ) -> None:
        """Replace the local draft only after the user explicitly chooses Reload."""

        store = self.overlay_store
        if store is None:
            return
        overlay = await asyncio.to_thread(store.load, ref)
        if overlay is None or ref != self._overlay_ref:
            return
        self._overlay_revision = overlay.revision
        self._overlay_committed_revisions[ref] = overlay.revision
        self.pane_preferences = overlay.preferences
        self._source_folders = overlay.source_folders
        self._source_annotations = overlay.source_annotations
        self._apply_pane_layout(max(1, self.size.width), relocate_hidden_focus=True)
        self._start_sources_refresh()

    def _device_overlay_recovery_export(
        self, ref: QualifiedWorkspaceRef
    ) -> str:
        """Return bounded metadata-only recovery JSON with opaque IDs hashed."""

        def opaque(value: str) -> str:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

        payload = {
            "schema_version": 1,
            "owner": {
                "data_source": ref.data_source.value,
                "workspace": opaque(ref.workspace_id),
                "server_profile": opaque(ref.server_profile_id)
                if ref.server_profile_id
                else "",
                "principal": opaque(ref.principal_id) if ref.principal_id else "",
            },
            "preferences": {
                "sources_open": self.pane_preferences.sources_open,
                "studio_open": self.pane_preferences.studio_open,
                "preferred_companion": self.pane_preferences.preferred_companion,
            },
            "folders": [
                {
                    "id": opaque(folder.folder_id),
                    "name": folder.name,
                    "parent": opaque(folder.parent_folder_id)
                    if folder.parent_folder_id
                    else "",
                    "sources": [opaque(source_id) for source_id in folder.source_ids],
                }
                for folder in self._source_folders
            ],
            "annotations": [
                {
                    "id": opaque(annotation.annotation_id),
                    "source": opaque(annotation.source_id),
                }
                for annotation in self._source_annotations
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _set_overlay_ref(self, ref: QualifiedWorkspaceRef | None) -> None:
        """Change overlay ownership and invalidate revisions from the prior ref."""

        if ref == self._overlay_ref:
            return
        self._overlay_owner_generation += 1
        self._overlay_ref = ref
        self._overlay_revision = 0

    async def _submit_intake_request(
        self,
        ref: QualifiedWorkspaceRef,
        request: ResearchSourceIntakeRequest,
    ) -> None:
        """Persist each captured intent before submitting it to Library ingest."""

        await self.controller.require_workspace_capability(ref, "attach_existing")
        if request.kind in {"existing", "catalog"}:
            for catalog_item_id in request.values:
                await self._attach_existing(ref, catalog_item_id)
            return
        for source_value in request.values:
            operation = ResearchSourceOperation(
                operation_id=self._operation_id_factory(),
                idempotency_key=f"research-intake-{uuid4().hex}",
                data_source=ref.data_source,
                server_profile_id=ref.server_profile_id,
                principal_id=ref.principal_id,
                workspace_id=ref.workspace_id,
                canonical_item_type=(
                    CanonicalItemType.LOCAL_LIBRARY
                    if ref.data_source is WorkspaceDataSource.LOCAL
                    else CanonicalItemType.SERVER_MEDIA
                ),
                desired_selected=True,
                created_at=self._now_factory(),
                updated_at=self._now_factory(),
            )
            if self.operation_store is None:
                raise RuntimeError("Durable Research source intake is unavailable.")
            operation = await asyncio.to_thread(self.operation_store.create, operation)
            operation_id = operation.operation_id
            source_path = source_value
            staged_paste = False
            if request.kind == "paste":
                staging_store = self.paste_staging_store
                if staging_store is None:
                    await asyncio.to_thread(
                        self.operation_store.advance_stage,
                        operation.operation_id,
                        stage=SourceOperationStage.CATALOG,
                        status=SourceOperationStatus.FAILED,
                        expected_revision=operation.revision,
                        error_code="paste_staging_unavailable",
                        error_message="Private paste staging is unavailable.",
                    )
                    continue
                try:
                    staged_path = await asyncio.to_thread(
                        staging_store.stage,
                        operation.operation_id,
                        title=request.title,
                        body=source_value,
                    )
                except Exception:
                    await asyncio.to_thread(
                        self.operation_store.advance_stage,
                        operation.operation_id,
                        stage=SourceOperationStage.CATALOG,
                        status=SourceOperationStatus.FAILED,
                        expected_revision=operation.revision,
                        error_code="paste_staging_failed",
                        error_message="Private paste staging could not be created.",
                    )
                    continue
                source_path = str(staged_path)
                staged_paste = True
            try:
                job = self.app_instance.submit_library_ingest_job(
                    source_path=source_path,
                    title=request.title,
                    research_source_operation_id=operation.operation_id,
                    required_origin=ref.data_source.value,
                )
            except Exception:
                await asyncio.to_thread(
                    self.operation_store.advance_stage,
                    operation.operation_id,
                    stage=SourceOperationStage.CATALOG,
                    status=SourceOperationStatus.FAILED,
                    expected_revision=operation.revision,
                    error_code="catalog_submit_failed",
                    error_message="Catalog intake could not be started for the selected authority.",
                )
                if staged_paste:
                    await asyncio.to_thread(
                        self.paste_staging_store.delete,
                        operation.operation_id,
                    )
                continue
            operation = await asyncio.to_thread(
                self.operation_store.advance_stage,
                operation.operation_id,
                stage=SourceOperationStage.CATALOG,
                status=SourceOperationStatus.IN_PROGRESS,
                expected_revision=operation.revision,
                ingest_job_id=job.job_id,
            )
            job_state = str(getattr(getattr(job, "state", None), "value", ""))
            if job_state in {"done", "failed", "cancelled", "skipped"}:
                scheduler = self.association_scheduler
                if scheduler is not None:
                    operation = await scheduler.resume(operation_id)
                if (
                    staged_paste
                    and (
                        job_state in {"cancelled", "skipped"}
                        or (
                            operation is not None
                            and operation.catalog_status
                            is SourceOperationStatus.SUCCEEDED
                        )
                    )
                ):
                    await asyncio.to_thread(
                        self.paste_staging_store.delete,
                        operation_id,
                    )
        if self.is_mounted:
            self._start_sources_refresh()

    async def _attach_existing(
        self, ref: QualifiedWorkspaceRef, catalog_item_id: str
    ) -> None:
        """Run receipt-first attach against the durable captured owner."""

        await self.controller.attach_existing(
            ref,
            catalog_item_id=catalog_item_id,
            idempotency_key=f"research-existing-{uuid4().hex}",
        )
        if ref == self.controller.selected_ref and self.is_mounted:
            self._start_sources_refresh()
