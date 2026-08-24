"""Responsive, compose-once Research Workspace foundation screen."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Resize
from textual.widgets import Button, Static

from ...Research_Workspace import (
    ResearchPresentationOverlayStore,
    ResearchWorkspaceCatalogState,
    ResearchWorkspaceController,
)
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
    ResearchSourcesRegion,
    ResearchStudioRegion,
)
from ..Research_Workspace_Modules.pane_handle import ResearchSidePane


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
        **kwargs: Any,
    ) -> None:
        super().__init__(app_instance, "research_workspace", **kwargs)
        self.controller = controller or ResearchWorkspaceController({})
        self.overlay_store = overlay_store
        self.pane_preferences = ResearchPanePreferences()
        self.active_pane: ResearchPaneName = "chat"
        self._pane_layout: ResearchPaneLayout | None = None
        self._overlay_revision = 0
        self._overlay_ref = None

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
        self._overlay_ref = None
        self._overlay_revision = 0
        header = self.query_one("#research-workspace-header", ResearchHeaderRegion)
        header.sync_data_source(message.data_source)
        self.query_one("#research-workspace-selection", Static).update(
            f"Loading {message.data_source.value.title()} research workspaces..."
        )
        self._start_catalog_refresh()

    @on(ResearchPaneModeStrip.Selected)
    def select_pane_mode(self, message: ResearchPaneModeStrip.Selected) -> None:
        """Reveal the selected pane according to the current responsive band."""
        width = max(1, self.size.width)
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
        shell.set_class(self.size.height < 24, "height-compact")

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
        data_source = self.controller.selected_data_source
        state = await self.controller.refresh_workspace_catalog()
        if data_source is not self.controller.selected_data_source:
            return
        await self._apply_catalog_state(state)

    async def _apply_catalog_state(self, state: ResearchWorkspaceCatalogState) -> None:
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
            self._overlay_ref = None
            self._overlay_revision = 0
            return
        if not state.workspaces:
            selection.update(
                f"No {state.data_source.value.title()} research workspaces."
            )
            status.update(
                f"Foundation ready · {state.data_source.value.title()} catalog ready · "
                "0 workspaces · Foundation only"
            )
            self._overlay_ref = None
            self._overlay_revision = 0
            return

        workspace = state.workspaces[0]
        self.controller.select_workspace(workspace.ref)
        selection.update(
            f"{workspace.name} · {workspace.ref.data_source.value.title()}"
        )
        status.update(
            f"Foundation ready · {state.data_source.value.title()} catalog ready · "
            f"{len(state.workspaces)} workspace(s) · Foundation only"
        )
        self._overlay_ref = workspace.ref
        self._overlay_revision = 0
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
        self._overlay_revision = overlay.revision
        self.pane_preferences = overlay.preferences
        self._apply_pane_layout(max(1, self.size.width), relocate_hidden_focus=True)

    def _start_overlay_save(self) -> None:
        if self.overlay_store is None or self._overlay_ref is None:
            return
        self.run_worker(
            self._save_overlay_preferences(),
            group="research-workspace-overlay",
            exclusive=True,
        )

    async def _save_overlay_preferences(self) -> None:
        if self.overlay_store is None or self._overlay_ref is None:
            return
        ref = self._overlay_ref
        preferences = self.pane_preferences
        expected_revision = self._overlay_revision
        try:
            saved = await asyncio.to_thread(
                self.overlay_store.save,
                ref,
                preferences,
                expected_revision=expected_revision,
            )
        except (OSError, ValueError, RuntimeError):
            self.notify(
                "Device-only pane preference was not saved; retry the pane action.",
                severity="warning",
            )
            return
        if ref == self._overlay_ref and preferences == self.pane_preferences:
            self._overlay_revision = saved.revision
