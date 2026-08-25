"""Pinned identity and authority summary for Research Workspace."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

from ...Research_Workspace import (
    ResearchWorkspaceCatalogState,
    WorkspaceDataSource,
)
from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Workbench.workbench_widgets import DestinationHeader
from .mode_bar import ResearchModeStrip
from .workspace_menu import ResearchWorkspaceMenu


class ResearchHeaderRegion(Vertical):
    """Pinned foundation header with explicit data-authority selection."""

    class DataSourceSelected(Message):
        """Request one explicit Research catalog authority."""

        def __init__(self, data_source: WorkspaceDataSource) -> None:
            super().__init__()
            self.data_source = data_source

    def compose(self) -> ComposeResult:
        yield DestinationHeader(
            WorkbenchHeaderState(
                title="Research Workspace",
                subtitle="Bounded sources, grounded answers, durable outputs.",
                status="empty",
                status_label="Setup required",
            ),
            id="research-destination-header",
        )
        yield ResearchModeStrip(
            active_route="research_workspace", id="research-mode-strip"
        )
        with Horizontal(id="research-authority-summary"):
            yield Static("Workspace data:", id="research-data-source-label")
            yield Button(
                "Local",
                id="research-data-source-local",
                classes="research-data-source-button is-active",
                name="Use Local workspace data",
                tooltip="Use this device's Research workspace catalog.",
                compact=True,
            )
            yield Static("|", id="research-data-source-separator")
            yield Button(
                "Server",
                id="research-data-source-server",
                classes="research-data-source-button",
                name="Use Server workspace data",
                tooltip="Use the selected server's Research workspace catalog.",
                compact=True,
            )
        with Horizontal(id="research-processing-summary"):
            yield Static("Processing: not configured", id="research-processing-route")
            yield Static("Sources: 0 ready", id="research-readiness-summary")
        yield Static(
            "",
            id="research-authority-recovery",
            classes="research-recovery-callout",
            markup=False,
        )
        yield ResearchWorkspaceMenu(id="research-workspace-menu")

    def sync_data_source(self, data_source: WorkspaceDataSource) -> None:
        """Paint the explicit selected authority without changing it implicitly."""

        selected = WorkspaceDataSource(data_source)
        for candidate in WorkspaceDataSource:
            self.query_one(
                f"#research-data-source-{candidate.value}", Button
            ).set_class(candidate is selected, "is-active")

    def sync_catalog_state(self, state: ResearchWorkspaceCatalogState) -> None:
        """Show selected-authority recovery while leaving its selector active."""

        self.sync_data_source(state.data_source)
        recovery = self.query_one("#research-authority-recovery", Static)
        if state.recovery is None:
            recovery.update("")
            recovery.display = False
            return
        recovery.update(
            f"{state.recovery.user_message} {state.recovery.recovery_action}".strip()
        )
        recovery.display = True

    @on(Button.Pressed, ".research-data-source-button")
    def select_data_source(self, event: Button.Pressed) -> None:
        """Post a selection request; the screen owns catalog orchestration."""

        event.stop()
        value = str(event.button.id or "").removeprefix("research-data-source-")
        if value in {item.value for item in WorkspaceDataSource}:
            self.post_message(self.DataSourceSelected(WorkspaceDataSource(value)))
