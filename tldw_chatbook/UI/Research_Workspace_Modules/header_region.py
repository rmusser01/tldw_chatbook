"""Pinned identity and authority summary for Research Workspace."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Workbench.workbench_widgets import DestinationHeader
from .mode_bar import ResearchModeStrip
from .workspace_menu import ResearchWorkspaceMenu


class ResearchHeaderRegion(Vertical):
    """Stable foundation header; service-backed state arrives in Task 5."""

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
            yield Static("Workspace data: Local", id="research-data-source")
            yield Static("Processing: not configured", id="research-processing-route")
            yield Static("Sources: 0 ready", id="research-readiness-summary")
        yield ResearchWorkspaceMenu(id="research-workspace-menu")
