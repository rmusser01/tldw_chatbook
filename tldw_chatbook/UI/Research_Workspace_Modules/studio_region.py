"""Studio foundation region."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from .quick_notes_section import ResearchQuickNotesSection


class ResearchStudioRegion(VerticalScroll):
    """Honest output shell with no advertised generator before its owner exists."""

    can_focus = True

    def compose(self) -> ComposeResult:
        yield Static(
            "Studio", id="research-studio-heading", classes="research-pane-title"
        )
        yield Static(
            "0 generated outputs · Quick Notes use the selected canonical owner",
            classes="research-pane-summary",
        )
        yield ResearchQuickNotesSection(id="research-quick-notes-section")
        yield Static(
            "Outputs become available after a workspace has eligible ready sources.",
            classes="research-pane-empty",
        )
