"""Studio foundation region."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static


class ResearchStudioRegion(VerticalScroll):
    """Honest output shell with no advertised generator before its owner exists."""

    can_focus = True

    def compose(self) -> ComposeResult:
        yield Static(
            "Studio", id="research-studio-heading", classes="research-pane-title"
        )
        yield Static(
            "0 outputs · Quick Notes unavailable", classes="research-pane-summary"
        )
        yield Static(
            "Outputs become available after a workspace has eligible ready sources.",
            classes="research-pane-empty",
        )
