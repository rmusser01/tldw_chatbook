"""Sources foundation region."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static


class ResearchSourcesRegion(VerticalScroll):
    """Visible source authority and honest empty state."""

    can_focus = True

    def compose(self) -> ComposeResult:
        yield Static(
            "Sources", id="research-sources-heading", classes="research-pane-title"
        )
        yield Static("0 attached · 0 ready", classes="research-pane-summary")
        yield Static(
            "No sources are attached. Manage workspaces in Settings; source intake arrives in the Sources phase.",
            classes="research-pane-empty",
        )
