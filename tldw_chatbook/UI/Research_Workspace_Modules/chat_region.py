"""Grounded Chat foundation region."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static


class ResearchChatRegion(VerticalScroll):
    """Dominant evidence-focused chat region without an inert composer."""

    can_focus = True

    def compose(self) -> ComposeResult:
        yield Static(
            "Grounded Chat", id="research-chat-heading", classes="research-pane-title"
        )
        yield Static(
            "Selected sources: 0 · Retrieval unavailable",
            classes="research-pane-summary",
        )
        yield Static(
            "Choose a workspace and attach ready sources before asking a grounded question.",
            classes="research-pane-empty",
        )
