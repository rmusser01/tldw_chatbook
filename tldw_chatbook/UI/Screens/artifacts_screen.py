"""Artifacts destination shell for generated outputs and Chatbooks."""

from textual.app import ComposeResult
from textual import on
from textual.containers import Vertical
from textual.widgets import Button, Static

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen


class ArtifactsScreen(BaseAppScreen):
    """Generated outputs, portable bundles, reports, datasets, and Chatbooks."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "artifacts", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="artifacts-shell"):
            yield Static("Artifacts", id="artifacts-title", classes="ds-destination-header")
            yield Static(
                "Generated outputs, bundles, reports, datasets, and Chatbooks.",
                id="artifacts-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="artifacts-sections", classes="ds-panel"):
                yield Button("Open Chatbooks", id="artifacts-open-chatbooks")
                yield Static(
                    "Generated outputs from local and server output services will appear here.",
                    id="artifacts-output-status",
                    classes="destination-purpose",
                )

    @on(Button.Pressed, "#artifacts-open-chatbooks")
    def open_chatbooks(self) -> None:
        self.post_message(NavigateToScreen("chatbooks"))
