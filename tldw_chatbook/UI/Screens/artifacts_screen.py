"""Artifacts destination shell for generated outputs and Chatbooks."""

from textual.app import ComposeResult
from textual import on
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
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
                yield Button(
                    "Open Chatbooks",
                    id="artifacts-open-chatbooks",
                    tooltip="Open portable Chatbook bundles.",
                )
                yield Static(
                    "Generated outputs from local and server output services will appear here.",
                    id="artifacts-output-status",
                    classes="destination-purpose",
                )
                yield Button(
                    "Use in Console",
                    id="artifacts-use-in-console",
                    tooltip="Stage artifact context in Console.",
                )

    @on(Button.Pressed, "#artifacts-open-chatbooks")
    def open_chatbooks(self) -> None:
        self.post_message(NavigateToScreen("chatbooks"))

    @on(Button.Pressed, "#artifacts-use-in-console")
    def use_in_console(self) -> None:
        self.app_instance.open_chat_with_handoff(
            ChatHandoffPayload(
                source="artifacts",
                item_type="artifact-context",
                title="Artifact context",
                body="Stage generated outputs, reports, datasets, exports, or Chatbook context.",
            )
        )
