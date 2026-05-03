"""Artifacts destination shell for generated outputs and Chatbooks."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


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
                yield Static("Chatbooks | Reports | Datasets | Exports")
