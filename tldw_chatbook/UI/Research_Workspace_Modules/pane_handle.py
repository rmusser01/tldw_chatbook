"""Accessible fixed-width collapse and reveal controls."""

from __future__ import annotations

from typing import Any, Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button


ResearchSidePane = Literal["sources", "studio"]

_HANDLE_COPY = {
    "sources": {
        "collapse": ("<---", "Collapse Sources pane"),
        "reveal": ("--->", "Expand Sources pane"),
    },
    "studio": {
        "collapse": ("--->", "Collapse Studio pane"),
        "reveal": ("<---", "Expand Studio pane"),
    },
}


class ResearchPaneHandle(Vertical):
    """Mount both handle states once and expose one at a time."""

    class Toggled(Message):
        """Request collapse or reveal for one side pane."""

        def __init__(self, pane: ResearchSidePane, reveal: bool) -> None:
            super().__init__()
            self.pane = pane
            self.reveal = reveal

    def __init__(self, pane: ResearchSidePane, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pane = pane

    def compose(self) -> ComposeResult:
        for action in ("collapse", "reveal"):
            label, accessible_name = _HANDLE_COPY[self.pane][action]
            yield Button(
                label,
                id=f"research-{self.pane}-{action}",
                classes=f"research-pane-handle-button research-pane-{action}",
                name=accessible_name,
                tooltip=accessible_name,
                compact=True,
            )

    def on_mount(self) -> None:
        self.sync_expanded(True)

    def sync_expanded(self, expanded: bool, *, handle_visible: bool = True) -> None:
        """Swap visible control state without rebuilding the handle."""
        self.query_one(f"#research-{self.pane}-collapse", Button).display = (
            handle_visible and expanded
        )
        self.query_one(f"#research-{self.pane}-reveal", Button).display = (
            handle_visible and not expanded
        )

    @on(Button.Pressed, ".research-pane-handle-button")
    def toggle_pane(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(
            self.Toggled(self.pane, event.button.id == f"research-{self.pane}-reveal")
        )
