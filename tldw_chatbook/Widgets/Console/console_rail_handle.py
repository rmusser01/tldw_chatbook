"""Compact Console rail handle widgets."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static


class ConsoleRailHandle(Vertical):
    """Focusable compact handle for opening a collapsed Console rail."""

    def __init__(
        self,
        *,
        label: str,
        badge: str = "",
        button_id: str,
        badge_id: str,
        side: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.badge = badge
        self.button_id = button_id
        self.badge_id = badge_id
        self.side = side
        self.add_class("console-rail-handle")
        self.add_class(f"console-rail-handle-{side}")

    def compose(self) -> ComposeResult:
        button = Button(self.label, id=self.button_id, compact=True)
        button.add_class("console-rail-handle-button")
        button.add_class(f"console-rail-handle-button-{self.side}")
        button.tooltip = (
            "Open Context rail"
            if self.side == "left"
            else "Open Inspector rail"
        )
        yield button
        if self.badge:
            badge = Static(self.badge, id=self.badge_id, markup=False)
            badge.add_class("console-rail-handle-badge")
            yield badge

    def sync_state(self, label: str, badge: str) -> None:
        """Refresh this handle's label and badge without recomposing the screen."""
        if self.label == label and self.badge == badge:
            return
        self.label = label
        self.badge = badge
        self.refresh(recompose=True)
