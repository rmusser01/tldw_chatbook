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
        button_width = 9 if self.side == "right" else 11
        button_height: int | str = 3 if self.side == "right" else "100%"
        button = Button(self._display_label(), id=self.button_id, compact=True)
        button.add_class("console-rail-handle-button")
        button.add_class(f"console-rail-handle-button-{self.side}")
        button.styles.width = button_width
        button.styles.min_width = 0
        button.styles.max_width = button_width
        button.styles.height = button_height
        button.styles.min_height = button_height
        button.styles.max_height = button_height
        button.tooltip = (
            "Open Context rail"
            if self.side == "left"
            else "Open Inspector rail"
        )
        yield button
        if self.badge:
            badge = Static(self._display_badge(), id=self.badge_id, markup=False)
            badge.add_class("console-rail-handle-badge")
            badge.tooltip = self.badge
            yield badge

    def sync_state(self, label: str, badge: str) -> None:
        """Refresh this handle's label and badge without recomposing the screen."""
        if self.label == label and self.badge == badge:
            return
        self.label = label
        self.badge = badge
        self.call_later(self.recompose)

    def _display_label(self) -> str:
        """Return a compact visible label while preserving full tooltips."""
        if self.side != "right":
            return self.label
        return "Inspect" if self.label == "< Inspector" else self.label

    def _display_badge(self) -> str:
        """Return badge copy that fits the collapsed inspector affordance."""
        if self.side != "right":
            return self.badge
        if self.badge == "1 approval":
            return "1 appr"
        if self.badge.endswith(" approvals"):
            count = self.badge.split(maxsplit=1)[0]
            return f"{count} appr"
        if self.badge == "artifact":
            return "art"
        return self.badge
