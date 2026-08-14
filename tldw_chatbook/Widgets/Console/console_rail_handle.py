"""Console's rail handle: the shared base plus Console's own vocabulary."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_glyphs import GLYPH_COLLAPSE_LEFT, GLYPH_COLLAPSED
from tldw_chatbook.Chat.console_rail_state import (
    CONSOLE_RAIL_CONTEXT_LABEL,
    CONSOLE_RAIL_INSPECTOR_LABEL,
)
from tldw_chatbook.Widgets.destination_rail import DestinationRailHandle


class ConsoleRailHandle(DestinationRailHandle):
    """Rail handle carrying Console's tooltips and compact vocabulary."""

    VERTICAL_WIDTH = 3
    VERTICAL_CONTENT_WIDTH = 1

    def __init__(
        self,
        *,
        label: str,
        badge: str = "",
        button_id: str,
        badge_id: str,
        side: str,
        vertical: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a Console rail handle.

        Args:
            label: Rail name shown on the handle button.
            badge: Optional secondary line under the button.
            button_id: DOM id for the open button.
            badge_id: DOM id for the badge static.
            side: ``"left"`` for Context or ``"right"`` for Inspector.
            vertical: Whether to stack the compact handle text top-to-bottom.
            kwargs: Forwarded to ``DestinationRailHandle``.
        """
        super().__init__(
            label=label,
            badge=badge,
            button_id=button_id,
            badge_id=badge_id,
            side=side,
            open_tooltip=(
                "Open Context rail" if side == "left" else "Open Inspector rail"
            ),
            **kwargs,
        )
        self.vertical = vertical
        if side == "right":
            self.add_class("console-inspector-rail-handle")
        if self.vertical:
            self.add_class("console-rail-handle-vertical")
            self.styles.width = self.VERTICAL_WIDTH
            self.styles.min_width = self.VERTICAL_WIDTH
            self.styles.max_width = self.VERTICAL_WIDTH

    def compose(self) -> ComposeResult:
        """Render the Inspector fill or opt-in vertical child geometry.

        Returns:
            Child widgets with Console-specific handle geometry.
        """
        for child in super().compose():
            if self.vertical and isinstance(child, Button):
                child.add_class("console-rail-handle-button-vertical")
                child.styles.width = self.VERTICAL_CONTENT_WIDTH
                child.styles.max_width = self.VERTICAL_CONTENT_WIDTH
                child.styles.height = "1fr"
                child.styles.clear_rule("min_height")
                child.styles.clear_rule("max_height")
                child.styles.line_pad = 0
            elif self.vertical and isinstance(child, Static):
                child.add_class("console-rail-handle-badge-vertical")
                child.styles.width = self.VERTICAL_CONTENT_WIDTH
                child.styles.min_width = 0
                child.styles.max_width = self.VERTICAL_CONTENT_WIDTH
            elif self.side == "right" and isinstance(child, Button):
                child.styles.width = "100%"
                child.styles.max_width = "100%"
                child.styles.height = "1fr"
                child.styles.min_height = 0
                child.styles.max_height = "100%"
                child.styles.line_pad = 0
            yield child

    def _display_label(self) -> str:
        """Return compact visible text while preserving full tooltips."""
        if self.vertical:
            return self._stack_vertical_label(self.label)
        if self.side == "left":
            return (
                "Context->" if self.label == CONSOLE_RAIL_CONTEXT_LABEL else self.label
            )
        return "<-Inspect" if self.label == CONSOLE_RAIL_INSPECTOR_LABEL else self.label

    def _display_badge(self) -> str:
        """Return badge copy that fits the collapsed inspector affordance."""
        display_badge = self.badge
        if self.side == "right":
            if self.badge == "1 approval":
                display_badge = "1 appr"
            elif self.badge.endswith(" approvals"):
                count = self.badge.split(maxsplit=1)[0]
                display_badge = f"{count} appr"
            elif self.badge == "artifact":
                display_badge = "art"
        if self.vertical:
            return self._stack_vertical_text(display_badge)
        return display_badge

    @staticmethod
    def _stack_vertical_label(label: str) -> str:
        """Normalize known rail labels before rendering them one cell per row."""
        normalized_label = " ".join(label.split())
        if normalized_label == " ".join(CONSOLE_RAIL_CONTEXT_LABEL.split()):
            normalized_label = normalized_label.removesuffix(GLYPH_COLLAPSED).rstrip()
        elif normalized_label == " ".join(CONSOLE_RAIL_INSPECTOR_LABEL.split()):
            normalized_label = normalized_label.removeprefix(
                GLYPH_COLLAPSE_LEFT
            ).lstrip()
        return ConsoleRailHandle._stack_vertical_text(normalized_label)

    @staticmethod
    def _stack_vertical_text(text: str) -> str:
        """Normalize whitespace and render each remaining character on its own row."""
        return "\n".join(" ".join(text.split()))
