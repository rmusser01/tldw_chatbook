"""Collapsible Console left-rail section header chrome."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Static

CONSOLE_RAIL_SECTION_TOGGLE_PREFIX = "console-rail-section-toggle-"


class ConsoleRailSectionHeader(Horizontal):
    """One-line rail section header with a collapse/expand toggle.

    Attributes:
        title: User-facing section title.
        section_id: Stable section id used in child widget ids.
        open: Whether the associated section body is currently visible.
    """

    def __init__(
        self,
        title: str,
        *,
        section_id: str,
        open: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(classes="console-rail-section-header", **kwargs)
        self.title = title
        self.section_id = section_id
        self.open = open
        self.styles.height = 1
        self.styles.min_height = 1
        self.styles.max_height = 1

    def compose(self) -> ComposeResult:
        title = Static(
            self.title,
            id=f"console-rail-section-title-{self.section_id}",
            classes="console-rail-section-title",
            markup=False,
        )
        title.styles.width = "1fr"
        yield title
        toggle = Button(
            self._toggle_label(),
            id=f"{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}{self.section_id}",
            classes="console-workspace-action console-rail-section-toggle",
            compact=True,
        )
        toggle.tooltip = self._toggle_tooltip()
        toggle.styles.width = 3
        toggle.styles.min_width = 3
        toggle.styles.max_width = 3
        yield toggle

    def sync_open(self, open: bool) -> None:
        """Refresh the toggle affordance after the section body visibility changes."""
        self.open = open
        toggle = self.query_one(
            f"#{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}{self.section_id}",
            Button,
        )
        toggle.label = self._toggle_label()
        toggle.tooltip = self._toggle_tooltip()

    def _toggle_label(self) -> str:
        return "-" if self.open else "+"

    def _toggle_tooltip(self) -> str:
        return f"Collapse {self.title}" if self.open else f"Expand {self.title}"
