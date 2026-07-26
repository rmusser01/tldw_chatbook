"""Shared, destination-agnostic rail chrome.

Extracted from ``Widgets/Console/console_rail_handle.py`` and
``console_rail_section.py``, which had six consumers across Console, Home,
Library, and Personas while living in a Console-private namespace and
importing from the Chat layer. This module carries no Chat import and no
Console vocabulary; ``ConsoleRailHandle`` subclasses it and keeps its own.

The ``.console-rail-*`` CSS class names are retained deliberately so the
generated bundle sees no diff. The TCSS references these widgets only by
class, never by type, so the new type names are invisible to CSS. Renaming
the classes is a deferred cleanup.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static


#: Toggle-button id prefix. Unchanged from the Console original so existing
#: selectors and tests keep resolving.
RAIL_SECTION_TOGGLE_PREFIX = "console-rail-section-toggle-"

#: Default collapse/expand affordance glyphs. Literals rather than an import
#: from ``Chat.console_glyphs`` so this module stays free of the Chat layer;
#: the values match that module exactly.
GLYPH_EXPANDED = "▾"
GLYPH_COLLAPSED = "▸"


class DestinationRailHandle(Vertical):
    """Focusable compact handle for opening a collapsed destination rail."""

    def __init__(
        self,
        *,
        label: str,
        badge: str = "",
        button_id: str,
        badge_id: str,
        side: str,
        open_tooltip: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a rail handle.

        Args:
            label: Rail name shown on the handle button.
            badge: Optional secondary line under the button.
            button_id: DOM id for the open button.
            badge_id: DOM id for the badge static.
            side: ``"left"`` or ``"right"``; drives height and CSS class.
            open_tooltip: Button tooltip. Defaults to ``"Open <label> rail"``.
            kwargs: Forwarded to ``Vertical``.
        """
        super().__init__(**kwargs)
        self.label = label
        self.badge = badge
        self.button_id = button_id
        self.badge_id = badge_id
        self.side = side
        self.open_tooltip = open_tooltip or f"Open {label} rail"
        self.add_class("console-rail-handle")
        self.add_class(f"console-rail-handle-{side}")

    def compose(self) -> ComposeResult:
        """Render the open button and, when set, the badge."""
        button_width = 11
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
        button.tooltip = self.open_tooltip
        yield button
        if self.badge:
            badge = Static(self._display_badge(), id=self.badge_id, markup=False)
            badge.add_class("console-rail-handle-badge")
            badge.tooltip = self.badge
            yield badge

    def sync_state(self, label: str, badge: str) -> None:
        """Refresh label and badge without recomposing the whole screen."""
        if self.label == label and self.badge == badge:
            return
        self.label = label
        self.badge = badge
        self.call_later(self.recompose)

    def _display_label(self) -> str:
        """Visible button text. Override to abbreviate."""
        return self.label

    def _display_badge(self) -> str:
        """Visible badge text. Override to abbreviate."""
        return self.badge


class DestinationRailSectionHeader(Horizontal):
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

    def compose(self) -> ComposeResult:
        """Render the section title and its collapse/expand toggle."""
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
            id=f"{RAIL_SECTION_TOGGLE_PREFIX}{self.section_id}",
            classes="console-workspace-action console-rail-section-toggle",
            compact=True,
        )
        toggle.tooltip = self._toggle_tooltip()
        toggle.styles.width = 3
        toggle.styles.min_width = 3
        toggle.styles.max_width = 3
        yield toggle

    def sync_open(self, open: bool) -> None:
        """Refresh the toggle affordance after body visibility changes."""
        self.open = open
        toggle = self.query_one(
            f"#{RAIL_SECTION_TOGGLE_PREFIX}{self.section_id}",
            Button,
        )
        toggle.label = self._toggle_label()
        toggle.tooltip = self._toggle_tooltip()

    def _toggle_label(self) -> str:
        return GLYPH_EXPANDED if self.open else GLYPH_COLLAPSED

    def _toggle_tooltip(self) -> str:
        return f"Collapse {self.title}" if self.open else f"Expand {self.title}"
