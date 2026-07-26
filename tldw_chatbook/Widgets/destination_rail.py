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
            open_tooltip: Fixed button tooltip. When omitted, the tooltip is
                derived from ``label`` and tracks it across ``sync_state``.
            kwargs: Forwarded to ``Vertical``.
        """
        super().__init__(**kwargs)
        self.label = label
        self.badge = badge
        self.button_id = button_id
        self.badge_id = badge_id
        self.side = side
        #: Caller-supplied override, or None to derive from the live label.
        self._explicit_open_tooltip = open_tooltip
        self.add_class("console-rail-handle")
        self.add_class(f"console-rail-handle-{side}")

    @property
    def open_tooltip(self) -> str:
        """Tooltip for the open button.

        Resolved on read rather than captured in ``__init__``: ``sync_state``
        can rename the rail, and a tooltip derived from the label would
        otherwise keep naming the previous one after the recompose.

        Returns:
            The caller's fixed tooltip when one was supplied, else
            ``"Open <label> rail"`` for the current label.
        """
        if self._explicit_open_tooltip is not None:
            return self._explicit_open_tooltip
        return f"Open {self.label} rail"

    def compose(self) -> ComposeResult:
        """Render the open button and, when set, the badge.

        Returns:
            A ``ComposeResult`` yielding the open button, followed by the
            badge ``Static`` only when ``badge`` is non-empty.
        """
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
        """Refresh label and badge without recomposing the whole screen.

        Args:
            label: New rail name for the handle button.
            badge: New badge text; empty string removes the badge.
        """
        if self.label == label and self.badge == badge:
            return
        self.label = label
        self.badge = badge
        self.call_later(self.recompose)

    def _display_label(self) -> str:
        """Return the visible button text.

        Subclass extension point: override to abbreviate a long rail name
        for the eleven-column handle while leaving the tooltip full.

        Returns:
            The label as rendered on the button.
        """
        return self.label

    def _display_badge(self) -> str:
        """Return the visible badge text.

        Subclass extension point: override to abbreviate badge copy that
        would not fit the collapsed handle.

        Returns:
            The badge as rendered under the button.
        """
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
        """Create a rail section header.

        Args:
            title: User-facing section title, also used in the toggle tooltip.
            section_id: Stable id fragment for this section's child widget ids.
            open: Whether the associated section body starts visible; drives
                which chevron the toggle shows.
            kwargs: Forwarded to ``Horizontal``.
        """
        super().__init__(classes="console-rail-section-header", **kwargs)
        self.title = title
        self.section_id = section_id
        self.open = open

    def compose(self) -> ComposeResult:
        """Render the section title and its collapse/expand toggle.

        Returns:
            A ``ComposeResult`` yielding the title ``Static`` then the
            toggle ``Button``.
        """
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
        """Refresh the toggle affordance after body visibility changes.

        Args:
            open: Whether the section body is now visible.
        """
        self.open = open
        toggle = self.query_one(
            f"#{RAIL_SECTION_TOGGLE_PREFIX}{self.section_id}",
            Button,
        )
        toggle.label = self._toggle_label()
        toggle.tooltip = self._toggle_tooltip()

    def _toggle_label(self) -> str:
        """Return the chevron matching the current open state.

        Returns:
            The expanded glyph when open, the collapsed glyph otherwise.
        """
        return GLYPH_EXPANDED if self.open else GLYPH_COLLAPSED

    def _toggle_tooltip(self) -> str:
        """Return the toggle tooltip describing the action it performs.

        Returns:
            ``"Collapse <title>"`` when open, ``"Expand <title>"`` otherwise.
        """
        return f"Collapse {self.title}" if self.open else f"Expand {self.title}"
