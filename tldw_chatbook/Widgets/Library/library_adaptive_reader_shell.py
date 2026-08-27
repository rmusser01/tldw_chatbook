"""Retained three-role structure for Library adaptive readers."""

from __future__ import annotations

from typing import Any, Literal

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.events import DescendantFocus
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button

from tldw_chatbook.Utils.adaptive_reader_state import (
    PANE_GRIP_WIDTH,
    AdaptiveReaderEffectiveLayout,
    PaneName,
)


class PaneToggleRequested(Message):
    """Request a manual toggle of one optional pane."""

    def __init__(self, pane: Literal["library", "items"]) -> None:
        super().__init__()
        self.pane = pane


class AdaptiveReaderShellResized(Message):
    """Report that the settled shell allocation may need resolving."""


class LibraryAdaptiveReaderPaneGrip(Button):
    """Five-column keyboard and pointer control for one optional pane."""

    BINDINGS = [Binding("enter,space", "press", "Press button", show=False)]

    def __init__(
        self,
        pane: PaneName,
        *,
        open: bool,
        pane_label: str,
        extra_classes: str = "",
        **kwargs: Any,
    ) -> None:
        self.pane = pane
        self.pane_label = pane_label
        classes = "library-adaptive-reader-pane-grip"
        if extra_classes:
            classes = f"{classes} {extra_classes}"
        super().__init__(compact=True, flat=True, classes=classes, **kwargs)
        self.styles.width = PANE_GRIP_WIDTH
        self.styles.min_width = PANE_GRIP_WIDTH
        self.styles.max_width = PANE_GRIP_WIDTH
        self.styles.height = "100%"
        self.styles.padding = 0
        self.styles.line_pad = 0
        self.styles.border = ("none", "transparent")
        self.styles.content_align = ("center", "middle")
        self.sync_open(open)

    def sync_open(self, open: bool) -> None:
        """Patch arrow and action copy without changing geometry."""
        action = "Collapse" if open else "Expand"
        copy = f"{action} {self.pane_label} pane"
        self.label = "<---" if open else "--->"
        self._name = copy
        self.tooltip = copy

    @on(Button.Pressed)
    def request_toggle(self, event: Button.Pressed) -> None:
        """Translate native Button activation into the shell message."""
        if event.button is not self:
            return
        event.stop()
        self.post_message(PaneToggleRequested(self.pane))


class LibraryAdaptiveReaderShell(Horizontal):
    """Own adaptive reader structure while callers own state and behavior."""

    def __init__(
        self,
        library: Widget,
        items: Widget,
        work: Widget,
        layout: AdaptiveReaderEffectiveLayout,
        *,
        id_prefix: str,
        library_label: str,
        items_label: str,
        grip_classes: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_class("library-adaptive-reader-shell")
        self.library = library
        self.items = items
        self.work = work
        self.library.add_class("library-adaptive-reader-library")
        self.items.add_class("library-adaptive-reader-items")
        self.work.add_class("library-adaptive-reader-work")
        self.library_grip = LibraryAdaptiveReaderPaneGrip(
            "library",
            open=layout.library_open,
            pane_label=library_label,
            extra_classes=grip_classes,
            id=f"{id_prefix}-library-grip",
        )
        self.items_grip = LibraryAdaptiveReaderPaneGrip(
            "items",
            open=layout.items_open,
            pane_label=items_label,
            extra_classes=grip_classes,
            id=f"{id_prefix}-items-grip",
        )
        self._last_focused_descendant: dict[PaneName, Widget | None] = {
            "library": None,
            "items": None,
        }
        self.effective_layout = layout

    def compose(self) -> ComposeResult:
        """Compose retained Library, Items, grips, and Work widgets."""
        yield self.library
        yield self.library_grip
        yield self.items
        yield self.items_grip
        yield self.work

    def on_mount(self) -> None:
        """Apply initial geometry and request a settled resize projection."""
        self.sync_layout(self.effective_layout)
        self.call_after_refresh(self.post_message, AdaptiveReaderShellResized())

    def on_resize(self) -> None:
        """Request layout resolution after the shell allocation changes."""
        self.post_message(AdaptiveReaderShellResized())

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Remember optional-pane focus before a grip activation moves it."""
        target = event.widget
        for pane_name, pane in (
            ("library", self.library),
            ("items", self.items),
        ):
            if self._is_valid_focus_target(pane, target):
                self._last_focused_descendant[pane_name] = target
                return

    def _pane_focus_chain(self, pane: Widget) -> list[Widget]:
        """Return currently reachable pane targets in Textual focus order."""
        if not self.is_mounted:
            return []
        return [
            target
            for target in self.app.screen.focus_chain
            if target is pane or pane in target.ancestors
        ]

    def _is_valid_focus_target(self, pane: Widget, target: Widget | None) -> bool:
        """Return whether ``target`` is currently reachable within ``pane``."""
        return target is not None and target in self._pane_focus_chain(pane)

    def sync_layout(
        self,
        layout: AdaptiveReaderEffectiveLayout,
        *,
        manual_reopen: PaneName | None = None,
    ) -> None:
        """Patch pane display and exact cell widths in place."""
        previous_layout = self.effective_layout
        self.effective_layout = layout
        focused = self.app.focused if self.is_mounted else None
        evacuation_target: Widget | None = None
        manual_reopen_pane: Widget | None = None
        for pane_name, pane, grip, was_open, open, width in (
            (
                "library",
                self.library,
                self.library_grip,
                previous_layout.library_open,
                layout.library_open,
                layout.library_width,
            ),
            (
                "items",
                self.items,
                self.items_grip,
                previous_layout.items_open,
                layout.items_open,
                layout.items_width,
            ),
        ):
            if (
                not open
                and focused is not None
                and (focused is pane or pane in focused.ancestors)
            ):
                if focused is not pane and self._is_valid_focus_target(pane, focused):
                    self._last_focused_descendant[pane_name] = focused
                evacuation_target = grip
            pane.display = open
            pane.disabled = not open
            pane.styles.width = width
            pane.styles.min_width = width
            pane.styles.max_width = width
            pane.styles.height = "100%"
            grip.sync_open(open)
            if not was_open and open and manual_reopen == pane_name:
                manual_reopen_pane = pane
        self.work.display = True
        self.work.styles.width = "1fr"
        self.work.styles.min_width = 0
        self.work.styles.height = "100%"
        if evacuation_target is not None:
            evacuation_target.focus(scroll_visible=False)
        elif manual_reopen_pane is not None:
            focus_chain = self._pane_focus_chain(manual_reopen_pane)
            target = self._last_focused_descendant[manual_reopen]
            if target not in focus_chain:
                target = next(iter(focus_chain), None)
            if target is not None:
                target.focus(scroll_visible=False)
