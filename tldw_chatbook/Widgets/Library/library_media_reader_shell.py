"""Permanent three-role shell for Library Media."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button

from tldw_chatbook.Library.library_media_reader_state import (
    PANE_GRIP_WIDTH,
    MediaReaderEffectiveLayout,
    PaneName,
)


class PaneToggleRequested(Message):
    """Request a manual toggle of one optional Media pane."""

    def __init__(self, pane: PaneName) -> None:
        super().__init__()
        self.pane = pane


class MediaShellResized(Message):
    """Report that the settled shell allocation may need resolving."""


class LibraryMediaPaneGrip(Button):
    """Five-column keyboard and pointer grip for one Media pane."""

    BINDINGS = [Binding("enter,space", "press", "Press button", show=False)]

    def __init__(self, pane: PaneName, *, open: bool, **kwargs: Any) -> None:
        self.pane = pane
        super().__init__(
            compact=True,
            flat=True,
            classes="library-media-pane-grip",
            **kwargs,
        )
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
        copy = f"{action} {self.pane.title()} pane"
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


class LibraryMediaReaderShell(Horizontal):
    """Own Media geometry while leaving state and services to LibraryScreen."""

    def __init__(
        self,
        library: Widget,
        items: Widget,
        reader: Widget,
        layout: MediaReaderEffectiveLayout,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.library = library
        self.items = items
        self.reader = reader
        self.library_grip = LibraryMediaPaneGrip(
            "library", open=layout.library_open, id="library-media-library-grip"
        )
        self.items_grip = LibraryMediaPaneGrip(
            "items", open=layout.items_open, id="library-media-items-grip"
        )
        self.effective_layout = layout

    def compose(self) -> ComposeResult:
        yield self.library
        yield self.library_grip
        yield self.items
        yield self.items_grip
        yield self.reader

    def on_mount(self) -> None:
        self.sync_layout(self.effective_layout)
        collapse = self.query("#library-rail-collapse")
        if collapse:
            collapse.first().display = False
        self.call_after_refresh(self.post_message, MediaShellResized())

    def on_resize(self) -> None:
        self.post_message(MediaShellResized())

    def sync_layout(self, layout: MediaReaderEffectiveLayout) -> None:
        """Patch pane display and exact cell widths in place."""
        self.effective_layout = layout
        for pane, open, width in (
            (self.library, layout.library_open, layout.library_width),
            (self.items, layout.items_open, layout.items_width),
        ):
            pane.display = open
            pane.styles.width = width
            pane.styles.min_width = width
            pane.styles.max_width = width
            pane.styles.height = "100%"
        self.library_grip.sync_open(layout.library_open)
        self.items_grip.sync_open(layout.items_open)
        self.reader.display = True
        self.reader.styles.width = "1fr"
        self.reader.styles.min_width = 0
        self.reader.styles.height = "100%"
