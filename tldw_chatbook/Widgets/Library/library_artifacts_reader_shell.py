"""Concrete Artifacts consumer of Library's retained adaptive reader."""

from __future__ import annotations

from textual import on
from textual.events import Key
from textual.widgets import Input, Select

from .library_adaptive_reader_shell import (
    AdaptiveReaderShellResized,
    LibraryAdaptiveReaderShell,
    PaneToggleRequested,
)
from .library_artifacts_widgets import ArtifactItems, ArtifactWork


class LibraryArtifactsReaderShell(LibraryAdaptiveReaderShell):
    def __init__(self, library, controller):
        self.controller = controller
        super().__init__(
            library,
            ArtifactItems(controller, id="library-canvas"),
            ArtifactWork(controller, id="library-artifacts-work"),
            controller.layout,
            id_prefix="library-artifacts",
            library_label="Library",
            items_label="Items",
            id="library-artifacts-reader-shell",
        )

    def on_mount(self) -> None:
        self.call_after_refresh(self.controller.attach, self)

    @on(AdaptiveReaderShellResized)
    def resized(self, event: AdaptiveReaderShellResized) -> None:
        event.stop()
        self.controller.resize()

    @on(PaneToggleRequested)
    def toggle(self, event: PaneToggleRequested) -> None:
        event.stop()
        self.controller.toggle_pane(event.pane)

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            # Editable fields and expanded selectors get first refusal.
            focused = self.app.focused
            if isinstance(focused, Select) and focused.expanded:
                return
            if isinstance(focused, Input) and focused.value:
                return
            event.stop()
            event.prevent_default()
            self.controller.focus_items()
        elif event.key == "/" and not isinstance(self.app.focused, Input):
            event.stop()
            event.prevent_default()
            self.controller.focus_items(search=True)

    def sync(self) -> None:
        self.items.sync()
        self.work.sync()
