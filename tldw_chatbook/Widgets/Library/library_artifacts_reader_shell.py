"""Concrete Artifacts consumer of Library's retained adaptive reader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.events import DescendantFocus, Key
from textual.widgets import Input, Select

from ...Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
from ..workbench_focus import WorkbenchPaneTarget
from .library_adaptive_reader_shell import (
    AdaptiveReaderShellResized,
    LibraryAdaptiveReaderShell,
    PaneToggleRequested,
)
from .library_artifacts_widgets import ArtifactItems, ArtifactWork

if TYPE_CHECKING:
    from ...UI.Library_Modules.library_artifacts_controller import (
        LibraryArtifactsController,
    )
    from .library_rail import LibraryRail


class LibraryArtifactsReaderShell(LibraryAdaptiveReaderShell):
    def __init__(
        self, library: LibraryRail, controller: LibraryArtifactsController
    ) -> None:
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

    def workbench_focus_targets(self) -> tuple[WorkbenchPaneTarget, ...]:
        """Cycle Library, Items and reader, using grips for collapsed panes."""
        targets = (
            WorkbenchPaneTarget(
                "library-rail",
                (
                    "library-search-input",
                    f"library-row-{LIBRARY_ROW_INGEST_MEDIA}",
                    "library-rail-explore-all",
                    "library-rail-collapse",
                ),
            )
            if self.library.display
            else WorkbenchPaneTarget(
                "library-artifacts-library-grip", ("library-artifacts-library-grip",)
            ),
            WorkbenchPaneTarget("library-canvas", ("library-artifacts-list",))
            if self.items.display
            else WorkbenchPaneTarget(
                "library-artifacts-items-grip", ("library-artifacts-items-grip",)
            ),
        )
        if self.controller.selected is not None or self.controller.layout.reader_width:
            targets += (
                WorkbenchPaneTarget(
                    "library-artifacts-work", ("library-artifacts-body",)
                ),
            )
        return targets

    @on(DescendantFocus)
    def reveal_focused_reader(self, event: DescendantFocus) -> None:
        if (
            event.widget.id == "library-artifacts-body"
            and self.app.focused is event.widget
            and (
                not self.controller.reader_open
                or self.controller.layout.priority_pane is not None
            )
        ):
            self.controller.focus_reader(request_focus=False)

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
