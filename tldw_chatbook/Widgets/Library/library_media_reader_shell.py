"""Media-compatible adapter for the shared adaptive reader shell."""

from __future__ import annotations

from typing import Any

from textual.widget import Widget

from tldw_chatbook.Library.library_media_reader_state import (
    MEDIA_READER_LAYOUT_PROFILE,
    MediaReaderEffectiveLayout,
    PaneName,
)

from .library_adaptive_reader_shell import (
    AdaptiveReaderShellResized,
    LibraryAdaptiveReaderPaneGrip,
    LibraryAdaptiveReaderShell,
    PaneToggleRequested as SharedPaneToggleRequested,
)

MediaShellResized = AdaptiveReaderShellResized
PaneToggleRequested = SharedPaneToggleRequested


class LibraryMediaPaneGrip(LibraryAdaptiveReaderPaneGrip):
    """Preserve the public Media grip constructor and visual class."""

    def __init__(self, pane: PaneName, *, open: bool, **kwargs: Any) -> None:
        super().__init__(
            pane,
            open=open,
            pane_label=pane.title(),
            extra_classes="library-media-pane-grip",
            width=MEDIA_READER_LAYOUT_PROFILE.grip_width,
            **kwargs,
        )


class LibraryMediaReaderShell(LibraryAdaptiveReaderShell):
    """Preserve the Media shell API and selector contract."""

    def __init__(
        self,
        library: Widget,
        items: Widget,
        reader: Widget,
        layout: MediaReaderEffectiveLayout,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            library=library,
            items=items,
            work=reader,
            layout=layout,
            id_prefix="library-media",
            library_label="Library",
            items_label="Items",
            grip_classes="library-media-pane-grip",
            grip_width=MEDIA_READER_LAYOUT_PROFILE.grip_width,
            **kwargs,
        )
        self.reader = reader

    def on_mount(self) -> None:
        """Hide the redundant Media rail control after shared shell setup."""
        super().on_mount()
        collapse = self.query("#library-rail-collapse")
        if collapse:
            collapse.first().display = False
