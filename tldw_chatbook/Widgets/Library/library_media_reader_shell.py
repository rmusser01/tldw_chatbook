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
        """Build a Media grip at the profile's width.

        Args:
            pane: Which optional pane this grip toggles.
            open: Whether that pane is open right now.
            **kwargs: Forwarded to ``LibraryAdaptiveReaderPaneGrip``; the
                Media label, class and ``grip_width`` are fixed here.
        """
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
        """Adapt Media's pane vocabulary onto the shared shell.

        Media calls its work pane the ``reader``, and the attribute is part
        of this class's public surface (callers reach for ``shell.reader``),
        so the shared ``work`` name is bridged here rather than renamed.

        Args:
            library: Widget for the Library rail pane.
            items: Widget for the media list pane.
            reader: Widget for the Reader pane -- the shared shell's ``work``.
            layout: The resolved Media layout to mount with.
            **kwargs: Forwarded to ``LibraryAdaptiveReaderShell`` (``id``,
                ``classes``, ...); the Media identity arguments (id prefix,
                pane labels, grip classes and the profile's one-cell
                ``grip_width``) are fixed here.
        """
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
        """Hide the redundant Media rail control after shared shell setup.

        No super().on_mount(): the dispatcher already invokes
        LibraryAdaptiveReaderShell.on_mount separately for this Mount event
        (TASK-31822).
        """
        collapse = self.query("#library-rail-collapse")
        if collapse:
            collapse.first().display = False
