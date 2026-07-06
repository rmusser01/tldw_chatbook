"""Library media viewer canvas: full metadata + content, with a Back control."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_media_viewer_state import LibraryMediaViewerState


class LibraryMediaViewer(Vertical):
    """Render the full Library media item: metadata, content, and actions.

    Attributes:
        viewer: Current media viewer display state.
    """

    def __init__(
        self,
        viewer: LibraryMediaViewerState,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.viewer = viewer
        self.styles.width = "13fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        """Render the back control, title, metadata, content, and actions.

        Uses only render-verified widgets (``Static``/``VerticalScroll``/
        ``Button``) stacked full-width in this ``Vertical`` — horizontal rows
        that mix a ``1fr`` sibling with a fixed-width widget are the known
        non-rendering failure mode, so every row here is either a single
        full-width widget or the plain ``ds-toolbar`` action row (already
        proven to render by the conversations/media list canvases).

        Returns:
            ComposeResult for the media viewer canvas.
        """
        yield Button(
            "‹ Back to list",
            id="library-media-back",
            classes="library-canvas-action",
            compact=True,
        )
        yield Static(
            self.viewer.title,
            id="library-media-viewer-title",
            markup=False,
        )
        yield Static(
            "\n".join(self.viewer.metadata_lines),
            id="library-media-viewer-meta",
            markup=False,
        )
        yield Static(
            "Content",
            id="library-media-viewer-content-title",
            classes="destination-section",
        )
        with VerticalScroll(id="library-media-viewer-content"):
            yield Static(
                self.viewer.content or "No stored content.",
                id="library-media-viewer-content-text",
                markup=False,
            )
        if self.viewer.has_analysis:
            yield Static(
                "Analysis",
                id="library-media-viewer-analysis-title",
                classes="destination-section",
            )
            yield Static(
                self.viewer.analysis,
                id="library-media-viewer-analysis-text",
                markup=False,
            )

        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                "Edit",
                id="library-media-edit",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Delete",
                id="library-media-delete",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Read it later",
                id="library-media-read-later",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Use in Chat",
                id="library-media-use-in-chat",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Open in Media",
                id="library-media-open",
                classes="library-canvas-action",
                compact=True,
            )
