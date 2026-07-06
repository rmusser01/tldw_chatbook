"""Library media viewer canvas: full metadata + content, with a Back control."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_media_viewer_state import LibraryMediaViewerState


class LibraryMediaViewer(Vertical):
    """Render the full Library media item: metadata, content, and actions.

    Attributes:
        viewer: Current media viewer display state.
        editing: Whether the metadata edit form should render in place of
            the read-only metadata block and action row.
    """

    def __init__(
        self,
        viewer: LibraryMediaViewerState,
        *,
        editing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.viewer = viewer
        self.editing = editing
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
        if self.editing:
            yield from self._compose_edit_form()
        else:
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
            if self.editing:
                yield Button(
                    "Save",
                    id="library-media-edit-save",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Cancel",
                    id="library-media-edit-cancel",
                    classes="library-canvas-action",
                    compact=True,
                )
            else:
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

    def _compose_edit_form(self) -> ComposeResult:
        """Render the metadata edit inputs, prefilled from ``viewer.edit_fields``.

        Stacked full-width ``Input`` widgets in a plain ``Vertical`` --
        matching the render-verified pattern already used by the Library
        Collections create/rename form.

        Returns:
            ComposeResult for the metadata edit form.
        """
        with Vertical(id="library-media-edit-form"):
            yield Input(
                value=self.viewer.edit_fields.get("title", ""),
                placeholder="Title",
                id="library-media-edit-title",
            )
            yield Input(
                value=self.viewer.edit_fields.get("author", ""),
                placeholder="Author",
                id="library-media-edit-author",
            )
            yield Input(
                value=self.viewer.edit_fields.get("url", ""),
                placeholder="URL",
                id="library-media-edit-url",
            )
            yield Input(
                value=self.viewer.edit_fields.get("keywords", ""),
                placeholder="Keywords (comma-separated)",
                id="library-media-edit-keywords",
            )
