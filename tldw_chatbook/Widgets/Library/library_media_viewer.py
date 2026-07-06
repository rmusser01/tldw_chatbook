"""Library media viewer canvas: full metadata + content, with a Back control."""

from __future__ import annotations

from typing import Any, Sequence

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_media_viewer_state import (
    LibraryMediaHighlightRow,
    LibraryMediaViewerState,
)


class LibraryMediaViewer(Vertical):
    """Render the full Library media item: metadata, content, and actions.

    Attributes:
        viewer: Current media viewer display state.
        editing: Whether the metadata edit form should render in place of
            the read-only metadata block and action row.
        confirming_delete: Whether the inline delete-confirmation affordance
            should render in place of the normal action row.
        highlights: Reading highlights for this media item, in display order.
    """

    def __init__(
        self,
        viewer: LibraryMediaViewerState,
        *,
        editing: bool = False,
        confirming_delete: bool = False,
        highlights: Sequence[LibraryMediaHighlightRow] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.viewer = viewer
        self.editing = editing
        self.confirming_delete = confirming_delete
        self.highlights = tuple(highlights)
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

        yield from self._compose_highlights()

        if self.confirming_delete and not self.editing:
            # A single full-width Static above the toolbar, not inside it --
            # mixing a Static with the toolbar's Buttons is the known
            # non-rendering failure mode called out on ``compose`` above.
            yield Static(
                "Delete this media? This moves it to trash.",
                id="library-media-delete-confirm-copy",
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
            elif self.confirming_delete:
                yield Button(
                    "Delete",
                    id="library-media-delete-confirm",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Cancel",
                    id="library-media-delete-cancel",
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

    def _compose_highlights(self) -> ComposeResult:
        """Render the highlights section: existing rows, then the add form.

        Each highlight is a full-width ``Static`` (quote, plus a
        ``Color: .../Note: ...`` line when present) immediately followed by
        its own full-width delete ``Button`` -- stacked, not paired inside a
        ``Horizontal``, matching the render-safety rule on ``compose`` above.
        The delete button carries the highlight's id as a plain attribute
        (mirroring ``LibraryMediaCanvas`` setting ``button.media_id``) so the
        screen's class-selector handler can read it back.

        Returns:
            ComposeResult for the highlights section.
        """
        yield Static(
            "Highlights",
            id="library-media-viewer-highlights-title",
            classes="destination-section",
        )
        if not self.highlights:
            yield Static(
                "No highlights yet.",
                id="library-media-viewer-highlights-empty",
                markup=False,
            )
        else:
            for index, highlight in enumerate(self.highlights):
                yield Static(
                    highlight.display_text,
                    id=f"library-media-highlight-{index}",
                    markup=False,
                )
                delete_button = Button(
                    "Delete highlight",
                    id=f"library-media-highlight-delete-{index}",
                    classes="library-canvas-action library-media-highlight-delete",
                    compact=True,
                )
                delete_button.highlight_id = highlight.highlight_id
                yield delete_button
        with Vertical(id="library-media-highlight-form"):
            yield Input(
                placeholder="Quote",
                id="library-media-highlight-quote",
            )
            yield Input(
                placeholder="Note (optional)",
                id="library-media-highlight-note",
            )
            yield Input(
                placeholder="Color (optional)",
                id="library-media-highlight-color",
            )
            yield Button(
                "Add highlight",
                id="library-media-highlight-add",
                classes="library-canvas-action",
                compact=True,
            )
