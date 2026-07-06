"""Library Browse ▸ Media canvas: media list, type filter, and preview."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_media_state import LibraryMediaCanvasState
from tldw_chatbook.Widgets.Library.library_rail import _visible_row_title


class LibraryMediaCanvas(Vertical):
    """Render the Library media list with a type filter and preview.

    Attributes:
        canvas: Current media canvas display state.
    """

    def __init__(
        self,
        canvas: LibraryMediaCanvasState,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.canvas = canvas
        self.styles.width = "13fr"
        self.styles.min_width = 40

    def sync_state(self, canvas: LibraryMediaCanvasState) -> None:
        """Refresh the canvas from new state.

        Args:
            canvas: Latest media canvas display state.

        Returns:
            None.
        """
        self.canvas = canvas
        self.refresh(recompose=True)

    def compose(self) -> ComposeResult:
        """Render the header/filter, status line, media rows, and preview.

        Returns:
            ComposeResult for the media canvas.
        """
        yield Static(
            f"Media ({self.canvas.count})",
            id="library-media-title",
        )
        yield Button(
            f"type: {self.canvas.active_type} ▸",
            id="library-media-type-filter",
            classes="library-canvas-action",
            compact=True,
        )

        status_text = self.canvas.status_copy or self.canvas.empty_copy
        status = Static(
            status_text,
            id="library-media-status",
            markup=False,
        )
        status.display = bool(status_text)
        yield status

        media_list = Vertical(id="library-media-list")
        media_list.styles.height = "auto"
        with media_list:
            for index, row in enumerate(self.canvas.rows):
                marker = "▸" if row.selected else " "
                button = Button(
                    f"{marker} {_visible_row_title(row.title)}"
                    f"\n    {row.secondary}",
                    id=f"library-media-row-{index}",
                    classes="library-media-row",
                    compact=True,
                )
                button.media_id = row.media_id
                button.tooltip = row.title
                button.set_class(row.selected, "library-media-row-selected")
                button.styles.height = 2
                button.styles.min_height = 2
                yield button

        preview = Vertical(id="library-media-preview")
        preview.styles.height = "auto"
        has_preview = bool(self.canvas.selected_id and self.canvas.preview_lines)
        preview.display = has_preview
        with preview:
            yield Static(
                "\n".join(self.canvas.preview_lines),
                id="library-media-preview-lines",
                markup=False,
            )
            toolbar = Horizontal(classes="ds-toolbar")
            toolbar.styles.height = "auto"
            with toolbar:
                yield Button(
                    "Open in Media",
                    id="library-media-open",
                    classes="library-canvas-action",
                    compact=True,
                )
