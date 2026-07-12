"""Library Browse ▸ Media canvas: media list, type filter, and preview."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
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
        select_mode = getattr(self.canvas, "select_mode", False)
        # Gate/label off the RENDERED rows, not ``canvas.count`` -- the latter
        # is the pre-filter total across ALL media types, so with a media-type
        # filter active it overstates what's shown (and stays > 0 when the
        # filter renders nothing). ``handle_library_media_select_all`` already
        # selects only the rendered rows, so this keeps the copy/gate honest.
        # Also portable to the conversations canvas state, which has no
        # ``.count`` field.
        rendered_count = len(self.canvas.rows)
        export_btn = Button("Export…", id="library-media-export",
                            classes="library-canvas-action", compact=True)
        export_btn.display = not select_mode
        yield export_btn
        select_btn = Button("Done" if select_mode else "Select",
                            id="library-media-select-toggle",
                            classes="library-canvas-action", compact=True)
        select_btn.disabled = rendered_count == 0
        yield select_btn
        if select_mode:
            action_row = Horizontal(classes="ds-toolbar")
            action_row.styles.height = "auto"
            with action_row:
                yield Static(f"{self.canvas.selected_count} selected",
                             id="library-media-selected-count", markup=False)
                yield Button(f"Select all {rendered_count} shown",
                             id="library-media-select-all",
                             classes="library-canvas-action", compact=True)
                yield Button("Clear", id="library-media-select-clear",
                             classes="library-canvas-action", compact=True)
                export_selected = Button("Export selected",
                                         id="library-media-export-selected",
                                         classes="library-canvas-action", compact=True)
                export_selected.disabled = self.canvas.selected_count == 0
                yield export_selected

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
                if select_mode:
                    marker = "☑" if row.checked else "☐"
                else:
                    marker = "▸" if row.selected else " "
                button = Button(
                    f"{marker} {_visible_row_title(row.title)}"
                    f"\n    {row.secondary}",
                    id=f"library-media-row-{index}",
                    classes="library-media-row",
                    compact=True,
                )
                button.media_id = row.media_id
                # Tooltips are rendered as markup too -- escape user titles.
                button.tooltip = escape_markup(row.title)
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
                # Opens the selected item in the IN-LIBRARY media viewer
                # (nav stays on Library), so the label must not promise the
                # legacy Media manager -- that escape hatch lives on the
                # full viewer's own action row (`#library-media-open`,
                # `LibraryMediaViewer`), which genuinely navigates there.
                yield Button(
                    "Open in viewer",
                    id="library-media-open-viewer",
                    classes="library-canvas-action",
                    compact=True,
                )
