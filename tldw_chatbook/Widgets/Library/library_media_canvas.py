"""Library Browse ▸ Media canvas: media list, type filter, and preview."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_media_state import LibraryMediaCanvasState
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_DELETE_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_DELETE_SELECTED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_TOOLTIP,
)
from tldw_chatbook.Widgets.Library.library_rail import _visible_row_title
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


class LibraryMediaCanvas(RecomposeCaptureGuard, Vertical):
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
        export_btn = Button(
            "Export…",
            id="library-media-export",
            classes="library-canvas-action",
            compact=True,
        )
        export_btn.display = not select_mode
        yield export_btn
        select_btn = Button(
            "Done" if select_mode else "Select",
            id="library-media-select-toggle",
            classes="library-canvas-action",
            compact=True,
        )
        # Disable only when there's nothing to select AND we're not already in
        # select mode -- in select mode the button is "Done" and must always be
        # pressable so the user can exit even if the rows dropped to zero
        # (e.g. a background snapshot refresh emptied the list).
        select_btn.disabled = rendered_count == 0 and not select_mode
        yield select_btn
        confirming_bulk_delete = getattr(self.canvas, "confirming_bulk_delete", False)
        if select_mode:
            if confirming_bulk_delete:
                # A single full-width Static above the toolbar, not inside it
                # -- mixing a long sentence Static with the toolbar's fixed-
                # width Buttons in one Horizontal is the known non-rendering
                # failure mode (see LibraryMediaViewer.compose's delete-
                # confirm copy, the same pattern this mirrors). The short
                # "N selected" Static below is unaffected -- it is already
                # proven to render alongside Buttons in this exact row.
                item_word = "item" if self.canvas.selected_count == 1 else "items"
                yield Static(
                    f"Delete {self.canvas.selected_count} selected {item_word}? "
                    "This moves them to trash.",
                    id="library-media-bulk-delete-confirm-copy",
                    markup=False,
                )
            action_row = Horizontal(classes="ds-toolbar")
            action_row.styles.height = "auto"
            with action_row:
                selected_count_static = Static(
                    f"{self.canvas.selected_count} selected",
                    id="library-media-selected-count",
                    markup=False,
                )
                # Bug found via task-2853's OWN live tmux verification
                # (reproduced against pre-task-8 HEAD too, so it predates
                # this task): with no explicit width, this Static's width
                # resolves as unbounded inside the ``ds-toolbar``
                # ``Horizontal`` -- live capture showed it claiming ~1700
                # columns on a 170-column terminal, pushing every sibling
                # Button entirely off-screen (invisible, though still
                # present in the DOM -- which is why headless ``query_one``
                # pilot tests never caught it). Pinning ``width: auto`` (the
                # same value ``Button``'s own DEFAULT_CSS already uses)
                # makes it hug its own text instead.
                selected_count_static.styles.width = "auto"
                yield selected_count_static
                if confirming_bulk_delete:
                    yield Button(
                        "Delete",
                        id="library-media-bulk-delete-confirm",
                        classes="library-canvas-action library-media-action-danger",
                        compact=True,
                    )
                    yield Button(
                        "Cancel",
                        id="library-media-bulk-delete-cancel",
                        classes="library-canvas-action",
                        compact=True,
                    )
                else:
                    yield Button(
                        f"Select all {rendered_count} shown",
                        id="library-media-select-all",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield Button(
                        "Clear",
                        id="library-media-select-clear",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    export_selected = Button(
                        "Export selected",
                        id="library-media-export-selected",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    export_selected.disabled = self.canvas.selected_count == 0
                    # F-018: a disabled action says why.
                    export_selected.tooltip = (
                        LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP
                        if export_selected.disabled
                        else LIBRARY_EXPORT_SELECTED_TOOLTIP
                    )
                    yield export_selected
                    # task-2853: the second real bulk action -- "Delete
                    # selected" -- pushed to the far end (CSS margin, same
                    # library-media-action-danger idiom the single-item
                    # viewer's own Delete uses) so it is never adjacent to
                    # Export selected.
                    delete_selected = Button(
                        "Delete selected",
                        id="library-media-delete-selected",
                        classes="library-canvas-action library-media-action-danger",
                        compact=True,
                    )
                    delete_selected.disabled = self.canvas.selected_count == 0
                    delete_selected.tooltip = (
                        LIBRARY_DELETE_SELECTED_DISABLED_TOOLTIP
                        if delete_selected.disabled
                        else LIBRARY_DELETE_SELECTED_TOOLTIP
                    )
                    yield delete_selected

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
                # task-281 (PR #665 review): the in-place toggle needs the
                # marker-less RAW label to rebuild from -- reading it back
                # off the mounted Button un-escapes user titles (both
                # ``.plain`` and Textual 8's ``str(Content)`` return
                # rendered text), so the raw remainder is stashed here at
                # the single point of truth.
                label_rest = f" {_visible_row_title(row.title)}\n    {row.secondary}"
                button = Button(
                    f"{marker}{label_rest}",
                    id=f"library-media-row-{index}",
                    classes="library-media-row",
                    compact=True,
                )
                button.media_id = row.media_id
                button._library_row_label_rest = label_rest
                # Tooltips are rendered as markup too -- escape user titles.
                button.tooltip = escape_markup(row.title)
                button.set_class(row.selected, "library-media-row-selected")
                button.styles.height = 2
                button.styles.min_height = 2
                yield button

        preview = Vertical(id="library-media-preview")
        preview.styles.height = "auto"
        # task-2853 AC4: while Select mode is active, the preview must never
        # show an item outside the current (multi-item) selection context --
        # ``canvas.selected_id``/``preview_lines`` still carry whatever was
        # focused before Select was entered (the UAT's "bottom preview pane
        # meanwhile shows a previously-selected different item" finding), so
        # the whole block is hidden entirely rather than tracking a second,
        # separate "focused row" concept select mode has no use for.
        has_preview = (
            not select_mode
            and bool(self.canvas.selected_id and self.canvas.preview_lines)
        )
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
                # (nav stays on Library), distinct from the full viewer's
                # own action row (`#library-media-open`, `LibraryMediaViewer`
                # -- "Open in Library ▸ Media", task-2857), which posts a
                # fresh ``NavigateToScreen`` for the "media" route.
                yield Button(
                    "Open in viewer",
                    id="library-media-open-viewer",
                    classes="library-canvas-action",
                    compact=True,
                )
