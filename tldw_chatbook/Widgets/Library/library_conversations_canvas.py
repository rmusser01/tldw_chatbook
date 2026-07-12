"""Library Browse ▸ Conversations canvas: saved-chat list and preview."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_conversations_state import (
    LibraryConversationsCanvasState,
)
from tldw_chatbook.Widgets.Library.library_rail import _visible_row_title


class LibraryConversationsCanvas(Vertical):
    """Render the saved-conversation list with a preview + Console handoff.

    Attributes:
        canvas: Current conversations canvas display state.
    """

    def __init__(
        self,
        canvas: LibraryConversationsCanvasState,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.canvas = canvas
        self.styles.width = "13fr"
        self.styles.min_width = 40

    def sync_state(self, canvas: LibraryConversationsCanvasState) -> None:
        """Refresh the canvas from new state.

        Args:
            canvas: Latest conversations canvas display state.

        Returns:
            None.
        """
        self.canvas = canvas
        self.refresh(recompose=True)

    def compose(self) -> ComposeResult:
        """Render the status line, conversation rows, and selection preview.

        Returns:
            ComposeResult for the conversations canvas.
        """
        select_mode = getattr(self.canvas, "select_mode", False)
        # Gate/label off the RENDERED rows, not a pre-filter total -- the
        # conversations canvas state has no ``.count`` field at all (unlike
        # media), so ``len(self.canvas.rows)`` is the only correct source for
        # "how many rows are shown right now".
        rendered_count = len(self.canvas.rows)
        export_btn = Button(
            "Export…",
            id="library-conversations-export",
            classes="library-canvas-action",
            compact=True,
        )
        export_btn.display = not select_mode
        yield export_btn
        select_btn = Button("Done" if select_mode else "Select",
                            id="library-conversations-select-toggle",
                            classes="library-canvas-action", compact=True)
        # Disable only when nothing to select AND not already in select mode --
        # in select mode "Done" must stay pressable so the user can always exit,
        # even if the rows dropped to zero (e.g. a background snapshot refresh).
        select_btn.disabled = rendered_count == 0 and not select_mode
        yield select_btn
        if select_mode:
            action_row = Horizontal(classes="ds-toolbar")
            action_row.styles.height = "auto"
            with action_row:
                yield Static(f"{self.canvas.selected_count} selected",
                             id="library-conversations-selected-count", markup=False)
                yield Button(f"Select all {rendered_count} shown",
                             id="library-conversations-select-all",
                             classes="library-canvas-action", compact=True)
                yield Button("Clear", id="library-conversations-select-clear",
                             classes="library-canvas-action", compact=True)
                export_selected = Button("Export selected",
                                         id="library-conversations-export-selected",
                                         classes="library-canvas-action", compact=True)
                export_selected.disabled = self.canvas.selected_count == 0
                yield export_selected

        status_text = self.canvas.status_copy or self.canvas.empty_copy
        status = Static(
            status_text,
            id="library-conversations-status",
            markup=False,
        )
        status.display = bool(status_text)
        yield status

        yield Input(
            value=self.canvas.query,
            placeholder="Filter conversations… (Enter)",
            id="library-conversations-filter",
        )

        conversation_list = Vertical(id="library-conversations-list")
        conversation_list.styles.height = "auto"
        with conversation_list:
            for index, row in enumerate(self.canvas.rows):
                if select_mode:
                    marker = "☑" if row.checked else "☐"
                else:
                    marker = "▸" if row.selected else " "
                button = Button(
                    f"{marker} {_visible_row_title(row.title)}"
                    f"\n    {row.secondary}",
                    id=f"library-conversation-row-{index}",
                    classes="library-conversation-row",
                    compact=True,
                )
                button.conversation_id = row.conversation_id
                # Tooltips are rendered as markup too -- escape user titles.
                button.tooltip = escape_markup(row.title)
                button.set_class(row.selected, "library-conversation-row-selected")
                button.styles.height = 2
                button.styles.min_height = 2
                yield button

        preview = Vertical(id="library-conversation-preview")
        preview.styles.height = "auto"
        has_preview = bool(self.canvas.selected_id and self.canvas.preview_lines)
        preview.display = has_preview
        with preview:
            yield Static(
                "\n".join(self.canvas.preview_lines),
                id="library-conversation-preview-lines",
                markup=False,
            )
            toolbar = Horizontal(classes="ds-toolbar")
            toolbar.styles.height = "auto"
            with toolbar:
                yield Button(
                    "Open in Console",
                    id="library-conversation-open-console",
                    classes="library-canvas-action",
                    compact=True,
                )
