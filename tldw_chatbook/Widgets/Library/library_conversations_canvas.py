"""Library Browse ▸ Conversations canvas: saved-chat list and preview."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_TOOLTIP,
    LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP,
    library_disabled_action_label,
)
from tldw_chatbook.Library.library_conversations_state import (
    LibraryConversationsCanvasState,
)
from tldw_chatbook.Widgets.Library.library_rail import _visible_row_title
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


class LibraryConversationsCanvas(PostRecomposeCallback, RecomposeCaptureGuard, Vertical):
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
        self.styles.width = "100%"

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
        rendered_count = len(self.canvas.rows)
        pager = self.canvas.pager
        title_count = pager.title_count if pager is not None else None
        title = (
            "Conversations"
            if title_count is None
            else f"Conversations ({title_count})"
        )
        yield Static(
            title,
            id="library-conversations-title",
            classes="destination-section",
            markup=False,
        )
        actions_disabled = self.canvas.actions_disabled
        export_btn = Button(
            library_disabled_action_label("Export…", actions_disabled),
            id="library-conversations-export",
            classes="library-canvas-action",
            compact=True,
        )
        export_btn.disabled = actions_disabled
        export_btn.display = not select_mode
        yield export_btn
        # Disable only when nothing to select AND not already in select mode --
        # in select mode "Done" must stay pressable so the user can always exit,
        # even if the rows dropped to zero (e.g. a background snapshot refresh).
        select_disabled = actions_disabled or (
            rendered_count == 0 and not select_mode
        )
        select_btn = Button(
            # task-4023 AC#1 (RC-07): disabled carries the non-colour "○"
            # marker; the F-018 reason tooltip below says why.
            library_disabled_action_label(
                "Done" if select_mode else "Select", select_disabled
            ),
            id="library-conversations-select-toggle",
            classes="library-canvas-action",
            compact=True,
        )
        select_btn.disabled = select_disabled
        if select_disabled:
            select_btn.tooltip = LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP
        yield select_btn
        if select_mode:
            action_row = Horizontal(classes="ds-toolbar")
            action_row.styles.height = "auto"
            with action_row:
                # task-2853 review round 2: the SAME unbounded-width defect
                # proved live in the Media canvas's identical counter --
                # see library_media_canvas.py's compose() for the live
                # tmux evidence. Fixed via the shared
                # ``library-toolbar-count`` class (css/components/
                # _agentic_terminal.tcss's ``width: auto``) rather than
                # repeating a Python-side one-off here.
                yield Static(
                    f"{self.canvas.selected_count} selected",
                    id="library-conversations-selected-count",
                    classes="library-toolbar-count",
                    markup=False,
                )
                yield Button(
                    library_disabled_action_label(
                        f"Select all {rendered_count} shown", actions_disabled
                    ),
                    id="library-conversations-select-all",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=actions_disabled,
                )
                yield Button(
                    library_disabled_action_label("Clear", actions_disabled),
                    id="library-conversations-select-clear",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=actions_disabled,
                )
                export_disabled = actions_disabled or self.canvas.selected_count == 0
                export_selected = Button(
                    # task-4023 AC#1 (RC-07): "○" disabled marker; base
                    # label stashed for `_apply_library_row_toggle`'s
                    # in-place patch.
                    library_disabled_action_label(
                        "Export selected", export_disabled
                    ),
                    id="library-conversations-export-selected",
                    classes="library-canvas-action",
                    compact=True,
                )
                export_selected._library_disabled_marker_base = "Export selected"
                export_selected.disabled = export_disabled
                # F-018: a disabled action says why.
                export_selected.tooltip = (
                    LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP
                    if export_selected.disabled
                    else LIBRARY_EXPORT_SELECTED_TOOLTIP
                )
                yield export_selected

        # task-2859 item 1: the filter box now renders ABOVE the empty-state/
        # status text, matching Notes/Prompts (title -> filter -> toolbar ->
        # empty-or-rows) -- it used to sit below the empty-state Static,
        # which read as an afterthought under "No conversations yet.".
        yield Input(
            value=self.canvas.query,
            placeholder="Filter conversations… (Enter)",
            id="library-conversations-filter",
        )

        status_text = self.canvas.status_copy or self.canvas.empty_copy
        status = Static(
            status_text,
            id="library-conversations-status",
            markup=False,
        )
        status.display = bool(status_text)
        yield status
        selection_notice = self.canvas.selection_notice
        if selection_notice and selection_notice != status_text:
            yield Static(
                selection_notice,
                id="library-conversations-selection-notice",
                markup=False,
            )

        conversation_list = VerticalScroll(id="library-conversations-list")
        with conversation_list:
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
                    id=f"library-conversation-row-{index}",
                    classes="library-conversation-row",
                    compact=True,
                )
                button.conversation_id = row.conversation_id
                button._library_row_label_rest = label_rest
                # Tooltips are rendered as markup too -- escape user titles.
                button.tooltip = escape_markup(row.title)
                button.set_class(row.selected, "library-conversation-row-selected")
                button.disabled = actions_disabled
                button.styles.height = 2
                button.styles.min_height = 2
                yield button

        previous_disabled = (
            pager.previous_disabled
            if pager is not None
            else self.canvas.previous_disabled
        )
        next_disabled = (
            pager.next_disabled if pager is not None else self.canvas.next_disabled
        )
        range_copy = pager.range_copy if pager is not None else self.canvas.range_copy
        page_copy = pager.page_copy if pager is not None else self.canvas.page_copy
        previous_reason = pager.previous_reason if pager is not None else ""
        next_reason = pager.next_reason if pager is not None else ""
        disabled_reasons = tuple(
            dict.fromkeys(
                reason
                for disabled, reason in (
                    (previous_disabled, previous_reason),
                    (next_disabled, next_reason),
                )
                if disabled and reason
            )
        )
        with Vertical(
            id="library-conversations-pager",
            classes="library-source-pager",
        ):
            yield Static(
                " · ".join(copy for copy in (range_copy, page_copy) if copy),
                id="library-conversations-page-status",
                classes="library-source-pager-status",
                markup=False,
            )
            if disabled_reasons:
                yield Static(
                    " · ".join(disabled_reasons),
                    id="library-conversations-disabled-reason",
                    classes="library-source-pager-status",
                    markup=False,
                )
            with Horizontal(classes="library-source-pager-controls"):
                previous = Button(
                    library_disabled_action_label("Previous", previous_disabled),
                    id="library-conversations-previous",
                    classes="library-canvas-action",
                    compact=True,
                )
                previous.disabled = previous_disabled
                if previous_disabled:
                    previous.tooltip = previous_reason
                yield previous
                if pager is not None and pager.retry_visible:
                    yield Button(
                        "Try again",
                        id="library-conversations-retry",
                        classes="library-canvas-action",
                        compact=True,
                    )
                next_page = Button(
                    library_disabled_action_label("Next", next_disabled),
                    id="library-conversations-next",
                    classes="library-canvas-action",
                    compact=True,
                )
                next_page.disabled = next_disabled
                if next_disabled:
                    next_page.tooltip = next_reason
                yield next_page

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
                    library_disabled_action_label(
                        "Open in Console", actions_disabled
                    ),
                    id="library-conversation-open-console",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=actions_disabled,
                )
