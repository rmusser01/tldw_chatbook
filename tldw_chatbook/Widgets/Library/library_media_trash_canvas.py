"""Library media Trash view: browsable, restorable trashed-media list.

task-4025: the third ``_library_media_view`` value ("trash") of the Browse ▸
Media canvas -- entered from the media list's "Trash" toolbar action, exited
via "‹ Media"/Escape. Lists every ``is_trash=1`` item (via the
``list_media_trash`` seam) and restores per item through the existing
``restore_media_item`` seam. Restore deliberately has NO receipt: ADR-055's
receipts accompany destruction, and restore is recovery -- its feedback is
the row leaving this list, the counts moving, and the transient notice line.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Library.library_media_state import (
    LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_EMPTY_TOOLTIP,
    LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_ERROR_TOOLTIP,
    LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_LOADING_TOOLTIP,
    LIBRARY_MEDIA_TRASH_RESTORE_TOOLTIP,
    LibraryMediaTrashState,
)
from tldw_chatbook.Library.library_pager_state import LibraryPagerDisplay
from tldw_chatbook.Library.library_shell_state import (
    library_disabled_action_label,
)
from tldw_chatbook.Widgets.Library.library_rail import _visible_row_title
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


class LibraryMediaTrashCanvas(PostRecomposeCallback, RecomposeCaptureGuard, Vertical):
    """Render the Library media Trash list with per-item restore.

    Attributes:
        canvas: Current Trash view display state.
    """

    def __init__(
        self,
        canvas: LibraryMediaTrashState,
        *,
        pager: LibraryPagerDisplay | None = None,
        types: tuple[str, ...] = (),
        query_draft: str = "",
        applied_scope_label: str = "",
        applied_type: str | None = None,
        type_choices_visible: bool = False,
        action_disabled_reason: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._set_presentation(
            canvas,
            pager=pager,
            types=types,
            query_draft=query_draft,
            applied_scope_label=applied_scope_label,
            applied_type=applied_type,
            type_choices_visible=type_choices_visible,
            action_disabled_reason=action_disabled_reason,
        )
        # Same width contract as ``LibraryMediaCanvas`` -- this view swaps
        # in for the list inside the same canvas host. 1fr (never 13fr):
        # see that canvas's constructor comment -- an independent 13fr
        # resolves ~13x wider than the host and clips children.
        self.styles.width = "1fr"
        self.styles.min_width = 0
        self.styles.height = "100%"
        self.styles.min_height = 0
        self.styles.overflow = ("hidden", "hidden")

    def _set_presentation(
        self,
        canvas: LibraryMediaTrashState,
        *,
        pager: LibraryPagerDisplay | None,
        types: tuple[str, ...],
        query_draft: str,
        applied_scope_label: str,
        applied_type: str | None,
        type_choices_visible: bool,
        action_disabled_reason: str,
    ) -> None:
        """Store the screen-owned presentation without deriving authority."""
        self.canvas = canvas
        self.pager = pager
        self.types = tuple(types)
        self.query_draft = query_draft
        self.applied_scope_label = applied_scope_label
        self.applied_type = applied_type
        self.type_choices_visible = type_choices_visible
        self.action_disabled_reason = action_disabled_reason

    def sync_state(
        self,
        canvas: LibraryMediaTrashState,
        *,
        pager: LibraryPagerDisplay | None = None,
        types: tuple[str, ...] = (),
        query_draft: str = "",
        applied_scope_label: str = "",
        applied_type: str | None = None,
        type_choices_visible: bool = False,
        action_disabled_reason: str = "",
    ) -> None:
        """Refresh the canvas from new state.

        Args:
            canvas: Latest Trash view display state.

        Returns:
            None.
        """
        self._set_presentation(
            canvas,
            pager=pager,
            types=types,
            query_draft=query_draft,
            applied_scope_label=applied_scope_label,
            applied_type=applied_type,
            type_choices_visible=type_choices_visible,
            action_disabled_reason=action_disabled_reason,
        )
        self.refresh(recompose=True)

    def compose(self) -> ComposeResult:
        """Render the heading, status/notice lines, trash rows, and Restore.

        Returns:
            ComposeResult for the Trash view.
        """
        bounded = self.pager is not None
        heading = Horizontal(classes="ds-toolbar", id="library-media-trash-heading")
        heading.styles.height = 1
        heading.styles.min_height = 1
        heading.styles.overflow = ("hidden", "hidden")
        with heading:
            yield Button(
                "‹ Media",
                id="library-media-trash-back",
                classes="library-canvas-action",
                compact=True,
            )
            title = (
                "Local Trash"
                if self.pager is None or self.pager.title_count is None
                else (
                    f"Local Trash · {self.pager.title_count} matching"
                    if self.applied_scope_label
                    else f"Local Trash · {self.pager.title_count} items"
                )
            )
            yield Static(
                title if bounded else f"Trash ({self.canvas.count})",
                id="library-media-trash-title",
                classes="library-toolbar-count",
                markup=False,
            )

        if bounded:
            filters = Horizontal(id="library-media-trash-filters")
            filters.styles.height = 1
            filters.styles.min_height = 1
            filters.styles.overflow = ("hidden", "hidden")
            with filters:
                if self.type_choices_visible:
                    options: list[Option] = []
                    highlighted = 0
                    for index, value in enumerate((None, *self.types)):
                        display = "All types" if value is None else value
                        option = Option(
                            f"✓ {display}" if value == self.applied_type else display,
                            id=f"library-media-trash-type-option-{index}",
                        )
                        option.choice_value = value
                        options.append(option)
                        if value == self.applied_type:
                            highlighted = index
                    choices = OptionList(
                        *options,
                        id="library-media-trash-type-choices",
                        compact=True,
                        markup=False,
                    )
                    choices.highlighted = highlighted
                    choices.styles.width = "100%"
                    choices.styles.height = 1
                    choices.styles.min_height = 1
                    yield choices
                else:
                    search = Input(
                        self.query_draft,
                        placeholder="Search Trash",
                        max_length=200,
                        id="library-media-trash-search",
                        compact=True,
                    )
                    search.styles.width = "1fr"
                    search.styles.min_width = 5
                    search.styles.height = 1
                    search.styles.min_height = 1
                    search.styles.padding = 0
                    search.styles.border = ("none", "transparent")
                    yield search
                    yield Button(
                        f"Type: {self.applied_type or 'All'}",
                        id="library-media-trash-type-filter",
                        classes="library-canvas-action",
                        compact=True,
                        tooltip="Choose a Trash media type.",
                    )
                    scope = Static(
                        self.applied_scope_label,
                        id="library-media-trash-scope",
                        classes="destination-purpose",
                        markup=False,
                    )
                    scope.styles.width = "auto"
                    scope.styles.max_width = 28
                    scope.styles.height = 1
                    yield scope
        else:
            # Compatibility for the pre-paging standalone canvas contract.
            notice = Static(
                self.canvas.notice,
                id="library-media-trash-notice",
                markup=False,
            )
            notice.display = bool(self.canvas.notice)
            yield notice

        # One status line, in honesty order: a fetch error outranks
        # everything; loading outranks the empty copy (an unloaded Trash
        # must never claim to be empty); then the truncation line or the
        # honest empty state.
        if self.canvas.error:
            status_text = self.canvas.error
        elif self.canvas.loading:
            status_text = "Loading Trash…"
        elif bounded and self.canvas.notice:
            status_text = self.canvas.notice
        elif bounded and self.pager is not None and self.pager.status_copy:
            status_text = self.pager.status_copy
        else:
            status_text = self.canvas.status_copy or self.canvas.empty_copy
        status = Static(
            status_text,
            id="library-media-trash-status",
            markup=False,
        )
        status.styles.height = 1 if bounded else "auto"
        status.styles.min_height = 1 if bounded else 0
        status.styles.max_height = 3
        status.styles.overflow = ("hidden", "hidden")
        status.tooltip = status_text or None
        status.display = bounded or bool(status_text)
        yield status

        trash_list = VerticalScroll(id="library-media-trash-list")
        # PR-1505 review (the L3a clipping lesson -- a plain auto-height
        # Vertical clips content past the fold, and a 200-item trash page
        # pushed the Restore toolbar ~100 rows off a 24-row terminal): the
        # list owns the remaining height between the heading/status above
        # and the Restore toolbar below, and scrolls its own overflow --
        # the same geometry `#library-media-list` gets from the stylesheet
        # in the wide split. Inline like the rest of this widget's
        # geometry; min_height 0 so the 1fr can actually shrink.
        trash_list.styles.height = "1fr"
        trash_list.styles.min_height = 0
        trash_list.styles.overflow_y = "auto"
        trash_list.styles.overflow_x = "hidden"
        with trash_list:
            for index, row in enumerate(self.canvas.rows):
                # Selected-row grammar: leading "▸ " (task-4023 AC#5).
                marker = "▸" if row.selected else " "
                label_rest = f" {_visible_row_title(row.title)}\n    {row.secondary}"
                button = Button(
                    f"{marker}{label_rest}",
                    id=f"library-media-trash-row-{index}",
                    classes="library-media-trash-row",
                    compact=True,
                )
                button.media_id = row.media_id
                button._library_row_label_rest = label_rest
                # Tooltips render markup -- escape user titles.
                button.tooltip = escape_markup(row.title)
                # Reuses the media list's selected-row styling class -- one
                # CSS contract, not a fork.
                button.set_class(row.selected, "library-media-row-selected")
                button.styles.height = 2
                button.styles.min_height = 2
                yield button

        if bounded and self.pager is not None:
            pager = Vertical(id="library-media-trash-pager")
            pager.styles.height = 2
            pager.styles.min_height = 2
            pager.styles.overflow = ("hidden", "hidden")
            with pager:
                copy = Horizontal(id="library-media-trash-pager-copy")
                copy.styles.height = 1
                copy.styles.min_height = 1
                copy.styles.overflow = ("hidden", "hidden")
                with copy:
                    yield Static(
                        self.pager.range_copy,
                        id="library-media-trash-range",
                        markup=False,
                    )
                    yield Static(
                        self.pager.page_copy,
                        id="library-media-trash-page",
                        markup=False,
                    )
                controls = Horizontal(
                    classes="ds-toolbar", id="library-media-trash-pager-controls"
                )
                controls.styles.height = 1
                controls.styles.min_height = 1
                controls.styles.overflow = ("hidden", "hidden")
                with controls:
                    previous = Button(
                        library_disabled_action_label(
                            "Previous", self.pager.previous_disabled
                        ),
                        id="library-media-trash-previous",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=self.pager.previous_disabled,
                        tooltip=self.pager.previous_reason or None,
                    )
                    previous.styles.min_width = 0
                    previous.styles.padding = 0
                    yield previous
                    if self.pager.retry_visible:
                        retry = Button(
                            "Retry",
                            id="library-media-trash-retry",
                            classes="library-canvas-action",
                            compact=True,
                            tooltip="Retry the failed Trash request.",
                        )
                        retry.styles.min_width = 0
                        retry.styles.padding = 0
                        yield retry
                    next_button = Button(
                        library_disabled_action_label("Next", self.pager.next_disabled),
                        id="library-media-trash-next",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=self.pager.next_disabled,
                        tooltip=self.pager.next_reason or None,
                    )
                    next_button.styles.min_width = 0
                    next_button.styles.padding = 0
                    yield next_button

        toolbar = Horizontal(classes="ds-toolbar", id="library-media-trash-actions")
        toolbar.styles.height = 1
        toolbar.styles.min_height = 1
        toolbar.styles.padding = 0
        toolbar.styles.overflow = ("hidden", "hidden")
        with toolbar:
            if bounded:
                disabled_reason = self.action_disabled_reason
                action_disabled = bool(disabled_reason)
                restore_tooltip = (
                    disabled_reason
                    if action_disabled
                    else LIBRARY_MEDIA_TRASH_RESTORE_TOOLTIP
                )
            else:
                action_disabled = self.canvas.loading or not self.canvas.rows
                if not action_disabled:
                    restore_tooltip = LIBRARY_MEDIA_TRASH_RESTORE_TOOLTIP
                elif self.canvas.error:
                    restore_tooltip = LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_ERROR_TOOLTIP
                elif self.canvas.loading:
                    restore_tooltip = (
                        LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_LOADING_TOOLTIP
                    )
                else:
                    restore_tooltip = LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_EMPTY_TOOLTIP
            restore = Button(
                # F-018 + task-4023 AC#1: disabled carries the non-colour
                # "○" marker and a reason tooltip.
                library_disabled_action_label("Restore", action_disabled),
                id="library-media-trash-restore",
                classes="library-canvas-action",
                compact=True,
            )
            restore._library_disabled_marker_base = "Restore"
            restore.disabled = action_disabled
            restore.tooltip = restore_tooltip
            restore.styles.min_width = 0
            restore.styles.padding = 0
            # Textual's non-removable Button line-pad reserves one blank cell
            # on each edge. At the shipped 32-column compact Items allocation,
            # the two exact disabled labels need 33 content/line-pad cells.
            # Overlap only those adjacent blank edge cells; both labels and
            # both focus targets remain whole and inside the action row.
            restore.styles.margin = (0, -1, 0, 0)
            yield restore
            if bounded:
                delete = Button(
                    library_disabled_action_label(
                        "Delete permanently", action_disabled
                    ),
                    id="library-media-trash-delete",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=action_disabled,
                    tooltip=(
                        disabled_reason
                        if action_disabled
                        else "Delete this Trash item permanently."
                    ),
                )
                delete._library_disabled_marker_base = "Delete permanently"
                delete.styles.min_width = 0
                delete.styles.padding = 0
                delete.styles.margin = 0
                delete.styles.offset = (-1, 0)
                yield delete
