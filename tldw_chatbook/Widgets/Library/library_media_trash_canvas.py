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
    MediaTrashMutationTarget,
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
        retry_visible: bool | None = None,
        controls_disabled_reason: str = "",
        confirmation_target: MediaTrashMutationTarget | None = None,
        commit_pending: bool = False,
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
            retry_visible=retry_visible,
            controls_disabled_reason=controls_disabled_reason,
            confirmation_target=confirmation_target,
            commit_pending=commit_pending,
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
        # Measured row-list cap (task-28015); see ``_cap_trash_list``. Kept on
        # the instance so a recompose re-applies it without a blank frame.
        self._list_cap: int | None = None

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
        retry_visible: bool | None,
        controls_disabled_reason: str,
        confirmation_target: MediaTrashMutationTarget | None,
        commit_pending: bool,
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
        self.retry_visible = (
            bool(pager and pager.retry_visible)
            if retry_visible is None
            else retry_visible
        )
        self.controls_disabled_reason = controls_disabled_reason
        self.confirmation_target = confirmation_target
        self.commit_pending = commit_pending

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
        retry_visible: bool | None = None,
        controls_disabled_reason: str = "",
        confirmation_target: MediaTrashMutationTarget | None = None,
        commit_pending: bool = False,
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
            retry_visible=retry_visible,
            controls_disabled_reason=controls_disabled_reason,
            confirmation_target=confirmation_target,
            commit_pending=commit_pending,
        )
        self.refresh(recompose=True)

    def on_mount(self) -> None:
        """Measure the list cap and status overflow after the first layout."""
        self.call_after_refresh(self._measure_after_layout)

    def on_resize(self) -> None:
        """Re-measure against the current Items allocation."""
        self.call_after_refresh(self._measure_after_layout)

    def _after_recompose(self) -> None:
        """Re-measure the newly mounted children without delaying focus."""
        self.call_after_refresh(self._measure_after_layout)

    def _measure_after_layout(self) -> None:
        """Post-layout measurements: the row-list cap, then the status fold."""
        self._cap_trash_list()
        self._update_status_fold()

    def _cap_trash_list(self) -> None:
        """Bound the auto-height row list to the space its siblings leave.

        task-28015: the list was ``height: 1fr``, so a one-item Trash held
        ~36 blank rows between the row and the Restore action docked at the
        panel bottom. Auto height puts Restore back beside the item, but
        Textual has no "auto up to the remaining space" scalar -- a
        ``max-height`` of ``1fr``/``100%`` resolves against the CONTAINER,
        not the remainder, which would let a full page push the pager and
        the actions off the terminal again (the reachability contract
        ``_assert_trash_rows_and_restore_reachable`` pins). So the cap is
        measured from the laid-out siblings and re-applied on resize and
        after every recompose; the list scrolls its own overflow as before.
        """
        try:
            trash_list = self.query_one("#library-media-trash-list", VerticalScroll)
        except Exception:
            return
        available = self.content_size.height
        if available < 1:
            return
        above = trash_list.region.y - self.content_region.y
        below = max(
            (
                child.region.bottom
                for child in self.children
                if child.display and child.region.y >= trash_list.region.bottom
            ),
            default=trash_list.region.bottom,
        ) - trash_list.region.bottom
        cap = max(2, available - above - below)
        if cap != self._list_cap:
            self._list_cap = cap
            trash_list.styles.max_height = cap

    def _update_status_fold(self) -> None:
        """Expose a fold row only when the status needs more than two rows."""
        if self.pager is None:
            return
        try:
            status = self.query_one("#library-media-trash-status", Static)
            fold = self.query_one("#library-media-trash-status-fold", Static)
        except Exception:
            return
        width = status.content_size.width or status.region.width
        if width < 1:
            return
        fold.display = status.visual.get_height(status.styles, width) > 2

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
            back = Button(
                "‹ Media",
                id="library-media-trash-back",
                classes="library-canvas-action",
                compact=True,
            )
            back.disabled = self.commit_pending
            back.tooltip = "Finishing this action…" if self.commit_pending else None
            # task-28015: Textual Button's 16-cell min-width floor left the
            # ~38-column Items pane too little for the count, which painted
            # as "Local Trash · 1 i". Same escape the pager buttons below
            # already take; the label keeps its own content width.
            back.styles.min_width = 0
            yield back
            count = self.pager.title_count if self.pager is not None else None
            title = (
                "Local Trash"
                if count is None
                else (
                    f"Local Trash · {count} matching"
                    if self.applied_scope_label
                    else f"Local Trash · {count} {'item' if count == 1 else 'items'}"
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
                    choices.disabled = bool(self.controls_disabled_reason)
                    choices.tooltip = self.controls_disabled_reason or None
                    yield choices
                else:
                    search = Input(
                        self.query_draft,
                        placeholder="Search Trash",
                        max_length=200,
                        id="library-media-trash-search",
                        compact=True,
                        disabled=bool(self.controls_disabled_reason),
                    )
                    search.styles.width = "1fr"
                    search.styles.min_width = 5
                    search.styles.height = 1
                    search.styles.min_height = 1
                    search.styles.padding = 0
                    search.styles.border = ("none", "transparent")
                    search.tooltip = self.controls_disabled_reason or None
                    yield search
                    type_disabled = bool(self.controls_disabled_reason)
                    yield Button(
                        library_disabled_action_label(
                            f"Type: {self.applied_type or 'All'}", type_disabled
                        ),
                        id="library-media-trash-type-filter",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=type_disabled,
                        tooltip=(
                            self.controls_disabled_reason
                            or "Choose a Trash media type."
                        ),
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

        # One bounded status region, in honesty order: a fetch error outranks
        # everything; loading outranks the empty copy (an unloaded Trash
        # must never claim to be empty); then the truncation line or the
        # honest empty state.
        if self.commit_pending:
            status_text = "Finishing this action…"
        elif self.canvas.error:
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
        status.styles.height = "auto"
        status.styles.min_height = 1 if bounded else 0
        status.styles.max_height = 2 if bounded else 3
        status.styles.overflow = ("hidden", "hidden")
        status.tooltip = status_text or None
        status.display = bounded or bool(status_text)
        yield status
        if bounded:
            fold = Static(
                "▼ more status",
                id="library-media-trash-status-fold",
                classes="destination-purpose",
                markup=False,
            )
            fold.styles.height = 1
            fold.styles.min_height = 1
            fold.styles.overflow = ("hidden", "hidden")
            fold.tooltip = status_text or None
            fold.display = False
            yield fold

        trash_list = VerticalScroll(id="library-media-trash-list")
        # PR-1505 review (the L3a clipping lesson -- a plain auto-height
        # Vertical clips content past the fold, and a 200-item trash page
        # pushed the Restore toolbar ~100 rows off a 24-row terminal): the
        # list owns the height between the heading/status above and the
        # Restore toolbar below, and scrolls its own overflow. task-28015
        # keeps that ceiling but drops the floor: auto height so a short
        # page ends where its rows end and Restore stays beside the item,
        # with the ceiling measured by `_cap_trash_list` (Textual cannot
        # express "auto up to the remaining space" in CSS). Inline like the
        # rest of this widget's geometry; min_height 0 so it can shrink.
        trash_list.styles.height = "auto"
        trash_list.styles.min_height = 0
        if self._list_cap is not None:
            trash_list.styles.max_height = self._list_cap
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
                    controls_disabled = bool(self.controls_disabled_reason)
                    previous_disabled = (
                        self.pager.previous_disabled or controls_disabled
                    )
                    previous = Button(
                        library_disabled_action_label("Previous", previous_disabled),
                        id="library-media-trash-previous",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=previous_disabled,
                        tooltip=(
                            self.controls_disabled_reason
                            or self.pager.previous_reason
                            or None
                        ),
                    )
                    previous.styles.min_width = 0
                    previous.styles.padding = 0
                    yield previous
                    if self.retry_visible:
                        retry = Button(
                            library_disabled_action_label("Retry", controls_disabled),
                            id="library-media-trash-retry",
                            classes="library-canvas-action",
                            compact=True,
                            disabled=controls_disabled,
                            tooltip=(
                                self.controls_disabled_reason
                                or "Retry the failed Trash request."
                            ),
                        )
                        retry.styles.min_width = 0
                        retry.styles.padding = 0
                        yield retry
                    next_disabled = self.pager.next_disabled or controls_disabled
                    next_button = Button(
                        library_disabled_action_label("Next", next_disabled),
                        id="library-media-trash-next",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=next_disabled,
                        tooltip=(
                            self.controls_disabled_reason
                            or self.pager.next_reason
                            or None
                        ),
                    )
                    next_button.styles.min_width = 0
                    next_button.styles.padding = 0
                    yield next_button

        if self.confirmation_target is not None:
            confirmation = Vertical(id="library-media-trash-delete-confirmation")
            confirmation.styles.height = 5
            confirmation.styles.min_height = 5
            confirmation.styles.overflow = ("hidden", "hidden")
            with confirmation:
                consequence = Static(
                    "This cannot be undone.",
                    id="library-media-trash-delete-confirm-consequence",
                    markup=False,
                )
                consequence.styles.height = 1
                consequence.styles.min_height = 1
                consequence.styles.overflow = ("hidden", "hidden")
                yield consequence

                details = VerticalScroll(
                    id="library-media-trash-delete-confirm-details"
                )
                details.styles.height = 1
                details.styles.min_height = 1
                details.styles.overflow_y = "auto"
                details.styles.overflow_x = "hidden"
                with details:
                    yield Static(
                        self.confirmation_target.title,
                        id="library-media-trash-delete-confirm-title",
                        markup=False,
                    )

                identity = Vertical(id="library-media-trash-delete-confirm-identity")
                identity.styles.height = 2
                identity.styles.min_height = 2
                identity.styles.overflow = ("hidden", "hidden")
                with identity:
                    media_type = self.confirmation_target.media_type or "Unknown type"
                    trash_date = (
                        self.confirmation_target.trash_date or "Unknown deletion time"
                    )
                    type_identity = Static(
                        media_type,
                        id="library-media-trash-delete-confirm-type",
                        markup=False,
                    )
                    type_identity.styles.height = 1
                    type_identity.styles.min_height = 1
                    type_identity.styles.overflow = ("hidden", "hidden")
                    yield type_identity
                    time_identity = Static(
                        trash_date,
                        id="library-media-trash-delete-confirm-time",
                        markup=False,
                    )
                    time_identity.styles.height = 1
                    time_identity.styles.min_height = 1
                    time_identity.styles.overflow = ("hidden", "hidden")
                    yield time_identity

                buttons = Horizontal(
                    classes="ds-toolbar",
                    id="library-media-trash-delete-confirm-actions",
                )
                buttons.styles.height = 1
                buttons.styles.min_height = 1
                buttons.styles.overflow = ("hidden", "hidden")
                with buttons:
                    cancel = Button(
                        "Cancel",
                        id="library-media-trash-delete-cancel",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    cancel.styles.min_width = 0
                    cancel.styles.padding = 0
                    cancel.styles.margin = (0, -1, 0, 0)
                    yield cancel
                    confirm = Button(
                        "Delete permanently",
                        id="library-media-trash-delete-confirm",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    confirm.styles.min_width = 0
                    confirm.styles.padding = 0
                    confirm.styles.margin = 0
                    confirm.styles.offset = (-1, 0)
                    yield confirm
            return

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
