"""Library Browse ▸ Media canvas: media list, type filter, and preview."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.markup import escape as escape_markup
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.geometry import Size
from textual.message import Message
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Library.library_pager_state import LibraryPagerDisplay
from tldw_chatbook.Library.library_media_state import (
    LibraryMediaCanvasState,
    MEDIA_SORT_CHOICES,
)
from tldw_chatbook.Widgets.Library.library_choice_strip import (
    compose_library_choice_strip,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_DELETE_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_DELETE_SELECTED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_TOOLTIP,
    LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP,
    library_choice_label,
    library_choice_tooltip,
    library_disabled_action_label,
)
from tldw_chatbook.Widgets.Library.library_rail import _visible_row_title
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


_MEDIA_ROW_COMPACT_HEIGHT = 1
_MEDIA_ROW_WIDE_HEIGHT = 2


@dataclass(frozen=True)
class LibraryMediaRowGeometry:
    """One public Textual geometry revision from a Media row-scroll owner."""

    revision: int
    size: Size
    virtual_size: Size
    container_size: Size | None


class LibraryMediaRowGeometryChanged(Message):
    """Report one concrete Media row-scroll owner's revised geometry."""

    def __init__(
        self,
        owner: "LibraryMediaRowScroll",
        geometry: LibraryMediaRowGeometry,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.geometry = geometry


class LibraryMediaRowScroll(VerticalScroll):
    """Publish distinct Resize-derived geometry for the owning Media list."""

    latest_geometry: LibraryMediaRowGeometry | None = None

    def on_resize(self, event: events.Resize) -> None:
        """Publish distinct, monotonically revised owner geometry after reflow."""
        previous = self.latest_geometry
        geometry_values = (event.size, event.virtual_size, event.container_size)
        if previous is not None and geometry_values == (
            previous.size,
            previous.virtual_size,
            previous.container_size,
        ):
            return
        geometry = LibraryMediaRowGeometry(
            revision=1 if previous is None else previous.revision + 1,
            size=event.size,
            virtual_size=event.virtual_size,
            container_size=event.container_size,
        )
        self.latest_geometry = geometry
        self.post_message(LibraryMediaRowGeometryChanged(self, geometry))


def _media_row_label_rest(
    title: str,
    secondary: str,
    *,
    compact: bool,
    loading: bool = False,
    loaded: bool = False,
) -> str:
    """Return the marker-free Media row label for one responsive density."""
    visible_title = _visible_row_title(title)
    state = "Loading" if loading else "Loaded" if loaded else ""
    if compact:
        prefix = f"{state} · " if state else ""
        return f" {prefix}{visible_title} · {secondary}"
    prefix = (
        "Selected · loading preview  "
        if loading
        else "Loaded in Reader            "
        if loaded
        else ""
    )
    return f" {prefix}{visible_title}\n    {secondary}"


class LibraryMediaCanvas(PostRecomposeCallback, RecomposeCaptureGuard, Vertical):
    """Render the Library media list with a type filter and preview.

    Attributes:
        canvas: Current media canvas display state.
    """

    def __init__(
        self,
        canvas: LibraryMediaCanvasState,
        *,
        pager: LibraryPagerDisplay | None = None,
        type_options: tuple[str | None, ...] | None = None,
        stale_action_reason: str = "",
        mutation_action_reason: str = "",
        compact: bool = False,
        show_preview: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.canvas = canvas
        self.pager = pager
        self.type_options = (
            canvas.type_options if type_options is None else type_options
        )
        self.stale_action_reason = stale_action_reason
        self.mutation_action_reason = mutation_action_reason
        self.compact = compact
        self.show_preview = show_preview
        # Fill the (already 13fr) canvas host, not an independent 13fr --
        # ``LibraryMediaViewer`` documented this trap first: an `fr` width
        # here resolves against the HOST's content width per fraction, so
        # 13fr laid this canvas out ~13x wider than visible (measured 1703
        # cols on a 170-col terminal) and children clipped instead of
        # ellipsizing. task-14900's side-by-side split needs the panes to
        # divide the REAL width, so the canvas must be bounded like the
        # viewer already is.
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def sync_state(
        self,
        canvas: LibraryMediaCanvasState,
        *,
        pager: LibraryPagerDisplay | None = None,
        type_options: tuple[str | None, ...] | None = None,
        stale_action_reason: str = "",
        mutation_action_reason: str = "",
        compact: bool = False,
        show_preview: bool = True,
    ) -> None:
        """Refresh the canvas from new state.

        Args:
            canvas: Latest media canvas display state.

        Returns:
            None.
        """
        self.canvas = canvas
        self.pager = pager
        self.type_options = (
            canvas.type_options if type_options is None else type_options
        )
        self.stale_action_reason = stale_action_reason
        self.mutation_action_reason = mutation_action_reason
        self.compact = compact
        self.show_preview = show_preview
        self.refresh(recompose=True)

    def apply_compact_presentation(self, compact: bool) -> None:
        """Patch mounted Media density and preview participation in place."""
        self.compact = compact
        select_mode = getattr(self.canvas, "select_mode", False)
        row_height = (
            _MEDIA_ROW_COMPACT_HEIGHT if compact else _MEDIA_ROW_WIDE_HEIGHT
        )
        for button in self.query(".library-media-row"):
            title = button._library_media_title
            secondary = button._library_media_secondary
            label_rest = _media_row_label_rest(
                title,
                secondary,
                compact=compact,
                loading=button._library_media_loading,
                loaded=button._library_media_loaded,
            )
            button._library_row_label_rest = label_rest
            if select_mode:
                marker = "☑" if button._library_media_checked else "☐"
            else:
                marker = (
                    "▸"
                    if button._library_media_selected and not compact
                    else " "
                )
            button.label = f"{marker}{label_rest}"
            button.set_class(
                button._library_media_selected and not compact and not select_mode,
                "library-media-row-selected",
            )
            button.styles.height = row_height
            button.styles.min_height = row_height
            self._gate_stale_action(button, label_rest.lstrip())
        try:
            preview = self.query_one("#library-media-preview")
            open_viewer = self.query_one("#library-media-open-viewer", Button)
        except NoMatches:
            return
        preview.display = self.show_preview and self._has_preview and not compact
        open_viewer.can_focus = self.show_preview and not compact

    def apply_reader_state(self, canvas: LibraryMediaCanvasState) -> None:
        """Patch Reader row state without replacing row widgets.

        Args:
            canvas: Fresh media canvas state carrying selection and load flags.
        """
        self.canvas = canvas
        rows = {row.media_id: row for row in canvas.rows}
        select_mode = getattr(canvas, "select_mode", False)
        for button in self.query(".library-media-row"):
            row = rows.get(str(button.media_id))
            if row is None:
                continue
            button._library_media_selected = row.selected
            button._library_media_checked = row.checked
            button._library_media_loading = row.loading
            button._library_media_loaded = row.loaded
            label_rest = _media_row_label_rest(
                row.title,
                row.secondary,
                compact=self.compact,
                loading=row.loading,
                loaded=row.loaded,
            )
            button._library_row_label_rest = label_rest
            if select_mode:
                marker = "☑" if row.checked else "☐"
            else:
                marker = "▸" if row.selected and not self.compact else " "
            button.label = f"{marker}{label_rest}"
            button.set_class(
                row.selected and not self.compact and not select_mode,
                "library-media-row-selected",
            )

    def _gate_stale_action(self, button: Button, base_label: str) -> Button:
        """Apply the controller's stale-page gate to one unsafe action."""
        reason = self.mutation_action_reason or self.stale_action_reason
        if reason:
            button.label = library_disabled_action_label(base_label, True)
            button.disabled = True
            button.tooltip = reason
        return button

    def _gate_mutation_action(self, button: Button, base_label: str) -> Button:
        """Disable even recovery controls only while a write is unsettled."""
        if self.mutation_action_reason:
            button.label = library_disabled_action_label(base_label, True)
            button.disabled = True
            button.tooltip = self.mutation_action_reason
        return button

    def compose(self) -> ComposeResult:
        """Render the header/filter, status line, media rows, and preview.

        Returns:
            ComposeResult for the media canvas.
        """
        title_count = self.pager.title_count if self.pager is not None else self.canvas.count
        title = "Media" if title_count is None else f"Media ({title_count})"
        yield Static(title, id="library-media-title")
        filter_row = Horizontal(classes="ds-toolbar")
        filter_row.styles.height = "auto"
        with filter_row:
            yield Input(
                value=self.canvas.query,
                placeholder="Filter media",
                id="library-media-filter",
            )
            clear_filter = Button(
                "Clear filter",
                id="library-media-filter-clear",
                compact=True,
            )
            clear_filter.disabled = not bool(self.canvas.query)
            yield clear_filter
        select_mode = getattr(self.canvas, "select_mode", False)
        if (
            self.pager is not None
            and title_count == 0
            and not self.canvas.rows
            and not select_mode
            and not self.canvas.delete_receipt_count
            and not self.stale_action_reason
            and not self.mutation_action_reason
            and not self.pager.status_copy
            and not self.pager.retry_visible
        ):
            yield Static(
                self.canvas.empty_copy,
                id="library-media-status",
                markup=False,
            )
            if self.canvas.active_type is None:
                yield Button(
                    "Import media",
                    id="library-media-empty-import",
                    classes="library-canvas-action",
                    compact=True,
                )
            else:
                yield Button(
                    "Show all types",
                    id="library-media-empty-clear-type",
                    classes="library-canvas-action",
                    compact=True,
                )
            return
        # Gate/label off the RENDERED rows, not ``canvas.count`` -- the latter
        # is the pre-filter total across ALL media types, so with a media-type
        # filter active it overstates what's shown (and stays > 0 when the
        # filter renders nothing). ``handle_library_media_select_all`` already
        # selects only the rendered rows, so this keeps the copy/gate honest.
        # Also portable to the conversations canvas state, which has no
        # ``.count`` field.
        rendered_count = len(self.canvas.rows)
        # task-4023 AC#5: one toolbar grammar across the list canvases --
        # these three actions used to stack VERTICALLY (one full-width
        # button per row) while Notes/Prompts/Skills lay theirs out in
        # horizontal ``ds-toolbar`` rows. Same render-safe shape as those
        # canvases: fixed-width compact Buttons only, never mixed with a
        # 1fr sibling.
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        # task-14902: while the type choice strip is open it REPLACES this
        # toolbar row (the Notes Sort precedent -- browse actions hide while
        # the chooser is showing), keeping the vertical budget flat.
        type_choices_visible = getattr(self.canvas, "type_choices_visible", False)
        sort_choices_visible = getattr(self.canvas, "sort_choices_visible", False)
        toolbar.display = not (type_choices_visible or sort_choices_visible)
        sort_labels = dict(MEDIA_SORT_CHOICES)
        current_sort = getattr(self.canvas, "sort_by", "last_modified_desc")
        with toolbar:
            type_filter = Button(
                # task-14902: a chooser-opener, no longer a cycler -- press
                # opens the direct-pick strip below instead of advancing.
                library_choice_label(
                    "type",
                    "All types"
                    if self.canvas.active_type is None
                    else self.canvas.active_type,
                ),
                id="library-media-type-filter",
                classes="library-canvas-action",
                compact=True,
                tooltip=library_choice_tooltip(
                    "media type",
                    tuple(
                        "All types" if value is None else value
                        for value in self.type_options
                    ),
                ),
            )
            yield self._gate_mutation_action(type_filter, str(type_filter.label))
            # task-28013: sort chooser opener -- hidden in select mode like
            # Export/Trash (Select's toolbar acts on the selection).
            sort_btn = Button(
                library_choice_label(
                    "sort", sort_labels.get(current_sort, "Newest")
                ),
                id="library-media-sort",
                classes="library-canvas-action",
                compact=True,
                tooltip=library_choice_tooltip(
                    "the sort order", tuple(label for _, label in MEDIA_SORT_CHOICES)
                ),
            )
            sort_btn.display = not select_mode
            yield self._gate_stale_action(sort_btn, str(sort_btn.label))
            export_btn = Button(
                "Export…",
                id="library-media-export",
                classes="library-canvas-action",
                compact=True,
            )
            export_btn.display = not select_mode
            yield self._gate_stale_action(export_btn, "Export…")
            # task-4025: the browsable Trash surface's entry point -- a
            # plain navigation action (never a `type:` cycle value: `type:`
            # cycles CONTENT types derived from the records, and trash is a
            # STATE). Always enabled: the trash count isn't known until its
            # view fetches, and an empty Trash shows its honest empty copy
            # rather than this button lying disabled. Hidden in select mode
            # like "Export…" -- Select's toolbar is for acting on the
            # selection, not navigating away from it.
            trash_btn = Button(
                "Trash",
                id="library-media-trash-open",
                classes="library-canvas-action",
                compact=True,
                tooltip="Browse and restore deleted media.",
            )
            trash_btn.display = not select_mode
            yield trash_btn
            # task-28242: "Review these" pins the WHOLE filtered result as an
            # ordered review set and walks it in the Reader. A list-level
            # action, hidden in select mode like Export/Trash.
            review_btn = Button(
                "Review these",
                id="library-media-review",
                classes="library-canvas-action",
                compact=True,
                tooltip="Review every item in this list, one by one.",
            )
            review_btn.display = not select_mode
            yield self._gate_stale_action(review_btn, "Review these")
            # Disable only when there's nothing to select AND we're not
            # already in select mode -- in select mode the button is "Done"
            # and must always be pressable so the user can exit even if the
            # rows dropped to zero (e.g. a background snapshot refresh
            # emptied the list).
            select_disabled = rendered_count == 0 and not select_mode
            select_btn = Button(
                # task-4023 AC#1 (RC-07): disabled carries the non-colour
                # "○" marker; the F-018 reason tooltip below says why.
                library_disabled_action_label(
                    "Done" if select_mode else "Select", select_disabled
                ),
                id="library-media-select-toggle",
                classes="library-canvas-action",
                compact=True,
            )
            select_btn.disabled = select_disabled
            if select_disabled:
                select_btn.tooltip = LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP
            yield self._gate_stale_action(
                select_btn, "Done" if select_mode else "Select"
            )
        if type_choices_visible:
            options: list[Option] = []
            highlighted = 0
            for index, value in enumerate(self.type_options):
                display = "All types" if value is None else value
                option = Option(
                    f"✓ {display}" if value == self.canvas.active_type else display,
                    id=f"library-media-type-option-{index}",
                )
                option.choice_value = value
                options.append(option)
                if value == self.canvas.active_type:
                    highlighted = index
            choices = OptionList(
                *options,
                id="library-media-type-choices",
                compact=True,
                markup=False,
            )
            choices.highlighted = highlighted
            choices.styles.height = min(8, max(1, len(options)))
            yield choices
        if sort_choices_visible:
            # task-28013: the sort chooser's direct-pick strip, replacing the
            # toolbar row exactly like the type chooser (shared helper).
            yield from compose_library_choice_strip(
                strip_id="library-media-sort-choices",
                choice_class="library-media-sort-choice",
                options=tuple(
                    (f"library-media-sort-{value}", value, label)
                    for value, label in MEDIA_SORT_CHOICES
                ),
                active_value=current_sort,
            )
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
                # task-4025 AC3 (ADR-055 Pattern A): the confirm copy names
                # the durable recovery path -- the Trash view this task
                # built (the list toolbar's own "Trash" action) -- on top
                # of the receipt's immediate Undo. Supersedes task-4022
                # AC3's honest "there's no Trash view" copy, which was
                # true only until this surface existed.
                item_word = "item" if self.canvas.selected_count == 1 else "items"
                yield Static(
                    f"Delete {self.canvas.selected_count} selected {item_word}? "
                    "You can undo right away, or restore later from Trash.",
                    id="library-media-bulk-delete-confirm-copy",
                    markup=False,
                )
            action_row = Horizontal(classes="ds-toolbar")
            action_row.styles.height = "auto"
            with action_row:
                # Bug found via task-2853's OWN live tmux verification
                # (reproduced against pre-task-8 HEAD too, so it predates
                # this task, and against the Conversations canvas too, the
                # identical pattern -- see review round 2): with no
                # explicit width, this Static resolved as unbounded inside
                # the ``ds-toolbar`` ``Horizontal`` -- live capture showed
                # it claiming ~1700 columns on a 170-column terminal,
                # pushing every sibling Button entirely off-screen
                # (invisible, though still present in the DOM -- which is
                # why headless ``query_one`` pilot tests never caught it).
                # Fixed as the general rule via the shared
                # ``library-toolbar-count`` class (css/components/
                # _agentic_terminal.tcss), not a per-widget Python
                # one-off, so every canvas's counter is covered by one
                # declaration.
                yield Static(
                    f"{self.canvas.selected_count} selected",
                    id="library-media-selected-count",
                    classes="library-toolbar-count",
                    markup=False,
                )
                if confirming_bulk_delete:
                    confirm = Button(
                        "Delete",
                        id="library-media-bulk-delete-confirm",
                        classes="library-canvas-action library-media-action-danger",
                        compact=True,
                    )
                    yield self._gate_stale_action(confirm, "Delete")
                    cancel = Button(
                        "Cancel",
                        id="library-media-bulk-delete-cancel",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(cancel, "Cancel")
                else:
                    select_all = Button(
                        f"Select all {rendered_count} shown",
                        id="library-media-select-all",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_stale_action(
                        select_all, f"Select all {rendered_count} shown"
                    )
                    clear = Button(
                        "Clear",
                        id="library-media-select-clear",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_stale_action(clear, "Clear")
                    export_disabled = self.canvas.selected_count == 0
                    export_selected = Button(
                        # task-4023 AC#1 (RC-07): "○" disabled marker --
                        # these are the very buttons the user entered
                        # Select mode looking for, previously colour-only
                        # at a measured 1.39:1. The base label is stashed
                        # so `_apply_library_row_toggle`'s in-place patch
                        # can rebuild it when the selection count crosses 0.
                        library_disabled_action_label(
                            "Export selected", export_disabled
                        ),
                        id="library-media-export-selected",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    export_selected._library_disabled_marker_base = (
                        "Export selected"
                    )
                    export_selected.disabled = export_disabled
                    # F-018: a disabled action says why.
                    export_selected.tooltip = (
                        LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP
                        if export_selected.disabled
                        else LIBRARY_EXPORT_SELECTED_TOOLTIP
                    )
                    yield self._gate_stale_action(
                        export_selected, "Export selected"
                    )
                    # task-28242: the third real bulk action -- "Review
                    # selected" -- pins the selection as an ordered review set.
                    # Sits between Export and the far-end danger Delete.
                    review_disabled = self.canvas.selected_count == 0
                    review_selected = Button(
                        library_disabled_action_label(
                            "Review selected", review_disabled
                        ),
                        id="library-media-review-selected",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    review_selected._library_disabled_marker_base = (
                        "Review selected"
                    )
                    review_selected.disabled = review_disabled
                    review_selected.tooltip = (
                        "Select items to review them one by one."
                        if review_disabled
                        else "Review the selected items, one by one."
                    )
                    yield self._gate_stale_action(
                        review_selected, "Review selected"
                    )
                    # task-2853: the second real bulk action -- "Delete
                    # selected" -- pushed to the far end (CSS margin, same
                    # library-media-action-danger idiom the single-item
                    # viewer's own Delete uses) so it is never adjacent to
                    # Export selected.
                    delete_disabled = self.canvas.selected_count == 0
                    delete_selected = Button(
                        library_disabled_action_label(
                            "Delete selected", delete_disabled
                        ),
                        id="library-media-delete-selected",
                        classes="library-canvas-action library-media-action-danger",
                        compact=True,
                    )
                    delete_selected._library_disabled_marker_base = (
                        "Delete selected"
                    )
                    delete_selected.disabled = delete_disabled
                    delete_selected.tooltip = (
                        LIBRARY_DELETE_SELECTED_DISABLED_TOOLTIP
                        if delete_selected.disabled
                        else LIBRARY_DELETE_SELECTED_TOOLTIP
                    )
                    yield self._gate_stale_action(
                        delete_selected, "Delete selected"
                    )

        # task-4022 AC2: a completed bulk delete's receipt, naming the
        # count with an Undo affordance right at the point of action --
        # mirrors the ingest queue's own done-row grammar ("✓ done · file
        # · 1s" + a jump action) rather than a toast, which this canvas
        # has none of on the success path today. Rendered OUTSIDE
        # select_mode: a full-success delete exits select mode, so this is
        # the only place left to show it. Uses the same
        # ``library-toolbar-count`` class as "N selected" above -- proven
        # safe for a short Static sharing a ``ds-toolbar`` Horizontal with
        # Buttons (see the comment on that Static; an earlier long-
        # sentence Static in this same row went unbounded and pushed every
        # Button off-screen).
        receipt_count = getattr(self.canvas, "delete_receipt_count", 0)
        if receipt_count:
            receipt_word = "item" if receipt_count == 1 else "items"
            receipt_row = Horizontal(
                classes="ds-toolbar", id="library-media-bulk-delete-receipt"
            )
            receipt_row.styles.height = "auto"
            with receipt_row:
                yield Static(
                    # task-4025 (ADR-055 Pattern A): the receipt names the
                    # durable path too -- "· in Trash" points at the Trash
                    # view that outlives this receipt's Undo/Dismiss.
                    f"✓ deleted · {receipt_count} {receipt_word} · in Trash",
                    id="library-media-bulk-delete-receipt-copy",
                    classes="library-toolbar-count",
                    markup=False,
                )
                undo = Button(
                    "Undo",
                    id="library-media-bulk-delete-undo",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield self._gate_stale_action(undo, "Undo")
                dismiss = Button(
                    "Dismiss",
                    id="library-media-bulk-delete-receipt-dismiss",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield self._gate_mutation_action(dismiss, "Dismiss")

        status_text = (
            self.pager.status_copy
            if self.pager is not None and self.pager.status_copy
            else self.canvas.status_copy or self.canvas.empty_copy
        )
        status = Static(
            status_text,
            id="library-media-status",
            markup=False,
        )
        status.display = bool(status_text)
        yield status

        # task-2853 AC4: while Select mode is active, the preview must never
        # show an item outside the current (multi-item) selection context --
        # ``canvas.selected_id``/``preview_lines`` still carry whatever was
        # focused before Select was entered (the UAT's "bottom preview pane
        # meanwhile shows a previously-selected different item" finding), so
        # the whole block is hidden entirely rather than tracking a second,
        # separate "focused row" concept select mode has no use for.
        has_preview = self.show_preview and (
            not select_mode
            and bool(self.canvas.selected_id and self.canvas.preview_lines)
        )
        self._has_preview = has_preview

        # task-14900: the list and its preview share a workbench container
        # (Collections' `#library-collections-workbench` grammar). Above the
        # screen's one measured width regime it lays them out side by side
        # (this Horizontal's default); below it, the host's existing
        # `library-notes-compact` class gives the list the full canvas and
        # suppresses the preview via CSS -- the conditional is keyed off a
        # class the screen already maintains at compose time AND on every
        # resize crossing, so no compose branch here can drift from an
        # in-place updater. Geometry (heights/overflow) moved from inline
        # styles into the same CSS tiers, because inline styles outrank the
        # class-flipped rules.
        workbench = Horizontal(id="library-media-workbench")
        workbench.set_class(has_preview, "has-preview")
        with workbench:
            media_list = Vertical(id="library-media-list")
            with media_list:
                with LibraryMediaRowScroll(id="library-media-row-scroll"):
                    row_height = (
                        _MEDIA_ROW_COMPACT_HEIGHT
                        if self.compact
                        else _MEDIA_ROW_WIDE_HEIGHT
                    )
                    for index, row in enumerate(self.canvas.rows):
                        if select_mode:
                            marker = "☑" if row.checked else "☐"
                        else:
                            marker = "▸" if row.selected and not self.compact else " "
                        # task-281 (PR #665 review): the in-place toggle needs the
                        # marker-less RAW label to rebuild from -- reading it back
                        # off the mounted Button un-escapes user titles (both
                        # ``.plain`` and Textual 8's ``str(Content)`` return
                        # rendered text), so the raw remainder is stashed here at
                        # the single point of truth.
                        label_rest = _media_row_label_rest(
                            row.title,
                            row.secondary,
                            compact=self.compact,
                            loading=row.loading,
                            loaded=row.loaded,
                        )
                        button = Button(
                            f"{marker}{label_rest}",
                            id=f"library-media-row-{index}",
                            classes="library-media-row",
                            compact=True,
                        )
                        button.media_id = row.media_id
                        button._library_row_label_rest = label_rest
                        button._library_media_title = row.title
                        button._library_media_secondary = row.secondary
                        button._library_media_selected = row.selected
                        button._library_media_checked = row.checked
                        button._library_media_loading = row.loading
                        button._library_media_loaded = row.loaded
                        button.tooltip = escape_markup(row.title)
                        button.set_class(
                            row.selected and not self.compact and not select_mode,
                            "library-media-row-selected",
                        )
                        button.styles.height = row_height
                        button.styles.min_height = row_height
                        yield self._gate_stale_action(button, label_rest.lstrip())
                if self.pager is not None:
                    yield from self._compose_pager(self.pager)

            preview = Vertical(id="library-media-preview")
            preview.display = has_preview and not self.compact
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
                    open_viewer = Button(
                        "Open in viewer",
                        id="library-media-open-viewer",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    open_viewer.can_focus = self.show_preview and not self.compact
                    yield self._gate_stale_action(open_viewer, "Open in viewer")

            # task-14900: the wide split's detail half never sits blank --
            # when the preview is hidden (Select mode, or an empty list) a
            # placeholder explains the pane, Collections' own detail-pane
            # grammar ("No Collection selected."). CSS-only visibility
            # (never a Python ``display`` write, which would outrank the
            # compact rule that hides it in the preserved stacked layout):
            # hidden while the workbench carries ``has-preview``, and hidden
            # entirely below the breakpoint.
            detail_empty = Static(
                (
                    "No preview in Select mode."
                    if select_mode
                    else "No media item selected."
                ),
                id="library-media-detail-empty",
                markup=False,
            )
            detail_empty.display = self.show_preview
            yield detail_empty

    def _compose_pager(self, pager: LibraryPagerDisplay) -> ComposeResult:
        """Render the controller-owned Media pager below the row viewport."""
        # task-28016: a single-page result has nowhere to page to, so the
        # "Page 1 of 1" counter and the boundary reasons ("Already on the
        # first page.", "No more results.") are pure noise. Show only the item
        # range and keep the (disabled) controls; both return the moment a
        # second page exists.
        disabled_reasons = (
            ()
            if pager.single_page
            else tuple(
                dict.fromkeys(
                    reason
                    for disabled, reason in (
                        (pager.previous_disabled, pager.previous_reason),
                        (pager.next_disabled, pager.next_reason),
                    )
                    if disabled and reason
                )
            )
        )
        status_parts = (
            (pager.range_copy,)
            if pager.single_page
            else (pager.range_copy, pager.page_copy)
        )
        with Vertical(id="library-media-pager", classes="library-source-pager"):
            yield Static(
                " · ".join(copy for copy in status_parts if copy),
                id="library-media-page-status",
                classes="library-source-pager-status",
                markup=False,
            )
            if disabled_reasons:
                yield Static(
                    " · ".join(disabled_reasons),
                    id="library-media-disabled-reason",
                    classes="library-source-pager-status",
                    markup=False,
                )
            with Horizontal(classes="library-source-pager-controls"):
                previous = Button(
                    library_disabled_action_label(
                        "Previous", pager.previous_disabled
                    ),
                    id="library-media-previous",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=pager.previous_disabled,
                )
                if pager.previous_disabled:
                    previous.tooltip = pager.previous_reason
                yield self._gate_mutation_action(previous, "Previous")
                if pager.retry_visible:
                    retry = Button(
                        "Retry",
                        id="library-media-retry",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(retry, "Retry")
                next_page = Button(
                    library_disabled_action_label("Next", pager.next_disabled),
                    id="library-media-next",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=pager.next_disabled,
                )
                if pager.next_disabled:
                    next_page.tooltip = pager.next_reason
                yield self._gate_mutation_action(next_page, "Next")
