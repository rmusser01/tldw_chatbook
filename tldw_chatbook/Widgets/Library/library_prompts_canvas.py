"""Library prompts canvas: list mode (rows + filter + sort).

Structural template copy of ``library_notes_canvas.py``'s list-view
``compose`` -- prompts and notes diverge (two-part editor, no sync), so only
the list shape (header count line, filter Input, fixed-button
``ds-toolbar`` rows, row Buttons with escaped labels) is mirrored here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, Input, Static, TextArea

from tldw_chatbook.Library.library_prompts_state import (
    LibraryPromptDeleteReceipt,
    PromptBrowseResult,
    PromptEditorState,
    PromptHistoryState,
    PromptMembershipState,
    PromptsListState,
    definition_state_display_label,
    prompt_editor_meta_line,
)
from tldw_chatbook.Library.library_pager_state import LibraryPagerDisplay
from tldw_chatbook.Library.library_shell_state import (
    library_choice_label,
    library_choice_tooltip,
    library_disabled_action_label,
)
from tldw_chatbook.Prompt_Management.prompt_batch_models import (
    PromptBatchDeleteResult,
)
from tldw_chatbook.Widgets.Library.library_choice_strip import (
    compose_library_choice_strip,
)
from tldw_chatbook.UI.Library_Modules.prompt_history_region import (
    LibraryPromptHistoryRegion,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import PromptBlockEditor
from tldw_chatbook.Widgets.Prompts.prompt_block_editor_state import (
    PromptBlockEditorState,
)
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)

_SORT_LABELS = {"newest": "Newest", "name": "Name"}
_EMPTY_PROMPTS_COPY = "No prompts yet."
_EMPTY_PROMPTS_FILTER_COPY = "No prompts match your filter."
_EMPTY_PROMPT_LIBRARY_COPY = "No prompts yet. Create or import a prompt to begin."
_EMPTY_PROMPT_COLLECTION_COPY = (
    "This collection has no prompts. Choose another collection or add prompts."
)
# Task 8c U7: one-line dim hints under the System/User prompt labels,
# explaining the two-part prompt model to a new user.
_SYSTEM_PROMPT_HINT = "Instructions the model always follows."
_USER_PROMPT_HINT = "The message inserted into the composer."
_NOTHING_TO_SELECT = "Nothing here to select yet."
_SELECT_FIRST = "Select one or more items first."
_SELECTION_EMPTY_REASON = "Select one or more items to use bulk actions."
_PAGE_UNAVAILABLE = "Current page is unavailable."
_PAGE_UNAVAILABLE_REASON = (
    "Current page is unavailable; selected items remain available for Export or Delete."
)
_MUTATION_PROGRESS = "Updating selected items…"
_STALE_PAGE_ACTIONS = "List may be out of date. Retry or change the scope."
PROMPT_DISCARD_TOOLTIP_CLEAN = "No unsaved Prompt changes to discard."
PROMPT_DISCARD_TOOLTIP_DIRTY = "Return to the Prompt list without saving these changes."
PROMPT_DISCARD_TOOLTIP_BUSY = (
    "Prompt changes are still in progress. Try again when they finish."
)


def _compact_receipt_name(value: str, limit: int = 42) -> str:
    """Keep an untrusted artifact name literal and bounded in the action row."""
    normalized = " ".join(value.splitlines()).strip() or "Untitled"
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


class LibraryPromptsListCanvas(PostRecomposeCallback, Vertical):
    """Render the Library prompts canvas: the list view, or the prompt editor.

    Attributes:
        state: List-view display state (rows, count, sort). ``None``
            renders nothing (mirrors ``LibraryNotesCanvas``'s guard for a
            not-yet-available list state). Only used when ``mode == "list"``.
        sort_mode: Current prompts sort mode key (``"newest"``/``"name"``),
            used to label the sort control.
        filter_value: Current prompts filter text, prefilled into the
            filter ``Input``.
        browse_result: Exact immutable service-backed page and request state.
            When omitted, the legacy ``state``-only rendering remains available
            to existing widget callers; the Library screen always supplies it.
        mode: ``"list"`` renders the prompts list; ``"editor"`` renders the
            in-canvas prompt editor for ``editor_state``.
        editor_state: The prompt to render in editor mode. Required when
            ``mode == "editor"``.
        conflict: When ``True`` (editor mode only), renders the save
            conflict banner -- a quiet explanatory line plus Save-as-new/
            Reload actions -- in place of the normal action row. Mirrors
            ``LibraryNotesCanvas.conflict``. ``editor_state`` must already
            reflect the user's kept text (never the stale server detail)
            when this is set.
        status: Save-outcome status text shown below the meta line (e.g.
            ``"Saved."`` or a name-conflict explanation), or ``""`` when
            idle. Not shown while ``conflict`` is set -- the conflict
            banner communicates the outcome instead.
        show_open_existing: Editor mode only. When ``True``, renders the
            "Open existing" action (Task 8b D3) directly under the status
            line -- shown only while ``status`` is the name-in-use outcome,
            giving that status copy's "...or open the existing prompt" a
            real affordance. Never shown together with ``conflict``.
        dirty: Editor mode only (Task 8c U6). Whether the open prompt has
            unsaved in-progress edits -- threaded into the meta line's
            trailing "Unsaved changes" marker via ``prompt_editor_meta_line``
            on this initial compose. Per-keystroke updates never recompose
            this widget at all (the screen updates ``#library-prompt-meta``
            in place instead -- see
            ``LibraryScreen._update_library_prompt_meta_static``); this
            constructor argument only matters for the handful of flows that
            already do a full recompose while dirty (initial load, Duplicate,
            conflict entry/resolution).
        write_in_flight: Editor mode only. Keeps Discard disabled while an
            admitted Prompt writer may still persist the working copy.
        import_open: List-view only. When ``True``, renders the inline
            Import row (a path ``Input`` for a file OR folder, plus
            Import/Cancel actions) below the Sort/Import…/Export… toolbar.
        import_path: The Import row's path ``Input`` prefilled value.
            Only meaningful while ``import_open`` is ``True``.
        import_status: Muted outcome line shown below the Import row
            (e.g. ``"2 imported · 1 skipped (duplicate name)"``), or
            ``""`` when idle/not yet run.
    """

    def __init__(
        self,
        state: PromptsListState | None = None,
        *,
        sort_mode: str = "newest",
        filter_value: str = "",
        browse_result: PromptBrowseResult | None = None,
        pager: LibraryPagerDisplay | None = None,
        mode: str = "list",
        editor_state: PromptEditorState | None = None,
        conflict: bool = False,
        status: str = "",
        show_open_existing: bool = False,
        import_open: bool = False,
        import_path: str = "",
        import_status: str = "",
        mutation_status: str = "",
        dirty: bool = False,
        can_update_original: bool = False,
        include_starter_content: bool = False,
        history_state: PromptHistoryState | None = None,
        history_current_compatible: bool = True,
        collection_label: str = "All prompts",
        membership_state: PromptMembershipState | None = None,
        sort_choices_visible: bool = False,
        delete_receipt: LibraryPromptDeleteReceipt
        | PromptBatchDeleteResult
        | None = None,
        mutation_in_flight: bool = False,
        page_actions_disabled: bool = False,
        write_in_flight: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.sort_mode = sort_mode
        self.sort_choices_visible = sort_choices_visible
        self.filter_value = filter_value
        self.browse_result = browse_result
        self.pager = pager
        self.mode = mode
        self.editor_state = editor_state
        self.conflict = conflict
        self.status = status
        self.show_open_existing = show_open_existing
        self.import_open = import_open
        self.import_path = import_path
        self.import_status = import_status
        self.mutation_status = mutation_status
        self.dirty = dirty
        self.can_update_original = can_update_original
        self.include_starter_content = include_starter_content
        self.history_state = history_state
        self.history_current_compatible = history_current_compatible
        self.collection_label = collection_label
        self.membership_state = membership_state
        self.delete_receipt = delete_receipt
        self.mutation_in_flight = mutation_in_flight
        self.page_actions_disabled = page_actions_disabled
        self.write_in_flight = write_in_flight
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        if self.mode == "loading":
            yield Static(
                "Loading prompt…",
                id="library-prompt-loading",
                classes="destination-purpose",
                markup=False,
            )
            return
        if self.mode == "editor":
            yield from self._compose_editor()
            return
        yield from self._compose_list()

    def sync_state(
        self,
        *,
        state: PromptsListState | None,
        sort_mode: str,
        filter_value: str,
        browse_result: PromptBrowseResult | None,
        pager: LibraryPagerDisplay | None,
        mode: str,
        editor_state: PromptEditorState | None,
        conflict: bool,
        status: str,
        show_open_existing: bool,
        import_open: bool,
        import_path: str,
        import_status: str,
        dirty: bool,
        can_update_original: bool,
        include_starter_content: bool,
        history_state: PromptHistoryState | None,
        history_current_compatible: bool,
        collection_label: str,
        membership_state: PromptMembershipState | None,
        sort_choices_visible: bool,
        page_actions_disabled: bool,
    ) -> None:
        """Apply a complete prompt snapshot within the mounted canvas.

        Args:
            state: Prompt list snapshot, or ``None`` outside list mode.
            sort_mode: Active prompt sort identifier.
            filter_value: Current prompt filter text.
            browse_result: Paginated prompt browse result.
            pager: Controller-derived Prompt pager presentation.
            mode: Canvas surface to render.
            editor_state: Prompt editor snapshot.
            conflict: Whether the open prompt has an edit conflict.
            status: Current prompt editor status copy.
            show_open_existing: Whether to offer opening an existing prompt.
            import_open: Whether the prompt import form is expanded.
            import_path: Current prompt import path.
            import_status: Current prompt import outcome copy.
            dirty: Whether the prompt editor has unsaved changes.
            can_update_original: Whether save may update the original prompt.
            include_starter_content: Whether create mode uses starter content.
            history_state: Prompt version-history snapshot.
            history_current_compatible: Whether history matches the open prompt.
            collection_label: Collection membership label for the prompt.
            membership_state: Prompt collection-membership snapshot.
            sort_choices_visible: Whether the sort chooser is expanded.
            page_actions_disabled: Whether stale retained rows and bulk actions
                are read-only until an authoritative refresh succeeds.
        """
        self.state = state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.browse_result = browse_result
        self.pager = pager
        self.mode = mode
        self.editor_state = editor_state
        self.conflict = conflict
        self.status = status
        self.show_open_existing = show_open_existing
        self.import_open = import_open
        self.import_path = import_path
        self.import_status = import_status
        self.dirty = dirty
        self.can_update_original = can_update_original
        self.include_starter_content = include_starter_content
        self.history_state = history_state
        self.history_current_compatible = history_current_compatible
        self.collection_label = collection_label
        self.membership_state = membership_state
        self.sort_choices_visible = sort_choices_visible
        self.page_actions_disabled = page_actions_disabled
        self.refresh(recompose=True)

    def _compose_list(self) -> ComposeResult:
        state = self.state
        if state is None:
            return
        browse_result = self.browse_result
        pager = self.pager
        total: int | str | None = pager.title_count if pager is not None else state.count
        if pager is None and browse_result is not None:
            total = (
                "…"
                if browse_result.status in {"loading", "error"}
                else browse_result.total_items
            )
        yield Static(
            "Prompts" if total is None else f"Prompts ({total})",
            id="library-prompts-header",
            classes="destination-section",
            markup=False,
        )
        fresh_empty_status = (
            browse_result.status
            if browse_result is not None
            and pager is not None
            and pager.title_count == 0
            and not state.rows
            and not state.select_mode
            and self.delete_receipt is None
            and not self.mutation_in_flight
            and not self.mutation_status
            and not self.import_open
            and not pager.status_copy
            and not pager.retry_visible
            else None
        )
        if fresh_empty_status in {"empty_library", "empty_collection", "no_matches"}:
            if fresh_empty_status == "no_matches":
                yield Input(
                    placeholder="Filter prompts… (Enter)",
                    id="library-prompts-filter",
                    value=self.filter_value,
                )
                empty_copy = (
                    f'No prompts match "{browse_result.scope.query}". '
                    "Clear the search or try different words."
                )
                action_label = "Clear filter"
                action_id = "library-prompts-empty-clear-filter"
            elif fresh_empty_status == "empty_collection":
                yield Static(
                    library_choice_label("collection", self.collection_label),
                    id="library-prompts-empty-collection-label",
                    markup=False,
                )
                empty_copy = _EMPTY_PROMPT_COLLECTION_COPY
                action_label = "All prompts"
                action_id = "library-prompts-empty-all-prompts"
            else:
                empty_copy = _EMPTY_PROMPT_LIBRARY_COPY
                action_label = "New prompt"
                action_id = "library-prompts-empty-new"
            yield Static(
                empty_copy,
                id="library-prompts-empty",
                markup=False,
            )
            yield Button(
                action_label,
                id=action_id,
                classes=(
                    "library-canvas-action console-action-primary"
                    if fresh_empty_status == "empty_library"
                    else "library-canvas-action"
                ),
                compact=True,
            )
            if fresh_empty_status == "empty_library":
                yield Button(
                    "Import…",
                    id="library-prompts-import",
                    classes="library-canvas-action",
                    compact=True,
                )
            return
        if self.delete_receipt is not None:
            receipt = self.delete_receipt
            if isinstance(receipt, PromptBatchDeleteResult):
                if len(receipt.entries) == 1:
                    entry = receipt.entries[0]
                    receipt_copy = (
                        f"✓ deleted · {entry.artifact_type.title()} · "
                        f"{_compact_receipt_name(entry.title)}"
                    )
                else:
                    receipt_copy = f"✓ deleted · {len(receipt.entries)} items"
            else:
                receipt_copy = (
                    f"✓ deleted · {receipt.artifact_type.title()} · "
                    f"{_compact_receipt_name(receipt.title)}"
                )
            receipt_copy_row = Horizontal(
                id="library-prompts-delete-receipt", classes="ds-toolbar"
            )
            receipt_copy_row.styles.height = "auto"
            with receipt_copy_row:
                yield Static(
                    receipt_copy,
                    id="library-prompts-delete-receipt-copy",
                    classes="library-toolbar-count",
                    markup=False,
                )
            receipt_actions = Horizontal(classes="ds-toolbar")
            receipt_actions.styles.height = "auto"
            with receipt_actions:
                yield Button(
                    library_disabled_action_label("Undo", self.mutation_in_flight),
                    id="library-prompts-delete-undo",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=self.mutation_in_flight,
                    tooltip=(_MUTATION_PROGRESS if self.mutation_in_flight else None),
                )
                yield Button(
                    library_disabled_action_label("Dismiss", self.mutation_in_flight),
                    id="library-prompts-delete-receipt-dismiss",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=self.mutation_in_flight,
                    tooltip=(_MUTATION_PROGRESS if self.mutation_in_flight else None),
                )
        yield Input(
            placeholder="Filter prompts… (Enter)",
            id="library-prompts-filter",
            value=self.filter_value,
            disabled=self.mutation_in_flight,
            tooltip=(_MUTATION_PROGRESS if self.mutation_in_flight else None),
        )
        collection_label = library_choice_label(
            "collection", escape_markup(self.collection_label)
        )
        yield Button(
            # task-14902: a chooser, not a cycler -- pressing opens the
            # collection manager modal (browse lane: the full, unbounded
            # collection set with a direct pick), so the label must not
            # carry the press-advances "⇄" glyph.
            library_disabled_action_label(collection_label, self.mutation_in_flight),
            id="library-prompts-collection",
            classes="library-canvas-action",
            compact=True,
            disabled=self.mutation_in_flight,
            # The collection set is user-data (dynamic), so the tooltip
            # names the pick's shape rather than a stale enumeration.
            tooltip=(
                _MUTATION_PROGRESS
                if self.mutation_in_flight
                else "Press to pick the prompt scope: All prompts, or one collection."
            ),
        )
        page_unavailable = bool(
            browse_result is not None and browse_result.status in {"loading", "error"}
        )
        page_actions_disabled = (
            self.mutation_in_flight or self.page_actions_disabled
        )
        page_action_reason = (
            _MUTATION_PROGRESS
            if self.mutation_in_flight
            else _STALE_PAGE_ACTIONS
            if self.page_actions_disabled
            else None
        )
        page_selectable = bool(state.rows) and not page_unavailable
        selection_reason = ""
        if state.select_mode:
            yield Static(
                f"{state.total_selected} selected · "
                f"{state.selected_on_page} on this page",
                id="library-prompts-selection-summary",
                classes="library-toolbar-count",
                markup=False,
            )
            select_page_disabled = not page_selectable or page_actions_disabled
            zero_selection = state.total_selected == 0
            clear_disabled = zero_selection or self.mutation_in_flight
            selection_disabled = zero_selection or page_actions_disabled
            management_toolbar = Horizontal(classes="ds-toolbar")
            management_toolbar.styles.height = "auto"
            with management_toolbar:
                yield Button(
                    library_disabled_action_label("Select page", select_page_disabled),
                    id="library-prompts-select-page",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=select_page_disabled,
                    tooltip=(
                        page_action_reason
                        if page_actions_disabled
                        else (
                            _PAGE_UNAVAILABLE
                            if page_unavailable
                            else _NOTHING_TO_SELECT
                        )
                        if select_page_disabled
                        else None
                    ),
                )
                yield Button(
                    library_disabled_action_label("Clear all", clear_disabled),
                    id="library-prompts-clear-selection",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=clear_disabled,
                    tooltip=(
                        _MUTATION_PROGRESS
                        if self.mutation_in_flight
                        else _SELECT_FIRST
                        if clear_disabled
                        else None
                    ),
                )
            done_toolbar = Horizontal(classes="ds-toolbar")
            done_toolbar.styles.height = "auto"
            with done_toolbar:
                yield Button(
                    library_disabled_action_label("Done", self.mutation_in_flight),
                    id="library-prompts-selection-done",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=self.mutation_in_flight,
                    tooltip=(_MUTATION_PROGRESS if self.mutation_in_flight else None),
                )
            for label, action_id in (
                ("Export selected", "library-prompts-export-selected"),
                ("Delete selected", "library-prompts-delete-selected"),
            ):
                selection_toolbar = Horizontal(classes="ds-toolbar")
                selection_toolbar.styles.height = "auto"
                with selection_toolbar:
                    yield Button(
                        library_disabled_action_label(label, selection_disabled),
                        id=action_id,
                        classes="library-canvas-action",
                        compact=True,
                        disabled=selection_disabled,
                        tooltip=(
                            page_action_reason
                            if page_actions_disabled
                            else _SELECT_FIRST
                            if selection_disabled
                            else None
                        ),
                    )
            if not self.mutation_in_flight:
                if zero_selection:
                    selection_reason = _SELECTION_EMPTY_REASON
                elif page_unavailable:
                    selection_reason = _PAGE_UNAVAILABLE_REASON
                elif not state.rows:
                    selection_reason = _NOTHING_TO_SELECT
        else:
            select_disabled = not page_selectable or page_actions_disabled
            sort_label = library_choice_label(
                "sort", _SORT_LABELS.get(self.sort_mode, "Newest")
            )
            management_toolbar = Horizontal(classes="ds-toolbar")
            management_toolbar.styles.height = "auto"
            # task-14902: the sort choice strip replaces only this row;
            # Import/Export remain available below it.
            management_toolbar.display = not self.sort_choices_visible
            with management_toolbar:
                yield Button(
                    library_disabled_action_label(sort_label, self.mutation_in_flight),
                    id="library-prompts-sort",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=self.mutation_in_flight,
                    tooltip=(
                        _MUTATION_PROGRESS
                        if self.mutation_in_flight
                        else library_choice_tooltip(
                            "the sort order", tuple(_SORT_LABELS.values())
                        )
                    ),
                )
                yield Button(
                    library_disabled_action_label("Select", select_disabled),
                    id="library-prompts-select",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=select_disabled,
                    tooltip=(
                        page_action_reason
                        if page_actions_disabled
                        else _NOTHING_TO_SELECT
                        if select_disabled
                        else None
                    ),
                )
            if self.sort_choices_visible:
                yield from compose_library_choice_strip(
                    strip_id="library-prompts-sort-choices",
                    choice_class="library-prompts-sort-choice",
                    options=tuple(
                        (f"library-prompts-sort-{mode}", mode, label)
                        for mode, label in _SORT_LABELS.items()
                    ),
                    active_value=self.sort_mode,
                    disabled=self.mutation_in_flight,
                )
            transfer_toolbar = Horizontal(classes="ds-toolbar")
            transfer_toolbar.styles.height = "auto"
            with transfer_toolbar:
                for label, action_id in (
                    ("Import…", "library-prompts-import"),
                    ("Export…", "library-prompts-export"),
                ):
                    disabled = self.mutation_in_flight or (
                        self.page_actions_disabled
                        and action_id == "library-prompts-export"
                    )
                    yield Button(
                        library_disabled_action_label(label, disabled),
                        id=action_id,
                        classes="library-canvas-action",
                        compact=True,
                        disabled=disabled,
                        tooltip=(
                            page_action_reason if disabled else None
                        ),
                    )
            if not page_actions_disabled and select_disabled:
                selection_reason = _NOTHING_TO_SELECT
        if self.mutation_in_flight:
            yield Static(
                _MUTATION_PROGRESS,
                id="library-prompts-mutation-progress",
                classes="destination-purpose",
                markup=False,
            )
        elif self.mutation_status:
            yield Static(
                self.mutation_status,
                id="library-prompts-mutation-status",
                classes="destination-purpose",
                markup=False,
            )
        elif selection_reason:
            yield Static(
                selection_reason,
                id="library-prompts-selection-reason",
                classes="destination-purpose",
                markup=False,
            )
        if self.import_open and not state.select_mode:
            yield from self._compose_import_row()
        if (
            pager is None
            and browse_result is not None
            and browse_result.status == "loading"
        ):
            yield Static(
                "Loading prompts…",
                id="library-prompts-loading",
                classes="destination-purpose",
                markup=False,
            )
            return
        if (
            pager is None
            and browse_result is not None
            and browse_result.status == "error"
        ):
            yield Static(
                browse_result.error,
                id="library-prompts-error",
                classes="destination-purpose",
                markup=False,
            )
            yield Button(
                library_disabled_action_label("Retry", self.mutation_in_flight),
                id="library-prompts-retry",
                classes="library-canvas-action",
                compact=True,
                disabled=self.mutation_in_flight,
                tooltip=(_MUTATION_PROGRESS if self.mutation_in_flight else None),
            )
            return
        if (
            pager is None
            and browse_result is not None
            and browse_result.total_pages > 1
        ):
            yield from self._compose_paging(browse_result)
        if not state.rows:
            if pager is not None and pager.title_count is None:
                yield from self._compose_pager(pager)
                return
            if browse_result is not None:
                if browse_result.status == "empty_collection":
                    empty_copy = _EMPTY_PROMPT_COLLECTION_COPY
                elif browse_result.status == "no_matches":
                    empty_copy = (
                        f'No prompts match "{browse_result.scope.query}". '
                        "Clear the search or try different words."
                    )
                else:
                    empty_copy = _EMPTY_PROMPT_LIBRARY_COPY
            else:
                empty_copy = (
                    _EMPTY_PROMPTS_FILTER_COPY
                    if self.filter_value
                    else _EMPTY_PROMPTS_COPY
                )
            yield Static(
                empty_copy,
                id="library-prompts-empty",
                markup=False,
            )
            if pager is not None:
                yield from self._compose_pager(pager)
            return
        with VerticalScroll(id="library-prompts-list"):
            for row in state.rows:
                # Button labels are parsed as Rich markup: escape the
                # user-supplied name AND secondary line (details/description)
                # so "[draft] Q3 plan [wip]" renders verbatim instead of
                # eating bracketed segments as tags (or crashing on an
                # unmatched closing tag) -- same fix class as the notes list
                # row / search-history Button labels. The secondary line is
                # equally user-controlled (the prompt's free-text
                # description) and must be escaped too, not just the name.
                name = escape_markup(row.name)
                secondary = escape_markup(row.secondary) if row.secondary else ""
                artifact_summary = escape_markup(
                    f"{row.type_label} · {row.source_label} · {row.lane_summary}"
                )
                selection_prefix = ""
                if state.select_mode:
                    selection_prefix = "☑ " if row.checked else "☐ "
                label_parts = (
                    f"{selection_prefix}{name}",
                    artifact_summary,
                    secondary,
                )
                button = Button(
                    library_disabled_action_label(
                        "\n".join(part for part in label_parts if part),
                        page_actions_disabled,
                    ),
                    id=f"library-prompt-row-{row.prompt_id}",
                    classes="library-prompt-row",
                    compact=True,
                    disabled=page_actions_disabled,
                    tooltip=page_action_reason,
                )
                button.prompt_id = row.prompt_id
                button.prompt_version = row.version
                button.artifact_type = row.artifact_type
                button.prompt_name = row.name
                yield button
        if pager is not None:
            yield from self._compose_pager(pager)

    def _compose_pager(self, pager: LibraryPagerDisplay) -> ComposeResult:
        """Render the controller-derived Prompt pager without recalculation."""
        with Vertical(id="library-prompts-pager"):
            copy = " · ".join(part for part in (pager.range_copy, pager.page_copy) if part)
            yield Static(
                copy,
                id="library-prompts-page-label",
                markup=False,
            )
            reasons = tuple(
                dict.fromkeys(
                    reason
                    for reason in (pager.previous_reason, pager.next_reason)
                    if reason
                )
            )
            yield Static(
                " · ".join((pager.status_copy, *reasons)).strip(" ·"),
                id="library-prompts-page-status",
                classes="destination-purpose",
                markup=False,
            )
            previous_disabled = pager.previous_disabled or self.mutation_in_flight
            next_disabled = pager.next_disabled or self.mutation_in_flight
            toolbar = Horizontal(classes="ds-toolbar")
            toolbar.styles.height = "auto"
            with toolbar:
                yield Button(
                    library_disabled_action_label("Previous", previous_disabled),
                    id="library-prompts-page-previous",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=previous_disabled,
                    tooltip=(
                        _MUTATION_PROGRESS
                        if self.mutation_in_flight
                        else pager.previous_reason or None
                    ),
                )
                yield Button(
                    library_disabled_action_label("Next", next_disabled),
                    id="library-prompts-page-next",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=next_disabled,
                    tooltip=(
                        _MUTATION_PROGRESS
                        if self.mutation_in_flight
                        else pager.next_reason or None
                    ),
                )
            if pager.retry_visible:
                yield Button(
                    library_disabled_action_label("Retry", self.mutation_in_flight),
                    id="library-prompts-retry",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=self.mutation_in_flight,
                    tooltip=(
                        _MUTATION_PROGRESS if self.mutation_in_flight else None
                    ),
                )

    def _compose_paging(self, result: PromptBrowseResult) -> ComposeResult:
        """Render the minimal bounded page controls required for exact browse."""
        first = (result.page - 1) * result.scope.page_size + 1
        last = min(result.total_items, first + len(result.items) - 1)
        page_label = Static(
            f"Page {result.page} of {result.total_pages} · "
            f"showing {first}–{last} of {result.total_items}",
            id="library-prompts-page-label",
            markup=False,
        )
        page_label.styles.height = "auto"
        yield page_label
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                library_disabled_action_label("Previous", self.mutation_in_flight),
                id="library-prompts-page-previous",
                classes="library-canvas-action",
                compact=True,
                disabled=result.page <= 1 or self.mutation_in_flight,
                tooltip=(_MUTATION_PROGRESS if self.mutation_in_flight else None),
            )
            yield Button(
                library_disabled_action_label("Next", self.mutation_in_flight),
                id="library-prompts-page-next",
                classes="library-canvas-action",
                compact=True,
                disabled=result.page >= result.total_pages or self.mutation_in_flight,
                tooltip=(_MUTATION_PROGRESS if self.mutation_in_flight else None),
            )

    def _compose_import_row(self) -> ComposeResult:
        """Render the inline Import row: a path Input, then a Run/Cancel
        action toolbar, then the outcome line.

        The path ``Input`` is its own full-width sibling -- NOT packed into
        a ``Horizontal`` alongside the action Buttons -- mirroring
        ``LibraryIngestCanvas``'s documented render-safe shape for this
        canvas family: a ``Horizontal`` mixing a 1fr-width Input with
        fixed-width compact Buttons is this family's known non-rendering
        failure mode. The Run/Cancel Buttons instead get their own
        ``ds-toolbar`` row underneath, the same fixed-width-only shape as
        the sort/Import… toolbar above.
        """
        yield Input(
            placeholder="File or folder path…",
            id="library-prompts-import-path",
            value=self.import_path,
            disabled=self.mutation_in_flight,
        )
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            # Task 8b D4: Browse… picks a FILE via the same FileOpen dialog
            # the media-ingest form's Browse action uses -- that dialog has
            # no directory-selection mode, so a folder import still has to
            # be typed by hand into the path Input above; this only covers
            # the file case (see handle_library_prompts_import_browse).
            yield Button(
                library_disabled_action_label("Browse…", self.mutation_in_flight),
                id="library-prompts-import-browse",
                classes="library-canvas-action",
                compact=True,
                disabled=self.mutation_in_flight,
                tooltip=(_MUTATION_PROGRESS if self.mutation_in_flight else None),
            )
            yield Button(
                library_disabled_action_label("Import", self.mutation_in_flight),
                id="library-prompts-import-run",
                classes="library-canvas-action",
                compact=True,
                disabled=self.mutation_in_flight,
                tooltip=(_MUTATION_PROGRESS if self.mutation_in_flight else None),
            )
            yield Button(
                library_disabled_action_label("Cancel", self.mutation_in_flight),
                id="library-prompts-import-cancel",
                classes="library-canvas-action",
                compact=True,
                disabled=self.mutation_in_flight,
                tooltip=(_MUTATION_PROGRESS if self.mutation_in_flight else None),
            )
        yield Static(
            self.import_status,
            id="library-prompts-import-status",
            markup=False,
        )

    def _compose_editor(self) -> ComposeResult:
        """Render the prompt editor with a scrolling body and fixed actions.

        The shell has one vertical scroll owner for the editable content and
        an intrinsically sized action region below it. This keeps the action
        controls reachable without covering the final editor field.

        Field order (Task 8b U2): Name, Description, System prompt, User
        prompt, Keywords, Author -- Author is demoted from 2nd position to
        last, beside Keywords (ids unchanged; only compose order moved).

        Task 8c: each of System prompt/User prompt now gets a one-line dim
        hint Static right under its label (U7); the meta line's initial
        render threads ``self.dirty`` through so a full recompose while
        dirty (initial load, Duplicate, conflict entry/resolution) still
        shows the unsaved marker (U6 -- per-keystroke dirty updates instead
        patch ``#library-prompt-meta`` in place, never recomposing this
        widget).
        """
        editor_state = self.editor_state
        if editor_state is None:
            return
        with Vertical(id="library-prompt-editor-shell"):
            with VerticalScroll(id="library-prompt-editor-content"):
                if self.mutation_in_flight:
                    yield Static(
                        _MUTATION_PROGRESS,
                        id="library-prompts-mutation-progress",
                        classes="destination-purpose",
                        markup=False,
                    )
                yield Button(
                    library_disabled_action_label(
                        "‹ Back to list", self.mutation_in_flight
                    ),
                    id="library-prompt-back",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=self.mutation_in_flight,
                )
                yield Static("Name", classes="library-prompt-field-label", markup=False)
                yield Input(
                    value=editor_state.name,
                    id="library-prompt-name",
                    disabled=self.mutation_in_flight,
                )
                # Task 8b U4: rendered label only -- the DB/record field name
                # (``details``, ``#library-prompt-details``) is untouched.
                yield Static(
                    "Description", classes="library-prompt-field-label", markup=False
                )
                yield Input(
                    value=editor_state.details,
                    id="library-prompt-details",
                    disabled=self.mutation_in_flight,
                )
                yield Static(
                    (
                        f"{editor_state.artifact_type.title()} · "
                        f"{editor_state.source.title()} · "
                        f"{definition_state_display_label(editor_state.definition_state)}"
                    ),
                    id="library-prompt-artifact-status",
                    classes="destination-purpose",
                    markup=False,
                )
                block_state = editor_state.block_editor_state
                if block_state is not None:
                    block_editor = PromptBlockEditor(
                        block_state,
                        can_update_original=self.can_update_original,
                        allow_apply_system=False,
                        apply_system_unavailable_reason=(
                            "System apply is unavailable in Library; use the Console "
                            "prompt workbench to apply it to the session."
                        ),
                        embedded=True,
                        id="library-prompt-block-editor",
                    )
                    block_editor.disabled = self.mutation_in_flight
                    yield block_editor
                    yield Checkbox(
                        "Include current text as starter content",
                        value=self.include_starter_content,
                        id="library-prompt-recipe-starter",
                        disabled=self.mutation_in_flight,
                    )
                else:
                    yield Static(
                        editor_state.compatibility_reason,
                        id="library-prompt-compatibility",
                        classes="destination-purpose",
                        markup=False,
                    )
                    convert = Button(
                        "Convert and save as new Prompt",
                        id="library-prompt-convert",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=(
                            self.mutation_in_flight
                            or not editor_state.can_convert_as_new
                        ),
                    )
                    if convert.disabled:
                        convert.tooltip = (
                            "Conversion unavailable — this artifact has no compatible "
                            "System or User text."
                        )
                    yield convert
                yield Static(
                    "Compiled System preview",
                    classes="library-prompt-field-label",
                    markup=False,
                )
                yield Static(
                    _SYSTEM_PROMPT_HINT,
                    classes="library-prompt-field-hint",
                    markup=False,
                )
                yield TextArea(
                    editor_state.compiled_system_preview,
                    read_only=True,
                    id="library-prompt-system",
                )
                yield Static(
                    "Compiled User preview",
                    classes="library-prompt-field-label",
                    markup=False,
                )
                yield Static(
                    _USER_PROMPT_HINT, classes="library-prompt-field-hint", markup=False
                )
                yield TextArea(
                    editor_state.compiled_user_preview,
                    read_only=True,
                    id="library-prompt-user",
                )
                yield Input(
                    value=editor_state.keywords_csv,
                    placeholder="Keywords (comma-separated)",
                    id="library-prompt-keywords",
                    disabled=self.mutation_in_flight,
                )
                yield Static(
                    "Author", classes="library-prompt-field-label", markup=False
                )
                yield Input(
                    value=editor_state.author,
                    id="library-prompt-author",
                    disabled=self.mutation_in_flight,
                )
                yield Static(
                    prompt_editor_meta_line(editor_state, dirty=self.dirty),
                    id="library-prompt-meta",
                    markup=False,
                )
                if self.conflict:
                    yield Static(
                        "This item changed elsewhere — Reload the current version or "
                        "save your kept blocks as a new item.",
                        id="library-prompt-conflict-copy",
                        classes="destination-purpose",
                        markup=False,
                    )
                else:
                    yield Static(
                        self.status,
                        id="library-prompt-save-status",
                        markup=False,
                    )
                    if self.show_open_existing:
                        # Task 8b D3: makes the status copy's "...or open the
                        # existing prompt" a real affordance -- only shown in the
                        # name-in-use state (never alongside the conflict banner
                        # above, which has its own Save-as-new/Reload actions).
                        yield Button(
                            "Open existing",
                            id="library-prompt-open-existing",
                            classes="library-canvas-action",
                            compact=True,
                            disabled=self.mutation_in_flight,
                        )
                if self.membership_state is not None:
                    yield Static(
                        "Collections",
                        classes="library-prompt-field-label",
                        markup=False,
                    )
                    yield Static(
                        self._membership_summary(self.membership_state),
                        id="library-prompt-memberships-summary",
                        classes="destination-purpose",
                        markup=False,
                    )
                    yield Button(
                        self._membership_manage_label(self.membership_state),
                        id="library-prompt-memberships-manage",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=not (
                            not self.mutation_in_flight
                            and (
                                self.membership_state.can_manage
                                or self.membership_state.can_retry_load
                            )
                        ),
                    )
                    yield Button(
                        "Apply memberships",
                        id="library-prompt-memberships-apply",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=(
                            self.mutation_in_flight
                            or not self.membership_state.can_apply
                        ),
                    )
                    yield Static(
                        self._membership_status(self.membership_state),
                        id="library-prompt-memberships-status",
                        classes="destination-purpose",
                        markup=False,
                    )
                # Keep the empty region mounted for an unsaved editor so its
                # first successful create can reveal history without
                # remounting the editor fields or persistent action strip.
                history_region = LibraryPromptHistoryRegion(
                    self.history_state,
                    dirty=self.dirty,
                    current_compatible=self.history_current_compatible,
                    id="library-prompt-history-region",
                )
                history_region.disabled = self.mutation_in_flight
                yield history_region

            with Vertical(id="library-prompt-editor-actions"):
                if self.conflict:
                    yield Button(
                        "Save as new",
                        id="library-prompt-conflict-save-new",
                        classes="library-canvas-action console-action-primary",
                        compact=True,
                        disabled=self.mutation_in_flight,
                    )
                    yield Button(
                        "Reload",
                        id="library-prompt-conflict-reload",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=self.mutation_in_flight,
                    )
                else:
                    with Vertical(id="library-prompt-actions-primary"):
                        yield Button(
                            (
                                f"Save {editor_state.artifact_type.title()}"
                                if editor_state.prompt_id is None
                                else "Update original"
                            ),
                            id="library-prompt-save",
                            classes="library-canvas-action console-action-primary",
                            compact=True,
                            disabled=(
                                self.mutation_in_flight
                                or (
                                    editor_state.prompt_id is not None
                                    and (
                                        block_state is None
                                        or not self.can_update_original
                                    )
                                )
                            ),
                        )
                    with Vertical(id="library-prompt-actions-content"):
                        yield Button(
                            "Use in Console",
                            id="library-prompt-insert-console",
                            classes="library-canvas-action",
                            compact=True,
                            disabled=self.mutation_in_flight,
                        )

                        yield Button(
                            "Export…",
                            id="library-prompt-export",
                            classes="library-canvas-action",
                            compact=True,
                            disabled=self.mutation_in_flight,
                        )
                        yield Button(
                            "Copy Markdown",
                            id="library-prompt-copy",
                            classes="library-canvas-action",
                            compact=True,
                            disabled=self.mutation_in_flight,
                        )
                    with Vertical(id="library-prompt-actions-lifecycle"):
                        yield Button(
                            "Duplicate prompt",
                            id="library-prompt-duplicate",
                            classes="library-canvas-action",
                            compact=True,
                            disabled=self.mutation_in_flight,
                        )
                        yield Button(
                            "Delete",
                            id="library-prompt-delete",
                            classes="library-canvas-action library-media-action-danger",
                            compact=True,
                            disabled=self.mutation_in_flight,
                        )
                yield Button(
                    "Discard changes",
                    id="library-prompt-discard",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=(
                        self.mutation_in_flight
                        or self.write_in_flight
                        or not self.dirty
                    ),
                    tooltip=(
                        PROMPT_DISCARD_TOOLTIP_BUSY
                        if self.mutation_in_flight or self.write_in_flight
                        else (
                            PROMPT_DISCARD_TOOLTIP_DIRTY
                            if self.dirty
                            else PROMPT_DISCARD_TOOLTIP_CLEAN
                        )
                    ),
                )

    @staticmethod
    def _membership_ids_summary(
        state: PromptMembershipState, collection_ids: tuple[int, ...]
    ) -> str:
        labels = dict(state.labels)
        if not collection_ids:
            return "No collections"
        return ", ".join(
            labels.get(collection_id, f"Collection #{collection_id}")
            for collection_id in collection_ids
        )

    @classmethod
    def _membership_summary(cls, state: PromptMembershipState) -> str:
        if state.status == "disabled":
            return "Memberships unavailable"
        if state.status == "loading":
            return "Loading memberships…"
        if state.status == "load_error":
            return "Memberships not loaded"
        current = cls._membership_ids_summary(state, state.applied_ids)
        if state.staged_ids == state.applied_ids:
            return current
        staged = cls._membership_ids_summary(state, state.staged_ids)
        return f"Current: {current} · Staged: {staged}"

    @staticmethod
    def _membership_status(state: PromptMembershipState) -> str:
        if state.status == "disabled":
            return state.disabled_reason
        if state.status == "loading":
            return "Loading memberships…"
        if state.status == "applying":
            return "Applying memberships…"
        if state.outcome:
            return state.outcome
        if state.can_apply:
            return "Membership changes staged — apply separately from Prompt Save."
        return "Memberships are current."

    @staticmethod
    def _membership_manage_label(state: PromptMembershipState) -> str:
        return "Retry memberships" if state.can_retry_load else "Manage collections"

    def sync_memberships(self, state: PromptMembershipState) -> None:
        """Patch only membership controls, preserving every editor field widget."""
        self.membership_state = state
        self.query_one("#library-prompt-memberships-summary", Static).update(
            self._membership_summary(state)
        )
        self.query_one("#library-prompt-memberships-status", Static).update(
            self._membership_status(state)
        )
        manage = self.query_one("#library-prompt-memberships-manage", Button)
        manage.label = self._membership_manage_label(state)
        manage.disabled = not (state.can_manage or state.can_retry_load)
        self.query_one(
            "#library-prompt-memberships-apply", Button
        ).disabled = not state.can_apply

    def on_prompt_block_editor_block_field_changed(
        self, event: PromptBlockEditor.BlockFieldChanged
    ) -> None:
        """Patch compiled previews in place while the child editor stays mounted."""
        self._sync_block_preview(event.state)

    def on_prompt_block_editor_block_action_requested(
        self, event: PromptBlockEditor.BlockActionRequested
    ) -> None:
        """Patch previews after add/move/duplicate/delete without recomposition."""
        self._sync_block_preview(event.state)

    def _sync_block_preview(self, state: PromptBlockEditorState) -> None:
        current = self.editor_state
        if current is not None:
            self.editor_state = replace(
                current,
                block_editor_state=state,
                artifact_type=state.artifact_type,
                compiled_system_preview=state.compiled_system,
                compiled_user_preview=state.compiled_user,
                system_prompt=state.compiled_system,
                user_prompt=state.compiled_user,
            )
        for selector, value in (
            ("#library-prompt-system", state.compiled_system),
            ("#library-prompt-user", state.compiled_user),
        ):
            try:
                preview = self.query_one(selector, TextArea)
            except NoMatches:
                continue
            if preview.text != value:
                with preview.prevent(TextArea.Changed):
                    preview.load_text(value)
