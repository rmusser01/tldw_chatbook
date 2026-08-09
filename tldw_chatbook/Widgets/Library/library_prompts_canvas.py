"""Library prompts canvas: list mode (rows + filter + sort).

Structural template copy of ``library_notes_canvas.py``'s list-view
``compose`` -- prompts and notes diverge (two-part editor, no sync), so only
the list shape (header count line, filter Input, single-row
``ds-toolbar``, row Buttons with escaped labels) is mirrored here.
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
    PromptBrowseResult,
    PromptEditorState,
    PromptHistoryState,
    PromptsListState,
    definition_state_display_label,
    prompt_editor_meta_line,
)
from tldw_chatbook.UI.Library_Modules.prompt_history_region import (
    LibraryPromptHistoryRegion,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import PromptBlockEditor
from tldw_chatbook.Widgets.Prompts.prompt_block_editor_state import (
    PromptBlockEditorState,
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


class LibraryPromptsListCanvas(Vertical):
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
        import_open: List-view only. When ``True``, renders the inline
            Import row (a path ``Input`` for a file OR folder, plus
            Import/Cancel actions) below the sort/Import… toolbar (Task 8c
            D5 removed the dead Export… button that used to sit here).
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
        mode: str = "list",
        editor_state: PromptEditorState | None = None,
        conflict: bool = False,
        status: str = "",
        show_open_existing: bool = False,
        import_open: bool = False,
        import_path: str = "",
        import_status: str = "",
        dirty: bool = False,
        can_update_original: bool = False,
        include_starter_content: bool = False,
        history_state: PromptHistoryState | None = None,
        history_current_compatible: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.browse_result = browse_result
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
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        if self.mode == "editor":
            yield from self._compose_editor()
            return
        yield from self._compose_list()

    def _compose_list(self) -> ComposeResult:
        state = self.state
        if state is None:
            return
        browse_result = self.browse_result
        total = browse_result.total_items if browse_result is not None else state.count
        yield Static(
            f"Prompts ({total})",
            id="library-prompts-header",
            classes="destination-section",
            markup=False,
        )
        yield Input(
            placeholder="Filter prompts… (Enter)",
            id="library-prompts-filter",
            value=self.filter_value,
        )
        # One horizontal ds-toolbar row for sort/Import -- mirrors
        # library_notes_canvas.py's toolbar exactly (same render-safe shape:
        # every child is a fixed-width compact Button, never mixed with a
        # 1fr sibling). Task 8c D5: the list-toolbar "Export…" button that
        # used to sit here had no handler anywhere -- pressing it silently
        # no-op'd -- so it was removed rather than wired to a fake bulk
        # export. Bulk export is deferred to backlog task-197; per-prompt
        # export lives in the editor's own #library-prompt-export and still
        # works.
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                f"sort: {_SORT_LABELS.get(self.sort_mode, 'Newest')} ▸",
                id="library-prompts-sort",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Import…",
                id="library-prompts-import",
                classes="library-canvas-action",
                compact=True,
            )
        if self.import_open:
            yield from self._compose_import_row()
        if browse_result is not None and browse_result.status == "loading":
            yield Static(
                "Loading prompts…",
                id="library-prompts-loading",
                classes="destination-purpose",
                markup=False,
            )
            return
        if browse_result is not None and browse_result.status == "error":
            yield Static(
                browse_result.error,
                id="library-prompts-error",
                classes="destination-purpose",
                markup=False,
            )
            yield Button(
                "Retry",
                id="library-prompts-retry",
                classes="library-canvas-action",
                compact=True,
            )
            return
        if browse_result is not None and browse_result.total_pages > 1:
            yield from self._compose_paging(browse_result)
        if not state.rows:
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
            return
        with Vertical(id="library-prompts-list"):
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
                label_parts = (name, artifact_summary, secondary)
                button = Button(
                    "\n".join(part for part in label_parts if part),
                    id=f"library-prompt-row-{row.prompt_id}",
                    classes="library-prompt-row",
                    compact=True,
                )
                button.prompt_id = row.prompt_id
                yield button

    def _compose_paging(self, result: PromptBrowseResult) -> ComposeResult:
        """Render the minimal bounded page controls required for exact browse."""
        first = (result.page - 1) * result.scope.page_size + 1
        last = min(result.total_items, first + len(result.items) - 1)
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                "Previous",
                id="library-prompts-page-previous",
                classes="library-canvas-action",
                compact=True,
                disabled=result.page <= 1,
            )
            yield Static(
                f"Page {result.page} of {result.total_pages} · "
                f"showing {first}–{last} of {result.total_items}",
                id="library-prompts-page-label",
                markup=False,
            )
            yield Button(
                "Next",
                id="library-prompts-page-next",
                classes="library-canvas-action",
                compact=True,
                disabled=result.page >= result.total_pages,
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
                "Browse…",
                id="library-prompts-import-browse",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Import",
                id="library-prompts-import-run",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Cancel",
                id="library-prompts-import-cancel",
                classes="library-canvas-action",
                compact=True,
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
                yield Button(
                    "‹ Back to list",
                    id="library-prompt-back",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Static("Name", classes="library-prompt-field-label", markup=False)
                yield Input(value=editor_state.name, id="library-prompt-name")
                # Task 8b U4: rendered label only -- the DB/record field name
                # (``details``, ``#library-prompt-details``) is untouched.
                yield Static(
                    "Description", classes="library-prompt-field-label", markup=False
                )
                yield Input(value=editor_state.details, id="library-prompt-details")
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
                    yield block_editor
                    yield Checkbox(
                        "Include current text as starter content",
                        value=self.include_starter_content,
                        id="library-prompt-recipe-starter",
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
                        disabled=not editor_state.can_convert_as_new,
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
                )
                yield Static(
                    "Author", classes="library-prompt-field-label", markup=False
                )
                yield Input(value=editor_state.author, id="library-prompt-author")
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
                        )
                # Keep the empty region mounted for an unsaved editor so its
                # first successful create can reveal history without
                # remounting the editor fields or persistent action strip.
                yield LibraryPromptHistoryRegion(
                    self.history_state,
                    dirty=self.dirty,
                    current_compatible=self.history_current_compatible,
                    id="library-prompt-history-region",
                )

            with Vertical(id="library-prompt-editor-actions"):
                if self.conflict:
                    yield Button(
                        "Save as new",
                        id="library-prompt-conflict-save-new",
                        classes="library-canvas-action console-action-primary",
                        compact=True,
                    )
                    yield Button(
                        "Reload",
                        id="library-prompt-conflict-reload",
                        classes="library-canvas-action",
                        compact=True,
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
                                editor_state.prompt_id is not None
                                and (
                                    block_state is None or not self.can_update_original
                                )
                            ),
                        )
                    with Vertical(id="library-prompt-actions-content"):
                        yield Button(
                            "Use in Console",
                            id="library-prompt-insert-console",
                            classes="library-canvas-action",
                            compact=True,
                        )
                        yield Button(
                            "Export…",
                            id="library-prompt-export",
                            classes="library-canvas-action",
                            compact=True,
                        )
                        yield Button(
                            "Copy Markdown",
                            id="library-prompt-copy",
                            classes="library-canvas-action",
                            compact=True,
                        )
                    with Vertical(id="library-prompt-actions-lifecycle"):
                        yield Button(
                            "Duplicate prompt",
                            id="library-prompt-duplicate",
                            classes="library-canvas-action",
                            compact=True,
                        )
                        yield Button(
                            "Delete",
                            id="library-prompt-delete",
                            classes="library-canvas-action library-media-action-danger",
                            compact=True,
                        )

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
