"""Library prompts canvas: list mode (rows + filter + sort).

Structural template copy of ``library_notes_canvas.py``'s list-view
``compose`` -- prompts and notes diverge (two-part editor, no sync), so only
the list shape (header count line, filter Input, single-row
``ds-toolbar``, row Buttons with escaped labels) is mirrored here.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Library.library_prompts_state import (
    PromptEditorState,
    PromptsListState,
    prompt_editor_meta_line,
)

_SORT_LABELS = {"newest": "Newest", "name": "Name"}
_EMPTY_PROMPTS_COPY = "No prompts yet."
_EMPTY_PROMPTS_FILTER_COPY = "No prompts match your filter."


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
        mode: ``"list"`` renders the prompts list; ``"editor"`` renders the
            in-canvas prompt editor for ``editor_state``.
        editor_state: The prompt to render in editor mode. Required when
            ``mode == "editor"``.
        conflict: When ``True`` (editor mode only), renders the save
            conflict banner -- a quiet explanatory line plus Overwrite/
            Reload actions -- in place of the normal action row. Mirrors
            ``LibraryNotesCanvas.conflict``. ``editor_state`` must already
            reflect the user's kept text (never the stale server detail)
            when this is set.
        status: Save-outcome status text shown below the meta line (e.g.
            ``"Saved."`` or a name-conflict explanation), or ``""`` when
            idle. Not shown while ``conflict`` is set -- the conflict
            banner communicates the outcome instead.
        import_open: List-view only. When ``True``, renders the inline
            Import row (a path ``Input`` for a file OR folder, plus
            Import/Cancel actions) below the sort/Import…/Export…
            toolbar.
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
        mode: str = "list",
        editor_state: PromptEditorState | None = None,
        conflict: bool = False,
        status: str = "",
        import_open: bool = False,
        import_path: str = "",
        import_status: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.mode = mode
        self.editor_state = editor_state
        self.conflict = conflict
        self.status = status
        self.import_open = import_open
        self.import_path = import_path
        self.import_status = import_status
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
        yield Static(
            f"Prompts ({state.count})",
            id="library-prompts-header",
            classes="destination-section",
            markup=False,
        )
        yield Input(
            placeholder="Filter prompts… (Enter)",
            id="library-prompts-filter",
            value=self.filter_value,
        )
        # One horizontal ds-toolbar row for sort/Import/Export -- mirrors
        # library_notes_canvas.py's toolbar exactly (same render-safe shape:
        # every child is a fixed-width compact Button, never mixed with a
        # 1fr sibling).
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                f"sort: {_SORT_LABELS.get(self.sort_mode, 'Newest')} ▸",
                id="library-prompts-sort", classes="library-canvas-action", compact=True,
            )
            yield Button(
                "Import…", id="library-prompts-import",
                classes="library-canvas-action", compact=True,
            )
            yield Button(
                "Export…", id="library-prompts-export",
                classes="library-canvas-action", compact=True,
            )
        if self.import_open:
            yield from self._compose_import_row()
        if not state.rows:
            yield Static(
                _EMPTY_PROMPTS_FILTER_COPY if self.filter_value else _EMPTY_PROMPTS_COPY,
                id="library-prompts-empty",
                markup=False,
            )
            return
        with Vertical(id="library-prompts-list"):
            for row in state.rows:
                # Button labels are parsed as Rich markup: escape the
                # user-supplied name so "[draft] Q3 plan [wip]" renders
                # verbatim instead of eating bracketed segments as tags (or
                # crashing on an unmatched closing tag) -- same fix class as
                # the notes list row / search-history Button labels.
                name = escape_markup(row.name)
                button = Button(
                    f"{name}\n{row.secondary}" if row.secondary else name,
                    id=f"library-prompt-row-{row.prompt_id}",
                    classes="library-prompt-row",
                    compact=True,
                )
                button.prompt_id = row.prompt_id
                yield button

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
        the sort/Import…/Export… toolbar above.
        """
        yield Input(
            placeholder="File or folder path…",
            id="library-prompts-import-path",
            value=self.import_path,
        )
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                "Import", id="library-prompts-import-run",
                classes="library-canvas-action", compact=True,
            )
            yield Button(
                "Cancel", id="library-prompts-import-cancel",
                classes="library-canvas-action", compact=True,
            )
        yield Static(
            self.import_status,
            id="library-prompts-import-status",
            markup=False,
        )

    def _compose_editor(self) -> ComposeResult:
        """Render the prompt editor: Back, six fields, meta, actions.

        Structural template copy of ``LibraryNotesCanvas._compose_editor``:
        stacked full-width widgets plus a single plain ``ds-toolbar`` action
        row. Unlike the notes editor there is no autosave/Preview toggle
        here (prompts use an explicit Save button only), and no inline
        delete-confirmation state -- Delete is a single, confirm-free
        action (styled the same danger tier as the notes delete button).
        """
        editor_state = self.editor_state
        if editor_state is None:
            return
        yield Button(
            "‹ Back to list",
            id="library-prompt-back",
            classes="library-canvas-action",
            compact=True,
        )
        yield Static("Name", classes="library-prompt-field-label", markup=False)
        yield Input(value=editor_state.name, id="library-prompt-name")
        yield Static("Author", classes="library-prompt-field-label", markup=False)
        yield Input(value=editor_state.author, id="library-prompt-author")
        yield Static("Details", classes="library-prompt-field-label", markup=False)
        yield Input(value=editor_state.details, id="library-prompt-details")
        yield Static("System prompt", classes="library-prompt-field-label", markup=False)
        yield TextArea(editor_state.system_prompt, id="library-prompt-system")
        yield Static("User prompt", classes="library-prompt-field-label", markup=False)
        yield TextArea(editor_state.user_prompt, id="library-prompt-user")
        yield Input(
            value=editor_state.keywords_csv,
            placeholder="Keywords (comma-separated)",
            id="library-prompt-keywords",
        )
        yield Static(
            prompt_editor_meta_line(editor_state),
            id="library-prompt-meta",
            markup=False,
        )
        if self.conflict:
            yield Static(
                "This prompt changed elsewhere — Overwrite saves your text; "
                "Reload discards it.",
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
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            if self.conflict:
                yield Button(
                    "Overwrite",
                    id="library-prompt-conflict-overwrite",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Reload",
                    id="library-prompt-conflict-reload",
                    classes="library-canvas-action",
                    compact=True,
                )
            else:
                yield Button(
                    "Save",
                    id="library-prompt-save",
                    classes="library-canvas-action",
                    compact=True,
                )
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
                    "Copy",
                    id="library-prompt-copy",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Delete",
                    id="library-prompt-delete",
                    classes="library-canvas-action library-media-action-danger",
                    compact=True,
                )
