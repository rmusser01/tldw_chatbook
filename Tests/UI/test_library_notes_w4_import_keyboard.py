"""task-32540/32553/32554: Import once by keyboard, its copy, one back cue.

Critique #3 (dev 5fd502dbac), both assessors, Obsidian import workflow.
Reproduced live at 235x52 on a scratch power profile before these pins were
written (captures ``wave4-caps/import-kbd/import-00-tree-focused.txt``,
``import-00b-buttons-colour-only.txt``, ``import-01-tab-leak.txt``):

* the picker opened with ``DirectoryNavigation`` focused, so a typed path was
  swallowed by the listing's type-ahead and never reached the field;
* Open / Select folder / Cancel carried only a colour + label-underline focus
  treatment -- no geometry changed between focused and unfocused;
* after "Select folder" the confirmation pane's Tab walked *out* of the
  canvas: source switch, body, then the rail's "Search Library…" box, never
  marking Change selection / Clear / Check selection.

task-32554's copy defects were captured in the same walk
(``import-02-copy.txt``): every New row read "Content: create 1 new note: …",
the pane printed "1 folder selected." and then "1 folder selected: /…" with a
48-character elision inside a 190-column pane, and the picker's breadcrumb
rendered every segment of a 100+ character path off the dialog's right edge.
task-32553's three spellings of "go back" are in ``import-00-tree-focused.txt``
("Back to Notes") and the Folder-files surfaces ("Back to navigator").

Every assertion below reads what production renders or focuses, never a
string the test itself supplied.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Input, Label, Static

import tldw_chatbook
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_note_import_state import (
    initial_note_import_snapshot,
    project_library_note_import_snapshot,
    select_folder,
)
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import (
    FileSystemPickerScreen,
    InputBar,
)
from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation
from tldw_chatbook.Widgets.Library.library_note_import_canvas import (
    LibraryNoteImportCanvas,
)
from tldw_chatbook.Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas
from tldw_chatbook.Widgets.Library.library_notes_sync_roots_canvas import (
    LibraryNotesSyncRootsCanvas,
)

pytestmark = pytest.mark.asyncio

_BUNDLED_STYLESHEET = (
    Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
)

#: The three actions the confirmation pane offers once a folder is chosen,
#: in compose order (``library_note_import_canvas._compose_selection`` plus
#: ``_compose_primary_action``).
_SELECTION_PANE_ACTIONS = (
    "note-import-change-source",
    "note-import-clear-source",
    "note-import-check",
)


class _PickerHost(ConsolidatedCSSApp):
    """Push one real picker under the real app stylesheet.

    ``CSS_PATH`` is the generated bundle, because the focus treatment under
    test is an APP-tier rule: ``components/_buttons.tcss`` sets
    ``Button:focus { outline: none; }``, which Textual's cascade ranks above
    any widget ``DEFAULT_CSS`` regardless of specificity, so a shape cue
    added to the dialog's own default CSS would never paint.
    """

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self, dialog: FileSystemPickerScreen) -> None:
        super().__init__()
        self._dialog = dialog
        self.results: list[object] = []

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        await self.push_screen(self._dialog, callback=self.results.append)


def _import_once_picker(location: Path) -> FileOpen:
    """The exact dialog Import once pushes (library_screen.py)."""
    return FileOpen(
        title="Import once (files or one folder)",
        offer_select_folder=True,
        location=str(location),
    )


def _field(dialog: FileSystemPickerScreen) -> Input:
    return dialog.query_one(InputBar).query_one(Input)


async def _press_until_focused(pilot, widget_id: str, *, limit: int = 60) -> int:
    """Tab until ``widget_id`` holds focus; return how many presses it took."""
    for presses in range(1, limit + 1):
        await pilot.press("tab")
        focused = pilot.app.focused
        if focused is not None and focused.id == widget_id:
            return presses
    raise AssertionError(
        f"#{widget_id} was never reached by Tab "
        f"(stopped on {getattr(pilot.app.focused, 'id', None)!r})"
    )


# --- AC#1: the picker opens with its path field focused --------------------


async def test_the_import_picker_opens_with_its_path_field_focused(tmp_path) -> None:
    """AC#1. Typed text must land in the field, not in the listing."""
    (tmp_path / "vault").mkdir()
    dialog = _import_once_picker(tmp_path)
    app = _PickerHost(dialog)

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        await pilot.pause()

        field = _field(dialog)
        assert app.focused is field, (
            "Import once's picker must open focused on its path field; "
            f"focus was on {type(app.focused).__name__}"
        )

        await pilot.press(*"vault")
        assert field.value == "vault", (
            "typed text never reached the field -- the listing swallowed it"
        )

        # Enter on a typed directory browses into it (file_dialog._confirm_file).
        await pilot.press("enter")
        await pilot.pause()
        assert dialog.query_one(DirectoryNavigation).location == tmp_path / "vault"


async def test_a_plain_file_picker_still_opens_on_its_listing(tmp_path) -> None:
    """Negative control: only the folder-offering picker changes.

    Every other caller of ``FileOpen`` (character import, skill folders, TTS
    models, …) keeps the listing-first behaviour it has always had.
    """
    dialog = FileOpen(location=str(tmp_path))
    app = _PickerHost(dialog)

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.focused, DirectoryNavigation)


# --- AC#2: a shape-based focus cue, not colour alone -----------------------


async def test_picker_buttons_carry_a_shape_focus_cue(tmp_path) -> None:
    """AC#2. Open / Select folder / Cancel must change shape on focus.

    Asserts the COMPUTED style of the mounted button, so the rule has to win
    the real cascade (the app-tier ``Button:focus { outline: none; }`` is what
    a widget-tier rule would lose to).
    """
    dialog = _import_once_picker(tmp_path)
    app = _PickerHost(dialog)

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        await pilot.pause()

        for button_id in ("select", "select-current-folder", "cancel"):
            button = dialog.query_one(f"#{button_id}", Button)
            button.focus()
            await pilot.pause()
            assert app.focused is button
            edges = (
                button.styles.outline_left[0],
                button.styles.outline_right[0],
            )
            assert any(edge not in ("", "none") for edge in edges), (
                f"#{button_id} has no shape-based focus cue: "
                f"outline edges {edges!r}"
            )


# --- AC#3: Tab stays in the canvas and marks the three actions -------------


async def _open_import_once_with_folder(screen, pilot, folder: Path):
    """Reach the confirmation pane through the shipped chooser + real picker."""
    await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
    screen.query_one("#library-notes-add-from-files").press()
    await _wait_for_selector(screen, pilot, "#notes-add-import-once")
    screen.query_one("#notes-add-import-once").press()

    dialog = await _wait_for_picker(pilot)
    _field(dialog).value = str(folder)
    await pilot.pause()
    dialog.query_one("#select-current-folder", Button).press()
    await _wait_for_selector(screen, pilot, "#note-import-change-source")


async def _wait_for_picker(pilot, *, attempts: int = 200) -> FileSystemPickerScreen:
    for _ in range(attempts):
        top = pilot.app.screen_stack[-1]
        if isinstance(top, FileSystemPickerScreen):
            await pilot.pause()
            return top
        await pilot.pause()
    raise AssertionError("the Import once picker never opened")


async def test_tab_from_the_selection_pane_cycles_the_three_actions(
    tmp_path,
) -> None:
    """AC#3. Tab walks Change selection → Clear → Check selection, in canvas."""
    folder = tmp_path / "vault"
    folder.mkdir()
    (folder / "note.md").write_text("# Note\n", encoding="utf-8")

    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _open_import_once_with_folder(screen, pilot, folder)

        # The confirmation itself holds focus once the picker returns, so the
        # summary is what a screen reader lands on and Tab starts from there.
        assert host.focused is not None and host.focused.id == "note-import-body", (
            "focus never returned to the import canvas after Select folder; "
            f"it was on {getattr(host.focused, 'id', None)!r}"
        )

        # task-32554 AC#2 on the real route, not a bare host: the path budget
        # follows the mounted pane rather than the 48-character compact
        # floor. (This harness's work pane is narrower than a real 235-column
        # terminal, so a pytest tmp path still elides -- what is pinned here
        # is that the line uses the width it has.)
        canvas = screen.query_one("#library-note-import-canvas")
        summary = screen.query_one("#note-import-source-summary", Static)
        summary_text = getattr(summary.renderable, "plain", str(summary.renderable))
        assert len(summary_text) <= canvas.content_size.width, summary_text
        assert len(summary_text) > len("1 folder selected: ") + 48, summary_text

        visited: list[str | None] = []
        for _ in range(len(_SELECTION_PANE_ACTIONS)):
            await pilot.press("tab")
            focused = host.focused
            visited.append(getattr(focused, "id", None))
            assert focused is not None
            # The COMPUTED cue, not merely the class that carries it: a
            # regression that drops the app-tier
            # `Button.library-canvas-action:focus` rule would leave a
            # class-membership assertion green (review finding 4).
            edges = (
                focused.styles.outline_left[0],
                focused.styles.outline_right[0],
            )
            assert any(edge not in ("", "none") for edge in edges), (
                f"{getattr(focused, 'id', None)!r} shows no shape-based "
                f"focus mark: outline edges {edges!r}"
            )
        assert tuple(visited) == _SELECTION_PANE_ACTIONS

        # Keep tabbing: the cycle must close inside the stepper's pane
        # rather than leaking into the rail's search box (the live defect).
        for _ in range(12):
            await pilot.press("tab")
            focused = host.focused
            assert focused is not None
            assert getattr(focused, "id", None) != "library-rail-search"
            assert any(
                node.id == "library-note-work-pane"
                for node in focused.ancestors_with_self
            ), (
                f"Tab left the Import once pane and landed on "
                f"{getattr(focused, 'id', None)!r}"
            )


# --- AC#4: the whole flow, keyboard only -----------------------------------


async def test_import_once_completes_by_keyboard_alone(tmp_path) -> None:
    """AC#4. Add from files… → receipt with ``pilot.press`` only.

    The AC starts at **Add from files…**, so reaching the notes list presses
    ``#library-row-browse-notes`` directly; from there every step is a
    keystroke, through the real chooser, the real ``FileOpen`` dialog and the
    real import worker. Nothing after that press touches a Button object or
    sets a widget value.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
    from tldw_chatbook.UI.Screens import library_screen as library_screen_module

    folder = tmp_path / "vault"
    folder.mkdir()
    (folder / "note.md").write_text("# Keyboard note\n\nBody\n", encoding="utf-8")

    database = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="w4-import-kbd")
    folders = LocalNoteFolderRepository(database)
    interop = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="w4-import-kbd",
        global_db_to_use=database,
    )
    scope_service = NotesScopeService(
        local_notes_service=interop,
        server_service=None,
        folder_repository=folders,
    )
    receipt_path = tmp_path / "import-receipts.sqlite"

    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=[])
    app.chachanotes_db = database
    app.notes_scope_service = scope_service
    host = LibraryHarness(app)

    original = library_screen_module.get_notes_sync_state_db_path
    library_screen_module.get_notes_sync_state_db_path = lambda: receipt_path
    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-notes").press()
            await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")

            await _press_until_focused(pilot, "library-notes-add-from-files")
            await pilot.press("enter")
            await _wait_for_selector(screen, pilot, "#notes-add-import-once")

            await _press_until_focused(pilot, "notes-add-import-once")
            await pilot.press("enter")

            dialog = await _wait_for_picker(pilot)
            await pilot.press(*str(folder))
            await pilot.pause()
            await _press_until_focused(pilot, "select-current-folder")
            await pilot.press("enter")
            await _wait_for_selector(screen, pilot, "#note-import-check")

            await _press_until_focused(pilot, "note-import-check")
            await pilot.press("enter")
            await _wait_for_selector(screen, pilot, "#note-import-import")

            # The review's own primary action holds focus the moment the
            # review renders -- the guide's recipe says "Enter" here, not
            # "Tab past sixty row controls first".
            assert host.focused is not None
            assert host.focused.id == "note-import-import", host.focused.id
            await pilot.press("enter")
            await _wait_for_selector(screen, pilot, "#note-import-receipt")
            await _wait_for_condition(
                pilot,
                lambda: database.count_notes() == 1,
                message="the keyboard-only import never created the note",
            )

            assert (
                screen._library_note_import_controller.snapshot.phase == "receipt"
            )
    finally:
        library_screen_module.get_notes_sync_state_db_path = original
        interop.close_all_user_connections()
        database.close_connection()


# --- task-32554: the import copy ------------------------------------------


class _SelectionHost(ConsolidatedCSSApp):
    """Mount the import canvas alone at a stated pane width."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self, folder: str) -> None:
        super().__init__()
        # The REAL projection of a real folder selection: status_line and
        # selected_names both come from production, not from this test.
        self._snapshot = project_library_note_import_snapshot(
            select_folder(initial_note_import_snapshot(), Path(folder))
        )

    def compose(self) -> ComposeResult:
        yield LibraryNoteImportCanvas(
            self._snapshot, id="library-note-import-canvas"
        )


def _summary_text(app: _SelectionHost) -> str:
    static = app.query_one("#note-import-source-summary", Static)
    return getattr(static.renderable, "plain", str(static.renderable))


async def test_the_selection_is_stated_once_with_the_full_path_when_it_fits() -> None:
    """32554 AC#2. A path that fits a 190-column pane is not elided."""
    folder = "/Users/reader/" + "/".join(["segment"] * 8) + "/fresh/vault"
    assert 80 <= len(folder) <= 120, len(folder)
    app = _SelectionHost(folder)

    async with app.run_test(size=(190, 40)) as pilot:
        await pilot.pause()
        await pilot.pause()
        text = _summary_text(app)

    assert "…" not in text, f"a {len(folder)}-char path was elided in 190 columns: {text}"
    assert folder in text


async def test_a_path_too_long_for_the_pane_still_middle_elides() -> None:
    """32554 AC#2, other half: elision only when the pane cannot hold it."""
    folder = "/Users/reader/" + "/".join([f"segment{i:02d}" for i in range(28)]) + "/vault"
    assert len(folder) > 280, len(folder)
    app = _SelectionHost(folder)

    async with app.run_test(size=(190, 40)) as pilot:
        await pilot.pause()
        await pilot.pause()
        text = _summary_text(app)

    assert "…" in text
    # The basename is the part the reader recognises; it must survive.
    assert text.endswith("vault")
    assert len(text) <= 190


async def test_the_selection_is_not_stated_twice_on_the_same_pane() -> None:
    """32554 AC#2. The status line must not repeat the summary's sentence."""
    folder = "/Users/reader/vault"
    app = _SelectionHost(folder)

    async with app.run_test(size=(190, 40)) as pilot:
        await pilot.pause()
        status = app.query_one("#note-import-status", Static)
        status_text = getattr(status.renderable, "plain", str(status.renderable))
        summary_text = _summary_text(app)

    assert summary_text.startswith("1 folder selected")
    assert "folder selected" not in status_text, (
        f"the pane states the selection twice: {status_text!r} then "
        f"{summary_text!r}"
    )


async def test_the_picker_breadcrumb_is_one_line_with_a_middle_ellipsis(
    tmp_path,
) -> None:
    """32554 AC#3. A deep path collapses in the middle instead of running off."""
    deep = tmp_path
    for part in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot"):
        deep = deep / part
    deep.mkdir(parents=True)

    dialog = _import_once_picker(deep)
    app = _PickerHost(dialog)

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        await pilot.pause()
        crumbs = dialog.query_one("#path-breadcrumbs")
        buttons = list(crumbs.query(Button))
        ellipses = [
            label
            for label in crumbs.query(Label)
            if "breadcrumb-ellipsis" in label.classes
        ]
        width = crumbs.size.width
        painted = sum(child.outer_size.width for child in crumbs.children)

    assert len(deep.parts) > 8, deep
    assert ellipses, (
        f"a {len(deep.parts)}-segment path rendered every crumb "
        f"({[str(b.label) for b in buttons]})"
    )
    assert len(buttons) < len(deep.parts)
    assert str(buttons[-1].label) == deep.name
    assert painted <= width, (
        f"the breadcrumb row painted {painted} columns into {width}"
    )


# --- task-32553: one back-cue grammar -------------------------------------


class _BackCueHost(ConsolidatedCSSApp):
    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        import_snapshot = replace(
            project_library_note_import_snapshot(initial_note_import_snapshot()),
            phase="select",
        )
        yield LibraryNotesCanvas(mode="import", import_snapshot=import_snapshot)
        yield LibraryNotesAddFromFilesCanvas(
            initial_lasting_sync_snapshot(), id="chooser"
        )
        yield LibraryNotesSyncRootsCanvas(
            initial_lasting_sync_snapshot(), id="roots"
        )


async def test_one_back_cue_grammar_across_the_notes_surfaces() -> None:
    """32553 AC#1. "‹ <where it goes>" everywhere, not three spellings."""
    app = _BackCueHost()

    async with app.run_test(size=(190, 60)) as pilot:
        await pilot.pause()
        labels = {
            button_id: str(app.query_one(f"#{button_id}", Button).label)
            for button_id in (
                "library-notes-import-back",
                "notes-sync-back",
                "notes-sync-roots-back",
            )
        }

    assert labels == {
        "library-notes-import-back": "‹ Notes",
        "notes-sync-back": "‹ Notes",
        "notes-sync-roots-back": "‹ Notes",
    }, labels
