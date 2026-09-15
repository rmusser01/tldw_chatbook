"""task-32606: Folder files' folder picker by keyboard, for the whole family.

Critique #4 (dev 77eb2601a6), assessor A P1, personas Sam (keyboard-only) and
Jordan. Reproduced headless on dev before these pins were written::

    SelectDirectory      focused=ProgressiveDirectoryNavigation
       after typing "tmp": value=None          <- the listing ate it
    FileOpen+folder      focused=FileNameInput
       after typing "tmp": value='tmp'

Wave 4's task-32540 taught ``FileOpen(offer_select_folder=True)`` to open on
its path field. Folder files pushes a *different* class -- the vendored
``SelectDirectory`` (``library_file_notes_workspace._open_root_picker``) --
which never got it, and neither did ``EnhancedSelectDirectory``. The fix is
the shared ``FileSystemPickerScreen._focus_initial_widget`` reading one
declarative flag, so the behaviour belongs to "dialogs that hand back a
folder" rather than to one subclass; these pins run the real Folder-files
door and then the whole family.

Every assertion reads what production focuses or renders, never a value the
test itself supplied to the widget under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, TextArea, Tree

# Stubs first in the local group: it registers the optional MLX modules the
# application imports below would otherwise probe.
import Tests.UI._optional_module_stubs  # noqa: F401
from Tests.UI.test_library_file_notes_workspace import (
    _production_workspace_context,
    _wait_until,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, SelectDirectory
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import (
    FileSystemPickerScreen,
    InputBar,
)
from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen,
    EnhancedSelectDirectory,
)
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)

pytestmark = pytest.mark.asyncio

WIDE = (235, 52)
CRITIQUE_COMPACT = (100, 30)


def _field(dialog: FileSystemPickerScreen) -> Input:
    """The dialog's one input-bar field, read the way production reads it."""
    return dialog.query_one(InputBar).query_one(Input)


class _PickerHost(App[None]):
    """Minimal host that pushes one real picker."""

    def __init__(self, dialog: FileSystemPickerScreen) -> None:
        super().__init__()
        self._dialog = dialog

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        await self.push_screen(self._dialog)


async def _wait_for_picker(pilot, *, attempts: int = 200) -> FileSystemPickerScreen:
    for _ in range(attempts):
        top = pilot.app.screen_stack[-1]
        if isinstance(top, FileSystemPickerScreen):
            await pilot.pause()
            await pilot.pause()
            return top
        await pilot.pause()
    raise AssertionError("the Folder files folder picker never opened")


def _vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "daily.md").write_text("# Daily\n\nfirst line\n", encoding="utf-8")
    (root / "sub").mkdir()
    return root


# --- AC#1: the Folder-files door, on the real route -------------------------


@pytest.mark.parametrize("size", [WIDE, CRITIQUE_COMPACT])
async def test_the_folder_files_picker_opens_with_its_path_field_focused(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """AC#1/AC#3: Enter on "Choose folder…" and the first keystroke lands.

    Drives the shipped button with the keyboard only -- no ``press()`` call
    and no click -- through ``_open_root_picker``'s real ``SelectDirectory``.
    """
    root = _vault(tmp_path)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    try:
        async with _production_workspace_context(workspace, size=size) as pilot:
            choose = workspace.query_one("#file-notes-choose-root", Button)
            choose.focus()
            await pilot.pause()
            assert pilot.app.focused is choose
            await pilot.press("enter")

            dialog = await _wait_for_picker(pilot)
            field = _field(dialog)
            assert pilot.app.focused is field, (
                "Choose File Notes Folder must open focused on its path "
                f"field; focus was on {type(pilot.app.focused).__name__}"
            )
            # Pre-filled by the dialog itself from the browsed location.
            assert field.value == str(root)
            assert field.selection.start != field.selection.end, (
                "the pre-filled folder is not selected, so the first "
                "keystroke appends to it instead of replacing it"
            )

            await pilot.press(*"sub")
            assert field.value == "sub", (
                "typed text never reached the field -- the listing's "
                f"type-ahead swallowed it (field holds {field.value!r})"
            )
    finally:
        replica.close()


async def test_folder_files_chooses_a_folder_and_edits_a_file_by_key_presses(
    tmp_path: Path,
) -> None:
    """AC#3: every step from the choose button to a changed file is a key.

    Honest scope (review round 1, minor 5): this pins what each control
    DOES when it holds focus, not the Tab route to it. Focus is placed
    programmatically at three points -- the choose button, the tree, the
    editor -- because the live Tab route into the Folder-files tree leaks
    into the Library rail, which is a separate open defect ridered by this
    task rather than fixed by it. The keyboard walk of the picker itself
    (the part this task owns) IS exercised end to end here and was
    re-walked live at 235x52 and 100x30.
    """
    root = _vault(tmp_path)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    try:
        async with _production_workspace_context(workspace, size=WIDE) as pilot:
            workspace.query_one("#file-notes-choose-root", Button).focus()
            await pilot.pause()
            await pilot.press("enter")

            dialog = await _wait_for_picker(pilot)
            field = _field(dialog)
            assert pilot.app.focused is field
            # Replace whatever the picker opened on with the vault path.
            await pilot.press("ctrl+a")
            for char in str(root):
                await pilot.press(char)
            await pilot.press("enter")
            await pilot.pause()
            assert dialog.query_one(DirectoryNavigation).location == root

            # Tab out of the field onto Select, and confirm with Enter.
            await pilot.press("tab")
            assert getattr(pilot.app.focused, "id", None) == "select"
            await pilot.press("enter")

            await _wait_until(
                pilot,
                lambda: workspace.root == root.resolve(),
                "the chosen folder never became the Folder files root",
            )
            await _wait_until(
                pilot,
                lambda: bool(workspace.query("#file-notes-tree")),
                "the Folder files tree never mounted",
            )

            tree = workspace.query_one("#file-notes-tree", Tree)
            tree.focus()
            await pilot.pause()
            for _ in range(12):
                if workspace.current_path == "daily.md":
                    break
                await pilot.press("down")
                await pilot.press("enter")
                await pilot.pause()
            assert workspace.current_path == "daily.md", (
                "the note never opened from the tree by keyboard"
            )

            editor = workspace.query_one("#file-notes-editor", TextArea)
            editor.focus()
            await pilot.pause()
            before = editor.text
            await pilot.press("X")
            assert editor.text != before, "the editor took no keyboard input"
    finally:
        replica.close()


# --- AC#2: the dialog advertises its own keys -------------------------------


async def test_the_folder_picker_renders_its_own_footer_chips(
    tmp_path: Path,
) -> None:
    """AC#2: a ModalScreen is translucent, so the host screen's chips show
    through. Three Tab presses left assessor A reading the Library's
    ``/ focus search | F6 next pane | esc notes`` while a modal was up.
    """
    from textual.widgets import Footer
    from textual.widgets._footer import FooterKey

    dialog = SelectDirectory(tmp_path, title="Choose File Notes Folder")
    app = _PickerHost(dialog)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        await pilot.pause()
        footer = dialog.query_one(Footer)
        keys = {chip.key: chip.description for chip in footer.query(FooterKey)}
        assert keys, "the picker's footer advertises nothing"
        # The dialog's own escape route, named by the dialog's own binding.
        assert "escape" in keys and keys["escape"], keys

        # The footer is a child of the SCREEN, not of `Dialog`. Docked at
        # screen level it REPLACES the host screen's chips; inside `Dialog`
        # it would merely add a second key row above a contradicting one,
        # which is the placement this task tried first and rejected
        # (capture 02-picker-open-235x52.txt). Without this the pin passes
        # for either placement (review round 1, minor 4).
        assert footer.parent is dialog, (
            "the footer must dock on the screen, over the host's own row; "
            f"it is a child of {type(footer.parent).__name__}"
        )


async def test_every_folder_offering_picker_gets_a_footer_of_its_own(
    tmp_path: Path,
) -> None:
    """AC#2 for the whole family, including the hand-mirrored one.

    `EnhancedFileDialog` re-implements `compose` instead of calling the
    base's, so the base's `Footer` never reached it and
    `EnhancedSelectDirectory` -- the second dialog this task fixed for
    initial focus -- still showed the HOST screen's chips through the
    translucent modal (review round 1, important 2).
    """
    from textual.widgets import Footer

    (tmp_path / "sub").mkdir()
    for name, dialog in _folder_offering_dialogs(tmp_path).items():
        app = _PickerHost(dialog)
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            await pilot.pause()
            footer = dialog.query_one(Footer)
            assert footer.parent is dialog, (
                f"{name}'s footer must dock on the screen, over the host's "
                f"own row; it is a child of {type(footer.parent).__name__}"
            )


@pytest.mark.parametrize(
    "dialog_name",
    ["SelectDirectory", "FileOpen(offer_select_folder)"],
)
async def test_the_footer_leads_with_the_keys_a_narrow_terminal_can_show(
    tmp_path: Path, dialog_name: str
) -> None:
    """AC#2 at 100x30: `Footer` scrolls its overflow off the right edge.

    Chips render in binding order, so the order IS the narrow-width
    priority. Before the reorder `esc Cancel` was off-screen entirely at
    100 columns; a dead `^s Select this folder` (vetoed by `check_action`
    on every dialog that does not offer it, yet still rendered, dimmed, by
    Textual) sat second and ate 23 of 60 columns ahead of every live
    action (review round 1, important 2 + minor 4).

    Scoped to the vendored family, whose `BINDINGS` this task owns.
    `EnhancedFileDialog` hides Escape on purpose (task-430's smart
    dismiss) and appends its own bindings AFTER the base's, so neither
    half of this rule is its to keep.
    """
    from textual.widgets._footer import FooterKey

    dialog = _folder_offering_dialogs(tmp_path)[dialog_name]
    app = _PickerHost(dialog)
    async with app.run_test(size=CRITIQUE_COMPACT) as pilot:
        await pilot.pause()
        await pilot.pause()
        order = [chip.key for chip in dialog.query(FooterKey)]
        assert order[0] == "escape", f"the way out must lead: {order}"
        live = [chip.key for chip in dialog.query(FooterKey) if not chip._disabled]
        dead = [chip.key for chip in dialog.query(FooterKey) if chip._disabled]
        # `FileOpen(offer_select_folder=True)` really does offer ctrl+s, so
        # it has nothing vetoed; the other two do.
        if dead:
            assert order.index(live[-1]) < order.index(dead[0]), (
                "a binding this dialog cannot run is ordered ahead of one it "
                f"can, so the narrow-width slots go to a dead chip: {order}"
            )


# --- Critical (review round 1): a footer chip click must not cancel --------


@pytest.mark.parametrize(
    "dialog_name",
    ["SelectDirectory", "FileOpen(offer_select_folder)", "EnhancedSelectDirectory"],
)
async def test_clicking_a_footer_chip_does_not_cancel_the_picker(
    tmp_path: Path, dialog_name: str
) -> None:
    """The footer docks OUTSIDE ``SAFE_MODAL_CONTENT``.

    `SafeModalDismissMixin.on_click` classifies a primary click whose
    target is neither the content nor a descendant, at a point outside the
    content region, as a backdrop click -- and cancels. Textual's
    `FooterKey.on_mouse_down` neither stops nor prevents the event, so a
    chip fires its key AND closes the dialog. Reproduced at
    `origin/dev`+this-task's-footer: screen stack 2 -> 1, result ``None``.
    The chips are styled `pointer: pointer` with a hover highlight, so
    they invite exactly that click, and on `FileSave` it discards a typed
    filename.
    """
    from textual.widgets._footer import FooterKey

    dialog = _folder_offering_dialogs(tmp_path)[dialog_name]
    app = _PickerHost(dialog)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        await pilot.pause()
        depth = len(app.screen_stack)
        # Every chip whose own action is not "close this dialog": `escape`
        # cancels by design, and on `FileOpen(offer_select_folder=True)`
        # ctrl+s confirms the folder and dismisses, also by design.
        closing = {"request_safe_cancel", "smart_dismiss", "select_current_folder"}
        keys = [c.key for c in dialog.query(FooterKey) if c.action not in closing]
        assert keys, "no non-closing chip to click"
        for key in keys:
            # Re-query every time: a chip whose key moves focus changes the
            # active bindings, so `Footer` recomposes and the widget from
            # the first pass is detached with a stale region.
            chip = next(c for c in dialog.query(FooterKey) if c.key == key)
            await pilot.click(chip)
            await pilot.pause()
            await pilot.pause()
            assert len(app.screen_stack) == depth, (
                f"clicking the {key!r} footer chip dismissed the picker "
                f"(stack {depth} -> {len(app.screen_stack)})"
            )
            assert app.screen is dialog


# --- AC#4: one shared behaviour, not a per-subclass override ----------------


def _folder_offering_dialogs(tmp_path: Path):
    """Every dialog in the family whose result is a folder."""
    return {
        "SelectDirectory": SelectDirectory(
            tmp_path, title="Choose File Notes Folder"
        ),
        "FileOpen(offer_select_folder)": FileOpen(
            location=str(tmp_path),
            title="Import once (files or one folder)",
            offer_select_folder=True,
        ),
        "EnhancedSelectDirectory": EnhancedSelectDirectory(
            location=str(tmp_path), title="Select directory"
        ),
    }


@pytest.mark.parametrize(
    "name",
    ["SelectDirectory", "FileOpen(offer_select_folder)", "EnhancedSelectDirectory"],
)
async def test_every_folder_offering_picker_opens_on_its_path_field(
    tmp_path: Path, name: str
) -> None:
    """AC#4: the behaviour belongs to the family, not to one subclass."""
    (tmp_path / "sub").mkdir()
    dialog = _folder_offering_dialogs(tmp_path)[name]
    app = _PickerHost(dialog)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        await pilot.pause()
        field = _field(dialog)
        assert app.focused is field, (
            f"{name} opened focused on "
            f"{type(app.focused).__name__}, not its path field"
        )


@pytest.mark.parametrize("name", ["FileOpen", "EnhancedFileOpen"])
async def test_a_plain_file_picker_still_opens_on_its_listing(
    tmp_path: Path, name: str
) -> None:
    """Negative control: a file-only picker keeps browsing-first focus.

    Character import, skill folders, TTS models and every other file-only
    caller are untouched -- there Enter on a listing row is the natural
    first keystroke.
    """
    dialog = {
        "FileOpen": FileOpen(location=str(tmp_path)),
        "EnhancedFileOpen": EnhancedFileOpen(location=str(tmp_path)),
    }[name]
    app = _PickerHost(dialog)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.focused, DirectoryNavigation), (
            f"{name} no longer opens on its listing: "
            f"focus is on {type(app.focused).__name__}"
        )
