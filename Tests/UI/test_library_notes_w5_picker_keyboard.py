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


async def test_folder_files_reaches_and_edits_a_file_with_no_mouse(
    tmp_path: Path,
) -> None:
    """AC#3: choose a folder and edit a file in it, keyboard only."""
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
