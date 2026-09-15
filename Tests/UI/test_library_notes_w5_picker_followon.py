"""task-32611 + task-32643: one folder picker, told apart from a file picker.

Critique #4 (dev 77eb2601a6) opened all three Notes folder doors and found
three dialogs:

* Import once            -- "File name or path", Open / Select folder / Cancel
* Folder files           -- "Folder path", Select / Cancel
* Keep a folder synced   -- "File name", though it can ONLY return a folder

...all three listing files and folders interleaved in "Discovery order", and
none of them saying how many notes a folder holds or which one is a vault.

These pins read what production constructs and renders, never a value the test
handed the widget under test. The cheap halves (`project_records`,
`count_folder_notes`) are pinned as the pure functions they are; the doors are
pinned on their real call sites.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, ListView, Select, Static

import Tests.UI._optional_module_stubs  # noqa: F401
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, FileSave, SelectDirectory
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import (
    FileSystemPickerScreen,
)
from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation
from tldw_chatbook.Third_Party.textual_fspicker.parts import (
    progressive_directory_navigation as pdn,
)
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedSelectDirectory

pytestmark = pytest.mark.asyncio

CRITIQUE_COMPACT = (100, 30)


class _PickerHost(App[None]):
    """Minimal host that pushes one real picker and keeps its result."""

    def __init__(self, dialog: FileSystemPickerScreen) -> None:
        super().__init__()
        self._dialog = dialog
        self.picked: Path | None = None
        self.dismissed = False

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        def done(result: Path | None) -> None:
            self.picked = result
            self.dismissed = True

        await self.push_screen(self._dialog, done)


async def _wait_for_picker(pilot, *, attempts: int = 200) -> FileSystemPickerScreen:
    for _ in range(attempts):
        top = pilot.app.screen_stack[-1]
        if isinstance(top, FileSystemPickerScreen):
            await pilot.pause()
            await pilot.pause()
            return top
        await pilot.pause()
    raise AssertionError("the picker never opened")


async def _wait_until(predicate, *, seconds: float = 4.0):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.01)


def _rows(nav: DirectoryNavigation) -> list[str]:
    return [
        option.location.name for option in nav.options if option.location.name != ".."
    ]


def _row_text(nav: DirectoryNavigation, name: str) -> str:
    """Everything one painted row says, read off the option's own prompt."""
    for option in nav.options:
        if option.location.name == name:
            record = option.record
            return f"{record.display_name}|{record.size_text}"
    raise AssertionError(f"no row for {name!r}; rows were {_rows(nav)}")


def _tree(tmp_path: Path) -> Path:
    """A vault-shaped folder: files and folders interleaved by creation."""
    root = tmp_path / "notes-root"
    root.mkdir()
    for name in ("Reading", "Inbox", "Archive"):
        (root / name).mkdir()
    (root / "scratch.txt").write_text("x", encoding="utf-8")
    (root / "README.md").write_text("x", encoding="utf-8")
    # "Reading" is an Obsidian vault holding two notes at its own top level
    # and a third one a level down, which must NOT be counted.
    (root / "Reading" / ".obsidian").mkdir()
    (root / "Reading" / "one.md").write_text("x", encoding="utf-8")
    (root / "Reading" / "two.markdown").write_text("x", encoding="utf-8")
    (root / "Reading" / "deeper").mkdir()
    (root / "Reading" / "deeper" / "three.md").write_text("x", encoding="utf-8")
    return root


# --- task-32611 AC#1/#2: the three doors, and which component each pushes ---


async def test_keep_a_folder_synced_pushes_the_folder_only_picker(tmp_path) -> None:
    """AC#1/#2: the door that can only answer with a folder uses that dialog.

    Reads the production handler body itself (no copy of it here). On dev it
    pushed ``FileOpen(offer_select_folder=True)`` -- the files-AND-folder
    dialog -- so the field said "File name", the placeholder said "File name
    or path", and every file in the folder was listed as though pickable,
    under the title "Choose a folder to keep synced".
    """
    from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
        LibraryNotesController,
    )

    pushed = []

    class _Event:
        def stop(self) -> None:
            pass

    class _Door:
        app = type("_App", (), {"push_screen": staticmethod(lambda s, c: pushed.append(s))})()

        def _library_notes_sync_browse_location(self) -> str:
            return str(tmp_path)

    LibraryNotesController.handle_library_notes_lasting_folder_requested(
        _Door(), _Event()
    )

    assert len(pushed) == 1
    dialog = pushed[0]
    assert isinstance(dialog, SelectDirectory), (
        '"Keep a folder synced" must push the folder-only picker; it pushed '
        f"{type(dialog).__name__}"
    )
    assert dialog.RETURNS_A_FOLDER is True
    # AC#2: the buttons that can act on a folder, and no third one that
    # cannot -- `_offer_select_folder` is what adds "Open"'s sibling.
    assert getattr(dialog, "_offer_select_folder", False) is False


async def test_every_folder_door_shares_one_hint_and_button_grammar(
    tmp_path,
) -> None:
    """AC#3: same hint wording, same name for the button that commits a folder.

    Critique #4 read "Select folder to use this folder" on Import once and
    "Select to use this folder" on Folder files -- two sentences for one
    decision. The dialogs are built exactly as the three doors build them.
    """
    import_once = FileOpen(
        tmp_path, title="Import once (files or one folder)", offer_select_folder=True
    )
    folder_files = SelectDirectory(tmp_path, title="Choose File Notes Folder")
    keep_synced = SelectDirectory(tmp_path, title="Choose a folder to keep synced")

    hints = {d._hint_text() for d in (import_once, folder_files, keep_synced)}
    assert len(hints) == 1, f"three doors, {len(hints)} different hints: {hints}"
    assert "Select folder to use this folder" in hints.pop()

    for dialog in (folder_files, keep_synced):
        host = _PickerHost(dialog)
        async with host.run_test(size=CRITIQUE_COMPACT) as pilot:
            picker = await _wait_for_picker(pilot)
            assert str(picker.query_one("#select", Button).label) == "Select folder", (
                "the folder-only doors must name the confirm button for what "
                "it returns, like the files-and-folder door already does"
            )

    # Negative control: a file-only picker still says nothing about folders.
    assert FileOpen(tmp_path, title="Open")._hint_text() == ""


# --- task-32611 AC#4 / task-32643 AC#1: folders first, name-ascending -------


def test_folders_first_orders_folders_then_names_in_both_directions() -> None:
    """The projection itself, with no UI and no filesystem."""
    records = [
        pdn.FileRecord(Path("/x/scratch.txt"), False),
        pdn.FileRecord(Path("/x/Reading"), True),
        pdn.FileRecord(Path("/x/Archive"), True),
        pdn.FileRecord(Path("/x/README.md"), False),
    ]
    visible, _ = pdn.project_records(
        records,
        show_hidden=False,
        query="",
        file_filter=None,
        sort_key="folders",
        descending=False,
        cancelled=Event(),
    )
    assert [r.location.name for r in visible] == [
        "Archive",
        "Reading",
        "README.md",
        "scratch.txt",
    ]

    visible, _ = pdn.project_records(
        records,
        show_hidden=False,
        query="",
        file_filter=None,
        sort_key="folders",
        descending=True,
        cancelled=Event(),
    )
    assert [r.location.name for r in visible] == [
        "Reading",
        "Archive",
        "scratch.txt",
        "README.md",
    ], "Descending must reverse the NAMES and keep folders on top"


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda p: SelectDirectory(p, title="Choose File Notes Folder"), "folders"),
        (
            lambda p: FileOpen(p, title="Import once", offer_select_folder=True),
            "folders",
        ),
        (lambda p: EnhancedSelectDirectory(location=p, title="Pick"), "folders"),
        # Negative controls: a picker that cannot return a folder is untouched.
        (lambda p: FileOpen(p, title="Open"), "discovery"),
        (lambda p: FileSave(p, title="Save"), "discovery"),
    ],
)
async def test_default_listing_order_follows_what_the_dialog_returns(
    tmp_path, build, expected
) -> None:
    """AC#4: folders first on a folder door, discovery order everywhere else.

    Asserted on the NAVIGATION, not just on the Select: the control and the
    listing disagreeing is the same defect one step later.
    """
    host = _PickerHost(build(tmp_path))
    async with host.run_test(size=CRITIQUE_COMPACT) as pilot:
        picker = await _wait_for_picker(pilot)
        nav = picker.query_one(DirectoryNavigation)
        await _wait_until(lambda: nav.sort_key == expected)
        assert str(picker.query_one("#listing-sort", Select).value) == expected
        # "Discovery order" stays available on both (AC#4's second half).
        assert "discovery" in [value for _, value in pdn.SORT_OPTIONS]


async def test_the_folder_door_lists_folders_before_files(tmp_path) -> None:
    """AC#4 end to end, through the real dialog and a real directory."""
    root = _tree(tmp_path)
    host = _PickerHost(
        FileOpen(root, title="Import once", offer_select_folder=True)
    )
    async with host.run_test(size=CRITIQUE_COMPACT) as pilot:
        picker = await _wait_for_picker(pilot)
        nav = picker.query_one(DirectoryNavigation)
        await _wait_until(lambda: len(_rows(nav)) == 5)
        assert _rows(nav) == [
            "Archive",
            "Inbox",
            "Reading",
            "README.md",
            "scratch.txt",
        ]


# --- task-32643 AC#2: a bounded note count ---------------------------------


def test_the_note_count_reads_one_folder_and_stops_at_the_ceiling(tmp_path) -> None:
    """AC#2: depth-1 and capped -- the badge may not walk a subtree."""
    root = _tree(tmp_path)
    # Two notes at the top of "Reading"; the third lives in "deeper/".
    assert pdn.count_folder_notes(root / "Reading") == 2
    assert pdn.count_folder_notes(root / "Inbox") == 0
    assert pdn.count_folder_notes(root / "missing") is None

    wide = tmp_path / "wide"
    wide.mkdir()
    for index in range(12):
        (wide / f"n{index}.md").write_text("x", encoding="utf-8")
    assert pdn.count_folder_notes(wide, ceiling=5) == 5

    # The ceiling is a read bound, not just a display bound: the scan stops.
    seen = 0
    real_scandir = pdn.os.scandir

    class _Counting:
        def __init__(self, path):
            self._path = path

        def __enter__(self):
            self._scan = real_scandir(self._path)
            return self._generate()

        def _generate(self):
            nonlocal seen
            for entry in self._scan:
                seen += 1
                yield entry

        def __exit__(self, *args):
            self._scan.close()

    pdn.os.scandir = _Counting
    try:
        assert pdn.count_folder_notes(wide, ceiling=3) == 3
    finally:
        pdn.os.scandir = real_scandir
    assert seen <= 3, f"the ceiling read {seen} entries, not at most 3"


async def test_folder_rows_carry_a_note_count_and_a_vault_marker(tmp_path) -> None:
    """AC#2/AC#4 as the user meets them: on the row, in the real dialog."""
    root = _tree(tmp_path)
    host = _PickerHost(
        SelectDirectory(root, title="Choose File Notes Folder", notes_context="probe")
    )
    async with host.run_test(size=CRITIQUE_COMPACT) as pilot:
        picker = await _wait_for_picker(pilot)
        nav = picker.query_one(DirectoryNavigation)
        await _wait_until(lambda: "2 notes" in _row_text(nav, "Reading"))
        assert pdn.VAULT_MARKER.strip() in _row_text(nav, "Reading")
        await _wait_until(lambda: "0 notes" in _row_text(nav, "Inbox"))
        assert pdn.VAULT_MARKER.strip() not in _row_text(nav, "Inbox"), (
            "a plain folder must say nothing extra"
        )


async def test_a_picker_without_a_notes_context_draws_no_badge(tmp_path) -> None:
    """Negative control: "12 notes" is noise in a model-file picker."""
    root = _tree(tmp_path)
    host = _PickerHost(SelectDirectory(root, title="Select directory"))
    async with host.run_test(size=CRITIQUE_COMPACT) as pilot:
        picker = await _wait_for_picker(pilot)
        nav = picker.query_one(DirectoryNavigation)
        await _wait_until(lambda: len(_rows(nav)) == 3)
        for _ in range(20):
            await pilot.pause()
        assert _row_text(nav, "Reading") == "Reading|"


# --- task-32643 AC#4: ONE vault predicate ----------------------------------


def test_the_listing_asks_the_apps_one_vault_predicate(tmp_path, monkeypatch) -> None:
    """AC#4: the picker CALLS the shared detection; it has no copy of it.

    Replacing `folder_is_obsidian_vault` -- the function
    `library_notes_sync_controller` also routes through -- must change what
    the listing marks. A second `.obsidian` test written inside the picker
    would keep this green while the two surfaces drifted.
    """
    from tldw_chatbook.Notes import note_import_discovery

    monkeypatch.setattr(
        note_import_discovery, "folder_is_obsidian_vault", lambda folder: True
    )
    plain = tmp_path / "plain"
    plain.mkdir()
    record = pdn.read_folder_summary(pdn.FileRecord(plain, True))
    assert record.is_vault is True
    assert pdn.VAULT_MARKER.strip() in record.display_name

    monkeypatch.setattr(
        note_import_discovery, "folder_is_obsidian_vault", lambda folder: False
    )
    record = pdn.read_folder_summary(pdn.FileRecord(plain, True))
    assert record.is_vault is False


def test_the_sync_setup_and_the_picker_share_that_predicate(tmp_path) -> None:
    """The other half of AC#4: task-32641's surface reads the same function."""
    from tldw_chatbook.Notes.note_import_discovery import folder_is_obsidian_vault
    from tldw_chatbook.UI.Library_Modules import library_notes_sync_controller

    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    plain = tmp_path / "plain"
    plain.mkdir()

    for folder, expected in ((vault, True), (plain, False)):
        assert folder_is_obsidian_vault(folder) is expected
        assert (
            library_notes_sync_controller._carries_obsidian_marker(str(folder))
            is expected
        )


# --- task-32643 AC#3: recent roots, one keystroke away ---------------------


async def test_a_recent_root_is_offered_and_enter_chooses_it(
    tmp_path, monkeypatch
) -> None:
    """AC#3: ctrl+r, Enter -- no navigation, and the root IS the answer."""
    root = _tree(tmp_path)
    recent = tmp_path / "last-week"
    recent.mkdir()

    from tldw_chatbook.Third_Party.textual_fspicker import base_dialog

    monkeypatch.setattr(
        base_dialog.FileSystemPickerScreen,
        "_get_recent_paths",
        lambda self: [recent] if self._notes_context else [],
    )

    host = _PickerHost(
        SelectDirectory(root, title="Choose File Notes Folder", notes_context="probe")
    )
    async with host.run_test(size=CRITIQUE_COMPACT) as pilot:
        picker = await _wait_for_picker(pilot)
        assert [
            str(item.data) for item in picker.query("#recent-list ListItem")
        ] == [str(recent)]

        await pilot.press("ctrl+r")
        await pilot.pause()
        listing = picker.query_one("#recent-list", ListView)
        assert pilot.app.focused is listing, (
            "the offered roots must take focus, or reaching one still costs "
            "the navigation they exist to replace"
        )
        await pilot.press("enter")
        await _wait_until(lambda: host.dismissed)
        assert host.picked == recent


async def test_recent_roots_write_and_read_through_one_store(
    tmp_path, monkeypatch
) -> None:
    """AC#3's persistence: the picker reads what a selection wrote.

    Both halves run for real against an isolated config; only the config file
    is redirected.
    """
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Library import library_browse_location

    profile = tmp_path / "profile.toml"
    profile.write_text("[general]\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(profile))
    config_module.load_settings(force_reload=True)

    chosen = tmp_path / "chosen"
    chosen.mkdir()
    context = library_browse_location.picker_recent_context("library.notes_sync")
    generation = library_browse_location.claim_browse_directory(
        "library.notes_sync", "last_directory"
    )
    library_browse_location.remember_browse_directory(
        "library.notes_sync",
        "last_directory",
        chosen,
        generation,
        recent_context=context,
    )

    dialog = SelectDirectory(tmp_path, title="Choose", notes_context=context)
    assert dialog._get_recent_paths() == [chosen]
    # The start directory it has always written is still written.
    assert library_browse_location.validated_browse_directory(
        config_module.get_cli_setting("library.notes_sync", "last_directory", None)
    ) == chosen


async def test_an_empty_recents_panel_still_opens_and_keeps_focus(tmp_path) -> None:
    """The panel's OTHER contract, which AC#3 must not cost.

    ``test_fspicker_keyboard_save`` opens all three transients on an empty
    picker to pin the Escape-peel order, so ctrl+r must keep opening with
    nothing to show -- and taking focus into an empty list would strand the
    keyboard user there.
    """
    host = _PickerHost(SelectDirectory(tmp_path, title="Select directory"))
    async with host.run_test(size=CRITIQUE_COMPACT) as pilot:
        picker = await _wait_for_picker(pilot)
        before = pilot.app.focused
        await pilot.press("ctrl+r")
        await pilot.pause()
        assert picker.show_recent is True
        assert picker.query_one("#recent-locations").has_class("visible")
        assert pilot.app.focused is before


async def test_the_hint_line_renders_the_shared_wording(tmp_path) -> None:
    """The hint is not just computed -- it reaches the screen on both doors."""
    for dialog in (
        SelectDirectory(tmp_path, title="Choose a folder to keep synced"),
        FileOpen(tmp_path, title="Import once", offer_select_folder=True),
    ):
        host = _PickerHost(dialog)
        async with host.run_test(size=CRITIQUE_COMPACT) as pilot:
            picker = await _wait_for_picker(pilot)
            painted = str(picker.query_one("#picker-hint-line", Static).renderable)
            assert painted == "Enter Open  ·  Select folder to use this folder"
