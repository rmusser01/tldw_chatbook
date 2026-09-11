"""Add from files and Import once layout/copy contracts (import-ux wave).

Covers tasks 32125 (chooser), 32130 (receipt copy), 32134 (source changes)
and 32135 (one-line review rows with per-group bulk actions).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Collapsible, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Library.library_note_import_state import (
    LibraryNoteImportItemSnapshot,
    LibraryNoteImportSnapshot,
)
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Library import library_browse_location as browse_location_module
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_note_import_canvas import (
    LibraryNoteImportCanvas,
)
from tldw_chatbook.Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)
from tldw_chatbook.Library.library_notes_state import LibraryNotesListState
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas
from tldw_chatbook.app import TldwCli

pytestmark = pytest.mark.asyncio


def _plain(widget: Static) -> str:
    return getattr(widget.renderable, "plain", str(widget.renderable))


def _import_snapshot(**changes: object) -> LibraryNoteImportSnapshot:
    snapshot = LibraryNoteImportSnapshot(
        phase="select",
        selected_names=(),
        selection_kind="",
        destination="",
        status_line="Choose one or more files, or one folder.",
        preview_items=(),
        page=1,
        page_count=1,
        can_check=False,
        check_disabled_reason="Choose a source first.",
        can_import=False,
        import_disabled_reason="Check the selection before importing.",
    )
    return replace(snapshot, **changes)


def _item(index: int, **changes: object) -> LibraryNoteImportItemSnapshot:
    item = LibraryNoteImportItemSnapshot(
        item_id=f"item-{index}",
        name=f"vault/Archive/note-{index}.md",
        classification="new",
        action="create_new",
        reason="Ready to import as a new note.",
        effect_summary="Content: create 1 new note.",
        membership_summary="Create in vault / Archive.",
    )
    return replace(item, **changes)


class _ChooserHost(ConsolidatedCSSApp):
    CSS_PATH = TldwCli.CSS_PATH

    def __init__(self) -> None:
        super().__init__()
        self.snapshot = initial_lasting_sync_snapshot()

    def compose(self) -> ComposeResult:
        yield LibraryNotesAddFromFilesCanvas(self.snapshot, id="chooser")


class _ImportHost(ConsolidatedCSSApp):
    CSS_PATH = TldwCli.CSS_PATH

    def __init__(self, snapshot: LibraryNoteImportSnapshot) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.messages: list[object] = []

    def compose(self) -> ComposeResult:
        yield LibraryNoteImportCanvas(self.snapshot, id="library-note-import-canvas")

    def on_library_note_import_canvas_change_source_requested(
        self, message: LibraryNoteImportCanvas.ChangeSourceRequested
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_clear_source_requested(
        self, message: LibraryNoteImportCanvas.ClearSourceRequested
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_group_action_requested(
        self, message: LibraryNoteImportCanvas.GroupActionRequested
    ) -> None:
        self.messages.append(message)


# --- task-32125 -----------------------------------------------------------


async def test_both_relationships_are_buttons_under_their_own_descriptions() -> None:
    """Import once sits with Keep a folder synced, not in the pinned bar."""
    app = _ChooserHost()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        body = app.query_one("#notes-sync-body")
        children = list(body.children)
        ids = [child.id for child in children]

        assert "notes-add-import-once" in ids
        assert ids.index("notes-add-import-once") < ids.index("notes-add-keep-synced")
        import_index = ids.index("notes-add-import-once")
        keep_index = ids.index("notes-add-keep-synced")
        assert "Import once —" in _plain(children[import_index - 1])
        assert "Keep a folder synced —" in _plain(children[keep_index - 1])

        pinned = app.query_one("#notes-sync-pinned-actions")
        assert [button.id for button in pinned.query(Button)] == ["notes-sync-back"]


async def test_chooser_header_names_no_relationship_before_one_is_chosen() -> None:
    """The authority line stops announcing Lasting sync on the chooser."""
    canvas = LibraryNotesCanvas(
        mode="lasting_add",
        lasting_sync_snapshot=initial_lasting_sync_snapshot(),
    )

    copy = canvas._authority_copy()

    assert "Lasting sync" not in copy
    assert "Add from files" in copy


# --- task-32130 -----------------------------------------------------------


async def test_receipt_discloses_every_skipped_path_and_reason() -> None:
    """The receipt names the skipped files instead of only counting them."""
    app = _ImportHost(
        _import_snapshot(
            phase="receipt",
            status_line="Import completed.",
            receipt_line="61 imported · 0 updated · 11 skipped · 0 failed",
            receipt_detail="Import finished · 61 notes created · 11 files skipped",
            skipped_count=2,
            skipped_items=(
                ("vault/.obsidian/app.json", "Not a note file (app configuration)."),
                ("vault/Inbox/Untitled.md", "Empty file — nothing to import."),
            ),
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        disclosure = app.query_one("#note-import-skipped", Collapsible)
        assert "Skipped (2)" in str(disclosure.title)
        rows = "\n".join(
            _plain(row) for row in disclosure.query(".note-import-skipped-row")
        )
        assert "vault/.obsidian/app.json" in rows
        assert "Not a note file (app configuration)." in rows
        assert "Empty file — nothing to import." in rows
        assert "61 notes created" in _plain(
            app.query_one("#note-import-receipt-detail", Static)
        )


# --- task-32134 -----------------------------------------------------------


async def test_a_chosen_folder_can_be_changed_or_cleared() -> None:
    """A wrong folder is recoverable without leaving Import once."""
    app = _ImportHost(
        _import_snapshot(
            phase="select",
            selected_names=("vault",),
            selection_kind="folder",
            status_line="1 folder selected.",
            can_check=True,
            check_disabled_reason="",
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # A folder import is exclusive, so Add another file stays absent.
        assert not app.query("#note-import-add-source")
        app.query_one("#note-import-change-source", Button).press()
        app.query_one("#note-import-clear-source", Button).press()
        await pilot.pause()

    assert isinstance(app.messages[0], LibraryNoteImportCanvas.ChangeSourceRequested)
    assert isinstance(app.messages[1], LibraryNoteImportCanvas.ClearSourceRequested)


async def test_selected_files_keep_add_another_file_beside_change_and_clear() -> None:
    """File selections keep their documented Add another file control."""
    app = _ImportHost(
        _import_snapshot(
            phase="destination",
            selected_names=("draft.md",),
            selection_kind="files",
            status_line="1 file selected.",
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.query_one("#note-import-add-source", Button)
        assert app.query_one("#note-import-change-source", Button)
        assert app.query_one("#note-import-clear-source", Button)


async def test_notes_list_offers_last_import_while_a_session_receipt_exists() -> None:
    """The list surfaces the retained receipt after Back to Notes (32134)."""
    class _ListHost(ConsolidatedCSSApp):
        CSS_PATH = TldwCli.CSS_PATH

        def compose(self) -> ComposeResult:
            yield LibraryNotesCanvas(
                list_state=LibraryNotesListState(
                    rows=(),
                    header_copy="Notes (0)",
                    status_copy="",
                    empty_copy="No notes yet. Create one to see it here.",
                ),
                mode="list",
                import_receipt_available=True,
                id="library-notes-canvas",
            )

    app = _ListHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.query_one("#library-notes-import-receipt", Button)


# --- task-32135 -----------------------------------------------------------


async def test_review_rows_are_one_line_with_their_controls_adjacent() -> None:
    """Path, action and destination share one row with Skip/Create."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 1 item before import.",
            preview_items=(_item(1),),
            can_import=True,
            import_disabled_reason="",
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        row = app.query_one(".note-import-row", Horizontal)
        text = _plain(row.query_one(".note-import-row-text", Static))
        assert "vault/Archive/note-1.md" in text
        assert "create 1 new note" in text
        assert "vault / Archive" in text
        assert row.query_one("#note-import-action-item-1-skip", Button)
        assert row.query_one("#note-import-action-item-1-create", Button)
        assert row.size.height == 1


async def test_each_review_group_offers_skip_all_and_create_all() -> None:
    """A 71-item review can be settled per action class, not row by row."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 3 items before import.",
            preview_items=(
                _item(1),
                _item(2),
                _item(
                    3,
                    classification="unsupported",
                    action="skip",
                    reason="This file type is not supported.",
                    effect_summary="Content: no change.",
                    membership_summary="Folder placement: no change.",
                ),
            ),
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        headings = [
            _plain(heading)
            for heading in app.query(".note-import-group-heading").results(Static)
        ]
        assert "New (2)" in headings
        assert "Unsupported (1)" in headings
        app.query_one("#note-import-group-new-skip", Button).press()
        await pilot.pause()
        # An unsupported group cannot create anything, so it offers Skip only.
        assert not app.query("#note-import-group-unsupported-create")
        assert app.query_one("#note-import-group-new-create", Button)

    message = app.messages[-1]
    assert isinstance(message, LibraryNoteImportCanvas.GroupActionRequested)
    assert (message.classification, message.action) == ("new", "skip")


async def test_review_shows_at_least_fifteen_rows_at_235x52() -> None:
    """A full review page fits on one screen at the wide terminal size."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 25 items before import.",
            preview_items=tuple(_item(index) for index in range(1, 26)),
            can_import=True,
            import_disabled_reason="",
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        visible = app.screen._compositor.visible_widgets
        rows = [row for row in app.query(".note-import-row") if row in visible]
        assert len(rows) >= 15


async def test_review_reserves_an_options_slot_above_the_groups() -> None:
    """Task 5's Obsidian toggle has a named home above the grouped rows."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 1 item before import.",
            preview_items=(_item(1),),
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        body = app.query_one("#note-import-body")
        ids = [child.id for child in body.children]
        assert "notes-import-review-options" in ids
        # An empty 1fr slot would push every group to the bottom of the body.
        assert app.query_one("#notes-import-review-options").size.height == 0
        first_heading = next(
            child
            for child in body.children
            if child.has_class("note-import-group-row")
        )
        assert ids.index("notes-import-review-options") < body.children.index(
            first_heading
        )


async def test_a_clipped_row_still_reaches_its_effect_and_destination() -> None:
    """At 60 columns the row clips, so the rest stays reachable (32135 review)."""
    from textual.containers import Container

    snapshot = _import_snapshot(
        phase="review",
        status_line="Review 1 item before import.",
        preview_items=(_item(1),),
        can_import=True,
        import_disabled_reason="",
    )

    class _CompactHost(ConsolidatedCSSApp):
        CSS_PATH = TldwCli.CSS_PATH

        def compose(self) -> ComposeResult:
            with Container(id="library-canvas", classes="library-notes-compact"):
                yield LibraryNoteImportCanvas(
                    snapshot,
                    compact=True,
                    id="library-note-import-canvas",
                )

    app = _CompactHost()
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        summary = app.query_one(".note-import-row-text", Static)
        # The row is clipped at this width…
        assert summary.size.width < len(_plain(summary))
        # …so the whole sentence is on the tooltip,
        assert "create 1 new note" in str(summary.tooltip)
        assert "vault / Archive" in str(summary.tooltip)
        # and the destination repeats on its own line.
        destination = app.query_one(".note-import-row-destination", Static)
        assert destination.display is True
        assert "vault / Archive" in _plain(destination)


async def test_every_review_control_stays_inside_a_narrow_viewport() -> None:
    """A row's trailing buttons must be reachable, not clipped off-screen."""
    from textual.containers import Container

    snapshot = _import_snapshot(
        phase="review",
        status_line="Review 2 items before import.",
        preview_items=(
            _item(
                1,
                classification="uncertain_match",
                action="create_new",
                reason="This source may match an existing note.",
                uncertain=True,
            ),
            _item(
                2,
                classification="changed_repeat",
                action="update_existing",
                reason="This source differs from an existing note.",
                can_update=True,
                effect_summary="Content: keep existing content.",
                membership_summary="Folder placement: unchanged.",
            ),
        ),
        can_import=True,
        import_disabled_reason="",
    )

    class _CompactHost(ConsolidatedCSSApp):
        CSS_PATH = TldwCli.CSS_PATH

        def compose(self) -> ComposeResult:
            with Container(id="library-canvas", classes="library-notes-compact"):
                yield LibraryNoteImportCanvas(
                    snapshot,
                    compact=True,
                    id="library-note-import-canvas",
                )

    app = _CompactHost()
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        body = app.query_one("#note-import-body")
        buttons = list(body.query(Button))
        assert len(buttons) >= 9  # both rows' full control sets are mounted
        overflowing = [
            button.id
            for button in buttons
            if button.region.right > body.content_region.right
            or button.region.x < body.content_region.x
        ]
        assert overflowing == []


async def test_a_wide_row_does_not_repeat_its_destination() -> None:
    """The second line is compact-only; the wide row already ends in it."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 1 item before import.",
            preview_items=(_item(1),),
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        assert not app.query(".note-import-row-destination")


# --- task-32174 (last-used picker start directory) -------------------------
#
# Import once and Keep a folder synced both open a vendored, plain
# ``FileOpen`` (``Third_Party.textual_fspicker``) with no built-in
# per-context memory of its own -- unlike ``EnhancedFileOpen``
# (``Widgets/enhanced_file_picker.py``), which already remembers a start
# directory per ``context=``. These pickers get the same behaviour
# caller-side instead, mirroring ``LibraryScreen._library_ingest_browse_
# location`` (see ``Tests/UI/test_library_screen.py::test_ingest_browse_
# location_prefers_last_used_then_home``). ``LibraryScreen.__init__`` is
# pure attribute setup -- no I/O, no compose(), no worker starts -- so
# constructing one with a throwaway app stand-in is enough to reach these
# flows, with no mount required.
#
# PR #2554 review: these go through the production picker-opening methods
# and their registered callbacks, not the private resolver/persist helpers
# -- dropping ``location=`` from the ``FileOpen`` call, or the persistence
# call from the callback, has to fail something here. The shared
# validation/ordering contract lives in
# ``Tests/Library/test_library_browse_location.py``.


def _minimal_notes_screen() -> LibraryScreen:
    return LibraryScreen(MagicMock())


class _PickerHost:
    """App stand-in that captures the picker a flow pushes, and its callback."""

    def __init__(self) -> None:
        self.pushed = None
        self.callback = None

    def push_screen(self, screen, callback=None):
        self.pushed = screen
        self.callback = callback
        return None


def _as_host(screen: LibraryScreen, host: _PickerHost):
    return patch.object(
        LibraryScreen, "app", new_callable=lambda: property(lambda self: host)
    )


def _capture_saved_directories(monkeypatch) -> list[tuple]:
    saved: list[tuple] = []
    monkeypatch.setattr(
        browse_location_module,
        "save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    return saved


async def test_import_once_picker_opens_at_the_remembered_directory(
    tmp_path, monkeypatch
) -> None:
    """task-32174 AC#1: the real picker instance gets the remembered start."""
    screen = _minimal_notes_screen()
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.get_cli_setting",
        lambda section, key=None, default=None: (
            str(tmp_path)
            if (section, key) == ("library.notes_import", "last_directory")
            else default
        ),
    )
    host = _PickerHost()
    with _as_host(screen, host):
        screen._push_library_note_import_picker()

    assert isinstance(host.pushed, FileOpen)
    assert Path(host.pushed._location) == tmp_path.resolve()


async def test_import_once_picker_opens_at_home_without_a_usable_memory(
    tmp_path, monkeypatch
) -> None:
    """A remembered value that is relative, traversing or gone is refused --
    it used to be handed to the picker after a bare ``is_dir()`` probe."""
    screen = _minimal_notes_screen()
    (tmp_path / "relative-dir").mkdir()
    monkeypatch.chdir(tmp_path)
    for remembered in ("relative-dir", "../..", str(tmp_path / "deleted"), None):
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.library_screen.get_cli_setting",
            lambda *args, _value=remembered, **kwargs: _value,
        )
        host = _PickerHost()
        with _as_host(screen, host):
            screen._push_library_note_import_picker()
        assert Path(host.pushed._location) == Path.home(), remembered


async def test_import_once_remembers_the_directory_it_selected(
    tmp_path, monkeypatch
) -> None:
    """AC#1: completing a selection through the picker's own callback is
    what persists the directory -- a picked file contributes its parent."""
    screen = _minimal_notes_screen()
    picked = tmp_path / "note.md"
    picked.write_text("# hi", encoding="utf-8")
    screen._library_note_import_controller = MagicMock()
    monkeypatch.setattr(screen, "run_worker", lambda work, **kwargs: work())
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.get_cli_setting",
        lambda *args, **kwargs: None,
    )
    saved = _capture_saved_directories(monkeypatch)

    host = _PickerHost()
    with _as_host(screen, host):
        screen._push_library_note_import_picker()
        await host.callback(picked)

    assert saved == [("library.notes_import", "last_directory", str(tmp_path))]
    screen._library_note_import_controller.accept_selected_path.assert_called_once()


async def test_import_once_and_ingest_keep_independent_last_directories(
    tmp_path, monkeypatch
) -> None:
    """Each picker context is keyed independently -- one cannot leak into
    another the way ``Tests/UI/test_file_picker_start_dir.py`` pins for
    ``EnhancedFileOpen`` contexts."""
    screen = _minimal_notes_screen()
    store: dict[tuple[str, str], str] = {}

    def fake_get(section, key=None, default=None):
        return store.get((section, key), default)

    def fake_save(section, key, value):
        store[(section, key)] = value
        return True

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.get_cli_setting", fake_get
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.save_setting_to_cli_config",
        fake_save,
    )
    monkeypatch.setattr(
        browse_location_module, "save_setting_to_cli_config", fake_save
    )
    monkeypatch.setattr(screen, "run_worker", lambda work, **kwargs: work())
    screen._library_note_import_controller = MagicMock()

    ingest_dir = tmp_path / "ingest-dir"
    notes_dir = tmp_path / "notes-import-dir"
    ingest_dir.mkdir()
    notes_dir.mkdir()
    (notes_dir / "picked.md").write_text("# hi", encoding="utf-8")

    screen._remember_library_ingest_location(ingest_dir / "picked.txt")
    host = _PickerHost()
    with _as_host(screen, host):
        screen._push_library_note_import_picker()
        await host.callback(notes_dir / "picked.md")

    assert store[("library.ingest", "last_directory")] == str(ingest_dir)
    assert store[("library.notes_import", "last_directory")] == str(notes_dir)
    assert screen._library_ingest_browse_location() == str(ingest_dir)
    assert screen._library_note_import_browse_location() == str(notes_dir)


def _folder_requested_event() -> MagicMock:
    event = MagicMock()
    event.stop = MagicMock()
    return event


async def test_notes_sync_picker_opens_at_the_remembered_directory(
    tmp_path, monkeypatch
) -> None:
    """task-32174 AC#2: Keep a folder synced remembers its own directory."""
    controller = _minimal_notes_screen()._notes_controller
    monkeypatch.setattr(
        "tldw_chatbook.UI.Library_Modules.library_notes_controller.get_cli_setting",
        lambda section, key=None, default=None: (
            str(tmp_path)
            if (section, key) == ("library.notes_sync", "last_directory")
            else default
        ),
    )
    host = _PickerHost()
    with _as_host(controller._screen, host):
        controller.handle_library_notes_lasting_folder_requested(
            _folder_requested_event()
        )

    assert isinstance(host.pushed, FileOpen)
    assert Path(host.pushed._location) == tmp_path.resolve()


async def test_notes_sync_picker_opens_at_home_without_a_usable_memory(
    tmp_path, monkeypatch
) -> None:
    controller = _minimal_notes_screen()._notes_controller
    (tmp_path / "relative-dir").mkdir()
    monkeypatch.chdir(tmp_path)
    for remembered in ("relative-dir", str(tmp_path / "deleted"), None):
        monkeypatch.setattr(
            "tldw_chatbook.UI.Library_Modules.library_notes_controller."
            "get_cli_setting",
            lambda *args, _value=remembered, **kwargs: _value,
        )
        host = _PickerHost()
        with _as_host(controller._screen, host):
            controller.handle_library_notes_lasting_folder_requested(
                _folder_requested_event()
            )
        assert Path(host.pushed._location) == Path.home(), remembered


async def test_notes_sync_remembers_the_folder_it_selected(tmp_path, monkeypatch) -> None:
    """AC#2: the folder just picked becomes the next open's start directory,
    persisted from the picker's own callback."""
    screen = _minimal_notes_screen()
    controller = screen._notes_controller
    picked_folder = tmp_path / "sync-folder"
    picked_folder.mkdir()
    monkeypatch.setattr(
        "tldw_chatbook.UI.Library_Modules.library_notes_controller.get_cli_setting",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(screen, "run_worker", lambda work, **kwargs: work())
    saved = _capture_saved_directories(monkeypatch)

    host = _PickerHost()
    with _as_host(screen, host):
        controller.handle_library_notes_lasting_folder_requested(
            _folder_requested_event()
        )
        await host.callback(picked_folder)

    assert saved == [
        ("library.notes_sync", "last_directory", str(picked_folder))
    ]
    assert (
        controller._library_notes_sync_controller.snapshot.setup.folder
        == str(picked_folder)
    )


# --- task-32176 -----------------------------------------------------------


async def test_group_actions_say_they_only_change_this_page() -> None:
    """A bulk action settles the rendered page, so its label says so."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 2 items before import.",
            preview_items=(
                _item(1),
                _item(
                    2,
                    classification="unsupported",
                    action="skip",
                    reason="This file type is not supported.",
                ),
            ),
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        labels = {
            str(button.label)
            for button in app.query(".note-import-group-action").results(Button)
        }
        assert labels == {"Skip all on this page", "Create all on this page"}


# --- task-32257 -----------------------------------------------------------


async def test_a_disabled_import_control_carries_its_reason_as_text() -> None:
    """The reason lived on the tooltip, so the control said only 'unavailable'."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 1 item before import.",
            preview_items=(_item(1),),
            can_import=False,
            import_disabled_reason="Choose how to handle the folder name collision.",
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        label = str(app.query_one("#note-import-import", Button).label)

    assert "Import selected items" in label
    assert "Choose how to handle the folder name collision" in label


async def test_a_disabled_check_control_carries_its_reason_as_text() -> None:
    """Check selection shares the helper, so it shared the defect (32257)."""
    app = _ImportHost(
        _import_snapshot(
            phase="select",
            can_check=False,
            check_disabled_reason="Choose a source first.",
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        label = str(app.query_one("#note-import-check", Button).label)

    assert "Check selection" in label
    assert "Choose a source first" in label


def test_the_canvas_takes_its_non_importable_set_from_the_planner_enum() -> None:
    """task-32176: one source of truth for what cannot be imported."""
    from tldw_chatbook.Notes.note_import_plan_models import (
        NON_IMPORTABLE_CLASSIFICATIONS,
        ImportClassification,
    )
    from tldw_chatbook.Widgets.Library import library_note_import_canvas

    assert library_note_import_canvas._NON_IMPORTABLE == {
        classification.value for classification in NON_IMPORTABLE_CLASSIFICATIONS
    }
    # Every classification still needs a group label, which is the third place
    # the set used to be spelled out by hand.
    assert set(library_note_import_canvas._CLASSIFICATION_LABELS) == {
        classification.value for classification in ImportClassification
    }


# --- task-32258 (one import, three surfaces) -------------------------------
#
# The numbers are produced by the real chain -- real discovery, parser,
# planner, receipt ledger and executor over a real temp ChaChaNotes database
# -- because the disagreement being pinned is between two real denominators:
# the review counts SOURCES, while the ledger counts PLANNED CHANGES (one per
# note a source creates). A fake receipt could not reproduce it.


def _real_import_controller(root: Path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
    from tldw_chatbook.Notes.note_import_discovery import discover_import_sources
    from tldw_chatbook.Notes.note_import_execution_models import (
        approve_note_import_plan,
    )
    from tldw_chatbook.Notes.note_import_executor import (
        LocalNoteImportTarget,
        NoteImportExecutor,
    )
    from tldw_chatbook.Notes.note_import_parsers import parse_import_sources
    from tldw_chatbook.Notes.note_import_plan_models import ImportBounds
    from tldw_chatbook.Notes.note_import_planner import (
        analyze_root_collision,
        apply_item_override,
        classify_import_batch,
        confirm_uncertain_match,
        resolve_root_collision,
    )
    from tldw_chatbook.Notes.note_import_receipts import NoteImportReceiptRepository
    from tldw_chatbook.UI.Library_Modules.library_note_import_controller import (
        LibraryNoteImportController,
    )

    database = CharactersRAGDB(root / "chachanotes.db", "wave-import-test")
    published: list[object] = []
    controller = LibraryNoteImportController(
        bounds=ImportBounds(
            max_files=1_000,
            max_file_bytes=16 * 1024 * 1024,
            max_total_bytes=256 * 1024 * 1024,
            max_depth=32,
        ),
        database=lambda: database,
        folder_repository=lambda: LocalNoteFolderRepository(database),
        receipt_repository=lambda: NoteImportReceiptRepository(root / "receipts.db"),
        discover_import_sources=discover_import_sources,
        parse_import_sources=parse_import_sources,
        classify_import_batch=classify_import_batch,
        analyze_root_collision=analyze_root_collision,
        resolve_root_collision=resolve_root_collision,
        confirm_uncertain_match=confirm_uncertain_match,
        apply_item_override=apply_item_override,
        approve_note_import_plan=approve_note_import_plan,
        executor_factory=lambda db, folders, ledger: NoteImportExecutor(
            target=LocalNoteImportTarget(db=db, folder_repository=folders),
            receipt_repository=ledger,
        ),
        publish_snapshot=published.append,
        refresh_after_settlement=lambda: None,
    )
    return controller, published


async def test_one_import_leaves_review_progress_and_receipt_reconciled(
    tmp_path: Path,
) -> None:
    """Review said 66, progress said 67, the receipt said 59 + 8 (task-32258)."""
    vault = tmp_path / "vault"
    (vault / "Notes").mkdir(parents=True)
    (vault / "Notes" / "one.md").write_text("# One\n\nBody.\n", encoding="utf-8")
    (vault / "Notes" / "two.md").write_text("# Two\n\nBody.\n", encoding="utf-8")
    # One source, two notes: this is where the two denominators separate.
    (vault / "rows.csv").write_text(
        "title,content\nCSV one,first\nCSV two,second\n", encoding="utf-8"
    )
    (vault / "picture.png").write_bytes(b"not a note")

    (tmp_path / "profile").mkdir()
    controller, published = _real_import_controller(tmp_path / "profile")
    controller.accept_selected_path(vault, is_folder=True)
    await controller.check()

    review = controller.presentation_snapshot
    sources = len(controller.snapshot.plan.items)
    assert review.status_line == f"Review {sources} sources before import."

    progress_totals: list[int] = []
    inner = controller._publish_snapshot

    def capture(snapshot):
        if snapshot.phase == "importing":
            progress_totals.append(snapshot.progress_total)
        inner(snapshot)

    controller._publish_snapshot = capture
    await controller.approve_and_execute()

    settled = controller.presentation_snapshot
    receipt = controller.snapshot.receipt
    # Every surface counts something it names, and the two denominators are
    # reconciled out loud rather than left to contradict each other.
    assert set(progress_totals) == {receipt.total}
    assert receipt.total > sources  # the CSV source creates two notes
    assert (
        receipt.imported + receipt.updated + receipt.skipped + receipt.failed
        == receipt.total
    )
    assert settled.receipt_detail == (
        f"{receipt.total} planned changes from {sources} reviewed sources."
    )
    assert settled.receipt_line == (
        f"{receipt.imported} notes created · {receipt.skipped} file skipped"
    )
    # …and the outcome is stated once: the header carries the session state,
    # the receipt line the counts, the detail the denominators.
    assert settled.status_line == "Import completed."
    assert "created" not in settled.status_line
    assert "created" not in settled.receipt_detail
