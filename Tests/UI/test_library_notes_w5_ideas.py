"""Library ▸ Notes critique #4, wave 5 -- the three accepted IDEAS (G8).

Tasks 32640, 32641 and 32642: the ideas task-32627 accepted. See
``backlog/tasks/task-<id>*.md`` for the acceptance criteria; each test names
the task it pins.

Everything here reads production output -- the real mounted screen at the
critique's own terminal sizes, the real session draft, the real import
controller -- never a value the test just handed the code. Where a walk
matters it is walked with ``pilot.press("tab")``, never ``widget.focus()``.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
)
from Tests.UI.test_library_notes_w4_editor import (
    _build_notes_host,
    _open_first_note,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    library_note_location_line,
    library_note_property_block,
)

pytestmark = pytest.mark.asyncio

#: The critique's own terminal sizes (task-32614 measured at these three).
WIDE = (235, 52)
COMPACT = (100, 30)
NARROW = (60, 20)

#: A note whose Created/Modified/Version/Words spell out four properties --
#: the timestamps are what the joined line could not fit in a compact pane.
_DATED_NOTE = [
    {
        "id": "note-dated",
        "title": "Garden redesign",
        "content": "Body text with several words in it\n",
        "version": 3,
        "created_at": "2026-07-01T03:00:00+00:00",
        "last_modified": "2026-09-14T11:02:00+00:00",
    }
]


def _meta_rows(screen) -> list[str]:
    """Info's Properties Static, row by row, as it is rendered."""
    meta = screen.query_one("#library-note-context-meta", Static)
    return str(meta.renderable).split("\n")


async def _open_info(screen, pilot) -> None:
    screen.query_one("#library-note-context", Button).press()
    await _wait_for_condition(
        pilot,
        lambda: screen.query_one("#library-note-context-region").display,
        message="Info never opened.",
    )
    await pilot.pause()


# --- task-32642 AC#1/#4: one row per property, nothing dropped -------------


async def test_info_gives_every_property_its_own_row_at_235x52():
    """task-32642 AC#1/AC#4.

    Born red against the unfixed tree, where the whole Static was ONE row
    155 columns wide inside a 36-row pane:
    ``assert ['Created 2026-06-30 20:00 · 10w ago · Modified 2026-09-14
    04:02 · 1d ago · v3 · 7 words'] == ['Created', 'Modified', 'Version',
    'Words']``. Four facts shared a row; the pane had 21 spare ones.
    """
    host = _build_notes_host(notes=_DATED_NOTE)
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_info(screen, pilot)

        rows = _meta_rows(screen)
        labels = [row.split("  ")[0].strip() for row in rows]
        assert labels == ["Created", "Modified", "Version", "Words"], rows
        # AC#4: arrangement, not loss -- every fact the joined line carried
        # is still on screen, and the Static is as tall as it has rows.
        meta = screen.query_one("#library-note-context-meta", Static)
        assert meta.region.height == len(rows), (rows, meta.region.height)
        # Local-zone rendered, so the DATE is not asserted literally -- the
        # shape is: an absolute stamp beside a relative age, per row.
        assert all(
            re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} · \S+ ago$", row)
            for row in rows[:2]
        ), rows
        assert rows[2].endswith("v3"), rows
        assert rows[3].endswith("words"), rows


async def test_the_compact_property_block_keeps_one_column_and_no_truncation():
    """task-32642 AC#2.

    The pane is 46 columns wide at 100x30 and the compact sheet pinned this
    Static to ``height: 1``, so the 84-character joined line was cut. One
    property per row, unpadded, and every value present in full.

    Born red on the unfixed tree -- and the RED is worse than the finding
    said. Measured on 67bfde41d1 with the Python AND the three stylesheets
    reverted, ``#library-note-context-meta`` painted exactly one row::

        Created 2026-06-30 20:00 · 10w ago · Modified

    The Modified value, the version and the word count were all off the
    pane at 100x30: three of Info's four properties were unreadable there,
    not merely crowded.
    """
    host = _build_notes_host(notes=_DATED_NOTE)
    async with host.run_test(size=COMPACT) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_info(screen, pilot)

        rows = _meta_rows(screen)
        assert [row.split(" ", 1)[0] for row in rows] == [
            "Created",
            "Modified",
            "Version",
            "Words",
        ], rows
        # No alignment padding in the compact spelling: one column.
        assert all("  " not in row for row in rows), rows
        # Values arrive whole: the age still ends each timestamp row, which
        # the truncated single line could not do.
        assert all(row.endswith(" ago") for row in rows[:2]), rows
        assert all(
            re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} · \S+ ago$", row)
            for row in rows[:2]
        ), rows


def test_the_property_block_falls_back_to_the_joined_line_without_pairs():
    """A caller that supplies no pairs still renders the sentence it had.

    This is the vacuity guard for the two tests above: if
    ``library_note_property_block`` returned the joined line for every
    input, they would both fail -- and if it ignored ``compact``, this
    would still pass. Each shape is asserted against an input that can
    only produce it.
    """
    pairs = (("Created", "2026-07-01 03:00 · 10w ago"), ("Words", "6 words"))
    assert library_note_property_block((), "Updated today", compact=False) == (
        "Updated today"
    )
    assert library_note_property_block(pairs, "ignored", compact=True) == (
        "Created 2026-07-01 03:00 · 10w ago\nWords 6 words"
    )
    assert library_note_property_block(pairs, "ignored", compact=False) == (
        "Created  2026-07-01 03:00 · 10w ago\nWords    6 words"
    )


# --- task-32642 AC#3: keywords without opening Info ------------------------


@pytest.mark.parametrize("size", (WIDE, COMPACT, NARROW))
async def test_keywords_are_reachable_and_editable_from_the_editor(size):
    """task-32642 AC#3, at all three critique sizes.

    Born red on the unfixed tree, where ``#library-note-keywords`` was
    mounted inside ``#library-note-wide-utilities`` -- a container
    ``apply_session_state`` sets ``display = False`` unconditionally -- so
    forward Tab from the title went straight to the body and the walk never
    reached it: ``assert 'library-note-keywords' in ['library-note-body',
    ...]``.

    Walked with ``pilot.press("tab")``: a focus-order pin that sets focus
    directly cannot see a control that is displayed but skipped.
    """
    host = _build_notes_host(notes=_DATED_NOTE)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        title = screen.query_one("#library-note-title", Input)
        title.focus()
        await pilot.pause()

        visited: list[str] = []
        for _ in range(4):
            await pilot.press("tab")
            await pilot.pause()
            focused = screen.focused
            visited.append("" if focused is None else (focused.id or ""))
            if visited[-1] == "library-note-keywords":
                break
        assert "library-note-keywords" in visited, visited

        # ...and it EDITS: the canonical draft, not a detached widget value.
        before = screen._library_note_session.snapshot.keywords_text
        await pilot.press("g", "a", "r", "d", "e", "n")
        await pilot.pause()
        after = screen._library_note_session.snapshot.keywords_text
        assert after == f"{before}garden", (before, after)
        # Info was never opened to get here.
        assert screen.query_one("#library-note-context-region").display is False


# --- task-32640: the editor header answers "where does this note live?" ----


class _BoundRuntime:
    """The one production seam the header reads, with its real signature.

    ``LibraryNotesController._load_library_note_location`` looks up
    ``app_instance.notes_sync_runtime_owner`` and awaits
    ``note_file_location(note_id)``. The real owner's implementation is
    pinned against a real store in
    ``Tests/Notes/test_notes_sync_note_location.py``; this stands in for it
    so the HEADER can be exercised without a started sync runtime, and it
    carries the same signature so a change to that contract breaks here too.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self.asked: list[str] = []

    async def note_file_location(self, note_id: str) -> str:
        self.asked.append(note_id)
        return self._path


def _location_row(screen) -> Static:
    return screen.query_one("#library-note-location", Static)


async def test_a_database_only_note_says_so_and_claims_no_file():
    """task-32640 AC#1/AC#2, the unbound world.

    Born red on the unfixed tree -- there was no row at all:
    ``NoMatches: No nodes match '#library-note-location'``.
    """
    host = _build_notes_host(notes=_DATED_NOTE)
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await pilot.pause()

        line = str(_location_row(screen).renderable)
        assert line == "In the Library database only — no file on disk", line
        # AC#2: no file, so no path and no write time are invented.
        assert "/" not in line and "written" not in line


async def test_a_synced_note_names_its_file_and_when_it_was_written(tmp_path):
    """task-32640 AC#2/AC#4: the path from the live binding, the time from
    the FILE -- not from the note record, which knows neither.

    Born red on the unfixed tree for the same reason as the test above (no
    such row); after the row existed but before the write time came from
    ``stat``, the line ended at the path.
    """
    note_file = tmp_path / "vault" / "People" / "Sam.md"
    note_file.parent.mkdir(parents=True)
    note_file.write_text("# Sam\n", encoding="utf-8")
    written_at = datetime(2026, 9, 15, 8, 6).timestamp()
    os.utime(note_file, (written_at, written_at))

    host = _build_notes_host(notes=_DATED_NOTE)
    runtime = _BoundRuntime(str(note_file))
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        # Attached after the shell builds: the header asks the app for the
        # runtime when a note opens, which is what makes AC#4's "live state"
        # true -- a runtime that arrives later is still asked.
        host.app_instance.notes_sync_runtime_owner = runtime
        await _open_first_note(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: "In a synced folder" in str(_location_row(screen).renderable),
            message=f"The header read {str(_location_row(screen).renderable)!r}.",
        )

        line = str(_location_row(screen).renderable)
        assert runtime.asked == ["note-dated"], runtime.asked
        assert "Sam.md" in line, line
        # The clock is the file's own mtime, stated in local time.
        assert "file written 2026-09-15 08:06" in line, line


def test_the_location_line_truncates_the_path_and_never_the_world():
    """task-32640 AC#3, as a rule rather than a specimen.

    The world is stated in full at every width; the path is
    middle-elided so its filename survives; the write time is dropped
    before the path is. The vacuity guard is the last assertion: a line
    that simply returned everything unelided would pass the first three.
    """
    path = "/Users/someone/Documents/Vaults/Power vault/People/Samantha Reed.md"
    wide = library_note_location_line(path, "2026-09-15 08:06", 120)
    assert wide.startswith("In a synced folder · ")
    assert wide.endswith("· file written 2026-09-15 08:06")
    assert path in wide

    narrow = library_note_location_line(path, "2026-09-15 08:06", 46)
    assert narrow.startswith("In a synced folder · ")
    assert "Samantha Reed.md" in narrow, narrow
    assert path not in narrow, narrow
    assert "file written" not in narrow, narrow

    # Unmeasured width elides nothing rather than guessing.
    assert library_note_location_line(path, "", 0) == (
        f"In a synced folder · {path}"
    )


# --- task-32641: a vault is recognised BEFORE the review -------------------


def _recognition_controller(tmp_path):
    """One import controller with PRODUCTION discovery and a real ledger.

    Only the seams ``recognise_selected_folder`` actually reaches are real:
    the bounded discovery (which is also what the review's check runs) and
    Import once's own receipt ledger. The rest of the constructor's dozen
    dependencies belong to phases this path never enters.
    """
    from tldw_chatbook.Notes.note_import_discovery import discover_import_sources
    from tldw_chatbook.Notes.note_import_plan_models import ImportBounds
    from tldw_chatbook.Notes.note_import_receipts import NoteImportReceiptRepository
    from tldw_chatbook.UI.Library_Modules.library_note_import_controller import (
        LibraryNoteImportController,
    )

    published: list[object] = []
    unused = lambda *args, **kwargs: None  # noqa: E731 - phases not entered here
    controller = LibraryNoteImportController(
        bounds=ImportBounds(
            max_files=1_000,
            max_file_bytes=16 * 1024 * 1024,
            max_total_bytes=256 * 1024 * 1024,
            max_depth=32,
        ),
        database=unused,
        folder_repository=unused,
        receipt_repository=lambda: NoteImportReceiptRepository(
            tmp_path / "receipts.sqlite3"
        ),
        discover_import_sources=discover_import_sources,
        parse_import_sources=unused,
        classify_import_batch=unused,
        analyze_root_collision=unused,
        resolve_root_collision=unused,
        confirm_uncertain_match=unused,
        apply_item_override=unused,
        approve_note_import_plan=unused,
        executor_factory=unused,
        publish_snapshot=published.append,
        refresh_after_settlement=unused,
    )
    return controller, published


def _vault(tmp_path, *, obsidian: bool = True):
    """A folder shaped like the critique's own power vault."""
    root = tmp_path / "Power vault"
    (root / "People").mkdir(parents=True)
    (root / "People" / "Sam.md").write_text("# Sam\n", encoding="utf-8")
    (root / "Daily.md").write_text("# Daily\n", encoding="utf-8")
    if obsidian:
        (root / ".obsidian").mkdir()
        (root / ".obsidian" / "app.json").write_text("{}", encoding="utf-8")
        (root / ".trash").mkdir()
        (root / ".trash" / "Deleted.md").write_text("gone\n", encoding="utf-8")
        (root / "Templates").mkdir()
        (root / "Templates" / "Daily.md").write_text("tpl\n", encoding="utf-8")
    return root


async def test_a_selected_vault_is_recognised_with_its_counts_and_skips(tmp_path):
    """task-32641 AC#1/AC#2, through production discovery.

    Born red on the unfixed tree: ``recognise_selected_folder`` did not
    exist -- ``AttributeError: 'LibraryNoteImportController' object has no
    attribute 'recognise_selected_folder'`` -- and nothing said "vault"
    anywhere before the review.
    """
    controller, published = _recognition_controller(tmp_path)
    controller.accept_selected_path(_vault(tmp_path), is_folder=True)
    await controller.recognise_selected_folder(already_synced=False)

    line = controller.presentation_snapshot.vault_recognition
    assert line.startswith("Obsidian vault · "), line
    # The two real notes; the vault's own folders are not counted as notes.
    assert "2 notes to read" in line, line
    assert "skips .obsidian/, .trash/, Templates/ and empty files" in line, line
    assert "already" not in line, line
    assert published, "the recognition was never published to the canvas"


async def test_a_plain_folder_says_nothing_extra(tmp_path):
    """task-32641 AC#3: no empty "0 detected" row.

    The vacuity guard for the test above: the two folders differ only by
    the ``.obsidian/`` marker, so a recognition that fired for any folder
    would pass there and fail here.
    """
    controller, _ = _recognition_controller(tmp_path)
    controller.accept_selected_path(_vault(tmp_path, obsidian=False), is_folder=True)
    await controller.recognise_selected_folder(already_synced=False)

    assert controller.presentation_snapshot.vault_recognition == ""


async def test_an_already_synced_vault_says_so_before_the_review(tmp_path):
    """task-32641 AC#4: the duplicate-vault case, where it can still be undone."""
    controller, _ = _recognition_controller(tmp_path)
    controller.accept_selected_path(_vault(tmp_path), is_folder=True)
    await controller.recognise_selected_folder(already_synced=True)

    line = controller.presentation_snapshot.vault_recognition
    assert line.endswith("· this folder is already kept in sync"), line


async def test_choosing_another_folder_drops_the_previous_recognition(tmp_path):
    """task-32641 AC#3: the sentence belongs to the folder that was scanned."""
    controller, _ = _recognition_controller(tmp_path)
    controller.accept_selected_path(_vault(tmp_path), is_folder=True)
    await controller.recognise_selected_folder(already_synced=False)
    assert controller.presentation_snapshot.vault_recognition

    plain = tmp_path / "plain"
    plain.mkdir()
    controller.accept_selected_path(plain, is_folder=True, replace=True)
    assert controller.presentation_snapshot.vault_recognition == ""


def test_the_recognition_line_states_counts_and_the_skip_rule():
    """task-32641 AC#2 as a rule, not a specimen.

    Empty files are named as always skipped because they always are
    (``ImportClassification.EMPTY`` -> ``ImportAction.SKIP`` in
    ``note_import_plan_models``), and no scan is needed to say so; the
    vault-owned folders are named only when the scan found them.
    """
    from tldw_chatbook.Library.library_note_import_state import (
        vault_recognition_line,
    )

    assert vault_recognition_line(
        notes=1, skipped_folders=(), already_imported=0, already_synced=False
    ) == "Obsidian vault · 1 note to read · skips empty files"
    assert vault_recognition_line(
        notes=54,
        skipped_folders=(".obsidian/",),
        already_imported=54,
        already_synced=False,
    ) == (
        "Obsidian vault · 54 notes to read · skips .obsidian/ and empty files"
        " · 54 already imported"
    )


@pytest.mark.parametrize("size", (WIDE, COMPACT, NARROW))
async def test_the_recognition_paints_under_the_folder_confirmation(size):
    """task-32641 AC#1: on the confirmation line, at every critique size.

    Born red on the unfixed tree: ``LibraryNoteImportSnapshot`` had no
    ``vault_recognition`` field and the canvas composed no such row --
    ``TypeError: __init__() got an unexpected keyword argument``, and after
    the field, ``NoMatches: No nodes match '#note-import-vault-recognition'``.
    """
    from dataclasses import replace as dataclass_replace

    from Tests.UI.test_library_notes_wave_import_ux import _ImportHost
    from tldw_chatbook.Library.library_note_import_state import (
        initial_note_import_snapshot,
        project_library_note_import_snapshot,
    )

    base = project_library_note_import_snapshot(initial_note_import_snapshot())
    snapshot = dataclass_replace(
        base,
        selected_names=("/Users/someone/Power vault",),
        selection_kind="folder",
        vault_recognition=(
            "Obsidian vault · 54 notes to read · skips .obsidian/, .trash/, "
            "Templates/ and empty files · 54 already imported"
        ),
    )
    host = _ImportHost(snapshot)
    async with host.run_test(size=size) as pilot:
        await pilot.pause()
        summary = host.query_one("#note-import-source-summary", Static)
        recognition = host.query_one("#note-import-vault-recognition", Static)

        assert "Obsidian vault" in str(recognition.renderable)
        # Under the confirmation, not inside the review below it.
        assert recognition.region.y == summary.region.y + summary.region.height, (
            summary.region,
            recognition.region,
        )

    # AC#3: nothing at all for a folder that is not a vault.
    plain = _ImportHost(dataclass_replace(snapshot, vault_recognition=""))
    async with plain.run_test(size=size) as pilot:
        await pilot.pause()
        assert not plain.query("#note-import-vault-recognition")


def test_a_folder_inside_a_sync_root_is_already_synced(tmp_path):
    """task-32641 AC#4's other half: the probe that answers it.

    Importing a SUB-FOLDER of a synced vault duplicates it exactly as
    thoroughly as importing the vault, so the predicate is "covered by",
    not "equal to" -- which is also the vacuity guard here: an equality
    check passes the first assertion and fails the second.
    """
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

    root = tmp_path / "vault"
    (root / "People").mkdir(parents=True)
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()

    owner = NotesSyncRuntimeOwner.__new__(NotesSyncRuntimeOwner)
    owner._root_paths = {"root-1": str(root)}
    assert owner.folder_is_sync_root(root) is True
    assert owner.folder_is_sync_root(root / "People") is True
    assert owner.folder_is_sync_root(elsewhere) is False


async def test_picking_a_vault_folder_recognises_it_on_the_real_screen(tmp_path):
    """task-32641 AC#1/AC#4 through the screen, not the controller alone.

    ``_accept_library_note_import_path`` is the whole of what the picker's
    callback does once the dialog closes, so this walks the production
    wiring: the import controller's scan AND the sync-root probe the screen
    answers for it. Born red with ``AttributeError:
    '_library_folder_is_sync_root'`` before that seam existed.
    """

    class _Runtime:
        def folder_is_sync_root(self, folder) -> bool:
            return True

    vault = _vault(tmp_path)
    host = _build_notes_host(notes=_DATED_NOTE)
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        host.app_instance.notes_sync_runtime_owner = _Runtime()

        screen._accept_library_note_import_path(vault, replace=True)
        await _wait_for_condition(
            pilot,
            lambda: bool(
                screen._library_note_import_controller.presentation_snapshot.vault_recognition
            ),
            message="The picked folder was never recognised.",
        )

        line = (
            screen._library_note_import_controller.presentation_snapshot.vault_recognition
        )
        assert line.startswith("Obsidian vault · 2 notes to read"), line
        assert line.endswith("· this folder is already kept in sync"), line
