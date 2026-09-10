"""Critique #8: structural waits get a deadline, a Cancel, and never block the exit.

task-32055. The observed wedge: a File Notes folder change sat on
``Changing folder…`` forever while Escape, the back cue, the palette and
Ctrl+Q were all swallowed. These tests drive the three Library waits with a
service call that never resolves and pin what the user can still do.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import Tests.UI._optional_module_stubs  # noqa: F401
from textual.widgets import Button

from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_service import (
    SCAN_PROGRESS_INTERVAL,
    FileNotesService,
    ScanCancelled,
)
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    ROOT_CHANGE_LANDED_COPY,
    LibraryFileNotesWorkspace,
)
import tldw_chatbook.Widgets.Library.library_file_notes_workspace as workspace_module
from Tests.UI.test_library_file_notes_workspace import (
    _production_workspace_context,
    _static_text,
    _wait_until,
)

STRUCTURAL_WAIT_CANCEL = "#library-structural-wait-cancel"


def _busy_row_cancel_labels(workspace) -> list[str]:
    """Every visible control in the folder row that reads as a Cancel."""
    row = workspace.query_one("#file-notes-root-row")
    labels = [
        str(button.label)
        for button in row.query(Button)
        if button.display and "Cancel" in str(button.label)
    ]
    if "Cancel" in _static_text(workspace, "#file-notes-root-status"):
        labels.append("status line")
    return labels


class _BlockedScan:
    """Stand in for a folder scan that never comes back.

    Patched in as a plain function (an instance is not a descriptor, so
    assigning one to the class would never bind ``self``).
    """

    def __init__(self, blocked_root: Path) -> None:
        self.blocked_root = blocked_root.resolve()
        self.started = threading.Event()
        self.release = threading.Event()
        original = FileNotesService.scan
        owner = self

        def scan(service: FileNotesService, **kwargs):
            if Path(service.root).resolve() == owner.blocked_root:
                owner.started.set()
                owner.release.wait(20)
            return original(service, **kwargs)

        self.scan = scan


@pytest.fixture
def blocked_root_change(tmp_path, monkeypatch):
    """A linked old root plus a new root whose scan never finishes."""
    old_root = tmp_path / "linked"
    old_root.mkdir()
    (old_root / "old.md").write_text("old note", encoding="utf-8")
    new_root = tmp_path / "slow"
    new_root.mkdir()
    (new_root / "new.md").write_text("new note", encoding="utf-8")
    blocked = _BlockedScan(new_root)
    monkeypatch.setattr(FileNotesService, "scan", blocked.scan)
    try:
        yield old_root, new_root, blocked
    finally:
        blocked.release.set()


async def _start_blocked_root_change(pilot, workspace, blocked, new_root):
    """Choose the slow folder and wait for its wait to become visible."""
    workspace._root_selected(new_root)
    await _wait_until(
        pilot,
        lambda: blocked.started.is_set() and workspace._structural_wait is not None,
        "the folder change never started its structural wait",
    )
    return workspace._structural_wait


@pytest.mark.asyncio
async def test_folder_change_wait_reports_still_working_and_offers_cancel(
    blocked_root_change,
) -> None:
    """A folder change that outlives the patience window says so, with a way out."""
    old_root, new_root, blocked = blocked_root_change
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
        wait = await _start_blocked_root_change(pilot, workspace, blocked, new_root)
        assert wait.label == "Changing folder"
        assert _static_text(workspace, "#file-notes-root-status") == "Changing folder…"

        # The real patience timer, not a back-dated clock.
        await _wait_until(
            pilot,
            lambda: "still working"
            in _static_text(workspace, "#file-notes-root-status"),
            "the wait never admitted it was still working",
            attempts=400,
        )
        line = _static_text(workspace, "#file-notes-root-status")
        assert line == "Changing folder… · still working"
        cancel = workspace.query_one(STRUCTURAL_WAIT_CANCEL, Button)
        assert cancel.display and not cancel.disabled
        # task-32121 AC4/AC5: the two decisions are buttons beside the one
        # Cancel; the line used to end in "· Cancel" as well, which read as
        # a second control with the same label in the same busy row.
        assert "Cancel" not in line
        assert workspace.query_one(
            "#file-notes-root-keep-waiting", Button
        ).display
        assert workspace.query_one(
            "#file-notes-root-choose-another", Button
        ).display
        assert _busy_row_cancel_labels(workspace) == ["Cancel"]
        assert workspace.root == old_root.resolve()
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_escape_leaves_files_while_a_folder_change_is_still_running(
    blocked_root_change,
) -> None:
    """Escape returns to Database Notes; the previously linked folder survives."""
    old_root, new_root, blocked = blocked_root_change
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
        screen = pilot.app.screen
        await _start_blocked_root_change(pilot, workspace, blocked, new_root)
        assert screen._notes_state.source == "files"

        await pilot.press("escape")
        await _wait_until(
            pilot,
            lambda: screen._notes_state.source == "database",
            "Escape was swallowed while a folder change was running",
        )
        assert workspace.root == old_root.resolve()
        assert workspace._structural_wait is None
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_quit_and_navigation_are_never_vetoed_by_a_structural_wait(
    blocked_root_change,
) -> None:
    """The gate refuses a second write, never the exit."""
    old_root, new_root, blocked = blocked_root_change
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
        screen = pilot.app.screen
        await _start_blocked_root_change(pilot, workspace, blocked, new_root)

        assert screen.check_action("quit", ()) is not False
        assert screen.check_action("library_notes_files_back", ()) is not False

        # The write IS gated: a second folder change never starts.
        assert await workspace.set_root(old_root, persist=False) is False
        assert blocked.started.is_set()

        # And the app-level navigation flush is not vetoed either.
        assert await screen.flush_pending_work() is True
        assert workspace.root == old_root.resolve()
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_cancelling_a_folder_change_keeps_the_previous_folder(
    blocked_root_change,
) -> None:
    """The Cancel button clears the wait and says what survived."""
    old_root, new_root, blocked = blocked_root_change
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
        screen = pilot.app.screen
        await _start_blocked_root_change(pilot, workspace, blocked, new_root)

        workspace.query_one(STRUCTURAL_WAIT_CANCEL, Button).press()
        await _wait_until(
            pilot,
            lambda: workspace._structural_wait is None,
            "Cancel did not clear the structural wait",
        )
        assert (
            _static_text(workspace, "#file-notes-action-status")
            == "Folder change cancelled · previous folder kept"
        )
        assert workspace.root == old_root.resolve()
        assert "old.md" in workspace.entries
        assert not workspace.query_one(STRUCTURAL_WAIT_CANCEL, Button).display
        assert screen._notes_state.source == "files"
        # The canvas is usable again: a second folder change is admitted.
        assert not workspace.query_one("#file-notes-choose-root", Button).disabled
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_skill_import_wait_reports_still_working_and_can_be_cancelled(
    tmp_path, monkeypatch
) -> None:
    """The skill import wait uses the same helper, line and Cancel button id."""
    from tldw_chatbook.UI.Library_Modules.library_skill_import_controller import (
        LibrarySkillImportCoordinator,
    )
    from tldw_chatbook.Library.library_structural_wait import WAIT_OWNER_SKILL_IMPORT
    from Tests.Skills.test_skills_import import (
        _open_skills_import_row,
        _real_skills_scope_service_with_trust,
    )
    from Tests.Skills.test_skills_library_flow import _wire_empty_non_skill_services
    from Tests.UI.test_library_shell import (
        LIBRARY_TEST_SIZE,
        LibraryHarness,
        _active_library_screen,
        _build_test_app,
        _wait_for_library_shell,
    )
    from textual.widgets import Input

    async def never_lands(self, raw_path: str):
        await asyncio.Event().wait()

    monkeypatch.setattr(LibrarySkillImportCoordinator, "_import", never_lands)
    _local, service = _real_skills_scope_service_with_trust(tmp_path / "store")
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Fixture.\n---\n\nBody.\n", encoding="utf-8"
    )
    app = _build_test_app(configured_default="library")
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)
        screen.query_one("#library-skills-import-path", Input).value = str(skill_dir)
        await pilot.pause()
        screen.query_one("#library-skills-import-run", Button).press()
        await _wait_until(
            pilot,
            lambda: screen._library_skills_import_in_flight,
            "the skill import never started",
        )
        assert (
            _static_text(screen, "#library-skills-import-status")
            == "Inspecting/importing…"
        )
        assert screen.check_action("quit", ()) is not False

        wait = screen._library_structural_wait_for(WAIT_OWNER_SKILL_IMPORT)
        assert wait is not None
        wait.started_at -= 5.0
        screen._repaint_library_skills_import_status()
        await pilot.pause()
        assert (
            _static_text(screen, "#library-skills-import-status")
            == "Inspecting/importing… · still working · Cancel"
        )

        # An in-flight refusal must still reach the row: the wait owns the
        # suffix, never the whole line (crit8 review #2).
        screen.handle_library_skills_import_run(SimpleNamespace(stop=lambda: None))
        await pilot.pause()
        assert _static_text(screen, "#library-skills-import-status") == (
            "An import is already in progress. · still working · Cancel"
        )

        cancel = screen.query_one(STRUCTURAL_WAIT_CANCEL, Button)
        assert cancel.display
        cancel.press()
        await _wait_until(
            pilot,
            lambda: not screen._library_skills_import_in_flight,
            "Cancel never released the skill import wait",
        )
        assert screen._library_skills_import_status == (
            "Import cancelled · check the skills list before retrying."
        )
        assert screen._library_structural_wait_for(WAIT_OWNER_SKILL_IMPORT) is None


@pytest.mark.asyncio
async def test_export_wait_reports_still_working_beside_its_cancel(
    tmp_path, monkeypatch
) -> None:
    """The export bundle write uses the same helper and its own Cancel."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.Library.library_structural_wait import WAIT_OWNER_EXPORT
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_EXPORT
    from Tests.UI.test_library_shell import (
        LIBRARY_TEST_SIZE,
        LibraryHarness,
        _active_library_screen,
        _build_test_app,
        _seed_conversations,
        _two_conversations,
        _wait_for_library_shell,
        _wait_for_selector,
        _wire_empty_export_prompts_db,
    )

    app = _build_test_app()
    _wire_empty_export_prompts_db(app, "crit8-waits-export-prompts")
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="crit8-waits-export-media")
    app.media_db.add_media_with_keywords(title="M1", content="c1", media_type="video")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="crit8-waits-export-ccn")
    app.chachanotes_db.add_conversation({"title": "Conv"})
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        # A bundle write that never lands: the real worker dispatch is
        # replaced, leaving the canvas exactly where a hung export leaves it.
        monkeypatch.setattr(
            screen, "_start_library_export_worker", lambda **kwargs: None
        )
        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-destination")
        await _wait_until(
            pilot,
            lambda: screen._export_state.counts is not None,
            "export counts never landed",
        )
        screen.refresh(recompose=True)
        await pilot.pause()
        screen._apply_library_export_destination(tmp_path / "out")
        await pilot.pause()
        screen.query_one("#library-export-submit", Button).press()
        await _wait_until(
            pilot,
            lambda: screen._export_state.running
            and bool(screen.query("#library-export-status-line")),
            "the export never started",
        )
        status = _static_text(screen, "#library-export-status-line")
        assert status.startswith("Exporting (") and status.endswith(")…")
        assert screen.check_action("quit", ()) is not False

        wait = screen._library_structural_wait_for(WAIT_OWNER_EXPORT)
        assert wait is not None
        wait.started_at -= 5.0
        screen._refresh_library_export_status_line()
        await pilot.pause()
        assert _static_text(screen, "#library-export-status-line").endswith(
            "… · still working · Cancel"
        )
        assert screen.query_one("#library-export-cancel", Button).display

        # Per-phase progress keeps the line; the wait only appends to it
        # (crit8 review #1).
        screen._apply_library_export_progress(
            screen._export_state.run_id, "notes", 3, 12
        )
        await pilot.pause()
        assert _static_text(screen, "#library-export-status-line") == (
            "Collecting notes…  3/12 · still working · Cancel"
        )

        screen.query_one("#library-export-cancel", Button).press()
        await pilot.pause()
        assert screen._library_structural_wait_for(WAIT_OWNER_EXPORT) is None
        assert _static_text(screen, "#library-export-status-line") == "Cancelling…"


@pytest.mark.asyncio
async def test_skill_import_patience_repaint_stays_off_other_rows(
    tmp_path, monkeypatch
) -> None:
    """The 3 s repaint must not touch whatever row the user left for.

    crit8 review #3: leaving Skills mid-import used to let the patience
    timer flip the File Notes workspace's identically-id'd Cancel button on,
    or recompose that unrelated canvas whole.
    """
    from tldw_chatbook.Library.library_structural_wait import WAIT_OWNER_SKILL_IMPORT
    import tldw_chatbook.UI.Screens.library_screen as library_screen_module

    root = tmp_path / "linked"
    root.mkdir()
    (root / "note.md").write_text("note", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
        screen = pilot.app.screen
        # An import started on the Skills row; the user is now on Folder files.
        assert screen._library_skill_import_coordinator.claim(str(tmp_path / "alpha"))
        wait = screen._begin_library_structural_wait(
            "Inspecting/importing",
            WAIT_OWNER_SKILL_IMPORT,
            cancel=lambda: None,
        )
        wait.started_at -= 5.0
        synced: list[str] = []
        monkeypatch.setattr(
            library_screen_module,
            "_sync_library_canvas",
            lambda _screen, kind: synced.append(kind),
        )

        screen._repaint_library_skills_import_status()
        await pilot.pause()

        assert "skills" not in synced
        assert not workspace.query_one(STRUCTURAL_WAIT_CANCEL, Button).display
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_a_folder_change_never_steals_another_surfaces_wait(
    blocked_root_change,
) -> None:
    """A folder change must leave the Skills row's own wait alone.

    crit8 review #4: the File Notes workspace used to publish into the
    screen's single wait slot, so a folder change overwrote a running skill
    import's wait and its Cancel silently stopped working.
    """
    from tldw_chatbook.Library.library_structural_wait import WAIT_OWNER_SKILL_IMPORT

    old_root, new_root, blocked = blocked_root_change
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
        screen = pilot.app.screen
        cancelled: list[str] = []
        skill_wait = screen._begin_library_structural_wait(
            "Inspecting/importing",
            WAIT_OWNER_SKILL_IMPORT,
            cancel=lambda: cancelled.append("skill-import"),
        )

        await _start_blocked_root_change(pilot, workspace, blocked, new_root)
        assert screen._library_structural_wait_for(WAIT_OWNER_SKILL_IMPORT) is skill_wait

        workspace.query_one(STRUCTURAL_WAIT_CANCEL, Button).press()
        await _wait_until(
            pilot,
            lambda: workspace._structural_wait is None,
            "Cancel did not clear the folder change",
        )
        assert screen._library_structural_wait_for(WAIT_OWNER_SKILL_IMPORT) is skill_wait

        # And the Skills row's Cancel still reaches the import it belongs to.
        screen._library_structural_wait_cancel_pressed(
            SimpleNamespace(stop=lambda: None)
        )
        assert cancelled == ["skill-import"]
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_folder_change_that_never_lands_times_out_and_keeps_the_folder(
    blocked_root_change, monkeypatch
) -> None:
    """The wait has a deadline; hitting it reports why and changes nothing."""
    old_root, new_root, blocked = blocked_root_change
    monkeypatch.setattr(workspace_module, "ROOT_CHANGE_TIMEOUT_SECONDS", 0.4)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
        await _start_blocked_root_change(pilot, workspace, blocked, new_root)
        await _wait_until(
            pilot,
            lambda: workspace._structural_wait is None,
            "the folder change never hit its deadline",
        )
        status = _static_text(workspace, "#file-notes-action-status")
        assert status == (
            "Folder change timed out · previous folder kept. "
            "Try again or choose a different folder."
        )
        assert workspace.root == old_root.resolve()
        assert not workspace._root_transitioning
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_cancel_that_loses_to_the_commit_reports_the_folder_that_landed(
    tmp_path, monkeypatch
) -> None:
    """A commit crossing the cancel must not claim the previous folder was kept.

    Root persistence past its atomic file replacement is deliberately
    unstoppable -- refusing to publish there would leave the on-disk config
    pointing at a folder the UI never adopted. So the cancel receipt, not
    the commit, is what has to stay honest (PR #2524 review).
    """
    old_root = tmp_path / "linked"
    old_root.mkdir()
    (old_root / "old.md").write_text("old note", encoding="utf-8")
    new_root = tmp_path / "next"
    new_root.mkdir()
    (new_root / "new.md").write_text("new note", encoding="utf-8")

    persistence_started = threading.Event()
    release_persistence = threading.Event()

    def persist_mutation(section_values):
        persistence_started.set()
        assert release_persistence.wait(timeout=20)
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        persist_mutation,
    )
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    try:
        async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: workspace.initialized,
                "the workspace never scanned its linked folder",
            )
            workspace._root_selected(new_root)
            await _wait_until(
                pilot,
                persistence_started.is_set,
                "the folder change never reached config persistence",
            )
            assert workspace.cancel_structural_wait()
            release_persistence.set()
            await _wait_until(
                pilot,
                lambda: workspace.root == new_root.resolve(),
                "the shielded commit never landed",
            )
            await _wait_until(
                pilot,
                lambda: workspace._structural_wait is None,
                "the cancelled wait never settled",
            )
            assert (
                _static_text(workspace, "#file-notes-action-status")
                == ROOT_CHANGE_LANDED_COPY
            )
            # Review round 2 (Qodo finding 4): the folder row is the line
            # the user is watching, and it survives repaints -- saying the
            # previous folder was kept while the new one is linked is the
            # one lie this branch exists to prevent.
            assert (
                _static_text(workspace, "#file-notes-root-status")
                == ROOT_CHANGE_LANDED_COPY
            )
            workspace._update_root_surface()
            await pilot.pause()
            assert (
                _static_text(workspace, "#file-notes-root-status")
                == ROOT_CHANGE_LANDED_COPY
            )
    finally:
        release_persistence.set()
        await workspace.shutdown()
        replica.close()


class _LockHoldingScan:
    """A scan that holds the shared service lock exactly like the real one.

    task-32121: ``FileNotesService.scan`` is ``@_serialized``, so an
    abandoned scan keeps the workspace's ``_service_lock`` until it
    returns. The fake used by the older waits tests replaces the decorated
    method outright and therefore never held that lock -- which is why the
    poisoning this class reproduces went unnoticed.
    """

    def __init__(self, blocked_root: Path, *, cooperative: bool = True) -> None:
        self.blocked_root = blocked_root.resolve()
        self.cooperative = cooperative
        self.started = threading.Event()
        self.release = threading.Event()
        self.observed_cancel = threading.Event()
        original = FileNotesService.scan
        owner = self

        def scan(service: FileNotesService, **kwargs):
            if Path(service.root).resolve() != owner.blocked_root:
                return original(service, **kwargs)
            should_cancel = kwargs.get("should_cancel")
            with service._operation_lock:
                owner.started.set()
                deadline = time.monotonic() + 20.0
                while time.monotonic() < deadline:
                    if owner.release.is_set():
                        break
                    if (
                        owner.cooperative
                        and should_cancel is not None
                        and should_cancel()
                    ):
                        owner.observed_cancel.set()
                        raise ScanCancelled()
                    time.sleep(0.01)
                return original(service, **kwargs)

        self.scan = scan


@pytest.fixture
def lock_holding_root_change(tmp_path, monkeypatch):
    """A linked root, a folder whose scan wedges the lock, and a small folder."""
    old_root = tmp_path / "linked"
    old_root.mkdir()
    (old_root / "old.md").write_text("old note", encoding="utf-8")
    wedged_root = tmp_path / "home-sized"
    wedged_root.mkdir()
    (wedged_root / "wedged.md").write_text("wedged", encoding="utf-8")
    small_root = tmp_path / "small"
    small_root.mkdir()
    (small_root / "small.md").write_text("small", encoding="utf-8")

    def install(*, cooperative: bool = True) -> _LockHoldingScan:
        blocked = _LockHoldingScan(wedged_root, cooperative=cooperative)
        monkeypatch.setattr(FileNotesService, "scan", blocked.scan)
        return blocked

    installed: list[_LockHoldingScan] = []

    def factory(**kwargs) -> _LockHoldingScan:
        blocked = install(**kwargs)
        installed.append(blocked)
        return blocked

    try:
        yield old_root, wedged_root, small_root, factory
    finally:
        for blocked in installed:
            blocked.release.set()


@pytest.mark.asyncio
async def test_abandoned_folder_scan_does_not_block_the_next_folder_change(
    lock_holding_root_change, monkeypatch
) -> None:
    """task-32121 AC1/AC2/AC6: the wedge, reproduced and then released.

    Proven live three times on dev c4a7b1911f: a small folder links in
    under two seconds, the home directory times out at thirty, and every
    later pick times out too. The abandoned scan thread keeps the shared
    service lock, so the next ``set_root`` waits behind it.
    """
    old_root, wedged_root, small_root, install_blocked_scan = (
        lock_holding_root_change
    )
    blocked = install_blocked_scan()
    monkeypatch.setattr(workspace_module, "ROOT_CHANGE_TIMEOUT_SECONDS", 0.4)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    try:
        async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
            workspace._root_selected(wedged_root)
            await _wait_until(
                pilot,
                lambda: blocked.started.is_set(),
                "the wedging scan never started",
            )
            await _wait_until(
                pilot,
                lambda: workspace._structural_wait is None,
                "the folder change never hit its deadline",
            )
            assert workspace.root == old_root.resolve()

            workspace._root_selected(small_root)
            await _wait_until(
                pilot,
                lambda: workspace.root == small_root.resolve(),
                "the next folder change stayed wedged behind the abandoned scan",
                attempts=100,
            )
            assert blocked.observed_cancel.is_set()
            assert (
                _static_text(workspace, "#file-notes-root-status")
                == "Linked · Local folder: small"
            )
    finally:
        blocked.release.set()
        await workspace.shutdown()
        replica.close()


@pytest.mark.asyncio
async def test_timeout_reason_owns_the_folder_row_until_the_next_attempt(
    blocked_root_change, monkeypatch
) -> None:
    """task-32121 AC3: the reason is painted where the user is looking.

    It used to be written only into the action-status line at the bottom
    of the editor pane while the folder row -- the line that had been
    saying "Changing folder…" -- silently reverted to the folder that was
    kept. Live sampling at 0.5 s never caught the reason at all.
    """
    old_root, new_root, blocked = blocked_root_change
    monkeypatch.setattr(workspace_module, "ROOT_CHANGE_TIMEOUT_SECONDS", 0.4)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    async with _production_workspace_context(workspace, size=(235, 52)) as pilot:
        await _start_blocked_root_change(pilot, workspace, blocked, new_root)
        await _wait_until(
            pilot,
            lambda: workspace._structural_wait is None,
            "the folder change never hit its deadline",
        )
        timeout_copy = (
            "Folder change timed out · previous folder kept. "
            "Try again or choose a different folder."
        )
        assert _static_text(workspace, "#file-notes-action-status") == timeout_copy
        assert _static_text(workspace, "#file-notes-root-status") == timeout_copy

        # The repaints that run after the wait settles must not scrub it.
        workspace._update_root_surface()
        workspace._update_controls()
        await pilot.pause()
        assert _static_text(workspace, "#file-notes-action-status") == timeout_copy
        assert _static_text(workspace, "#file-notes-root-status") == timeout_copy

        # Review round 1: it must not outlive the user's next real action
        # either -- one timeout used to hide the linked folder for the rest
        # of the session.
        assert await workspace.open_path("old.md")
        assert _static_text(workspace, "#file-notes-root-status") == (
            "Linked · Local folder: linked"
        )

        # ...and so does the next attempt, once the abandoned scan has let
        # go of the service lock. (While it still holds it, the next change
        # reports its own bounded wait -- also the timeout copy -- rather
        # than sitting silent, which is the AC1 half of this fix.)
        workspace._report_root_change_reason(timeout_copy)
        blocked.release.set()
        for _ in range(20):
            await pilot.pause(0.02)
        await workspace.set_root(old_root, persist=False)
        assert _static_text(workspace, "#file-notes-root-status") == (
            "Linked · Local folder: linked"
        )
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_a_slow_scan_reports_the_entries_it_has_seen(
    blocked_root_change,
) -> None:
    """task-32121 AC4: a bare "still working" becomes a count and two choices."""
    old_root, new_root, blocked = blocked_root_change
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    async with _production_workspace_context(workspace, size=(235, 52)) as pilot:
        wait = await _start_blocked_root_change(pilot, workspace, blocked, new_root)
        wait.started_at -= 5.0
        workspace._record_root_scan_progress(
            1240, generation=workspace._root_generation
        )
        workspace._update_root_surface()
        await pilot.pause()

        assert _static_text(workspace, "#file-notes-root-status") == (
            "Changing folder… · 1,240 entries so far"
        )
        assert _busy_row_cancel_labels(workspace) == ["Cancel"]
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_keep_waiting_grants_exactly_one_more_deadline(
    blocked_root_change, monkeypatch
) -> None:
    """task-32121 AC4: Keep waiting resets the deadline, once."""
    old_root, new_root, blocked = blocked_root_change
    monkeypatch.setattr(workspace_module, "ROOT_CHANGE_TIMEOUT_SECONDS", 2.0)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    async with _production_workspace_context(workspace, size=(235, 52)) as pilot:
        wait = await _start_blocked_root_change(pilot, workspace, blocked, new_root)
        wait.started_at -= 5.0
        workspace._update_root_surface()
        await pilot.pause()
        workspace.query_one("#file-notes-root-keep-waiting", Button).press()
        await pilot.pause()

        # Spent: the control goes away and a second grant is refused.
        assert not workspace.query_one(
            "#file-notes-root-keep-waiting", Button
        ).display
        assert workspace._root_change_extension == 2.0
        assert workspace.extend_root_change_deadline() is False

        # The first deadline arrives and spends the grant instead of the
        # folder change: the wait is still running on a second budget.
        await _wait_until(
            pilot,
            lambda: workspace._root_change_extension is None,
            "the first deadline never consumed the granted extension",
            attempts=300,
        )
        assert workspace._structural_wait is not None

        # And the second deadline still ends it.
        await _wait_until(
            pilot,
            lambda: workspace._structural_wait is None,
            "the extended folder change never hit its second deadline",
            attempts=300,
        )
        assert workspace.root == old_root.resolve()
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_choose_another_abandons_the_change_and_reopens_the_picker(
    blocked_root_change, monkeypatch
) -> None:
    """task-32121 AC4: the second choice is a real way to a different folder.

    The pushed screen is recorded rather than mounted: rendering the
    third-party picker pulls in a widget this task does not own, and doing
    so mid-file has produced a component-class KeyError under load.
    """
    from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory

    old_root, new_root, blocked = blocked_root_change
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    async with _production_workspace_context(workspace, size=(235, 52)) as pilot:
        wait = await _start_blocked_root_change(pilot, workspace, blocked, new_root)
        pushed: list[object] = []

        async def _record_push(screen, *args, **kwargs):
            pushed.append(screen)

        monkeypatch.setattr(pilot.app, "push_screen", _record_push)
        wait.started_at -= 5.0
        workspace._update_root_surface()
        await pilot.pause()
        workspace.query_one("#file-notes-root-choose-another", Button).press()

        await _wait_until(
            pilot,
            lambda: bool(pushed),
            "Choose another did not reopen the folder picker",
        )
        assert isinstance(pushed[0], SelectDirectory)
        assert workspace._structural_wait is None
        assert workspace.root == old_root.resolve()
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_a_scan_that_ignores_its_cancel_still_bounds_the_next_change(
    lock_holding_root_change, monkeypatch
) -> None:
    """task-32121 AC1: a wedged thread makes the next change report, not hang.

    Cooperative cancel cannot reach a scan parked inside one
    uninterruptible syscall (a dead network mount). The next change then
    waits on the service lock -- bounded by the same deadline, and it says
    so instead of sitting silent.
    """
    old_root, wedged_root, small_root, install_blocked_scan = (
        lock_holding_root_change
    )
    blocked = install_blocked_scan(cooperative=False)
    monkeypatch.setattr(workspace_module, "ROOT_CHANGE_TIMEOUT_SECONDS", 0.4)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    try:
        async with _production_workspace_context(workspace, size=(235, 52)) as pilot:
            workspace._root_selected(wedged_root)
            await _wait_until(
                pilot,
                lambda: blocked.started.is_set(),
                "the wedging scan never started",
            )
            await _wait_until(
                pilot,
                lambda: workspace._structural_wait is None,
                "the folder change never hit its deadline",
            )

            workspace._root_selected(small_root)
            await _wait_until(
                pilot,
                lambda: _static_text(workspace, "#file-notes-root-status")
                == (
                    "Folder change timed out · previous folder kept. "
                    "Try again or choose a different folder."
                ),
                "the queued folder change never reported its own deadline",
                attempts=200,
            )
            assert workspace.root == old_root.resolve()
    finally:
        blocked.release.set()
        await workspace.shutdown()
        replica.close()


@pytest.mark.asyncio
async def test_closing_folder_files_stops_the_progress_timer(
    blocked_root_change,
) -> None:
    """Review round 2 (Qodo finding 7): teardown stops the patience timer.

    The busy row's repaint is a repeating interval now, and only the wait
    settling stops it -- which a scan parked in an uninterruptible syscall
    never does. Closing Folder files must not leave it ticking against a
    workspace nobody is looking at.
    """
    old_root, new_root, blocked = blocked_root_change
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    try:
        async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
            await _start_blocked_root_change(pilot, workspace, blocked, new_root)
            assert workspace._structural_wait_timer is not None
            await workspace.shutdown()
            assert workspace._structural_wait_timer is None
    finally:
        blocked.release.set()
        replica.close()


@pytest.mark.asyncio
async def test_keep_waiting_also_extends_the_wait_for_the_shared_lock(
    lock_holding_root_change, monkeypatch
) -> None:
    """Review round 2 (Qodo finding 6): one clock for one wait.

    The worker thread had a second, fixed deadline of its own on the
    service lock. Keep waiting moved only the coroutine's deadline, so a
    change queued behind a scan parked in an uninterruptible syscall --
    the one case the extra budget exists for -- still died a moment after
    the user asked for more time.
    """
    old_root, wedged_root, small_root, install_blocked_scan = (
        lock_holding_root_change
    )
    blocked = install_blocked_scan(cooperative=False)
    monkeypatch.setattr(workspace_module, "ROOT_CHANGE_TIMEOUT_SECONDS", 1.5)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    try:
        async with _production_workspace_context(workspace, size=(235, 52)) as pilot:
            workspace._root_selected(wedged_root)
            await _wait_until(
                pilot,
                lambda: blocked.started.is_set(),
                "the wedging scan never started",
            )
            await _wait_until(
                pilot,
                lambda: workspace._structural_wait is None,
                "the first folder change never hit its deadline",
                attempts=300,
            )

            # The wedged thread still owns the lock, so this change queues
            # behind it -- and the user chooses to keep waiting.
            workspace._root_selected(small_root)
            await _wait_until(
                pilot,
                lambda: workspace._structural_wait is not None,
                "the queued folder change never published its wait",
            )
            assert workspace.extend_root_change_deadline()
            await _wait_until(
                pilot,
                lambda: workspace._root_change_extension is None,
                "the first deadline never consumed the granted extension",
                attempts=300,
            )
            # Past the original deadline (where the worker's own timeout
            # used to fire) and still well inside the granted one.
            for _ in range(15):
                await pilot.pause(0.02)

            blocked.release.set()
            await _wait_until(
                pilot,
                lambda: workspace.root == small_root.resolve(),
                "the extended folder change gave up on the lock anyway",
                attempts=300,
            )
    finally:
        blocked.release.set()
        await workspace.shutdown()
        replica.close()


@pytest.mark.asyncio
async def test_abandoning_one_attempt_never_cancels_another_attempts_scan(
    tmp_path, monkeypatch
) -> None:
    """Review round 1: the cancel flag belongs to the attempt, not the widget.

    It used to live on the workspace, so ``_abandon_root_change_task`` set
    whichever flag was current when it ran. A superseded attempt being
    abandoned after a later one had installed its flag would then cancel
    the LATER one's scan, and ``set_root`` returned False with nothing
    written to the surface.
    """
    old_root = tmp_path / "linked"
    old_root.mkdir()
    (old_root / "old.md").write_text("old note", encoding="utf-8")
    next_root = tmp_path / "next"
    next_root.mkdir()
    (next_root / "new.md").write_text("new note", encoding="utf-8")

    scan_started = threading.Event()
    release_scan = threading.Event()
    scan_returned = threading.Event()
    original = FileNotesService.scan

    def scan(service: FileNotesService, **kwargs):
        if Path(service.root).resolve() == next_root.resolve():
            scan_started.set()
            assert release_scan.wait(20)
            result = original(service, **kwargs)
            scan_returned.set()
            return result
        return original(service, **kwargs)

    monkeypatch.setattr(FileNotesService, "scan", scan)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    try:
        async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
            # The later attempt is in flight with its own flag...
            later_event = threading.Event()
            later = asyncio.ensure_future(
                workspace.set_root(next_root, persist=False, cancel_event=later_event)
            )
            await _wait_until(
                pilot,
                scan_started.is_set,
                "the later attempt never reached its scan",
            )

            # ...when a superseded earlier attempt is abandoned.
            earlier_event = threading.Event()
            earlier = asyncio.ensure_future(asyncio.sleep(30))
            workspace._abandon_root_change_task(earlier, earlier_event)

            assert earlier_event.is_set()
            assert not later_event.is_set()
            release_scan.set()
            await later
            assert scan_returned.is_set(), (
                "the later attempt's scan was cancelled by the earlier "
                "attempt's abandonment"
            )
    finally:
        release_scan.set()
        await workspace.shutdown()
        replica.close()


@pytest.mark.asyncio
async def test_the_folder_change_reports_progress_from_the_real_scan(
    tmp_path,
) -> None:
    """Review round 1: pin the workspace-to-service progress join itself."""
    old_root = tmp_path / "linked"
    old_root.mkdir()
    (old_root / "old.md").write_text("old note", encoding="utf-8")
    big_root = tmp_path / "big"
    big_root.mkdir()
    for index in range(SCAN_PROGRESS_INTERVAL + 50):
        (big_root / f"note{index:04d}.md").write_text("body", encoding="utf-8")

    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=old_root, replica=replica)
    async with _production_workspace_context(workspace, size=(235, 52)) as pilot:
        assert await workspace.set_root(big_root, persist=False)
        await pilot.pause()

        assert workspace._root_scan_entries == SCAN_PROGRESS_INTERVAL + 50
    await workspace.shutdown()
    replica.close()
