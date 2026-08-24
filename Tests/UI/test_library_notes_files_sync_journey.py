"""Production-shaped verification for the reviewed Notes/Files/Sync journey."""

from __future__ import annotations

import asyncio
import ast
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.app_factory import _build_test_app
from Tests.Notes.test_notes_sync_runtime import (
    _Adapter as _RuntimeAdapter,
    _input as _runtime_input,
    _owner as _runtime_owner,
    _pending_operation,
    _store as _runtime_store,
)
from Tests.UI.test_library_file_notes_workspace import (
    _assert_legible_painted_text,
    _production_workspace_context,
    _wait_until,
)
from Tests.UI.test_library_file_notes_git import (
    _row,
    _wait_for_current_git_row_projection,
)
from Tests.UI.test_library_file_notes_git_push import (
    _push_destination_projection,
    _push_workspace_fixture,
)
from Tests.Notes.test_file_notes_git_push_service import _publish_candidate_on_owner
from Tests.UI.test_library_note_import_flow import _open_import_once
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Actor_Packs.persona_coordinator import (
    PersonaActorPackCoordinator,
    PersonaActorPackRecoveryResult,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.note_import_execution_models import (
    ImportExecutionReceipt,
    ImportSessionState,
)
from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesSyncBindingRecord,
    NotesSyncRootRecord,
    NotesSyncStoreSetting,
)
from tldw_chatbook.Notes.notes_sync_conflicts import (
    ConflictApplyResult,
    ConflictComparison,
    NotesSyncConflictChoice,
)
from tldw_chatbook.Notes.notes_sync_executor import (
    NotesSyncExecutionResult,
    NotesSyncExecutor,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncOperationState,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_git_commit import CommitOutcome
from tldw_chatbook.Notes.file_notes_git_push import (
    PushReviewHandle,
    PushReviewProjection,
    _push_destination_policy_result,
    push_outcome_copy,
)
from tldw_chatbook.Notes.file_notes_git_service import (
    PushExecutionResult,
    PushPreflightResult,
)
from tldw_chatbook.Notes.file_notes_session_owner import SessionGitStatus
from tldw_chatbook.Notes.notes_sync_models import NotesSyncAction, NotesSyncActionKind
from tldw_chatbook.Notes.notes_sync_reconciler import (
    ReconciliationAttention,
    ReconciliationAttentionKind,
    ReconciliationPlan,
)
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncRuntimeOwner,
    NotesSyncControlResult,
    NotesSyncRootRuntimeSnapshot,
    NotesSyncRuntimeSnapshot,
    RuntimeConflictLabel,
    RuntimeConflictReceipt,
    build_notes_sync_runtime_owner,
)
from tldw_chatbook.Notes.notes_sync_filesystem import (
    NotesSyncFilesystemError,
    PosixNotesSyncFilesystem,
)
from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
    LibraryNotesSyncController,
)
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)
from tldw_chatbook.Widgets.Library import (
    library_file_notes_git_panel as git_panel_module,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Screens import library_screen as library_screen_module


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIVE_HELPER = _REPO_ROOT / "Helper_Scripts" / "verify_notes_files_sync_tui.py"


class _JourneyHarness(LibraryHarness):
    """Mount the real Library hierarchy with the exact production CSS stack."""

    CSS_PATH = TldwCli.CSS_PATH


@pytest.fixture(autouse=True)
def _isolate_actor_pack_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unrelated startup recovery off the Notes journey test database."""

    monkeypatch.setattr(
        "tldw_chatbook.app.first_profile_created_this_session", lambda: False
    )
    monkeypatch.setattr(
        PersonaActorPackCoordinator,
        "recover",
        lambda _self: PersonaActorPackRecoveryResult(0, 0, 0, ()),
    )


def _runtime(*roots: NotesSyncRootRuntimeSnapshot):
    return SimpleNamespace(
        snapshot=lambda: NotesSyncRuntimeSnapshot("active", "sync_now", roots)
    )


def _assert_inside(outer, inner) -> None:
    assert outer.region.contains_region(inner.region), (
        f"{inner.id} {inner.region} escaped {outer.id} {outer.region}"
    )


def _painted_text(app) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


def _seed_real_conflict_authority(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create one real two-sided conflict in disposable Notes authorities."""

    notes_path = tmp_path / "notes.sqlite3"
    state_path = tmp_path / "sync.sqlite3"
    sync_root = tmp_path / "folder"
    sync_root.mkdir()
    target = sync_root / "note.md"
    target.write_text("baseline", encoding="utf-8")
    database = CharactersRAGDB(notes_path, client_id="task-97-seed")
    folders = LocalNoteFolderRepository(database)
    assert database.add_note("Joined conflict", "baseline", "note-1") == "note-1"
    folders.create_folder(name="Synced notes", parent_id=None, folder_id="folder-1")
    folders.reconcile_managed(owner_id="root-1", desired=(("folder-1", "note-1"),))
    with PosixNotesSyncFilesystem(sync_root) as filesystem:
        baseline_file = filesystem.observe("note.md")
    baseline_note = database.get_note_by_id("note-1")
    assert baseline_note is not None
    store = NotesDeviceStateStore(state_path)
    store.initialize()
    store.create_root(
        NotesSyncRootRecord(
            root_id="root-1",
            note_scope_id="local_note",
            logical_folder_id="folder-1",
            canonical_path=str(sync_root.resolve()),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.ACTIVE,
        )
    )
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-1",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest=NotesSyncExecutor.stable_identity_digest(
                baseline_file
            ),
            state=NotesSyncBindingState.ACTIVE,
            serialization=baseline_file.observation.serialization,
            content_digest=hashlib.sha256(b"baseline").hexdigest(),
            note_version=int(baseline_note["version"]),
        )
    )
    assert database.update_note(
        "note-1",
        {"title": "Joined conflict", "content": "note side"},
        int(baseline_note["version"]),
    )
    target.write_text("file side", encoding="utf-8")
    database.close_connection()
    return notes_path, state_path, sync_root


async def _start_real_conflict_stack(
    notes_path: Path, state_path: Path
) -> tuple[
    NotesSyncRuntimeOwner,
    CharactersRAGDB,
    NotesInteropService,
    LibraryNotesSyncController,
]:
    """Build fresh production runtime/executor/controller objects over disk state."""

    database = CharactersRAGDB(notes_path, client_id="task-97-runtime")
    interop = NotesInteropService(
        base_db_directory=notes_path.parent,
        api_client_id="task-97-runtime",
        global_db_to_use=database,
    )
    scope_service = NotesScopeService(
        local_notes_service=interop,
        server_service=None,
        folder_repository=LocalNoteFolderRepository(database),
    )
    owner = build_notes_sync_runtime_owner(
        notes_scope_service=scope_service,
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=state_path,
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )
    await owner.start()
    controller = LibraryNotesSyncController(
        runtime=owner,
        import_controller=SimpleNamespace(begin_selection=lambda: None),
    )
    return owner, database, interop, controller


async def _close_real_conflict_stack(
    owner: NotesSyncRuntimeOwner,
    database: CharactersRAGDB,
    interop: NotesInteropService,
) -> None:
    await owner.shutdown()
    interop.close_all_user_connections()
    database.close_connection()


def test_live_verifier_is_a_checked_in_isolated_entry_point(tmp_path: Path) -> None:
    """The helper proves scratch paths/checksums/teardown before live launch."""
    assert _LIVE_HELPER.is_file()
    evidence = tmp_path / "evidence"
    result = subprocess.run(
        (
            sys.executable,
            str(_LIVE_HELPER),
            "--dry-run",
            "--evidence-dir",
            str(evidence),
        ),
        cwd=_REPO_ROOT,
        check=False,
        env={
            **os.environ,
            "OPENAI_API_KEY": "must-not-reach-child",
            "HTTPS_PROXY": "https://secret-proxy.invalid",
            "SSH_AUTH_SOCK": "/private/secret-agent.sock",
            "GIT_CONFIG_GLOBAL": "/private/secret-gitconfig",
        },
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS scratch profile teardown" in result.stdout
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["profile_is_scratch"] is True
    assert manifest["scratch_paths_valid"] is True
    assert manifest["test_mode"] is True
    assert manifest["first_run_section"] == "first_run"
    assert manifest["first_run_completed"] is True
    assert manifest["model_downloads_offline"] is True
    assert manifest["caller_environment_inherited"] is False
    assert manifest["proxy_environment_scrubbed"] is True
    assert manifest["git_config_isolated"] is True
    assert manifest["repo_byte_stable"] is True
    assert (
        manifest["repo_sentinel_sha256_before"]
        == (manifest["repo_sentinel_sha256_after"])
    )
    assert set(manifest["repo_sentinel_sha256_before"]) == {
        "tldw_chatbook/css/screen_css_scoped.tcss",
        "tldw_chatbook/css/screen_css_self.tcss",
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/css/widget_defaults_scoped.tcss",
        "tldw_chatbook/css/widget_defaults_self.tcss",
    }
    assert "network_disabled" not in manifest
    child_keys = set(manifest["child_environment_keys"])
    assert not child_keys & {
        "OPENAI_API_KEY",
        "SSH_AUTH_SOCK",
    }
    assert (
        manifest["decoy_default_sha256_before"]
        == manifest["decoy_default_sha256_after"]
    )
    assert Path(manifest["effective_config_relative"]).is_absolute() is False
    assert Path(manifest["data_dir_relative"]).is_absolute() is False
    assert manifest["automated_companion"] == (
        "Tests/UI/test_library_notes_files_sync_journey.py"
    )
    assert manifest["planned_physical_journeys"] == [
        "library_shell",
        "database_notes_new",
        "database_notes_list",
    ]


def test_live_verifier_bounds_every_subprocess_call() -> None:
    source = _LIVE_HELPER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]

    assert calls
    assert all(
        any(keyword.arg == "timeout" for keyword in call.keywords) for call in calls
    )


def test_session_git_journey_does_not_delegate_to_other_pytest_tests() -> None:
    """TASK-19012 must drive its own physical journey, not call other tests."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    delegated = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name.startswith("test_")
    ]

    assert delegated == []


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((120, 36), (60, 20)))
async def test_database_notes_import_once_journey_is_painted_focused_and_retained(
    size: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the one production chooser and verify its compact physical path."""
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.notes_sync_runtime_owner = _runtime()
    host = _JourneyHarness(app)

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        notes_canvas = screen.query_one("#library-notes-canvas")
        authority = screen.query_one("#library-notes-authority", Static)
        painted = _painted_text(host)
        purpose = str(
            screen.query_one("#library-notes-database-purpose", Static).renderable
        )
        assert "Library notes · Library database" in str(authority.renderable)
        assert "use Sync to mirror" not in purpose
        assert "switch to Folder files" in purpose
        assert "choose Add from files" in purpose
        assert "Library notes" in painted
        assert "Add from files" in painted

        begin_selection = Mock(
            wraps=screen._library_note_import_controller.begin_selection
        )
        monkeypatch.setattr(
            screen._library_note_import_controller,
            "begin_selection",
            begin_selection,
        )
        dialogs: list[object] = []
        screen.app.push_screen = lambda dialog, callback=None: dialogs.append(dialog)
        add_from_files = screen.query_one("#library-notes-add-from-files", Button)
        add_from_files.focus()
        await pilot.pause()
        add_from_files.press()
        import_once = await _wait_for_selector(screen, pilot, "#notes-add-import-once")
        chooser = screen.query_one("#library-notes-canvas")
        await pilot.pause()

        assert chooser is notes_canvas
        assert screen.focused is import_once
        painted = _painted_text(host)
        assert "Import once" in painted
        assert "Keep a folder synced" in painted
        _assert_inside(chooser, import_once)
        _assert_inside(chooser, screen.query_one("#notes-sync-back", Button))

        import_once.press()
        await pilot.pause()
        assert begin_selection.call_count == 1
        assert len(dialogs) == 1


@pytest.mark.asyncio
async def test_import_once_checks_cancels_then_persists_a_reviewed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise check/review/cancel and a fresh durable import in one screen."""
    notes_path = tmp_path / "notes.sqlite"
    receipt_path = tmp_path / "import-receipts.sqlite"
    source = tmp_path / "review-me.md"
    source.write_text("# Review me\n\nBody", encoding="utf-8")
    database = CharactersRAGDB(notes_path, client_id="task-19012-import")
    folders = LocalNoteFolderRepository(database)
    interop = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="task-19012-import",
        global_db_to_use=database,
    )
    scope_service = NotesScopeService(
        local_notes_service=interop,
        server_service=None,
        folder_repository=folders,
    )
    monkeypatch.setattr(
        library_screen_module,
        "get_notes_sync_state_db_path",
        lambda: receipt_path,
    )
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=[])
    app.chachanotes_db = database
    app.notes_scope_service = scope_service
    host = _JourneyHarness(app)

    try:
        async with host.run_test(size=(120, 36)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-notes", Button).press()
            await _open_import_once(screen, pilot, source)
            screen.query_one("#note-import-destination").value = "Inbox"
            await pilot.pause()
            screen.query_one("#note-import-check", Button).press()
            await _wait_for_selector(screen, pilot, "#note-import-import")
            assert database.count_notes() == 0
            assert folders.get_folder_by_path(("Inbox",)) is None

            controller = screen._library_note_import_controller
            original_factory = controller._executor_factory
            started = asyncio.Event()
            release = asyncio.Event()
            observed_cancel = None

            class _GatedCancelledExecutor:
                async def execute_async(
                    self, approved, *, cancel_event, progress_callback
                ) -> ImportExecutionReceipt:
                    nonlocal observed_cancel
                    del progress_callback
                    observed_cancel = cancel_event
                    started.set()
                    await release.wait()
                    return ImportExecutionReceipt(
                        approval_id=approved.approval_id,
                        state=ImportSessionState.CANCELLED,
                        total=1,
                        completed=0,
                        imported=0,
                        updated=0,
                        skipped=0,
                        failed=0,
                        retryable=0,
                    )

            controller._executor_factory = lambda *_args: _GatedCancelledExecutor()
            screen.query_one("#note-import-import", Button).press()
            await asyncio.wait_for(started.wait(), timeout=2)
            cancel = await _wait_for_selector(screen, pilot, "#note-import-cancel")
            cancel.press()
            await pilot.pause()
            assert observed_cancel is not None and observed_cancel.is_set()
            assert controller.snapshot.cancel_requested is True
            release.set()
            await _wait_for_selector(screen, pilot, "#note-import-receipt")
            assert controller.snapshot.receipt.state is ImportSessionState.CANCELLED
            assert database.count_notes() == 0

            controller._executor_factory = original_factory
            screen.query_one("#library-notes-import-back", Button).press()
            await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
            await _open_import_once(screen, pilot, source)
            screen.query_one("#note-import-destination").value = "Inbox"
            await pilot.pause()
            screen.query_one("#note-import-check", Button).press()
            await _wait_for_selector(screen, pilot, "#note-import-import")
            assert database.count_notes() == 0
            screen.query_one("#note-import-import", Button).press()
            await _wait_for_selector(screen, pilot, "#note-import-receipt")
            await _wait_for_condition(
                pilot,
                lambda: database.count_notes() == 1,
                message="reviewed import did not persist its one note",
            )

            receipt = controller.snapshot.receipt
            assert receipt.state is ImportSessionState.COMPLETED
            assert receipt.imported == 1
            assert folders.get_folder_by_path(("Inbox",)) is not None
            assert receipt_path.exists()
    finally:
        interop.close_all_user_connections()
        database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((120, 36), (60, 20)))
async def test_lasting_setup_keeps_server_unavailable_copy_painted(
    size: tuple[int, int],
) -> None:
    """The local setup never makes the missing server capability ambiguous."""
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.notes_sync_runtime_owner = _runtime()
    host = _JourneyHarness(app)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        screen.query_one("#library-notes-add-from-files", Button).press()
        await _wait_for_selector(screen, pilot, "#notes-add-keep-synced")
        screen.query_one("#notes-add-keep-synced", Button).press()
        server_reason = await _wait_for_selector(
            screen, pilot, "#notes-sync-server-disabled-reason"
        )
        server = screen.query_one("#notes-sync-destination-server", Button)
        assert server.disabled
        assert "server sync-folder capability not installed" in str(
            server_reason.renderable
        )
        screen.query_one("#notes-sync-body").scroll_to_widget(
            server_reason, animate=False
        )
        await pilot.pause()
        assert "server sync-folder capability not installed" in _painted_text(host)
        assert (
            screen.query_one("#notes-sync-back", Button)
            in host.screen._compositor.visible_widgets
        )
        screen.query_one("#notes-sync-back", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        assert screen._library_notes_view == "list"


@pytest.mark.asyncio
async def test_lasting_conflict_comparison_uses_named_worker_without_stealing_moved_focus() -> (
    None
):
    """A delayed comparison publishes inline but respects newer keyboard focus."""

    started = asyncio.Event()
    release = asyncio.Event()
    root = NotesSyncRootRuntimeSnapshot("root-1", "needs_attention", "review_changes")
    plan = ReconciliationPlan(
        root_id="root-1",
        observation_token="c" * 64,
        safe_actions=(),
        attention=(
            ReconciliationAttention(
                ReconciliationAttentionKind.CONFLICT,
                "both_sides_changed",
                "bind-1",
            ),
        ),
        skips=(),
        managed_placement_effects=(),
        deletion_groups=(),
    )

    class _ComparisonRuntime:
        def snapshot(self) -> NotesSyncRuntimeSnapshot:
            return NotesSyncRuntimeSnapshot("active", "sync_now", (root,))

        async def check_root(self, root_id: str) -> ReconciliationPlan:
            assert root_id == "root-1"
            return plan

        async def compare_conflict(
            self, root_id: str, observation_token: str, binding_id: str
        ) -> ConflictComparison:
            assert (root_id, observation_token, binding_id) == (
                "root-1",
                "c" * 64,
                "bind-1",
            )
            started.set()
            await release.wait()
            return ConflictComparison(
                binding_id="bind-1",
                note_title="Release note",
                relative_path="notes/release.md",
                note_version=3,
                note_updated_at=None,
                file_modified_ns=42,
                note_character_count=12,
                note_line_count=2,
                file_character_count=20,
                file_line_count=2,
                diff="--- Note\n+++ File\n-old\n+new\n",
                input_elided=False,
                output_elided=False,
            )

        async def conflict_labels(
            self, root_id: str, observation_token: str
        ) -> tuple[RuntimeConflictLabel, ...]:
            assert (root_id, observation_token) == ("root-1", "c" * 64)
            return (RuntimeConflictLabel("bind-1", "Release note", "notes/release.md"),)

    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.notes_sync_runtime_owner = _ComparisonRuntime()
    host = _JourneyHarness(app)
    async with host.run_test(size=(60, 20)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        manage = await _wait_for_selector(
            screen, pilot, "#library-notes-manage-sync-folders"
        )
        manage.press()
        review = await _wait_for_selector(screen, pilot, "#notes-sync-root-review-0")
        review.press()
        view = await _wait_for_selector(screen, pilot, "#notes-sync-conflict-view-0")
        view.focus()
        view.press()
        await asyncio.wait_for(started.wait(), timeout=2)
        comparison_workers = [
            worker
            for worker in screen.workers
            if worker.group == "library_notes_sync_comparison"
        ]
        assert len(comparison_workers) == 1

        moved_focus = screen.query_one("#notes-sync-history-open", Button)
        moved_focus.focus()
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is moved_focus,
            message="focus did not move away from the originating View button",
        )
        release.set()
        await _wait_for_condition(
            pilot,
            lambda: (
                bool(screen.query("#notes-sync-comparison-diff-0"))
                and screen.query_one("#notes-sync-comparison-0").display
            ),
            message="named comparison worker did not publish inline",
        )
        assert screen.focused is moved_focus

        returned = screen.query_one("#notes-sync-comparison-return-0", Button)
        returned.scroll_visible(immediate=True)
        returned.press()
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is view,
            message="Return did not restore the originating View control",
        )

        started.clear()
        release.clear()
        view.press()
        await asyncio.wait_for(started.wait(), timeout=2)
        back = screen.query_one("#notes-sync-back", Button)
        back.press()
        await _wait_for_selector(screen, pilot, "#notes-sync-roots-back")
        release.set()
        await pilot.pause()
        assert screen._library_notes_view == "lasting_roots"
        assert screen._library_notes_sync_controller.snapshot.comparison is None
        assert not screen.query("#notes-sync-comparison-diff-0")


def test_library_unmount_invalidates_lasting_review_before_first_await() -> None:
    """The screen fence is synchronous; controller races cover late facts."""

    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(library_screen_module.LibraryScreen.on_unmount)
        )
    )
    function = tree.body[0]
    assert isinstance(function, ast.AsyncFunctionDef)
    first_statement = function.body[1]
    assert isinstance(first_statement, ast.Expr)
    assert isinstance(first_statement.value, ast.Call)
    call = first_statement.value.func
    assert isinstance(call, ast.Attribute)
    assert call.attr == "invalidate_for_remount"


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery_action", ("apply", "undo"))
async def test_lasting_recovery_returns_to_roots_without_blank_add_canvas(
    recovery_action: str,
) -> None:
    token = "c" * 64
    root = NotesSyncRootRuntimeSnapshot("root-1", "needs_attention", "review_changes")
    plan = ReconciliationPlan(
        root_id="root-1",
        observation_token=token,
        safe_actions=(
            NotesSyncAction("action-1", NotesSyncActionKind.UPDATE_NOTE, "bind-1"),
        ),
        attention=(),
        skips=(),
        managed_placement_effects=(),
        deletion_groups=(),
    )

    class _RecoveryRuntime:
        def __init__(self) -> None:
            self.receipts: tuple[RuntimeConflictReceipt, ...] = ()

        def snapshot(self) -> NotesSyncRuntimeSnapshot:
            return NotesSyncRuntimeSnapshot("active", "sync_now", (root,))

        async def check_root(self, root_id: str) -> ReconciliationPlan:
            assert root_id == "root-1"
            return plan

        async def conflict_labels(
            self, root_id: str, observation_token: str
        ) -> tuple[RuntimeConflictLabel, ...]:
            assert (root_id, observation_token) == ("root-1", token)
            return ()

        async def active_conflict_receipts(
            self, root_id: str
        ) -> tuple[RuntimeConflictReceipt, ...]:
            assert root_id == "root-1"
            return self.receipts

        async def apply_reviewed(
            self,
            root_id: str,
            observation_token: str,
            action_ids: tuple[str, ...],
            selections: tuple[object, ...],
        ) -> ConflictApplyResult:
            assert (root_id, observation_token, action_ids, selections) == (
                "root-1",
                token,
                ("action-1",),
                (),
            )
            return ConflictApplyResult((), 0, 0, 0, False, True, True, None)

        async def undo_resolution(
            self, root_id: str, operation_id: str
        ) -> NotesSyncExecutionResult:
            assert (root_id, operation_id) == ("root-1", "operation-1")
            return NotesSyncExecutionResult(
                "undo-operation-1", NotesSyncOperationState.VERIFIED, False
            )

    runtime = _RecoveryRuntime()
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.notes_sync_runtime_owner = runtime
    host = _JourneyHarness(app)
    async with host.run_test(size=(60, 20)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        manage = await _wait_for_selector(
            screen, pilot, "#library-notes-manage-sync-folders"
        )
        manage.press()
        review = await _wait_for_selector(screen, pilot, "#notes-sync-root-review-0")
        review.press()
        await _wait_for_selector(screen, pilot, "#notes-sync-apply")

        if recovery_action == "apply":
            screen.query_one("#notes-sync-apply", Button).press()
        else:
            runtime.receipts = (
                RuntimeConflictReceipt(
                    "operation-1",
                    "Release note",
                    NotesSyncConflictChoice.KEEP_FILE,
                    "completed",
                    True,
                ),
            )
            controller = screen._library_notes_sync_controller
            await controller.refresh_conflict_receipts("root-1")
            controller._state = replace(  # noqa: SLF001 - mounted receipt setup
                controller.snapshot,
                phase="receipt",
                receipt_line="1 applied · durable receipt recorded",
            )
            controller._publish()  # noqa: SLF001 - mounted receipt setup
            undo = await _wait_for_selector(screen, pilot, "#notes-sync-receipt-undo-0")
            undo.press()

        await _wait_for_selector(screen, pilot, "#notes-sync-roots-back")
        assert screen._library_notes_view == "lasting_roots"
        assert not screen.query("#notes-sync-apply")
        assert screen._library_notes_sync_controller.snapshot.phase == "roots"


@pytest.mark.asyncio
async def test_lasting_review_activation_receipt_and_remount_recovery_journey(
    tmp_path: Path,
) -> None:
    """Review attention, activate a clean retry, then recover after remount."""
    token = "a" * 64
    folder = tmp_path / "lasting-root"
    folder.mkdir()

    class _DurableRuntime:
        def __init__(self) -> None:
            self.attention = True
            self.roots: tuple[NotesSyncRootRuntimeSnapshot, ...] = ()
            self.calls: list[tuple[object, ...]] = []

        def snapshot(self) -> NotesSyncRuntimeSnapshot:
            return NotesSyncRuntimeSnapshot("active", "sync_now", self.roots)

        async def review_setup(self, setup) -> ReconciliationPlan:
            self.calls.append(("review_setup", setup.display_name))
            return ReconciliationPlan(
                root_id="setup-root",
                observation_token=token,
                safe_actions=(
                    NotesSyncAction(
                        "create-note-1",
                        NotesSyncActionKind.CREATE_NOTE,
                        "binding-1",
                    ),
                ),
                attention=(
                    ReconciliationAttention(
                        ReconciliationAttentionKind.CONFLICT,
                        "both_changed",
                        "binding-attention",
                    ),
                )
                if self.attention
                else (),
                skips=(),
                managed_placement_effects=(),
                deletion_groups=(),
            )

        async def conflict_labels(
            self, root_id: str, observation_token: str
        ) -> tuple[RuntimeConflictLabel, ...]:
            del root_id, observation_token
            return ()

        async def abandon_setup(self, root_id: str) -> None:
            self.calls.append(("abandon_setup", root_id))

        async def activate_root(
            self, root_id: str, authorization: object
        ) -> NotesSyncControlResult:
            self.calls.append(("activate_root", root_id, authorization))
            self.roots = (
                NotesSyncRootRuntimeSnapshot(root_id, "up_to_date", "sync_now"),
            )
            return NotesSyncControlResult(
                True, "up_to_date", "sync_now", applied_count=1
            )

        async def resolve_cleanup(self, root_id: str, operation_id: str) -> object:
            self.calls.append(("resolve_cleanup", root_id, operation_id))
            self.roots = (
                NotesSyncRootRuntimeSnapshot(root_id, "up_to_date", "sync_now"),
            )
            return object()

    runtime = _DurableRuntime()

    async def enter_setup(screen, pilot) -> None:
        add_from_files = screen.query_one("#library-notes-add-from-files", Button)
        add_from_files.press()
        await _wait_for_selector(screen, pilot, "#notes-add-keep-synced")
        screen.query_one("#notes-add-keep-synced", Button).press()
        await _wait_for_selector(screen, pilot, "#notes-sync-display-name")
        controller = screen._library_notes_sync_controller
        controller.set_setup("display_name", "Reviewed folder")
        controller.set_setup("folder", str(folder))
        await pilot.pause()

    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.notes_sync_runtime_owner = runtime
    host = _JourneyHarness(app)
    async with host.run_test(size=(60, 20)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        await enter_setup(screen, pilot)
        screen.query_one("#notes-sync-check", Button).press()
        activate = await _wait_for_selector(screen, pilot, "#notes-sync-activate")
        attention_choice = screen.query_one("#notes-sync-attention-1-0", Button)
        assert activate.disabled
        assert attention_choice.disabled
        assert "1 safe · 1 need attention" in _painted_text(host)

        screen.query_one("#notes-sync-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_notes_view == "list",
            message="lasting Back did not return to Notes",
        )
        assert ("abandon_setup", "setup-root") in runtime.calls

    runtime.attention = False
    activation_app = _build_test_app()
    _seed_conversations(activation_app, [], notes=_two_notes())
    activation_app.notes_sync_runtime_owner = runtime
    activation_host = _JourneyHarness(activation_app)
    async with activation_host.run_test(size=(60, 20)) as pilot:
        screen = _active_library_screen(activation_host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        await enter_setup(screen, pilot)
        screen.query_one("#notes-sync-check", Button).press()
        activate = await _wait_for_selector(screen, pilot, "#notes-sync-activate")
        assert not activate.disabled
        activate.press()
        await _wait_for_selector(screen, pilot, "#notes-sync-receipt")
        assert "1 applied · durable receipt recorded" in _painted_text(activation_host)
        assert ("activate_root", "setup-root", token) in runtime.calls

    runtime.roots = (
        NotesSyncRootRuntimeSnapshot(
            "setup-root", "partial", "resolve_cleanup", "operation-1"
        ),
    )
    restarted_app = _build_test_app()
    _seed_conversations(restarted_app, [], notes=_two_notes())
    restarted_app.notes_sync_runtime_owner = runtime
    restarted = _JourneyHarness(restarted_app)
    async with restarted.run_test(size=(60, 20)) as pilot:
        screen = _active_library_screen(restarted)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        manage = await _wait_for_selector(
            screen, pilot, "#library-notes-manage-sync-folders"
        )
        manage.press()
        recover = await _wait_for_selector(screen, pilot, "#notes-sync-root-recover-0")
        assert recover.has_class("console-action-primary")
        assert "Partial · Next: Resolve recovery" in _painted_text(restarted)
        recover.press()
        await _wait_for_condition(
            pilot,
            lambda: ("resolve_cleanup", "setup-root", "operation-1") in runtime.calls,
            message="restart recovery did not reach the runtime",
        )
        await _wait_for_condition(
            pilot,
            lambda: "Recovery reviewed" in _painted_text(restarted),
            message="recovery status did not reach the compositor",
        )
        assert runtime.roots[0].status == "up_to_date"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ("setup", "root", "migration"))
async def test_check_again_routes_to_its_rendered_review_source(
    tmp_path: Path, source: str
) -> None:
    token = "a" * 64
    root_id = "root-1"
    folder = tmp_path / source
    folder.mkdir()
    root = (
        NotesSyncRootRuntimeSnapshot(
            root_id,
            "paused" if source == "migration" else "needs_attention",
            "review_migration" if source == "migration" else "review_changes",
        ),
    )
    release_migration = asyncio.Event()

    class _SourceRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def snapshot(self) -> NotesSyncRuntimeSnapshot:
            return NotesSyncRuntimeSnapshot(
                "active", "sync_now", () if source == "setup" else root
            )

        @staticmethod
        def _plan(selected_root: str) -> ReconciliationPlan:
            return ReconciliationPlan(
                root_id=selected_root,
                observation_token=token,
                safe_actions=(
                    NotesSyncAction(
                        "action-1", NotesSyncActionKind.UPDATE_NOTE, "binding-1"
                    ),
                ),
                attention=(),
                skips=(),
                managed_placement_effects=(),
                deletion_groups=(),
            )

        async def review_setup(self, _setup: object) -> ReconciliationPlan:
            self.calls.append(("review_setup", "setup-root"))
            return self._plan("setup-root")

        async def check_root(self, selected_root: str) -> ReconciliationPlan:
            self.calls.append(("check_root", selected_root))
            check_count = len([call for call in self.calls if call[0] == "check_root"])
            if source == "migration" and check_count == 1:
                await release_migration.wait()
            return self._plan(selected_root)

        async def conflict_labels(
            self, _root_id: str, _token: str
        ) -> tuple[RuntimeConflictLabel, ...]:
            return ()

        async def apply_reviewed(self, *_args: object) -> ConflictApplyResult:
            raise ValueError("stale_review")

        async def activate_root(self, *_args: object) -> NotesSyncControlResult:
            raise RuntimeError("activation failed")

        async def abandon_setup(self, selected_root: str) -> None:
            self.calls.append(("abandon_setup", selected_root))

    runtime = _SourceRuntime()
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.notes_sync_runtime_owner = runtime
    host = _JourneyHarness(app)
    async with host.run_test(size=(60, 20)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        if source == "setup":
            screen.query_one("#library-notes-add-from-files", Button).press()
            await _wait_for_selector(screen, pilot, "#notes-add-keep-synced")
            screen.query_one("#notes-add-keep-synced", Button).press()
            await _wait_for_selector(screen, pilot, "#notes-sync-display-name")
            controller = screen._library_notes_sync_controller
            controller.set_setup("display_name", "Source review")
            controller.set_setup("folder", str(folder))
            screen.query_one("#notes-sync-check", Button).press()
        else:
            screen.query_one("#library-notes-manage-sync-folders", Button).press()
            selector = (
                "#notes-sync-root-migration-0"
                if source == "migration"
                else "#notes-sync-root-review-0"
            )
            action = await _wait_for_selector(screen, pilot, selector)
            action.press()
            if source == "migration":
                await _wait_for_condition(
                    pilot,
                    lambda: any(call[0] == "check_root" for call in runtime.calls),
                    message="migration review did not reach its awaited check",
                )
                opened_before_await = (
                    screen._library_notes_view == "lasting_add"
                    and screen._library_notes_lasting_origin == "roots"
                )
                release_migration.set()
                assert opened_before_await

        action_selector = (
            "#notes-sync-apply" if source == "root" else "#notes-sync-activate"
        )
        action = await _wait_for_selector(screen, pilot, action_selector)
        assert screen._library_notes_sync_controller.snapshot.review.source == source
        action.press()
        check_again = await _wait_for_selector(screen, pilot, "#notes-sync-check-again")
        call_name = "review_setup" if source == "setup" else "check_root"
        before = len([call for call in runtime.calls if call[0] == call_name])
        check_again.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                len([call for call in runtime.calls if call[0] == call_name])
                == before + 1
            ),
            message=f"{source} Check again did not route to {call_name}",
        )
        assert screen._library_notes_sync_controller.snapshot.review.source == source


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ("root", "migration"))
@pytest.mark.parametrize("outcome", ("exception", "malformed"))
async def test_failed_persisted_check_again_uses_rendered_source_in_mounted_screen(
    source: str, outcome: str
) -> None:
    token = "a" * 64
    root_id = "root-1"
    root = NotesSyncRootRuntimeSnapshot(
        root_id,
        "paused" if source == "migration" else "needs_attention",
        "review_migration" if source == "migration" else "review_changes",
    )

    class _FailedCheckRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []
            self.attempts = 0

        def snapshot(self) -> NotesSyncRuntimeSnapshot:
            return NotesSyncRuntimeSnapshot("active", "sync_now", (root,))

        async def check_root(self, selected_root: str) -> ReconciliationPlan:
            self.attempts += 1
            self.calls.append(("check_root", selected_root))
            if self.attempts == 1:
                if outcome == "exception":
                    raise RuntimeError("check failed")
                return object()  # type: ignore[return-value]
            return ReconciliationPlan(
                root_id=selected_root,
                observation_token=token,
                safe_actions=(),
                attention=(),
                skips=(),
                managed_placement_effects=(),
                deletion_groups=(),
            )

        async def conflict_labels(
            self, _root_id: str, _token: str
        ) -> tuple[RuntimeConflictLabel, ...]:
            return ()

    runtime = _FailedCheckRuntime()
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.notes_sync_runtime_owner = runtime
    host = _JourneyHarness(app)
    async with host.run_test(size=(60, 20)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        screen.query_one("#library-notes-manage-sync-folders", Button).press()
        selector = (
            "#notes-sync-root-migration-0"
            if source == "migration"
            else "#notes-sync-root-review-0"
        )
        (await _wait_for_selector(screen, pilot, selector)).press()
        retry = await _wait_for_selector(screen, pilot, "#notes-sync-check-again")
        failed = screen._library_notes_sync_controller.snapshot.review

        assert failed.source == source
        assert failed.root_id == root_id
        assert failed.activation is (source == "migration")
        retry.press()
        await _wait_for_condition(
            pilot,
            lambda: runtime.attempts == 2,
            message=f"{source} failed Check again did not reach check_root",
        )
        await _wait_for_selector(
            screen,
            pilot,
            "#notes-sync-activate" if source == "migration" else "#notes-sync-apply",
        )

    assert runtime.calls == [("check_root", root_id), ("check_root", root_id)]


@pytest.mark.asyncio
async def test_lasting_attention_survives_a_fresh_screen_and_prioritizes_review() -> (
    None
):
    """A fresh remount keeps projected attention and its next action."""
    root = NotesSyncRootRuntimeSnapshot(
        "root.attention:opaque", "needs_attention", "review_changes", "op-1"
    )
    for visit in range(2):
        app = _build_test_app()
        _seed_conversations(app, [], notes=_two_notes())
        app.notes_sync_runtime_owner = _runtime(root)
        host = _JourneyHarness(app)
        async with host.run_test(size=(60, 20)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-notes", Button).press()
            manage = await _wait_for_selector(
                screen, pilot, "#library-notes-manage-sync-folders"
            )
            manage.press()
            review = await _wait_for_selector(
                screen, pilot, "#notes-sync-root-review-0"
            )
            await pilot.pause()

            assert review.has_class("console-action-primary"), visit
            assert "Needs attention · Next: Review changes" in _painted_text(host)
            assert "name unavailable before cutover" in _painted_text(host)
            assert screen.query_one("#notes-sync-root-retarget-0", Button).disabled
            retarget = screen.query_one("#notes-sync-root-retarget-0", Button)
            assert retarget.disabled
            assert screen.query_one("#notes-sync-root-disconnect-0", Button).disabled
            _assert_legible_painted_text(
                host,
                retarget,
                "○ Retarget",
                theme_name="textual-dark",
            )
            assert review in host.screen._compositor.visible_widgets
            assert (
                screen.query_one("#notes-sync-roots-back", Button)
                in host.screen._compositor.visible_widgets
            )


@pytest.mark.asyncio
async def test_lasting_runtime_reopens_and_resumes_a_durable_incomplete_journal(
    tmp_path: Path,
) -> None:
    """A new runtime instance claims and resumes an on-disk pending operation."""
    original = _runtime_store(tmp_path)
    _pending_operation(original)
    assert (
        original.get_operation("operation-1").state is NotesSyncOperationState.PENDING
    )

    reopened = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    adapter = _RuntimeAdapter([_runtime_input()])
    owner, coordinator, _watcher = _runtime_owner(
        store=reopened,
        admitted=True,
        adapter=adapter,
    )

    await owner.start()
    try:
        assert coordinator.acquire_calls == 1
        assert adapter.executor.reconstructed == ["operation-1"]
        assert adapter.executor.resumed == ["operation-1"]
        assert reopened.get_operation("operation-1").state is (
            NotesSyncOperationState.PENDING
        )
        assert owner.snapshot().roots[0].status == "up_to_date"
    finally:
        await owner.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("choice", "label"),
    (
        (NotesSyncConflictChoice.KEEP_FILE, "Keep file"),
        (NotesSyncConflictChoice.KEEP_NOTE, "Keep note"),
        (NotesSyncConflictChoice.KEEP_BOTH, "Keep both"),
        (NotesSyncConflictChoice.SKIP, "Skip for now"),
    ),
)
async def test_real_authority_conflict_choice_journey_is_durable_and_undoable(
    tmp_path: Path,
    choice: NotesSyncConflictChoice,
    label: str,
) -> None:
    """Join Check, comparison, staging, apply, restart, history, and Undo."""

    case = tmp_path / choice.value
    case.mkdir()
    notes_path, state_path, sync_root = _seed_real_conflict_authority(case)
    outside = case / "outside.md"
    outside.write_text("outside sentinel", encoding="utf-8")
    owner, database, interop, controller = await _start_real_conflict_stack(
        notes_path, state_path
    )
    operation_id: str | None = None
    try:
        await controller.check_root("root-1")
        reviewed = controller.snapshot.review
        assert reviewed.rows, controller.snapshot.status_line
        row = next(item for item in reviewed.rows if item.item_id == "binding-1")
        assert row.conflict_eligible
        await controller.show_conflict_comparison(
            "root-1", reviewed.observation_token, "binding-1"
        )
        comparison = controller.snapshot.comparison
        assert comparison is not None
        assert comparison.diff.startswith("--- Note\n+++ File\n")
        controller.return_to_conflict_choices(
            "root-1", reviewed.observation_token, "binding-1"
        )
        controller.stage_attention_choice(
            "root-1", reviewed.observation_token, "binding-1", label
        )
        staged_note = database.get_note_by_id("note-1")
        assert staged_note is not None and staged_note["content"] == "note side"
        assert (sync_root / "note.md").read_text(encoding="utf-8") == "file side"

        await controller.apply_reviewed("root-1", reviewed.observation_token)

        if choice is NotesSyncConflictChoice.SKIP:
            assert controller.snapshot.review.can_apply is False
            assert controller.snapshot.receipts == ()
            assert "cannot be applied" in controller.snapshot.status_line
        else:
            receipt = controller.snapshot.receipts
            assert len(receipt) == 1
            assert receipt[0].choice is choice
            assert receipt[0].undo_available
            operation_id = receipt[0].operation_id
            note = database.get_note_by_id("note-1")
            assert note is not None
            expected_note = (
                "note side"
                if choice is NotesSyncConflictChoice.KEEP_NOTE
                else "file side"
            )
            expected_file = (
                "note side"
                if choice is NotesSyncConflictChoice.KEEP_NOTE
                else "file side"
            )
            assert note["content"] == expected_note
            assert (sync_root / "note.md").read_text(encoding="utf-8") == expected_file
            assert database.count_notes() == (
                2 if choice is NotesSyncConflictChoice.KEEP_BOTH else 1
            )
    finally:
        await _close_real_conflict_stack(owner, database, interop)

    assert outside.read_text(encoding="utf-8") == "outside sentinel"
    with PosixNotesSyncFilesystem(sync_root) as filesystem:
        with pytest.raises(NotesSyncFilesystemError, match="invalid_relative_path"):
            filesystem.observe("../outside.md")

    (
        fresh_owner,
        fresh_database,
        fresh_interop,
        fresh_controller,
    ) = await _start_real_conflict_stack(notes_path, state_path)
    try:
        await fresh_controller.check_root("root-1")
        fresh_token = fresh_controller.snapshot.review.observation_token
        await fresh_controller.show_resolution_history("root-1")
        history = fresh_controller.snapshot.history
        if choice is NotesSyncConflictChoice.SKIP:
            assert history.rows == ()
            unchanged_note = fresh_database.get_note_by_id("note-1")
            assert unchanged_note is not None
            assert unchanged_note["content"] == "note side"
            assert (sync_root / "note.md").read_text(encoding="utf-8") == "file side"
            return

        assert operation_id is not None
        assert len(history.rows) == 1
        assert history.rows[0].operation_id == operation_id
        assert history.rows[0].choice is choice
        assert history.rows[0].undo_available
        await fresh_controller.undo_conflict_resolution(
            "root-1", fresh_token, operation_id, history_page=1
        )
        restored = fresh_database.get_note_by_id("note-1")
        assert restored is not None and restored["content"] == "note side"
        assert (sync_root / "note.md").read_text(encoding="utf-8") == "file side"
        assert fresh_database.count_notes() == 1
        assert fresh_controller.snapshot.history.rows[0].state == "undone"
    finally:
        await _close_real_conflict_stack(fresh_owner, fresh_database, fresh_interop)


@pytest.mark.asyncio
async def test_folder_files_and_session_git_use_supported_40x20_navigator(
    tmp_path: Path,
) -> None:
    """Exercise Folder files and Session Git through the real Library route."""
    root = tmp_path / "scratch-folder-notes"
    root.mkdir()
    (root / "entry.md").write_text("scratch\n", encoding="utf-8")
    subprocess.run(("git", "init", "--quiet", str(root)), check=True, timeout=10)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=10,
    )
    try:
        async with _production_workspace_context(workspace, size=(40, 20)) as pilot:
            await pilot.pause()
            authority = workspace.query_one("#file-notes-authority", Static)
            session_git = workspace.query_one("#file-notes-session-changes", Button)
            assert "Folder files" in _painted_text(pilot.app)
            assert authority in pilot.app.screen._compositor.visible_widgets
            assert session_git in pilot.app.screen._compositor.visible_widgets

            session_git.focus()
            session_git.press()
            await _wait_until(
                pilot,
                lambda: (
                    bool(workspace.query("#file-notes-git-back"))
                    and workspace.query_one("#file-notes-git-back", Button).display
                ),
                "Session Git did not open at 40x20",
            )
            await pilot.pause()
            assert "Session Git" in _painted_text(pilot.app)
            assert (
                workspace.query_one("#file-notes-git-back", Button)
                in pilot.app.screen._compositor.visible_widgets
            )
            assert getattr(pilot.app.focused, "id", None) in {
                "file-notes-git-back",
                "file-notes-git-rows",
            }
    finally:
        replica.close()


@pytest.mark.asyncio
async def test_session_git_executes_stage_commit_cancel_push_cancel_and_result(
    tmp_path: Path,
) -> None:
    """Physically drive one staged commit and guarded push through Library."""
    owner, binding, replica, service, workspace = _push_workspace_fixture(tmp_path)
    try:
        async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
            session_git = workspace.query_one(
                "#file-notes-session-changes",
                Button,
            )
            session_git.press()
            await _wait_until(
                pilot,
                lambda: len(service.status_calls) == 1,
                "Session Git status did not load",
            )
            await _wait_for_current_git_row_projection(workspace)

            service.rows = (
                _row("owned", group_id=1, unstage_eligible=True),
                _row("owned", group_id=2, unstage_eligible=True),
            )
            workspace.query_one(
                "#file-notes-git-stage-selected",
                Button,
            ).press()
            await _wait_until(
                pilot,
                lambda: (
                    service.stage_calls == [(1,)] and len(service.status_calls) == 2
                ),
                "physical Stage did not publish the refreshed staged rows",
            )
            await _wait_for_current_git_row_projection(workspace)
            commit = workspace.query_one("#file-notes-git-commit-staged", Button)
            assert commit.display and str(commit.label) == "Commit staged (2)"

            commit.press()
            await _wait_until(
                pilot,
                lambda: workspace._git_panel_widget.commit_phase == "form",
                "commit form did not open",
            )
            workspace.query_one(
                "#file-notes-git-commit-subject", Input
            ).value = "Cancelled draft"
            workspace.query_one("#file-notes-git-commit-cancel", Button).press()
            await _wait_until(
                pilot,
                lambda: workspace._git_panel_widget.commit_phase == "list",
                "commit Cancel did not return to Session Git",
            )
            assert service.commit_calls == []

            service.commit_outcomes.append(
                CommitOutcome(
                    "succeeded",
                    "Committed 2 reviewed session notes.",
                    commit_object_id="b" * 40,
                    committed_note_count=2,
                )
            )
            status_generation = owner.next_status_generation(binding)
            assert status_generation is not None
            service.published_commit_status = SessionGitStatus(
                binding_generation=binding.generation,
                status_generation=status_generation,
                state="ready",
                rows=(),
                repository=service.repository,
                head=service.head,
            )
            commit.press()
            await _wait_until(
                pilot,
                lambda: workspace._git_panel_widget.commit_phase == "form",
                "fresh commit form did not open",
            )
            workspace.query_one(
                "#file-notes-git-commit-subject", Input
            ).value = "Reviewed session notes"
            await pilot.pause()
            workspace.query_one("#file-notes-git-commit-review", Button).press()
            await _wait_until(
                pilot,
                lambda: workspace._git_panel_widget.commit_phase == "review",
                "commit review did not render",
            )
            workspace.query_one("#file-notes-git-commit-confirm", Button).press()
            await _wait_until(
                pilot,
                lambda: (
                    len(service.commit_calls) == 1
                    and workspace._git_panel_widget.commit_phase == "list"
                ),
                "reviewed commit result did not settle",
            )
            assert workspace._action_detail == "Committed 2 reviewed session notes."

            _publish_candidate_on_owner(
                owner,
                binding,
                service.repository,
                parent_oid="a" * 40,
                candidate_oid="d" * 40,
            )
            candidate_a = owner.snapshot(binding).push_candidate
            assert candidate_a is not None
            service.head = replace(
                service.head,
                branch=candidate_a.candidate.local_branch_ref,
                object_id=candidate_a.candidate.candidate_oid,
            )
            workspace._rehydrate_git_presentation()
            cancel_release = asyncio.Event()
            service.plan_push_operation(
                "local_proof",
                _push_destination_policy_result("blocked"),
                cancel_release,
            )
            service.cancel_push_result = True
            workspace.query_one("#file-notes-git-push-review", Button).press()
            await _wait_until(
                pilot,
                lambda: (
                    service.push_review_calls == [binding]
                    and workspace._git_panel_widget.push_phase == "checking_candidate"
                ),
                "push review did not enter its cancellable check",
            )
            observer = workspace._push_observer_task
            workspace.query_one("#file-notes-git-push-cancel", Button).press()
            await _wait_until(
                pilot,
                lambda: workspace._git_panel_widget.push_phase == "list",
                "push Cancel did not return to Session Git",
            )
            assert service.cancel_push_calls == [binding]
            cancel_release.set()
            assert observer is not None
            await observer

            _publish_candidate_on_owner(
                owner,
                binding,
                service.repository,
                parent_oid="d" * 40,
                candidate_oid="e" * 40,
            )
            candidate_b = owner.snapshot(binding).push_candidate
            assert candidate_b is not None and candidate_b != candidate_a
            service.head = replace(
                service.head,
                branch=candidate_b.candidate.local_branch_ref,
                object_id=candidate_b.candidate.candidate_oid,
            )
            workspace._rehydrate_git_presentation()
            destination = _push_destination_projection()
            local_release = asyncio.Event()
            preflight_release = asyncio.Event()
            push_release = asyncio.Event()
            handle = object.__new__(PushReviewHandle)
            review = PushReviewProjection(candidate_b.candidate, destination, "origin")
            push_result = PushExecutionResult(
                "succeeded",
                push_outcome_copy("succeeded"),
            )
            service.plan_push_operation(
                "local_proof",
                _push_destination_policy_result("ready", destination),
                local_release,
            )
            service.plan_push_operation(
                "preflight",
                PushPreflightResult("review", handle, review),
                preflight_release,
            )
            service.plan_push_operation("push", push_result, push_release)

            workspace.query_one("#file-notes-git-push-review", Button).press()
            await _wait_until(
                pilot,
                lambda: service.push_review_calls == [binding, binding],
                "fresh candidate push review did not start",
            )
            local_release.set()
            await _wait_until(
                pilot,
                lambda: isinstance(
                    workspace.app.screen,
                    git_panel_module.PushDestinationAuthorizationDialog,
                ),
                "push destination authorization did not open",
            )
            workspace.app.screen.query_one(
                "#file-notes-push-auth-confirm",
                Button,
            ).press()
            await _wait_until(
                pilot,
                lambda: len(service.authorize_and_check_calls) == 1,
                "authorized push preflight did not start",
            )
            preflight_release.set()
            await _wait_until(
                pilot,
                lambda: workspace._git_panel_widget.push_phase == "review",
                "immutable push review did not render",
            )
            workspace.query_one("#file-notes-git-push-confirm", Button).press()
            await _wait_until(
                pilot,
                lambda: service.push_calls == [binding],
                "reviewed push did not start",
            )
            service.mark_push_child_started()
            push_release.set()
            await _wait_until(
                pilot,
                lambda: workspace._git_panel_widget.push_phase == "result",
                "push result did not render",
            )
            result_copy = workspace.query_one(
                "#file-notes-git-push-result-copy",
                TextArea,
            )
            assert result_copy.text == push_result.outcome.message
            workspace.query_one("#file-notes-git-push-back-session", Button).press()
            await _wait_until(
                pilot,
                lambda: workspace._git_panel_widget.push_phase == "list",
                "push result Back did not return to Session Git",
            )
    finally:
        await workspace.shutdown()
        owner.shutdown()
        replica.close()
