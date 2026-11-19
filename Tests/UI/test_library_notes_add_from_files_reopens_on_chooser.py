"""TASK-34000.17: Notes "Add from files…" always opens on the chooser.

Review finding N-17 (notes-library-ux-review-2026-10-02): "Add from files…"
is the one door into both Import once and Keep a folder synced, and the
lasting-sync phase was set to ``choose`` exactly once, at construction.
After activating a folder, after a check that found nothing to review, after
a failed Recovery, and after leaving the Keep synced form with ‹ Notes, the
button reopened whatever page the sync controller had been left on -- a
stale receipt, a blank ``roots`` page with no way back, or the configure
form. Only a restart cleared it.

The walks here are the review's repros on the real Library stack (one real
``NotesSyncRuntimeOwner`` over a disposable profile for the activation walk;
a mounted fake runtime for the arms whose outcome a real vault cannot be
made to produce on demand: an empty manual check, a Recovery that fails, and
a check that is held open). Each test boots once.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_notes_files_sync_journey import (
    _isolate_actor_pack_recovery,  # noqa: F401 - autouse fixture, re-exported
    _JourneyHarness,
    _painted_text,
    _painted_widget_text,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _seed_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Notes.notes_sync_models import NotesSyncAction, NotesSyncActionKind
from tldw_chatbook.Notes.notes_sync_reconciler import ReconciliationPlan
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncRootRuntimeSnapshot,
    NotesSyncRuntimeSnapshot,
    RuntimeConflictLabel,
    build_notes_sync_runtime_owner,
)
from tldw_chatbook.Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

_CHOOSER_SELECTORS = (
    "#notes-add-import-once",
    "#notes-add-keep-synced",
    "#notes-sync-back",
)


def _empty_plan(root_id: str) -> ReconciliationPlan:
    return ReconciliationPlan(
        root_id=root_id,
        observation_token="d" * 64,
        safe_actions=(),
        attention=(),
        skips=(),
        managed_placement_effects=(),
        deletion_groups=(),
    )


async def _open_notes(screen, pilot) -> None:
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")


async def _reopen_add_from_files(screen, pilot) -> None:
    """Press Add from files… and wait until the lasting canvas has painted."""

    screen.query_one("#library-notes-add-from-files", Button).press()
    controller = screen._library_notes_sync_controller

    def _canvas_settled() -> bool:
        if screen._notes_state.view != "lasting_add":
            return False
        canvases = list(screen.query(LibraryNotesAddFromFilesCanvas))
        return bool(canvases) and (
            canvases[0].snapshot.phase == controller.snapshot.phase
        )

    await _wait_for_condition(
        pilot,
        _canvas_settled,
        message="Add from files… never mounted the lasting canvas",
    )
    await pilot.pause()


def _assert_chooser(screen, host) -> None:
    """The reopened page IS the chooser: both choices, a way back, Esc named."""

    controller = screen._library_notes_sync_controller
    painted = _painted_text(host)
    context = (
        f"phase={controller.snapshot.phase!r} "
        f"status={controller.snapshot.status_line!r}\n{painted}"
    )
    for selector in _CHOOSER_SELECTORS:
        assert screen.query(selector), (
            f"{selector} is not on the reopened Add from files page:\n{context}"
        )
    # Availability is the label, not ``disabled``: the inert chooser keeps its
    # button pressable so it can explain the gate, and wears a "○" prefix.
    assert (
        str(screen.query_one("#notes-add-import-once", Button).label) == "Import once"
    )
    keep_label = str(screen.query_one("#notes-add-keep-synced", Button).label)
    assert keep_label == "Keep a folder synced", f"Keep synced is inert: {keep_label!r}"
    assert "Nothing chosen yet." in painted, context
    assert controller.snapshot.phase == "choose", context
    assert "esc" in dict(screen._library_notes_footer_shortcuts()), (
        f"footer does not name Esc: {screen._library_notes_footer_shortcuts()!r}"
    )


async def _leave_with_notes_back(
    screen, pilot, selector: str = "#notes-sync-back"
) -> None:
    # The settled canvas may still be mid-recompose when the status line
    # flipped; wait for the live back control rather than racing the prune.
    back = await _wait_for_selector(screen, pilot, selector)
    back.press()
    await _wait_for_condition(
        pilot,
        lambda: screen._notes_state.view == "list",
        message=f"{selector} did not return to the Notes list",
    )
    await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")


async def _activate_root(
    screen, pilot, controller, *, display_name: str, folder: Path, from_chooser: bool
) -> None:
    """The j3 walk: Keep a folder synced ▸ check ▸ Activate, up to the receipt."""

    if not from_chooser:
        screen.query_one("#library-notes-add-from-files", Button).press()
    await _wait_for_selector(screen, pilot, "#notes-add-keep-synced")
    screen.query_one("#notes-add-keep-synced", Button).press()
    await _wait_for_selector(screen, pilot, "#notes-sync-display-name")
    controller.set_setup("display_name", display_name)
    controller.set_setup("folder", str(folder))
    await pilot.pause()
    screen.query_one("#notes-sync-check", Button).press()
    await _wait_for_selector(screen, pilot, "#notes-sync-activate")
    await _wait_for_condition(
        pilot,
        lambda: not screen.query_one("#notes-sync-activate", Button).disabled,
        message=f"review never admitted activation: {controller.snapshot.status_line}",
    )
    screen.query_one("#notes-sync-activate", Button).press()
    await _wait_for_selector(screen, pilot, "#notes-sync-receipt")


async def test_add_from_files_reopens_on_the_chooser_after_activating_a_root(
    tmp_path: Path,
) -> None:
    """AC#1 (after activation) and AC#4 on the real runtime.

    j3's walk (captures 59-60): activate a folder, press ‹ Notes, press Add
    from files… -- the receipt came back. Then the receipt's own bar offers
    Manage sync folders, and a second folder is set up from the chooser
    without a restart.
    """

    notes_path = tmp_path / "notes.sqlite3"
    state_path = tmp_path / "sync.sqlite3"
    vaults: list[Path] = []
    for suffix in ("a", "b"):
        vault = (tmp_path / f"vault-{suffix}").resolve()
        vault.mkdir()
        for index in range(3):
            (vault / f"synced-{suffix}-{index}.md").write_text(
                f"# Synced {suffix} {index}\n\nbody {index}\n", encoding="utf-8"
            )
        vaults.append(vault)
    database = CharactersRAGDB(notes_path, client_id="task-34000-17")
    folders = LocalNoteFolderRepository(database)
    for index in range(2):
        assert database.add_note(
            f"Existing {index}", f"existing body {index}", f"existing-{index}"
        )
    interop = NotesInteropService(
        base_db_directory=notes_path.parent,
        api_client_id="task-34000-17",
        global_db_to_use=database,
    )
    scope_service = NotesScopeService(
        local_notes_service=interop,
        server_service=None,
        folder_repository=folders,
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
    app = _build_test_app()
    _seed_conversations(app, [])
    app.notes_scope_service = scope_service
    app.notes_sync_runtime_owner = owner
    host = _JourneyHarness(app)
    try:
        async with host.run_test(size=(120, 40)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_notes(screen, pilot)
            controller = screen._library_notes_sync_controller

            await _activate_root(
                screen,
                pilot,
                controller,
                display_name="Vault sync",
                folder=vaults[0],
                from_chooser=False,
            )
            # The receipt widget mounts before its line paints; on a loaded CI
            # runner the paint lands a frame later (PR #3234 UI Fast Lane (2)).
            await _wait_for_condition(
                pilot,
                lambda: "3 applied · listed under Receipts" in _painted_text(host),
                message=lambda: f"receipt line never painted: {controller.snapshot.status_line}",
            )
            assert database.count_notes() == 5

            # AC#1, first case: ‹ Notes, then Add from files… again.
            await _leave_with_notes_back(screen, pilot)
            await _reopen_add_from_files(screen, pilot)
            _assert_chooser(screen, host)

            # AC#4: the second folder is set up from that chooser, and its
            # receipt's bar routes to Manage sync folders.
            await _activate_root(
                screen,
                pilot,
                controller,
                display_name="Vault sync 2",
                folder=vaults[1],
                from_chooser=True,
            )
            # The receipt widget mounts before its line paints; on a loaded CI
            # runner the paint lands a frame later (PR #3234 UI Fast Lane (2)).
            await _wait_for_condition(
                pilot,
                lambda: "3 applied · listed under Receipts" in _painted_text(host),
                message=lambda: f"receipt line never painted: {controller.snapshot.status_line}",
            )
            assert database.count_notes() == 8
            manage = screen.query_one("#notes-sync-manage-roots", Button)
            assert str(manage.label) == "Manage sync folders"
            await _wait_for_condition(
                pilot,
                lambda: "Manage sync folders" in _painted_widget_text(host, manage),
                message="the receipt bar never painted Manage sync folders",
            )
            manage.press()
            await _wait_for_selector(screen, pilot, "#notes-sync-roots-back")
            assert screen._notes_state.view == "lasting_roots"
            await _wait_for_condition(
                pilot,
                lambda: (
                    len(controller.snapshot.roots) == 2
                    and "Vault sync 2" in _painted_text(host)
                ),
                message="Manage sync folders did not list both roots",
            )
    finally:
        await owner.shutdown()
        interop.close_all_user_connections()
        database.close_connection()


class _RootsRuntime:
    """One bound root whose manual check finds nothing and whose Recovery fails."""

    def __init__(self) -> None:
        self.root = NotesSyncRootRuntimeSnapshot("root-1", "up_to_date", "sync_now")
        self.calls: list[tuple[object, ...]] = []

    def snapshot(self) -> NotesSyncRuntimeSnapshot:
        return NotesSyncRuntimeSnapshot("active", "sync_now", (self.root,))

    async def request_sync_now(self, root_id: str) -> ReconciliationPlan:
        self.calls.append(("request_sync_now", root_id))
        return _empty_plan(root_id)

    async def conflict_labels(
        self, _root_id: str, _token: str
    ) -> tuple[RuntimeConflictLabel, ...]:
        return ()

    async def resolve_cleanup(self, root_id: str, operation_id: str) -> object:
        self.calls.append(("resolve_cleanup", root_id, operation_id))
        raise RuntimeError("vault unreadable")

    async def abandon_setup(self, root_id: str) -> None:
        self.calls.append(("abandon_setup", root_id))


async def test_add_from_files_reopens_on_the_chooser_after_nothing_to_review_and_after_a_failed_recovery() -> (
    None
):
    """AC#1 (second and third cases) and AC#5 in a mounted screen.

    verify/nl-v-n-17/13 and j6's captures 92/94: a check that ends in
    "Nothing to review." and a Recovery that fails both leave phase ``roots``,
    for which the add canvas painted a heading, the status line and nothing
    else -- no Import once, no Keep synced, no ‹ Notes.
    """

    runtime = _RootsRuntime()
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.notes_sync_runtime_owner = runtime
    host = _JourneyHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes(screen, pilot)
        controller = screen._library_notes_sync_controller

        # Nothing to review.
        screen.query_one("#library-notes-manage-sync-folders", Button).press()
        check = await _wait_for_selector(screen, pilot, "#notes-sync-root-check-0")
        check.press()
        await _wait_for_condition(
            pilot,
            lambda: controller.snapshot.status_line == "Nothing to review.",
            message=f"manual check did not settle: {controller.snapshot.status_line!r}",
        )
        assert controller.snapshot.phase == "roots"
        await _leave_with_notes_back(screen, pilot, "#notes-sync-roots-back")
        await _reopen_add_from_files(screen, pilot)
        _assert_chooser(screen, host)

        # Recovery failed.
        runtime.root = NotesSyncRootRuntimeSnapshot(
            "root-1", "failed", "resolve_cleanup", action_id="operation-1"
        )
        await _leave_with_notes_back(screen, pilot)
        screen.query_one("#library-notes-manage-sync-folders", Button).press()
        recover = await _wait_for_selector(screen, pilot, "#notes-sync-root-recover-0")
        recover.press()
        await _wait_for_condition(
            pilot,
            lambda: controller.snapshot.status_line.startswith("Recovery failed"),
            message=f"Recovery did not fail: {controller.snapshot.status_line!r}",
        )
        assert controller.snapshot.status_line == (
            "Recovery failed — RuntimeError. Next: Check changes."
        )
        assert controller.snapshot.phase == "roots"
        await _leave_with_notes_back(screen, pilot, "#notes-sync-roots-back")
        await _reopen_add_from_files(screen, pilot)
        _assert_chooser(screen, host)
        assert ("resolve_cleanup", "root-1", "operation-1") in runtime.calls


async def test_leaving_keep_synced_with_back_then_reopening_shows_the_chooser_not_the_form() -> (
    None
):
    """AC#1, fourth case (verify/nl-v-n-17/03): Keep synced, ‹ Notes, reopen."""

    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.notes_sync_runtime_owner = SimpleNamespace(
        snapshot=lambda: NotesSyncRuntimeSnapshot("active", "sync_now")
    )
    host = _JourneyHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes(screen, pilot)

        screen.query_one("#library-notes-add-from-files", Button).press()
        await _wait_for_selector(screen, pilot, "#notes-add-keep-synced")
        screen.query_one("#notes-add-keep-synced", Button).press()
        await _wait_for_selector(screen, pilot, "#notes-sync-display-name")
        assert screen._library_notes_sync_controller.snapshot.phase == "configure"

        await _leave_with_notes_back(screen, pilot)
        await _reopen_add_from_files(screen, pilot)
        _assert_chooser(screen, host)
        assert not screen.query("#notes-sync-display-name"), (
            "the configure form came back instead of the chooser"
        )


def _held_setup_runtime(
    started: asyncio.Event, release: asyncio.Event, abandoned: list[str]
):
    """A runtime whose setup check is held open until ``release`` is set."""

    class _HeldSetupRuntime:
        def snapshot(self) -> NotesSyncRuntimeSnapshot:
            return NotesSyncRuntimeSnapshot("active", "sync_now")

        async def review_setup(self, _setup: object) -> ReconciliationPlan:
            started.set()
            await release.wait()
            return ReconciliationPlan(
                root_id="provisional-root",
                observation_token="f" * 64,
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

        async def conflict_labels(
            self, _root_id: str, _token: str
        ) -> tuple[RuntimeConflictLabel, ...]:
            return ()

        async def abandon_setup(self, root_id: str) -> None:
            abandoned.append(root_id)

    return _HeldSetupRuntime()


async def _enter_setup_form(screen, pilot, controller, folder: Path) -> None:
    """Add from files… ▸ Keep a folder synced, with a name and a folder set."""

    screen.query_one("#library-notes-add-from-files", Button).press()
    await _wait_for_selector(screen, pilot, "#notes-add-keep-synced")
    screen.query_one("#notes-add-keep-synced", Button).press()
    await _wait_for_selector(screen, pilot, "#notes-sync-display-name")
    controller.set_setup("display_name", "Held")
    controller.set_setup("folder", str(folder))
    await pilot.pause()


async def test_a_running_check_is_shown_not_discarded(tmp_path: Path) -> None:
    """AC#2 and the sharpened AC#3: a check that is still running is kept.

    The check runs the production way: the real Check changes press, whose
    screen handler awaits the runtime inline, so the screen's message pump is
    held for the whole check. What the user can observe while it runs is
    exactly what is asserted: the phase is ``checking`` and the footer names
    ``wait current step`` -- nothing else can be pressed until the step
    settles (TASK-34000.72 tracks the blank work pane the held pump leaves
    behind). Two refusals are asserted as the contracts they are, not as user
    gestures: the screen guard's (``action_library_notes_escape`` is called
    directly -- a real Escape key mid-check would queue behind the held pump
    and land only after the step settled) and the controller's
    (``begin_add_from_files``). Once the check settles into its review and
    the user leaves it, the next open is the chooser, not that review.
    """

    started = asyncio.Event()
    release = asyncio.Event()
    abandoned: list[str] = []
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.notes_sync_runtime_owner = _held_setup_runtime(started, release, abandoned)
    host = _JourneyHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes(screen, pilot)
        controller = screen._library_notes_sync_controller
        await _enter_setup_form(screen, pilot, controller, tmp_path)

        screen.query_one("#notes-sync-check", Button).press()
        await asyncio.wait_for(started.wait(), timeout=5)
        assert controller.snapshot.phase == "checking"
        assert screen._library_notes_footer_shortcuts() == (("wait", "current step"),)

        # Screen-guard contract: the Escape action refuses to leave a running
        # step. Called directly -- the held pump would queue a real keypress.
        await screen.action_library_notes_escape()
        assert screen._notes_state.view == "lasting_add"
        assert controller.snapshot.phase == "checking"

        # Controller contract: a running step is kept and nothing is published.
        assert controller.begin_add_from_files() is False
        assert controller.snapshot.phase == "checking"
        assert abandoned == []

        release.set()
        activate = await _wait_for_selector(screen, pilot, "#notes-sync-activate")
        assert controller.snapshot.phase == "review"
        assert str(activate.label) == "Activate reviewed root"

        await _leave_with_notes_back(screen, pilot)
        assert abandoned == ["provisional-root"]
        await _reopen_add_from_files(screen, pilot)
        _assert_chooser(screen, host)
        assert not screen.query("#notes-sync-activate"), (
            "the settled review came back instead of the chooser"
        )


async def test_the_rail_route_out_of_a_setup_review_releases_the_provisional_root(
    tmp_path: Path,
) -> None:
    """TASK-34000.17 fix round 1 (I-1): the rail's Notes row leaves the lasting
    canvas without ‹ Notes or Esc, and the provisional setup review's runtime
    lease must be released on that route too -- otherwise it outlives the
    visible flow, and the chooser reset the next Add from files makes would
    orphan it (nothing left to release).
    """

    started = asyncio.Event()
    release = asyncio.Event()
    release.set()
    abandoned: list[str] = []
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.notes_sync_runtime_owner = _held_setup_runtime(started, release, abandoned)
    host = _JourneyHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes(screen, pilot)
        controller = screen._library_notes_sync_controller
        await _enter_setup_form(screen, pilot, controller, tmp_path)
        screen.query_one("#notes-sync-check", Button).press()
        await _wait_for_selector(screen, pilot, "#notes-sync-activate")
        assert controller.snapshot.review.root_id == "provisional-root"
        assert controller.snapshot.review.source == "setup"

        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._notes_state.view == "list",
            message="the rail's Notes row did not return to the list",
        )
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        await _wait_for_condition(
            pilot,
            lambda: abandoned == ["provisional-root"],
            timeout=3.0,
            message=f"the rail route left the provisional root held: {abandoned!r}",
        )

        await _reopen_add_from_files(screen, pilot)
        _assert_chooser(screen, host)
        await _leave_with_notes_back(screen, pilot)
        # The rail route released the root BEFORE the reopen (asserted above);
        # the handler's own belt-and-braces release may call the runtime for
        # the same root again, which the runtime treats as a no-op
        # (``NotesSyncRuntimeOwner.abandon_setup`` returns when the root is no
        # longer a held setup review). Only this root was ever named.
        assert set(abandoned) == {"provisional-root"}
