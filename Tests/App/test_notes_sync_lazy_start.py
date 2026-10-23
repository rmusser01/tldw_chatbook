"""TASK-34000.71: the Library side's lazy, non-forced notes-sync start.

Boot's TASK-21247 gate constructs and starts the runtime only on
configuration evidence. With no root, the Library screen builds the owner
lazily and -- before this task -- never started it, so it stayed at its
constructor status "starting", which the setup gate refuses, and first-time
setup (the one force-start) sat behind the option that gate refused.
``ensure_notes_sync_runtime_started`` is the seam that breaks the deadlock:
one NON-forced start, through the boot path's own start-task bookkeeping.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import ScreenStackError

from tldw_chatbook.app import TldwCli
from tldw_chatbook.app_service_wiring import ServiceWiringMixin
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncRootRuntimeSnapshot,
    NotesSyncRuntimeSnapshot,
    build_notes_sync_runtime_owner,
)
from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
    LibraryNotesController,
)
from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
    LibraryNotesSyncController,
)

pytestmark = pytest.mark.unit


class _Owner:
    """An owner double that records how it was started."""

    def __init__(self, status: str = "starting", roots=()) -> None:
        self.force_calls: list[bool] = []
        self.status = status
        self.roots = tuple(roots)

    async def start(self, *, force: bool = False) -> None:
        self.force_calls.append(force)
        self.status = "not_configured"

    def snapshot(self) -> NotesSyncRuntimeSnapshot:
        next_action = {"starting": "wait", "not_configured": "none"}.get(
            self.status, "none"
        )
        return NotesSyncRuntimeSnapshot(self.status, next_action, self.roots)


class _LazyApp(TldwCli):
    """The slots the seam reads, with the owner built from a double.

    ``App.__init__`` is deliberately not run (the boot-gate tests'
    ``_RecordingOwnerApp`` precedent): only the owner slots, the lock and
    the start-task slot matter here.
    """

    def __init__(self, owner: object) -> None:  # pragma: no cover - slots only
        self._notes_sync_runtime_owner = None
        self._notes_sync_runtime_owner_lock = threading.Lock()
        self._notes_sync_runtime_start_task = None
        self._notes_sync_runtime_lazy_start_task = None
        self._owner = owner
        self.refreshed: list[str] = []
        self._screen_double: object | None = None

    def _construct_notes_sync_runtime_owner(self):  # type: ignore[override]
        return self._owner

    @property
    def screen(self):  # type: ignore[override]
        if self._screen_double is None:
            raise ScreenStackError("No screens on stack")
        return self._screen_double

    def call_after_refresh(self, callback, *args, **kwargs):  # type: ignore[override]
        callback(*args, **kwargs)
        return True


class _ExplodingOwnerApp(_LazyApp):
    def _construct_notes_sync_runtime_owner(self):  # type: ignore[override]
        raise AttributeError("construction exploded")


# --- the seam ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_call_schedules_one_non_forced_start_and_reuses_it() -> None:
    owner = _Owner()
    app = _LazyApp(owner)

    first = app.ensure_notes_sync_runtime_started()
    second = app.ensure_notes_sync_runtime_started()

    assert first is not None
    assert second is first, "a second call must reuse the recorded task"
    assert app._notes_sync_runtime_start_task is first
    # The boot path's name: shutdown and diagnostics treat both alike.
    assert first.get_name() == "start_notes_sync_runtime"
    await first
    assert owner.force_calls == [False], "the lazy start must not force"
    assert owner.snapshot().status == "not_configured"


@pytest.mark.asyncio
async def test_boot_scheduled_start_is_returned_not_doubled() -> None:
    """AC#5 seam: a configured profile's in-flight boot start is left alone."""

    root = NotesSyncRootRuntimeSnapshot("root-" + "a" * 40, "up_to_date", "none")
    owner = _Owner(status="starting", roots=(root,))
    app = _LazyApp(owner)
    boot_start = asyncio.create_task(asyncio.sleep(60), name="start_notes_sync_runtime")
    app._notes_sync_runtime_start_task = boot_start
    try:
        assert app.ensure_notes_sync_runtime_started() is boot_start
        assert owner.force_calls == [], "no second start over boot's own"
        assert owner.snapshot().status == "starting"
        # And what the Library reads while that start is in flight: not
        # offered as ready -- "starting" is not setup-ready.
        controller = LibraryNotesSyncController(
            runtime=owner,
            import_controller=SimpleNamespace(begin_selection=lambda: None),
        )
        assert controller.snapshot.lasting_available is False
    finally:
        boot_start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await boot_start


@pytest.mark.asyncio
async def test_build_failure_schedules_nothing() -> None:
    app = _ExplodingOwnerApp(_Owner())

    assert app.ensure_notes_sync_runtime_started() is None
    assert app._notes_sync_runtime_start_task is None
    assert isinstance(
        app.__dict__.get("_notes_sync_runtime_build_failure"), AttributeError
    )


@pytest.mark.asyncio
async def test_owner_double_without_start_is_left_alone() -> None:
    app = _LazyApp(SimpleNamespace(snapshot=lambda: None))

    assert app.ensure_notes_sync_runtime_started() is None
    assert app._notes_sync_runtime_start_task is None


def test_without_a_running_loop_nothing_is_scheduled() -> None:
    owner = _Owner()
    app = _LazyApp(owner)

    assert app.ensure_notes_sync_runtime_started() is None
    assert app._notes_sync_runtime_start_task is None
    # No coroutine was created and left un-awaited.
    assert owner.force_calls == []


@pytest.mark.asyncio
async def test_real_owner_defers_to_not_configured_without_creating_the_store(
    tmp_path: Path,
) -> None:
    """AC#3's mechanism at the seam: the deferred start opens no state DB."""

    state_path = tmp_path / "tldw_chatbook_notes_sync_state.db"
    owner = build_notes_sync_runtime_owner(
        # A deferred start never reaches the notes side; the builder only
        # checks the type.
        notes_scope_service=NotesScopeService(
            local_notes_service=SimpleNamespace(), server_service=None
        ),
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=state_path,
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
        start_evidence=lambda: False,
    )
    app = _LazyApp(owner)
    assert owner.snapshot().status == "starting"

    task = app.ensure_notes_sync_runtime_started()
    assert task is not None
    await task

    assert owner.snapshot().status == "not_configured"
    assert not state_path.exists(), "a deferred start must not create the store"
    # AC#6: the same shutdown a boot-started owner gets settles it cleanly.
    await owner.shutdown()
    assert task.done() and task.exception() is None


# --- the done-callback ---------------------------------------------------------


@pytest.mark.asyncio
async def test_settled_start_refreshes_the_library_screen() -> None:
    app = _LazyApp(_Owner())
    app._screen_double = SimpleNamespace(
        refresh_notes_sync_runtime=lambda: app.refreshed.append("refreshed")
    )

    task = app.ensure_notes_sync_runtime_started()
    assert task is not None
    await task
    await asyncio.sleep(0)  # let the done-callback run

    assert app.refreshed == ["refreshed"]


@pytest.mark.asyncio
async def test_done_callback_tolerates_an_app_with_no_screen_stack() -> None:
    """A harness mounts the Library over an un-run app; the callback must not raise."""

    app = _LazyApp(_Owner())
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, object]] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        task = app.ensure_notes_sync_runtime_started()
        assert task is not None
        await task
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous)

    assert unhandled == []
    assert app.refreshed == []


def test_entry_point_lives_on_service_wiring_mixin() -> None:
    """Like the safe reader: part of the app's composition surface."""
    assert "ensure_notes_sync_runtime_started" in vars(ServiceWiringMixin)


# --- fix round 1 --------------------------------------------------------------


@pytest.mark.asyncio
async def test_lazy_slot_marks_only_the_task_the_lazy_path_scheduled() -> None:
    """M-1: boot's task is returned but never recorded as the lazy one."""

    boot_app = _LazyApp(_Owner())
    boot_start = asyncio.create_task(asyncio.sleep(60), name="start_notes_sync_runtime")
    boot_app._notes_sync_runtime_start_task = boot_start
    try:
        assert boot_app.ensure_notes_sync_runtime_started() is boot_start
        assert getattr(boot_app, "_notes_sync_runtime_lazy_start_task", None) is None
    finally:
        boot_start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await boot_start

    lazy_app = _LazyApp(_Owner())
    task = lazy_app.ensure_notes_sync_runtime_started()
    assert task is not None
    assert lazy_app._notes_sync_runtime_lazy_start_task is task
    await task


class _FlakyOwner(_Owner):
    """Raises from ``start`` the first ``failures`` times, like a paused producer."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    async def start(self, *, force: bool = False) -> None:
        self.force_calls.append(force)
        if len(self.force_calls) <= self.failures:
            raise RuntimeError("runtime_producer_paused")
        self.status = "not_configured"


@pytest.mark.asyncio
async def test_a_start_that_raised_is_rearmed_on_the_next_call() -> None:
    """M-2: a refused start must not leave setup inert for the session."""

    owner = _FlakyOwner(failures=1)
    app = _LazyApp(owner)
    first = app.ensure_notes_sync_runtime_started()
    assert first is not None
    with pytest.raises(RuntimeError):
        await first
    assert owner.snapshot().status == "starting"

    second = app.ensure_notes_sync_runtime_started()
    assert second is not None and second is not first, "the failed task was kept"
    assert app._notes_sync_runtime_start_task is second
    assert app._notes_sync_runtime_lazy_start_task is second
    await second
    assert owner.snapshot().status == "not_configured"
    assert owner.force_calls == [False, False]
    # Settled: no further re-arm.
    assert app.ensure_notes_sync_runtime_started() is second


@pytest.mark.asyncio
async def test_a_cancelled_start_is_rearmed_while_the_owner_still_starts() -> None:
    owner = _Owner()
    app = _LazyApp(owner)
    cancelled = asyncio.create_task(asyncio.sleep(60), name="start_notes_sync_runtime")
    app._notes_sync_runtime_start_task = cancelled
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    task = app.ensure_notes_sync_runtime_started()
    assert task is not None and task is not cancelled
    await task
    assert owner.snapshot().status == "not_configured"


@pytest.mark.asyncio
async def test_rearm_is_bounded_to_one_new_start_per_call() -> None:
    """M-2: a permanently failing start retries once per call, never in a loop."""

    owner = _FlakyOwner(failures=10)
    app = _LazyApp(owner)
    first = app.ensure_notes_sync_runtime_started()
    assert first is not None
    # A second call while that attempt is still pending reuses it.
    assert app.ensure_notes_sync_runtime_started() is first
    with pytest.raises(RuntimeError):
        await first
    assert owner.force_calls == [False]

    second = app.ensure_notes_sync_runtime_started()
    assert second is not first
    assert app.ensure_notes_sync_runtime_started() is second
    with pytest.raises(RuntimeError):
        await second
    assert owner.force_calls == [False, False]


@pytest.mark.asyncio
async def test_a_failed_task_is_not_rearmed_once_the_owner_settled() -> None:
    """A task that raised AFTER the owner left "starting" is history, not a retry."""

    owner = _Owner(status="failed")
    app = _LazyApp(owner)

    async def _raise() -> None:
        raise RuntimeError("late")

    failed = asyncio.create_task(_raise(), name="start_notes_sync_runtime")
    app._notes_sync_runtime_start_task = failed
    with pytest.raises(RuntimeError):
        await failed

    assert app.ensure_notes_sync_runtime_started() is failed
    assert owner.force_calls == []


@pytest.mark.asyncio
async def test_a_double_with_a_sync_start_is_a_no_op() -> None:
    """M-3: a Mock-style synchronous ``start`` must not raise in the screen."""

    calls: list[bool] = []

    def sync_start(*, force: bool = False) -> None:
        calls.append(force)

    app = _LazyApp(SimpleNamespace(start=sync_start, snapshot=lambda: None))

    assert app.ensure_notes_sync_runtime_started() is None
    assert app._notes_sync_runtime_start_task is None
    assert calls == [False]


@pytest.mark.asyncio
async def test_an_awaitable_start_that_is_not_a_coroutine_is_awaited() -> None:
    """M-3: a Future-returning ``start`` (not a coroutine) still schedules."""

    loop = asyncio.get_running_loop()
    future: asyncio.Future[None] = loop.create_future()
    app = _LazyApp(SimpleNamespace(start=lambda **_kw: future, snapshot=lambda: None))

    task = app.ensure_notes_sync_runtime_started()
    assert task is not None and not task.done()
    future.set_result(None)
    await task
    assert task.exception() is None


# --- the Library side's settle helper (M-1, M-4) -------------------------------


def _settle(app_instance):
    """Run the opener's settle helper over a minimal host."""

    host = SimpleNamespace(app_instance=app_instance)
    return LibraryNotesController._settle_library_notes_sync_start(host)


@pytest.mark.asyncio
async def test_settle_waits_only_on_the_lazily_scheduled_task() -> None:
    """M-1 at the helper: boot's own pending task is not waited on."""

    pending = asyncio.create_task(asyncio.sleep(60))
    app = SimpleNamespace(
        ensure_notes_sync_runtime_started=lambda: pending,
        _notes_sync_runtime_lazy_start_task=None,
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        await _settle(app)
        assert loop.time() - started < 0.1, "the helper waited on boot's start"
        assert not pending.done()
    finally:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending


@pytest.mark.asyncio
async def test_settle_returns_when_the_lazy_start_is_cancelled() -> None:
    """M-4: a cancelled start is read as its status, not raised into the handler."""

    lazy = asyncio.create_task(asyncio.sleep(60))
    app = SimpleNamespace(
        ensure_notes_sync_runtime_started=lambda: lazy,
        _notes_sync_runtime_lazy_start_task=lazy,
    )
    asyncio.get_running_loop().call_later(0.05, lazy.cancel)

    await _settle(app)  # must not raise
    assert lazy.cancelled()


@pytest.mark.asyncio
async def test_settle_propagates_its_own_cancellation() -> None:
    """M-4: the handler being cancelled is not the start being cancelled."""

    lazy = asyncio.create_task(asyncio.sleep(60))
    app = SimpleNamespace(
        ensure_notes_sync_runtime_started=lambda: lazy,
        _notes_sync_runtime_lazy_start_task=lazy,
    )
    settling = asyncio.create_task(_settle(app))
    await asyncio.sleep(0.05)
    settling.cancel()
    with pytest.raises(asyncio.CancelledError):
        await settling
    assert not lazy.done(), "cancelling the handler must not cancel the start"
    lazy.cancel()
    with pytest.raises(asyncio.CancelledError):
        await lazy


# --- the screen's refresh against a suspended screen (M-5) ---------------------


def test_refresh_is_a_no_op_on_a_suspended_screen() -> None:
    """TASK-31521: a deferred refresh landing after suspend touches nothing."""

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    refreshed: list[str] = []
    controller = SimpleNamespace(refresh_roots=lambda: refreshed.append("roots"))
    suspended = SimpleNamespace(
        _library_screen_suspended=True,
        _library_notes_sync_controller=controller,
        is_mounted=False,
    )
    LibraryScreen.refresh_notes_sync_runtime(suspended)
    assert refreshed == []

    live = SimpleNamespace(
        _library_screen_suspended=False,
        _library_notes_sync_controller=controller,
        is_mounted=False,
    )
    LibraryScreen.refresh_notes_sync_runtime(live)
    assert refreshed == ["roots"]
