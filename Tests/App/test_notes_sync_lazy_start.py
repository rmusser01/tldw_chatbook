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
