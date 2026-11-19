"""TASK-34000.71 AC#4 through the real screen (fix round 1, I-1 and M-1).

Outside the PR lane census on purpose: three real-app boots of ~6-8 s each
would push the lane file (`test_library_notes_keep_synced_without_a_root.py`,
one boot) past its budget. These arms pin the bounded settle wait and the
repaint, which the lane's no-root walk cannot see (its start settles in
milliseconds):

- (a) a HELD lazily scheduled start: the chooser paints the honest not-ready
  option only after the bounded wait, the start is NOT cancelled by the
  timeout, and the option repaints to available once the start settles (the
  done-callback path);
- (b) a start released inside the bound: the first paint is already
  available;
- (c) M-1: a start BOOT scheduled (configured evidence on disk) is not
  waited on -- the done-callback repaints it, so blocking the screen pump
  for the bound buys nothing. The press paints at once.

Mutation check (recorded in the task notes): with the opener's
``await self._settle_library_notes_sync_start()`` replaced by ``pass``, (a)
fails on its elapsed-time floor and (b) on its first paint.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.App.test_notes_sync_boot_gate import _make_state_db
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_notes_files_sync_journey import (
    _isolate_actor_pack_recovery,  # noqa: F401 - autouse fixture, re-exported
)
from Tests.UI.test_library_shell import (
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeSnapshot
from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
    _NOTES_SYNC_START_SETTLE_SECONDS,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: The press must have waited for (about) the whole bound in arm (a); a
#: paint that comes back sooner than this means the wait is gone.
_WAIT_FLOOR_SECONDS = _NOTES_SYNC_START_SETTLE_SECONDS * 0.8
#: And in arm (c) it must NOT have waited: well under the bound.
_NO_WAIT_CEILING_SECONDS = _NOTES_SYNC_START_SETTLE_SECONDS * 0.6


class _HeldStartRuntime:
    """An owner double whose non-forced start blocks until released."""

    def __init__(self) -> None:
        self.status = "starting"
        self.release = asyncio.Event()
        self.start_calls: list[bool] = []
        self.shutdowns = 0

    def snapshot(self) -> NotesSyncRuntimeSnapshot:
        return NotesSyncRuntimeSnapshot(
            self.status, "wait" if self.status == "starting" else "none"
        )

    async def start(self, *, force: bool = False) -> None:
        self.start_calls.append(force)
        await self.release.wait()
        self.status = "not_configured"

    async def shutdown(self) -> None:
        self.shutdowns += 1
        self.release.set()


def _label(screen) -> str:
    return str(screen.query_one("#notes-add-keep-synced", Button).label)


def _app(runtime: _HeldStartRuntime):
    app = _build_test_app(configured_default="library")
    app.app_config["_first_run"] = False
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
    _seed_conversations(app, [])
    app.notes_sync_runtime_owner = runtime
    return app


async def _boot_to_notes(app, pilot):
    await _wait_for_condition(
        pilot,
        lambda: isinstance(app.screen, LibraryScreen),
        message="never routed to Library",
        timeout=30.0,
    )
    screen = app.screen
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
    return screen


async def _press_add_from_files(screen, pilot) -> tuple[str, float]:
    """Press Add from files…; return the first Keep-synced label and the wait."""

    loop = asyncio.get_running_loop()
    started = loop.time()
    screen.query_one("#library-notes-add-from-files", Button).press()
    await _wait_for_selector(screen, pilot, "#notes-add-keep-synced")
    return _label(screen), loop.time() - started


async def test_held_lazy_start_paints_honest_option_then_repaints_when_it_settles() -> (
    None
):
    """Arm (a): bounded wait, start left running, repaint on settle."""

    runtime = _HeldStartRuntime()
    app = _app(runtime)
    async with app.run_test(size=(160, 45)) as pilot:
        screen = await _boot_to_notes(app, pilot)
        task = app._notes_sync_runtime_start_task
        assert task is not None and not task.done(), task
        assert app._notes_sync_runtime_lazy_start_task is task

        first, elapsed = await _press_add_from_files(screen, pilot)
        assert first == "○ Keep a folder synced", first
        assert elapsed >= _WAIT_FLOOR_SECONDS, (
            f"the chooser painted after {elapsed:.3f}s: the bounded wait is gone"
        )
        assert not task.done(), "the bounded wait must not cancel the start"
        assert runtime.start_calls == [False]

        runtime.release.set()
        await _wait_for_condition(
            pilot,
            lambda: _label(screen) == "Keep a folder synced",
            message=f"chooser never repainted: {_label(screen)!r}",
        )
        assert task.done() and not task.cancelled()
        assert screen._library_notes_sync_controller.snapshot.lasting_available
    assert runtime.start_calls == [False]
    assert runtime.shutdowns == 1


async def test_start_released_inside_the_bound_paints_available_first() -> None:
    """Arm (b): a start that settles inside the bound never paints inert."""

    runtime = _HeldStartRuntime()
    app = _app(runtime)
    async with app.run_test(size=(160, 45)) as pilot:
        screen = await _boot_to_notes(app, pilot)
        task = app._notes_sync_runtime_start_task
        assert task is not None and not task.done(), task
        asyncio.get_running_loop().call_later(0.15, runtime.release.set)

        first, _elapsed = await _press_add_from_files(screen, pilot)
        assert first == "Keep a folder synced", first
        assert task.done() and not task.cancelled()
    assert runtime.start_calls == [False]


async def test_boot_scheduled_start_is_not_waited_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arm (c), M-1: a configured profile's in-flight BOOT start is left alone.

    On-disk evidence (one configured root) makes ``on_mount`` schedule the
    start; the Library side must neither wait on it (the press paints at
    once, honestly not ready) nor double it, and the done-callback still
    repaints the chooser when it settles.
    """

    state_path = tmp_path / "tldw_chatbook_notes_sync_state.db"
    _make_state_db(state_path, with_root=True)
    monkeypatch.setattr(
        "tldw_chatbook.app_service_wiring.get_notes_sync_state_db_path",
        lambda: state_path,
    )
    runtime = _HeldStartRuntime()
    app = _app(runtime)
    async with app.run_test(size=(160, 45) ) as pilot:
        screen = await _boot_to_notes(app, pilot)
        task = app._notes_sync_runtime_start_task
        assert task is not None and not task.done(), task
        assert task.get_name() == "start_notes_sync_runtime"
        assert app._notes_sync_runtime_lazy_start_task is None, (
            "the Library must not have scheduled a start of its own"
        )

        first, elapsed = await _press_add_from_files(screen, pilot)
        assert first == "○ Keep a folder synced", first
        assert elapsed < _NO_WAIT_CEILING_SECONDS, (
            f"the press waited {elapsed:.3f}s on boot's own start"
        )
        assert not task.done()
        assert runtime.start_calls == [False], "boot's start was doubled"

        runtime.release.set()
        await _wait_for_condition(
            pilot,
            lambda: _label(screen) == "Keep a folder synced",
            message=f"chooser never repainted: {_label(screen)!r}",
        )
    assert runtime.start_calls == [False]
