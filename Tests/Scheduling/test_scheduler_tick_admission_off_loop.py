"""TASK-33620.15.24: the scheduler tick's storage admission never blocks the event loop.

Every 30 seconds the scheduler entered ``execution_scope`` (storage admission:
stats, realpaths, control-record reads) synchronously on the app's event loop,
twice per tick. Live, a tick that landed while the user typed or right after
Enter held keyboard input for up to ~60 ms.

Each test holds the REAL admission (the profile's ``execution_scope``, wrapped)
on a ``threading.Event`` and releases it from the event loop itself. A build
that resolves admission on the loop cannot do that: its hold can only time out
while the loop is blocked, and ``outcomes`` records the timeout. That is the
RED. Each test then pins one semantic the move must keep: no job before
admission resolves and its lease held while it runs, a refusal still refuses,
ticks never overlap or pile up, cancellation with admission in flight neither
hangs nor leaks a lease, and a pause or stop() that lands mid-admission skips
that tick.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from contextlib import contextmanager

import pytest

from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.scheduler import loop as loop_module
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop

# The real execution scope reads the guarded profile selected at collection.
pytestmark = pytest.mark.bootstrap_profile


class _HeldAdmission:
    """Wrap the real ``execution_scope``; hold armed entries until released.

    ``outcomes`` gets one entry per held admission: True when the event loop
    released it while it was held, False when the hold timed out instead --
    which is all a loop-blocking admission allows.
    """

    #: Bounded, so the RED (admission blocking the loop) fails instead of hanging.
    HOLD_SECONDS = 5.0

    def __init__(self, real, *, armed=True, refuse=False):
        self.real = real
        self.armed = armed
        self.refuse = refuse
        self.entered = threading.Event()
        self.release = threading.Event()
        self.outcomes: list[bool] = []
        self.enter_threads: list[threading.Thread] = []
        self.exit_threads: list[threading.Thread] = []
        self.entries = 0
        self.exits = 0
        #: Scopes closed only by garbage collection (GeneratorExit): a lease
        #: nobody released, at whatever moment and thread dropped it last.
        self.finalized = 0
        self.resolving = 0
        self.max_resolving = 0
        self._lock = threading.Lock()

    @contextmanager
    def __call__(self, owners, path=None, **kwargs):
        held = self.armed
        with self._lock:
            self.entries += 1
            self.resolving += 1
            self.max_resolving = max(self.max_resolving, self.resolving)
        self.enter_threads.append(threading.current_thread())
        try:
            if held:
                self.entered.set()
                self.outcomes.append(self.release.wait(self.HOLD_SECONDS))
            scope = self.real(owners, path, **kwargs)
            allowed = scope.__enter__()
        finally:
            with self._lock:
                self.resolving -= 1
        try:
            yield allowed and not (held and self.refuse)
        except GeneratorExit:
            self.finalized += 1
            raise
        finally:
            try:
                scope.__exit__(None, None, None)
            finally:
                self.exit_threads.append(threading.current_thread())
                with self._lock:
                    self.exits += 1


@pytest.fixture
def db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "scheduler.db")
    try:
        yield database
    finally:
        database.close()


def _due_reminder(database, title="due"):
    return database.create_reminder_task(
        owner_id="local",
        title=title,
        schedule_kind="one_time",
        next_run_at="2026-01-01T00:00:00+00:00",
    )


def _hold(monkeypatch, **kwargs) -> _HeldAdmission:
    held = _HeldAdmission(loop_module.execution_scope, **kwargs)
    monkeypatch.setattr(loop_module, "execution_scope", held)
    return held


def _scope_leases() -> set:
    """Live storage leases not attached to a SQLite connection (scope leases)."""
    with storage_admission._lock:
        return {
            lease
            for lease in storage_admission._live_leases
            if lease.resource_policy is None
        }


async def _observe_hold(held: _HeldAdmission) -> None:
    """Wait, on the event loop, until an armed admission is being held."""
    deadline = time.monotonic() + held.HOLD_SECONDS * 2
    while not held.entered.is_set():
        assert time.monotonic() < deadline, "the held admission was never entered"
        await asyncio.sleep(0.005)


_BLOCKED = (
    "storage admission resolved on the event loop: the loop could not run "
    "until the held admission timed out"
)


@pytest.mark.asyncio
async def test_tick_admission_resolves_off_the_loop_before_any_dispatch(
    db, monkeypatch
):
    task_id = _due_reminder(db)
    ran = []
    held_while_running = []
    leases_before = _scope_leases()

    async def handler(row):
        ran.append(row["id"])
        # The lease is the point of admission: another process's native
        # maintenance waits on it, so it must be HELD while the job runs --
        # tick()'s scope and dispatch_reminder()'s, not merely resolved.
        held_while_running.append(
            (held.entries - held.exits, len(_scope_leases() - leases_before))
        )

    held = _hold(monkeypatch)
    scheduler = SchedulerLoop(db, {"reminder": handler}, handler_timeout_seconds=0)
    scheduler.queue.load()
    loop_thread = threading.current_thread()
    tick = asyncio.create_task(scheduler.tick())
    try:
        await _observe_hold(held)
        assert held.outcomes == [], _BLOCKED
        await asyncio.sleep(0.05)  # the loop keeps running while admission is held
        assert ran == [], "a job must never run before its admission resolves"
        assert not tick.done()
        held.release.set()
        await asyncio.wait_for(tick, 5)
    finally:
        held.release.set()
        await asyncio.gather(tick, return_exceptions=True)

    assert held.outcomes and all(held.outcomes), _BLOCKED
    assert ran == [task_id]
    (open_scopes, live_leases), = held_while_running
    assert open_scopes >= 2 and live_leases >= 1, (
        "the job ran without its storage admission held: "
        f"{open_scopes} open scopes, {live_leases} live scope leases"
    )
    assert db.get_reminder_task(task_id)["last_status"] == "completed"
    # Entry AND release stayed off the loop: no admission syscall ran on it.
    assert loop_thread not in held.enter_threads
    assert loop_thread not in held.exit_threads
    gc.collect()
    assert held.finalized == 0, "an admission was never released, only garbage-collected"
    assert held.exits == held.entries


@pytest.mark.asyncio
async def test_tick_refused_off_the_loop_still_refuses_and_stops_the_scheduler(
    db, monkeypatch
):
    task_id = _due_reminder(db)
    ran = []

    async def handler(row):
        ran.append(row["id"])

    # The startup scope is admitted; the first tick's admission is held, then refused.
    held = _hold(monkeypatch, armed=False, refuse=True)
    scheduler = SchedulerLoop(db, {"reminder": handler}, poll_interval=0.001)
    scheduler.report_configuration = lambda: setattr(held, "armed", True)
    runner = asyncio.create_task(scheduler.run())
    try:
        await _observe_hold(held)
        assert held.outcomes == [], _BLOCKED
        held.release.set()
        # A refused tick admission ends the loop, exactly as before.
        await asyncio.wait_for(runner, 5)
    finally:
        held.release.set()
        scheduler.stop()
        await asyncio.gather(runner, return_exceptions=True)

    assert held.outcomes == [True], _BLOCKED
    assert ran == [], "a refused admission must not dispatch"
    assert db.get_reminder_task(task_id)["last_status"] is None
    assert len(scheduler.queue) == 1, "the refused tick consumed nothing"
    assert not scheduler.running
    gc.collect()
    assert held.finalized == 0, "an admission was never released, only garbage-collected"
    assert held.exits == held.entries


@pytest.mark.asyncio
async def test_slow_tick_admission_never_overlaps_or_piles_up_ticks(
    db, monkeypatch
):
    held = _hold(monkeypatch, armed=False)
    scheduler = SchedulerLoop(db, {}, poll_interval=0.001)
    scheduler.report_configuration = lambda: setattr(held, "armed", True)
    dispatched = []

    async def dispatch_due(now):
        dispatched.append(now)

    scheduler._dispatch_due = dispatch_due
    runner = asyncio.create_task(scheduler.run())
    try:
        await _observe_hold(held)
        assert held.outcomes == [], _BLOCKED
        entries_while_held = held.entries
        # Two hundred poll intervals pass on a free loop while admission is slow.
        await asyncio.sleep(0.2)
        assert held.entries == entries_while_held, "a tick started behind a slow one"
        assert dispatched == [], "no tick dispatches before its admission resolves"
        held.armed = False
        held.release.set()
        deadline = time.monotonic() + 5
        while len(dispatched) < 3:
            assert time.monotonic() < deadline, "ticks did not resume after admission"
            await asyncio.sleep(0.005)
    finally:
        held.release.set()
        scheduler.stop()
        await asyncio.wait_for(runner, 5)

    assert held.outcomes == [True], _BLOCKED
    # Ticks stayed sequential: no two admissions were ever resolving at once.
    assert held.max_resolving == 1
    gc.collect()
    assert held.finalized == 0, "an admission was never released, only garbage-collected"
    assert held.exits == held.entries


@pytest.mark.asyncio
async def test_cancelled_scheduler_with_admission_in_flight_releases_its_lease(
    db, monkeypatch
):
    held = _hold(monkeypatch, armed=False)
    scheduler = SchedulerLoop(db, {}, poll_interval=0.001)
    scheduler.report_configuration = lambda: setattr(held, "armed", True)
    dispatched = []

    async def dispatch_due(now):
        dispatched.append(now)

    scheduler._dispatch_due = dispatch_due
    leases_before = _scope_leases()
    runner = asyncio.create_task(scheduler.run())
    try:
        await _observe_hold(held)
        assert held.outcomes == [], _BLOCKED
        # Shutdown while the admission is still resolving: the loop exits at once.
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(runner, 2)
        assert held.outcomes == [], "cancellation waited for the held admission"
        # The in-flight admission is still owned: maintenance cannot drain under it.
        scheduler._maintenance_close_admission()
        assert not await scheduler._maintenance_drain(time.monotonic())
        held.release.set()
        assert await scheduler._maintenance_drain(time.monotonic() + 5)
    finally:
        held.release.set()
        scheduler.stop()
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)

    assert held.outcomes == [True], _BLOCKED
    assert dispatched == [], "a cancelled tick never dispatches"
    # The admission that resolved after its waiter left closed its own lease.
    gc.collect()
    assert held.finalized == 0, "an admission was never released, only garbage-collected"
    assert held.exits == held.entries
    assert _scope_leases() <= leases_before, "a scope lease outlived its admission"


def test_scope_lease_acquired_on_one_thread_closes_on_another(db):
    """Guard: the off-loop seam relies on storage leases having no thread affinity.

    ``_OffLoopExecutionScope`` enters admission on one ``to_thread`` worker and
    releases it on whichever worker the release hop lands on. If a storage lease
    ever gains thread affinity (a close that refuses, or silently skips, a
    foreign thread), that seam must change -- this fails first.
    """
    leases_before = _scope_leases()
    scope = loop_module._OffLoopExecutionScope(("db.scheduled_tasks",), db.db_path)
    outcome = {}

    def enter():
        outcome["allowed"] = scope.enter()
        outcome["live"] = len(_scope_leases() - leases_before)

    def release():
        try:
            scope.release()
        except BaseException as exc:  # noqa: BLE001 -- reported below
            outcome["error"] = exc

    for target, name in ((enter, "admission-enter"), (release, "admission-release")):
        worker = threading.Thread(target=target, name=name)
        worker.start()
        worker.join(5)
        assert not worker.is_alive(), f"{name} did not finish"

    assert outcome["allowed"] is True, "the real admission was refused"
    assert outcome["live"] >= 1, "admission held no storage lease"
    assert "error" not in outcome, f"cross-thread release raised: {outcome.get('error')!r}"
    assert _scope_leases() <= leases_before, "a lease closed from another thread stayed live"


# --- A pause or stop() that lands while the tick's admission is resolving ---
#
# Synchronous admission made "fence open" and "storage admitted" one instant.
# Off the loop they are two, and a capture pause (or stop()) can land between
# them (review of TASK-33620.15.24: median ~10 ms live, max 129 ms). The drain
# would still wait for such a tick, but a tick the fence closed on before it was
# admitted has not begun: it must not dispatch, and the loop must not end.


async def _run_until_tick_admission_held(db, monkeypatch):
    held = _hold(monkeypatch, armed=False)
    scheduler = SchedulerLoop(db, {}, poll_interval=0.001)
    # Startup is admitted at once; the first tick's admission is held.
    scheduler.report_configuration = lambda: setattr(held, "armed", True)
    dispatched = []

    async def dispatch_due(now):
        dispatched.append(now)

    scheduler._dispatch_due = dispatch_due
    runner = asyncio.create_task(scheduler.run())
    await _observe_hold(held)
    assert held.outcomes == [], _BLOCKED
    held.armed = False
    return held, scheduler, dispatched, runner


@pytest.mark.asyncio
async def test_pause_landing_while_tick_admission_resolves_skips_that_tick(
    db, monkeypatch
):
    held, scheduler, dispatched, runner = await _run_until_tick_admission_held(
        db, monkeypatch
    )
    try:
        scheduler._maintenance_close_admission()
        held.release.set()
        # The drain completes without the paused tick ever dispatching.
        assert await scheduler._maintenance_drain(time.monotonic() + 5)
        await asyncio.sleep(0.05)
        assert dispatched == [], "a tick dispatched after the fence closed on it"
        assert not runner.done(), "a pause must hold the scheduler, not end it"
        assert scheduler.running
        scheduler._maintenance_resume()
        deadline = time.monotonic() + 5
        while not dispatched:
            assert time.monotonic() < deadline, "ticks did not resume after the pause"
            await asyncio.sleep(0.005)
    finally:
        held.release.set()
        scheduler.stop()
        await asyncio.wait_for(runner, 5)

    assert held.outcomes == [True], _BLOCKED
    gc.collect()
    assert held.finalized == 0, "an admission was never released, only garbage-collected"
    assert held.exits == held.entries


@pytest.mark.asyncio
async def test_stop_landing_while_tick_admission_resolves_skips_that_tick(
    db, monkeypatch
):
    held, scheduler, dispatched, runner = await _run_until_tick_admission_held(
        db, monkeypatch
    )
    try:
        scheduler.stop()
        held.release.set()
        await asyncio.wait_for(runner, 5)
    finally:
        held.release.set()
        scheduler.stop()
        await asyncio.gather(runner, return_exceptions=True)

    assert held.outcomes == [True], _BLOCKED
    assert dispatched == [], "a tick dispatched after stop() landed before its admission"
    assert not scheduler.running
    gc.collect()
    assert held.finalized == 0, "an admission was never released, only garbage-collected"
    assert held.exits == held.entries


# --- The tick's emergency-stop check: its default path is a guarded config read ---
#
# Live (2026-10-27, branch arm), the first tick after a send spent 98 ms on the
# loop in `default_emergency_stop_path()`: `get_user_data_dir()` is a guarded
# config read that enters storage admission for the config participant. Only the
# stop-state file read had been moved off the loop (TASK-31507).


def _hold_stop_path_lookup(monkeypatch, tmp_path, *, refuse=False):
    from tldw_chatbook import emergency_stop

    held = _HeldAdmission(None)

    def lookup():
        held.enter_threads.append(threading.current_thread())
        held.entered.set()
        held.outcomes.append(held.release.wait(held.HOLD_SECONDS))
        if refuse:
            raise RuntimeError("config read refused")
        return tmp_path / "emergency_stop.json"

    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", lookup)
    return held


@pytest.mark.asyncio
async def test_tick_emergency_stop_path_lookup_runs_off_the_loop(
    db, monkeypatch, tmp_path
):
    task_id = _due_reminder(db)
    ran = []

    async def handler(row):
        ran.append(row["id"])

    held = _hold_stop_path_lookup(monkeypatch, tmp_path)
    scheduler = SchedulerLoop(db, {"reminder": handler}, handler_timeout_seconds=0)
    scheduler.queue.load()
    loop_thread = threading.current_thread()
    tick = asyncio.create_task(scheduler.tick())
    try:
        await _observe_hold(held)
        assert held.outcomes == [], (
            "the emergency-stop path lookup (a guarded config read) ran on the "
            "event loop: the loop could not run until it timed out"
        )
        assert ran == [], "nothing dispatches before the stop check resolves"
        held.release.set()
        await asyncio.wait_for(tick, 5)
    finally:
        held.release.set()
        await asyncio.gather(tick, return_exceptions=True)

    assert held.outcomes == [True]
    assert loop_thread not in held.enter_threads
    assert ran == [task_id]


@pytest.mark.asyncio
async def test_refused_emergency_stop_path_lookup_still_holds_dispatch(
    db, monkeypatch, tmp_path
):
    task_id = _due_reminder(db)
    ran = []

    async def handler(row):
        ran.append(row["id"])

    held = _hold_stop_path_lookup(monkeypatch, tmp_path, refuse=True)
    scheduler = SchedulerLoop(db, {"reminder": handler}, handler_timeout_seconds=0)
    scheduler.queue.load()
    tick = asyncio.create_task(scheduler.tick())
    try:
        await _observe_hold(held)
        assert held.outcomes == [], (
            "the emergency-stop path lookup (a guarded config read) ran on the "
            "event loop: the loop could not run until it timed out"
        )
        held.release.set()
        await asyncio.wait_for(tick, 5)
    finally:
        held.release.set()
        await asyncio.gather(tick, return_exceptions=True)

    assert held.outcomes == [True]
    # Doubt still holds work: an unreadable stop state reads as stopped (26004).
    assert ran == []
    assert len(scheduler.queue) == 1, "the held occurrence stays queued"
    assert db.get_reminder_task(task_id)["last_status"] is None
