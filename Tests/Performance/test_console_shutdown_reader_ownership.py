"""Proposed exact callback shutdown regressions; install beneath Tests before running.

No production method, native guard or callback is replaced. An owned executor
initializer observes the original metadata/ledger code objects. Finally always
releases the original callback, joins it and restores the test loop executor.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import inspect
import sys
import threading
import time

import pytest
from textual.app import App

from Tests.Backup_Recovery.test_finite_db_counted_interval import live_operations
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root
from Tests.UI.test_console_character_context import _CharacterApp, _controller
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.UI.Console_Modules.character_context import (
    ConsoleCharacterContextController,
)
from tldw_chatbook.UI.Console_Modules.view_workers import capture_console_view_workers
from tldw_chatbook.Widgets.Console.console_character_context import (
    ConsoleCharacterContext,
)

pytestmark = pytest.mark.bootstrap_profile


async def _turns():
    # Deliver already queued cancellation. No native completion can race these
    # turns because its original callback remains held by the unreleased event.
    for _ in range(20):
        await asyncio.sleep(0)


async def _retired(database):
    deadline = time.monotonic() + 5
    while live_operations(database) or worker_leases(database):
        assert time.monotonic() < deadline, "released native callback did not retire"
        await asyncio.sleep(0.005)


def _held_original_reader(database, code, *, target=1, at_line=None):
    entered, release = threading.Event(), threading.Event()
    seen, guard = [], threading.Lock()

    def profile(frame, event, _arg):
        if frame.f_code is not code:
            return
        if at_line is None and event != "return":
            return
        if at_line is not None and (event != "line" or frame.f_lineno != at_line):
            return
        receiver = frame.f_locals.get("database", frame.f_locals.get("self"))
        if receiver is not database:
            return
        with guard:
            if len(seen) >= target:
                return
            seen.append(threading.current_thread())
            if len(seen) == target:
                entered.set()
        assert release.wait(10), "test must release its original finite callback"

    def trace(frame, event, arg):
        profile(frame, event, arg)
        # A global trace must return the local trace for this original frame;
        # otherwise Python never delivers the ledger's line events.
        return trace if frame.f_code is code else None

    def initializer():
        if at_line is None:
            sys.setprofile(profile)
        else:
            sys.settrace(trace)

    executor = ThreadPoolExecutor(
        max_workers=max(2, target + 1), initializer=initializer
    )
    return executor, entered, release, seen


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
async def test_character_capture_cancellation_waits_for_original_native_pair(tmp_path):
    database = CharactersRAGDB(tmp_path / "character.db", "shutdown-proof")
    controller = _controller(
        database_accessor=lambda: database,
        current_character_accessor=lambda: None,
        service_factory=CharacterConversationNavigationService,
    )
    executor, entered, release, seen = _held_original_reader(
        database,
        ConsoleCharacterContextController._read_database_scope_metadata.__code__,
    )
    loop = asyncio.get_running_loop()
    prior = loop._default_executor
    loop.set_default_executor(executor)
    pending = asyncio.create_task(controller._capture_scope())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert seen and live_operations(database) and worker_leases(database)
        pending.cancel()
        await _turns()
        assert (
            not pending.done()
        ), "Character awaiter retired before its actual DB callback"
        pending.cancel()
        await _turns()
        assert (
            not pending.done()
        ), "repeated cancellation abandoned Character native custody"
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await _retired(database)
        loop._default_executor = prior
        executor.shutdown(wait=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
async def test_initial_and_concurrent_character_reads_have_issued_host_workers(
    tmp_path,
):
    database = CharactersRAGDB(tmp_path / "mounted.db", "shutdown-proof")
    controller = _controller(
        database_accessor=lambda: database,
        current_character_accessor=lambda: None,
        service_factory=CharacterConversationNavigationService,
    )
    executor, entered, release, seen = _held_original_reader(
        database,
        ConsoleCharacterContextController._read_database_scope_metadata.__code__,
        target=2,
    )
    loop = asyncio.get_running_loop()
    prior = loop._default_executor
    loop.set_default_executor(executor)
    app = _CharacterApp(controller, controller.state)
    original_tasks = []
    try:
        async with app.run_test():
            widget = app.screen.query_one(ConsoleCharacterContext)
            original_tasks.append(widget._controller_task)
            widget._start(controller._capture_scope())
            original_tasks.append(widget._controller_task)
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                assert (
                    len(seen) == 2
                    and live_operations(database)
                    and worker_leases(database)
                )
                selected = capture_console_view_workers(app)[-1]
                actual = [
                    task for worker, node, task, work in selected if node is widget
                ]
                assert len(actual) == 2 and all(
                    not task.done() for task in actual
                ), "widget started real native reads outside the host worker drain"
                assert set(actual) == set(original_tasks)
            finally:
                release.set()
                await asyncio.gather(*original_tasks, return_exceptions=True)
                await _retired(database)
    finally:
        release.set()
        await _retired(database)
        loop._default_executor = prior
        executor.shutdown(wait=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
async def test_scheduler_worker_wait_cannot_outrun_original_ledger_offload(tmp_path):
    database = ScheduledTasksDB(tmp_path / "scheduled.db")
    scheduler = SchedulerLoop(database, {})
    # Stop at the original return statement inside the actual transaction:
    # UPDATE executed, but neither transaction nor callback has exited.
    original = ScheduledTasksDB.fail_interrupted_task_runs
    lines, first_line = inspect.getsourcelines(original)
    return_line = first_line + next(
        index
        for index, text in enumerate(lines)
        if "return int(cursor.rowcount)" in text
    )
    executor, entered, release, seen = _held_original_reader(
        database, original.__code__, at_line=return_line
    )
    loop = asyncio.get_running_loop()
    prior = loop._default_executor
    loop.set_default_executor(executor)
    app = App()
    try:
        async with app.run_test():
            worker = app.run_worker(
                scheduler.run(), group="scheduler", exit_on_error=False
            )
            waiter = None
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                assert seen and scheduler._maintenance_db_tasks
                assert live_operations(database) and worker_leases(database)
                scheduler.stop()
                worker.cancel()
                waiter = asyncio.create_task(worker.wait())
                await _turns()
                assert (
                    not waiter.done()
                ), "Worker.wait finished with the actual ledger callback live"
                worker.cancel()
                await _turns()
                assert (
                    not waiter.done()
                ), "repeated cancellation abandoned retained ledger work"
            finally:
                release.set()
                if waiter is not None:
                    await asyncio.gather(waiter, return_exceptions=True)
                else:
                    worker.cancel()
                    await asyncio.gather(worker.wait(), return_exceptions=True)
                scheduler._maintenance_close_admission()
                assert await scheduler._maintenance_drain(time.monotonic() + 5)
                assert not scheduler._maintenance_db_tasks
    finally:
        release.set()
        loop._default_executor = prior
        executor.shutdown(wait=True)
        database.close()
