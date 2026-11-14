"""PYTEST_DONT_REWRITE: original deferred Collections callback and native lifetime."""

import asyncio
import os
import sqlite3
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.thread import _WorkItem
from pathlib import Path

import pytest

from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["path", "legacy"])
@private_profile_test
async def test_original_deferred_collections_callback_keeps_loop_responsive(
    request, tmp_path, route
):
    await _exercise_deferred_collections_callback(route)


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["shutdown", "borrowed", "first_use", "all_created", "scope", "policy", "source"])
@private_profile_test
async def test_original_collections_setup_preserves_native_owner(
    request, tmp_path, control
):
    await _exercise_deferred_collections_callback("legacy", control=control)


async def _exercise_deferred_collections_callback(route, *, control=None):
    from tldw_chatbook import app as app_source
    from tldw_chatbook import app_service_wiring as wiring
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.Library.collections_legacy_recovery import (
        LegacyCollectionsRecovery,
    )

    creators = OriginalPrivateAppCreators(
        app_source, Path(os.environ["XDG_DATA_HOME"]).parent
    )
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    runtime = app.console_runtime
    database = app.local_library_collections_db
    assert type(database) is LibraryCollectionsDB and not database.is_memory_db
    assert (
        type(app.collections_capture_scope_service)
        is wiring._DeferredCollectionsCaptureScope
    )
    creator = database._thread_local.conn
    original = wiring.ServiceWiringMixin._deferred_wire_collections_capture_services
    callback = (
        config.get_library_collections_db_path
        if route == "path"
        else LegacyCollectionsRecovery.list_collections
    )
    pin = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for function in (original, callback, _WorkItem.run):
        pin._pin(function)
    pin.slots.extend(
        (
            (
                wiring.ServiceWiringMixin,
                "_deferred_wire_collections_capture_services",
                original,
            ),
            (
                config if route == "path" else LegacyCollectionsRecovery,
                "get_library_collections_db_path"
                if route == "path"
                else "list_collections",
                callback,
            ),
        )
    )
    entered, release, progress = threading.Event(), threading.Event(), threading.Event()
    loop, main = asyncio.get_running_loop(), threading.current_thread()
    held, errors, declined = {}, [], []
    controlled = threading.Event()
    control_task = None
    shutdown = None
    borrowed = None
    _unused_original_scope = app.collections_capture_scope_service
    original_policy = app.runtime_policy
    original_keywords = callback.__kwdefaults__
    before_tasks = asyncio.all_tasks()
    if control == "borrowed":
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        borrowed = await asyncio.to_thread(database._held_connection)
    tool = None

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def hold(code, offset, value):
        if code is not callback.__code__ or held:
            return
        frame = ancestor = None
        try:
            frame = sys._getframe(1)
            if (
                route == "legacy"
                and frame.f_locals.get("self")._database is not database
            ):
                return
            assert frame.f_code is callback.__code__
            assert frame.f_globals is callback.__globals__
            thread = threading.current_thread()
            held["thread"] = thread
            if thread is not main:
                ancestor = frame.f_back
                while (
                    ancestor is not None
                    and ancestor.f_code is not _WorkItem.run.__code__
                ):
                    ancestor = ancestor.f_back
                assert ancestor is not None
                item = ancestor.f_locals["self"]
                assert type(item) is _WorkItem and type(item.future) is Future
                assert item.future.running() and not item.future.done()
                held["future"] = item.future
            if route == "legacy":
                connection = database._thread_local.conn
                participant = database._maintenance_participant
                with storage._lock:
                    lease = participant.connections[connection]
                    assert lease in storage._live_leases
                    assert lease.resource_thread is thread and not closed(connection)
                held.update(connection=connection, participant=participant, lease=lease)
            entered.set()
            assert release.wait(10), "Collections callback was never released"
        except BaseException as error:  # noqa: BLE001 - record observer failures without skipping physical test cleanup.
            errors.append(type(error).__name__)
            entered.set()
        finally:
            frame = ancestor = None

    import ast
    import inspect
    import textwrap

    refusal_lines = {}
    refusal_records = {}
    diagnostic_functions = tuple(
        function for name in (
            "_require_collections_setup_current",
            "_build_deferred_collections_capture_parts",
        ) if (function := vars(wiring).get(name)) is not None
    )  # The baseline has no optional initializer refusal callbacks yet.
    for function in diagnostic_functions:
        lines, first = inspect.getsourcelines(function)
        defining = ast.parse(textwrap.dedent("".join(lines)))
        locations = set()
        for node in ast.walk(defining):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            target = node.exc.func
            if (
                isinstance(target, ast.Name)
                and target.id == "_CollectionsSetupObsolete"
            ) or (
                isinstance(target, ast.Attribute)
                and target.attr == "_CollectionsSetupObsolete"
                and isinstance(target.value, ast.Name)
                and target.value.id == "_source_wiring"
            ):
                locations.add(first + node.lineno - 1)
        refusal_lines[function.__code__] = locations
        refusal_records[function.__code__] = (function, function.__globals__)

    def observe_decline(code, line):
        if line not in refusal_lines[code]:
            return
        frame = None
        try:
            frame = sys._getframe(1)
            function, namespace = refusal_records[code]
            assert function.__code__ is code and function.__globals__ is namespace
            assert frame.f_code is code and frame.f_globals is namespace
            declined.append((code.co_name, line))
        except BaseException as error:  # noqa: BLE001 - passive observer records failure after original native drain.
            errors.append("refusal_observer:" + type(error).__name__)
        finally:
            frame = None

    async def cancel_during_original_callback():
        nonlocal shutdown
        try:
            initializer = app._collections_capture_initializer_task
            if control == "scope":
                app.collections_capture_scope_service = wiring._DeferredCollectionsCaptureScope(app)
                return
            if control == "policy":
                from types import SimpleNamespace
                app.runtime_policy = SimpleNamespace(state=original_policy.state)
                return
            if control == "source":
                callback.__kwdefaults__ = {**original_keywords, "size": 21}
                return
            if control == "first_use":
                app.ensure_collections_capture_services()
                held["first_use_repository"] = app.collections_capture_repository
                held["first_use_reconciliation_scheduled"] = any(
                    task.get_coro().cr_code
                    is app_source.TldwCli._reconcile_collections_capture_startup.__code__
                    for task in app._deferred_startup_tasks
                )
                return
            shutdown = asyncio.create_task(app._shutdown_collections_capture_runtime())
            for _ in range(3):
                await asyncio.sleep(0)
            assert not initializer.done() and not shutdown.done()
            assert held["future"].running() and not closed(held["connection"])
            if control == "all_created":
                for task in asyncio.all_tasks() - before_tasks - {asyncio.current_task()}:
                    task.cancel()
            for _ in range(2):
                shutdown.cancel()
                await asyncio.sleep(0)
                assert not shutdown.done() and not initializer.done()
            held["shutdown_retained_exact_native_callback"] = True
        finally:
            controlled.set()

    def start_control():
        nonlocal control_task
        control_task = asyncio.create_task(cancel_during_original_callback())

    def coordinate():
        if not entered.wait(10):
            errors.append("original_callback_never_entered")
            release.set()
            return
        loop.call_soon_threadsafe(progress.set)
        held["loop_progress_while_original_callback_held"] = progress.wait(0.35)
        if control in {"shutdown", "first_use", "all_created", "scope", "policy", "source"}:
            loop.call_soon_threadsafe(start_control)
            if not controlled.wait(10):
                errors.append("shutdown_control_never_completed")
        release.set()

    supervisor = threading.Thread(target=coordinate, name="collections-test-release")
    try:
        pin._pin(hold)
        pin._pin(coordinate)
        pin._pin(closed)
        for candidate in range(5, 0, -1):
            if candidate == sys.monitoring.DEBUGGER_ID:
                continue
            try:
                sys.monitoring.use_tool_id(candidate, "collections-deferred-loop")
            except ValueError:
                continue
            tool = candidate
            break
        assert tool is not None
        assert sys.monitoring.get_events(tool) == 0
        assert (
            sys.monitoring.register_callback(
                tool, sys.monitoring.events.PY_RETURN, hold
            )
            is None
        )
        assert sys.monitoring.get_local_events(tool, callback.__code__) == 0
        sys.monitoring.set_local_events(
            tool, callback.__code__, sys.monitoring.events.PY_RETURN
        )
        assert (
            sys.monitoring.register_callback(
                tool, sys.monitoring.events.LINE, observe_decline
            )
            is None
        )
        diagnostic_codes = tuple(function.__code__ for function in diagnostic_functions)
        for code in diagnostic_codes:
            assert sys.monitoring.get_local_events(tool, code) == 0
            sys.monitoring.set_local_events(tool, code, sys.monitoring.events.LINE)
        supervisor.start()
        original(app)
        while app._deferred_startup_tasks:
            results = await asyncio.gather(
                *tuple(app._deferred_startup_tasks), return_exceptions=True
            )
            assert all(
                not isinstance(value, BaseException)
                or (control in {"shutdown", "all_created"} and isinstance(value, asyncio.CancelledError))
                for value in results
            ), results
        await asyncio.to_thread(supervisor.join, 10)
        assert not supervisor.is_alive() and not errors, errors
        assert held and not closed(creator)
        if control in {"shutdown", "all_created"}:
            assert control_task is not None and shutdown is not None
            await control_task
            with pytest.raises(asyncio.CancelledError):
                await shutdown
            assert held["shutdown_retained_exact_native_callback"]
            assert app.collections_capture_repository is None
            assert (
                type(app.collections_capture_scope_service)
                is wiring._DeferredCollectionsCaptureScope
            )
        elif control in {"scope", "policy", "source"}:
            assert control_task is not None
            await control_task
            assert declined and app.collections_capture_repository is None
            assert app.local_collections_capture_service is None
        else:
            assert app.collections_capture_repository is not None, declined
            assert app.collections_capture_repository.db is database
            assert (
                app.local_collections_capture_service.repository
                is app.collections_capture_repository
            )
            assert (
                app.collections_capture_scope_service.active_authority
                is app.local_collections_capture_authority
            )
        if control == "first_use":
            assert app.collections_capture_repository is held["first_use_repository"]
        if "future" in held:
            assert held["future"].done()
            if route == "legacy" and control != "borrowed":
                assert closed(held["connection"])
                with storage._lock:
                    assert held["connection"] not in held["participant"].connections
                    assert held["lease"] not in storage._live_leases
        if control == "borrowed":
            assert held["connection"] is borrowed and not closed(borrowed)
            assert await asyncio.to_thread(database._held_connection) is borrowed
    finally:
        release.set()
        if supervisor.ident is not None:
            await asyncio.to_thread(supervisor.join, 10)
            assert not supervisor.is_alive()
        if tool is not None:
            for code in diagnostic_codes:
                sys.monitoring.set_local_events(tool, code, 0)
            assert (
                sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, None)
                is observe_decline
            )
            sys.monitoring.set_local_events(tool, callback.__code__, 0)
            assert (
                sys.monitoring.register_callback(
                    tool, sys.monitoring.events.PY_RETURN, None
                )
                is hold
            )
            assert sys.monitoring.get_events(tool) == 0
            sys.monitoring.free_tool_id(tool)
        callback.__kwdefaults__ = original_keywords
        app.runtime_policy = original_policy
        receipt = pin.close()
        assert receipt["original_source_current"]
        await app._shutdown_collections_capture_runtime()
        if borrowed is not None:
            await asyncio.to_thread(database.close)
            assert closed(borrowed)
        await runtime.dispose()
        creators.retire()
    if control == "first_use":
        assert held[
            "first_use_reconciliation_scheduled"
        ], "First use displaced deferred setup without scheduling startup reconciliation"
    assert held[
        "loop_progress_while_original_callback_held"
    ], "Original deferred Collections setup blocked the shared event loop"


@pytest.mark.asyncio
async def test_collections_shutdown_preserves_cancellation_pending_at_entry():
    from types import SimpleNamespace

    from tldw_chatbook.app_service_wiring import _retire_deferred_collections_capture

    entered, release = asyncio.Event(), asyncio.Event()

    async def initializer():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            entered.set()
            await release.wait()
            raise

    child = asyncio.create_task(initializer())
    await asyncio.sleep(0)
    app = SimpleNamespace(_collections_capture_initializer_task=child)

    async def shutdown():
        asyncio.current_task().cancel()
        cancelled = await _retire_deferred_collections_capture(app)
        assert cancelled is not None
        raise cancelled

    task = asyncio.create_task(shutdown())
    try:
        await entered.wait()
        assert not task.done() and not child.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert child.done() and child.cancelled()
    finally:
        release.set()
        await asyncio.gather(child, task, return_exceptions=True)


@pytest.mark.parametrize("change", ["missing_key", "foreign_equality"])
def test_collections_source_check_rejects_changed_keyword_defaults(change):
    from tldw_chatbook import app_service_wiring as wiring
    from tldw_chatbook.Library import collections_capture_service as service

    original = service.LocalCollectionsCaptureService.__init__
    keywords = original.__kwdefaults__
    prior = keywords.copy()
    called = []

    class Foreign:
        def __eq__(self, other):
            called.append(True)
            return True

    try:
        assert wiring._collections_setup_sources_current(
            (service._COLLECTIONS_SETUP_SOURCE,)
        )
        key = next(key for key, value in keywords.items() if value is None)
        if change == "missing_key":
            del keywords[key]
            keywords["foreign_keyword"] = None
        else:
            keywords[key] = Foreign()
        assert not wiring._collections_setup_sources_current(
            (service._COLLECTIONS_SETUP_SOURCE,)
        )
        assert not called
    finally:
        keywords.clear()
        keywords.update(prior)
