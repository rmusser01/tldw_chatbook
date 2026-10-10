"""PYTEST_DONT_REWRITE: qualify local observers against original source bytes."""

import ast
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.thread import _WorkItem
from functools import partial
import inspect
import json
from pathlib import Path
import sqlite3
import sys
import threading
from types import CodeType, FunctionType, MethodType, SimpleNamespace

import pytest

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
@private_profile_test
async def test_original_host_shutdown_joins_issued_browser_worker(
    request, tmp_path, borrowed
):
    from textual.app import App
    from textual.worker import Worker
    from tldw_chatbook.app_lifecycle import LifecycleMixin
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB import private_sqlite
    from tldw_chatbook.DB import base_db as base_source
    from tldw_chatbook.DB import ChaChaNotes_DB as notes_source
    from tldw_chatbook.UI.Console_Modules import console_browser_read as browser
    from tldw_chatbook.UI.Console_Modules import workspace
    from tldw_chatbook.UI.Console_Modules.view_workers import (
        capture_console_view_workers,
        drain_console_view_workers,
    )
    from Tests.UI.test_console_workspace_controller import _workspace_controller

    database = CharactersRAGDB(
        tmp_path / "browser.sqlite", client_id="browser-lifetime"
    )
    service = ChatConversationService(database)
    service.create_conversation(title="Global Native", scope_type="global")
    service.create_conversation(
        title="Default Native", scope_type="workspace", workspace_id="workspace-default"
    )
    participant = database._maintenance_participant
    host = App()
    host._console_runtime_shutdown_task = None
    host._plugin_service = None
    host.console_runtime = None
    controller = _workspace_controller(
        screen=host,
        app_instance=SimpleNamespace(local_chat_conversation_service=service),
    )
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-custody")
    loop.set_default_executor(executor)
    previous = (
        await loop.run_in_executor(executor, database.get_connection)
        if borrowed
        else None
    )
    original_read = browser._StockBrowserRead._read_pair
    original_run = browser._StockBrowserRead.run
    original_shutdown = LifecycleMixin._shutdown_console_runtime
    tree = ast.parse(Path(inspect.getsourcefile(browser)).read_bytes())
    read_body = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "_StockBrowserRead"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_read_pair"
    )
    assigned = next(
        node.lineno
        for node in ast.walk(read_body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "connection"
            for target in node.targets
        )
    )
    hold_line = min(
        node.lineno
        for node in ast.walk(read_body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_require_worker_current"
        and node.lineno > assigned
    )
    pin = OriginalStorageUnitObserver({}, False, lambda _name: None)
    stock_connection = inspect.getattr_static(base_source, "_QuiescentSQLiteConnection")
    assert (
        inspect.getattr_static(notes_source, "_QuiescentSQLiteConnection")
        is stock_connection
    )
    pin._pin(inspect.getattr_static(stock_connection, "close"))
    pin._pin(inspect.getattr_static(stock_connection, "__init__"))
    pin.slots.extend(
        (
            (base_source, "_QuiescentSQLiteConnection", stock_connection),
            (notes_source, "_QuiescentSQLiteConnection", stock_connection),
            (
                stock_connection,
                "__init__",
                inspect.getattr_static(stock_connection, "__init__"),
            ),
            (
                stock_connection,
                "close",
                inspect.getattr_static(stock_connection, "close"),
            ),
        )
    )
    admitted = inspect.getattr_static(private_sqlite, "_connect_registered_sqlite")
    admitted_body = inspect.getattr_static(admitted, "__wrapped__")
    assert (
        dict(zip(admitted.__code__.co_freevars, admitted.__closure__))[
            "function"
        ].cell_contents
        is admitted_body
    )
    for function in (
        private_sqlite._with_storage_admission,
        admitted,
        admitted_body,
        private_sqlite.connect_private_sqlite,
    ):
        pin._pin(function)
    for name in (
        "_with_storage_admission",
        "_connect_registered_sqlite",
        "connect_private_sqlite",
        "_ordinary_connections",
    ):
        pin.slots.append(
            (private_sqlite, name, inspect.getattr_static(private_sqlite, name))
        )
    pin.slots.append((admitted, "__wrapped__", admitted_body))
    dynamic_code = next(
        code
        for code in admitted.__code__.co_consts
        if type(code) is CodeType and code.co_name == "AdmittedConnection"
    )
    dynamic_method_codes = {
        code.co_name: code for code in dynamic_code.co_consts if type(code) is CodeType
    }
    bases_descriptor = inspect.getattr_static(type, "__bases__")
    mro_descriptor = inspect.getattr_static(type, "__mro__")
    pin.slots.extend(
        ((type, "__bases__", bases_descriptor), (type, "__mro__", mro_descriptor))
    )
    for owner, name in (
        (browser._StockBrowserRead, "_read_pair"),
        (browser._StockBrowserRead, "run"),
        (browser._StockBrowserRead, "_retire_created_handle"),
        (workspace.ConsoleWorkspaceController, "_sync_persisted_console_browser_rows"),
        (workspace.ConsoleWorkspaceController, "_refresh_console_persisted_rows_cache"),
        (workspace.ConsoleWorkspaceController, "_persisted_console_browser_rows"),
        (App, "run_worker"),
        (App, "_context"),
        (Worker, "_run"),
        (Worker, "cancel"),
        (Worker, "wait"),
        (LifecycleMixin, "_shutdown_console_runtime"),
        (_WorkItem, "run"),
    ):
        function = inspect.getattr_static(owner, name)
        pin._pin(function)
        pin.slots.append((owner, name, function))
    for function in (
        browser.capture_stock_browser_read,
        browser._sources_current,
        workspace._capture_stock_browser_read,
        _workspace_controller,
        asyncio.to_thread,
        asyncio.futures._chain_future,
        capture_console_view_workers,
        drain_console_view_workers,
    ):
        pin._pin(function)
        pin.slots.append(
            (sys.modules[function.__module__], function.__name__, function)
        )
    entered, release = asyncio.Event(), threading.Event()
    held, issued, invalid = {}, {}, []
    entry_facts = {}
    shutdown_returned = {"observed": False}
    tool = None
    worker = outer = shutdown_waiter = host_retirement = None
    body_error = None
    monitor_retired = False
    monitoring = sys.monitoring

    def native_status(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError as error:
            return "closed" if "closed" in str(error).lower() else "unknown"
        return "open"

    def yielded(code, _offset, _value):
        frame = None
        stage = "source_frame"
        try:
            if code is not original_run.__code__:
                return
            frame = sys._getframe(1)
            owner = frame.f_locals.get("self")
            if (
                frame.f_code is not code
                or frame.f_globals is not original_run.__globals__
            ):
                raise RuntimeError("browser_run_source_changed")
            if (
                type(owner) is not browser._StockBrowserRead
                or owner.controller is not controller
            ):
                return
            stage = "inner_task"
            inner = frame.f_locals["worker"]
            assert type(inner) is asyncio.Task and inner.get_loop() is loop
            issuer = asyncio.current_task()
            stage = "issuer_task"
            assert type(issuer) is asyncio.Task and issuer is not inner
            if issued:
                stage = "same_reyield"
                assert issued == {"owner": owner, "inner": inner, "outer": issuer}
            else:
                issued.update(owner=owner, inner=inner, outer=issuer)
        except BaseException as error:
            if len(invalid) < 16:
                invalid.append("yield." + stage + ":" + type(error).__name__)
        finally:
            frame = None

    def returned(code, _offset, value):
        frame = None
        try:
            frame = sys._getframe(1)
            if (
                code is original_shutdown.__code__
                and frame.f_locals.get("self") is host
            ):
                assert (
                    frame.f_code is code
                    and frame.f_globals is original_shutdown.__globals__
                    and value is None
                )
                shutdown_returned["observed"] = True
        except BaseException as error:
            invalid.append("return:" + type(error).__name__)
        finally:
            frame = None

    def hold(code, line):
        frame = ancestor = item = None
        stage = "source_frame"
        try:
            if code is not original_read.__code__ or line != hold_line or held:
                return
            frame = sys._getframe(1)
            owner = frame.f_locals.get("self")
            if (
                frame.f_code is not code
                or frame.f_globals is not original_read.__globals__
            ):
                raise RuntimeError("browser_native_source_changed")
            if (
                type(owner) is not browser._StockBrowserRead
                or owner.controller is not controller
            ):
                return
            assert owner.database is database and owner.service is service
            stage = "native_actor"
            assert threading.current_thread() is not threading.main_thread()
            connection = frame.f_locals["connection"]
            entry_facts.update(
                native_type=type(connection).__name__,
                native_module=type(connection).__module__,
                connection_id=id(connection),
            )
            stage = "native_type"
            native_type = type(connection)
            assert type(native_type) is type and owner.connection is connection
            assert bases_descriptor.__get__(native_type, type) == (stock_connection,)
            native_mro = mro_descriptor.__get__(native_type, type)
            assert native_mro == (
                native_type,
                *mro_descriptor.__get__(stock_connection, type),
            )
            stage = "admitted_methods"
            new_descriptor = inspect.getattr_static(native_type, "__new__")
            assert type(new_descriptor) is staticmethod
            new = new_descriptor.__func__
            assert (
                type(new) is FunctionType
                and new.__code__ is dynamic_method_codes["__new__"]
            )
            assert new.__globals__ is private_sqlite.__dict__
            assert (
                dict(zip(new.__code__.co_freevars, new.__closure__))[
                    "factory"
                ].cell_contents
                is stock_connection
            )
            for name in ("close", "__del__"):
                function = inspect.getattr_static(native_type, name)
                assert (
                    type(function) is FunctionType
                    and function.__code__ is dynamic_method_codes[name]
                )
                assert function.__globals__ is private_sqlite.__dict__
                pin._pin(function)
                pin.slots.append((native_type, name, function))
            pin._pin(new)
            pin.slots.append((native_type, "__new__", new_descriptor))
            assert inspect.getattr_static(
                native_type, "__init__"
            ) is inspect.getattr_static(stock_connection, "__init__")
            close = inspect.getattr_static(native_type, "close")
            close_cells = dict(zip(close.__code__.co_freevars, close.__closure__))
            assert close_cells["__class__"].cell_contents is native_type
            assert close_cells["capture_lease"].cell_contents is False
            assert close_cells["constructing"].cell_contents is False
            stage = "native_open"
            assert native_status(connection) == "open"
            assert owner.registry.is_registered(connection)
            stage = "work_item_frame"
            ancestor = frame.f_back
            for _ in range(16):
                if ancestor is None or ancestor.f_code is _WorkItem.run.__code__:
                    break
                ancestor = ancestor.f_back
            assert (
                ancestor is not None and ancestor.f_globals is _WorkItem.run.__globals__
            )
            item = ancestor.f_locals["self"]
            stage = "work_item_future"
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            assert type(item.fn) is partial
            stage = "dispatch_callable"
            callback = item.fn.args[0]
            assert type(callback) is MethodType and callback.__self__ is owner
            assert callback.__func__ is original_read
            with storage._lock:
                stage = "operation_lease"
                lease = participant.connections[connection]
                assert private_sqlite._ordinary_connections.get(connection) is lease
                assert close_cells["lease"].cell_contents is lease
                operation = getattr(storage._operation_local, "operation", None)
                assert (
                    type(operation) is storage._Operation
                    and operation in storage._operations
                )
                assert (
                    operation.participant is participant
                    and operation.thread is threading.current_thread()
                )
                assert (
                    lease in storage._live_leases
                    and lease.resource_thread is threading.current_thread()
                )
                assert lease.resource_participant is participant
            held.update(
                owner=owner,
                connection=connection,
                lease=lease,
                operation=operation,
                future=item.future,
                work_item_id=id(item),
                callable_id=id(item.fn),
                thread=threading.current_thread(),
                native_type=native_type,
            )
            loop.call_soon_threadsafe(entered.set)
            stage = "release"
            assert release.wait(10), "Original browser callback was never released"
        except BaseException as error:
            if len(invalid) < 16:
                invalid.append("hold." + stage + ":" + type(error).__name__)
            loop.call_soon_threadsafe(entered.set)
        finally:
            frame = ancestor = item = None

    def inspect_worker():
        assert threading.current_thread() is held["thread"]
        with storage._lock:
            assert held["operation"] not in storage._operations
            if borrowed:
                assert (
                    held["connection"] is previous and native_status(previous) == "open"
                )
                assert (
                    database._local.conn is previous
                    and participant.connections[previous] is held["lease"]
                )
                assert (
                    held["lease"] in storage._live_leases
                    and database._connection_quiescence.is_registered(previous)
                )
            else:
                assert native_status(held["connection"]) == "closed"
                assert held["connection"] not in participant.connections
                assert held["connection"] not in private_sqlite._ordinary_connections
                assert held["lease"] not in storage._live_leases
                assert getattr(database._local, "conn", None) is None
                assert not database._connection_quiescence.is_registered(
                    held["connection"]
                )

    def close_worker():
        if held:
            assert threading.current_thread() is held["thread"]
        database.close()
        if held:
            assert native_status(held["connection"]) == "closed"
            with storage._lock:
                assert held["connection"] not in participant.connections
                assert held["lease"] not in storage._live_leases

    def failure_coordinates(error, step):
        trace = error.__traceback__
        location = {"step": step, "error_type": type(error).__name__}
        try:
            for _ in range(16):
                if trace is None:
                    break
                if trace.tb_frame.f_code.co_filename == __file__:
                    location.update(
                        source_function=trace.tb_frame.f_code.co_name,
                        source_line=trace.tb_lineno,
                    )
                trace = trace.tb_next
            return location
        finally:
            trace = None

    try:
        for function in (
            native_status,
            yielded,
            returned,
            hold,
            inspect_worker,
            close_worker,
            failure_coordinates,
        ):
            pin._pin(function)
        for candidate in range(5, 0, -1):
            if candidate == monitoring.DEBUGGER_ID:
                continue
            try:
                monitoring.use_tool_id(candidate, "original-browser-host-custody")
            except ValueError:
                continue
            tool = candidate
            break
        assert tool is not None and monitoring.get_events(tool) == 0
        callbacks = {
            monitoring.events.LINE: hold,
            monitoring.events.PY_YIELD: yielded,
            monitoring.events.PY_RETURN: returned,
        }
        masks = {
            original_read.__code__: monitoring.events.LINE,
            original_run.__code__: monitoring.events.PY_YIELD,
            original_shutdown.__code__: monitoring.events.PY_RETURN,
        }
        for event, callback in callbacks.items():
            assert monitoring.register_callback(tool, event, callback) is None
        for code, mask in masks.items():
            assert monitoring.get_local_events(tool, code) == 0
            monitoring.set_local_events(tool, code, mask)
        with host._context():
            controller._sync_persisted_console_browser_rows()
        scheduled = tuple(
            candidate
            for candidate in host.workers
            if type(candidate) is Worker
            and vars(candidate).get("_node") is host
            and vars(candidate).get("group") == "console-persisted-browser-cache"
        )
        assert len(scheduled) == 1
        worker = scheduled[0]
        outer = worker._task
        assert type(outer) is asyncio.Task and outer.get_loop() is loop
        await asyncio.wait_for(entered.wait(), 10)
        if not (held and issued and not invalid):
            print(
                "browser_host_entry_failure="
                + json.dumps(
                    {
                        "borrowed": borrowed,
                        "held_present": bool(held),
                        "issued_present": bool(issued),
                        "invalid": invalid[:16],
                        "original_hold_line": hold_line,
                        "entry_facts": entry_facts,
                        "worker_id": id(worker),
                        "outer_task_id": id(outer),
                        "group": worker.group,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        assert held and issued and not invalid
        assert issued["owner"] is held["owner"]
        assert issued["outer"] is outer
        workers = tuple(w for w in host.workers if vars(w).get("_task") is outer)
        assert len(workers) == 1 and type(workers[0]) is Worker
        worker = workers[0]
        assert (
            worker._node is host and worker.group == "console-persisted-browser-cache"
        )
        assert (
            type(worker._work) is partial and worker._work.func.__self__ is controller
        )
        assert (
            worker._work.func.__func__
            is workspace.ConsoleWorkspaceController._refresh_console_persisted_rows_cache
        )
        inner = issued["inner"]
        coroutine = inner.get_coro()
        assert coroutine.cr_code is asyncio.to_thread.__code__
        dispatched = coroutine.cr_frame.f_locals["func"]
        assert (
            type(dispatched) is MethodType
            and dispatched.__self__ is held["owner"]
            and dispatched.__func__ is original_read
        )
        waiter = inner._fut_waiter
        chain_code = next(
            code
            for code in asyncio.futures._chain_future.__code__.co_consts
            if type(code) is CodeType and code.co_name == "_call_set_state"
        )
        chain = tuple(
            callback
            for callback in held["future"]._done_callbacks
            if type(callback) is FunctionType
            and callback.__code__ is chain_code
            and callback.__globals__ is asyncio.futures.__dict__
        )
        assert len(chain) == 1
        cells = dict(zip(chain[0].__code__.co_freevars, chain[0].__closure__))
        assert (
            cells["destination"].cell_contents is waiter and waiter.get_loop() is loop
        )
        pin._pin(chain[0])
        assert browser._sources_current()
        captured = capture_console_view_workers(host)
        shutdown_waiter = asyncio.create_task(original_shutdown(host))
        progress = []
        loop.call_soon(progress.append, True)
        await asyncio.sleep(0.35)
        host_retirement = host._console_runtime_shutdown_task
        assert type(host_retirement) is asyncio.Task
        with storage._lock:
            active = held["operation"] in storage._operations
            live_lease = held["lease"] in storage._live_leases
            registered = (
                participant.connections.get(held["connection"]) is held["lease"]
            )
        print(
            "browser_host_before_release="
            + json.dumps(
                {
                    "borrowed": borrowed,
                    "host_id": id(host),
                    "worker_id": id(worker),
                    "node_id": id(worker._node),
                    "work_id": id(worker._work),
                    "group": worker.group,
                    "outer_task_id": id(outer),
                    "inner_task_id": id(inner),
                    "inner_future_id": id(waiter),
                    "work_item_id": held["work_item_id"],
                    "work_item_callable_id": held["callable_id"],
                    "future_chain_callback_id": id(chain[0]),
                    "thread_ident": held["thread"].ident,
                    "concurrent_future_id": id(held["future"]),
                    "database_id": id(database),
                    "participant_id": id(participant),
                    "connection_id": id(held["connection"]),
                    "operation_id": id(held["operation"]),
                    "lease_id": id(held["lease"]),
                    "original_shutdown_return_observed": shutdown_returned["observed"],
                    "selected_exact_worker": any(
                        row[0] is worker for row in captured[-1]
                    ),
                    "native_open_at_held_entry": True,
                    "operation_active": active,
                    "lease_live": live_lease,
                    "registered": registered,
                    "future_done": held["future"].done(),
                    "future_running": held["future"].running(),
                    "inner_done": inner.done(),
                    "outer_done": outer.done(),
                    "loop_progress": progress == [True],
                    "invalid": invalid,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        assert progress == [True] and active and live_lease and registered
        assert not held["future"].done() and not inner.done() and not outer.done()
        assert (
            not shutdown_returned["observed"] and not host_retirement.done()
        ), "Original host shutdown returned while its issued browser callback remained held"
    except BaseException as error:
        body_error = error
        raise
    finally:
        release.set()
        cleanup_error = None
        cleanup_location = None
        cleanup_step = "join_issued_tasks"
        try:
            if outer is not None:
                await asyncio.gather(outer, return_exceptions=True)
            if issued:
                await asyncio.gather(
                    issued["outer"], issued["inner"], return_exceptions=True
                )
            if host._console_runtime_shutdown_task is not None:
                await asyncio.gather(
                    host._console_runtime_shutdown_task, return_exceptions=True
                )
            if shutdown_waiter is not None:
                await asyncio.gather(shutdown_waiter, return_exceptions=True)
            if held:
                assert held["future"].done()
                cleanup_step = "inspect_same_worker_retirement_or_borrower"
                await loop.run_in_executor(executor, inspect_worker)
                print(
                    "browser_host_after_release="
                    + json.dumps(
                        {
                            "borrowed": borrowed,
                            "original_shutdown_return_observed": shutdown_returned[
                                "observed"
                            ],
                            "future_done": held["future"].done(),
                            "inner_done": issued["inner"].done(),
                            "outer_done": issued["outer"].done(),
                            "captured_native_retirement_or_borrower_preservation_verified": True,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        except BaseException as error:
            cleanup_error = error
            cleanup_location = failure_coordinates(error, cleanup_step)
        try:
            cleanup_step = "retire_original_local_monitor"
            if tool is not None:
                assert monitoring.get_tool(tool) == "original-browser-host-custody"
                assert monitoring.get_events(tool) == 0
                assert all(
                    monitoring.get_local_events(tool, code) == mask
                    for code, mask in masks.items()
                )
                for event, expected in callbacks.items():
                    actual = monitoring.register_callback(tool, event, expected)
                    if actual is not expected:
                        monitoring.register_callback(tool, event, actual)
                        raise RuntimeError("browser_monitor_foreign_callback_preserved")
                for code in masks:
                    monitoring.set_local_events(tool, code, 0)
                for event, expected in callbacks.items():
                    actual = monitoring.register_callback(tool, event, None)
                    if actual is not expected:
                        monitoring.register_callback(tool, event, actual)
                        raise RuntimeError("browser_monitor_callback_owner_changed")
                assert monitoring.get_events(tool) == 0
                monitoring.free_tool_id(tool)
                monitor_retired = True
        except BaseException as error:
            cleanup_error = cleanup_error or error
            cleanup_location = cleanup_location or failure_coordinates(
                error, cleanup_step
            )
        try:
            cleanup_step = "explicit_original_worker_close"
            await loop.run_in_executor(executor, close_worker)
            cleanup_step = "explicit_original_creator_close"
            database.close()
            with storage._lock:
                assert not participant.connections and not participant.retiring_threads
            changed_cells = []
            for (
                function,
                code,
                _globals,
                _defaults,
                _kwdefaults,
                _items,
                _closure,
                cells,
            ) in pin.pins:
                for name, (cell, expected) in zip(code.co_freevars, cells):
                    actual = cell.cell_contents
                    if actual is not expected and len(changed_cells) < 16:
                        changed_cells.append(
                            {
                                "function": function.__qualname__,
                                "freevar": name,
                                "expected_type": type(expected).__name__,
                                "actual_type": type(actual).__name__,
                            }
                        )
            receipt = pin.close()
            if changed_cells:
                print(
                    "browser_host_changed_source_cells="
                    + json.dumps(changed_cells, sort_keys=True),
                    flush=True,
                )
            if held:
                assert type(held["connection"]) is held["native_type"]
                assert bases_descriptor.__get__(held["native_type"], type) == (
                    stock_connection,
                )
                assert mro_descriptor.__get__(held["native_type"], type) == (
                    held["native_type"],
                    *mro_descriptor.__get__(stock_connection, type),
                )
            receipt["original_local_monitor_retired"] = monitor_retired
            print(
                "browser_host_source_receipt=" + json.dumps(receipt, sort_keys=True),
                flush=True,
            )
            cleanup_step = "final_original_source_monitor_and_callback_validity"
            assert monitor_retired and (
                receipt["original_source_current"]
                and not receipt["invalid"]
                and not invalid
            )
        except BaseException as error:
            cleanup_error = cleanup_error or error
            cleanup_location = cleanup_location or failure_coordinates(
                error, cleanup_step
            )
        if cleanup_error is not None:
            print(
                "browser_host_cleanup_failure="
                + json.dumps(cleanup_location, sort_keys=True),
                flush=True,
            )
            if body_error is None:
                raise cleanup_error
            body_error.add_note(
                "Browser host cleanup failed: "
                + json.dumps(cleanup_location, sort_keys=True)
            )
