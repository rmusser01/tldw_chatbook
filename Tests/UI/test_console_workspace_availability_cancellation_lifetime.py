"""Held original Workspace SQL callbacks must outlive their cancelled awaiter."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
from types import CodeType, FunctionType, MethodType, SimpleNamespace

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]
assert route == 'workspace_availability_cancel'
selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
root = Path(os.environ['XDG_DATA_HOME']).absolute()
selector.write_text('[general]\nusers_name="qualifier"\n[paths]\ndata_dir="'
                    + root.as_posix() + '"\n', encoding='utf-8')
selector.chmod(0o600)


def shape(code):
    return (
        code.co_name, code.co_qualname, code.co_firstlineno,
        code.co_code, code.co_exceptiontable, code.co_stacksize,
        code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
        code.co_nlocals, code.co_flags, code.co_names, code.co_varnames,
        code.co_freevars, code.co_cellvars,
        tuple(shape(value) if isinstance(value, CodeType) else value
              for value in code.co_consts),
    )


def nested(code, qualname):
    if code.co_qualname == qualname:
        return code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            found = nested(value, qualname)
            if found is not None:
                return found
    return None


def closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError as error:
        assert 'closed' in str(error).lower()
        return True
    return False


async def wait_flag(event):
    deadline = asyncio.get_running_loop().time() + 10
    while not event.is_set():
        assert asyncio.get_running_loop().time() < deadline, 'original callback stage not reached'
        await asyncio.sleep(.01)


def main():
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission as storage, participants
    from tldw_chatbook.Chat import console_preparation_reads as preparation
    from tldw_chatbook.DB import base_db, private_sqlite, Workspace_DB as database_module
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.UI.Console_Modules import workspace as workspace_module
    from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID
    from tldw_chatbook.Workspaces import registry_service as registry_module
    from Tests.UI import test_console_workspace_controller as helpers

    controller_class = workspace_module.ConsoleWorkspaceController
    registry_class = registry_module.LocalWorkspaceRegistryService
    roots = (
        (workspace_module, controller_class, '__init__'),
        (workspace_module, controller_class, '_request_workspace_files_availability_refresh'),
        (workspace_module, controller_class, '_refresh_workspace_files_availability_snapshot'),
        (workspace_module, controller_class, '_read_workspace_files_availability'),
        (workspace_module, controller_class, '_read_owned_workspace_availability'),
        (workspace_module, controller_class, '_workspace_availability_runtime'),
        (preparation, preparation, 'run_preparation_read'),
        (preparation, preparation, '_retire'),
        (preparation, preparation, '_observe_locked'),
        (participants, participants, '_core_operation'),
        (workspace_module, controller_class, '_capture_workspace_files_availability_for_registry'),
        (registry_module, registry_class, '__init__'),
        (registry_module, registry_class, 'list_runtime_bindings'),
        (registry_module, registry_class, '_delete_default_runtime_bindings'),
        (database_module, WorkspaceDB, '__init__'),
        (database_module, WorkspaceDB, '_initialize_schema'),
        (database_module, WorkspaceDB, '_get_connection'),
        (database_module, WorkspaceDB, '_held_connection'),
        (database_module, WorkspaceDB, 'connection'),
        (database_module, WorkspaceDB, 'close'),
        (base_db, base_db, 'run_owned_db_call'),
        (base_db, base_db, 'operation_owned_connection'),
        (base_db, base_db.BaseDB, '__init__'),
        (private_sqlite, private_sqlite, 'connect_private_sqlite'),
        (helpers, helpers, '_workspace_controller'),
        (helpers, helpers._AsyncWorkerScreen, 'run_worker'),
        (helpers, helpers._NoMountScreen, '__init__'),
    )
    bindings, anchors, modules, sources, compiled_sources = [], [], {}, {}, {}

    def pin(function):
        assert type(function) is FunctionType
        if any(function is item[0] for item in anchors):
            return
        module = sys.modules[function.__globals__['__name__']]
        assert function.__globals__ is module.__dict__
        path = Path(module.__file__).absolute()
        assert path == Path(module.__spec__.origin).absolute()
        assert Path(function.__code__.co_filename).absolute() == path
        # Only the original repository bodies and their actual stdlib decorator
        # wrappers are admitted. No alternate file/namespace/body is invoked.
        assert path.is_relative_to(Path.cwd().resolve()) or (
            module.__name__ == 'contextlib'
            and path.is_relative_to(Path(sys.base_prefix).resolve())
        )
        if path not in sources:
            sources[path] = path.read_bytes()
            compiled_sources[path] = compile(sources[path], str(path), 'exec', dont_inherit=True)
        declared = nested(compiled_sources[path], function.__code__.co_qualname)
        assert declared is not None and shape(declared) == shape(function.__code__)
        closure = function.__closure__
        cells = tuple(cell.cell_contents for cell in closure or ())
        keywords = function.__kwdefaults__
        keyword_items = tuple((key, value) for key, value in (keywords or {}).items())
        wrapped = getattr(function, '__wrapped__', None)
        type_params = getattr(function, '__type_params__', ())
        anchors.append((function, function.__code__, module, function.__globals__,
                        function.__defaults__, keywords, keyword_items, closure,
                        cells, wrapped, type_params))
        modules[module.__name__] = (module, module.__spec__, module.__loader__, path)
        # In particular, a contextlib helper and a participants guard retain the
        # exact original body in their declared closures, not just a public name.
        for value in (*cells, wrapped):
            if type(value) is FunctionType:
                pin(value)

    for module, owner, name in roots:
        function = inspect.getattr_static(owner, name)
        pin(function)
        bindings.append((module, owner, name, function))
    assert helpers.ConsoleWorkspaceController is controller_class
    assert workspace_module.WorkspaceDB is WorkspaceDB
    assert workspace_module.LocalWorkspaceRegistryService is registry_class
    assert workspace_module.run_owned_db_call is base_db.run_owned_db_call
    generic = base_db.run_owned_db_call
    assert tuple(value.__name__ for value in generic.__type_params__) == ('_CallParameters', '_CallResult')
    assert generic.__code__.co_freevars == ('_CallResult',)
    assert len(generic.__closure__) == 1
    assert generic.__closure__[0].cell_contents is generic.__type_params__[1]
    for name in ('tldw_chatbook.Backup_Recovery.participants',
                 'tldw_chatbook.Backup_Recovery.storage_admission',
                 'tldw_chatbook.DB.private_sqlite',
                 'tldw_chatbook.Utils.private_paths',
                 'Tests.network_guard', 'Tests.real_profile_guard'):
        module = sys.modules[name]
        path = Path(module.__file__).absolute()
        assert path == Path(module.__spec__.origin).absolute()
        modules[name] = (module, module.__spec__, module.__loader__, path)
        sources[path] = path.read_bytes()

    def source_current():
        assert workspace_module.ConsoleWorkspaceController is controller_class
        assert registry_module.LocalWorkspaceRegistryService is registry_class
        assert database_module.WorkspaceDB is WorkspaceDB
        assert helpers.ConsoleWorkspaceController is controller_class
        assert workspace_module.WorkspaceDB is WorkspaceDB
        assert workspace_module.LocalWorkspaceRegistryService is registry_class
        assert workspace_module.run_owned_db_call is base_db.run_owned_db_call
        for module, owner, name, function in bindings:
            assert inspect.getattr_static(owner, name) is function
        for function, code, module, namespace, defaults, keywords, keyword_items, closure, cells, wrapped, type_params in anchors:
            assert function.__code__ is code and function.__globals__ is namespace
            assert function.__defaults__ is defaults and function.__kwdefaults__ is keywords
            assert function.__closure__ is closure and getattr(function, '__wrapped__', None) is wrapped
            assert getattr(function, '__type_params__', ()) is type_params
            assert len(keywords or {}) == len(keyword_items)
            assert all((keywords or {}).get(key) is value for key, value in keyword_items)
            assert all(cell.cell_contents is value for cell, value in zip(closure or (), cells, strict=True))
            assert sys.modules[module.__name__] is module and module.__dict__ is namespace
        for name, (module, spec, loader, path) in modules.items():
            assert sys.modules[name] is module and module.__spec__ is spec and module.__loader__ is loader
            assert Path(module.__file__).absolute() == path == Path(spec.origin).absolute()
        assert all(path.read_bytes() == source for path, source in sources.items())

    database = WorkspaceDB(config.get_user_data_dir() / 'availability-cancel.sqlite',
                           client_id='availability-cancel')
    assert type(database) is WorkspaceDB and not database.is_memory_db
    registry = registry_class(database)
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id='availability-owned', name='Availability owned',
                              assistant_defaults=None)
    database.close()  # Only the exact test creator's seed cache before submission.
    screen = helpers._AsyncWorkerScreen()
    sync_calls = []
    controller = helpers._workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
        sync_workspace_context=lambda: sync_calls.append('sync'),
    )
    assert type(controller) is controller_class and controller._screen is screen
    default_route = outcome in {'default_double', 'default_success'}
    workspace_id = DEFAULT_WORKSPACE_ID if default_route else 'availability-owned'
    workspace_ids = tuple(sorted((workspace_id, 'availability-owned'))) if default_route else (workspace_id,)
    count = {'single': 1, 'double': 2, 'triple': 3, 'default_double': 2,
             'borrowed_double': 2, 'current_success': 0, 'default_success': 0}[outcome]
    borrowed = outcome == 'borrowed_double'
    list_function = registry_class.list_runtime_bindings
    delete_function = registry_class._delete_default_runtime_bindings
    reader_function = controller_class._read_workspace_files_availability
    owned_function = controller_class._read_owned_workspace_availability
    invoke_code = nested(owned_function.__code__,
                         'ConsoleWorkspaceController._read_owned_workspace_availability.<locals>.read_captured_availability')
    current_code = nested(owned_function.__code__,
                          'ConsoleWorkspaceController._read_owned_workspace_availability.<locals>.current')
    producer_code = nested(preparation.run_preparation_read.__code__,
                           'run_preparation_read.<locals>.invoke')
    assert invoke_code is not None and current_code is not None and producer_code is not None
    tree = ast.parse(sources[Path(registry_module.__file__).absolute()])
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'LocalWorkspaceRegistryService')
    list_body = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'list_runtime_bindings')
    # Hold after the unchanged real SELECT has fetched rows. Default uses its
    # actual no-binding cleanup SELECT; neither branch substitutes a query/result.
    list_hold = next(node.lineno for node in list_body.body if isinstance(node, ast.Return)
                     and isinstance(node.value, ast.Call))
    delete_body = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                       and node.name == '_delete_default_runtime_bindings')
    delete_hold = next(node.lineno for node in ast.walk(delete_body)
                       if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                       and isinstance(node.test.operand, ast.Name)
                       and node.test.operand.id == 'has_bindings')
    held_code = delete_function.__code__ if default_route else list_function.__code__
    hold_line = delete_hold if default_route else list_hold
    entered, release, finished = (threading.Event() for _ in range(3))
    held, invocations, invalid, cancel_states, violations = [], [], [], [], []
    post_retirement = None

    def selected_invoke(frame):
        operation = frame.f_locals.get('reader')
        current = frame.f_locals.get('current')
        producer = frame.f_back
        read = producer.f_locals.get('read') if producer is not None else None
        return (
            frame.f_code is invoke_code and frame.f_globals is workspace_module.__dict__
            and frame.f_locals.get('database') is database
            and frame.f_locals.get('registry') is registry
            and frame.f_locals.get('workspace_ids') == workspace_ids
            and type(operation) is MethodType and operation.__self__ is controller
            and operation.__func__ is reader_function
            and frame.f_locals.get('_core_operation') is participants._core_operation
            and frame.f_locals.get('operation_owned_connection') is base_db.operation_owned_connection
            and type(current) is FunctionType and current.__code__ is current_code
            and producer.f_code is producer_code
            and producer.f_globals is preparation.__dict__
            and type(read) is preparation.ConsolePreparationRead
            and read.creator is controller and read.session_id is None
            and type(read.callback) is FunctionType and read.callback.__code__ is invoke_code
            and producer.f_locals.get('callback') is read.callback
        )

    def observe_start(code, offset):
        frame = sys._getframe(1)
        if code is not invoke_code or not selected_invoke(frame):
            return
        if len(invocations) >= 8:
            invalid.append('selected_invoke_capacity')
            return
        read = frame.f_back.f_locals['read']
        assert read in controller._preparation_reads and not read.retired.done()
        invocations.append(dict(frame=frame, thread=threading.current_thread(), returned=False,
                                physical_read=read))

    def observe_line(code, line):
        frame = sys._getframe(1)
        if code is not held_code or line != hold_line or held:
            return
        if frame.f_locals.get('self') is not registry:
            return
        ancestry, current = [], frame.f_back
        for _ in range(12):
            if current is None:
                break
            ancestry.append(current)
            current = current.f_back
        invoke = next((item for item in ancestry if selected_invoke(item)), None)
        read = next((item for item in ancestry if item.f_code is reader_function.__code__
                     and item.f_locals.get('self') is controller), None)
        if invoke is None or read is None:
            return
        try:
            assert read.f_locals.get('registry') is registry and read.f_locals.get('database') is database
            assert read.f_locals.get('workspace_ids') == workspace_ids
            assert registry.db is database
            connection = frame.f_locals['conn']
            assert isinstance(connection, sqlite3.Connection) and not closed(connection)
            assert getattr(database._thread_local, 'conn', None) is connection
            assert frame.f_locals['has_bindings'] is False if default_route else frame.f_locals['rows'] == []
            thread = threading.current_thread()
            participant = database._maintenance_participant
            operation = getattr(storage._operation_local, 'operation', None)
            with storage._lock:
                lease = participant.connections.get(connection)
                assert lease is not None and lease in storage._live_leases
                assert lease.resource_thread is thread and lease.resource_participant is participant
                assert operation in storage._operations and operation.participant is participant
            record = next(item for item in invocations if item['frame'] is invoke)
            held.append(dict(frame=frame, invoke=record, connection=connection, thread=thread,
                             participant=participant, lease=lease, operation=operation))
            entered.set()
            if not release.wait(10):
                invalid.append('original_callback_release_timeout')
        except BaseException as error:
            invalid.append('held_query:' + type(error).__name__)
            entered.set()

    def observe_return(code, offset, value):
        frame = sys._getframe(1)
        if code is not invoke_code:
            return
        record = next((item for item in invocations if item['frame'] is frame), None)
        if record is None:
            return
        record['returned'] = True
        if held and held[0]['invoke'] is record:
            # invoke RETURN occurs after its original repository interval and
            # operation-owned connection finally, not merely after inner SQL.
            finished.set()

    monitor = sys.monitoring
    tool = next(slot for slot in range(6) if monitor.get_tool(slot) is None)
    callbacks = ((monitor.events.PY_START, observe_start),
                 (monitor.events.LINE, observe_line),
                 (monitor.events.PY_RETURN, observe_return))
    masks = {invoke_code: monitor.events.PY_START | monitor.events.PY_RETURN,
             held_code: monitor.events.LINE}
    installed_callbacks, touched_masks, installed_masks = [], [], []
    owns_tool = False

    def install_monitoring():
        nonlocal owns_tool
        monitor.use_tool_id(tool, 'workspace-availability-cancellation')
        owns_tool = True
        for event, callback in callbacks:
            previous = monitor.register_callback(tool, event, callback)
            installed_callbacks.append((event, callback, previous))
            assert previous is None
        for code, mask in masks.items():
            assert monitor.get_local_events(tool, code) == 0
            # Record before mutation so a partial installation failure still
            # removes this exact code-local mask under unconditional retirement.
            touched_masks.append((code, mask))
            monitor.set_local_events(tool, code, mask)
            installed_masks.append(code)
        assert monitor.get_events(tool) == 0

    def retire_monitoring():
        failures = []
        if not owns_tool:
            return failures
        try:
            try:
                if monitor.get_events(tool) != 0:
                    failures.append('global_events_changed')
            except BaseException as error:
                failures.append('global_mask_read_failed:' + type(error).__name__)
            for code, mask in touched_masks:
                try:
                    actual = monitor.get_local_events(tool, code)
                    if (code in installed_masks and actual != mask) or actual not in {0, mask}:
                        failures.append('local_mask_changed')
                except BaseException as error:
                    failures.append('local_mask_read_failed:' + type(error).__name__)
                finally:
                    try:
                        monitor.set_local_events(tool, code, 0)
                    except BaseException as error:
                        failures.append('local_mask_retirement_failed:' + type(error).__name__)
            for event, callback, previous in installed_callbacks:
                try:
                    if monitor.register_callback(tool, event, previous) is not callback:
                        failures.append('local_callback_changed')
                except BaseException as error:
                    failures.append('local_callback_retirement_failed:' + type(error).__name__)
        finally:
            monitor.free_tool_id(tool)
        return failures

    async def exercise(executor):
        nonlocal post_retirement
        loop = asyncio.get_running_loop()
        loop.set_default_executor(executor)
        borrowed_connection = borrowed_thread = None
        if borrowed:
            def borrow():
                connection = database._held_connection()
                connection.execute('BEGIN')
                return connection, threading.current_thread()
            borrowed_connection, borrowed_thread = await loop.run_in_executor(executor, borrow)
        original = None
        try:
            controller._request_workspace_files_availability_refresh(workspace_ids)
            assert len(screen.workers) == 1
            original = screen.workers[0][0]
            assert type(original) is asyncio.Task and not original.done()
            assert controller._workspace_files_availability_refresh_in_flight
            assert dict(controller._workspace_files_availability_by_id) == {}
            assert dict(controller._workspace_files_runtime_bindings_by_id) == {}
            await wait_flag(entered)
            assert held and not invalid and len(invocations) == 1
            assert held[0]['thread'] is not threading.current_thread()
            if borrowed:
                assert held[0]['connection'] is borrowed_connection and held[0]['thread'] is borrowed_thread
                assert sqlite3.Connection.in_transaction.__get__(borrowed_connection)
            for number in range(1, count + 1):
                cancel_requested = original.cancel()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                with storage._lock:
                    live = (held[0]['lease'] in storage._live_leases and
                            held[0]['connection'] in held[0]['participant'].connections and
                            held[0]['operation'] in storage._operations)
                assert live and not closed(held[0]['connection']) and not finished.is_set()
                assert not held[0]['invoke']['physical_read'].retired.done()
                state = dict(cancel_number=number, cancel_requested=cancel_requested,
                             outer_done=original.done(), outer_cancelling=original.cancelling(),
                             in_flight=controller._workspace_files_availability_refresh_in_flight,
                             exact_callback_live=True, native_lease_live=True,
                             publication_count=len(sync_calls), worker_count=len(screen.workers))
                cancel_states.append(state)
                if original.done() or not state['in_flight']:
                    violations.append('availability_owner_released_before_native_callback_retired')
                if sync_calls or dict(controller._workspace_files_availability_by_id):
                    violations.append('availability_published_before_native_callback_retired')
                # This unchanged scheduling entry must coalesce while the exact
                # native callback is still alive; no state is forged for the test.
                controller._request_workspace_files_availability_refresh(workspace_ids)
                if len(screen.workers) != 1:
                    violations.append('availability_rearmed_before_native_callback_retired')
            if count == 0:
                controller._request_workspace_files_availability_refresh(workspace_ids)
                assert len(screen.workers) == 1 and not sync_calls
                assert controller._workspace_files_availability_refresh_in_flight
        finally:
            release.set()
            # Drain only the known original producer tasks and exact held native
            # invoke. No global task/owner census and no foreign-thread close.
            deadline = loop.time() + 10
            reported_task_errors = set()
            while not finished.is_set():
                for index, (task, _) in enumerate(screen.workers):
                    if task.done() and not task.cancelled() and index not in reported_task_errors:
                        error = task.exception()
                        if error is not None:
                            invalid.append('known_producer_task_error:' + type(error).__name__)
                            reported_task_errors.add(index)
                assert loop.time() < deadline, 'held_original_invoke_normal_return_missing'
                await asyncio.sleep(.01)
            deadline = loop.time() + 10
            while any(not task.done() for task, _ in screen.workers):
                assert loop.time() < deadline, 'known original availability tasks did not retire'
                await asyncio.sleep(.01)
            results = await asyncio.gather(*(task for task, _ in screen.workers), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    invalid.append('known_producer_task_error:' + type(result).__name__)
            assert held and not invalid and all(item['returned'] for item in invocations)
            assert all(item['physical_read'].retired.done()
                       and not item['physical_read'].retired.cancelled()
                       and item['physical_read'] not in controller._preparation_reads
                       for item in invocations)
            connection = held[0]['connection']
            with storage._lock:
                lease_live = held[0]['lease'] in storage._live_leases
                registered = connection in held[0]['participant'].connections
                operation_live = held[0]['operation'] in storage._operations
            post_retirement = dict(closed=closed(connection), lease_live=lease_live,
                                   registered=registered, operation_live=operation_live,
                                   original_invoke_normal_return=held[0]['invoke']['returned'])
            assert not operation_live
            if borrowed:
                assert not closed(connection) and lease_live and registered
                assert sqlite3.Connection.in_transaction.__get__(connection)
                def retire_borrowed():
                    assert threading.current_thread() is borrowed_thread
                    assert getattr(database._thread_local, 'conn', None) is borrowed_connection
                    borrowed_connection.rollback()
                    database.close()  # Only the original borrower on its owner Thread.
                await loop.run_in_executor(executor, retire_borrowed)
                assert closed(connection)
                with storage._lock:
                    assert held[0]['lease'] not in storage._live_leases
                    assert connection not in held[0]['participant'].connections
            else:
                assert closed(connection) and not lease_live and not registered
        assert not controller._workspace_files_availability_refresh_in_flight
        if count:
            assert original.cancelled()
            if not violations:
                assert not sync_calls and dict(controller._workspace_files_availability_by_id) == {}
                before = len(screen.workers)
                controller._request_workspace_files_availability_refresh(workspace_ids)
                assert len(screen.workers) == before + 1
                await screen.workers[-1][0]
                assert len(sync_calls) == 1
        else:
            assert not original.cancelled() and len(sync_calls) == 1
        if not violations:
            assert dict(controller._workspace_files_availability_by_id) == {item: False for item in workspace_ids}
            assert dict(controller._workspace_files_runtime_bindings_by_id) == {item: () for item in workspace_ids}

    try:
        install_monitoring()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='availability-original') as executor:
            asyncio.run(exercise(executor))
    finally:
        release.set()
        try:
            failures = retire_monitoring()
        finally:
            database.close()  # Creator cache only after actual worker retirement.
        assert not failures and (not owns_tool or monitor.get_tool(tool) is None)
    source_current()
    receipt = dict(outcome=outcome, held_stock_callback=True, source_current=True,
                   callback_route='default_cleanup' if default_route else 'runtime_binding_list',
                   guards_replaced=False, global_events=0, hooks_retired=True,
                   repeated_cancel_states=cancel_states, post_retirement=post_retirement,
                   original_invoke_count=len(invocations), violation_reasons=violations,
                   source_hashes={str(path): hashlib.sha256(source).hexdigest()
                                  for path, source in sources.items()})
    (selector.parent.parent / 'workspace-availability-cancel-receipt.json').write_text(
        json.dumps(receipt, indent=2), encoding='utf-8')
    assert not violations, violations
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize(
    "outcome",
    [
        "single",
        "double",
        "triple",
        "default_double",
        "borrowed_double",
        "current_success",
        "default_success",
    ],
)
def test_workspace_availability_retains_native_callback_across_cancellation(
    tmp_path, outcome
):
    _run(tmp_path, "workspace_availability_cancel", outcome, script=_SCRIPT)
