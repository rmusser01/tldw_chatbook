"""Repeated cancellation retains unchanged finite historical callback ownership."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
from types import CodeType

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]
assert route == 'historical_read_cancel'
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
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook import config
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB import base_db
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module
    from tldw_chatbook.UI.Console_Modules import agent as agent_module
    from Tests.UI import test_console_refresh_read_batching as helpers

    functions = (
        (agent_module, agent_module.ConsoleAgentController, '_load_historical_presentation'),
        (agent_module, agent_module.ConsoleAgentController, '_presentation_historical_snapshot'),
        (bridge_module, bridge_module.ConsoleAgentBridge, '_derive_historical_snapshot'),
        (base_db, base_db, 'run_owned_db_call'),
        (helpers, helpers, '_agent'),
        (helpers, helpers, '_historical_bridge'),
    )
    anchors, sources = [], {}
    for module, owner, name in functions:
        function = inspect.getattr_static(owner, name)
        path = Path(module.__file__).absolute()
        assert path == Path(module.__spec__.origin).absolute()
        assert path.is_relative_to(Path.cwd().resolve())
        source = path.read_bytes()
        compiled = compile(source, str(path), 'exec', dont_inherit=True)
        assert shape(nested(compiled, function.__qualname__)) == shape(function.__code__)
        assert function.__globals__ is module.__dict__
        closure = function.__closure__
        type_params = function.__type_params__
        if module is base_db and name == 'run_owned_db_call':
            definition = next(node for node in ast.parse(source).body
                              if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
            assert tuple(node.name for node in definition.type_params) == ('_CallParameters', '_CallResult')
            assert tuple(value.__name__ for value in type_params) == ('_CallParameters', '_CallResult')
            assert function.__code__.co_freevars == ('_CallResult',)
            assert closure is not None and len(closure) == 1
            assert closure[0].cell_contents is type_params[1]
        else:
            assert closure is None and type_params == ()
        cells = tuple(cell.cell_contents for cell in closure or ())
        anchors.append((module, owner, name, function, function.__code__, function.__globals__,
                        function.__defaults__, function.__kwdefaults__, module.__spec__, module.__loader__,
                        closure, cells, type_params))
        sources[path] = source
    # Native ownership/guard modules stay installed byte for byte; no replacement
    # callback, getter, transaction, admission predicate or configuration guard.
    for name in ('tldw_chatbook.DB.AgentRuns_DB',
                 'tldw_chatbook.Backup_Recovery.participants',
                 'tldw_chatbook.Backup_Recovery.storage_admission'):
        module = sys.modules[name]
        sources[Path(module.__file__).absolute()] = Path(module.__file__).read_bytes()

    loader = agent_module.ConsoleAgentController._load_historical_presentation
    derive = bridge_module.ConsoleAgentBridge._derive_historical_snapshot
    load_code, derive_code = loader.__code__, derive.__code__
    tree = ast.parse(sources[Path(bridge_module.__file__).absolute()])
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'ConsoleAgentBridge')
    body = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == '_derive_historical_snapshot')
    after_query = next(node.lineno for node in body.body if isinstance(node, ast.If)
                       and isinstance(node.test, ast.UnaryOp)
                       and isinstance(node.test.operand, ast.Name)
                       and node.test.operand.id == 'primary_records')
    load_tree = ast.parse(sources[Path(agent_module.__file__).absolute()])
    load_class = next(node for node in load_tree.body if isinstance(node, ast.ClassDef)
                      and node.name == 'ConsoleAgentController')
    load_body = next(node for node in load_class.body if isinstance(node, ast.AsyncFunctionDef)
                     and node.name == '_load_historical_presentation')
    shield_lines = sorted({node.lineno for node in ast.walk(load_body)
                           if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute)
                           and isinstance(node.func.value, ast.Name)
                           and node.func.value.id == 'asyncio' and node.func.attr == 'shield'})
    assert len(shield_lines) == 2
    drain_line = shield_lines[1]

    database = AgentRunsDB(config.get_user_data_dir() / 'historical-cancel.sqlite',
                           client_id='historical-cancel')
    bridge, child = helpers._historical_bridge(database)
    database.close()  # Close only the test creator's seed cache before the worker.
    agent, tasks = helpers._agent(bridge)
    entered, release, returned, drain_entered = (threading.Event() for _ in range(4))
    held, inner_tasks, invalid, observed_states = [], [], [], []
    borrowed = outcome == 'borrowed_double'

    def observe_line(code, line):
        frame = sys._getframe(1)
        if code is load_code and frame.f_locals.get('self') is agent:
            if line == drain_line:
                drain_entered.set()
            return
        if code is not derive_code or line != after_query or held:
            return
        if frame.f_locals.get('self') is not bridge or frame.f_locals.get('database') is not database:
            return
        try:
            assert frame.f_code is derive_code and frame.f_locals['primary_records']
            connection = getattr(database._thread_local, 'conn', None)
            thread = threading.current_thread()
            assert isinstance(connection, sqlite3.Connection) and not closed(connection)
            participant = database._maintenance_participant
            operation = getattr(storage._operation_local, 'operation', None)
            with storage._lock:
                lease = participant.connections.get(connection)
                assert lease is not None and lease in storage._live_leases
                assert lease.resource_thread is thread
                assert lease.resource_participant is participant
                assert operation in storage._operations and operation.participant is participant
            held.append(dict(frame=frame, connection=connection, thread=thread,
                             participant=participant, lease=lease, operation=operation))
            entered.set()
            if not release.wait(10):
                invalid.append('original_callback_release_timeout')
        except BaseException as error:
            invalid.append(type(error).__name__)
            entered.set()

    def observe_yield(code, offset, value):
        frame = sys._getframe(1)
        if code is not load_code or frame.f_locals.get('self') is not agent:
            return
        worker = frame.f_locals.get('worker')
        if worker is None or any(worker is item for item in inner_tasks):
            return
        if len(inner_tasks) >= 4:
            invalid.append('inner_task_capacity')
            return
        assert type(worker) is asyncio.Task
        inner_tasks.append(worker)

    def observe_return(code, offset, value):
        frame = sys._getframe(1)
        if code is derive_code and held and frame is held[0]['frame']:
            returned.set()

    tool = next(slot for slot in range(6) if sys.monitoring.get_tool(slot) is None)
    monitor = sys.monitoring
    monitor.use_tool_id(tool, 'historical-read-repeated-cancel')
    callbacks = ((monitor.events.LINE, observe_line),
                 (monitor.events.PY_YIELD, observe_yield),
                 (monitor.events.PY_RETURN, observe_return))
    for event, callback in callbacks:
        assert monitor.register_callback(tool, event, callback) is None
    masks = {derive_code: monitor.events.LINE | monitor.events.PY_RETURN,
             load_code: monitor.events.LINE | monitor.events.PY_YIELD}
    for code, mask in masks.items():
        assert monitor.get_local_events(tool, code) == 0
        monitor.set_local_events(tool, code, mask)
    assert monitor.get_events(tool) == 0
    violations = []
    post_retirement = None

    async def exercise(executor):
        nonlocal post_retirement
        loop = asyncio.get_running_loop()
        loop.set_default_executor(executor)
        if borrowed:
            def borrow():
                connection = database._held_connection()
                connection.execute('BEGIN')
                return connection, threading.current_thread()
            borrowed_connection, borrowed_thread = await loop.run_in_executor(executor, borrow)
        else:
            borrowed_connection = borrowed_thread = None
        original = None
        try:
            assert agent._console_agent_fleet_rows() == ()
            assert len(tasks) == 1
            original = tasks[0]
            state = agent._console_historical_read
            assert state is not None and state['pending']
            await wait_flag(entered)
            assert held and not invalid and inner_tasks
            assert held[0]['thread'] is not threading.current_thread()
            if borrowed:
                assert held[0]['connection'] is borrowed_connection
                assert held[0]['thread'] is borrowed_thread
                assert sqlite3.Connection.in_transaction.__get__(borrowed_connection)
            count = {'single': 1, 'double': 2, 'triple': 3,
                     'borrowed_double': 2, 'current_success': 0}[outcome]
            for number in range(1, count + 1):
                cancel_requested = original.cancel()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                if number == 1:
                    assert drain_entered.is_set(), 'first original shield-drain never reached'
                with storage._lock:
                    live = (held[0]['lease'] in storage._live_leases and
                            held[0]['connection'] in held[0]['participant'].connections and
                            held[0]['operation'] in storage._operations)
                assert live and not closed(held[0]['connection']) and not returned.is_set()
                snapshot = dict(cancel_number=number, cancel_requested=cancel_requested,
                                outer_done=original.done(), outer_cancelling=original.cancelling(),
                                state_same=agent._console_historical_read is state,
                                pending=state['pending'], inner_done=inner_tasks[0].done(),
                                exact_callback_live=True, native_lease_live=live)
                observed_states.append(snapshot)
                if original.done() or agent._console_historical_read is not state or not state['pending']:
                    violations.append('pending_owner_released_before_native_callback_retired')
                # This is the unchanged actual presentation request, not a forged
                # final state. The retained pending owner must prevent a new read.
                agent._console_agent_fleet_rows()
                if len(tasks) != 1:
                    violations.append('historical_read_rearmed_before_native_callback_retired')
        finally:
            release.set()
            # Await only the known actual original callbacks/tasks; no owner
            # census, foreign native close or global cleanup is performed.
            deadline = loop.time() + 10
            while any(not task.done() for task in tasks) or any(not task.done() for task in inner_tasks):
                assert loop.time() < deadline, 'known original historical tasks did not retire'
                await asyncio.sleep(.01)
            await asyncio.gather(*tasks, *inner_tasks, return_exceptions=True)
            assert returned.is_set() and not invalid
            assert held and held[0]['thread'].is_alive()
            connection = held[0]['connection']
            with storage._lock:
                lease_live = held[0]['lease'] in storage._live_leases
                registered = connection in held[0]['participant'].connections
                operation_live = held[0]['operation'] in storage._operations
            post_retirement = dict(closed=closed(connection), lease_live=lease_live,
                                   registered=registered, operation_live=operation_live,
                                   actual_inner_task_terminal=inner_tasks[0].done())
            assert not operation_live and inner_tasks[0].done()
            if borrowed:
                assert not closed(connection) and lease_live and registered
                assert sqlite3.Connection.in_transaction.__get__(connection)
                def retire_borrowed():
                    assert threading.current_thread() is borrowed_thread
                    assert getattr(database._thread_local, 'conn', None) is borrowed_connection
                    borrowed_connection.rollback()
                    database.close()  # Exact borrower on its original owner Thread.
                await loop.run_in_executor(executor, retire_borrowed)
                assert closed(connection)
                with storage._lock:
                    assert held[0]['lease'] not in storage._live_leases
                    assert connection not in held[0]['participant'].connections
            else:
                assert closed(connection) and not lease_live and not registered

        if outcome == 'current_success':
            assert agent._console_historical_read is state and not state['pending']
            assert state['value'].subagents[0].run_id == child
        else:
            assert original.cancelled()
            if not violations:
                assert agent._console_historical_read is None and not state['pending']
                before = len(tasks)
                assert agent._console_agent_fleet_rows() == ()
                assert len(tasks) == before + 1
                await asyncio.gather(*tasks[before:])
                assert agent._console_agent_fleet_rows()[0].row_id == child
        assert bridge._historical_cache == {}, 'presentation read changed authority cache'

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='historical-original') as executor:
            asyncio.run(exercise(executor))
    finally:
        release.set()
        failures = []
        try:
            assert monitor.get_events(tool) == 0
            for code, mask in masks.items():
                assert monitor.get_local_events(tool, code) == mask
                monitor.set_local_events(tool, code, 0)
            for event, callback in callbacks:
                assert monitor.register_callback(tool, event, None) is callback
        except BaseException as error:
            failures.append(type(error).__name__)
        finally:
            monitor.free_tool_id(tool)
        database.close()  # Creator cache only, after known worker retirement.
        assert not failures and monitor.get_tool(tool) is None
    for module, owner, name, function, code, namespace, defaults, kwdefaults, spec, loader_, closure, cells, type_params in anchors:
        assert sys.modules[module.__name__] is module and module.__spec__ is spec
        assert module.__loader__ is loader_ and Path(spec.origin).absolute() == Path(module.__file__).absolute()
        assert inspect.getattr_static(owner, name) is function
        assert function.__code__ is code and function.__globals__ is namespace
        assert function.__defaults__ is defaults and function.__kwdefaults__ is kwdefaults
        assert function.__closure__ is closure and function.__type_params__ is type_params
        assert all(cell.cell_contents is value for cell, value in zip(closure or (), cells, strict=True))
    assert all(path.read_bytes() == source for path, source in sources.items())
    receipt = dict(outcome=outcome, held_stock_callback=True, source_current=True,
                   guards_replaced=False, global_events=0, hooks_retired=True,
                   repeated_cancel_states=observed_states, post_retirement=post_retirement,
                   violation_reasons=violations,
                   source_hashes={str(path): hashlib.sha256(source).hexdigest()
                                  for path, source in sources.items()})
    (selector.parent.parent / 'historical-read-cancel-receipt.json').write_text(
        json.dumps(receipt, indent=2), encoding='utf-8')
    assert not violations, violations
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize(
    "outcome", ["single", "double", "triple", "borrowed_double", "current_success"]
)
def test_historical_read_retains_native_callback_across_cancellation(tmp_path, outcome):
    _run(tmp_path, "historical_read_cancel", outcome, script=_SCRIPT)
