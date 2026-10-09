"""Actual original settlement retirement precedes the probe's durable oracle."""

import os

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


pytestmark = pytest.mark.bootstrap_profile


_SCRIPT = r"""
import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from types import CodeType

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]
assert route == 'trace_retirement'


def shape(code):
    return (code.co_code, code.co_exceptiontable, code.co_stacksize,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_nlocals, code.co_flags, code.co_names, code.co_varnames,
            code.co_freevars, code.co_cellvars,
            tuple(shape(value) if isinstance(value, CodeType) else value
                  for value in code.co_consts))


def nested(code, name):
    if code.co_qualname == name:
        return code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            found = nested(value, name)
            if found is not None:
                return found
    return None


def physically_closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError as error:
        assert 'closed' in str(error).lower()
        return True
    return False


async def main():
    root = Path(os.environ['XDG_DATA_HOME']).absolute()
    selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    selector.write_text('[general]\nusers_name="qualifier"\n[paths]\ndata_dir="'
                        + root.as_posix() + '"\n', encoding='utf-8')
    selector.chmod(0o600)
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB import private_sqlite
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.Chat.console_trace_settlement import ConsoleTraceSettlementCoordinator
    from tldw_chatbook.Chat.console_trace_models import TraceCallState
    from Tests.Chat.test_console_trace_settlement import _call, _request

    test_path = Path.cwd() / 'Tests/Performance/test_console_native_pause_probe.py'
    test_raw = test_path.read_bytes()
    node = next(item for item in ast.parse(test_raw).body
                if isinstance(item, ast.AsyncFunctionDef)
                and item.name == '_await_probe_trace_settlement')
    namespace = dict(asyncio=asyncio, time=time)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(test_path), 'exec'), namespace)
    await_retirement = namespace[node.name]
    prepared = ConsoleTraceSettlementCoordinator._settle_prepared
    methods = (prepared, ConsoleTraceSettlementCoordinator._settle_claimed,
               ConsoleChatStore._run_provider_trace_settlement,
               ConsoleChatStore._drain_provider_trace_settlement_work,
               ConsoleChatStore.pending_provider_trace_settlement_work_count)
    records, sources = [], {test_path: test_raw}
    for method in methods:
        code = method.__code__
        path = Path(code.co_filename).absolute()
        assert path.is_relative_to(Path.cwd().resolve())
        raw = path.read_bytes()
        assert shape(nested(compile(raw, str(path), 'exec'), code.co_qualname)) == shape(code)
        records.append((method, code, method.__globals__, method.__defaults__,
                        method.__kwdefaults__, method.__closure__))
        sources[path] = raw

    database = CharactersRAGDB(str(root / 'trace-retirement.sqlite'), 'trace-retirement')
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _, _, call_id = _call(database, repository, state=TraceCallState.RESPONSE_STARTED)
    handoff = coordinator.prepare_handoff(database, _request(call_id))
    database.close_connection()  # Creator's cache only; worker starts without a borrower.
    store = ConsoleChatStore(settle_provider_traces_off_thread=True)
    entered, release = threading.Event(), threading.Event()
    captured, invalid = {}, []
    tool = next(value for value in range(6) if sys.monitoring.get_tool(value) is None)
    monitoring = sys.monitoring
    monitoring.use_tool_id(tool, 'probe-original-trace-retirement')

    def returned(code, offset, value):
        if code is not prepared.__code__ or captured:
            return
        try:
            actor = threading.current_thread()
            connection = database._local.conn
            assert isinstance(connection, sqlite3.Connection) and not physically_closed(connection)
            assert database._connection_quiescence.is_registered(connection)
            with storage._lock:
                lease = private_sqlite._ordinary_connections[connection]
                assert lease in storage._live_leases and lease.resource_thread is actor
            captured.update(actor=actor, connection=connection, lease=lease)
            entered.set()
            assert release.wait(10), 'original prepared-return hold was not released'
        except BaseException as error:
            invalid.append(type(error).__name__)
            entered.set()

    monitoring.register_callback(tool, monitoring.events.PY_RETURN, returned)
    monitoring.set_local_events(tool, prepared.__code__, monitoring.events.PY_RETURN)
    assert monitoring.get_events(tool) == 0
    waiter = None
    try:
        store.register_provider_trace_settlement('missing-owner', handoff)
        assert await asyncio.to_thread(entered.wait, 10), 'actual prepared-return boundary missing'
        assert not invalid and captured
        # SQL has committed, but the original outer owned handle is still live.
        assert store.pending_provider_trace_settlement_work_count() == 1
        assert not physically_closed(captured['connection'])
        deadline = time.perf_counter() + 15
        waiter = asyncio.create_task(await_retirement(store, deadline=deadline))
        await asyncio.sleep(0.02)
        assert not waiter.done(), 'work count cannot become zero before actual owned retirement'
        assert store.pending_provider_trace_settlement_work_count() == 1
        if outcome == 'expired':
            with_error = asyncio.create_task(await_retirement(store, deadline=time.perf_counter() - 1))
            try:
                await with_error
            except AssertionError:
                pass
            else:
                raise AssertionError('expired original send allowance accepted live settlement')
        elif outcome == 'cancel':
            waiter.cancel()
            try:
                await waiter
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('cancelled fixture waiter survived')
            assert store.pending_provider_trace_settlement_work_count() == 1
            assert not physically_closed(captured['connection'])
            waiter = asyncio.create_task(await_retirement(store, deadline=deadline))
        release.set()
        await waiter
        assert store.pending_provider_trace_settlement_work_count() == 0
        assert physically_closed(captured['connection'])
        assert not database._connection_quiescence.is_registered(captured['connection'])
        with storage._lock:
            assert captured['lease'] not in storage._live_leases
        # This is a fresh original repository reread, unlike the CI teardown snapshot.
        with database.transaction() as cursor:
            assert repository.get_call(cursor, call_id).state is TraceCallState.COMPLETE
            assert repository.get_response_link(cursor, call_id) is not None
    finally:
        release.set()
        if waiter is not None:
            await asyncio.gather(waiter, return_exceptions=True)
        await asyncio.to_thread(store.end_app_runtime)
        monitoring.set_local_events(tool, prepared.__code__, 0)
        assert monitoring.register_callback(tool, monitoring.events.PY_RETURN, None) is returned
        assert monitoring.get_events(tool) == 0
        monitoring.free_tool_id(tool)
        database.close_connection()
    assert captured and not captured['actor'].is_alive() and not invalid
    assert all(path.read_bytes() == raw for path, raw in sources.items())
    assert all(method.__code__ is code and method.__globals__ is defining
               and method.__defaults__ is defaults and method.__kwdefaults__ is kwdefaults
               and method.__closure__ is closure
               for method, code, defining, defaults, kwdefaults, closure in records)
    guard_counts = dict(network=len(network_guard.blocked_attempts()),
                        real_profile=len(real_profile_guard._violations))
    assert all(value == 0 for value in guard_counts.values())
    with storage._lock:
        worker_leases = sum(lease.resource_thread is captured['actor'] for lease in storage._live_leases)
    assert worker_leases == 0
    receipt = dict(outcome=outcome, actual_prepared_native_return_held=True,
                   pending_until_owned_connection_retired=True, fresh_complete_state_reread=True,
                   response_link_reread=True, original_methods_current=True, source_stable=True,
                   global_events=0, hooks_retired=monitoring.get_tool(tool) is None,
                   original_worker_positively_retired=True, worker_leases=worker_leases,
                   guard_counts=guard_counts, invalid=invalid,
                   sources={str(path): hashlib.sha256(raw).hexdigest() for path, raw in sources.items()})
    (root / ('trace-retirement-' + outcome + '.receipt.json')).write_text(
        json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


with user_fixture_default_owner():
    asyncio.run(main())
print('retired and reopened')
"""


@pytest.mark.skipif(
    os.name != "nt",
    reason="Native sqlite HANDLE retirement control is qualified on Windows",
)
@pytest.mark.parametrize("outcome", ["success", "expired", "cancel"])
def test_original_pending_work_includes_native_settlement_retirement(tmp_path, outcome):
    _run(tmp_path, "trace_retirement", outcome, script=_SCRIPT)
