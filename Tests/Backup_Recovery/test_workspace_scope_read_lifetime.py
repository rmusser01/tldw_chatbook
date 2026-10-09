"""Stock workspace scope reads retire only operation-owned worker handles."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


pytestmark = pytest.mark.bootstrap_profile


_SCRIPT = r"""
import asyncio
import hashlib
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import CodeType

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()

route, outcome = sys.argv[1:]
assert route == 'workspace_scope'


def shape(code):
    return (
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
            result = nested(value, qualname)
            if result is not None:
                return result
    return None


def native_closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError as error:
        assert 'closed' in str(error).lower()
        return True
    return False


def main():
    root = Path(os.environ['XDG_DATA_HOME']).absolute()
    selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    selector.write_text('[general]\nusers_name="qualifier"\n[paths]\ndata_dir="'
                        + root.as_posix() + '"\n', encoding='utf-8')
    selector.chmod(0o600)
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
    from tldw_chatbook.Workspaces import registry_service as module

    method = module.LocalWorkspaceRegistryService.get_workspace_scope
    code, globals_ = method.__code__, method.__globals__
    source_path = Path(module.__file__).absolute()
    assert source_path == Path(module.__spec__.origin).absolute()
    assert source_path.is_relative_to(Path.cwd().resolve())
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    compiled = compile(source_bytes, str(source_path), 'exec')
    assert shape(nested(compiled, code.co_qualname)) == shape(code)
    assert globals_ is module.__dict__
    defaults, kwdefaults = method.__defaults__, method.__kwdefaults__
    class_ = module.LocalWorkspaceRegistryService
    module_spec = module.__spec__

    class CustomWorkspaceDB(WorkspaceDB):
        pass

    expected = RagScope((ScopeItem('note', 'scope-note'),), '2026-10-05T00:00:00Z')
    database = None
    service = None
    owner_thread = None
    observed = []
    snapshots = []
    invalid = []
    entered = threading.Event()
    release = threading.Event()
    callback_returned = threading.Event()
    held_once = False

    def seed(database):
        service = class_(database)
        service.create_workspace(workspace_id='scope-owned', name='Scope owned',
                                 assistant_defaults=None)
        service.set_workspace_scope('scope-owned', expected)
        return service

    if outcome not in {'memory', 'custom'}:
        database = WorkspaceDB(config.get_user_data_dir() / 'scope-owned.sqlite',
                               client_id='scope-native')
        service = seed(database)
        if outcome == 'error':
            with database.transaction() as connection:
                connection.execute('DROP TABLE workspace_rag_scopes')
        database.close()

    def observe(observed_code, line):
        nonlocal held_once
        if observed_code is not code:
            return
        frame = sys._getframe(1)
        if frame.f_code is not code or frame.f_locals.get('self') is not service:
            return
        connection = frame.f_locals.get('conn')
        if connection is None:
            return
        if any(row['frame'] is frame for row in observed):
            return
        try:
            assert threading.current_thread() is owner_thread
            assert isinstance(connection, sqlite3.Connection)
            assert not native_closed(connection)
            participant = getattr(database, '_maintenance_participant', None)
            with storage._lock:
                lease = None if participant is None else participant.connections.get(connection)
                if outcome not in {'memory', 'custom'}:
                    assert participant is not None
                    assert lease in storage._live_leases
                    assert lease.resource_thread is owner_thread
                    assert lease.resource_participant is participant
            observed.append(dict(frame=frame, connection=connection, lease=lease,
                                 participant=participant, thread=owner_thread))
            if outcome == 'cancel' and not held_once:
                held_once = True
                entered.set()
                if not release.wait(10):
                    invalid.append('held_stock_read_release_timeout')
                callback_returned.set()
        except BaseException as error:
            invalid.append(type(error).__name__)
            entered.set()
            callback_returned.set()

    tool = next(value for value in range(6) if sys.monitoring.get_tool(value) is None)
    sys.monitoring.use_tool_id(tool, 'workspace-scope-read-lifetime')
    sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, observe)
    sys.monitoring.set_local_events(tool, code, sys.monitoring.events.LINE)
    assert sys.monitoring.get_events(tool) == 0

    def snapshot(row):
        with storage._lock:
            lease_live = row['lease'] in storage._live_leases if row['lease'] else False
            registered = (row['participant'] is not None and
                          row['connection'] in row['participant'].connections)
        snapshots.append(dict(closed=native_closed(row['connection']),
                              lease_live=lease_live, registered=registered))

    def worker():
        nonlocal database, service, owner_thread
        owner_thread = threading.current_thread()
        borrowed = None
        if outcome in {'memory', 'custom'}:
            database = (WorkspaceDB(':memory:', client_id='scope-native')
                        if outcome == 'memory' else
                        CustomWorkspaceDB(config.get_user_data_dir() / 'scope-custom.sqlite',
                                          client_id='scope-native'))
            service = seed(database)
        elif outcome == 'borrowed':
            borrowed = database._held_connection()
            borrowed.execute('BEGIN')
        try:
            try:
                result = service.get_workspace_scope('scope-owned')
            except module.WorkspaceRegistryServiceError as error:
                assert outcome == 'error'
                assert isinstance(error.__cause__, sqlite3.Error)
            else:
                assert outcome != 'error'
                assert result == expected
            assert observed, 'stock read never acquired its actual connection'
            snapshot(observed[-1])
            if outcome == 'borrowed':
                assert database._held_connection() is borrowed
                assert sqlite3.Connection.in_transaction.__get__(borrowed)
                assert borrowed.execute('SELECT 1').fetchone()[0] == 1
            if outcome in {'new', 'cancel'}:
                result = service.get_workspace_scope('scope-owned')
                assert result == expected
                snapshot(observed[-1])
        finally:
            # Test-owned cleanup stays on this exact worker, after all snapshots.
            if borrowed is not None and not native_closed(borrowed):
                borrowed.rollback()
            database.close()

    async def cancel_waiter(native):
        waiter_entered = asyncio.Event()
        async def await_worker():
            waiter_entered.set()
            await asyncio.wrap_future(native)
        task = asyncio.create_task(await_worker())
        try:
            await waiter_entered.wait()
            deadline = asyncio.get_running_loop().time() + 10
            while not entered.is_set():
                assert asyncio.get_running_loop().time() < deadline, 'actual stock read not held'
                await asyncio.sleep(.01)
            assert observed and not invalid
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('awaiting task did not cancel')
            assert native.running() and not native.done()
            assert not native_closed(observed[0]['connection'])
            with storage._lock:
                assert observed[0]['lease'] in storage._live_leases
        finally:
            release.set()
        # A second waiter observes actual executor completion, not cancellation.
        await asyncio.wait_for(asyncio.wrap_future(native), 10)
        assert callback_returned.is_set()

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='scope-owned') as executor:
            native = executor.submit(worker)
            if outcome == 'cancel':
                asyncio.run(cancel_waiter(native))
            else:
                native.result(timeout=10)
        assert not invalid, invalid
        assert observed and all(row['thread'] is owner_thread for row in observed)
        assert not owner_thread.is_alive()
        preserve = outcome in {'borrowed', 'memory', 'custom'}
        assert snapshots and all(row['closed'] is not preserve for row in snapshots), snapshots
        if not preserve:
            assert all(not row['lease_live'] and not row['registered'] for row in snapshots), snapshots
        elif outcome == 'borrowed':
            assert all(row['lease_live'] and row['registered'] for row in snapshots), snapshots
        if outcome in {'new', 'cancel'}:
            assert len(observed) == 2
            assert observed[0]['connection'] is not observed[1]['connection']
    finally:
        release.set()
        sys.monitoring.set_local_events(tool, code, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, None)
        sys.monitoring.free_tool_id(tool)
        if database is not None:
            database.close()  # Only this creator thread's cache, never the worker.
    assert sys.monitoring.get_tool(tool) is None
    assert source_path.read_bytes() == source_bytes
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
    assert module.__spec__ is module_spec
    assert Path(module.__file__).absolute() == source_path
    assert Path(module.__spec__.origin).absolute() == source_path
    assert module.LocalWorkspaceRegistryService is class_
    assert class_.get_workspace_scope is method
    assert method.__code__ is code and method.__globals__ is globals_
    assert method.__defaults__ is defaults and method.__kwdefaults__ is kwdefaults
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize(
    "outcome", ["new", "borrowed", "error", "cancel", "memory", "custom"]
)
def test_workspace_scope_read_retires_only_owned_handle(tmp_path, outcome):
    _run(tmp_path, "workspace_scope", outcome, script=_SCRIPT)
