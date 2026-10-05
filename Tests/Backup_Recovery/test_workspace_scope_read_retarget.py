"""An owned Workspace read must consume the same captured database receiver."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


pytestmark = pytest.mark.bootstrap_profile


_SCRIPT = r"""
import hashlib, json, os, sqlite3, sys, threading
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
assert (route, outcome) == ('workspace_scope_retarget', 'same_file')


def shape(code):
    return (code.co_code, code.co_exceptiontable, code.co_stacksize,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_nlocals, code.co_flags, code.co_names, code.co_varnames,
            code.co_freevars, code.co_cellvars,
            tuple(shape(value) if type(value) is CodeType else value for value in code.co_consts))


def nested(code, qualname):
    if code.co_qualname == qualname:
        return code
    for value in code.co_consts:
        if type(value) is CodeType:
            result = nested(value, qualname)
            if result is not None:
                return result
    return None


def physically_closed(connection):
    if connection is None:
        return None
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError as error:
        assert 'closed' in str(error).lower()
        return True
    return False


def main():
    data = Path(os.environ['XDG_DATA_HOME']).absolute()
    selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    selector.write_text('[general]\nusers_name="scope-retarget"\n[paths]\ndata_dir="'
                        + data.as_posix() + '"\n', encoding='utf-8')
    selector.chmod(0o600)
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission as storage, participants
    from tldw_chatbook.DB import base_db, Workspace_DB
    from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
    from tldw_chatbook.Workspaces import registry_service as module
    original = module.LocalWorkspaceRegistryService.get_workspace_scope
    cached = participants._core_cached_connection
    owned = base_db.operation_owned_connection
    owned_body = owned.__wrapped__
    assert owned.__closure__ and owned.__closure__[0].cell_contents is owned_body
    selected = (original, cached, owned_body)
    sources = {owner.__name__: Path(owner.__file__).absolute()
               for owner in (module, participants, base_db, Workspace_DB)}
    assert all(path.is_relative_to(Path.cwd().resolve()) for path in sources.values())
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    records = tuple((callback, callback.__code__, callback.__globals__, callback.__defaults__,
                     callback.__kwdefaults__) for callback in selected)
    for callback, code, defining, defaults, kwdefaults in records:
        source_path = sources[defining['__name__']]
        compiled = compile(source_path.read_bytes(), str(source_path), 'exec')
        assert shape(nested(compiled, code.co_qualname)) == shape(code)
    code, cache_code, owned_code = (callback.__code__ for callback in selected)
    database_path = config.get_user_data_dir() / 'scope-retarget.sqlite'
    a = Workspace_DB.WorkspaceDB(database_path, client_id='scope-a')
    service = module.LocalWorkspaceRegistryService(a)
    expected = RagScope((ScopeItem('note', 'owned-scope'),), '2026-10-05T00:00:00Z')
    service.create_workspace(workspace_id='scope-owned', name='Owned scope', assistant_defaults=None)
    service.set_workspace_scope('scope-owned', expected)
    b = Workspace_DB.WorkspaceDB(database_path, client_id='scope-b')
    assert type(a) is type(b) is Workspace_DB.WorkspaceDB and a is not b
    a.close()
    b.close()
    actor, borrowed, queried, retargeted, armed = None, None, None, False, False
    before, after, cleaned = {}, {}, {}
    native_connections = []

    def state(database):
        connection = getattr(database._thread_local, 'conn', None)
        participant = database._maintenance_participant
        with storage._lock:
            lease = participant.connections.get(connection)
            result = dict(cache_present=connection is not None,
                          physically_closed=physically_closed(connection),
                          lease_live=lease in storage._live_leases if lease is not None else False,
                          registered=connection in participant.connections)
            if lease is not None:
                assert lease.resource_thread is actor and lease.resource_participant is participant
        return result

    def cache_return(observed_code, offset, value):
        nonlocal retargeted
        if not armed or retargeted:
            return
        frame = sys._getframe(1)
        if (observed_code is not cache_code or frame.f_code is not cache_code
                or frame.f_globals is not cached.__globals__
                or frame.f_locals.get('repository') is not a):
            return
        owner_frame, read_frame = None, None
        parent = frame.f_back
        for _depth in range(64):
            if parent is None:
                break
            if parent.f_code is owned_code and parent.f_locals.get('database') is a:
                owner_frame = parent
            if parent.f_code is code and parent.f_locals.get('self') is service:
                read_frame = parent
            parent = parent.f_back
        if owner_frame is None or read_frame is None:
            return
        assert threading.current_thread() is actor
        assert value is borrowed and physically_closed(borrowed) is False
        assert service.db is a
        before.update(a=state(a), b=state(b))
        assert before['a']['lease_live'] and before['a']['registered']
        assert not before['b']['cache_present']
        service.db = b
        retargeted = True

    def read_line(observed_code, line):
        nonlocal queried
        frame = sys._getframe(1)
        if observed_code is not code or frame.f_locals.get('self') is not service:
            return
        connection = frame.f_locals.get('conn')
        if connection is not None and queried is None:
            assert retargeted and threading.current_thread() is actor
            assert physically_closed(connection) is False
            queried = connection
            native_connections.append(connection)

    monitoring = sys.monitoring
    tool = next(index for index in range(6) if monitoring.get_tool(index) is None)
    monitoring.use_tool_id(tool, 'workspace-scope-captured-owner-retarget')
    monitoring.register_callback(tool, monitoring.events.PY_RETURN, cache_return)
    monitoring.register_callback(tool, monitoring.events.LINE, read_line)
    monitoring.set_local_events(tool, cache_code, monitoring.events.PY_RETURN)
    monitoring.set_local_events(tool, code, monitoring.events.LINE)
    assert monitoring.get_events(tool) == 0

    def worker():
        nonlocal actor, borrowed, armed
        actor = threading.current_thread()
        try:
            borrowed = a._held_connection()
            native_connections.append(borrowed)
            borrowed.execute('BEGIN')
            armed = True
            actual = service.get_workspace_scope('scope-owned')
            assert actual == expected
            assert retargeted and service.db is b and queried is not None
            after.update(a=state(a), b=state(b), queried_original=queried is borrowed,
                         a_transaction=sqlite3.Connection.in_transaction.__get__(borrowed))
        finally:
            armed = False
            service.db = a
            if borrowed is not None and physically_closed(borrowed) is False:
                borrowed.rollback()
            # Only these exact test-owned caches close on their actual worker.
            for database in (b, a):
                connection = getattr(database._thread_local, 'conn', None)
                if connection is not None and all(connection is not old for old in native_connections):
                    native_connections.append(connection)
                database.close()
            cleaned.update(a=state(a), b=state(b),
                           all_recorded_handles_closed=all(physically_closed(c) for c in native_connections))

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='workspace-retarget') as executor:
            future = executor.submit(worker)
            future.result(timeout=10)
        assert actor is not None and not actor.is_alive()
    finally:
        armed = False
        monitoring.set_local_events(tool, cache_code, 0)
        monitoring.set_local_events(tool, code, 0)
        assert monitoring.register_callback(tool, monitoring.events.PY_RETURN, None) is cache_return
        assert monitoring.register_callback(tool, monitoring.events.LINE, None) is read_line
        assert monitoring.get_events(tool) == 0
        monitoring.free_tool_id(tool)
        a.close()
        b.close()
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]
               for name, path in sources.items())
    assert all(callback.__code__ is saved_code and callback.__globals__ is defining
               and callback.__defaults__ is defaults and callback.__kwdefaults__ is kwdefaults
               for callback, saved_code, defining, defaults, kwdefaults in records)
    assert module.LocalWorkspaceRegistryService.get_workspace_scope is original
    assert participants._core_cached_connection is cached
    assert base_db.operation_owned_connection is owned and owned.__wrapped__ is owned_body
    assert cleaned['all_recorded_handles_closed']
    assert all(not cleaned[name]['cache_present'] and not cleaned[name]['lease_live']
               and not cleaned[name]['registered'] for name in ('a', 'b'))
    with storage._lock:
        worker_leases = sum(lease.resource_thread is actor for lease in storage._live_leases)
    guard_counts = dict(network=len(network_guard.blocked_attempts()),
                        real_profile=len(real_profile_guard._violations))
    assert worker_leases == 0 and all(value == 0 for value in guard_counts.values())
    receipt = dict(diagnostic_only=False, same_file=True, real_borrowed_a=True,
                   retargeted=retargeted, before=before, after=after, cleaned=cleaned,
                   original_sources_unchanged=True, global_events=monitoring.get_events(tool),
                   hooks_retired=monitoring.get_tool(tool) is None,
                   worker_positively_retired=not actor.is_alive(), sources=hashes,
                   worker_leases_after_test_owned_cleanup=worker_leases, guard_counts=guard_counts)
    output = selector.parent.parent / 'workspace-scope-retarget.json'
    output.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    output.chmod(0o600)
    assert after['a_transaction'], receipt
    assert after['queried_original'], receipt
    assert not after['b']['cache_present'] and not after['b']['lease_live'], receipt
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


def test_workspace_scope_read_uses_the_operation_owned_database(tmp_path):
    _run(tmp_path, "workspace_scope_retarget", "same_file", script=_SCRIPT)
