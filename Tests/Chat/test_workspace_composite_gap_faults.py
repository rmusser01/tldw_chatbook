"""Actual native source-refusal retirement and nonstock service lookup."""

import pytest
from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile
_SCRIPT = r"""
import dis
import hashlib
import inspect
import json
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import CodeType

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()

case = sys.argv[1]
route, ownership = 'status', 'cold'
assert route in {'admit', 'status', 'capture', 'save_binding'}
assert ownership in {'cold', 'borrowed'}
assert route != 'save_binding' or ownership == 'cold'


def shape(code):
    return (code.co_code, code.co_exceptiontable, code.co_stacksize,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_nlocals, code.co_flags, code.co_names, code.co_varnames,
            code.co_freevars, code.co_cellvars,
            tuple(shape(value) if isinstance(value, CodeType) else value
                  for value in code.co_consts))


def nested(code, qualname):
    if code.co_qualname == qualname:
        return code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            result = nested(value, qualname)
            if result is not None:
                return result
    return None


def physically_closed(connection):
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
                        + root.as_posix() + '"\n[change_review]\nenabled=true\n',
                        encoding='utf-8')
    selector.chmod(0o600)
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import participants
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB import base_db
    from tldw_chatbook.DB import private_sqlite
    from tldw_chatbook.DB import Workspace_DB as database_module
    from tldw_chatbook.Workspaces import change_review_consent as consent_module
    from tldw_chatbook.Workspaces import file_inspector as inspector_module
    from tldw_chatbook.Workspaces import registry_service as registry_module

    WorkspaceDB = database_module.WorkspaceDB
    Registry = registry_module.LocalWorkspaceRegistryService
    Consent = consent_module.ChangeReviewConsentService
    Inspector = inspector_module.WorkspaceFileInspector
    opener = base_db.BaseDB._get_connection
    held_body = WorkspaceDB._held_connection.__wrapped__
    register = participants._register_core_connection
    query_names = ('get_workspace', 'read_change_review_consent',
                   'list_runtime_bindings', 'get_runtime_binding')
    query_methods = {name: Registry.__dict__[name] for name in query_names}
    methods = (
        opener, register, held_body, *query_methods.values(), Registry.save_runtime_binding,
        Registry.list_folder_bindings, Consent.admit_turn, Consent.status,
        Consent._current_bindings, Inspector.capture_binding, Inspector._current_scope,
    )
    records = tuple((method, method.__code__, method.__globals__, method.__defaults__,
                     method.__kwdefaults__, method.__closure__) for method in methods)
    modules = (config, base_db, private_sqlite, database_module, participants,
               storage, registry_module, consent_module, inspector_module)
    sources = {}
    origins = {}
    compiled = {}
    for module in modules:
        path = Path(module.__file__).resolve()
        origin = Path(module.__spec__.origin).resolve()
        assert path == origin and path.is_relative_to(Path.cwd().resolve())
        raw = path.read_bytes()
        sources[path] = raw
        origins[module] = (module.__file__, module.__spec__, module.__spec__.origin)
        compiled[module.__dict__['__name__']] = compile(raw, str(path), 'exec')
    for method, code, defining, defaults, kwdefaults, closure in records:
        declared = nested(compiled[defining['__name__']], code.co_qualname)
        assert declared is not None and shape(declared) == shape(code)
    guarded_bindings = tuple((module, name, module.__dict__[name]) for module, name in (
        (storage, '_acquire_storage'), (storage, '_check_operation'),
        (participants, '_core_closing'), (participants, '_core_operation'), (participants, '_core_getter'),
        (participants, '_core_transaction'), (private_sqlite, 'connect_private_sqlite'),
        (base_db, 'operation_owned_connection'),
    ))
    guarded_records = tuple((method, method.__code__, method.__globals__,
                             method.__defaults__, method.__kwdefaults__, method.__closure__)
                            for module, name, method in guarded_bindings)
    descriptors = tuple((name, WorkspaceDB.__dict__[name]) for name in
                        ('_get_connection', '_held_connection', 'connection', 'transaction', 'close'))


    assert case in {'new_empty', 'new_foreign', 'borrowed_empty', 'consent_lookup', 'consent_descriptor', 'inspector_lookup', 'inspector_descriptor'}
    retirement_case = case in {'new_empty', 'new_foreign', 'borrowed_empty'}
    database = WorkspaceDB(config.get_user_data_dir() / 'gap-A.sqlite', client_id='gap-A')
    registry = Registry(database)
    registry.create_workspace(workspace_id='gap', name='A-literal', assistant_defaults=None)
    root_a = Path(os.environ['HOME']).absolute() / 'gap-folder-A'
    root_a.mkdir(mode=0o700)
    binding_a = registry.add_folder_binding('gap', root_a)
    registry.set_change_review_enabled('gap', True)
    consent, inspector = Consent(registry), Inspector(registry)
    database.close()
    foreign = WorkspaceDB(config.get_user_data_dir() / 'gap-B.sqlite', client_id='gap-B')
    registry_b = Registry(foreign)
    registry_b.create_workspace(workspace_id='gap', name='B-literal', assistant_defaults=None)
    root_b = Path(os.environ['HOME']).absolute() / 'gap-folder-B'
    root_b.mkdir(mode=0o700)
    binding_b = registry_b.add_folder_binding('gap', root_b)
    registry_b.save_runtime_binding(replace(binding_b, binding_id=binding_a.binding_id))
    registry_b.set_change_review_enabled('gap', True)
    expected_consent_b = registry_b.read_change_review_consent('gap')
    assert expected_consent_b.state is consent_module.ChangeReviewState.ENABLED
    foreign.close()
    service = inspector if case.startswith('inspector') else consent
    owner = Inspector if case.startswith('inspector') else Consent
    original_lookup = inspect.getattr_static(owner, '__getattribute__')
    assert original_lookup is object.__getattribute__ and '_registry' not in vars(owner)
    body_codes = (Inspector._current_scope.__code__, Consent.status.__code__, Consent._current_bindings.__code__)
    lookup_calls = []

    def selection(frame):
        if frame.f_code in body_codes:
            assert frame.f_globals is (inspector_module.__dict__ if frame.f_code is Inspector._current_scope.__code__ else consent_module.__dict__)
            instruction = next((item for item in dis.get_instructions(frame.f_code) if item.offset > frame.f_lasti), None)
            if instruction is not None and instruction.argval in {'get_workspace', 'get_runtime_binding', 'read_change_review_consent', 'list_folder_bindings'}:
                lookup_calls.append('body-B')
                return registry_b
        lookup_calls.append('capture-or-check-A')
        return registry

    def lookup(self, name):
        if self is service and name == '_registry':
            return selection(sys._getframe(1))
        return original_lookup(self, name)

    def property_getter(self):
        assert self is service
        return selection(sys._getframe(1))

    if case.endswith('_lookup'):
        owner.__getattribute__ = lookup
    elif case.endswith('_descriptor'):
        owner._registry = property(property_getter)
    actor = None
    armed = False
    local_a = local_b = connection_a = connection_b = provenance_a = None
    opens_a, opens_b, swaps, query_receivers, invalid = [], [], [], [], []
    before = None
    result_checked = cleanup_checked = False

    def state(repository, connection):
        participant = repository._maintenance_participant
        with storage._lock:
            lease = participant.connections.get(connection)
            return dict(closed=physically_closed(connection), registered=lease is not None,
                        live=lease is not None and lease in storage._live_leases,
                        actor_matches=lease is not None and lease.resource_thread is actor,
                        participant_matches=lease is not None and lease.resource_participant is participant,
                        path_matches=lease is not None and lease.resource_path == participant.path)

    def returned(code, offset, value):
        nonlocal connection_a, provenance_a
        if not armed:
            return
        frame = sys._getframe(1)
        try:
            assert frame.f_code is code and threading.current_thread() is actor
            if code is opener.__code__:
                receiver = frame.f_locals.get('self')
                if receiver is database or receiver is foreign:
                    assert isinstance(value, sqlite3.Connection) and not physically_closed(value)
                    (opens_a if receiver is database else opens_b).append(value)
            elif code is held_body.__code__ and frame.f_locals.get('self') is database and retirement_case and not swaps:
                assert database._thread_local is local_a and local_a.conn is value
                assert isinstance(value, sqlite3.Connection) and not physically_closed(value)
                assert state(database, value) == dict(closed=False, registered=True, live=True, actor_matches=True, participant_matches=True, path_matches=True)
                connection_a = value
                participant = database._maintenance_participant
                with storage._lock:
                    provenance_a = participant, participant.connections[value]
                database._thread_local = local_b
                swaps.append('original-held-return-A-to-B')
            elif code in query_codes:
                receiver = frame.f_locals.get('self')
                if receiver is registry or receiver is registry_b:
                    query_receivers.append('A' if receiver is registry else 'B')
        except BaseException as error:
            invalid.append(type(error).__name__)

    monitoring = sys.monitoring
    tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(tool, 'workspace-captured-resource-lookup-faults')
    query_codes = {method.__code__ for method in query_methods.values()}
    selected = (opener.__code__, held_body.__code__, *query_codes)
    monitoring.register_callback(tool, monitoring.events.PY_RETURN, returned)
    for code in selected:
        monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
    assert monitoring.get_events(tool) == 0

    def worker():
        nonlocal actor, armed, local_a, local_b, connection_b, before, result_checked, cleanup_checked
        actor = threading.current_thread()
        local_a, local_b = database._thread_local, threading.local()
        assert getattr(local_a, 'conn', None) is None
        if case == 'new_foreign':
            connection_b = foreign._held_connection()
            local_b.conn = connection_b
            assert state(foreign, connection_b)['live']
        if case == 'borrowed_empty':
            borrowed = database._held_connection()
            assert local_a.conn is borrowed and not physically_closed(borrowed)
        try:
            armed = True
            if retirement_case:
                try:
                    consent.status('gap')
                except RuntimeError as error:
                    assert str(error) == 'workspace_composite_source_changed'
                else:
                    raise AssertionError('thread-local swap was accepted')
                assert len(swaps) == 1 and connection_a is not None and database._thread_local is local_b
                before = state(database, connection_a)
                before['A_cache_exact_handle'] = local_a.conn is connection_a
                before['B_cache_unchanged'] = getattr(local_b, 'conn', None) is connection_b
                if connection_b is not None:
                    before['foreign_B'] = state(foreign, connection_b)
            elif case.startswith('consent'):
                result = consent.status('gap')
                assert result.capability.state is consent_module.ChangeReviewState.ENABLED
                assert result.consent == expected_consent_b and result.roots == ()
                assert 'body-B' in lookup_calls
            else:
                result = inspector.capture_binding('gap', binding_a.binding_id)
                assert result is not None and result.canonical_root == str(root_b.resolve())
                actual = os.lstat(root_b)
                assert (result.root_device, result.root_inode) == (actual.st_dev, actual.st_ino)
                assert 'body-B' in lookup_calls
            result_checked = True
        finally:
            armed = False
            # Only test-owned native cleanup AFTER the captured product boundary;
            # no restoration or adoption of either thread-local/cache field.
            if connection_a is not None and not physically_closed(connection_a):
                participant, lease = provenance_a
                with storage._lock:
                    assert participant is database._maintenance_participant
                    assert participant.connections.get(connection_a) is lease and lease in storage._live_leases
                    assert lease.resource_thread is actor and lease.resource_participant is participant
                    assert lease.resource_path == participant.path
                with participants._core_closing(database, connection_a) as allowed:
                    assert allowed
                    connection_a.close()
            if connection_b is not None:
                foreign.close()
            if not retirement_case:
                database.close()
                foreign.close()
            handles = [*opens_a, *opens_b, *([connection_a] if connection_a is not None else []), *([connection_b] if connection_b is not None else [])]
            assert all(physically_closed(handle) for handle in handles)
            cleanup_checked = True

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='workspace-gap') as executor:
            executor.submit(worker).result(timeout=20)
        assert actor is not None and not actor.is_alive()
    finally:
        armed = False
        try:
            for code in selected:
                monitoring.set_local_events(tool, code, 0)
            assert monitoring.register_callback(tool, monitoring.events.PY_RETURN, None) is returned
            assert monitoring.get_events(tool) == 0
        finally:
            monitoring.free_tool_id(tool)
            if case.endswith('_lookup'):
                del owner.__getattribute__
                assert inspect.getattr_static(owner, '__getattribute__') is original_lookup
            elif case.endswith('_descriptor'):
                del owner._registry
            consent.shutdown()
            database.close()
            foreign.close()
    assert not invalid, invalid
    assert result_checked and cleanup_checked
    with storage._lock:
        leases = sum(lease.resource_thread is actor for lease in storage._live_leases)
        registered = sum(lease.resource_thread is actor for repository in (database, foreign) for lease in repository._maintenance_participant.connections.values())
    assert leases == 0 and registered == 0
    assert all(path.read_bytes() == raw for path, raw in sources.items())
    assert all(module.__file__ == saved[0] and module.__spec__ is saved[1] and module.__spec__.origin == saved[2] for module, saved in origins.items())
    assert all(function.__code__ is code and function.__globals__ is defining and function.__defaults__ is defaults and function.__kwdefaults__ is kwdefaults and function.__closure__ is closure for function, code, defining, defaults, kwdefaults, closure in (*records, *guarded_records))
    assert all(module.__dict__[name] is function for module, name, function in guarded_bindings)
    assert all(inspect.getattr_static(WorkspaceDB, name) is descriptor for name, descriptor in descriptors)
    assert not network_guard.blocked_attempts() and not real_profile_guard._violations
    assert monitoring.get_events(tool) == 0 and monitoring.get_tool(tool) is None
    receipt = dict(case=case, before_test_owned_cleanup=before, new_A_handles=len(opens_a), new_B_handles=len(opens_b), query_receivers=query_receivers, lookup_calls=lookup_calls, result_checked=result_checked, physical_fixture_cleanup_checked=cleanup_checked, worker_leases=leases, worker_registered=registered, source_hashes={str(path): hashlib.sha256(raw).hexdigest() for path, raw in sources.items()}, invalid=invalid, global_events=0, tool_retired=True, worker_retired=not actor.is_alive())
    (root / ('workspace-gap-' + case + '.receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: receipt[key] for key in ('case', 'new_A_handles', 'new_B_handles', 'worker_leases', 'tool_retired')}, sort_keys=True))
    if retirement_case:
        assert before['B_cache_unchanged'], 'replacement/foreign B cache was modified'
        if connection_b is not None:
            assert before['foreign_B'] == dict(closed=False, registered=True, live=True, actor_matches=True, participant_matches=True, path_matches=True), 'foreign B borrower was retired'
        if case == 'borrowed_empty':
            assert not before['closed'] and before['registered'] and before['live'] and before['A_cache_exact_handle'], 'A borrower was retired'
        else:
            assert before['closed'] and not before['registered'] and not before['live'], 'new captured A native handle survived source-refused composite'
    else:
        assert not opens_a, 'custom service lookup incorrectly qualified added stock A interval'
        assert len(opens_b) == 2 and query_receivers == ['B', 'B'], 'preceding custom B-reader contract changed'
    print('retired and reopened')


with user_fixture_default_owner():
    main()

"""


@pytest.mark.parametrize(
    "case",
    [
        "new_empty",
        "new_foreign",
        "borrowed_empty",
        "consent_lookup",
        "consent_descriptor",
        "inspector_lookup",
        "inspector_descriptor",
    ],
)
def test_workspace_composite_captured_retirement_and_lookup(tmp_path, case):
    _run(tmp_path, case, "ownership", script=_SCRIPT)
