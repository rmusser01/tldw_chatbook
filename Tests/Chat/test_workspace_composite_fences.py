"""Bounded genuine SQLite controls for the finite stock Workspace composite."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import ast
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
route, ownership = ('save_binding' if case in {'write_retarget', 'helper_drift'} else 'status'), 'cold'
assert case in {'custom_reader', 'subclass', 'db_retarget', 'body_drift', 'missing_table', 'write_retarget', 'helper_drift'}
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
    register = participants._register_core_connection
    query_names = ('get_workspace', 'read_change_review_consent',
                   'list_runtime_bindings', 'get_runtime_binding')
    query_methods = {name: Registry.__dict__[name] for name in query_names}
    methods = (
        opener, register, *query_methods.values(), Registry.save_runtime_binding,
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
        (participants, '_core_operation'), (participants, '_core_getter'),
        (participants, '_core_transaction'), (private_sqlite, 'connect_private_sqlite'),
        (base_db, 'operation_owned_connection'),
    ))
    guarded_records = tuple((method, method.__code__, method.__globals__,
                             method.__defaults__, method.__kwdefaults__, method.__closure__)
                            for module, name, method in guarded_bindings)
    descriptors = tuple((name, WorkspaceDB.__dict__[name]) for name in
                        ('_get_connection', '_held_connection', 'connection', 'transaction', 'close'))

    database = WorkspaceDB(config.get_user_data_dir() / 'composite-owned.sqlite',
                           client_id='finite-composite')
    selected_registry_type = Registry
    custom_calls = []
    if case == 'subclass':
        class CustomRegistry(Registry):
            def read_change_review_consent(self, workspace_id):
                custom_calls.append('subclass')
                value = super().read_change_review_consent(workspace_id)
                return consent_module.ChangeReviewConsent(value.state, 'custom-literal-revision')
        selected_registry_type = CustomRegistry
    registry = selected_registry_type(database)
    registry.create_workspace(workspace_id='composite-owned', name='Composite owned',
                              description='seeded', assistant_defaults=None)
    project = Path(os.environ['HOME']).absolute() / 'project-folder'
    project.mkdir(mode=0o700)
    binding = (None if route == 'admit'
               else registry.add_folder_binding('composite-owned', project))
    registry.set_change_review_enabled('composite-owned', True)
    consent = Consent(registry)  # Actual default capability and initializer callbacks.
    inspector = Inspector(registry)
    assert consent_module._default_capability_reader().state is consent_module.ChangeReviewState.ENABLED
    database.close()
    assert getattr(database._thread_local, 'conn', None) is None
    if case == 'missing_table':
        with database.transaction() as connection:
            connection.execute('DROP TABLE workspace_change_review')
        database.close()
    other_database = None
    if case in {'db_retarget', 'write_retarget'}:
        other_database = WorkspaceDB(config.get_user_data_dir() / 'foreign-owned.sqlite', client_id='other-owner')
        other_registry = Registry(other_database)
        other_registry.create_workspace(workspace_id='composite-owned', name='Foreign literal', assistant_defaults=None)
        other_registry.set_change_review_enabled('composite-owned', True)
        other_database.close()
    if case == 'custom_reader':
        from types import MethodType
        def custom_reader(self, workspace_id):
            custom_calls.append('instance')
            value = query_methods['read_change_review_consent'](self, workspace_id)
            return consent_module.ChangeReviewConsent(value.state, 'custom-literal-revision')
        registry.read_change_review_consent = MethodType(custom_reader, registry)
    switched = []
    changed_body_calls = []
    changed_target = (registry_module._metadata_to_json if case == 'helper_drift' else Registry.list_runtime_bindings)
    original_changed_body = changed_target.__code__
    if case in {'body_drift', 'helper_drift'}:
        defining = changed_target.__globals__
        assert '_workspace_test_changed_calls' not in defining
        defining['_workspace_test_changed_calls'] = changed_body_calls
        exec('def _workspace_test_changed_body(self, workspace_id):\n'
             '    _workspace_test_changed_calls.append(True)\n'
             '    return ()\n', defining)
        changed_body = defining.pop('_workspace_test_changed_body').__code__
        if case == 'helper_drift':
            exec('def _workspace_test_changed_helper(value):\n'
                 '    _workspace_test_changed_calls.append(True)\n'
                 '    return "{}"\n', defining)
            changed_body = defining.pop('_workspace_test_changed_helper').__code__

    save_tree = ast.parse(sources[Path(registry_module.__file__).resolve()])
    save_owner = next(node for node in save_tree.body if isinstance(node, ast.ClassDef) and node.name == 'LocalWorkspaceRegistryService')
    save_node = next(node for node in save_owner.body if isinstance(node, ast.FunctionDef) and node.name == 'save_runtime_binding')
    write_call = next(node for node in ast.walk(save_node) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == 'execute'
                      and isinstance(node.func.value, ast.Name) and node.func.value.id == 'conn')
    write_args_seen = []
    actor = None
    armed = False
    invalid = []
    opened = []
    registered = []
    reads = []
    borrowed = None
    boundary = None
    result_checked = False
    preserved = False
    calls = {name: 0 for name in query_names}
    code_names = {method.__code__: name for name, method in query_methods.items()}

    def started(code, offset):
        if not armed or code not in code_names:
            return
        frame = sys._getframe(1)
        if frame.f_code is code and frame.f_locals.get('self') is registry:
            if threading.current_thread() is not actor:
                invalid.append('query_actor_changed')
            calls[code_names[code]] += 1

    def returned(code, offset, value):
        if not armed:
            return
        frame = sys._getframe(1)
        try:
            assert frame.f_code is code
            if code is opener.__code__:
                if frame.f_locals.get('self') is not database:
                    return
                assert threading.current_thread() is actor
                assert isinstance(value, sqlite3.Connection) and not physically_closed(value)
                assert all(connection is not value for connection in opened)
                opened.append(value)  # Original physical connector returned a new handle.
            elif code is register.__code__:
                if frame.f_locals.get('repository') is not database:
                    return
                assert threading.current_thread() is actor
                connection = frame.f_locals['connection']
                assert value is connection and isinstance(connection, sqlite3.Connection)
                participant = database._maintenance_participant
                with storage._lock:
                    lease = participant.connections.get(connection)
                    assert lease in storage._live_leases and lease.resource_thread is actor
                    assert lease.resource_participant is participant
                    assert lease.resource_path == participant.path
                registered.append((connection, participant, lease))
            elif code in code_names and frame.f_locals.get('self') is registry:
                assert threading.current_thread() is actor
                connection = frame.f_locals.get('conn')
                assert isinstance(connection, sqlite3.Connection)
                assert connection is borrowed or any(connection is item for item in opened)
                reads.append((code_names[code], connection))
                if code is query_methods['read_change_review_consent'].__code__ and not switched:
                    if case in {'db_retarget', 'write_retarget'}:
                        registry.db = other_database
                        switched.append('database')
                    elif case in {'body_drift', 'helper_drift'}:
                        Registry.list_runtime_bindings.__code__ = changed_body
                        switched.append('body')
                if case == 'helper_drift' and code is query_methods['get_workspace'].__code__ and not switched:
                    changed_target.__code__ = changed_body
                    switched.append('helper')
                if code is query_methods['list_runtime_bindings'].__code__:
                    assert isinstance(value, tuple)
                    if route == 'admit':
                        assert value == ()
                    else:
                        assert len(value) == 1 and value[0].binding_id == binding.binding_id
                        assert value[0].label == 'project-folder'
        except BaseException as error:
            invalid.append(type(error).__name__)

    def lined(code, line):
        if not armed or case != 'write_retarget' or code is not Registry.save_runtime_binding.__code__:
            return
        frame = sys._getframe(1)
        if frame.f_code is not code or frame.f_locals.get('self') is not registry:
            return
        try:
            assert threading.current_thread() is actor
            if write_call.lineno <= line <= write_call.end_lineno:
                write_args_seen.append(True)
                return
            connection = frame.f_locals.get('conn')
            if write_args_seen and not switched and isinstance(connection, sqlite3.Connection):
                # First subsequent actual LINE after the original execute:
                # the new inside-body check, or the old context exit.
                assert sqlite3.Connection.in_transaction.__get__(connection)
                assert any(connection is item for item in opened)
                registry.db = other_database
                switched.append('inside_write_transaction')
        except BaseException as error:
            invalid.append('write_boundary_' + type(error).__name__)

    monitoring = sys.monitoring
    tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(tool, 'workspace-cold-finite-composite')
    monitoring.register_callback(tool, monitoring.events.PY_START, started)
    monitoring.register_callback(tool, monitoring.events.PY_RETURN, returned)
    selected = (opener.__code__, register.__code__, *code_names)
    for code in selected:
        monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN |
                                    (monitoring.events.PY_START if code in code_names else 0))
    monitoring.register_callback(tool, monitoring.events.LINE, lined)
    monitoring.set_local_events(tool, Registry.save_runtime_binding.__code__, monitoring.events.LINE)
    assert monitoring.get_events(tool) == 0

    def snapshot():
        participant = database._maintenance_participant
        connections = opened if ownership == 'cold' else [borrowed]
        with storage._lock:
            return [dict(physically_closed=physically_closed(connection),
                         cached=getattr(database._thread_local, 'conn', None) is connection,
                         registered=connection in participant.connections,
                         lease_live=participant.connections.get(connection) in storage._live_leases)
                    for connection in connections]

    def worker():
        nonlocal actor, armed, borrowed, boundary, result_checked, preserved
        actor = threading.current_thread()
        assert getattr(database._thread_local, 'conn', None) is None
        try:
            with database.transaction() if ownership == 'borrowed' else nullcontext() as transaction:
                if ownership == 'borrowed':
                    borrowed = transaction
                    assert isinstance(borrowed, sqlite3.Connection)
                    assert sqlite3.Connection.in_transaction.__get__(borrowed)
                    borrowed.execute('UPDATE workspace_records SET description = ? WHERE workspace_id = ?',
                                     ('borrowed-uncommitted', 'composite-owned'))
                armed = True
                with base_db.operation_owned_connection(database):
                    if case in {'db_retarget', 'body_drift', 'write_retarget', 'helper_drift'}:
                        try:
                            if case in {'write_retarget', 'helper_drift'}:
                                registry.save_runtime_binding(replace(binding, label='Updated literal'))
                            else:
                                consent.status('composite-owned')
                        except RuntimeError as error:
                            assert str(error) == 'workspace_composite_source_changed'
                        else:
                            raise AssertionError('changed receiver/body accepted before next query')
                        assert switched and not changed_body_calls
                        assert calls['list_runtime_bindings'] == 0, 'foreign/changed next reader was entered'
                    else:
                        result = consent.status('composite-owned')
                        assert result.capability.state is consent_module.ChangeReviewState.ENABLED
                        expected_state = (consent_module.ChangeReviewState.UNAVAILABLE if case == 'missing_table'
                                          else consent_module.ChangeReviewState.ENABLED)
                        assert result.consent.state is expected_state
                        if case in {'custom_reader', 'subclass'}:
                            assert result.consent.revision == 'custom-literal-revision'
                            assert len(custom_calls) == 1
                        assert result.roots == () and not consent._workers_started
                    result_checked = True
                armed = False
                boundary = snapshot()  # Before any test-owned physical cleanup.
                if ownership == 'borrowed':
                    assert database._thread_local.conn is borrowed and not physically_closed(borrowed)
                    assert sqlite3.Connection.in_transaction.__get__(borrowed)
                    row = borrowed.execute('SELECT description FROM workspace_records WHERE workspace_id = ?',
                                           ('composite-owned',)).fetchone()
                    assert row[0] == 'borrowed-uncommitted'
                    with storage._lock:
                        lease = database._maintenance_participant.connections.get(borrowed)
                        assert lease in storage._live_leases and lease.resource_thread is actor
                    preserved = True
                    borrowed.rollback()  # Test-owned borrower cleanup only after preservation proof.
        finally:
            armed = False
            database.close()  # Original worker closes only its own cache.
            if other_database is not None:
                foreign = getattr(other_database._thread_local, 'conn', None)
                other_database.close()  # Only this test-owned receiver on its actual worker.
                assert foreign is None or physically_closed(foreign)
            if case in {'body_drift', 'helper_drift'}:
                changed_target.__code__ = original_changed_body
                changed_target.__globals__.pop('_workspace_test_changed_calls')
            if case in {'db_retarget', 'write_retarget'}:
                registry.db = database
            if case == 'custom_reader':
                del registry.read_change_review_consent

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='workspace-composite') as executor:
            executor.submit(worker).result(timeout=20)
        assert actor is not None and not actor.is_alive()
    finally:
        armed = False
        try:
            for code in selected:
                monitoring.set_local_events(tool, code, 0)
            assert monitoring.register_callback(tool, monitoring.events.PY_START, None) is started
            assert monitoring.register_callback(tool, monitoring.events.PY_RETURN, None) is returned
            monitoring.set_local_events(tool, Registry.save_runtime_binding.__code__, 0)
            assert monitoring.register_callback(tool, monitoring.events.LINE, None) is lined
            assert monitoring.get_events(tool) == 0
        finally:
            monitoring.free_tool_id(tool)
            consent.shutdown()
            database.close()  # Creator's already closed cache, never a foreign-worker close.


    if case in {'db_retarget', 'write_retarget'}:
        other_database.close()
    assert not invalid, invalid
    expected_calls = dict(get_workspace=0, read_change_review_consent=1,
                          list_runtime_bindings=(1 if case in {'custom_reader', 'subclass'} else 0),
                          get_runtime_binding=0)
    if case in {'write_retarget', 'helper_drift'}:
        expected_calls = dict(get_workspace=1, read_change_review_consent=0,
                              list_runtime_bindings=0, get_runtime_binding=0)
    assert calls == expected_calls, calls
    assert reads and result_checked and boundary is not None
    if ownership == 'cold':
        assert opened and len(registered) == len(opened)
        assert all(row['physically_closed'] and not row['cached'] and
                   not row['registered'] and not row['lease_live'] for row in boundary), boundary
    else:
        assert borrowed is not None and preserved and not opened
        assert boundary == [dict(physically_closed=False, cached=True,
                                 registered=True, lease_live=True)], boundary
    connections = opened if ownership == 'cold' else [borrowed]
    assert all(physically_closed(connection) for connection in connections)
    with storage._lock:
        worker_leases = sum(lease.resource_thread is actor for lease in storage._live_leases)
        worker_registered = sum(lease.resource_thread is actor
                                for lease in database._maintenance_participant.connections.values())
    assert worker_leases == 0 and worker_registered == 0

    if route == 'save_binding':
        # Independent original read after worker retirement proves the real write persisted.
        persisted = registry.get_runtime_binding(binding.binding_id)
        verify_connection = database._thread_local.conn
        assert persisted.label == 'project-folder', 'stale-owner write committed instead of rollback'
        assert isinstance(verify_connection, sqlite3.Connection)
        database.close()
        assert physically_closed(verify_connection)
    assert all(path.read_bytes() == raw for path, raw in sources.items())
    assert all(module.__file__ == saved[0] and module.__spec__ is saved[1]
               and module.__spec__.origin == saved[2] for module, saved in origins.items())
    assert all(method.__code__ is code and method.__globals__ is defining
               and method.__defaults__ is defaults and method.__kwdefaults__ is kwdefaults
               and method.__closure__ is closure
               for method, code, defining, defaults, kwdefaults, closure in (*records, *guarded_records))
    assert all(module.__dict__[name] is method for module, name, method in guarded_bindings)
    assert all(inspect.getattr_static(WorkspaceDB, name) is descriptor for name, descriptor in descriptors)
    assert database_module.WorkspaceDB is WorkspaceDB and type(database) is WorkspaceDB
    assert registry_module.LocalWorkspaceRegistryService is Registry and type(registry) is selected_registry_type
    guards = dict(network=len(network_guard.blocked_attempts()),
                  real_profile=len(real_profile_guard._violations))
    assert guards == dict(network=0, real_profile=0)
    receipt = dict(case=case, route=route, ownership=ownership, physical_new_handles=len(opened),
                   registered_new_handles=len(registered), calls=calls,
                   query_handle_count=len({id(connection) for name, connection in reads}),
                   boundary=boundary, borrowed_preserved=preserved, result_checked=result_checked,
                   all_handles_physically_closed_after_owned_cleanup=True,
                   worker_positively_retired=not actor.is_alive(), worker_leases=worker_leases,
                   worker_registered=worker_registered, global_events=monitoring.get_events(tool),
                   monitoring_retired=monitoring.get_tool(tool) is None, invalid=invalid, guard_counts=guards,
                   source_hashes={str(path): hashlib.sha256(raw).hexdigest()
                                  for path, raw in sources.items()})
    (root / ('workspace-composite-control-' + case + '.receipt.json')).write_text(
        json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: receipt[key] for key in
                      ('route', 'ownership', 'physical_new_handles', 'query_handle_count',
                       'worker_leases', 'monitoring_retired')}, sort_keys=True))
    # Sole expected cold RED is reached only after source/results/physical retirement qualify.
    expected_handles = 2 if case in {'custom_reader', 'subclass'} else 1
    assert len(opened) == expected_handles, (case, len(opened))
    assert len({id(connection) for name, connection in reads}) == expected_handles
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize(
    "case",
    [
        "custom_reader",
        "subclass",
        "db_retarget",
        "body_drift",
        "missing_table",
        "write_retarget",
        "helper_drift",
    ],
)
def test_workspace_composite_keeps_fallback_and_source_fences(tmp_path, case):
    _run(tmp_path, case, "ownership", script=_SCRIPT)
