"""Real two-scope browser work count; no App, fake DB or replaced guard."""

import ast
import subprocess
import sys

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
from types import CodeType, FunctionType, MethodType, SimpleNamespace

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()

route, outcome = sys.argv[1:]
assert route == 'browser_two_scope'
assert outcome in {
    'cold', 'borrowed', 'instance', 'subclass', 'memory', 'lookup',
    'error_first', 'error_second', 'preinstalled_body',
    'retarget_database', 'retarget_app_service', 'midpoint_body', 'admission_body',
    'cache_local_empty', 'cache_local_foreign', 'cache_conn_foreign',
    'body_error_close_refused',
}


def shape(code):
    return (code.co_name, code.co_qualname, code.co_firstlineno,
            code.co_code, code.co_exceptiontable, code.co_stacksize,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_nlocals, code.co_flags, code.co_names, code.co_varnames,
            code.co_freevars, code.co_cellvars,
            tuple(shape(item) if isinstance(item, CodeType) else item
                  for item in code.co_consts))


def nested(code, qualname):
    if code.co_qualname == qualname:
        return code
    for item in code.co_consts:
        if isinstance(item, CodeType):
            result = nested(item, qualname)
            if result is not None:
                return result
    return None


def closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError as error:
        assert 'closed' in str(error).lower()
        return True
    return False


def main():
    selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    root = Path(os.environ['XDG_DATA_HOME']).absolute()
    selector.write_text('[general]\nusers_name="qualifier"\n[paths]\ndata_dir="'
                        + root.as_posix() + '"\n', encoding='utf-8')
    selector.chmod(0o600)
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import participants
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB import ChaChaNotes_DB as notes_module
    from tldw_chatbook.DB import base_db, private_sqlite
    from tldw_chatbook.DB import private_sqlite_process as helper_module
    from tldw_chatbook.Chat import chat_conversation_service as service_module
    from tldw_chatbook.UI.Console_Modules import workspace as workspace_module
    from Tests.UI import test_console_workspace_controller as fixture_module

    Database = notes_module.CharactersRAGDB
    Service = service_module.ChatConversationService
    Controller = workspace_module.ConsoleWorkspaceController
    list_method = inspect.getattr_static(Service, 'list_conversations')
    browser = inspect.getattr_static(Controller, '_persisted_console_browser_rows')
    register = participants._register_core_connection
    getter_wrapper = inspect.getattr_static(Database, '_get_thread_connection')
    assert type(getter_wrapper) is FunctionType
    getter = vars(getter_wrapper)['__wrapped__']
    assert type(getter) is FunctionType
    assert getter_wrapper.__globals__ is participants.__dict__
    assert getter_wrapper.__code__.co_freevars == ('function',)
    assert getter_wrapper.__closure__ is not None and len(getter_wrapper.__closure__) == 1
    assert getter_wrapper.__closure__[0].cell_contents is getter
    assert getter.__globals__ is notes_module.__dict__ and getter.__closure__ is None
    assert notes_module._register_core_connection is register
    helper = inspect.getattr_static(private_sqlite.HelperLease, 'start').__func__
    assert private_sqlite.HelperLease is helper_module.HelperLease
    assert helper.__globals__ is helper_module.__dict__
    close_method = inspect.getattr_static(Database, 'close_connection')
    core_close_body = vars(participants._core_closing)['__wrapped__']
    assert type(core_close_body) is FunctionType and core_close_body.__globals__ is participants.__dict__
    assert notes_module._core_closing is participants._core_closing
    registry_type = base_db.SQLiteConnectionQuiescenceRegistry
    registry_unregister = inspect.getattr_static(registry_type, 'unregister')
    methods = (list_method, browser, getter, register, helper, close_method, core_close_body, registry_unregister,
               fixture_module._workspace_controller)
    modules = (config, participants, storage, notes_module, base_db, private_sqlite, helper_module,
               service_module, workspace_module, fixture_module)
    sources, origins, compiled = {}, {}, {}
    for module in modules:
        path = Path(module.__file__).resolve()
        assert path == Path(module.__spec__.origin).resolve()
        assert path.is_relative_to(Path.cwd().resolve())
        raw = path.read_bytes()
        sources[path] = raw
        origins[module] = (module.__file__, module.__spec__, module.__spec__.origin)
        compiled[module.__dict__['__name__']] = compile(raw, str(path), 'exec', dont_inherit=True)
    for method in methods:
        assert type(method) is FunctionType
        declared = nested(compiled[method.__globals__['__name__']], method.__code__.co_qualname)
        assert declared is not None and shape(declared) == shape(method.__code__)
    declared_wrapper = nested(compiled[participants.__name__], '_core_getter.<locals>.accessed')
    assert declared_wrapper is not None and shape(declared_wrapper) == shape(getter_wrapper.__code__)
    notes_tree = ast.parse(sources[Path(notes_module.__file__).resolve()])
    database_node = next(node for node in notes_tree.body if isinstance(node, ast.ClassDef)
                         and node.name == 'CharactersRAGDB')
    getter_node = next(node for node in database_node.body if isinstance(node, ast.FunctionDef)
                       and node.name == '_get_thread_connection')
    assert len(getter_node.decorator_list) == 1
    assert isinstance(getter_node.decorator_list[0], ast.Name)
    assert getter_node.decorator_list[0].id == '_core_getter'
    records = tuple((fn, fn.__code__, fn.__globals__, fn.__defaults__, fn.__kwdefaults__,
                     fn.__closure__, tuple(cell.cell_contents for cell in fn.__closure__ or ()))
                    for fn in (*methods, getter_wrapper))
    guarded_bindings = tuple((module, name, module.__dict__[name]) for module, name in (
        (storage, '_acquire_storage'), (storage, '_check_operation'),
        (participants, '_core_operation'), (participants, '_core_getter'),
        (participants, '_core_transaction'), (participants, '_core_closing'),
        (private_sqlite, 'connect_private_sqlite'),
        (base_db, 'operation_owned_connection'),
    ))
    guarded_records = tuple((fn, fn.__code__, fn.__globals__, fn.__defaults__, fn.__kwdefaults__,
                            fn.__closure__) for _, _, fn in guarded_bindings)

    database = Database(':memory:' if outcome == 'memory' else root / 'browser-a.sqlite',
                        client_id='browser-finite')
    foreign = Database(root / 'browser-b.sqlite', client_id='browser-foreign')
    source_service = Service(database)
    foreign_service = Service(foreign)
    global_id = source_service.create_conversation(title='Global Native', scope_type='global')
    default_id = source_service.create_conversation(title='Default Native', scope_type='workspace',
                                                    workspace_id='workspace-default')
    foreign_service.create_conversation(title='Foreign Native', scope_type='global')
    seed = getattr(database._local, 'conn', None)
    assert isinstance(seed, sqlite3.Connection) and not closed(seed)
    if outcome != 'memory':
        database.close_connection()
        assert closed(seed) and getattr(database._local, 'conn', None) is None
    foreign.close_connection()
    assert getattr(foreign._local, 'conn', None) is None

    custom_calls = []
    clone = FunctionType(list_method.__code__, list_method.__globals__,
                         list_method.__name__, list_method.__defaults__, list_method.__closure__)
    clone.__kwdefaults__ = (None if list_method.__kwdefaults__ is None
                           else dict(list_method.__kwdefaults__))

    def custom(self, query=None, **kwargs):
        custom_calls.append(kwargs['scope_type'])
        result = clone(self, query, **kwargs)
        if outcome == 'error_first' or (outcome in {'error_second', 'body_error_close_refused'} and len(custom_calls) == 2):
            raise RuntimeError('declared reader failed after real SQL')
        return result

    class SubclassService(Service):
        def list_conversations(self, query=None, **kwargs):
            return custom(self, query, **kwargs)

    service = SubclassService(database) if outcome == 'subclass' else source_service
    if outcome in {'instance', 'error_first', 'error_second', 'body_error_close_refused'}:
        service.list_conversations = MethodType(custom, service)
    original_lookup = vars(Service).get('__getattribute__')
    if outcome == 'lookup':
        def declared_lookup(self, name):
            if name == 'db':
                return foreign
            return object.__getattribute__(self, name)
        Service.__getattribute__ = declared_lookup
    test_global_names = ('_BROWSER_TEST_CLONE', '_BROWSER_TEST_BODY_CALLS', '_BROWSER_TEST_CHANGED')
    assert not any(name in service_module.__dict__ for name in test_global_names)
    body_calls = []
    # Only this unguarded service body is deliberately customized; every
    # storage/native/config/getter/transaction callback retains its original.
    exec('def _BROWSER_TEST_CHANGED(self, query=None, **kwargs):\n'
         '    _BROWSER_TEST_BODY_CALLS.append(True)\n'
         '    return _BROWSER_TEST_CLONE(self, query, **kwargs)\n',
         service_module.__dict__)
    changed = service_module.__dict__.pop('_BROWSER_TEST_CHANGED')
    service_module.__dict__['_BROWSER_TEST_CLONE'] = clone
    service_module.__dict__['_BROWSER_TEST_BODY_CALLS'] = body_calls
    saved_list_code = list_method.__code__
    if outcome == 'preinstalled_body':
        list_method.__code__ = changed.__code__
    app = SimpleNamespace(local_chat_conversation_service=service)
    controller = fixture_module._workspace_controller(app_instance=app)
    assert type(controller) is Controller

    armed = False
    main_actor = threading.current_thread()
    actors, queries, registered, getter_values, helpers, invalid = [], [], [], [], [], []
    mutated = False
    borrower = None
    borrower_context = None
    borrower_preserved = False
    result = None
    boundary = None
    cache_routes = {'cache_local_empty', 'cache_local_foreign', 'cache_conn_foreign'}
    captured_local = database._local
    foreign_local = foreign._local
    captured_a = None
    cache_boundary = None
    foreign_borrower = None
    retirement_refusals = []
    explicit_close_refusal_before = None
    selected = {saved_list_code, changed.__code__, getter.__code__, register.__code__, helper.__code__, core_close_body.__code__}
    monitoring = sys.monitoring
    tool = next(value for value in range(6) if monitoring.get_tool(value) is None)

    def started(code, offset):
        if not armed:
            return
        frame = sys._getframe(1)
        try:
            assert frame.f_code is code
            if code in {saved_list_code, changed.__code__} and frame.f_locals.get('self') is service:
                actor = threading.current_thread()
                if all(actor is not value for value in actors):
                    actors.append(actor)
                if code is saved_list_code:
                    assert frame.f_locals['limit'] == 75
                    assert frame.f_locals['archive_scope'] == 'active'
                    assert frame.f_locals['include_deleted'] is False
                    queries.append((vars(service)['db'], frame.f_locals['scope_type'],
                                    frame.f_locals['workspace_id'], frame.f_locals['query'],
                                    frame.f_locals['character_scope'], frame.f_locals['offset']))
            elif code is helper.__code__:
                request = frame.f_locals['request']
                if type(request) is helper_module.PrepareRequest and object.__getattribute__(request, 'path') in (
                    database.db_path_str, foreign.db_path_str,
                ):
                    helpers.append(threading.current_thread())
        except BaseException as error:
            invalid.append(type(error).__name__)

    def returned(code, offset, value):
        nonlocal mutated, captured_a
        if not armed:
            return
        frame = sys._getframe(1)
        try:
            assert frame.f_code is code
            actor = threading.current_thread()
            if code is getter.__code__ and any(frame.f_locals.get('self') is owner for owner in (database, foreign)):
                assert isinstance(value, sqlite3.Connection) and not closed(value)
                getter_values.append((frame.f_locals['self'], actor, value))
            elif code is register.__code__ and any(frame.f_locals.get('repository') is owner for owner in (database, foreign)):
                owner = frame.f_locals['repository']
                connection = frame.f_locals['connection']
                assert value is connection and isinstance(connection, sqlite3.Connection)
                assert not closed(connection)
                if not owner.is_memory_db:
                    participant = owner._maintenance_participant
                    with storage._lock:
                        lease = participant.connections.get(connection)
                        assert lease in storage._live_leases and lease.resource_thread is actor
                        assert lease.resource_participant is participant
                        assert lease.resource_path == participant.path
                    assert all(connection is not prior[2] for prior in registered)
                    registered.append((owner, actor, connection, participant, lease))
                    if all(actor is not prior for prior in actors):
                        actors.append(actor)
                    if outcome == 'admission_body' and owner is database and not mutated:
                        assert actor is not main_actor
                        assert list_method.__code__ is saved_list_code
                        list_method.__code__ = changed.__code__
                        mutated = True
            elif code is saved_list_code and frame.f_locals.get('self') is service:
                assert isinstance(value, dict) and isinstance(value['items'], list)
                if not mutated and frame.f_locals['scope_type'] == 'global':
                    if outcome in cache_routes:
                        assert actor is not main_actor
                        assert database._local is captured_local
                        captured_a = captured_local.conn
                        assert isinstance(captured_a, sqlite3.Connection) and not closed(captured_a)
                        assert database._connection_quiescence.is_registered(captured_a)
                        participant = database._maintenance_participant
                        with storage._lock:
                            lease = participant.connections.get(captured_a)
                            assert lease in storage._live_leases and lease.resource_thread is actor
                            assert lease.resource_participant is participant and lease.resource_path == participant.path
                        assert any(owner is database and connection is captured_a and thread is actor
                                   for owner, thread, connection, _, _ in registered)
                        if outcome == 'cache_local_empty':
                            database._local = threading.local()
                        else:
                            assert foreign._local is foreign_local and foreign_local.conn is foreign_borrower
                            assert isinstance(foreign_borrower, sqlite3.Connection) and not closed(foreign_borrower)
                            assert foreign._connection_quiescence.is_registered(foreign_borrower)
                            if outcome == 'cache_local_foreign':
                                database._local = foreign_local
                            else:
                                captured_local.conn = foreign_borrower
                        mutated = True
                    elif outcome == 'retarget_database':
                        assert vars(service)['db'] is database
                        service.db = foreign
                        mutated = True
                    elif outcome == 'retarget_app_service':
                        assert app.local_chat_conversation_service is service
                        app.local_chat_conversation_service = foreign_service
                        mutated = True
                    elif outcome == 'midpoint_body':
                        assert list_method.__code__ is saved_list_code
                        list_method.__code__ = changed.__code__
                        mutated = True
        except BaseException as error:
            invalid.append(type(error).__name__)

    def yielded_close(code, offset, value):
        if not armed or outcome != 'body_error_close_refused':
            return
        frame = sys._getframe(1)
        try:
            assert frame.f_code is core_close_body.__code__ is code
            assert frame.f_globals is participants.__dict__
            if frame.f_locals.get('repository') is database and frame.f_locals.get('connection') is borrower and value is False:
                actor = threading.current_thread()
                participant = database._maintenance_participant
                with storage._lock:
                    lease = participant.connections.get(borrower)
                    assert lease in storage._live_leases and lease.resource_thread is actor
                    assert any(operation.participant is participant and operation.thread is actor for operation in storage._operations)
                assert not closed(borrower) and sqlite3.Connection.in_transaction.__get__(borrower)
                retirement_refusals.append(actor)
        except BaseException as error:
            invalid.append(type(error).__name__)

    monitoring.use_tool_id(tool, 'browser-two-scope-finite')
    monitoring.register_callback(tool, monitoring.events.PY_START, started)
    monitoring.register_callback(tool, monitoring.events.PY_RETURN, returned)
    for code in selected:
        monitoring.set_local_events(tool, code, monitoring.events.PY_START | monitoring.events.PY_RETURN)
    monitoring.register_callback(tool, monitoring.events.PY_YIELD, yielded_close)
    monitoring.set_local_events(tool, core_close_body.__code__, monitoring.events.PY_YIELD)
    assert monitoring.get_events(tool) == 0

    def snapshot():
        with storage._lock:
            return [dict(owner='a' if owner is database else 'b',
                         physically_closed=closed(connection),
                         lease_live=lease in storage._live_leases,
                         registered=connection in participant.connections)
                    for owner, actor, connection, participant, lease in registered]

    async def run(executor):
        nonlocal armed, borrower, borrower_context, borrower_preserved, result, boundary, foreign_borrower, cache_boundary, explicit_close_refusal_before
        loop = asyncio.get_running_loop()
        loop.set_default_executor(executor)
        if outcome in {'cache_local_foreign', 'cache_conn_foreign'}:
            def enter_foreign_borrower():
                nonlocal foreign_borrower
                assert foreign._local is foreign_local
                foreign_borrower = foreign.get_connection()
                assert foreign_local.conn is foreign_borrower and not closed(foreign_borrower)
                assert foreign._connection_quiescence.is_registered(foreign_borrower)
                assert foreign_borrower.execute('SELECT 1').fetchone()[0] == 1
            armed = True
            try:
                await asyncio.to_thread(enter_foreign_borrower)
            finally:
                armed = False
        if outcome in {'borrowed', 'body_error_close_refused'}:
            def enter_borrower():
                nonlocal borrower, borrower_context
                borrower_context = database.transaction()
                cursor = borrower_context.__enter__()
                borrower = database._local.conn
                assert isinstance(borrower, sqlite3.Connection)
                assert sqlite3.Connection.in_transaction.__get__(borrower)
                cursor.execute('UPDATE conversations SET title=? WHERE id=?',
                               ('Borrowed Native', global_id))
            await asyncio.to_thread(enter_borrower)
        try:
            armed = True
            result = await controller._persisted_console_browser_rows('Native')
            armed = False
            boundary = snapshot()
            if outcome in cache_routes:
                def inspect_cache_boundary():
                    assert captured_a is not None and mutated
                    participant = database._maintenance_participant
                    with storage._lock:
                        a_lease = next(item[4] for item in registered if item[0] is database and item[2] is captured_a)
                        a_live = a_lease in storage._live_leases
                        a_registered = captured_a in participant.connections
                    facts = dict(a_physically_closed=closed(captured_a), a_lease_live=a_live,
                                 a_core_registered=a_registered,
                                 a_quiescence_registered=database._connection_quiescence.is_registered(captured_a),
                                 original_local_still_selected=database._local is captured_local)
                    if foreign_borrower is not None:
                        foreign_participant = foreign._maintenance_participant
                        with storage._lock:
                            b_lease = next(item[4] for item in registered if item[0] is foreign and item[2] is foreign_borrower)
                            b_live = b_lease in storage._live_leases
                            b_registered = foreign_borrower in foreign_participant.connections
                        b_closed = closed(foreign_borrower)
                        b_usable = False
                        if not b_closed:
                            b_usable = foreign_borrower.execute('SELECT 1').fetchone()[0] == 1
                        facts.update(b_physically_closed=b_closed, b_lease_live=b_live,
                                     b_core_registered=b_registered, b_usable=b_usable,
                                     b_quiescence_registered=foreign._connection_quiescence.is_registered(foreign_borrower),
                                     b_caller_cache_preserved=getattr(foreign_local, 'conn', None) is foreign_borrower,
                                     replacement_cache_preserved=(database._local is foreign_local if outcome == 'cache_local_foreign'
                                                                  else database._local is captured_local and getattr(captured_local, 'conn', None) is foreign_borrower))
                    return facts
                cache_boundary = await asyncio.to_thread(inspect_cache_boundary)
            if outcome in {'borrowed', 'body_error_close_refused'}:
                def check_borrower():
                    nonlocal borrower_preserved, explicit_close_refusal_before
                    assert database._local.conn is borrower and not closed(borrower)
                    assert sqlite3.Connection.in_transaction.__get__(borrower)
                    assert borrower.execute('SELECT title FROM conversations WHERE id=?',
                                            (global_id,)).fetchone()[0] == 'Borrowed Native'
                    participant = database._maintenance_participant
                    with storage._lock:
                        lease = participant.connections.get(borrower)
                        assert lease in storage._live_leases and lease.resource_thread is threading.current_thread()
                    if outcome == 'body_error_close_refused':
                        # Real explicit close enters the original _core_closing body and
                        # yields False while the original caller transaction is live.
                        previous_refusals = explicit_close_refusal_before = len(retirement_refusals)
                        database.close_connection()
                        assert len(retirement_refusals) == previous_refusals + 1
                        assert retirement_refusals[-1] is threading.current_thread()
                        assert database._local.conn is borrower and not closed(borrower)
                        assert sqlite3.Connection.in_transaction.__get__(borrower)
                    borrower_preserved = True
                if outcome == 'body_error_close_refused':
                    armed = True
                    try:
                        await asyncio.to_thread(check_borrower)
                    finally:
                        armed = False
                else:
                    await asyncio.to_thread(check_borrower)
        finally:
            armed = False
            def retire_test_owners():
                nonlocal borrower_context
                if borrower_context is not None:
                    borrower_context.__exit__(RuntimeError, RuntimeError('test-owned rollback'), None)
                    borrower_context = None
                if outcome in cache_routes:
                    # Restore only fixture-owned references, after recording the production boundary.
                    # Original owners positively close their own physical handles; no foreign handle
                    # is adopted by A. A previously wrong-close B may already be closed, in which
                    # case its actual quiescence registry is unregistered only after physical proof.
                    database._local = captured_local
                    foreign._local = foreign_local
                    for owner, local in ((database, captured_local), (foreign, foreign_local)):
                        for recorded_owner, actor, connection, participant, lease in registered:
                            if recorded_owner is not owner:
                                continue
                            assert actor is threading.current_thread()
                            if not closed(connection):
                                local.conn = connection
                                owner.close_connection()
                            assert closed(connection), 'fixture-owned physical retirement failed'
                            owner._connection_quiescence.unregister(connection)
                        local.conn = None
                    assert all(not owner._connection_quiescence._connections for owner in (database, foreign))
                database.close_connection()
                foreign.close_connection()
            await asyncio.to_thread(retire_test_owners)

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='browser-finite') as executor:
            asyncio.run(run(executor))
    finally:
        armed = False
        list_method.__code__ = saved_list_code
        if 'list_conversations' in vars(service):
            del service.list_conversations
        service.db = database
        app.local_chat_conversation_service = service
        if outcome == 'lookup':
            if original_lookup is None:
                del Service.__getattribute__
            else:
                Service.__getattribute__ = original_lookup
        for name in test_global_names:
            service_module.__dict__.pop(name, None)
        try:
            for code in selected:
                monitoring.set_local_events(tool, code, 0)
            assert monitoring.register_callback(tool, monitoring.events.PY_START, None) is started
            assert monitoring.register_callback(tool, monitoring.events.PY_RETURN, None) is returned
            assert monitoring.register_callback(tool, monitoring.events.PY_YIELD, None) is yielded_close
            assert monitoring.get_events(tool) == 0
        finally:
            monitoring.free_tool_id(tool)
            database.close_connection()
            foreign.close_connection()

    assert not invalid, invalid
    assert result is not None and boundary is not None
    assert actors and getter_values
    if outcome == 'memory':
        assert actors == [threading.current_thread()]
    else:
        assert len(actors) == 1 and actors[0] is not threading.current_thread()
        assert not actors[0].is_alive()
    assert all(closed(value) for _, _, value in getter_values)
    assert all(closed(connection) for _, _, connection, _, _ in registered)
    if borrower is not None:
        assert closed(borrower)
    with storage._lock:
        leases = sum(any(lease.resource_thread is actor for actor in actors)
                     for lease in storage._live_leases)
        operations = sum(any(operation.thread is actor for actor in actors)
                         for operation in storage._operations)
        registrations = sum(any(lease.resource_thread is actor for actor in actors)
                            for owner in (database, foreign)
                            for lease in getattr(getattr(owner, '_maintenance_participant', None),
                                                 'connections', {}).values())
    assert leases == operations == registrations == 0
    assert all(path.read_bytes() == raw for path, raw in sources.items())
    assert all(module.__file__ == values[0] and module.__spec__ is values[1]
               and module.__spec__.origin == values[2] for module, values in origins.items())
    assert all(fn.__code__ is code and fn.__globals__ is defining
               and fn.__defaults__ is defaults and fn.__kwdefaults__ is kwdefaults
               and fn.__closure__ is closure
               and all(cell.cell_contents is value for cell, value in zip(closure or (), cells))
               for fn, code, defining, defaults, kwdefaults, closure, cells in records)
    assert all(module.__dict__[name] is fn for module, name, fn in guarded_bindings)
    assert all(fn.__code__ is code and fn.__globals__ is defining
               and fn.__defaults__ is defaults and fn.__kwdefaults__ is kwdefaults
               and fn.__closure__ is closure
               for fn, code, defining, defaults, kwdefaults, closure in guarded_records)
    assert vars(participants._core_closing)['__wrapped__'] is core_close_body
    assert notes_module._core_closing is participants._core_closing
    assert inspect.getattr_static(Database, 'close_connection') is close_method
    assert base_db.SQLiteConnectionQuiescenceRegistry is registry_type
    assert inspect.getattr_static(registry_type, 'unregister') is registry_unregister
    assert inspect.getattr_static(Database, '_get_thread_connection') is getter_wrapper
    assert vars(getter_wrapper)['__wrapped__'] is getter
    assert getter_wrapper.__closure__[0].cell_contents is getter
    assert type(database) is Database and notes_module.CharactersRAGDB is Database
    assert service_module.ChatConversationService is Service
    assert vars(Service).get('__getattribute__') is original_lookup
    assert private_sqlite.HelperLease is helper_module.HelperLease
    assert inspect.getattr_static(helper_module.HelperLease, 'start').__func__ is helper
    assert Controller._persisted_console_browser_rows is browser
    assert not network_guard.blocked_attempts() and not real_profile_guard._violations
    rows, total, error = result
    opened_a = [row for row in registered if row[0] is database]
    opened_b = [row for row in registered if row[0] is foreign]
    receipt = dict(route=route, outcome=outcome, original_query_count=len(queries),
                   original_query_scopes=[scope for _, scope, _, _, _, _ in queries],
                   foreign_db_field_queries=sum(owner is foreign for owner, *_ in queries),
                   physical_a_handles=len(opened_a), physical_b_handles=len(opened_b),
                   helper_starts=len(helpers), boundary=boundary,
                   custom_callback_calls=len(custom_calls), changed_body_calls=len(body_calls),
                   retarget_observed=mutated, borrowed_preserved=borrower_preserved,
                   cache_boundary=cache_boundary, actual_close_refusals=len(retirement_refusals),
                   explicit_close_refusal_before=explicit_close_refusal_before,
                   explicit_close_refusal_delta=(None if explicit_close_refusal_before is None else len(retirement_refusals) - explicit_close_refusal_before),
                   result_rows=len(rows), result_total=total, error_present=bool(error),
                   all_native_handles_closed=True, worker_leases=leases,
                   worker_registrations=registrations, worker_operations=operations,
                   monitoring_retired=monitoring.get_tool(tool) is None,
                   global_events=monitoring.get_events(tool), invalid=invalid,
                   source_hashes={str(path): hashlib.sha256(raw).hexdigest()
                                  for path, raw in sources.items()})
    (root / ('browser-two-scope-' + outcome + '.receipt.json')).write_text(
        json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: receipt[key] for key in ('outcome', 'original_query_count',
                     'physical_a_handles', 'physical_b_handles', 'helper_starts',
                     'borrowed_preserved', 'changed_body_calls', 'monitoring_retired')}, sort_keys=True))

    if outcome in cache_routes:
        assert mutated and cache_boundary is not None, 'actual original first-query return was not qualified'
        assert captured_a is not None and any(item[2] is captured_a for item in opened_a)
        if foreign_borrower is not None:
            assert len(opened_b) == 1 and opened_b[0][2] is foreign_borrower, 'actual caller B was not attributed'
        else:
            assert not opened_b
        assert cache_boundary['a_physically_closed'], 'new A survived the actual callback boundary'
        assert not cache_boundary['a_lease_live'] and not cache_boundary['a_core_registered']
        assert not cache_boundary['a_quiescence_registered']
        if foreign_borrower is not None:
            assert not cache_boundary['b_physically_closed'] and cache_boundary['b_usable'], 'caller B was closed or changed'
            assert cache_boundary['b_lease_live'] and cache_boundary['b_core_registered']
            assert cache_boundary['b_quiescence_registered'] and cache_boundary['b_caller_cache_preserved']
            assert cache_boundary['replacement_cache_preserved'], 'replacement caller cache was altered or restored'
        assert len(queries) == 1 and rows == [] and error, 'changed local/connection source escaped publication'
    elif outcome in {'retarget_database', 'retarget_app_service', 'midpoint_body', 'admission_body'}:
        assert mutated, 'original first-query return boundary never reached'
        assert not opened_b and not body_calls, 'changed owner/body executed before refusal'
        assert len(queries) <= 1, 'stock source changed before a later fresh query'
        assert rows == [] and error, 'changed stock source returned an accepted browser result'
    elif outcome == 'body_error_close_refused':
        assert borrower_preserved and retirement_refusals
        assert len(custom_calls) == len(queries) == 2 and error and total == 1
        assert [row.title for row in rows] == ['Borrowed Native']
        assert not opened_a and not opened_b, 'borrower unexpectedly acquired a new native connection'
    elif outcome in {'error_first', 'error_second'}:
        expected = 1 if outcome == 'error_first' else 2
        assert len(custom_calls) == len(queries) == expected
        assert error and total == (None if expected == 1 else 1)
        assert [row.title for row in rows] == ([] if expected == 1 else ['Global Native'])
        assert all(item['physically_closed'] and not item['lease_live'] and
                   not item['registered'] for item in boundary)
    elif outcome == 'lookup':
        assert not error and total == 1 and [row.title for row in rows] == ['Foreign Native']
        assert len(queries) == 2 and not opened_a and len(opened_b) == 2
        assert all(item['physically_closed'] and not item['lease_live'] and
                   not item['registered'] for item in boundary)
    else:
        assert not error and total == 2
        assert [row.conversation_id for row in rows] == [global_id, default_id]
        assert [row.title for row in rows] == [
            'Borrowed Native' if outcome == 'borrowed' else 'Global Native', 'Default Native']
        assert len(queries) == 2
        assert [(scope, workspace, query, character, offset) for _, scope, workspace, query, character, offset in queries] == [
            ('global', None, 'Native', 'generic', 0),
            ('workspace', 'workspace-default', 'Native', 'generic', 0)]
        if outcome == 'cold':
            assert all(item['physically_closed'] and not item['lease_live'] and
                       not item['registered'] for item in boundary)
            if os.name != 'nt':
                assert len(helpers) == len(opened_a), 'helper/native-handle attribution incomplete'
            # Sole cold count RED, after result/source/physical retirement proof.
            assert len(opened_a) <= 1, 'stock flat browser reopened physical SQLite: ' + str(len(opened_a))
        elif outcome == 'borrowed':
            assert borrower_preserved and not opened_a
        elif outcome == 'instance':
            assert custom_calls == ['global', 'workspace'] and len(opened_a) == 2
        elif outcome == 'subclass':
            assert custom_calls == ['global', 'workspace']
            assert boundary and any(not item['physically_closed'] for item in boundary)
        elif outcome == 'preinstalled_body':
            assert len(body_calls) == 2 and len(opened_a) == 2
        else:
            assert outcome == 'memory' and not registered and not helpers
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


def _compact_inline_child_script(script: str) -> str:
    """Keep the literal readable while fitting Windows' inline argv bound."""
    original = ast.parse(script)
    canonical = ast.unparse(original)
    lines = []
    for line in canonical.splitlines():
        depth = len(line) - len(line.lstrip(" "))
        assert depth % 4 == 0
        lines.append(" " * (depth // 4) + line.lstrip(" "))
    compact = "\n".join(lines)
    assert ast.dump(ast.parse(compact)) == ast.dump(original)
    command = subprocess.list2cmdline(
        [sys.executable, "-c", compact, "browser_two_scope", "body_error_close_refused"]
    )
    assert len(command.encode("utf-16-le")) // 2 + 1 <= 32767
    return compact


# The original runner still executes the complete child with its unchanged45s
# limit and argv/env/profile. Only equivalent whitespace is transported.
_SCRIPT = _compact_inline_child_script(_SCRIPT)


def test_stock_browser_two_scopes_open_one_physically_retired_handle(tmp_path):
    _run(tmp_path, "browser_two_scope", "cold", script=_SCRIPT)


def test_stock_browser_two_scopes_preserve_actual_borrowed_transaction(tmp_path):
    _run(tmp_path, "browser_two_scope", "borrowed", script=_SCRIPT)


@pytest.mark.parametrize(
    "outcome", ["instance", "subclass", "memory", "lookup", "preinstalled_body"]
)
def test_browser_declared_custom_or_memory_route_retains_original_abi(
    tmp_path, outcome
):
    _run(tmp_path, "browser_two_scope", outcome, script=_SCRIPT)


@pytest.mark.parametrize("outcome", ["error_first", "error_second"])
def test_browser_declared_reader_error_keeps_original_partial_rows(tmp_path, outcome):
    _run(tmp_path, "browser_two_scope", outcome, script=_SCRIPT)


@pytest.mark.parametrize(
    "outcome",
    ["retarget_database", "retarget_app_service", "midpoint_body", "admission_body"],
)
def test_stock_browser_midpoint_drift_refuses_before_changed_reader(tmp_path, outcome):
    _run(tmp_path, "browser_two_scope", outcome, script=_SCRIPT)


@pytest.mark.parametrize(
    "outcome", ["cache_local_empty", "cache_local_foreign", "cache_conn_foreign"]
)
def test_stock_browser_retires_captured_a_without_touching_caller_b(tmp_path, outcome):
    _run(tmp_path, "browser_two_scope", outcome, script=_SCRIPT)


def test_declared_body_error_preserves_actual_caller_after_original_close_refusal(
    tmp_path,
):
    _run(tmp_path, "browser_two_scope", "body_error_close_refused", script=_SCRIPT)
