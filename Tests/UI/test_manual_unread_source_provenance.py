"""Evidence-only draft: custom unread readers retain their worker lifetime.

No native qualification is claimed until the parent installs/runs these cases.
"""

import json

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

# Only the shared parent follows collection-time config; each real script
# selects its own profile before importing the original guarded readers.
pytestmark = pytest.mark.bootstrap_profile


_SCRIPT = r"""
import asyncio, hashlib, json, os, sqlite3, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import FunctionType, MethodType, SimpleNamespace
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]


def main():
    root = Path(os.environ['XDG_DATA_HOME']).absolute()
    selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    selector.write_text('[general]\nusers_name="qualifier"\n[paths]\ndata_dir="'
                        + root.as_posix() + '"\n', encoding='utf-8')
    selector.chmod(0o600)
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import participants
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.conversation_local_marks_service import ConversationLocalMarksService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB import base_db
    from tldw_chatbook.Utils import windows_files

    database = CharactersRAGDB(
        ':memory:' if route == 'memory' else config.get_user_data_dir() / 'unread.sqlite',
        client_id='unread-source-native')
    service = ConversationLocalMarksService(database)
    conversation_id = database.add_conversation({'title': 'Unread source lifetime'})
    service.mark_unread(conversation_id)
    initial_connection = database.get_connection()
    if route != 'memory':
        database.close_connection()
        try:
            sqlite3.Connection.in_transaction.__get__(initial_connection)
        except sqlite3.ProgrammingError:
            pass
        else:
            raise AssertionError('seeded SQLite connection remains physically open')

    source_reader = ConversationLocalMarksService.unread_ids_for
    source_code = source_reader.__code__
    source_globals = source_reader.__globals__
    source_defaults = source_reader.__defaults__
    source_kwdefaults = source_reader.__kwdefaults__
    assert type(source_reader) is FunctionType and not source_code.co_freevars
    defining_module = sys.modules[ConversationLocalMarksService.__module__]
    assert source_globals is defining_module.__dict__
    original_copy = FunctionType(source_code, source_globals,
                                 source_reader.__name__, source_defaults)
    original_copy.__kwdefaults__ = source_kwdefaults
    custom_calls = []
    held_connections = []
    actual_actors = []

    def custom_reader(receiver, ids):
        custom_calls.append(threading.current_thread())
        result = original_copy(receiver, ids)
        held_connections.append(receiver.db.get_connection())
        return result

    workspace_name = 'tldw_chatbook.UI.Console_Modules.workspace'
    module_absent_before_replacement = workspace_name not in sys.modules
    if route == 'pre_ui_import':
        assert module_absent_before_replacement, 'fixture did not precede the actual UI import'
        assert ConversationLocalMarksService.unread_ids_for is source_reader
        ConversationLocalMarksService.unread_ids_for = custom_reader
    from tldw_chatbook.UI.Console_Modules import workspace
    if route in ('body_code', 'memory'):
        assert ConversationLocalMarksService.unread_ids_for is source_reader
        defining_module.__dict__['_unread_provenance_original'] = original_copy
        defining_module.__dict__['_unread_provenance_custom_calls'] = custom_calls
        defining_module.__dict__['_unread_provenance_connections'] = held_connections
        defining_module.__dict__['_unread_provenance_threads'] = threading
        exec('def _unread_provenance_replacement(self, conversation_ids):\n'
             '    _unread_provenance_custom_calls.append(\n'
             '        _unread_provenance_threads.current_thread())\n'
             '    result = _unread_provenance_original(self, conversation_ids)\n'
             '    _unread_provenance_connections.append(self.db.get_connection())\n'
             '    return result\n', defining_module.__dict__)
        replacement_code = defining_module.__dict__['_unread_provenance_replacement'].__code__
        assert replacement_code.co_freevars == source_code.co_freevars == ()
        source_reader.__code__ = replacement_code
        assert ConversationLocalMarksService.unread_ids_for is source_reader
        assert service.unread_ids_for.__self__ is service
        assert service.unread_ids_for.__func__ is source_reader
        assert source_reader.__globals__ is source_globals
    elif route == 'custom_instance':
        service.unread_ids_for = MethodType(custom_reader, service)

    modules = (config, participants, storage, base_db, defining_module, workspace)
    repository = Path.cwd().resolve()
    files = {module.__name__: Path(module.__file__).absolute() for module in modules}
    assert all(path.is_relative_to(repository) for path in files.values())
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in files.items()}
    protected = (participants._core_operation, participants._core_access,
                 base_db.operation_owned_connection, base_db.run_owned_db_call,
                 database.get_connection.__func__, database.transaction.__func__,
                 database.close_connection.__func__, windows_files._Native.open_handle)
    protected_codes = tuple(function.__code__ for function in protected)
    protected_globals = tuple(function.__globals__ for function in protected)
    owned_scope_code = base_db.operation_owned_connection.__wrapped__.__code__
    owned_scope_calls = []
    authority = ('profile', 'source')
    state = {'key': (service, service.manual_revision, authority),
             'values': {}, 'pending': {conversation_id}}
    renders = []
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(conversation_local_marks_service=service,
                                     chachanotes_db=database),
        _manual_unread_cache=state,
        _console_switcher_authority=lambda: authority,
        _sync_console_workspace_context=lambda: renders.append(True))

    def selected(frame, event, arg):
        if (event == 'call' and frame.f_code is owned_scope_code
                and frame.f_locals.get('database') is database):
            owned_scope_calls.append(threading.current_thread())
        return None

    def inspect_owned_connection():
        actor = threading.current_thread()
        actual_actors.append(actor)
        assert custom_calls and all(entered is actor for entered in custom_calls)
        assert held_connections and all(connection is held_connections[0]
                                        for connection in held_connections)
        connection = held_connections[0]
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
            alive_before_test_cleanup = True
        except sqlite3.ProgrammingError:
            alive_before_test_cleanup = False
        return alive_before_test_cleanup

    async def exercise():
        nonlocal conversation_id
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='unread-source-owner')
        loop.set_default_executor(executor)
        try:
            if route == 'memory':
                # Each native actor's :memory: connection has its own schema.
                # Seed through the actual schema/write methods on this actor,
                # rather than relying on the constructor's main-thread DB.
                def seed_owned_memory():
                    database._initialize_schema()
                    selected = database.add_conversation({'title': 'Owned memory unread'})
                    service.mark_unread(selected)
                    return selected

                conversation_id = await asyncio.to_thread(seed_owned_memory)
                state['key'] = (service, service.manual_revision, authority)
                state['pending'] = {conversation_id}
            await workspace.ConsoleWorkspaceController._load_manual_unread_rows(
                screen, service, state, (conversation_id,))
            assert state['values'] == {conversation_id: True}
            assert not state['pending'] and renders == [True]
            return await asyncio.to_thread(inspect_owned_connection)
        finally:
            # This executor/database/recorded handle belongs only to this test.
            # Inspect first, then physically retire on that same native actor.
            await asyncio.to_thread(database.close_connection)

    assert sys.gettrace() is None and threading.gettrace() is None
    alive = None
    failure = None
    threading.settrace_all_threads(selected)
    try:
        try:
            alive = asyncio.run(exercise())
        except BaseException as error:
            failure = error
    finally:
        threading.settrace_all_threads(None)
        sys.settrace(None)
        ConversationLocalMarksService.unread_ids_for = source_reader
        source_reader.__code__ = source_code
        service.__dict__.pop('unread_ids_for', None)
        database.close_connection()
        assert ConversationLocalMarksService.unread_ids_for is source_reader
        assert source_reader.__code__ is source_code and source_reader.__globals__ is source_globals
        assert source_reader.__defaults__ is source_defaults
        assert source_reader.__kwdefaults__ is source_kwdefaults
        assert all(function.__code__ is code and function.__globals__ is defining
                   for function, code, defining in
                   zip(protected, protected_codes, protected_globals))
        for connection in held_connections:
            try:
                sqlite3.Connection.in_transaction.__get__(connection)
            except sqlite3.ProgrammingError:
                pass
            else:
                raise AssertionError('test-owned native connection did not physically retire')
        with storage._lock:
            census = {
                'ordinary': len(storage._live_leases - set(storage._startups.values())),
                'pending': len(storage._pending_acquisitions),
                'operations': len(storage._operations), 'raw': len(storage._raw_operations),
                'retiring': len(storage._retiring_holds),
            }
        assert not any(census.values()), census
        assert not network_guard.blocked_attempts()
        assert not real_profile_guard.take_violations()
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]
                   for name, path in files.items())
    receipt = {
        'route': route, 'module_absent_before_replacement': module_absent_before_replacement,
        'alive_before_owned_test_cleanup': alive, 'custom_callback_calls': len(custom_calls),
        'added_stock_owned_scope_calls': len(owned_scope_calls),
        'same_owned_worker_actor': bool(actual_actors) and all(
            actor is actual_actors[0] for actor in custom_calls + actual_actors),
        'connection_physically_closed_after_owned_cleanup': bool(held_connections),
        'original_guard_bodies_unchanged': True, 'original_reader_restored': True,
        'trace_restored': sys.gettrace() is None and threading.gettrace() is None,
        'source_hashes': hashes, 'final_census': census,
        'unexpected_error': type(failure).__name__ if failure is not None else None,
    }
    (root.parent / 'unread-source-provenance-receipt.json').write_text(
        json.dumps(receipt, sort_keys=True), encoding='utf-8')
    print(json.dumps(receipt, sort_keys=True), flush=True)
    if failure is not None:
        raise failure
    assert receipt['same_owned_worker_actor'] and len(custom_calls) == 1
    assert alive is True, 'custom unread reader lost its original worker connection lifetime'
    assert not owned_scope_calls, 'custom unread reader entered the added stock owner scope'
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize("route", ["pre_ui_import", "body_code", "custom_instance", "memory"])
def test_custom_unread_reader_keeps_original_worker_connection_lifetime(tmp_path, route):
    _run(tmp_path, route, "custom_unread", script=_SCRIPT, timeout=90)
    receipt = json.loads(
        (tmp_path / "unread-source-provenance-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["alive_before_owned_test_cleanup"]
    assert receipt["connection_physically_closed_after_owned_cleanup"]
    assert receipt["original_guard_bodies_unchanged"] and receipt["original_reader_restored"]
    assert receipt["trace_restored"] and not any(receipt["final_census"].values())
