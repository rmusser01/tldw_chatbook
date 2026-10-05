"""Original hook-key reads must retire newly owned Workspace worker handles."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


pytestmark = pytest.mark.bootstrap_profile


_SCRIPT = r"""
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import CodeType, SimpleNamespace

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()

route, outcome = sys.argv[1:]
assert route == 'hook_workspace'


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
    if connection is None:
        return True
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
    from tldw_chatbook.Chat import console_runtime as runtime_module
    from tldw_chatbook.Chat import console_chat_controller as controller_module
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.DB.base_db import operation_owned_connection
    from tldw_chatbook.Workspaces import registry_service as registry_module
    from tldw_chatbook.Workspaces.change_review_consent import ChangeReviewConsentService

    # These are original installed methods; no permission/native reader is replaced.
    key_method = runtime_module.ConsoleRuntime._hooks_v2_context_key
    authority_method = controller_module.ConsoleChatController._hook_authority_values
    consent_method = registry_module.LocalWorkspaceRegistryService.read_change_review_consent
    workspace_method = registry_module.LocalWorkspaceRegistryService.get_workspace
    methods = (key_method, authority_method, consent_method, workspace_method)
    original_runtime_class = runtime_module.ConsoleRuntime
    original_controller_class = controller_module.ConsoleChatController
    original_registry_class = registry_module.LocalWorkspaceRegistryService
    records = tuple((method, method.__code__, method.__globals__, method.__defaults__,
                     method.__kwdefaults__, method.__closure__) for method in methods)
    sources = {}
    for method, code, defining, defaults, kwdefaults, closure in records:
        module = sys.modules[defining['__name__']]
        path = Path(module.__file__).absolute()
        assert path == Path(module.__spec__.origin).absolute()
        assert path.is_relative_to(Path.cwd().resolve())
        raw = path.read_bytes()
        assert shape(nested(compile(raw, str(path), 'exec'), code.co_qualname)) == shape(code)
        sources[path] = raw

    class CustomWorkspaceDB(WorkspaceDB):
        pass

    class CustomRegistry(registry_module.LocalWorkspaceRegistryService):
        def get_workspace(self, workspace_id):
            assert workspace_id == 'hook-owned'
            return super().get_workspace(workspace_id)

    if outcome == 'runtime_alias':
        class ReplacementRuntime(original_runtime_class):
            pass
        runtime_module.ConsoleRuntime = ReplacementRuntime
    elif outcome == 'controller_alias':
        class ReplacementController(original_controller_class):
            pass
        controller_module.ConsoleChatController = ReplacementController
    elif outcome == 'registry_alias':
        class ReplacementRegistry(original_registry_class):
            pass
        registry_module.LocalWorkspaceRegistryService = ReplacementRegistry
    database = None
    foreign = None
    registry = None
    consent = None
    runtime = None
    controller = None
    actor = None
    borrowed = None
    observed = []
    snapshots = []
    invalid = []
    calls = {'key': 0, 'authority': 0, 'consent': 0, 'workspace': 0}
    code_names = {method.__code__: name for method, name in zip(methods, calls)}
    armed = False
    retargeted = False
    entered = threading.Event()
    release = threading.Event()
    callback_returned = threading.Event()
    refusal = False
    result_equal = False
    connection_count = None
    foreign_connection_count = None

    def seed(db):
        service_class = CustomRegistry if outcome == 'custom_reader' else registry_module.LocalWorkspaceRegistryService
        service = service_class(db)
        service.create_workspace(workspace_id='hook-owned', name='Hook owned', assistant_defaults=None)
        return service

    if outcome not in {'memory', 'custom_db'}:
        database = WorkspaceDB(config.get_user_data_dir() / 'hook-owned.sqlite', client_id='hook-a')
        registry = seed(database)
        if outcome == 'error':
            with database.transaction() as connection:
                connection.execute('DROP TABLE workspace_change_review')
        if outcome == 'retarget':
            foreign = WorkspaceDB(database.db_path, client_id='hook-b')
            foreign.close()
        database.close()

    def build_runtime():
        nonlocal runtime, controller, consent
        # Actual stock component constructors, with their documented headless seams.
        # No TldwCli, LLM request, fake permission reader, or fake storage participant.
        consent = ChangeReviewConsentService(registry)
        app = SimpleNamespace(app_config=config.load_settings(),
                              workspace_registry_service=registry,
                              change_review_consent_service=consent)
        store = ConsoleChatStore()
        session = store.create_session(workspace_id='hook-owned', ephemeral=True)
        controller = controller_module.ConsoleChatController(
            store=store, provider_gateway=None, agent_runtime_enabled=False)
        runtime = runtime_module.ConsoleRuntime(app)
        runtime.set_chat_store(store)
        runtime.set_chat_controller(controller)
        return session.id

    def snapshot_connection(row):
        db = row['database']
        participant = row['participant']
        connection = row['connection']
        with storage._lock:
            return dict(closed=physically_closed(connection),
                        lease_live=row['lease'] in storage._live_leases if row['lease'] else False,
                        registered=participant is not None and connection in participant.connections,
                        cached=getattr(db._thread_local, 'conn', None) is connection,
                        database_is_foreign=db is foreign)

    def read_line(code, line):
        if not armed or code not in (consent_method.__code__, workspace_method.__code__):
            return
        frame = sys._getframe(1)
        if frame.f_code is not code or frame.f_locals.get('self') is not registry:
            return
        connection = frame.f_locals.get('conn')
        if connection is None or any(row['connection'] is connection and row['code'] is code for row in observed):
            return
        try:
            assert threading.current_thread() is actor
            assert isinstance(connection, sqlite3.Connection) and not physically_closed(connection)
            db = foreign if foreign is not None and connection is getattr(foreign._thread_local, 'conn', None) else database
            assert connection is getattr(db._thread_local, 'conn', None)
            participant = getattr(db, '_maintenance_participant', None)
            with storage._lock:
                lease = None if participant is None else participant.connections.get(connection)
                if outcome not in {'memory', 'custom_db'}:
                    assert participant is not None and lease in storage._live_leases
                    assert lease.resource_thread is actor and lease.resource_participant is participant
            observed.append(dict(connection=connection, database=db, participant=participant,
                                 lease=lease, code=code))
            if outcome == 'cancel' and code is consent_method.__code__:
                entered.set()
                if not release.wait(10):
                    invalid.append('actual_reader_release_timeout')
                callback_returned.set()
        except BaseException as error:
            invalid.append(type(error).__name__)
            entered.set()

    def started(code, offset):
        if armed and code in code_names:
            assert threading.current_thread() is actor
            calls[code_names[code]] += 1

    def returned(code, offset, value):
        nonlocal retargeted
        if not armed or outcome != 'retarget' or retargeted or code is not consent_method.__code__:
            return
        frame = sys._getframe(1)
        if frame.f_code is not code or frame.f_locals.get('self') is not registry:
            return
        assert threading.current_thread() is actor and registry.db is database
        assert borrowed is not None and not physically_closed(borrowed)
        assert sqlite3.Connection.in_transaction.__get__(borrowed)
        registry.db = foreign  # Metadata-only retarget at the original completed read.
        retargeted = True

    monitoring = sys.monitoring
    tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(tool, 'hook-workspace-finite-reader')
    monitoring.register_callback(tool, monitoring.events.LINE, read_line)
    monitoring.register_callback(tool, monitoring.events.PY_START, started)
    monitoring.register_callback(tool, monitoring.events.PY_RETURN, returned)
    for method in methods:
        events = monitoring.events.PY_START
        if method in (consent_method, workspace_method):
            events |= monitoring.events.LINE | monitoring.events.PY_RETURN
        monitoring.set_local_events(tool, method.__code__, events)
    assert monitoring.get_events(tool) == 0

    def worker():
        nonlocal database, registry, actor, borrowed, armed, refusal, result_equal
        nonlocal connection_count, foreign_connection_count
        actor = threading.current_thread()
        if outcome in {'memory', 'custom_db'}:
            database = (WorkspaceDB(':memory:', client_id='hook-a') if outcome == 'memory'
                        else CustomWorkspaceDB(config.get_user_data_dir() / 'hook-custom.sqlite', client_id='hook-a'))
            registry = seed(database)
        session_id = build_runtime()
        if outcome in {'borrowed', 'retarget'}:
            borrowed = database._held_connection()
            borrowed.execute('BEGIN')
        expected = (runtime._chat_store.sessions()[0].workspace_id, (),
                    controller_module.capture_project_instruction_authority(
                        runtime._chat_store.sessions()[0], registry, include_bindings=False))
        try:
            armed = True
            try:
                # Positive mechanism keeps all original callback bodies, only adds
                # the proposed supported scope around the same actual WorkspaceDB.
                with operation_owned_connection(database) if outcome == 'hypothesis' else nullcontext():
                    result = runtime._hooks_v2_context_key(session_id)
            except RuntimeError as error:
                assert outcome == 'retarget'
                assert str(error) == 'Console hook workspace ownership changed.'
                refusal = True
            else:
                result_equal = result == expected
                assert result_equal
            snapshots.extend(snapshot_connection(row) for row in observed)
            connection_count = len({id(row['connection']) for row in observed
                                    if row['database'] is database})
            foreign_connection_count = len({id(row['connection']) for row in observed
                                            if row['database'] is foreign})
            if borrowed is not None:
                assert database._held_connection() is borrowed
                assert sqlite3.Connection.in_transaction.__get__(borrowed)
                assert borrowed.execute('SELECT 1').fetchone()[0] == 1
        finally:
            armed = False
            release.set()
            registry.db = database
            if borrowed is not None and not physically_closed(borrowed):
                borrowed.rollback()
            # Test-owned physical cleanup occurs on the original worker AFTER evidence.
            for db in (foreign, database):
                if db is not None:
                    db.close()
            if consent is not None:
                consent.shutdown()
            runtime_module.ConsoleRuntime = original_runtime_class
            controller_module.ConsoleChatController = original_controller_class
            registry_module.LocalWorkspaceRegistryService = original_registry_class
            if runtime is not None:
                asyncio.run(runtime.dispose())

    async def cancelled_waiter(native):
        async def await_native():
            await asyncio.wrap_future(native)
        task = asyncio.create_task(await_native())
        try:
            deadline = asyncio.get_running_loop().time() + 10
            while not entered.is_set():
                assert asyncio.get_running_loop().time() < deadline, 'original consent reader not reached'
                await asyncio.sleep(.01)
            assert observed and not invalid
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('original waiter not cancelled')
            assert native.running() and not native.done()
            assert not physically_closed(observed[0]['connection'])
            with storage._lock:
                assert observed[0]['lease'] in storage._live_leases
        finally:
            release.set()
        await asyncio.wait_for(asyncio.wrap_future(native), 10)
        assert callback_returned.is_set()

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='hook-workspace') as executor:
            native = executor.submit(worker)
            if outcome == 'cancel':
                asyncio.run(cancelled_waiter(native))
            else:
                native.result(timeout=10)
        assert actor is not None and not actor.is_alive()
    finally:
        armed = False
        release.set()
        for method in methods:
            monitoring.set_local_events(tool, method.__code__, 0)
        assert monitoring.register_callback(tool, monitoring.events.LINE, None) is read_line
        assert monitoring.register_callback(tool, monitoring.events.PY_START, None) is started
        assert monitoring.register_callback(tool, monitoring.events.PY_RETURN, None) is returned
        monitoring.free_tool_id(tool)
        runtime_module.ConsoleRuntime = original_runtime_class
        controller_module.ConsoleChatController = original_controller_class
        registry_module.LocalWorkspaceRegistryService = original_registry_class
        for db in (foreign, database):
            if db is not None:
                db.close()  # Creator's cache only, never a foreign worker cache.
    assert not invalid, invalid
    assert all(path.read_bytes() == raw for path, raw in sources.items())
    assert all(method.__code__ is code and method.__globals__ is defining
               and method.__defaults__ is defaults and method.__kwdefaults__ is kwdefaults
               and method.__closure__ is closure
               for method, code, defining, defaults, kwdefaults, closure in records)
    assert calls == dict(key=1, authority=1, consent=1, workspace=1), calls
    assert observed and snapshots, 'actual native query boundary was not observed'
    assert all(physically_closed(row['connection']) for row in observed)
    with storage._lock:
        worker_leases = sum(lease.resource_thread is actor for lease in storage._live_leases)
    guard_counts = dict(network=len(network_guard.blocked_attempts()),
                        real_profile=len(real_profile_guard._violations))
    assert worker_leases == 0 and all(count == 0 for count in guard_counts.values())
    preserve = outcome in {'borrowed', 'memory', 'custom_db'}
    foreign_rows = [row for row in snapshots if row['database_is_foreign']]
    expected_retirement = (
        refusal and retargeted and foreign_rows
        and all(row['closed'] and not row['lease_live'] and not row['registered']
                and not row['cached'] for row in foreign_rows)
        if outcome == 'retarget' else
        all(row['closed'] is not preserve for row in snapshots)
        and (preserve or all(not row['lease_live'] and not row['registered']
                             and not row['cached'] for row in snapshots))
    )
    optional_scope_supported = outcome not in {
        'custom_reader', 'runtime_alias', 'controller_alias', 'registry_alias'}
    # Original stock queries share one actual physical handle. Custom
    # aliases must decline optional grouping; their two direct stock
    # public reader bodies each retire their newly opened handle.
    expected_handle_count = (connection_count == 1 if optional_scope_supported
                             else connection_count == 2)
    if outcome == 'retarget':
        expected_handle_count = connection_count == 1 and foreign_connection_count == 1
    receipt = dict(outcome=outcome, calls=calls, snapshots=snapshots,
                   normal_workspace_connection_count=connection_count,
                   foreign_workspace_connection_count=foreign_connection_count,
                   expected_handle_count=expected_handle_count,
                   result_equal=result_equal, refusal=refusal, retargeted=retargeted,
                   expected_retirement=bool(expected_retirement), invalid=invalid,
                   global_events=monitoring.get_events(tool),
                   hooks_retired=monitoring.get_tool(tool) is None,
                   worker_positively_retired=not actor.is_alive(),
                   worker_leases_after_test_cleanup=worker_leases, guard_counts=guard_counts,
                   source_hashes={str(path): hashlib.sha256(raw).hexdigest()
                                  for path, raw in sources.items()})
    (root / ('hook-workspace-' + outcome + '.receipt.json')).write_text(
        json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    assert expected_retirement and expected_handle_count, json.dumps(receipt, sort_keys=True)
    print('retired and reopened')


with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize(
    "outcome",
    [
        "new",
        "hypothesis",
        "error",
        "cancel",
        "borrowed",
        "memory",
        "custom_db",
        "custom_reader",
        "retarget",
        "runtime_alias",
        "controller_alias",
        "registry_alias",
    ],
)
def test_original_hook_workspace_reads_retire_only_owned_handles(tmp_path, outcome):
    _run(tmp_path, "hook_workspace", outcome, script=_SCRIPT)
