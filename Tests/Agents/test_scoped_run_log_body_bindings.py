"""Evidence-only native original run-log body fault controls; not yet executed."""

import json

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""

import contextvars, functools, inspect, json, os, sqlite3, sys, threading
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from Tests.network_guard import install, blocked_attempts
install()
route = sys.argv[1]
selector = Path(os.environ["TLDW_CONFIG_PATH"])
base = selector.parent.parent
selector.write_text('[general]\nusers_name="scoped"\n[paths]\ndata_dir="'
    + (base / 'data').as_posix() + '"\n', encoding='utf-8')
selector.chmod(0o600)
from tldw_chatbook import config
from tldw_chatbook.Agents import activation, run_log
from tldw_chatbook.Agents.agent_models import AgentConfig, RUN_DONE
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Backup_Recovery import admission_runtime, storage_admission as storage
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Utils import windows_files

db = AgentRunsDB(config.get_user_data_dir() / 'agent_runs.db', client_id='scoped')
manager = ConsoleScratchSpaceManager(temp_parent=base / 'home')
snapshot = manager.snapshot('chat-a')
other = manager.snapshot('chat-b')
published = []
foreign_published = []
writer = None
scope_entered = False
foreign_dispatches = []
captured_contexts = []

@contextmanager
def source_scope():
    global scope_entered
    with manager.lease(snapshot):
        scope_entered = True
        if route == 'retarget-root':
            writer._explicit_root = other.root
        elif route == 'retarget-publisher':
            writer._on_bound = lambda *args: foreign_published.append(True)
        elif route == 'retarget-partial-publisher':
            writer._on_bound.keywords['session_id'] = 'chat-b'
        try:
            yield snapshot.root
        finally:
            scope_entered = False

def publish(run_id, root, *, session_id):
    if session_id == 'chat-a':
        published.append((run_id, root))
        if route == 'foreign-copy':
            captured_contexts.append((contextvars.copy_context(), run_log._SCOPED_LOG_SOURCE.get()))
    else:
        foreign_published.append(True)

writer = run_log.RunLogWriter(root=snapshot.root, access_scope=source_scope,
    on_bound=functools.partial(publish, session_id='chat-a'))
# The original constructor normalizes root into a different Path object.
expected_writer_root = writer._explicit_root
expected_normalized_root = Path(snapshot.root).resolve()
assert expected_writer_root == expected_normalized_root
original_selector = run_log.resolve_log_root
selector_calls = []
if route == 'custom-selector':
    def custom_selector(*args, **kwargs):
        selector_calls.append((args, kwargs))
        return snapshot.root
    run_log.resolve_log_root = custom_selector

provider_calls = []
def chat(**kwargs):
    provider_calls.append(True)
    return {'choices': [{'message': {'content': 'completed'}}]}
service = AgentService(db, ToolCatalogRegistry(), chat_call=chat, run_log_writer=writer)
originals = (activation.execution, activation._guard.execution.__func__,
    run_log.RunLogWriter.bind, run_log.RunLogWriter._bind_under_scope,
    storage.acquire_storage)
selector_code = original_selector.__code__
admission_code = inspect.unwrap(admission_runtime.RecoveryAdmissionGuard.execution).__code__
native_code = windows_files._Native.open_handle.__code__ if os.name == 'nt' else None
create_code = inspect.unwrap(AgentRunsDB.create_run).__code__
bind_code = run_log.RunLogWriter.bind.__code__
under_bind_code = run_log.RunLogWriter._bind_under_scope.__code__
check_code = run_log._check_scoped_writer.__code__
from tldw_chatbook.Tools.file_operation_tools import is_within
containment_code = is_within.__code__
# Two additive routes retain the original nine-route assertions below.
body_routes = ('late-under-bind-body', 'late-publisher-body')
original_under_bind = run_log.RunLogWriter._bind_under_scope
original_publish = publish
body_targets = (
    (original_under_bind, original_under_bind.__code__, original_under_bind.__defaults__,
     original_under_bind.__kwdefaults__, original_under_bind.__globals__),
    (original_publish, original_publish.__code__, original_publish.__defaults__,
     original_publish.__kwdefaults__, original_publish.__globals__),
)
protected_functions = (activation.execution, activation._guard.execution.__func__,
    storage.acquire_storage, run_log.RunLogWriter.bind, AgentService.run_turn,
    AgentService.run_turn.__wrapped__, original_selector)
if native_code is not None:
    protected_functions += (windows_files._Native.open_handle,)
protected_codes = tuple((function, function.__code__) for function in protected_functions)
source_root = Path(config.__file__).parent
source_names = ('Agents/activation.py', 'Agents/agent_service.py', 'Agents/run_log.py',
    'Backup_Recovery/storage_admission.py', 'Chat/console_scratch_space.py',
    'Utils/windows_files.py')
import hashlib
source_hashes = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                 for name in source_names}
fault_records = []
writer_active_after_run = None

def changed_under_bind(self, run_id):
    self.__dict__.setdefault('_scoped_body_dispatches', []).append('late-under-bind-body')

def changed_publish(run_id, root, *, session_id):
    foreign_dispatches.append(route)

fault_injected = False

def foreign_dispatch(self, *args):
    foreign_dispatches.append(route)

def late_return(frame, event, arg):
    global fault_injected
    if event != 'return' or fault_injected:
        return late_return
    caller = frame.f_back
    method = None
    if route == 'late-bind' and frame.f_code is create_code and isinstance(arg, str):
        method = 'bind'
    elif (route == 'late-under-bind' and frame.f_code is check_code
          and scope_entered and caller is not None and caller.f_code is bind_code):
        method = '_bind_under_scope'
    elif (route == 'late-migration' and frame.f_code is containment_code
          and arg is True and caller is not None and caller.f_code is under_bind_code):
        method = '_migrate_legacy_dir'
    if (route in body_routes and frame.f_code is check_code
            and scope_entered and caller is not None and caller.f_code is bind_code):
        record = run_log._SCOPED_LOG_SOURCE.get()
        assert record is not None, 'original writer check returned no issued scoped source'
        with manager._condition:
            scratch_record = manager._matching_record_locked(snapshot)
            scratch_live = scratch_record is not None and scratch_record.leases > 0
        with storage._lock:
            startup = set(storage._startups.values())
            live_count = len(storage._live_leases - startup)
        current_actor = admission_runtime.execution_identity()
        preconditions = {
            'original_check_returned_exact_record': arg is record,
            'issued_source_live': record.live,
            'exact_service_receiver': record.service is service,
            'exact_writer_receiver': record.writer is writer,
            'exact_writer_root_owner': writer._explicit_root is expected_writer_root,
            'exact_record_root_owner': record.root is expected_writer_root,
            'original_normalized_root_value': record.root == expected_normalized_root,
            'original_snapshot_root_value': record.root == snapshot.root,
            'exact_thread_owner': record.thread is threading.current_thread(),
            'original_execution_actor_value': record.actor == current_actor,
            'actual_current_root_admitted': counts['actual_root_admissions'] > 0,
            'actual_scratch_lease_live': scratch_live,
            'actual_ordinary_native_lease_live': live_count > 0,
        }
        # Scalar actor fields and normalized path values follow production's
        # equality semantics; exact retained receivers/root/thread use `is`.
        # Persist only booleans before mutation so an invalid precondition is
        # classified as fixture failure, even if original bind degrades it.
        (base / 'scoped-body-preconditions.json').write_text(json.dumps({
            'route': route, 'preconditions': preconditions,
            'snapshot_is_constructor_root': snapshot.root is expected_writer_root,
            'mutation_started': False,
        }), encoding='utf-8')
        assert all(preconditions.values()), preconditions
        if route == 'late-under-bind-body':
            target, saved_code, defaults, kwdefaults, defining = body_targets[0]
            replacement = changed_under_bind
            assert vars(run_log.RunLogWriter)['_bind_under_scope'] is target
            assert writer._bind_under_scope.__func__ is target
        else:
            target, saved_code, defaults, kwdefaults, defining = body_targets[1]
            replacement = changed_publish
            assert type(writer._on_bound) is functools.partial
            assert writer._on_bound.func is target
            assert writer._on_bound.keywords['session_id'] == 'chat-a'
        assert target.__code__ is saved_code and target.__globals__ is defining
        assert target.__defaults__ is defaults and target.__kwdefaults__ is kwdefaults
        # Only this unguarded original function's code object changes. The
        # descriptor, captured function, globals, defaults and callback binding
        # stay exact; no source/native/admission function is replaced.
        target.__code__ = replacement.__code__
        assert target.__code__ is replacement.__code__
        fault_records.append((record, live_count))
        fault_injected = True
        return late_return
    if method is not None:
        # Fault only changes instance metadata after the original producer
        # returned. DB, source lease, native guard and class bodies stay original.
        setattr(writer, method, MethodType(foreign_dispatch, writer))
        fault_injected = True
    return late_return
counts = {'unrelated_global_selectors': 0, 'native_opens': 0, 'actual_root_admissions': 0}
def observe(frame, event, arg):
    if event != 'call':
        return None
    if frame.f_code is selector_code:
        if frame.f_locals.get('sandbox_root') is None and frame.f_locals.get('workspace_id') is None:
            counts['unrelated_global_selectors'] += 1
    elif native_code is not None and frame.f_code is native_code:
        counts['native_opens'] += 1
    elif frame.f_code is admission_code and frame.f_locals.get('self') is activation._guard:
        if ('agents.history', snapshot.root) in frame.f_locals.get('sources', ()):
            counts['actual_root_admissions'] += 1
    if route.startswith('late-') and frame.f_code in (create_code, check_code, containment_code):
        frame.f_trace_lines = False
        frame.f_trace_opcodes = False
        return late_return
    return None

assert sys.gettrace() is None
try:
    sys.settrace(observe)
    try:
        run_id, outcome = service.run_turn(conversation_id='conversation',
            messages=[{'role': 'user', 'content': 'hello'}],
            config=AgentConfig(model='test', system_prompt=''), api_endpoint='OpenAI')
    finally:
        assert sys.gettrace() is observe
        sys.settrace(None)
    assert outcome.status == RUN_DONE and provider_calls == [True]
    print(json.dumps({'route': route, 'counts': counts}), flush=True)
    if route in ('ordinary', 'foreign-copy'):
        assert writer.is_active and writer.log_dir.is_relative_to(snapshot.root)
        assert published == [(run_id, snapshot.root)]
        assert counts['actual_root_admissions'] > 0, 'actual writer root was not independently admitted'
        assert counts['unrelated_global_selectors'] == 0, 'explicit writer discovered unrelated global roots'
    elif route in ('retarget-root', 'retarget-publisher', 'retarget-partial-publisher'):
        assert not writer.is_active, 'changed source was allowed to bind'
        assert not foreign_published, 'changed publisher was invoked'
        assert not list(other.root.glob('.agent-runs/*')), 'retargeted root received log effects'
    elif route in body_routes:
        # Defer only these new refusal assertions until after physical cleanup.
        # Otherwise an expected native RED would skip the original final census.
        assert fault_injected and len(fault_records) == 1
        assert run_log._SCOPED_LOG_SOURCE.get() is None
        assert not fault_records[0][0].live, 'issued source metadata survived the turn'
        assert counts['actual_root_admissions'] > 0
        assert counts['unrelated_global_selectors'] == 0
        foreign_dispatches.extend(writer.__dict__.get('_scoped_body_dispatches', ()))
        writer_active_after_run = writer.is_active
    elif route.startswith('late-'):
        assert fault_injected, 'original late boundary was not observed'
        assert not foreign_dispatches, 'late replaced dispatch ran before its source fence'
        assert not writer.is_active, 'late changed writer was allowed to bind'
        assert not foreign_published
    else:
        assert selector_calls == [((), {})], selector_calls
        assert writer.is_active and published == [(run_id, snapshot.root)]
    if route == 'foreign-copy':
        assert len(captured_contexts) == 1
        copied, record = captured_contexts[0]
        assert record is not None and not record.live
        def must_refuse(call):
            try:
                call()
            except PermissionError:
                return
            raise AssertionError('retired or foreign explicit source was accepted')
        must_refuse(lambda: copied.run(run_log.check_scoped_log_service, service))
        def enter_explicit():
            with run_log.scoped_log_source(record):
                raise AssertionError('foreign scope was entered')
        must_refuse(lambda: copied.run(enter_explicit))
        old_dir = writer.log_dir
        thread_errors = []
        @activation.worker_guard(service)
        def worker():
            assert threading.current_thread() is not record.thread
            must_refuse(enter_explicit)
            writer.bind('foreign-child')  # Documented idempotent binding latch.
            assert writer.is_active and writer.log_dir is old_dir
            number = writer.append(run_id='foreign-child', kind='subagent',
                type='model', content='independent-child-marker')
            assert number is not None, 'independent original child append was disabled'
        def target():
            try:
                copied.run(worker)
            except BaseException as error:
                thread_errors.append(error)
        child = threading.Thread(target=target, name='scoped-copy-native-control')
        child.start()
        child.join(30)
        assert not child.is_alive(), 'native control worker did not physically retire'
        assert not thread_errors, repr(thread_errors)
        assert published == [(run_id, snapshot.root)]
        assert any('independent-child-marker' in p.read_text(encoding='utf-8')
                   for p in old_dir.glob('logs.*.txt'))
    assert originals == (activation.execution, activation._guard.execution.__func__,
        run_log.RunLogWriter.bind, run_log.RunLogWriter._bind_under_scope,
        storage.acquire_storage), 'guard or original writer callback changed'
    assert not blocked_attempts()
finally:
    for function, code, defaults, kwdefaults, defining in body_targets:
        function.__code__ = code
        assert function.__code__ is code and function.__globals__ is defining
        assert function.__defaults__ is defaults and function.__kwdefaults__ is kwdefaults
    assert sys.gettrace() is None, 'original trace hook was not restored'
    assert all(function.__code__ is code for function, code in protected_codes), \
        'original source/native/admission body changed'
    connection = getattr(db._thread_local, 'conn', None)
    db.close()
    manager.close('chat-a')
    manager.close('chat-b')
    assert manager.wait_for_cleanup(timeout_seconds=2.0)
    run_log.resolve_log_root = original_selector
    if connection is not None:
        try:
            connection.execute('SELECT 1')
        except sqlite3.ProgrammingError:
            pass
        else:
            raise AssertionError('native SQLite connection did not physically close')
with storage._lock:
    startup = set(storage._startups.values())
    assert not (storage._live_leases - startup), 'ordinary native lease survived'
    assert not storage._operations and not storage._raw_operations
    assert not storage._pending_acquisitions and not storage._retiring_holds
assert source_hashes == {
    name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
    for name in source_names
}, 'managed source changed during the native fault'
assert all(function.__code__ is code for function, code in protected_codes)
assert all(function.__code__ is code and function.__globals__ is defining
           and function.__defaults__ is defaults and function.__kwdefaults__ is kwdefaults
           for function, code, defaults, kwdefaults, defining in body_targets)
assert not blocked_attempts()
with storage._lock:
    startup = set(storage._startups.values())
    final_census = {
        'ordinary_leases': len(storage._live_leases - startup),
        'operations': len(storage._operations),
        'raw_operations': len(storage._raw_operations),
        'pending_acquisitions': len(storage._pending_acquisitions),
        'retiring_holds': len(storage._retiring_holds),
    }
assert not any(final_census.values())
(base / 'scoped-body-receipt.json').write_text(json.dumps({
    'route': route, 'counts': counts, 'fault_injected': fault_injected,
    'foreign_dispatches': foreign_dispatches,
    'live_ordinary_leases_at_fault': fault_records[0][1],
    'provider_calls': len(provider_calls), 'provider_completed': outcome.status == RUN_DONE,
    'writer_active_after_run': writer_active_after_run,
    'source_metadata_retired': not fault_records[0][0].live,
    'final_census': final_census, 'source_hashes': source_hashes,
    'sqlite_connection_physically_closed': connection is not None,
    'original_guarded_bodies_unchanged': True, 'unguarded_bodies_restored': True,
    'trace_restored': sys.gettrace() is None,
}), encoding='utf-8')
print(json.dumps({'body_route': route, 'counts': counts, 'final_census': final_census,
                  'foreign_dispatches': foreign_dispatches}), flush=True)
(base / 'scoped-source-receipt.json').write_text(json.dumps({
    'route': route, 'counts': counts, 'fault_injected': fault_injected,
    'foreign_dispatches': foreign_dispatches, 'ordinary_resources_retired': True,
    'sqlite_connection_physically_closed': connection is not None,
    'guards_replaced': False,
}), encoding='utf-8')
print('retired and reopened')

assert not foreign_dispatches, 'changed same-identity original body ran after stock qualification'
assert not writer_active_after_run, 'changed same-identity source was allowed to bind'
assert not foreign_published, 'changed source published foreign authority'
"""


@pytest.mark.parametrize("route", ["late-under-bind-body", "late-publisher-body"])
def test_scoped_run_log_refuses_same_identity_body_drift(tmp_path, route):
    _run(tmp_path, route, "scoped", script=_SCRIPT, timeout=180)
    receipt = json.loads(
        (tmp_path / "scoped-body-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["provider_completed"] and receipt["provider_calls"] == 1
    assert receipt["live_ordinary_leases_at_fault"] > 0
    assert receipt["source_metadata_retired"]
    assert receipt["sqlite_connection_physically_closed"]
    assert not any(receipt["final_census"].values())
    assert (
        receipt["original_guarded_bodies_unchanged"]
        and receipt["unguarded_bodies_restored"]
    )
    assert receipt["trace_restored"]
    print(json.dumps(receipt))
