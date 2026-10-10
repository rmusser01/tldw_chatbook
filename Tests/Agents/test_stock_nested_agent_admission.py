"""Actual original run-turn admission work-count/source controls."""

import json
import pytest
from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile
_SCRIPT = r"""
import ast, hashlib, contextvars, functools, inspect, json, os, sqlite3, sys, threading
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from Tests.network_guard import install, blocked_attempts
from Tests import real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
real_profile_guard.install()
install()
route, outcome = sys.argv[1:]
assert outcome == 'ownership'
assert route in {'stock', 'custom-selector', 'retarget-root', 'retarget-method', 'retarget-source', 'pause'}
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
        if route == 'retarget-publisher':
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

from tldw_chatbook.Backup_Recovery import activation as scope_source
from tldw_chatbook.Agents import agent_service as service_source
from concurrent.futures import ThreadPoolExecutor
from types import FunctionType
modules = (config, activation, admission_runtime, scope_source, service_source, run_log, storage)
source_bytes, origins, compiled = {}, {}, {}
def shape(code):
    from types import CodeType
    return (code.co_code, code.co_exceptiontable, code.co_stacksize, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags, code.co_names,
            code.co_varnames, code.co_freevars, code.co_cellvars,
            tuple(shape(value) if isinstance(value, CodeType) else value for value in code.co_consts))
def nested(code, name):
    from types import CodeType
    if code.co_qualname == name:return code
    for value in code.co_consts:
        if isinstance(value,CodeType):
            result=nested(value,name)
            if result is not None:return result
def record(function):
    return (function,function.__code__,function.__globals__,function.__defaults__,
            function.__kwdefaults__,tuple((function.__kwdefaults__ or {}).items()),
            function.__closure__,tuple((cell,cell.cell_contents) for cell in function.__closure__ or ()))
def current(row):
    function,code,namespace,defaults,kwdefaults,items,closure,cells=row
    return (function.__code__ is code and function.__globals__ is namespace
            and function.__defaults__ is defaults and function.__kwdefaults__ is kwdefaults
            and len(function.__kwdefaults__ or {})==len(items)
            and all((function.__kwdefaults__ or {}).get(key) is value for key,value in items)
            and function.__closure__ is closure and all(cell.cell_contents is value for cell,value in cells))
for module in modules:
    path=Path(module.__file__).resolve();origin=Path(module.__spec__.origin).resolve()
    assert path==origin and path.is_relative_to(Path.cwd().resolve())
    raw=path.read_bytes();source_bytes[path]=raw
    origins[module]=(module.__file__,module.__spec__,module.__spec__.origin)
    compiled[module.__name__]=compile(raw,str(path),'exec')
contextlib_module=sys.modules['contextlib']
contextlib_path=Path(contextlib_module.__file__).resolve()
assert contextlib_path==Path(contextlib_module.__spec__.origin).resolve()
contextlib_raw=contextlib_path.read_bytes()
source_bytes[contextlib_path]=contextlib_raw
origins[contextlib_module]=(contextlib_module.__file__,contextlib_module.__spec__,contextlib_module.__spec__.origin)
compiled['contextlib']=compile(contextlib_raw,str(contextlib_path),'exec')
scope_factory=scope_source.execution_scope
scope_body=scope_factory.__dict__['__wrapped__']
assert type(scope_factory) is FunctionType and type(scope_body) is FunctionType
assert dict(zip(scope_factory.__code__.co_freevars,scope_factory.__closure__))['func'].cell_contents is scope_body
assert admission_runtime.execution_scope is scope_factory
permission_body=scope_source._execution_permission_from_records
admit_sources=admission_runtime.RecoveryAdmissionGuard._admit_sources
assert type(permission_body) is FunctionType and type(admit_sources) is FunctionType
execution_factory=admission_runtime.RecoveryAdmissionGuard.execution
execution_body=execution_factory.__dict__['__wrapped__']
assert dict(zip(execution_factory.__code__.co_freevars,execution_factory.__closure__))['func'].cell_contents is execution_body
turn_anchor=service_source._SCOPED_RUN_TURN_ANCHOR
turn_wrapper,turn_body=turn_anchor[2:4]
assert AgentService.run_turn is turn_wrapper and turn_wrapper.__dict__['__wrapped__'] is turn_body
records=tuple(record(value) for value in (scope_factory,scope_body,permission_body,admit_sources,execution_factory,execution_body,
        activation.execution,activation.guarded,activation.worker_guard,turn_wrapper,turn_body,
        run_log.capture_scoped_log_source,run_log.scoped_log_source,run_log.RunLogWriter.bind,
        run_log.RunLogWriter._bind_under_scope,storage.acquire_storage))
for function,code,namespace,*_ in records:
    declared=nested(compiled[namespace['__name__']],code.co_qualname)
    assert declared is not None and shape(declared)==shape(code)
creator=getattr(db._thread_local,'conn',None)
assert isinstance(creator,sqlite3.Connection)
db.close()
try:sqlite3.Connection.in_transaction.__get__(creator)
except sqlite3.ProgrammingError:pass
else:raise AssertionError('creator native SQLite handle did not close')
actor=None;armed=False;rows=[];invalid=[];foreign_dispatches=[];pause=None;faults=[]
new_sql=[];expected_observed=[];result=None
opener=inspect.getattr_static(AgentRunsDB,'_get_connection').__dict__['__wrapped__']
opener_record=record(opener)
assert shape(nested(compile(Path(opener.__code__.co_filename).read_bytes(),opener.__code__.co_filename,'exec'),opener.__code__.co_qualname))==shape(opener.__code__)
def nearest(frame, code):
    for _ in range(96):
        if frame is None:return None
        if frame.f_code is code:return frame
        frame=frame.f_back
    invalid.append('ancestry_limit')
    return None
def guarded_turn(frame):
    for _ in range(96):
        if frame is None:return None
        if frame.f_code is turn_wrapper.__code__:
            return frame if frame.f_locals.get('function') is turn_body and frame.f_locals.get('self') is service else None
        frame=frame.f_back
    invalid.append('ancestry_limit')
    return None
def tag(owners,path):
    assert type(owners) is tuple and all(type(value) is str for value in owners)
    return tuple(owners),str(path)
def observed(code,offset,value=None):
    global pause
    if not armed:return
    frame=sys._getframe(2)
    try:
        assert frame.f_code is code and threading.current_thread() is actor
        if code is opener.__code__:
            if frame.f_locals.get('self') is db:
                assert isinstance(value,sqlite3.Connection)
                new_sql.append(value)
            return
        caller=guarded_turn(frame.f_back)
        if caller is None:return
        guard=nearest(frame.f_back,execution_body.__code__)
        assert guard is not None and guard.f_locals.get('self') is activation._guard
        assert guard.f_locals.get('service') is service
        assert guard.f_globals is vars(admission_runtime)
        assert frame.f_code is permission_body.__code__ and frame.f_globals is vars(scope_source)
        assert scope_source._execution_permission_from_records is permission_body
        admission=nearest(frame.f_back,admit_sources.__code__)
        assert admission is not None and admission.f_locals.get('self') is activation._guard
        assert admission.f_globals is vars(admission_runtime)
        if value is None:
            owners,path=frame.f_locals['owners'],frame.f_locals['path']
            key=tag(owners,path)
            rows.append(dict(key=key,phase='start',allowed=None))
            for owner,path in guard.f_locals['observed']:
                item=tag(activation._guard.owners(owner),path)
                if item not in expected_observed:expected_observed.append(item)
        else:
            assert type(value) is bool
            if not faults and route in {'retarget-root','retarget-method','retarget-source','pause'} and value is True:
                if route=='retarget-root':writer._explicit_root=other.root
                elif route=='retarget-method':
                    def foreign(self,*args,**kwargs):foreign_dispatches.append(True);raise AssertionError('foreign run-turn invoked')
                    service.run_turn=MethodType(foreign,service)
                elif route=='retarget-source':service_source._SCOPED_RUN_TURN_ANCHOR=tuple(list(turn_anchor))
                else:pause=storage._begin_local_pause()
                faults.append(route)
            key=tag(frame.f_locals['owners'],frame.f_locals['path'])
            rows.append(dict(key=key,phase='return',allowed=value))
    except BaseException as error:
        invalid.append(type(error).__name__)
def started(code,offset):observed(code,offset)
def returned(code,offset,value):observed(code,offset,value)
monitor=sys.monitoring
tool=next(value for value in range(6) if monitor.get_tool(value) is None)
monitor.use_tool_id(tool,'stock-run-turn-source-count')
try:
    monitor.register_callback(tool,monitor.events.PY_START,started)
    monitor.register_callback(tool,monitor.events.PY_RETURN,returned)
    monitor.set_local_events(tool,permission_body.__code__,monitor.events.PY_START|monitor.events.PY_RETURN)
    monitor.set_local_events(tool,opener.__code__,monitor.events.PY_RETURN)
    assert monitor.get_events(tool)==0
    @activation.worker_guard(service)
    def invoke():
        global actor,armed,result,pause
        actor=threading.current_thread();armed=True
        try:
            result=service.run_turn(conversation_id='conversation',
                messages=[{'role':'user','content':'hello'}],
                config=AgentConfig(model='test',system_prompt=''),api_endpoint='OpenAI')
        except PermissionError:
            assert route in {'retarget-root','retarget-method','retarget-source','pause'}
        finally:
            armed=False
            if pause is not None:pause.resume()
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='run-turn-admission') as pool:
        pool.submit(invoke).result(timeout=120)
    assert actor is not None and not actor.is_alive()
finally:
    armed=False
    try:
        monitor.set_local_events(tool,permission_body.__code__,0)
        monitor.set_local_events(tool,opener.__code__,0)
        monitor.register_callback(tool,monitor.events.PY_START,None)
        monitor.register_callback(tool,monitor.events.PY_RETURN,None)
        assert monitor.get_events(tool)==0
    finally:
        monitor.free_tool_id(tool)
        if 'run_turn' in vars(service):del service.run_turn
        service_source._SCOPED_RUN_TURN_ANCHOR=turn_anchor
        run_log.resolve_log_root=original_selector
        db.close()
        manager.close('chat-a');manager.close('chat-b')
        assert manager.wait_for_cleanup(timeout_seconds=2.0)
assert not invalid,invalid
assert rows and expected_observed
assert all(current(row) for row in records) and current(opener_record)
assert all(path.read_bytes()==raw for path,raw in source_bytes.items())
assert all(module.__file__==item[0] and module.__spec__ is item[1] and module.__spec__.origin==item[2] for module,item in origins.items())
assert admission_runtime.execution_scope is scope_factory and scope_source.execution_scope is scope_factory
assert scope_source._execution_permission_from_records is permission_body
assert admission_runtime.RecoveryAdmissionGuard._admit_sources is admit_sources
assert AgentService.run_turn is turn_wrapper and service_source._SCOPED_RUN_TURN_ANCHOR is turn_anchor
assert not foreign_dispatches
if route in {'stock','custom-selector'}:
    assert result is not None and result[1].status==RUN_DONE and provider_calls==[True]
    assert writer.is_active and published==[(result[0],snapshot.root)]
    assert new_sql,'original worker never opened its real RunsDB SQL resource'
else:
    assert faults==[route] and not provider_calls and not published and not writer.is_active
    assert not list(other.root.glob('.agent-runs/*'))
for connection in new_sql:
    try:sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:pass
    else:raise AssertionError('worker native SQLite resource did not physically retire')
with storage._lock:
    startup=set(storage._startups.values())
    assert not (storage._live_leases-startup)
    assert not storage._operations and not storage._raw_operations
    assert not storage._pending_acquisitions and not storage._retiring_holds
assert not blocked_attempts() and not real_profile_guard._violations
assert monitor.get_events(tool)==0 and monitor.get_tool(tool) is None
counts={}
for row in rows:
    if row['phase']=='start':counts[row['key']]=counts.get(row['key'],0)+1
if route=='pause':
    assert set(counts).issubset(set(expected_observed)) and counts, 'pause lost the actually entered source set'
else:
    assert set(counts)==set(expected_observed), 'actual original observed path set was not fully counted'
receipt=dict(route=route,source_current=True,physical_cleanup_checked=True,
    sql_handles=len(new_sql),counts=[dict(owners=key[0],path_basename=Path(key[1]).name,
    path_sha256=hashlib.sha256(key[1].encode()).hexdigest(),count=value) for key,value in counts.items()],
    scope_starts=sum(counts.values()),distinct_scope_keys=len(counts),faults=faults,
    observed_boundary='_execution_permission_from_records',
    returned_allowed=[row['allowed'] for row in rows if row['phase']=='return'],
    guard_callbacks_replaced=False,global_events=0,tool_retired=True,invalid=invalid,
    source_hashes={str(path):hashlib.sha256(raw).hexdigest() for path,raw in source_bytes.items()})
(base/'nested-agent-admission-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:receipt[key] for key in ('route','scope_starts','distinct_scope_keys','sql_handles','faults','tool_retired')},sort_keys=True))
if route=='stock':
    assert all(value==1 for value in counts.values()), 'stock run-turn revalidated already observed owner/path set'
    assert any(Path(key[1])==snapshot.root and 'agents.history' in key[0] for key in counts)
elif route=='custom-selector':
    assert selector_calls==[((),{})]
    assert any(value>1 for value in counts.values()), 'preceding scoped=None protected selector route changed'
print('retired and reopened')

"""


@pytest.mark.parametrize(
    "route",
    [
        "stock",
        "custom-selector",
        "retarget-root",
        "retarget-method",
        "retarget-source",
        "pause",
    ],
)
def test_stock_run_turn_enters_each_source_once(tmp_path, route):
    _run(tmp_path, route, "ownership", script=_SCRIPT, timeout=180)
    receipt = json.loads(
        (tmp_path / "nested-agent-admission-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["source_current"] and receipt["physical_cleanup_checked"]
    assert receipt["global_events"] == 0 and receipt["tool_retired"]
