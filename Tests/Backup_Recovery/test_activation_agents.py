"""Installed agent effects require local review of their actual restored sources."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_mcp_recovery_review import _SETUP as _MCP_SETUP

_SCRIPT = r"""
import os, sys, types
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
sys.modules.setdefault('parakeet_mlx', types.ModuleType('parakeet_mlx'))
route, state = sys.argv[1:]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
base = selector.parent.parent
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="' + str(base/'data') + '"\n[agents]\nmax_live_subagents=1\n')
selector.chmod(0o600)
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.agent_models import AgentConfig, AgentDefinition, RUN_DONE
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
data = config.get_user_data_dir()
db = AgentRunsDB(data/'agent_runs.db', client_id='test')
definition = AgentDefinition(name='saved', description='saved definition', instructions='saved')
db.create_agent_definition(definition)
db.close()
root = bootstrap.default_bootstrap_root()
startup = storage._startups.pop((os.getpid(), str(root)), None)
if startup is not None: startup.close()
authority = admission_authority(root)
tokenizers = base/'home'/'.config'/'tldw_cli'/'tokenizers'
tokenizers.mkdir(mode=0o700)
authority.register('profile', (selector.parent, base/'data', tokenizers))
control = base/'operation'
control.mkdir(mode=0o700)
owners = ('config','db.agent_runs','agents.history','db.workspaces','db.chachanotes.primary','mcp.local','mcp.permissions')
if state != 'ordinary':
    register_pending(root, 'restore', ('profile',), control, (selector,))
    with authority.maintenance(('profile',), 2) as session:
        bind_activation(root, 'restore', selector, 'generation', owners, session=session)
    (root/('pending-'+bootstrap._key('restore')+'.json')).unlink()
    activation = ActivationStore(control/'activation')
    for owner in owners:
        if state == 'approved' or (state == 'config_only' and owner == 'config') or (state == 'other_only' and owner != 'db.agent_runs') or (state == 'permission_missing' and owner != 'mcp.permissions'):
            activation.approve('generation', owner)
    if state == 'missing': (activation._generation('generation')/'required.json').unlink()
    if state == 'corrupt': (activation._generation('generation')/'required.json').write_bytes(b'{')
if state == 'shared':
    other = base/'other.toml'
    other.write_text('[general]\n')
    other.chmod(0o600)
    os.environ['TLDW_CONFIG_PATH'] = str(other)
denied = state not in ('ordinary','approved')
effects=[]
def chat(**kwargs):
    effects.append('provider')
    return {'choices':[{'message':{'content':'completed'}}]}
service = AgentService(db, ToolCatalogRegistry(), chat_call=chat)
kwargs = dict(conversation_id='conversation', messages=[{'role':'user','content':'hello'}], config=AgentConfig(model='test', system_prompt=''), api_endpoint='OpenAI')
if route == 'inspection':
    assert db.list_agent_definitions()[0]['name'] == 'saved'
    assert not db.list_runs('conversation')
else:
    try:
        if route == 'bridge':
            from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
            from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
            bridge = ConsoleAgentBridge(agent_runs_db=db, store=ConsoleChatStore(), provider_gateway=None)
            bridge.run_reply(conversation_id='conversation', session_id='session', resolution=None, assistant_message_id='reply', model='test', session_system_prompt='', agent_messages=[], should_cancel=lambda:False, resume_provider_continuation=True)
        else:
            run_id, outcome = service.run_turn(**kwargs, resume_provider_continuation=route=='resume')
            if not denied: assert outcome.status == RUN_DONE, outcome
    except PermissionError as exc:
        assert denied and str(exc) == 'agent_activation_required', str(exc)
    else:
        assert not denied, 'inactive agent execution succeeded'
    if denied:
        assert not effects
        assert not db.list_runs('conversation'), 'inactive execution created a run'
    else:
        assert effects == ['provider']
        assert db._thread_local.conn.execute('SELECT 1').fetchone()[0] == 1
db.close()
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["turn", "resume", "bridge", "inspection"])
def test_restored_agent_execution_is_inert(tmp_path, route):
    _run(tmp_path, route, "inactive", script=_SCRIPT)


@pytest.mark.parametrize(
    "state",
    [
        "ordinary",
        "approved",
        "config_only",
        "other_only",
        "missing",
        "corrupt",
        "shared",
    ],
)
def test_agent_actual_source_requires_paired_owner_review(tmp_path, state):
    _run(tmp_path, "turn", state, script=_SCRIPT)


_WORKERS = (
    _SCRIPT.split("effects=[]")[0]
    + r"""
import threading
from tldw_chatbook.Agents.agent_models import ToolResult, ToolCall, ToolSchema, ToolCatalogEntry, RunBudget
from tldw_chatbook.Agents.agent_service import _call_with_timeout
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
entered = threading.Event()
release = threading.Event()
finished = threading.Event()
threads = []
effect_file = data/'late-effect.txt'
def effect():
    threads.append(threading.current_thread())
    entered.set()
    assert release.wait(10)
    effect_file.write_text('completed after waiter')
    finished.set()
    return ToolResult(ok=True, content='completed')
if route in ('timeout','cancel'):
    result = _call_with_timeout(effect, .2, 'delayed', should_cancel=lambda:route=='cancel')
    assert not result.ok
    assert entered.is_set(), result
    assert not effect_file.exists()
    try:
        with authority.maintenance(('profile',), .04):
            raise AssertionError('timed-out live worker lost native admission')
    except AdmissionTimeout: pass
    pause=storage._begin_local_pause()
    try:
        fresh = _call_with_timeout(lambda: ToolResult(ok=True, content='unexpected'), 2, 'fresh')
        assert not fresh.ok and 'agent_activation_required' in fresh.error
    finally: pause.resume()
    release.set()
    threads[0].join(5)
    assert finished.is_set() and effect_file.read_text() == 'completed after waiter'
    with authority.maintenance(('profile',), 2): pass
else:
    raise AssertionError(route)
db.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["timeout", "cancel"])
def test_agent_native_tool_worker_retains_admission_after_waiter(tmp_path, route):
    _run(tmp_path, route, "approved", script=_WORKERS)


_FLEET = (
    _SCRIPT.split("effects=[]")[0]
    + r"""
import threading, sqlite3, time
from Tests.Agents.test_agent_service import FleetChat, fence
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
from tldw_chatbook.Tools import workspace_file_roots
entered=threading.Event()
release=threading.Event()
settled=threading.Event()
finish_release=threading.Event()
connections=[]
threads=[]
def child_reply():
    threads.append(threading.current_thread())
    entered.set()
    assert release.wait(10)
    return 'child complete'
def parent_reply():
    assert entered.wait(10)
    return 'parent complete'
def on_settled(run_id, status):
    assert db.get_run(run_id)['status'] == RUN_DONE
    connections.append(db._thread_local.conn)
    settled.set()
    assert finish_release.wait(10)
chat=FleetChat([fence('spawn_subagent', {'task':'child'}), parent_reply], {'child':[child_reply]})
service=AgentService(db, ToolCatalogRegistry(), chat_call=chat, fleet_coordinator=FleetCoordinator(max_live=3, clock=time.monotonic), on_child_settled=on_settled)
run_id, outcome=service.run_turn(conversation_id='conversation', messages=[{'role':'user','content':'delegate'}], config=AgentConfig(model='test',system_prompt='', allowed_tools=('spawn_subagent',), native_tools=False), api_endpoint='llama_cpp')
assert outcome.status == RUN_DONE and outcome.subagents_spawned == 1, outcome
db.close()
registry=workspace_file_roots._default_registry_instance
if registry is not None: registry.db.close()
startup=storage._startups.pop((os.getpid(), str(root)), None)
if startup is not None: startup.close()
release.set()
assert settled.wait(10)
try:
    with authority.maintenance(('profile',), .04):
        raise AssertionError('child released admission before terminal callback finished')
except AdmissionTimeout: pass
finish_release.set()
threads[0].join(5)
assert not threads[0].is_alive()
for connection in connections:
    try: sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError as exc: assert 'closed' in str(exc)
    else: raise AssertionError('completed child retained native AgentRunsDB connection')
with authority.maintenance(('profile',),2): pass
assert db.get_run(run_id)['status'] == RUN_DONE
assert any(row['status']==RUN_DONE and row['agent_kind']=='subagent' for row in db.list_runs('conversation'))
db.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_fleet_child_retains_scope_through_terminal_persistence_and_retires_native_db(
    tmp_path,
):
    _run(tmp_path, "fleet", "approved", script=_FLEET)


_BRIDGE = (
    _SCRIPT.split("effects=[]")[0]
    + r"""
import threading
from Tests.Chat.test_console_agent_bridge import _bridge_with_gateway, _run as run_bridge, _ChunkGateway
from tldw_chatbook.Chat import console_agent_bridge as bridge_module
from tldw_chatbook.Tools import workspace_file_roots
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
entered=threading.Event()
release=threading.Event()
threads=[]
effect_file=data/'late-provider.txt'
class Gateway:
    async def stream_chat(self, resolution, messages, **kwargs):
        threads.append(threading.current_thread())
        entered.set()
        assert release.wait(10)
        effect_file.write_text('provider completed')
        yield 'late answer'
gateway = _ChunkGateway([['answer']]) if route == 'bridge_positive' else Gateway()
bridge, bridge_db, chat_store, session, aid = _bridge_with_gateway(data/'bridge', gateway)
if route == 'bridge_timeout':
    bridge_module._CHAT_CALL_TIMEOUT_SECONDS=.2
    bridge_module._LOOP_THREAD_JOIN_SECONDS=.04
outcome=run_bridge(bridge,chat_store,session,aid)
# The controller owns these caller-thread connections; close explicitly here
# to isolate the native provider's admitted lifetime from that separate owner.
bridge_db.close()
registry=workspace_file_roots._default_registry_instance
if registry is not None: registry.db.close()
startup=storage._startups.pop((os.getpid(), str(root)), None)
if startup is not None: startup.close()
if route == 'bridge_positive':
    assert outcome.status == RUN_DONE and outcome.final_text == 'answer', outcome
else:
    assert outcome.status == 'error' and entered.is_set(), outcome
    assert not effect_file.exists()
    try:
        with authority.maintenance(('profile',), .04):
            raise AssertionError('live Console provider lost admission after timeout')
    except AdmissionTimeout: pass
    release.set()
    threads[0].join(5)
    assert not threads[0].is_alive() and effect_file.read_text() == 'provider completed'
with authority.maintenance(('profile',), 2): pass
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["bridge_positive", "bridge_timeout"])
def test_console_provider_scope_lasts_until_actual_completion(tmp_path, route):
    _run(tmp_path, route, "approved", script=_BRIDGE)


_PERMISSION = (
    _MCP_SETUP.replace(
        "('config','mcp.local','mcp.permissions','mcp.context','mcp.targets')",
        "('config','mcp.local','mcp.permissions','mcp.context','mcp.targets',"
        "'db.agent_runs','agents.history','db.workspaces','db.chachanotes.primary')",
    )
    + r"""
from Tests.Chat.test_console_agent_bridge import _bridge_with_gateway, _run as run_bridge, _ChunkGateway
from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
for owner in ('config','db.agent_runs','agents.history','db.workspaces','db.chachanotes.primary'):
 activation.approve(witness['generation'],owner)
plane=plane()
assert plane.permission_store.get_global_default()=='ask'
gate=BuiltinToolGate(plane)
gateway = _ChunkGateway([['answer']])
bridge, bridge_db, chat_store, session, aid = _bridge_with_gateway(user/'bridge', gateway)
try: run_bridge(bridge,chat_store,session,aid,builtin_gate=gate)
except PermissionError as exc: assert str(exc) == 'agent_activation_required'
else: raise AssertionError('unreviewed imported tool permissions activated Console agent')
assert not bridge_db.list_runs('conv-1') and not gateway.messages_seen
assert all((user/name).read_bytes()==value for name,value in history.items())
# The exact current MCP review, after the other owners are approved, reopens it.
plane.approve_recovery_review(plane.capture_recovery_review())
from tldw_chatbook.Agents.activation import execution, _permission_sources
with execution(bridge,sources=_permission_sources(gate)):
 assert not gateway.messages_seen
assert all((user/name).read_bytes()==value for name,value in history.items())
bridge_db.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_console_imported_permissions_do_not_activate_agent(tmp_path):
    _run(tmp_path, "bridge", "permission_missing", script=_PERMISSION)


_ADMISSION = (
    _SCRIPT.split("effects=[]")[0]
    + r"""
import asyncio, threading
from tldw_chatbook.Agents.activation import execution
effects=[]
def chat(**kwargs):
    effects.append('provider')
    return {'choices':[{'message':{'content':'answer'}}]}
service=AgentService(db, ToolCatalogRegistry(), chat_call=chat)
def invoke():
    return service.run_turn(conversation_id='conversation', messages=[], config=AgentConfig(model='test',system_prompt=''), api_endpoint='llama_cpp')
if route == 'generation':
    register_pending(root,'restore-again',('profile',),control,(selector,))
    with authority.maintenance(('profile',),2) as session:
        bind_activation(root,'restore-again',selector,'generation-2',owners,session=session)
    (root/('pending-'+bootstrap._key('restore-again')+'.json')).unlink()
    try: invoke()
    except PermissionError: pass
    else: raise AssertionError('old generation approval enabled a new agent')
elif route == 'pause':
    pause=storage._begin_local_pause()
    try:
        try: invoke()
        except PermissionError: pass
        else: raise AssertionError('paused process accepted fresh agent intake')
    finally: pause.resume()
elif route == 'copied':
    async def probe():
        with execution(service):
            try: await asyncio.to_thread(invoke)
            except PermissionError: pass
            else: raise AssertionError('copied thread borrowed agent scope')
    asyncio.run(probe())
else: raise AssertionError(route)
assert not effects and not db.list_runs('conversation')
db.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["generation", "pause", "copied"])
def test_agent_fresh_intake_needs_current_native_admission(tmp_path, route):
    _run(tmp_path, route, "approved", script=_ADMISSION)


_TOOL_DB = (
    _SCRIPT.split("effects=[]")[0]
    + r"""
import threading, sqlite3
from tldw_chatbook.Agents.agent_service import _call_with_timeout
from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Tools import workspace_file_roots
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
registry=workspace_file_roots._registry_factory()
registry.db.close()
service=AgentService(db, ToolCatalogRegistry(), chat_call=lambda **kwargs: None)
entered=threading.Event()
release=threading.Event()
connections=[]
threads=[]
def effect():
    db.list_agent_definitions()
    registry.get_active_workspace()
    connections.extend([db._thread_local.conn, registry.db._thread_local.conn])
    threads.append(threading.current_thread())
    entered.set()
    assert release.wait(10)
    return ToolResult(ok=True, content='complete')
result=_call_with_timeout(effect, .5, 'database-tool', execution_owner=service)
assert not result.ok and entered.is_set()
startup=storage._startups.pop((os.getpid(), str(root)), None)
if startup is not None: startup.close()
try:
    with authority.maintenance(('profile',), .04): raise AssertionError('live tool lost admission')
except AdmissionTimeout: pass
release.set()
threads[0].join(5)
assert not threads[0].is_alive()
for connection in connections:
    try: sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError as exc: assert 'closed' in str(exc)
    else: raise AssertionError('completed tool kept its new native connection')
with authority.maintenance(('profile',),2): pass
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_tool_worker_retires_its_new_agent_and_workspace_connections(tmp_path):
    _run(tmp_path, "tool_db", "approved", script=_TOOL_DB)


_CHILD_PAUSE = (
    _SCRIPT.split("effects=[]")[0]
    + r"""
import threading, time
from Tests.Agents.test_agent_service import FleetChat, fence
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
chat=FleetChat([fence('spawn_subagent', {'task':'child'}), 'parent complete'], {'child':['forbidden']}, allow_unconsumed=True)
service=AgentService(db, ToolCatalogRegistry(), chat_call=chat, fleet_coordinator=FleetCoordinator(max_live=3, clock=time.monotonic))
original_start=threading.Thread.start
def start(thread):
    if not thread.name.startswith('fleet-'): return original_start(thread)
    pause=storage._begin_local_pause()
    try:
        original_start(thread)
        thread.join(5)
        assert not thread.is_alive()
    finally: pause.resume()
threading.Thread.start=start
run_id, outcome=service.run_turn(conversation_id='conversation', messages=[{'role':'user','content':'delegate'}], config=AgentConfig(model='test',system_prompt='', allowed_tools=('spawn_subagent',), native_tools=False), api_endpoint='llama_cpp')
assert outcome.status == RUN_DONE, outcome
assert not chat.child_calls
handles=service.fleet_snapshot()
assert len(handles)==1 and handles[0].status=='error' and 'agent_activation_required' in handles[0].error, handles
rows=db.list_runs('conversation')
assert len(rows)==2, rows
child=next(row for row in rows if row['id']==handles[0].run_id)
assert child['status']=='error', child
db.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_paused_child_intake_settles_precreated_run_without_execution(tmp_path):
    _run(tmp_path, "child_pause", "approved", script=_CHILD_PAUSE)


_BRIDGE_WORKER = (
    _SCRIPT.split("effects=[]")[0]
    + r"""
import threading, sqlite3
from Tests.Chat.test_console_agent_bridge import _bridge_with_gateway, _run as run_bridge, _ChunkGateway
from tldw_chatbook.Agents.activation import worker_guard
from tldw_chatbook.Tools import workspace_file_roots
bridge, bridge_db, chat_store, session, aid = _bridge_with_gateway(data/'bridge', _ChunkGateway([['answer']]))
registry=workspace_file_roots._registry_factory()
caller_connections=[bridge_db._thread_local.conn, registry.db._thread_local.conn]
worker_connections=[]
outcomes=[]
def reply():
    outcomes.append(run_bridge(bridge,chat_store,session,aid))
    worker_connections.extend([bridge_db._thread_local.conn, registry.db._thread_local.conn])
thread=threading.Thread(target=worker_guard(bridge)(reply))
thread.start()
thread.join(20)
assert not thread.is_alive()
assert len(outcomes)==1 and outcomes[0].status==RUN_DONE, outcomes
assert len(worker_connections)==2
for connection in worker_connections:
    try: sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError as exc: assert 'closed' in str(exc)
    else: raise AssertionError('finished Bridge worker retained its native connection')
for connection in caller_connections:
    assert connection.execute('SELECT 1').fetchone()[0] == 1
bridge_db.close()
registry.db.close()
startup=storage._startups.pop((os.getpid(), str(root)), None)
if startup is not None: startup.close()
with authority.maintenance(('profile',),2): pass
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_bridge_worker_retires_only_its_new_native_connections(tmp_path):
    _run(tmp_path, "bridge_worker", "approved", script=_BRIDGE_WORKER)
