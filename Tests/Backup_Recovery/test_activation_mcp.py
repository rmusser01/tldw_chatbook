"""Restored MCP definitions stay inert through native accepted execution."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, os, sys, types
from pathlib import Path
from types import SimpleNamespace
from Tests.network_guard import install, blocked_attempts
install()
sys.modules.setdefault('parakeet_mlx', types.ModuleType('parakeet_mlx'))
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
route, state = sys.argv[1:]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
base = selector.parent.parent
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="' + str(base/'data') + '"\n')
selector.chmod(0o600)
from tldw_chatbook import config
data = config.get_user_data_dir()
from tldw_chatbook.MCP.client import MCPClient, _StdioJSONRPCConnection
from tldw_chatbook.MCP.local_runtime_delegate import LocalMCPRuntimeDelegate
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalMCPStore, LocalExternalMCPProfile
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService
paths = {name: data/name for name in ('local','targets','context','permissions')}
for path in paths.values(): path.mkdir(mode=0o700)
local_store = LocalMCPStore(paths['local']/'local_mcp_store.json')
local_store.save_profile(LocalExternalMCPProfile(profile_id='demo', command='disposable-sentinel'))
delegate = LocalMCPRuntimeDelegate(manifest_provider=lambda: {})
client = MCPClient()
local = LocalMCPControlService(store=local_store, client=client, runtime_delegate=delegate, manifest_provider=lambda: {})
plane = UnifiedMCPControlPlaneService(target_store=ConfiguredServerTargetStore(paths['targets']/'mcp_server_targets.json'), context_store=UnifiedMCPContextStore(paths['context']/'unified_mcp_context.json'), local_service=local, server_service=SimpleNamespace())
plane._permission_store = MCPPermissionStore(paths['permissions']/'mcp_permissions.json')
root = bootstrap.default_bootstrap_root()
startup = storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None: startup.close()
authority=admission_authority(root)
shared = state in paths
restored = paths[state] if shared else data
authority.register('profile',(selector.parent,restored))
control=base/'operation'; control.mkdir(mode=0o700)
owners=('config','mcp.local','mcp.targets','mcp.context','mcp.permissions')
if state not in ('ordinary','unqualified'):
    register_pending(root,'restore',('profile',),control,(selector,))
    with authority.maintenance(('profile',),2) as session:
        bind_activation(root,'restore',selector,'generation',owners,session=session)
    (root/('pending-'+bootstrap._key('restore')+'.json')).unlink()
    activation=ActivationStore(control/'activation')
    for owner in owners:
        if state=='approved' or state=='config_only' and owner=='config': activation.approve('generation',owner)
    if state=='missing': (activation._generation('generation')/'required.json').unlink()
    if state=='corrupt': (activation._generation('generation')/'required.json').write_bytes(b'{')
if shared:
    other=base/'unrelated.toml'; other.write_text('[general]\n'); other.chmod(0o600)
    os.environ['TLDW_CONFIG_PATH']=str(other)
if state=='unqualified': storage.qualified_for=lambda *args:(False,'native_unqualified')
denied=state not in ('ordinary','approved','unqualified')
events=[]
async def effect(*args,**kwargs):
    events.append('effect')
    return SimpleNamespace(content=['ok'],contents=[SimpleNamespace(text='ok',mimeType='text/plain')],messages=[])
client.sessions['demo']=SimpleNamespace(call_tool=effect,read_resource=effect,get_prompt=effect)
async def tool(payload):
    events.append('tool'); return {'ok':True}
delegate._tool_sentinel=tool
delegate._resources=SimpleNamespace(get_note_resource=effect)
delegate._prompts=SimpleNamespace(generate_document_prompt=effect)
def env(profile):
    events.append('credential'); return {}
local._build_spawn_env=env
async def spawn(*args,**kwargs):
    events.append('spawn'); raise RuntimeError('disposable transport sentry')
asyncio.create_subprocess_exec=spawn
async def run():
    if route=='raw':
        connection=_StdioJSONRPCConnection.__new__(_StdioJSONRPCConnection)
        connection._reader_unavailable=False; connection._cleanup_complete=False
        from itertools import count
        connection._request_ids=count(1); connection._pending_requests={}
        async def send(payload):
            events.append('wire')
            connection._pending_requests[payload['id']].set_result({})
        connection._send_message=send
        return await connection.request('tools/call',{},timeout_seconds=.1)
    if route=='client': return await client.call_tool('demo','sentinel',{})
    if route=='client_read': return await client.read_resource('demo','note://x')
    if route=='client_prompt': return await client.get_prompt('demo','generate_document',{})
    if route=='connect': return await client.connect_to_server('new','disposable-sentinel')
    if route=='local_connect': return await local.connect_profile('demo')
    if route=='delegate': return await delegate.execute_tool('sentinel',{})
    if route=='read': return await delegate.read_resource('note://x')
    if route=='prompt': return await delegate.get_prompt('generate_document',{})
    if route=='request': return await delegate.request('resources/read',{'uri':'note://x'})
    if route=='batch': return await delegate.batch([{'method':'resources/read','params':{'uri':'note://x'}}])
    if route=='permission_toggle':
        plane.set_tool_state('builtin:tldw_chatbook','sentinel','allow')
        assert not activation.allowed('generation','mcp.permissions')
        return await plane.execute_advanced_tool('sentinel',{})
    if route=='plane': return await plane.execute_hub_tool('builtin:tldw_chatbook','sentinel',{})
    if route=='action': return await plane.run_action('resource.read',{'resource_uri':'note://x'})
    if route=='inspection_batch':
        result=await plane.run_action('runtime.batch',{'requests':[{'method':'tools/list'}]})
        assert result['results'][0]['ok']
        return
    if route=='inspection_request':
        result=await plane.run_action('runtime.request',{'method':'tools/list'})
        assert result['result']=={'tools':[]}
        return
    if route=='inspection':
        assert local.get_external_servers()
        assert delegate.get_status()
        assert (await delegate.request('tools/list'))=={'tools':[]}
        return
    raise AssertionError(route)
try:
    result=asyncio.run(run())
    if denied and route in ('client','client_read','client_prompt'):
        assert result=={'error':'mcp_activation_required'}
    if denied and route=='connect': assert result is False
except (PermissionError,RuntimeError) as exc:
    if route.startswith('inspection') or not denied and route not in ('connect','local_connect'): raise
if denied or route.startswith('inspection'): assert events==[],events
else: assert events,route
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    [
        "raw",
        "client",
        "client_read",
        "client_prompt",
        "connect",
        "local_connect",
        "delegate",
        "read",
        "prompt",
        "request",
        "batch",
        "plane",
        "action",
        "inspection",
        "inspection_request",
        "inspection_batch",
        "permission_toggle",
    ],
)
def test_inactive_mcp_denies_before_effect(tmp_path, route):
    _run(tmp_path, route, "inactive", script=_SCRIPT)


@pytest.mark.parametrize(
    "state",
    [
        "ordinary",
        "approved",
        "unqualified",
        "config_only",
        "missing",
        "corrupt",
        "local",
        "targets",
        "context",
        "permissions",
    ],
)
def test_mcp_actual_independent_sources(tmp_path, state):
    _run(tmp_path, "plane", state, script=_SCRIPT)


_RETENTION = (
    _SCRIPT.split("events=[]")[0]
    + r"""
import threading
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
from tldw_chatbook.MCP.activation import execution, MCPActivationRequired

def assert_held():
    try:
        with authority.maintenance(('profile',), .03):
            raise AssertionError('MCP effect lost native admission')
    except AdmissionTimeout: pass

if route in ('nested','copied','pid'):
    async def effect(payload):
        assert_held()
        with execution(plane):
            pause=storage._begin_local_pause()
            try:
                if route=='nested':
                    with execution(delegate): pass
                elif route=='pid':
                    original=os.getpid
                    try:
                        os.getpid=lambda: original()+1
                        try:
                            with execution(delegate): pass
                        except MCPActivationRequired: pass
                        else: raise AssertionError('foreign PID borrowed admission')
                    finally: os.getpid=original
                else:
                    async def child():
                        try:
                            with execution(delegate): pass
                        except MCPActivationRequired: return
                        raise AssertionError('copied task borrowed admission')
                    await asyncio.create_task(child())
                    def thread():
                        try:
                            with execution(delegate): pass
                        except MCPActivationRequired: return
                        raise AssertionError('copied thread borrowed admission')
                    await asyncio.to_thread(thread)
                assert_held()
            finally: pause.resume()
        return {'ok':True}
    delegate._tool_sentinel=effect
    asyncio.run(plane.execute_hub_tool('builtin:tldw_chatbook','sentinel',{}))
elif route in ('worker_cancel','worker_timeout'):
    from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS
    entered=threading.Event(); release=threading.Event(); finished=threading.Event()
    effects=[]
    def invoke(*args):
        entered.set()
        assert release.wait(3)
        try:
            with execution(delegate):
                assert_held()
                import subprocess
                result=subprocess.run([sys.executable,'-c',"print('disposable-worker')"],capture_output=True,text=True,check=True)
                effects.append(result.stdout.strip())
            return {'ok':True}
        finally: finished.set()
    delegate._library_service=SimpleNamespace(invoke=invoke)
    async def run():
        waiter=asyncio.create_task(plane.execute_hub_tool('builtin:tldw_chatbook',next(iter(LIBRARY_TOOL_DESCRIPTORS)),{},timeout_seconds=.2 if route=='worker_timeout' else 3))
        assert await asyncio.to_thread(entered.wait,2)
        pause=storage._begin_local_pause()
        try:
            if route=='worker_cancel': waiter.cancel()
            try: await waiter
            except asyncio.CancelledError: assert route=='worker_cancel'
            except RuntimeError as exc: assert route=='worker_timeout' and 'Timed out' in str(exc)
            else: raise AssertionError('waiter unexpectedly returned')
            assert not finished.is_set()
            assert_held()
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait,3)
            pause.resume()
    asyncio.run(run())
    assert effects==['disposable-worker'],effects
else:
    raise AssertionError(route)
with authority.maintenance(('profile',),1): pass
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize(
    "route", ["nested", "copied", "pid", "worker_cancel", "worker_timeout"]
)
def test_mcp_native_accepted_lifetime(tmp_path, route):
    _run(tmp_path, route, "approved", script=_RETENTION)


_TRANSPORT = (
    _SCRIPT.split("events=[]")[0]
    + r'''
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
from tldw_chatbook.MCP import client as client_module

def assert_held():
    try:
        with authority.maintenance(('profile',), .03):
            raise AssertionError('transport lost native admission')
    except AdmissionTimeout: pass

server = """
import json, sys, time
for line in sys.stdin:
    p=json.loads(line)
    if 'id' not in p: continue
    method=p['method']
    if method=='initialize':
        if sys.argv[1]=='hang': time.sleep(10)
        result={'protocolVersion':'2025-03-26','capabilities':{},'serverInfo':{}}
    elif method=='tools/list': result={'tools':[{'name':'sentinel','inputSchema':{'type':'object'}}]}
    elif method=='resources/list': result={'resources':[]}
    elif method=='prompts/list': result={'prompts':[]}
    elif method=='tools/call': result={'content':[{'type':'text','text':'real-wire-effect'}]}
    else: result={}
    print(json.dumps({'jsonrpc':'2.0','id':p['id'],'result':result}),flush=True)
"""
processes=[]
spawned=asyncio.Event()
original_spawn=asyncio.create_subprocess_exec
async def spawn(*args,**kwargs):
    assert_held()
    process=await original_spawn(*args,**kwargs)
    processes.append(process); spawned.set()
    return process
asyncio.create_subprocess_exec=spawn
original_send=client_module._StdioJSONRPCConnection._send_message
async def send(self,payload):
    if payload.get('method') in ('initialize','tools/list','tools/call'): assert_held()
    return await original_send(self,payload)
client_module._StdioJSONRPCConnection._send_message=send
if route=='timeout': client_module.CONNECT_TIMEOUT_SECONDS=.15
async def run():
    if route=='lifecycle':
        local_store.save_profile(LocalExternalMCPProfile(profile_id='real',command=sys.executable,args=('-u','-c',server,'ok')))
        await plane.connect_local_profile('real')
        result=await client.call_tool('real','sentinel',{})
        assert result['result'][0]['text']=='real-wire-effect'
    elif route=='cancel':
        waiter=asyncio.create_task(client.connect_to_server('real',sys.executable,['-u','-c',server,'hang']))
        await spawned.wait()
        waiter.cancel()
        try: await waiter
        except asyncio.CancelledError: pass
        else: raise AssertionError('connect swallowed cancellation')
    else:
        result=await client.connect_to_server('real',sys.executable,['-u','-c',server,'hang' if route=='timeout' else 'ok'])
        assert result is (route!='timeout')
        if result:
            assert (await client.call_tool('real','sentinel',{}))['result'][0]['text']=='real-wire-effect'
    await client.disconnect_all()
asyncio.run(run())
assert processes and all(p.returncode is not None for p in processes)
assert not client.sessions and not client._pending_connections
with authority.maintenance(('profile',),1): pass
assert not blocked_attempts()
print('retired and reopened')
'''
)


@pytest.mark.parametrize("route", ["success", "lifecycle", "timeout", "cancel"])
def test_mcp_real_stdio_admission_and_cleanup(tmp_path, route):
    _run(tmp_path, route, "approved", script=_TRANSPORT)


_REQUEST_SETTLEMENT = _TRANSPORT.split('processes=[]')[0]
_REQUEST_SETTLEMENT = _REQUEST_SETTLEMENT.replace(
    "elif method=='tools/call': result={'content':[{'type':'text','text':'real-wire-effect'}]}",
    """elif method=='tools/call':
        from pathlib import Path
        Path(sys.argv[2]).write_text('entered')
        while not Path(sys.argv[3]).exists(): time.sleep(.01)
        Path(sys.argv[4]).write_text('finished')
        result={'content':[]}""",
) + r'''
async def run():
    entered=base/'entered'; release=base/'release'; finished=base/'finished'
    assert await client.connect_to_server('real',sys.executable,['-u','-c',server,'ok',str(entered),str(release),str(finished)])
    session=client.sessions['real']
    process=session.process
    session.request_timeout_seconds=.1
    try:
        waiter=asyncio.create_task(client.call_tool('real','sentinel',{}))
        for _ in range(200):
            if entered.exists(): break
            await asyncio.sleep(.01)
        assert entered.exists()
        if route=='cancel': waiter.cancel()
        try: result=await waiter
        except asyncio.CancelledError: assert route=='cancel'
        else: assert 'Timed out' in result['error'],result
        # Removing a request future is not completion of the child operation.
        assert process.returncode is not None, 'accepted tool still runs after waiter release'
        assert not finished.exists()
        with authority.maintenance(('profile',),.2):
            release.write_text('go')
            await asyncio.sleep(.05)
            assert not finished.exists()
        assert 'real' not in client.sessions
    finally:
        release.write_text('go')
        await client.disconnect_all()
asyncio.run(run())
assert not blocked_attempts()
print('retired and reopened')
'''


@pytest.mark.parametrize('route', ['timeout', 'cancel'])
def test_mcp_established_request_settles_before_admission_release(tmp_path, route):
    _run(tmp_path, route, 'approved', script=_REQUEST_SETTLEMENT)


_CONCURRENT_REQUESTS = _REQUEST_SETTLEMENT.split('async def run():')[0]
_CONCURRENT_REQUESTS += r'''
from tldw_chatbook.MCP import client as client_module
server=server.replace('import json, sys, time', 'import json, sys, time, signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)')
async def run():
    entered=base/'entered'; release=base/'release'; finished=base/'finished'
    assert await client.connect_to_server('real',sys.executable,['-u','-c',server,'ok',str(entered),str(release),str(finished)])
    session=client.sessions['real']; process=session.process
    session.request_timeout_seconds=10
    client_module._TERMINATE_TIMEOUT_SECONDS=.4
    sent=asyncio.Event(); stopping=asyncio.Event(); requests=[]
    original_send=session._send_message
    async def send(payload):
        await original_send(payload)
        if payload.get('method')=='tools/call':
            requests.append(payload['id'])
            if len(requests)==2: sent.set()
    session._send_message=send
    original_terminate=process.terminate
    def terminate():
        stopping.set(); original_terminate()
    process.terminate=terminate
    first=asyncio.create_task(client.call_tool('real','sentinel',{}))
    try:
        for _ in range(200):
            if entered.exists(): break
            await asyncio.sleep(.01)
        assert entered.exists()
        second=asyncio.create_task(client.call_tool('real','sentinel',{}))
        await sent.wait()
        first.cancel()
        await stopping.wait()
        await asyncio.sleep(.02)
        assert process.returncode is None
        assert not first.done() and not second.done(), 'sibling released before native exit'
        assert_held()
        if route=='repeated':
            first.cancel(); second.cancel()
            await asyncio.sleep(.02)
            assert not first.done() and not second.done()
            assert_held()
        results=await asyncio.gather(first,second,return_exceptions=True)
        assert isinstance(results[0],asyncio.CancelledError)
        if route=='repeated': assert isinstance(results[1],asyncio.CancelledError)
        else: assert 'MCP connection closed' in results[1]['error'],results
        assert process.returncode is not None
        assert not finished.exists()
        assert not client.sessions and not client._pending_connections
    finally:
        release.write_text('go')
        await client.disconnect_all()
asyncio.run(run())
with authority.maintenance(('profile',),1): pass
assert not blocked_attempts()
print('retired and reopened')
'''


@pytest.mark.parametrize('route', ['sibling', 'repeated'])
def test_mcp_interrupted_sibling_requests_wait_for_real_exit(tmp_path, route):
    _run(tmp_path, route, 'approved', script=_CONCURRENT_REQUESTS)


_FAILED_STOP = _REQUEST_SETTLEMENT.split('async def run():')[0] + r'''
from tldw_chatbook.MCP import client as client_module
async def run():
    entered=base/'entered'; release=base/'release'; finished=base/'finished'
    assert await client.connect_to_server('real',sys.executable,['-u','-c',server,'ok',str(entered),str(release),str(finished)])
    session=client.sessions['real']; process=session.process
    session.request_timeout_seconds=.05
    client_module._TERMINATE_TIMEOUT_SECONDS=.02
    attempted=asyncio.Event()
    original_terminate=process.terminate; original_kill=process.kill
    def refused():
        attempted.set(); raise PermissionError('disposable signal refusal')
    process.terminate=refused; process.kill=refused
    waiter=asyncio.create_task(client.call_tool('real','sentinel',{}))
    try:
        await attempted.wait()
        await asyncio.sleep(.1)
        assert entered.exists() and not finished.exists()
        assert process.returncode is None and not waiter.done()
        assert_held()
        release.write_text('go')
        result=await asyncio.wait_for(waiter,2)
        assert 'Timed out' in result['error'],result
        assert finished.exists() and process.returncode is not None
    finally:
        process.terminate=original_terminate; process.kill=original_kill
        release.write_text('go')
        await client.disconnect_all()
asyncio.run(run())
with authority.maintenance(('profile',),1): pass
assert not blocked_attempts()
print('retired and reopened')
'''


def test_mcp_failed_termination_retains_admission_until_natural_exit(tmp_path):
    _run(tmp_path, 'failed_stop', 'approved', script=_FAILED_STOP)


_RESPONSE_OUTCOMES = _REQUEST_SETTLEMENT.split('async def run():')[0] + r'''
if route=='server_error':
    server=server.replace("elif method=='tools/call':", """elif method=='tools/call':
        print(json.dumps({'jsonrpc':'2.0','id':p['id'],'error':{'code':-32000,'message':'completed server refusal'}}),flush=True)
        continue
    elif method=='unused':""")
else:
    server=server.replace("Path(sys.argv[2]).write_text('entered')", "Path(sys.argv[2]).write_text('entered')\n        import os; os.close(1)")
async def run():
    entered=base/'entered'; release=base/'release'; finished=base/'finished'
    assert await client.connect_to_server('real',sys.executable,['-u','-c',server,'ok',str(entered),str(release),str(finished)])
    session=client.sessions['real']; process=session.process
    try:
        result=await client.call_tool('real','sentinel',{})
        if route=='server_error':
            assert 'completed server refusal' in result['error']
            assert process.returncode is None and client.sessions['real'] is session
            assert (await session.list_tools()).tools[0].name=='sentinel'
        else:
            assert 'MCP transport unavailable' in result['error'],result
            assert process.returncode is not None and not finished.exists()
    finally:
        release.write_text('go')
        await client.disconnect_all()
    assert process.returncode is not None
asyncio.run(run())
with authority.maintenance(('profile',),1): pass
assert not blocked_attempts()
print('retired and reopened')
'''


@pytest.mark.parametrize('route', ['server_error', 'transport_loss'])
def test_mcp_response_completion_and_native_transport_loss(tmp_path, route):
    _run(tmp_path, route, 'approved', script=_RESPONSE_OUTCOMES)


_SHUTDOWN_SETTLEMENT = _FAILED_STOP.split('async def run():')[0] + r'''
import threading, time, signal
shutdown=threading.Event()
observed=[]
process=None

def during_shutdown():
    assert shutdown.wait(3)
    # Let normal asyncio.run shutdown cancel all its outstanding tasks.
    time.sleep(.15)
    try:
        with authority.maintenance(('profile',),.1):
            observed.append('maintenance entered with active child')
            (base/'release').write_text('go')
            for _ in range(200):
                if (base/'finished').exists(): break
                time.sleep(.01)
    except AdmissionTimeout:
        observed.append('admission retained')
    finally:
        (base/'release').write_text('go')

async def run():
    global process
    entered=base/'entered'; release=base/'release'; finished=base/'finished'
    assert await client.connect_to_server('real',sys.executable,['-u','-c',server,'ok',str(entered),str(release),str(finished)])
    session=client.sessions['real']; process=session.process
    session.request_timeout_seconds=.05
    client_module._TERMINATE_TIMEOUT_SECONDS=.02
    def refused(): raise PermissionError('disposable signal refusal')
    process.terminate=refused; process.kill=refused
    waiter=asyncio.create_task(client.call_tool('real','sentinel',{}))
    await asyncio.sleep(.3)
    assert entered.exists() and not finished.exists()
    assert process.returncode is None and not waiter.done()
    assert_held()
    shutdown.set()
    # Returning normally makes asyncio.run cancel the settlement task itself.
observer=threading.Thread(target=during_shutdown)
observer.start()
try:
    asyncio.run(run())
finally:
    shutdown.set()
    observer.join(4)
    if process is not None and process.returncode is None:
        try: os.kill(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
assert not observer.is_alive()
assert observed==['admission retained'],observed
assert process.returncode is not None
assert (base/'finished').exists()
with authority.maintenance(('profile',),1): pass
assert not blocked_attempts()
print('retired and reopened')
'''


def test_mcp_asyncio_run_shutdown_retains_cancelled_settlement(tmp_path):
    _run(tmp_path, 'shutdown', 'approved', script=_SHUTDOWN_SETTLEMENT)
