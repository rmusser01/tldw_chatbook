"""Concrete sync storage and idle local MCP subprocess maintenance boundaries."""

import pytest

from Tests.Backup_Recovery.test_activation_mcp import _SCRIPT as _MCP_ORDINARY_SETUP
from Tests.Backup_Recovery.test_activation_mcp import _TRANSPORT
from Tests.Backup_Recovery.test_mcp_recovery_review import _APPROVED_SETUP
from Tests.Backup_Recovery.test_activation_network_sync import _SCRIPT as _SYNC_SETUP
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SYNC = (
    _SYNC_SETUP.split("effects = []")[0]
    + r"""
import time
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.runtime_maintenance import _bind, _settle_stage, _resume_hooks
from Tests.Sync_Interop.test_local_first_sync_service import FakeLocalFirstServer
server = FakeLocalFirstServer(pull_envelopes=[])
service = LocalFirstSyncService(server_service=server, state_repository=repo,
    local_store=InMemoryNotesStore(), dataset_keys={'dataset-1':b'x'*32})
entered, release = asyncio.Event(), asyncio.Event()
original = server.pull_v2_envelopes
async def pull(**kwargs):
    entered.set()
    await release.wait()
    if route == 'error': raise RuntimeError('deliberate transport failure')
    return await original(**kwargs)
server.pull_v2_envelopes = pull
async def run():
    accepted = asyncio.create_task(service.sync_once(**scope_args, domains=['notes']))
    await asyncio.wait_for(entered.wait(), 5)
    closed = []
    hook = _bind(service, 'Sync_Interop.local_first_sync_service', 'LocalFirstSyncService')
    waiting = asyncio.create_task(_settle_stage([hook], closed, time.monotonic()+5))
    await asyncio.sleep(0)
    try:
        await service.sync_once(**scope_args, domains=['notes'])
    except RecoveryRequired: pass
    else: raise AssertionError('new sync admitted after intake closed')
    if route in ('timeout','cancel'):
        if route == 'cancel':
            waiting.cancel()
            try: await waiting
            except asyncio.CancelledError: pass
        else:
            waiting.cancel()
            try: await waiting
            except asyncio.CancelledError: pass
            assert not await service._maintenance_drain(time.monotonic())
        try: await _resume_hooks(closed)
        except RecoveryRequired: pass
        else: raise AssertionError('unsettled sync silently reopened')
        assert closed and not accepted.done()
    release.set()
    if route == 'error':
        try: await accepted
        except RuntimeError as error: assert str(error) == 'deliberate transport failure'
        else: raise AssertionError('failure missing')
    else:
        await accepted
    if route not in ('timeout','cancel'): await waiting
    assert await service._maintenance_drain(time.monotonic()+1)
    state = repo.get_sync_v2_profile_state(**scope_args)
    if route == 'error': assert state['last_error']
    else: assert state['dataset_cursors']['sync_v2'] == '9'
    repo.close()
    pause = storage._begin_local_pause()
    assert pause.drain(time.monotonic()+1)
    pause.resume()
    await _resume_hooks(closed)
    assert not closed
asyncio.run(run())
assert not blocked_attempts()
print('retired and reopened')
"""
)

_MCP = (
    _TRANSPORT.split("async def run():")[0]
    + r"""
import time
from tldw_chatbook.Backup_Recovery.runtime_maintenance import _bind, _settle_stage, _resume_hooks
async def run():
    assert await client.connect_to_server('real',sys.executable,['-u','-c',server,'ok'])
    process = processes[-1]
    assert process.returncode is None
    closed=[]
    hook=_bind(client,'MCP.client','MCPClient')
    await _settle_stage([hook],closed,time.monotonic()+5)
    assert process.returncode is not None
    assert not client.sessions and not client._pending_connections
    with authority.maintenance(tuple(witness['namespaces']),1): pass
    await _resume_hooks(closed)
    assert len(processes)==1, 'resume replayed a saved executable definition'
    assert await client.connect_to_server('real',sys.executable,['-u','-c',server,'ok'])
    assert len(processes)==2
    await client.disconnect_all()
asyncio.run(run())
assert all(p.returncode is not None for p in processes)
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["success", "error", "timeout", "cancel"])
def test_sync_accepted_result_commits_before_repository_pause(tmp_path, route):
    _run(tmp_path, route, "ordinary", script=_SYNC)


def test_idle_established_mcp_child_exits_before_maintenance(tmp_path):
    _run(tmp_path, "success", "approved", script=_MCP)


_MCP_REFUSAL = (
    _MCP.split("async def run():")[0]
    + r"""
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
async def run():
    assert await client.connect_to_server('real',sys.executable,['-u','-c',server,'ok'])
    session=client.sessions['real']
    process=session.process
    entered, release=asyncio.Event(),asyncio.Event()
    original=client._finish_connection_cleanup
    async def cleanup(*args):
        entered.set()
        await release.wait()
        if route=='failure' and process.returncode is None:
            raise RuntimeError('injected native cleanup refusal')
        if route=='uncertain':
            client.sessions.clear()
            client.servers.clear()
            return
        await original(*args)
    client._finish_connection_cleanup=cleanup
    closed=[]
    hook=_bind(client,'MCP.client','MCPClient')
    waiting=asyncio.create_task(_settle_stage([hook],closed,time.monotonic()+5))
    await entered.wait()
    waiting.cancel()
    try: await waiting
    except asyncio.CancelledError: pass
    assert process.returncode is None and not client._maintenance_cleanup.done()
    try: await _resume_hooks(closed)
    except RecoveryRequired: pass
    else: raise AssertionError('unsettled native cleanup reopened')
    assert not await client._maintenance_drain(time.monotonic())
    release.set()
    if route in ('uncertain','failure'):
        try: await client._maintenance_cleanup
        except RuntimeError: assert route=='failure'
        assert not await client._maintenance_drain(time.monotonic()+1)
        assert process.returncode is None
        if route=='uncertain': assert not client.sessions
        try: await _resume_hooks(closed)
        except RecoveryRequired: pass
        else: raise AssertionError('empty map substituted for native exit')
        await session.close()
    assert await client._maintenance_drain(time.monotonic()+5)
    assert process.returncode is not None
    await _resume_hooks(closed)
    assert not closed and len(processes)==1
asyncio.run(run())
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["cancel", "uncertain", "failure"])
def test_mcp_cancelled_or_uncertain_cleanup_keeps_capture_closed(tmp_path, route):
    _run(tmp_path, route, "approved", script=_MCP_REFUSAL)


def _ordinary_mcp_script(script):
    """Keep transport probes on the actual ordinary lease's observed namespace."""
    ordinary = _MCP_ORDINARY_SETUP.split("events=[]")[0]
    ordinary += (
        "\nwith storage.acquire_storage(selector) as lease:\n"
        " witness={'namespaces':lease.execution_scope()[1]}\n"
    )
    assert script.startswith(_APPROVED_SETUP)
    return ordinary + script[len(_APPROVED_SETUP) :]


def test_ordinary_idle_mcp_resume_requires_explicit_connect(tmp_path):
    # Use actual ordinary sources while retaining the same native-hold probes.
    script = _ordinary_mcp_script(_MCP)
    _run(tmp_path, "success", "ordinary", script=script)


def test_actual_app_drains_sync_tail_and_idle_mcp_before_storage(tmp_path):
    from Tests.Backup_Recovery.test_runtime_startup_handoff import _SCRIPT

    setup = r'''
    from pathlib import Path
    from tldw_chatbook.Sync_Interop.notes_local_store import InMemoryNotesStore
    scopes = dict(server_profile_id='accepted', authenticated_principal_id='user', workspace_scope='workspace')
    app.sync_state_repository.set_sync_v2_profile_state(**scopes,
        profile_mode='local_first', device_id='device', dataset_id='dataset',
        dataset_cursors={'sync_v2':'7'}, capabilities={'supported_domains':['notes']},
        dry_run_metadata={'dry_run':True})
    app.sync_v2_dataset_keys['dataset']=b'x'*32
    app.local_first_sync_service.local_store=InMemoryNotesStore()
    sync_entered,sync_release=asyncio.Event(),asyncio.Event()
    class NativeBoundaryClient:
        async def pull_sync_v2_envelopes(self, **kwargs):
            sync_entered.set()
            await sync_release.wait()
            return {'dataset_id':'dataset','envelopes':[],'next_cursor':'9','has_more':False}
    app.server_sync_service.client=NativeBoundaryClient()
    app.server_sync_service.policy_enforcer=None
    sync=asyncio.create_task(app.manual_sync_control_service.run_once(**scopes,domains=['notes']))
    await asyncio.wait_for(sync_entered.wait(),5)
    client=app.local_mcp_control_service._get_client()
    child_source="""
import json,sys
for line in sys.stdin:
 p=json.loads(line)
 if 'id' not in p: continue
 m=p['method']
 r=({'protocolVersion':'2025-03-26','capabilities':{},'serverInfo':{}} if m=='initialize' else
    {'tools':[{'name':'sentinel','inputSchema':{'type':'object'}}]} if m=='tools/list' else {'resources':[]} if m=='resources/list' else {'prompts':[]})
 print(json.dumps({'jsonrpc':'2.0','id':p['id'],'result':r}),flush=True)
"""
    assert await client.connect_to_server('native',sys.executable,['-u','-c',child_source])
    child=client.sessions['native'].process
    async def finish_sync():
        while not app.manual_sync_control_service._producer_lifetime.closed:
            await asyncio.sleep(.001)
        assert not app.local_first_sync_service._producer_lifetime.closed
        assert not app.server_sync_service._producer_lifetime.closed
        sync_release.set()
    finish=asyncio.create_task(finish_sync())
'''
    verify = r"""
        await finish
        result=await sync
        assert result.status=='success',result
        assert app.sync_state_repository.get_sync_v2_profile_state(**scopes)['dataset_cursors']['sync_v2']=='9'
        assert child.returncode is not None
        assert client._producer_lifetime.closed
"""
    script = _SCRIPT.replace(
        "    runtime = RuntimeMaintenance(app)",
        setup + "\n    runtime = RuntimeMaintenance(app)",
    )
    script = script.replace(
        "        runtime.retire_local_caches()",
        verify + "\n        runtime.retire_local_caches()",
    )
    script = script.replace(
        "        assert not errors",
        "        assert not client._producer_lifetime.closed\n        assert not client.sessions\n        assert not app.manual_sync_control_service._producer_lifetime.closed\n        assert not errors",
    )
    _run(tmp_path, "startup", "resume", script=script)


_MONITOR_BUSY = r"""
import asyncio,sys,threading,time
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'): sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import runtime_maintenance as maintenance
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Sync_Interop.notes_local_store import InMemoryNotesStore
async def main():
 app=TldwCli()
 scopes=dict(server_profile_id='accepted',authenticated_principal_id='user',workspace_scope='workspace')
 app.sync_state_repository.set_sync_v2_profile_state(**scopes,profile_mode='local_first',
  device_id='device',dataset_id='dataset',dataset_cursors={'sync_v2':'7'},
  capabilities={'supported_domains':['notes']},dry_run_metadata={'dry_run':True})
 app.sync_v2_dataset_keys['dataset']=b'x'*32
 app.local_first_sync_service.local_store=InMemoryNotesStore()
 entered,release=asyncio.Event(),asyncio.Event()
 class API:
  async def pull_sync_v2_envelopes(self,**kwargs):
   entered.set(); await release.wait()
   return {'dataset_id':'dataset','envelopes':[],'next_cursor':'9','has_more':False}
 app.server_sync_service.client=API()
 app.server_sync_service.policy_enforcer=None
 accepted=asyncio.create_task(app.manual_sync_control_service.run_once(**scopes,domains=['notes']))
 await asyncio.wait_for(entered.wait(),5)
 requested=True
 original_request=storage._local_pause_requested
 storage._local_pause_requested=lambda:requested
 native_entered,native_release=threading.Event(),threading.Event()
 retrying=asyncio.Event()
 original_readmit=maintenance._readmit_native_holds
 def held_readmit(holds):
  original_readmit(holds)
  # Deliver cancellation during the real readmission wait, then retry the
  # still-busy sync hook before releasing its accepted operation.
  if sys.argv[2]=='cancel' and not native_entered.is_set():
   native_entered.set()
   assert native_release.wait(5)
 maintenance._readmit_native_holds=held_readmit
 manual_type=type(app.manual_sync_control_service)
 original_drain=manual_type._maintenance_drain
 async def short_drain(self,deadline):
  if native_release.is_set(): retrying.set()
  # Earlier app owners keep their normal deadline; this is the busy producer.
  return await original_drain(self,min(deadline,time.monotonic()+.03))
 manual_type._maintenance_drain=short_drain
 monitoring=asyncio.create_task(maintenance.monitor_app(app))
 try:
  for _ in range(500):
   if getattr(app,'_backup_maintenance_error',None): break
   await asyncio.sleep(.01)
  assert app._backup_maintenance_error=='runtime_work_not_settled'
  assert not monitoring.done(),'monitor abandoned a recoverable fenced producer'
  assert app.manual_sync_control_service._producer_lifetime.closed
  assert not accepted.done()
  if sys.argv[2]=='cancel':
   assert await asyncio.to_thread(native_entered.wait,5)
   monitoring.cancel()
   await asyncio.sleep(.03)
   assert not monitoring.done()
   native_release.set()
   await asyncio.wait_for(retrying.wait(),5)
  requested=False
  release.set()
  assert (await accepted).status=='success'
  for _ in range(500):
   if not app.manual_sync_control_service._producer_lifetime.closed: break
   await asyncio.sleep(.01)
  assert not app.manual_sync_control_service._producer_lifetime.closed
  assert app._backup_runtime_maintenance is None
  if sys.argv[2]=='cancel':
   try: await asyncio.wait_for(asyncio.shield(monitoring),3)
   except asyncio.CancelledError: pass
   else: raise AssertionError('monitor swallowed cancellation')
  else: assert not monitoring.done()
 finally:
  requested=False; release.set(); native_release.set()
  await accepted
  monitoring.cancel()
  try: await monitoring
  except (asyncio.CancelledError,RuntimeError): pass
  runtime=getattr(app,'_backup_runtime_maintenance',None)
  # Failed baseline monitor cannot surrender its same-task authority here.
  # Its retained storage lease must not be replaced with a fake clean result.
  if runtime is None:
   await app._shutdown_app_owned_lifecycles()
   await app.tts_service.close()
  storage._local_pause_requested=original_request
  maintenance._readmit_native_holds=original_readmit
  manual_type._maintenance_drain=original_drain
asyncio.run(main())
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("state", ["ordinary", "cancel"])
def test_monitor_retries_failed_resume_until_accepted_sync_settles(tmp_path, state):
    _run(tmp_path, "monitor", state, script=_MONITOR_BUSY)


def test_remote_upper_and_server_tail_finish_before_storage_pause(tmp_path):
    script = (
        _APPROVED_SETUP
        + r"""
import time
from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget,SectionCapabilityFlags
target_store=plane.target_store
context_store=plane.context_store
target_store.save_targets([ConfiguredServerTarget(
 server_id='server-a',label='Server A',base_url='https://blocked.invalid/api',is_default=True,
)])
from tldw_chatbook.Backup_Recovery.runtime_maintenance import _bind,_settle_stage,_resume_hooks
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
entered,release=asyncio.Event(),asyncio.Event()
class API:
 async def get_status(self):
  entered.set(); await release.wait(); return {'status':'ok'}
server=ServerUnifiedMCPService(client=API(),target_store=target_store)
async def bootstrap_context(client):
 return SimpleNamespace(manageable_team_ids=(),manageable_org_ids=(),can_use_system_admin_scope=False,principal=None)
async def probe(**kwargs): return SectionCapabilityFlags(overview=True),{}
server._bootstrap_access_context=bootstrap_context
server._probe_section_capabilities=probe
plane.server_service=server
before=target_store.path.read_bytes(),context_store.path.read_bytes()
async def run():
 accepted=asyncio.create_task(plane.select_server_target('server-a'))
 entering=asyncio.create_task(entered.wait())
 done,_=await asyncio.wait((accepted,entering),timeout=5,return_when=asyncio.FIRST_COMPLETED)
 if accepted in done:
  await accepted
  raise AssertionError('remote selection returned before the held native API')
 assert entering in done,'remote selection never reached the native API'
 closed=[]
 upper=_bind(plane,'MCP.unified_control_plane_service','UnifiedMCPControlPlaneService')
 waiting=asyncio.create_task(_settle_stage([upper],closed,time.monotonic()+3))
 await asyncio.sleep(0)
 assert not server._producer_lifetime.closed
 try: await plane.select_server_target('server-a')
 except RecoveryRequired: pass
 else: raise AssertionError('upper MCP intake reopened')
 release.set()
 await accepted; await waiting
 assert target_store.path.read_bytes()!=before[0]
 assert context_store.path.read_bytes()!=before[1]
 await _settle_stage([_bind(server,'MCP.server_unified_service','ServerUnifiedMCPService')],closed,time.monotonic()+1)
 with authority.maintenance(tuple(witness['namespaces']),1): pass
 await _resume_hooks(closed)
 await plane.select_server_target('server-a')
asyncio.run(run())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "upper", "approved", script=script)


_MCP_UPPER = (
    _MCP.split("async def run():")[0]
    + r"""
async def run():
 local_store.save_profile(LocalExternalMCPProfile(profile_id='real',command=sys.executable,args=('-u','-c',server,'ok')))
 await plane.connect_local_profile('real')
 session=client.sessions['real']
 entered,release=asyncio.Event(),asyncio.Event()
 send=session._send_message
 async def held_send(payload):
  if payload.get('method')=='tools/call':
   entered.set(); await release.wait()
  await send(payload)
 session._send_message=held_send
 accepted=asyncio.create_task(plane.execute_hub_tool('local:real','sentinel',{}))
 await entered.wait()
 closed=[]
 upper=_bind(plane,'MCP.unified_control_plane_service','UnifiedMCPControlPlaneService')
 waiting=asyncio.create_task(_settle_stage([upper],closed,time.monotonic()+5))
 await asyncio.sleep(0)
 assert not local._producer_lifetime.closed and not client._producer_lifetime.closed
 release.set()
 await accepted; await waiting
 records=plane.execution_log.read_recent()
 assert any(r['server_key']=='local:real' and r['status']=='success' and r['ok'] for r in records),records
 await _settle_stage([_bind(local,'MCP.local_control_service','LocalMCPControlService')],closed,time.monotonic()+1)
 await _settle_stage([_bind(client,'MCP.client','MCPClient')],closed,time.monotonic()+5)
 assert session.process.returncode is not None
 with authority.maintenance(tuple(witness['namespaces']),1): pass
 await _resume_hooks(closed)
 # Exercise the original user-facing connect route after reverse resume.
 await plane.connect_local_profile('real')
 assert len(processes)==2
 await client.disconnect_all()
asyncio.run(run())
assert all(p.returncode is not None for p in processes)
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("state", ["ordinary", "approved"])
def test_local_upper_route_records_native_result_before_child_pause(tmp_path, state):
    script = _MCP_UPPER
    if state == "ordinary":
        script = _ordinary_mcp_script(script)
    _run(tmp_path, "upper", state, script=script)


_PENDING_CHILD = (
    _MCP.split("async def run():")[0]
    + r"""
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
async def run():
 original_cleanup=client._finish_connection_cleanup
 async def refuse(*args):
  raise client_module.MCPClientError('injected native cleanup refusal')
 client._finish_connection_cleanup=refuse
 malformed=server.replace("'protocolVersion':'2025-03-26'", "'protocolVersion':'bad'")
 assert not await client.connect_to_server('real',sys.executable,['-u','-c',malformed,'ok'])
 process=processes[-1]
 closed=[]
 try:
  assert process.returncode is None and client._pending_connections
  assert not client.sessions and not client._producer_lifetime.calls
  hook=_bind(client,'MCP.client','MCPClient')
  try: await _settle_stage([hook],closed,time.monotonic()+.1)
  except RecoveryRequired: pass
  else: raise AssertionError('pending native child admitted capture')
  try: await _resume_hooks(closed)
  except RecoveryRequired: pass
  else: raise AssertionError('live pending child lost its maintenance fence')
  assert closed and client._producer_lifetime.closed
  try: await client.connect_to_server('second',sys.executable,['-u','-c',server,'ok'])
  except RecoveryRequired: pass
  else: raise AssertionError('pending native child reopened new intake')
  assert len(processes)==1 and process.returncode is None
  # The fixture ends the child through stdin EOF. Maintenance does not kill
  # unresolved pending work just to obtain capture.
  process.stdin.close()
  await asyncio.wait_for(process.wait(),2)
  assert client._pending_connections
  client._finish_connection_cleanup=original_cleanup
  assert await client._maintenance_drain(time.monotonic()+3)
  assert not client._pending_connections
  await _resume_hooks(closed)
  assert not closed and not client._producer_lifetime.closed
  assert await client.connect_to_server('real',sys.executable,['-u','-c',server,'ok'])
  assert len(processes)==2
 finally:
  client._finish_connection_cleanup=original_cleanup
  if process.returncode is None:
   process.stdin.close(); await asyncio.wait_for(process.wait(),2)
  for key,pending in tuple(client._pending_connections.items()):
   await client._bounded_teardown_connection(key,pending=pending)
  for key,session in tuple(client.sessions.items()):
   await client._bounded_teardown_connection(key,session=session)
asyncio.run(run())
assert all(p.returncode is not None for p in processes)
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_failed_initialization_pending_child_retains_fence_until_native_exit(tmp_path):
    _run(tmp_path, "pending", "approved", script=_PENDING_CHILD)


_UNKNOWN_PENDING = (
    _MCP.split("async def run():")[0]
    + r"""
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
async def run():
 mapping = (client._connect_reservations if route=='reservation' else
            client.servers if route=='orphan' else client._pending_connections)
 marker=object()
 mapping['unknown']=marker
 client._maintenance_close_admission()
 assert not await client._maintenance_drain(time.monotonic()+.1)
 try: client._maintenance_resume()
 except RecoveryRequired: pass
 else: raise AssertionError('unknown native ownership reopened intake')
 assert client._producer_lifetime.closed
 # This fixture marker never represented a spawned process. Remove only it.
 assert mapping.pop('unknown') is marker
 assert await client._maintenance_drain(time.monotonic()+1)
 client._maintenance_resume()
asyncio.run(run())
assert not processes and not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["reservation", "orphan", "unknown"])
def test_unknown_connection_ownership_keeps_resume_fenced(tmp_path, route):
    _run(tmp_path, route, "approved", script=_UNKNOWN_PENDING)
