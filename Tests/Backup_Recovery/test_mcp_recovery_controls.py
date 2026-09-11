"""Mounted MCP Permissions uses actual restored owner review without reconnecting."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_mcp_recovery_review import _SETUP

_UI = r'''
from textual.app import App
from textual.widgets import Button
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
service=plane(); host=SimpleNamespace(unified_mcp_service=service)
mode=sys.argv[1]; effects=[]; applies=[]; captured=[]
original_capture=service.capture_recovery_review
original_approve=service.approve_recovery_review
def capture():
 result=original_capture();captured.append(result);return result
def approve(review):
 applies.append(review);return original_approve(review)
service.capture_recovery_review=capture;service.approve_recovery_review=approve
if mode=='cancel_accepted':
 from tldw_chatbook.Backup_Recovery import native_files
 entered=Event(); release=Event(); flushed=native_files._flush_private_tree
 def hold(path,*args):
  result=flushed(path,*args)
  if not entered.is_set():
   entered.set();assert release.wait(5)
  return result
 native_files._flush_private_tree=hold
original_client={name:getattr(service.local_service.client,name,None) for name in ('connect_to_server','list_tools','execute_tool')}

def forbidden(*args,**kwargs):
 effects.append('unexpected effect'); raise AssertionError('review contacted or mutated unrelated service')
import keyring
for name in ('get_keyring','get_password','set_password','delete_password'):setattr(keyring,name,forbidden)
for name in ('reload','_collect_snapshots','_refresh_server_discovery'):
 setattr(MCPWorkbench,name,forbidden)
for name in ('connect_to_server','list_tools','execute_tool'):
 setattr(service.local_service.client,name,forbidden)
service.set_global_default=forbidden
class Bench(MCPWorkbench):
 def _start_initial_load(self):
  self.is_loading=False;self._reloading=False
class Host(App):
 def compose(self): yield Bench(host,id='bench')
app=Host(); notices=[]
app.notify=lambda message,**kwargs:notices.append(str(message))
async def wait_for(predicate):
 for _ in range(150):
  if predicate():return
  await asyncio.sleep(.02)
 raise AssertionError('mounted review did not settle: '+repr(notices))
async def run():
 async with app.run_test(size=(150,55)) as pilot:
  bench=app.query_one(Bench)
  await bench._mount_deferred_canvases();bench.set_mode('permissions')
  await bench._sync_permissions_mode(effective={})
  await pilot.pause()
  buttons=bench.query('#mcp-perm-recovery-review')
  assert len(buttons)==1,'MCP Permissions has no restored-root review control'
  buttons.first(Button).press()
  await wait_for(lambda:isinstance(app.screen,ConfirmationDialog))
  dialog=app.screen
  await pilot.pause()
  assert dialog.query_one('#confirmation-dialog').region.height <= app.size.height,'review exceeds viewport'
  confirm=dialog.query_one('#confirm-button',Button);confirm.scroll_visible(animate=False);await pilot.pause()
  assert confirm.region.bottom <= app.size.height,'confirmation is unreachable'
  assert 'workspace' in dialog.message.lower()
  assert '\\[bold]review' in dialog.message
  assert 'old-approved' not in dialog.message and 'disposable-sentinel' not in dialog.message
  assert not activation.allowed(witness['generation'],'mcp.local')
  token=bench._mcp_recovery_token
  if mode=='navigation':bench.set_mode('tools');bench.set_mode('permissions')
  elif mode=='source':bench._source='server'
  elif mode=='target':bench._selected_server_key='server:changed'
  elif mode=='target_round_trip':
   from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCPRail
   await bench.on_mcp_rail_server_selected(MCPRail.ServerSelected('local:demo'))
   await bench.on_mcp_rail_server_selected(MCPRail.ServerSelected(None))
  elif mode=='scope':bench._scope='team'
  elif mode=='service':host.unified_mcp_service=plane()
  elif mode=='unmount':await bench.remove()
  elif mode=='definition':
   path=user/'local_mcp_store.json';value=json.loads(path.read_bytes());value['profiles'][0]['command']='changed';path.write_text(json.dumps(value));history[path.name]=path.read_bytes()
  elif mode=='workspace':
   selected=bootstrap.effective_config_path();selected.write_text(selected.read_text().replace('[bold]review','changed-review'));(base/'changed-review').mkdir(mode=0o700)
  elif mode=='generation':
   required=activation._generation(witness['generation'])/'required.json';value=json.loads(required.read_bytes());value['generation']='changed-generation';required.write_text(json.dumps(value))
  elif mode=='duplicate':buttons.first(Button).press()
  app.screen.query_one('#cancel-button' if mode=='cancel' else '#confirm-button',Button).press()
  await wait_for(lambda:not isinstance(app.screen,ConfirmationDialog))
  if mode=='cancel_accepted':
   try:
    await wait_for(entered.is_set)
    workers=[worker for worker in app.workers if worker.group=='mcp-recovery-confirm']
    assert len(workers)==1
    workers[0].cancel();await pilot.pause()
    assert bench._mcp_recovery_busy,'cancelled waiter abandoned accepted native write'
    assert not activation.allowed(witness['generation'],'mcp.local')
   finally:release.set()
  await wait_for(lambda:not bench._mcp_recovery_busy)
  if mode=='duplicate':bench._confirm_mcp_recovery_review(token,captured[0],True)
  positive=mode in ('approve','empty','duplicate','cancel_accepted')
  assert activation.allowed(witness['generation'],'mcp.local') == positive,notices
  assert len(applies)==(1 if positive or mode in ('definition','workspace','generation') else 0),notices
  assert not activation.allowed(witness['generation'],'config')
  assert not activation.allowed(witness['generation'],'skills')
  assert all((user/name).read_bytes()==data for name,data in history.items())
  if positive:
   assert service.permission_store.get_global_default()=='ask'
   assert service.selected_source=='local'
   with execution(service):pass
   from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCPRail
   await pilot.pause()
   assert bench.query_one(MCPRail).source=='local'
   assert bench.query_one(MCPRail).selected_server_key is None
  assert not effects and not blocked_attempts()
  if mode=='approve':
   # Explicit execution is a separate phase after the UI's owner-only review.
   client=service.local_service.client
   for name,value in original_client.items():setattr(client,name,value)
   server="""
import json,sys
for line in sys.stdin:
 p=json.loads(line)
 if 'id' not in p:continue
 method=p['method']
 if method=='initialize':result={'protocolVersion':'2025-03-26','capabilities':{},'serverInfo':{}}
 elif method=='tools/list':result={'tools':[{'name':'sentinel','inputSchema':{'type':'object'}}]}
 elif method=='resources/list':result={'resources':[]}
 elif method=='prompts/list':result={'prompts':[]}
 elif method=='tools/call':result={'content':[{'type':'text','text':'real-wire-effect'}]}
 else:result={}
 print(json.dumps({'jsonrpc':'2.0','id':p['id'],'result':result}),flush=True)
"""
   assert await client.connect_to_server('real',sys.executable,['-u','-c',server])
   process=client.sessions['real'].process
   try:assert (await client.call_tool('real','sentinel',{}))['result'][0]['text']=='real-wire-effect'
   finally:await client.disconnect_all()
   assert process.returncode is not None and not client.sessions
   assert not activation.allowed(witness['generation'],'config')
   assert not blocked_attempts()
asyncio.run(run())
print('retired and reopened')
'''


@pytest.mark.parametrize(
    "mode",
    [
        "approve",
        "cancel",
        "navigation",
        "source",
        "target",
        "target_round_trip",
        "scope",
        "service",
        "unmount",
        "definition",
        "workspace",
        "generation",
        "duplicate",
        "cancel_accepted",
        "empty",
    ],
)
def test_mounted_restored_mcp_root_review(tmp_path, mode):
    setup = _SETUP.replace(
        "user=source/'Local';",
        "workspace=base/'[bold]review';workspace.mkdir(mode=0o700)\n"
        "selector.write_text(selector.read_text()+'[console]\\nworkspace_root=\"'+str(workspace)+'\"\\n')\n"
        "user=source/'Local';",
    )
    if mode == "empty":
        setup = setup.replace(
            "'profiles':[{'profile_id':'demo','command':'disposable-sentinel','args':[]}]",
            "'profiles':[]",
        )
    _run(tmp_path, mode, "", script=setup + _UI)


_ORDINARY_UI = r"""
from textual.app import App
from textual.widgets import Button
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
service=plane();host=SimpleNamespace(unified_mcp_service=service)
class Bench(MCPWorkbench):
 def _start_initial_load(self):self.is_loading=False;self._reloading=False
class Host(App):
 def compose(self):yield Bench(host)
app=Host();notices=[];app.notify=lambda message,**kwargs:notices.append(str(message))
async def run():
 async with app.run_test(size=(150,55)) as pilot:
  bench=app.query_one(Bench);await bench._mount_deferred_canvases();bench.set_mode('permissions')
  await bench._sync_permissions_mode(effective={});await pilot.pause()
  before={p.name:p.read_bytes() for p in user.iterdir() if p.is_file()}
  bench.query_one('#mcp-perm-recovery-review',Button).press()
  for _ in range(150):
   if notices:break
   await asyncio.sleep(.02)
  assert notices and 'unavailable' in notices[-1],notices
  assert not isinstance(app.screen,ConfirmationDialog)
  assert {p.name:p.read_bytes() for p in user.iterdir() if p.is_file()}==before
  assert not service.get_kill_switch()
  bench.query_one('#mcp-perm-kill-switch',Button).press()
  for _ in range(150):
   if service.get_kill_switch():break
   await asyncio.sleep(.02)
  assert service.get_kill_switch(),'ordinary explicit permission control stopped working'
  assert not blocked_attempts()
asyncio.run(run());print('retired and reopened')
"""


def test_mounted_ordinary_mcp_review_preserves_existing_permission_controls(tmp_path):
    prefix = _SETUP.split("doc=manifest()", 1)[0]
    imports = (
        "from tldw_chatbook.MCP.local_store"
        + _SETUP.split("from tldw_chatbook.MCP.local_store", 1)[1].split(
            "user=destination/'data'", 1
        )[0]
    )
    factory = (
        "from tldw_chatbook import config"
        + _SETUP.split("from tldw_chatbook import config", 1)[1].split(
            "_,profiles,_=", 1
        )[0]
    )
    _run(tmp_path, "ordinary", "", script=prefix + imports + factory + _ORDINARY_UI)


_VIEW_STATE = r"""
from textual.app import App
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCPRail
service=plane();service.approve_recovery_review(service.capture_recovery_review())
host=SimpleNamespace(unified_mcp_service=service);effects=[]
def forbidden(*args,**kwargs):effects.append('connect');raise AssertionError('view restoration connected')
service.local_service.client.connect_to_server=forbidden
class Bench(MCPWorkbench):
 def _start_initial_load(self):self.is_loading=False;self._reloading=False
class Host(App):
 def compose(self):yield Bench(host)
async def run():
 app=Host()
 async with app.run_test(size=(150,55)) as pilot:
  bench=app.query_one(Bench);await bench._mount_deferred_canvases()
  bench._snapshots=await bench._collect_snapshots()
  assert bench._snapshot_for('local:demo') is not None
  bench.set_mode('permissions');await pilot.pause()
  cases=[
   ('absent','local:demo',{},'local:demo',False),
   ('unknown','local:demo',{'selected_server_key':'local:unknown'},'local:demo',False),
   ('wrong_type','local:demo',{'selected_server_key':17},'local:demo',False),
   ('mapping','local:demo',{'selected_server_key':{'bad':'key'}},'local:demo',False),
   ('clear','local:demo',{'selected_server_key':None},None,True),
   ('valid',None,{'selected_server_key':'local:demo'},'local:demo',True),
   ('same','local:demo',{'selected_server_key':'local:demo'},'local:demo',False),
  ]
  for name,previous,state,expected,changed in cases:
   bench._selected_server_key=previous;token=object();bench._mcp_recovery_token=token
   await bench._apply_view_state({'mode':'permissions',**state})
   assert bench._selected_server_key==expected,name
   assert (bench._mcp_recovery_token is None) if changed else (bench._mcp_recovery_token is token),name
  token=object();bench._mcp_recovery_token=token
  await bench.on_mcp_rail_server_selected(MCPRail.ServerSelected('local:demo'))
  assert bench._mcp_recovery_token is token,'same target invalidated review'
  await bench.on_mcp_rail_server_selected(MCPRail.ServerSelected(None))
  await bench.on_mcp_rail_server_selected(MCPRail.ServerSelected('local:demo'))
  assert bench._selected_server_key=='local:demo'
  assert bench._mcp_recovery_token is None,'target round-trip reused pending review'
  assert not effects and not blocked_attempts()
  assert not activation.allowed(witness['generation'],'config')
asyncio.run(run());print('retired and reopened')
"""


def test_actual_mcp_view_state_preserves_selection_validation_and_review_revision(
    tmp_path,
):
    _run(tmp_path, "view-state", "", script=_SETUP + _VIEW_STATE)
