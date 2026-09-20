"""Restored-root review must not publish over a later workbench view."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_mcp_recovery_review import _SETUP

_JOURNEY = r"""
from textual.app import App
from textual.screen import Screen
from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen
from textual.widgets import Button,DataTable,Select
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.MCP.readiness import local_profile_readiness
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
service=plane();host=SimpleNamespace(unified_mcp_service=service)
route,outcome=sys.argv[1:];entered=Event();release=Event();applies=[];effects=[]
original=service.approve_recovery_review

def approve(review):
 applies.append(review)
 if outcome=='success':original(review)
 entered.set()
 assert release.wait(8),'review release timed out'
 if outcome=='failure':original(review)
service.approve_recovery_review=approve

def forbidden(*args,**kwargs):
 effects.append('unexpected effect');raise AssertionError('review connected or reloaded')
for name in ('connect_to_server','list_tools','execute_tool'):
 setattr(service.local_service.client,name,forbidden)
import keyring
for name in ('get_keyring','get_password','set_password','delete_password'):setattr(keyring,name,forbidden)
class Bench(MCPWorkbench):
 def _start_initial_load(self):self.is_loading=False;self._reloading=False
 reload=forbidden
 _collect_snapshots=forbidden
 _refresh_server_discovery=forbidden
class Surface(Screen):
 def compose(self):
  self.workbench=Bench(host);yield self.workbench
 def _clear_footer_shortcuts(self):pass
 on_screen_suspend=MCPScreen.on_screen_suspend
class Host(App):pass
app=Host();notices=[];app.notify=lambda message,**kwargs:notices.append(str(message))
async def wait_for(predicate):
 async with asyncio.timeout(8):
  while not predicate():await asyncio.sleep(.02)
async def run():
 async with app.run_test(size=(150,55)) as pilot:
  await app.push_screen(Surface());await pilot.pause()
  bench=app.screen.query_one(Bench);await bench._mount_deferred_canvases()
  # A previously published catalog is UI input only, never launched.
  record={'profile_id':'demo','command':'disposable-sentinel','is_connected':False,
          'discovery_snapshot':{'tools':[{'name':name,'description':'Old catalog'}
                                        for name in ('historical_tool','later_tool')]}}
  bench._catalog_records={'demo':record};bench._snapshots=[local_profile_readiness(record)]
  bench._selected_server_key='local:demo';bench.set_mode('permissions')
  await bench._sync_children();await pilot.pause()
  assert bench.query_one('#mcp-tools-table',DataTable).row_count>0
  assert bench.query_one('#mcp-servers-table',DataTable).row_count==1
  bench.query_one('#mcp-perm-recovery-review',Button).press()
  await wait_for(lambda:isinstance(app.screen,ConfirmationDialog));await pilot.pause()
  app.screen.query_one('#confirm-button',Button).press()
  await wait_for(entered.is_set)
  try:
   assert bench._mcp_recovery_busy
   if outcome=='failure':
    path=user/'local_mcp_store.json';value=json.loads(path.read_bytes())
    value['profiles'][0]['command']='changed';path.write_text(json.dumps(value));history[path.name]=path.read_bytes()
   if route=='mode':bench.set_mode('tools')
   elif route=='round_trip':bench.set_mode('tools');bench.set_mode('permissions')
   elif route in {'screen','screen_round_trip'}:
    await app.push_screen(Screen());await pilot.pause()
    if route=='screen_round_trip':await app.pop_screen();await pilot.pause()
   elif route=='service':host.unified_mcp_service=plane()
   elif route.startswith('permission_'):
    if 'profile' in route:service.permission_store.ensure_profile('later')
    # An ordinary passive refresh can publish the now-current owner context
    # while the accepted review's worker still holds its completion receipt.
    await bench._sync_children();await pilot.pause()
    canvas=bench.query_one(MCPPermissionsMode)
    if 'profile' in route:
     field=canvas.query_one('#mcp-perm-tool-profile',Select);field.focus()
     await pilot.press('enter','end','enter');await pilot.pause()
     assert bench._tool_policy_profile_id=='later'
     if route.endswith('round_trip'):
      field.focus();await pilot.press('enter','home','enter');await pilot.pause()
      assert bench._tool_policy_profile_id=='default'
    else:
     names=['later_tool','historical_tool'] if route.endswith('round_trip') else ['later_tool']
     table=canvas.query_one('#mcp-perm-table',DataTable)
     for name in names:
      assert canvas.select_tool_row('local:demo',name)
      table.focus();await pilot.press('enter');await pilot.pause()
      detail=bench.query_one(MCPInspector).current_permission_tool
      assert detail is not None and detail.name==name,'new selection did not reach inspector'
     expected_tool=detail
   if route not in {'current','screen_round_trip'} and not route.startswith('permission_'):
    # A later selection must not be reset to All servers by the old receipt.
    bench._selected_server_key='local:later'
   expected=bench.get_view_state();notice_count=len(notices)
  finally:release.set()
  await wait_for(lambda:not bench._mcp_recovery_busy)
  await pilot.pause()
  assert len(applies)==1
  assert activation.allowed(witness['generation'],'mcp.local')==(outcome=='success')
  assert all((user/name).read_bytes()==data for name,data in history.items())
  assert not effects and not blocked_attempts()
  if route.startswith('permission_row'):
   assert bench.query_one(MCPInspector).current_permission_tool==expected_tool,'late review cleared newer Permissions detail'
  if route!='current':
   assert bench.get_view_state()==expected,'late review overwrote the newer view'
   assert len(notices)==notice_count,'late review notified a different view'
  elif outcome=='failure':
   assert 'Request a fresh review' in notices[-1]
   assert bench.query_one('#mcp-tools-table',DataTable).row_count>0
  else:
   assert service.permission_store.get_global_default()=='ask'
   assert bench._selected_server_key is None
   assert bench.query_one('#mcp-servers-table',DataTable).row_count==0,'old Servers rows survived review'
   table=bench.query_one('#mcp-tools-table',DataTable)
   assert all('historical_tool' not in str(table.get_row_at(i)) for i in range(table.row_count)),'old Tools row survived review'
   assert bench.query_one('#mcp-audit-table',DataTable).row_count==0
   assert 'Fresh MCP roots reviewed' in notices[-1]
  assert bench._mcp_recovery_token is None
asyncio.run(run());print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route", ["current", "mode", "round_trip", "screen", "screen_round_trip", "service"]
)
@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_restored_review_completion_owns_its_view(tmp_path, route, outcome):
    _run(tmp_path, route, outcome, script=_SETUP + _JOURNEY)


@pytest.mark.parametrize(
    "route",
    [
        "permission_row",
        "permission_row_round_trip",
        "permission_profile",
        "permission_profile_round_trip",
    ],
)
def test_restored_review_preserves_later_permission_selection(tmp_path, route):
    _run(tmp_path, route, "success", script=_SETUP + _JOURNEY)


@pytest.mark.parametrize(
    "seam", ["tool", "finding", "sync_lock", "overview", "detail", "readiness"]
)
def test_restored_review_stops_rendering_after_navigation(tmp_path, seam):
    script = (
        _SETUP
        + r"""
from textual.app import App
from textual.widgets import Button
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
service=plane();host=SimpleNamespace(unified_mcp_service=service);seam=sys.argv[1]
class Bench(MCPWorkbench):
 def _start_initial_load(self):self.is_loading=False;self._reloading=False
class Host(App):
 def compose(self):yield Bench(host)
app=Host();notices=[];app.notify=lambda message,**kwargs:notices.append(str(message))
async def wait_for(predicate):
 async with asyncio.timeout(8):
  while not predicate():await asyncio.sleep(.02)
async def run():
 async with app.run_test(size=(150,55)) as pilot:
  bench=app.query_one(Bench);await bench._mount_deferred_canvases();bench.set_mode('permissions')
  await bench._sync_children();await pilot.pause()
  inspector=bench.query_one(MCPInspector);servers=bench.query_one(MCPServersMode)
  entered=asyncio.Event();release=asyncio.Event()
  if seam=='sync_lock':await bench._sync_children_lock.acquire()
  else:
   target,name={'tool':(inspector,'show_tool'),'finding':(inspector,'show_finding'),
                'overview':(servers,'update_overview'),'detail':(bench,'_show_selected_detail'),
                'readiness':(inspector,'update_readiness')}[seam]
   original=getattr(target,name)
   async def gated(*args,**kwargs):
    await original(*args,**kwargs)
    entered.set();await release.wait()
   setattr(target,name,gated)
  bench.query_one('#mcp-perm-recovery-review',Button).press()
  await wait_for(lambda:isinstance(app.screen,ConfirmationDialog));await pilot.pause()
  app.screen.query_one('#confirm-button',Button).press()
  if seam=='sync_lock':await wait_for(lambda:bool(bench._sync_children_lock._waiters))
  else:await wait_for(entered.is_set)
  try:
   if seam!='sync_lock':setattr(target,name,original)
   bench.set_mode('audit')
   for worker in list(bench.workers):
    if worker.group=='mcp-tool-clear':await worker.wait()
   later={'severity':'warning','message':'Newly selected detail','finding_type':'native-review'}
   await inspector.show_finding(later,server_key='server:later')
   expected=bench.get_view_state();count=len(notices)
   # After navigation, no remaining recovery-driven sync may start.
   renders=[]
   def forbidden(*args,**kwargs):renders.append('stale render');raise AssertionError('stale recovery continued rendering')
   bench._sync_tools_mode=forbidden
  finally:
   if seam=='sync_lock':bench._sync_children_lock.release()
   else:release.set()
  await wait_for(lambda:not bench._mcp_recovery_busy);await pilot.pause()
  assert inspector._current_finding==later,'late recovery cleared newer finding'
  assert bench.get_view_state()==expected and len(notices)==count
  assert not renders,'late recovery continued rendering'
  assert activation.allowed(witness['generation'],'mcp.local')
  assert all((user/name).read_bytes()==data for name,data in history.items())
  assert not blocked_attempts()
asyncio.run(run());print('retired and reopened')
"""
    )
    _run(tmp_path, seam, "render", script=script)
