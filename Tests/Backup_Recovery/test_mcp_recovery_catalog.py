"""A reviewed restore publishes only fresh, passive local catalog records."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_mcp_recovery_review import _SETUP

_JOURNEY = r"""
from textual.app import App
from textual.screen import Screen
from textual.widgets import Button,DataTable,Static
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode
from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
service=plane();host=SimpleNamespace(unified_mcp_service=service);route=sys.argv[1]
entered=asyncio.Event();release=asyncio.Event();reads=[];effects=[]
original=service.local_external_catalog
async def catalog():
 reads.append('catalog')
 assert activation.allowed(witness['generation'],'mcp.local'),'catalog read before approval'
 records=await original()
 if route not in {'current','failure'}:
  entered.set();await release.wait()
 if route=='failure':raise OSError('private catalog path must not be exposed')
 return records
service.local_external_catalog=catalog

def forbidden(*args,**kwargs):
 effects.append('unexpected effect');raise AssertionError('review performed an active operation')
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
 def compose(self):self.workbench=Bench(host);yield self.workbench
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
  bench=app.screen.query_one(Bench);await bench._mount_deferred_canvases();bench.set_mode('permissions')
  await bench._sync_children();await pilot.pause()
  bench.query_one('#mcp-perm-recovery-review',Button).press()
  await wait_for(lambda:isinstance(app.screen,ConfirmationDialog));await pilot.pause()
  app.screen.query_one('#confirm-button',Button).press()
  if route not in {'current','failure'}:
   await wait_for(entered.is_set)
   try:
    if route=='mode':bench.set_mode('tools')
    elif route=='round_trip':bench.set_mode('tools');bench.set_mode('permissions')
    elif route=='screen_round_trip':
     await app.push_screen(Screen());await pilot.pause();await app.pop_screen();await pilot.pause()
    elif route=='service':host.unified_mcp_service=plane()
    expected=bench.get_view_state();notice_count=len(notices)
   finally:release.set()
  # The queued Button.Pressed may not have set busy yet. The captured review
  # token remains owned until admission, native write and publication finish.
  await wait_for(lambda:bench._mcp_recovery_token is None and not bench._mcp_recovery_busy);await pilot.pause()
  assert activation.allowed(witness['generation'],'mcp.local')
  assert service.permission_store.get_global_default()=='ask'
  assert all((user/name).read_bytes()==data for name,data in history.items())
  assert reads==['catalog'],'approved review did not read the passive catalog once'
  assert not effects and not blocked_attempts()
  if route not in {'current','failure'}:
   assert bench.get_view_state()==expected and len(notices)==notice_count
   assert not bench._catalog_records,'late catalog was published over a newer view'
  elif route=='failure':
   assert 'Fresh MCP roots reviewed' in notices[-1]
   assert 'Press r to retry' in notices[-1]
   assert 'private catalog path' not in notices[-1]
   assert not bench._catalog_records
  else:
   assert list(bench._catalog_records)==['demo']
   assert bench._catalog_records['demo']['command']=='disposable-sentinel'
   assert not bench._catalog_records['demo']['is_connected']
   assert not bench._catalog_records['demo']['discovery_snapshot']
   table=bench.query_one('#mcp-servers-table',DataTable)
   keys={table.coordinate_to_cell_key((i,0))[0].value for i in range(table.row_count)}
   assert 'local:demo' in keys,'reviewed definition is missing from Servers'
   assert len(bench._snapshots)==2,'built-in readiness disappeared during review'
   assert not service.local_service.store.list_governance_rules()
   assert not service.local_service.store.list_approval_requests()
   copy=' '.join(str(w.renderable) for w in bench.query_one(MCPPermissionsMode).query(Static))
   assert 'Historical rules and grants remain inactive' in copy
   assert 'stay inactive until reviewed' not in copy
asyncio.run(run());print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    ["current", "failure", "mode", "round_trip", "screen_round_trip", "service"],
)
def test_reviewed_catalog_publication_is_passive_and_owned(tmp_path, route):
    _run(tmp_path, route, "catalog", script=_SETUP + _JOURNEY)
