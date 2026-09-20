"""Restored-root review keeps literal details and decisions keyboard reachable."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_mcp_recovery_review import _SETUP

_SCRIPT = r"""
from textual.app import App
from textual.widgets import Button
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
service=plane(); host=SimpleNamespace(unified_mcp_service=service)
route, theme = sys.argv[1:]
effects=[]
def forbidden(*args, **kwargs):
 effects.append('unexpected connection or execution')
 raise AssertionError(effects[-1])
for name in ('connect_to_server','list_tools','execute_tool'):
 setattr(service.local_service.client,name,forbidden)
class Bench(MCPWorkbench):
 def _start_initial_load(self): self.is_loading=False; self._reloading=False
class Host(App):
 CSS_PATH=str(Path(sys.modules[MCPWorkbench.__module__].__file__).parents[2]/'css/tldw_cli_modular.tcss')
 def compose(self): yield Bench(host,id='bench')
app=Host();app.theme=theme
async def settle(pilot):
 await pilot.pause();await pilot.wait_for_scheduled_animations();await pilot.pause()
async def wait_for(pilot, predicate):
 async with asyncio.timeout(15):
  while not predicate():await pilot.pause(.03)
 await settle(pilot)
def compact(value):return ''.join(value.split())
def painted(region):
 return '\n'.join(strip.crop(region.x,region.right).text for strip in app.screen._compositor.render_strips()[region.y:region.bottom])
def visible_actions():
 for selector in ('#cancel-button','#confirm-button'):
  button=app.screen.query_one(selector,Button)
  assert button in app.screen._compositor.visible_widgets, 'review action is offscreen: '+selector
  region,clip=app.screen._compositor.visible_widgets[button]
  assert region.width>0 and region.height>0 and region.intersection(clip)==region
  assert compact(str(button.label)) in compact(painted(button.content_region))
  hit,_=app.screen.get_widget_at(*region.center)
  assert hit is button
async def run():
 async with app.run_test(size=(80,24)) as pilot:
  bench=app.query_one(Bench)
  await bench._mount_deferred_canvases();bench.set_mode('permissions')
  await bench._sync_permissions_mode(effective={});await settle(pilot)
  launch=bench.query_one('#mcp-perm-recovery-review',Button)
  launch.focus();await settle(pilot);await pilot.press('enter')
  await wait_for(pilot,lambda:isinstance(app.screen,ConfirmationDialog))
  dialog=app.screen
  assert '[bold]review' in dialog.message
  if route=='entry':
   for size in ((80,24),(170,48),(80,24)):
    await pilot.resize_terminal(*size);await settle(pilot)
    visible_actions()
   # Tab reaches both choices from the initially focused review body.
   await pilot.press('tab');await settle(pilot)
   assert app.focused.id=='cancel-button'
   await pilot.press('enter')
  else:
   # Opening the review must expose a keyboard scroll target immediately.
   await pilot.press('end');await settle(pilot)
   message=dialog.query_one('.dialog-message')
   assert compact('No server will connect and no tool permission will be granted.') in compact(painted(app.screen.region)), 'keyboard cannot read the end of the review'
   visible_actions()
   await pilot.press('home');await settle(pilot)
   assert 'Workspace:' in painted(app.screen.region)
   # Collect each visible message row while advancing with actual keys.
   rows={}
   for _ in range(300):
    region,clip=app.screen._compositor.visible_widgets[message]
    shown=region.intersection(clip)
    for y in range(shown.y,shown.bottom):
     rows[y-message.region.y]=app.screen._compositor.render_strips()[y].crop(shown.x,shown.right).text
    before=message.region.y
    await pilot.press('down');await settle(pilot)
    visible_actions()
    if message.region.y==before:break
   from rich.text import Text
   assert compact(Text.from_markup(dialog.message).plain) in compact('\n'.join(rows[i] for i in sorted(rows)))
   await pilot.press('escape')
  await wait_for(pilot,lambda:app.screen is not dialog)
  assert not activation.allowed(witness['generation'],'mcp.local')
  assert all((user/name).read_bytes()==value for name,value in history.items())
  assert not effects and not blocked_attempts()
asyncio.run(run())
print('retired and reopened')
"""


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("route", ["entry", "keyboard"])
def test_compact_recovery_review_keyboard_and_paint(tmp_path, route, theme):
    setup = _SETUP.replace(
        "user=source/'Local';",
        "workspace=base/'[bold]review-工具-long-workspace-directory';workspace.mkdir(mode=0o700)\n"
        "selector.write_text(selector.read_text()+'[console]\\nworkspace_root=\"'+str(workspace)+'\"\\n')\n"
        "user=source/'Local';",
    )
    _run(tmp_path, route, theme, script=setup + _SCRIPT, timeout=90)
