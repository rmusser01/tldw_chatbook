"""Console store observation preserves a successful Canvas view binding."""

import sys

import pytest

from Tests.Backup_Recovery.test_canvas_policy_maintenance import _SCRIPT as _NATIVE
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = _NATIVE.split("async def main():")[0] + r'''
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Chat.console_canvas_controller import ConsoleCanvasController
class View:
 _console_chat_store=runtime.chat_store
 def _console_runtime(self):return runtime
 def _console_ordered_resume_pending(self):return False
 def _console_canvas_scope(self,*args):return None
 def _prefill_console_canvas_repair(self,*args):pass
 def _prepare_console_canvas_submit(self,*args):pass
 def _schedule_console_canvas_tool_open(self,*args):pass
 def _console_canvas_publication_is_current(self,*args):return False
def callbacks(view):
 return dict(scope_resolver=view._console_canvas_scope,
             bridge_sink=view._prefill_console_canvas_repair,
             bridge_prepare=view._prepare_console_canvas_submit,
             auto_open=view._schedule_console_canvas_tool_open,
             publication_guard=view._console_canvas_publication_is_current)
def lookup(view):return ChatScreen._ensure_console_chat_store(view)
view=View();runtime.attach_view(view)
reader=runtime._canvas_enabled_reader;reads=[]
def observed():
 reads.append(1)
 return reader()
runtime._canvas_enabled_reader=observed
async def main():
 pause=None
 try:
  if case=='initial_pause':
   runtime._canvas_native_view_binding=None
   pause=storage._begin_local_pause()
   lookup(view)
   assert runtime._canvas_native_view_binding is None
   pause.resume();pause=None
  lookup(view)
  first=runtime._canvas_native_view_binding
  assert first is not None
  generation=authority._view_generation
  reads.clear()
  if case=='direct':
   runtime.bind_canvas_native_view(**callbacks(view))
   assert reads==[1] and runtime._canvas_native_view_binding is not first
   return
  if case=='successor':
   successor=View();runtime.attach_view(successor);lookup(successor)
   replacement=runtime._canvas_native_view_binding
   assert replacement is not first
   lookup(view)
   assert runtime._canvas_native_view_binding is replacement
   return
  if case=='callbacks':view._prefill_console_canvas_repair=lambda *args:None
  if case=='controller':
   runtime._canvas_controller=ConsoleCanvasController(profile_snapshot=runtime._ensure_canvas_profile_snapshot())
  if case=='reclaim':
   assert runtime.detach_view(view,runtime._attached_generation)
   runtime.attach_view(view)
  if case=='cleared':runtime._canvas_native_view_binding=None
  if case=='pause':
   pause=storage._begin_local_pause()
   lookup(view)
   assert runtime._materialize_canvas_native_authority() is None
   assert not runtime._canvas_disabled_latched
   pause.resume();pause=None
  lookup(view)
  if case in {'callbacks','controller','reclaim','cleared'}:
   assert runtime._canvas_native_view_binding is not first and reads==[1]
   assert runtime.canvas_controller._settlement_listeners==[runtime._canvas_settlement_listener]
  else:
   assert runtime._canvas_native_view_binding is first
   assert authority._view_generation==generation
   assert reads==([1] if case=='pause' else [])
  stable=runtime._canvas_native_view_binding
  reads.clear();lookup(view)
  assert runtime._canvas_native_view_binding is stable and reads==[]
  assert config.save_setting_to_cli_config('canvas','enabled',False)
  assert runtime._materialize_canvas_native_authority() is None
  assert runtime._canvas_disabled_latched
  await runtime.apply_canvas_policy()
  assert runtime._canvas_native_view_binding is None
 finally:
  if pause is not None:pause.resume()
  await runtime.dispose()
asyncio.run(main())
assert not storage._raw_operations
print('retired and reopened')
'''


@pytest.mark.parametrize("case", [
    "same", "direct", "successor", "callbacks", "controller", "reclaim",
    "cleared", "initial_pause", "pause",
])
def test_store_lookup_preserves_only_its_current_canvas_view_binding(tmp_path, case):
    _run(tmp_path, case, "canvas-binding", script=_SCRIPT,
         timeout=90 if sys.platform == "win32" else 35)


_IMPORT = _SCRIPT.split("async def main():")[0] + r'''
from tldw_chatbook.Canvas.models import CanvasScope
store=runtime.chat_store
session=store.create_session(ephemeral=True)
store.append_message(session.id,role='assistant',content='Synthetic import source')
def scope(self,requested):
 assert requested==store.active_session_id==session.id
 return CanvasScope(requested,requested,tuple(store.canvas_active_path_message_ids(requested)),None,None,'synthetic-import')
View._console_canvas_scope=scope
source='<!doctype html><title>Synthetic</title><p>Import lifecycle</p>'
try:
 lookup(view)
 capture=authority._capture_import(session.id,{})
 assert authority._apply_import(session_id=session.id,source=source,capture=capture) is None
 prepared=authority._compilation.run(lambda:authority._prepare_import(capture,source))
 if case=='successor':
  successor=View();runtime.attach_view(successor);lookup(successor)
 lookup(view)
 if case=='same':
  assert authority._apply_import(session_id=session.id,source=source,capture=capture,_prepared=prepared) is not None
 else:
  try:authority._apply_import(session_id=session.id,source=source,capture=capture,_prepared=prepared)
  except RuntimeError as error:assert error.args==('canvas_scope_unavailable',)
  else:raise AssertionError('superseded import was accepted')
  assert authority.import_html(session_id=session.id,source=source) is not None
 live=authority._scope_resolver(session.id)
 assert len(runtime.canvas_controller.list_session_canvases(live,temporary=True))==1
finally:asyncio.run(runtime.dispose())
assert not storage._raw_operations
print('retired and reopened')
'''


@pytest.mark.parametrize("case", ["same", "successor"])
def test_pending_canvas_import_survives_only_same_view_store_observation(tmp_path, case):
    _run(tmp_path, case, "canvas-import", script=_IMPORT,
         timeout=90 if sys.platform == "win32" else 35)
