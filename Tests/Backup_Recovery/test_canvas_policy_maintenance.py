"""Backup pauses Canvas policy observation without changing the enabled setting."""

import sys

import pytest

from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT as _BOUND
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = _BOUND.split("assert config.get_cli_setting")[0] + r'''
import asyncio,sys,time
from types import SimpleNamespace
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.runtime_maintenance import _bind,_settle_stage,_resume_hooks
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
runtime=ConsoleRuntime(SimpleNamespace(chachanotes_db=None))
runtime.ensure_chat_store()
resolver=lambda session:SimpleNamespace()
authority=runtime.ensure_canvas_native_authority(scope_resolver=resolver)
assert authority is not None and runtime.canvas_enabled()
binding=runtime._canvas_native_view_binding
case=sys.argv[1]
async def main():
 closed=[];pause=None
 runtime.start_async_lifecycles()
 try:
  watcher=runtime._canvas_policy_watch_task
  assert watcher is not None
  hook=_bind(runtime,'Chat.console_runtime','ConsoleRuntime','_canvas_maintenance')
  await _settle_stage([hook],closed,time.monotonic()+3)
  assert watcher.done()
  pause=storage._begin_local_pause()
  try:config.get_canvas_execution_enabled()
  except RecoveryRequired as error:assert error.args==('storage_locally_paused',)
  else:raise AssertionError('paused native config read was admitted')
  assert not runtime.canvas_enabled()
  await runtime.apply_canvas_policy()
  runtime.start_async_lifecycles()
  await asyncio.sleep(.3)
  assert runtime._canvas_policy_watch_task is watcher
  assert not runtime._canvas_disabled_latched
  assert runtime._canvas_native_authority is authority
  assert runtime._canvas_native_view_binding is binding
  if case=='explicit':runtime.latch_canvas_disabled()
  if case=='dispose':await runtime.dispose()
  pause.resume();pause=None
  if case=='external':assert config.save_setting_to_cli_config('canvas','enabled',False)
  await _resume_hooks(closed)
  if case in {'explicit','external'}:
   assert not runtime.canvas_enabled()
   await runtime.apply_canvas_policy()
   assert runtime._canvas_disabled_latched and runtime._canvas_native_authority is None
   assert config.save_setting_to_cli_config('canvas','enabled',True)
   assert not runtime.canvas_enabled()
  elif case=='dispose':
   assert not runtime.canvas_enabled() and runtime._canvas_policy_watch_task is None
  else:
   assert config.get_canvas_execution_enabled() and runtime.canvas_enabled()
   assert runtime._canvas_native_authority is authority
   assert runtime._canvas_native_view_binding is binding
   assert runtime._canvas_policy_watch_task is not watcher
 finally:
  if pause is not None:pause.resume()
  if closed:await _resume_hooks(closed)
  await runtime.dispose()
asyncio.run(main())
assert not storage._raw_operations
print('retired and reopened')
'''


@pytest.mark.parametrize("case", ["resume", "explicit", "external", "dispose"])
def test_canvas_policy_preserves_enabled_state_across_native_backup_pause(tmp_path, case):
    _run(tmp_path, case, "canvas-policy-maintenance", script=_SCRIPT,
         timeout=90 if sys.platform == "win32" else 25)

_LIFETIME = _SCRIPT.split("async def main():")[0] + r'''
async def main():
 closed=[];release=asyncio.Event();entered=asyncio.Event();finished=[]
 original=authority.dispose
 async def dispose():
  entered.set()
  await release.wait()
  original();finished.append(True)
 authority.dispose=dispose
 assert config.save_setting_to_cli_config('canvas','enabled',False)
 if case=='watcher':
  runtime.start_async_lifecycles();caller=runtime._canvas_policy_watch_task
 else:caller=asyncio.create_task(runtime.apply_canvas_policy())
 await asyncio.wait_for(entered.wait(),3)
 hook=_bind(runtime,'Chat.console_runtime','ConsoleRuntime','_canvas_maintenance')
 try:
  try:await _settle_stage([hook],closed,time.monotonic())
  except RecoveryRequired as error:assert error.args==('runtime_work_not_settled',)
  else:raise AssertionError('accepted policy cleanup was not retained')
  if case=='cancel':
   caller.cancel()
   try:await caller
   except asyncio.CancelledError:pass
   else:raise AssertionError('caller cancellation was swallowed')
  if case=='cancel_drain':
   drain=asyncio.create_task(hook.drain(runtime,time.monotonic()+3))
   await asyncio.sleep(0);drain.cancel()
   try:await drain
   except asyncio.CancelledError:pass
   else:raise AssertionError('drain cancellation was swallowed')
  assert not await hook.drain(runtime,time.monotonic())
  try:await _resume_hooks(closed)
  except RecoveryRequired as error:assert error.args==('runtime_work_not_settled',)
  else:raise AssertionError('busy policy cleanup resumed')
  assert not finished and len(closed)==1
 finally:
  release.set()
  await asyncio.gather(caller,return_exceptions=True)
  if closed:
   assert await hook.drain(runtime,time.monotonic()+3)
   await _resume_hooks(closed)
  await runtime.dispose()
 assert finished==[True] and runtime._canvas_disabled_latched
asyncio.run(main())
print('retired and reopened')
'''


@pytest.mark.parametrize("case", ["accepted", "watcher", "cancel", "cancel_drain"])
def test_backup_retains_canvas_policy_cleanup_until_it_finishes(tmp_path, case):
    _run(tmp_path, case, "canvas-policy-cleanup", script=_LIFETIME,
         timeout=90 if sys.platform == "win32" else 25)

_READ_RACE = _SCRIPT.split("async def main():")[0] + r'''
import threading
entered=threading.Event();read_now=threading.Event();read_done=threading.Event();finish=threading.Event()
original=runtime._read_canvas_enabled
results=[]
def read():
 entered.set();assert read_now.wait(5)
 result=original();read_done.set();assert finish.wait(5)
 return result
runtime._read_canvas_enabled=read
worker=threading.Thread(target=lambda:results.append(runtime.canvas_enabled()),daemon=True)
async def main():
 closed=[];pause=None
 hook=_bind(runtime,'Chat.console_runtime','ConsoleRuntime','_canvas_maintenance')
 try:
  worker.start();assert await asyncio.to_thread(entered.wait,3)
  await _settle_stage([hook],closed,time.monotonic()+3)
  pause=storage._begin_local_pause()
  read_now.set();assert await asyncio.to_thread(read_done.wait,3)
  if case=='resumed':
   pause.resume();pause=None
   await _resume_hooks(closed)
  finish.set();await asyncio.to_thread(worker.join,3)
  assert not worker.is_alive() and results==[False]
  assert not runtime._canvas_disabled_latched
 finally:
  read_now.set();finish.set();await asyncio.to_thread(worker.join,3)
  runtime._read_canvas_enabled=original
  if pause is not None:pause.resume()
  if closed:await _resume_hooks(closed)
  assert runtime.canvas_enabled()
  await runtime.dispose()
asyncio.run(main())
print('retired and reopened')
'''


@pytest.mark.parametrize("case", ["closed", "resumed"])
def test_native_policy_read_crossing_backup_pause_cannot_latch_disable(tmp_path, case):
    _run(tmp_path, case, "canvas-policy-race", script=_READ_RACE,
         timeout=90 if sys.platform == "win32" else 25)

_CONSUMERS = _SCRIPT.split("async def main():")[0] + r'''
from Tests.Agents.test_canvas_tool_provider import SCOPE,_Coordinator
from tldw_chatbook.Agents.canvas_tool_provider import CanvasToolProvider
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage,ConsoleMessageRole
service=ConsoleMessageActionService(canvas_enabled_reader=runtime.canvas_enabled,canvas_disabled_reader=runtime.canvas_disabled)
provider=CanvasToolProvider(_Coordinator(),scope=SCOPE,enabled_reader=runtime.canvas_enabled,disabled_reader=runtime.canvas_disabled)
message=ConsoleChatMessage(id='backup-canvas',role=ConsoleMessageRole.ASSISTANT,content='```html\n<title>Retained</title>\n```')
async def main():
 closed=[];pause=None
 hook=_bind(runtime,'Chat.console_runtime','ConsoleRuntime','_canvas_maintenance')
 try:
  assert service.dispatch('canvas-open-0',message).status=='canvas_open_requested'
  assert provider.list_catalog()
  if case=='coordinated':await _settle_stage([hook],closed,time.monotonic()+3)
  # The uncoordinated case also models raw native gate contention before the
  # app monitor reaches its own close hook: the exact config refusal is shared.
  pause=storage._begin_local_pause()
  runtime.start_async_lifecycles()
  await asyncio.sleep(.3)
  assert not runtime.canvas_enabled()
  await runtime.apply_canvas_policy()
  assert service.dispatch('canvas-open-0',message).status=='blocked'
  assert not provider.list_catalog()
  assert not runtime.canvas_disabled()
  pause.resume();pause=None
  if closed:await _resume_hooks(closed)
  assert runtime.canvas_enabled()
  assert service.dispatch('canvas-open-0',message).status=='canvas_open_requested'
  assert provider.list_catalog()
  runtime.latch_canvas_disabled()
  assert service.dispatch('canvas-open-0',message).status=='blocked'
  assert not provider.list_catalog()
  assert service._canvas_disabled_latched and provider._disabled_latched
 finally:
  if pause is not None:pause.resume()
  if closed:await _resume_hooks(closed)
  await runtime.dispose()
asyncio.run(main())
print('retired and reopened')
'''


@pytest.mark.parametrize("case", ["coordinated", "before_hook"])
def test_native_backup_preserves_existing_canvas_consumers(tmp_path, case):
    _run(tmp_path, case, "canvas-policy-consumers", script=_CONSUMERS,
         timeout=90 if sys.platform == "win32" else 25)
