"""Required policy polling must leave the UI and backup coordinator runnable."""

import sys

import pytest

from Tests.Backup_Recovery.test_canvas_policy_maintenance import _SCRIPT
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_RESPONSIVE = _SCRIPT.split("async def main():")[0] + r'''
import threading
held=threading.Event();release=threading.Event();entered=threading.Event()
watchdog_fired=[];writer_errors=[]
def writer():
 try:
  with config._config_write_lock(selected):
   held.set()
   assert release.wait(20)
 except BaseException as error:writer_errors.append(type(error).__name__)
def unblock():
 watchdog_fired.append(True);release.set()
original=runtime._canvas_enabled_reader
def read():
 entered.set()
 return original()
runtime._canvas_enabled_reader=read
worker=threading.Thread(target=writer,daemon=True)
async def main():
 worker.start();assert await asyncio.to_thread(held.wait,10)
 watchdog=threading.Timer(10 if sys.platform=='win32' else 3,unblock)
 watchdog.start()
 try:
  runtime.start_async_lifecycles()
  assert await asyncio.to_thread(entered.wait,10)
  # This callback cannot run if the watcher waits on the native config lock
  # on this event loop. Only the independent watchdog can release it then.
  release.set()
  assert not watchdog_fired,'native policy polling blocked the UI event loop'
 finally:
  release.set();watchdog.cancel()
  await asyncio.to_thread(worker.join,10)
  await runtime.dispose()
 assert not worker.is_alive() and not writer_errors
 assert not storage._raw_operations
asyncio.run(main())
print('retired and reopened')
'''


def test_native_policy_poll_does_not_block_ui_while_config_writer_is_active(tmp_path):
    _run(tmp_path, "responsive", "canvas-policy-worker", script=_RESPONSIVE,
         timeout=90 if sys.platform == "win32" else 25)


_RETAINED = _RESPONSIVE.split("async def main():")[0] + r'''
async def main():
 worker.start();assert await asyncio.to_thread(held.wait,10)
 watchdog=threading.Timer(15,unblock);watchdog.start()
 caller=None
 try:
  runtime.start_async_lifecycles()
  watcher=runtime._canvas_policy_watch_task
  assert await asyncio.to_thread(entered.wait,10)
  assert not watchdog_fired,'native policy polling blocked the UI event loop'
  if case=='dispose_cancel':
   caller=asyncio.create_task(runtime.dispose())
   await asyncio.sleep(.05)
   assert not caller.done(),'dispose abandoned an accepted native read'
   caller.cancel()
   try:await caller
   except asyncio.CancelledError:pass
   else:raise AssertionError('dispose cancellation was swallowed')
  else:
   runtime._canvas_maintenance_close_admission()
   if case=='watcher_cancel':
    watcher.cancel()
    try:await watcher
    except asyncio.CancelledError:pass
    else:raise AssertionError('watcher cancellation was swallowed')
   if case=='drain_cancel':
    caller=asyncio.create_task(runtime._canvas_maintenance_drain(time.monotonic()+10))
    await asyncio.sleep(0);caller.cancel()
    try:await caller
    except asyncio.CancelledError:pass
    else:raise AssertionError('drain cancellation was swallowed')
  runtime._canvas_maintenance_close_admission()
  assert not await runtime._canvas_maintenance_drain(time.monotonic()),'accepted native read was not retained'
  try:runtime._canvas_maintenance_resume()
  except RecoveryRequired as error:assert error.args==('runtime_work_not_settled',)
  else:raise AssertionError('backup resumed before native policy read completed')
 finally:
  release.set();watchdog.cancel()
  await asyncio.to_thread(worker.join,10)
  runtime._canvas_maintenance_close_admission()
  assert await runtime._canvas_maintenance_drain(time.monotonic()+10)
  if not runtime._disposed:
   runtime._canvas_maintenance_resume()
   assert runtime.canvas_enabled() and runtime._canvas_native_authority is authority
  await runtime.dispose()
 assert not worker.is_alive() and not writer_errors
 assert not storage._raw_operations
asyncio.run(main())
print('retired and reopened')
'''


@pytest.mark.parametrize("case", ["maintenance", "watcher_cancel", "drain_cancel", "dispose_cancel"])
def test_backup_and_disposal_retain_accepted_native_policy_read(tmp_path, case):
    _run(tmp_path, case, "canvas-policy-worker-retained", script=_RETAINED,
         timeout=90 if sys.platform == "win32" else 25)


_RESTART = _RESPONSIVE.split("async def main():")[0] + r'''
reads=[]
def read():
 reads.append(threading.get_ident());entered.set()
 return original()
runtime._canvas_enabled_reader=read
async def main():
 worker.start();assert await asyncio.to_thread(held.wait,10)
 watchdog=threading.Timer(15,unblock);watchdog.start()
 try:
  runtime.start_async_lifecycles()
  watcher=runtime._canvas_policy_watch_task
  assert await asyncio.to_thread(entered.wait,10)
  assert not watchdog_fired,'native policy polling blocked the UI event loop'
  reader=runtime._canvas_policy_read_task
  watcher.cancel()
  try:await watcher
  except asyncio.CancelledError:pass
  else:raise AssertionError('watcher cancellation was swallowed')
  runtime.start_async_lifecycles()
  successor=runtime._canvas_policy_watch_task
  runtime.start_async_lifecycles()
  await asyncio.sleep(0)
  assert successor is not watcher and runtime._canvas_policy_watch_task is successor
  assert runtime._canvas_policy_read_task is reader and len(reads)==1
  release.set();await asyncio.to_thread(worker.join,10)
  deadline=time.monotonic()+10
  while len(reads)<2 and time.monotonic()<deadline:await asyncio.sleep(.01)
  assert len(reads)>=2,'restarted watcher stopped observing native policy'
  assert not runtime._canvas_disabled_latched
 finally:
  release.set();watchdog.cancel()
  await asyncio.to_thread(worker.join,10)
  await runtime.dispose()
 assert not worker.is_alive() and not writer_errors
 assert not storage._raw_operations
asyncio.run(main())
print('retired and reopened')
'''


def test_cancelled_watcher_reuses_pending_native_read_when_restarted(tmp_path):
    _run(tmp_path, "restart", "canvas-policy-worker-restarted", script=_RESTART,
         timeout=90 if sys.platform == "win32" else 25)
