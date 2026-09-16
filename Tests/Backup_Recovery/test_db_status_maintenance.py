"""Database telemetry pauses with backup and retains real cancelled offloads."""

import sys

import pytest

from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT as _BOUND
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = _BOUND.split("assert config.get_cli_setting")[0] + r'''
import asyncio,threading,time,sys
from types import SimpleNamespace
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.runtime_maintenance import _bind,_settle_stage,_resume_hooks
from tldw_chatbook.Utils.db_status_manager import DBStatusManager
case=sys.argv[1]
db=config.get_chachanotes_db_lazy()
db.add_note('Telemetry fixture','Retained during backup')
db.close_connection()
app=SimpleNamespace(db_sizes_status={'previous':'kept'})
manager=DBStatusManager(app)
original=manager._collect_db_sizes
entered=threading.Event();release=threading.Event();returned=threading.Event()
calls=[]
failure=ValueError('synthetic collection failure')
def collect():
 calls.append(threading.get_ident());entered.set()
 try:
  assert release.wait(5), 'test failed to release accepted worker'
  value=original()
  if case=='error':raise failure
  return value
 finally:returned.set()
manager._collect_db_sizes=collect
async def main():
 closed=[];pause=None;accepted=None
 loop_errors=[]
 asyncio.get_running_loop().set_exception_handler(lambda loop,context:loop_errors.append(context))
 if case=='original_hook':
  def unexpected():raise AssertionError('instance hook override ran')
  manager._maintenance_close_admission=unexpected
 hook=_bind(manager,'Utils.db_status_manager','DBStatusManager')
 try:
  if case in {'late','original_hook','open_drain'}:
   if case=='open_drain':
    try:await manager._maintenance_drain(time.monotonic()+1)
    except RecoveryRequired as error:assert error.args==('runtime_producer_not_closed',)
    else:raise AssertionError('open producer was declared drained')
   if case=='original_hook':
    try:_bind(object(),'Utils.db_status_manager','DBStatusManager')
    except RecoveryRequired as error:assert error.args==('runtime_owner_unqualified',)
    else:raise AssertionError('foreign telemetry owner was accepted')
   await _settle_stage([hook],closed,time.monotonic()+1)
   # Delivery is already queued; gating only timer scheduling cannot stop it.
   await manager.update_db_sizes()
   assert not calls and app.db_sizes_status=={'previous':'kept'}
   pause=storage._begin_local_pause()
   await manager.update_db_sizes()
   assert not calls and app.db_sizes_status=={'previous':'kept'}
   assert pause.drain(time.monotonic()+1)
   pause.resume();pause=None
  else:
   accepted=asyncio.create_task(manager.update_db_sizes())
   assert await asyncio.to_thread(entered.wait,3)
   assert calls[0]!=threading.get_ident()
   try:await _settle_stage([hook],closed,time.monotonic())
   except RecoveryRequired as error:assert error.args==('runtime_work_not_settled',)
   else:raise AssertionError('accepted native offload was not retained')
   await manager.update_db_sizes()
   assert len(calls)==1
   if case=='cancel_drain':
    waiting=asyncio.create_task(manager._maintenance_drain(time.monotonic()+5))
    await asyncio.sleep(0)
    waiting.cancel()
    try:await waiting
    except asyncio.CancelledError:pass
    else:raise AssertionError('drain cancellation was swallowed')
   if case in {'cancel','cancel_error'}:
    accepted.cancel()
    try:await accepted
    except asyncio.CancelledError:pass
    else:raise AssertionError('caller cancellation was swallowed')
    assert not returned.is_set()
    assert app.db_sizes_status=={'previous':'kept'}
   assert not await manager._maintenance_drain(time.monotonic())
   try:await _resume_hooks(closed)
   except RecoveryRequired as error:assert error.args==('runtime_work_not_settled',)
   else:raise AssertionError('busy telemetry resumed before worker completion')
   assert len(closed)==1
   release.set()
   if case not in {'cancel','cancel_error'}:await accepted
   assert await manager._maintenance_drain(time.monotonic()+5)
   assert returned.is_set()
   if case in {'cancel','cancel_error','error'}:
    assert app.db_sizes_status=={'previous':'kept'}
   else:
    assert set(app.db_sizes_status)=={'prompts','chachanotes','media'}
    assert 'Error' not in app.db_sizes_status.values()
  await _resume_hooks(closed)
  assert not closed
  manager._collect_db_sizes=original
  release.set()
  await manager.update_db_sizes()
  assert set(app.db_sizes_status)=={'prompts','chachanotes','media'}
  assert 'Error' not in app.db_sizes_status.values()
  await asyncio.sleep(0)
  assert not loop_errors, 'an offload failure escaped its retained completion'
 finally:
  release.set()
  if accepted is not None and not accepted.done():await accepted
  if pause is not None:pause.resume()
asyncio.run(main())
print('retired and reopened')
'''


@pytest.mark.parametrize("case", [
    "late", "complete", "cancel", "error", "cancel_error", "cancel_drain",
    "original_hook", "open_drain",
])
def test_native_telemetry_intake_and_worker_lifetime(tmp_path, case):
    script = _SCRIPT
    if case == "cancel_error":
        script = script.replace("if case=='error':raise failure", "raise failure")
    _run(
        tmp_path, case, "telemetry", script=script,
        timeout=90 if sys.platform == "win32" else 25,
    )
