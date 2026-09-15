"""A settled worker must not strand the monitor thread's installed cache."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio,sys,threading,time
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import runtime_maintenance as maintenance
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import _core_operation
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from pathlib import Path

db=CharactersRAGDB(Path.home()/'notes.db',client_id='late-worker')
connection=db.get_connection()
participant=db._maintenance_participant
ready=threading.Event();release=threading.Event();finished=threading.Event();errors=[]
def worker():
 try:
  with _core_operation(db):
   foreign=db.get_connection()
   ready.set()
   assert release.wait(5)
   assert foreign.execute('SELECT 1').fetchone()[0]==1
 except BaseException as error:errors.append(error)
 finally:
  try:db.close_connection()
  except BaseException as error:errors.append(error)
  finished.set()
thread=threading.Thread(target=worker);thread.start()
assert ready.wait(5)

class Drained(BaseException):pass

async def main():
 first=asyncio.Event();reached=[]
 original_settle=maintenance.RuntimeMaintenance.settle_producers
 original_retire=maintenance.RuntimeMaintenance.retire_local_caches
 original_probe=maintenance._poll_local_pause_requested
 original_startup=storage._LocalPause.retire_startup
 # Isolate producer/probe orchestration; every cache, operation, lease, pause,
 # drain and close below is the installed native implementation.
 async def settled(self,deadline):self._settled=True
 async def requested():return True
 def retire(self):
  original_retire(self)
  assert db._local.conn is connection and not connection.in_transaction
  with storage._lock:
   assert any(op.participant is participant for op in storage._operations)
  first.set()
  if sys.argv[1]=='completed':release.set()
 def at_startup(pause,runtime):
  assert pause.drain(time.monotonic())
  assert finished.is_set() and not errors
  assert db._local.conn is None
  assert not participant.connections
  reached.append(True)
  # Observe successful native drain; never fake or grant startup/capture authority.
  raise Drained()
 maintenance.RuntimeMaintenance.settle_producers=settled
 maintenance.RuntimeMaintenance.retire_local_caches=retire
 maintenance._poll_local_pause_requested=requested
 storage._LocalPause.retire_startup=at_startup
 monitoring=asyncio.create_task(maintenance.monitor_app(object.__new__(TldwCli)))
 try:
  await asyncio.wait_for(first.wait(),2)
  if sys.argv[1]=='held':
   await asyncio.sleep(0.1)
   assert not reached and not monitoring.done()
   assert db._local.conn is connection and not connection.in_transaction
   with storage._lock:
    assert any(op.participant is participant for op in storage._operations)
  else:
   try:await asyncio.wait_for(monitoring,2)
   except Drained:pass
   except TimeoutError:
    assert finished.is_set() and not errors
    assert db._local.conn is connection and not connection.in_transaction
    raise AssertionError('completed worker left main-thread cache blocking local drain') from None
   else:raise AssertionError('monitor did not reach native drain')
   assert reached
 finally:
  release.set()
  monitoring.cancel()
  await asyncio.gather(monitoring,return_exceptions=True)
  thread.join(5);assert not thread.is_alive() and not errors
  maintenance.RuntimeMaintenance.settle_producers=original_settle
  maintenance.RuntimeMaintenance.retire_local_caches=original_retire
  maintenance._poll_local_pause_requested=original_probe
  storage._LocalPause.retire_startup=original_startup
 assert storage._pause is None
 assert db.get_connection().execute('SELECT 1').fetchone()[0]==1
 db.close_connection()

asyncio.run(main())
print('retired and reopened')
"""


@pytest.mark.parametrize("worker", ["completed", "held"])
def test_monitor_revisits_only_after_foreign_operation_finishes(tmp_path, worker):
    _run(tmp_path, worker, "late-worker", script=_SCRIPT, timeout=20)
