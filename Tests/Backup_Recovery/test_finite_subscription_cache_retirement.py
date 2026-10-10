"""Original admitted Subs close must retire new A and preserve caller-owned B."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import json,sqlite3,sys,threading
from pathlib import Path
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import participants,storage_admission as storage
from tldw_chatbook.DB import Subscriptions_DB as module
from tldw_chatbook.DB.private_sqlite import _ordinary_connections
from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
route,outcome=sys.argv[1:]
assert route=='subscriptions' and outcome in {'cache_empty','cache_foreign'}
db_a=module.SubscriptionsDB(Path.home()/'subs-A.db',client_id='native-retirement')
db_b=module.SubscriptionsDB(Path.home()/'subs-B.db',client_id='native-retirement')
main_a,main_b=db_a.conn,db_b.conn
main_a.execute('SELECT 1');main_b.execute('SELECT 1')
originals=(participants.run_finite_local_worker,participants._core_closing,
 participants._core_closing.__wrapped__,module.SubscriptionsDB,
 module.SubscriptionsDB.__dict__['close'],module.SubscriptionsDB.__dict__['conn'])
observer=OriginalStorageUnitObserver({},lambda:True,lambda unit:None)
for function in (originals[0],originals[1],originals[2],originals[4],originals[5].fget,
 originals[5].fget.__wrapped__):observer._pin(function)
observer.slots.extend([(participants,'run_finite_local_worker',originals[0]),
 (participants,'_core_closing',originals[1]),(originals[1],'__wrapped__',originals[2]),
 (module,'SubscriptionsDB',originals[3]),(originals[3],'close',originals[4]),
 (originals[3],'conn',originals[5])])
monitor=sys.monitoring
observer.tool=next(slot for slot in range(5,0,-1)
 if slot!=monitor.DEBUGGER_ID and monitor.get_tool(slot) is None)
monitor.use_tool_id(observer.tool,'finite-subscription-captured-close')
code=originals[2].__code__;observer.codes[code]='admitted_close'
observer.active=observer.installed=True
armed=False;worker_thread=None;a=b=None;local_a=local_b=None
lease_a=lease_b=None;gates=[];errors=[];facts={}

def alive(connection):
 try:return connection.execute('SELECT 1').fetchone()[0]==1
 except sqlite3.ProgrammingError:return False

def resource(connection,database,lease):
 with storage._lock:
  return dict(physical_live=alive(connection),
   ordinary_registered=_ordinary_connections.get(connection) is lease,
   lease_live=lease in storage._live_leases,
   core_registered=database._maintenance_participant.connections.get(connection) is lease)

def yielded(selected,offset,value):
 if not observer.active or selected is not code:return
 try:
  frame=observer._frame(code)
  assert frame.f_globals is originals[2].__globals__
  is_a=frame.f_locals.get('repository') is db_a
  if not armed or not is_a:return
  assert threading.current_thread() is worker_thread
  assert value is True and frame.f_locals.get('connection') is a
  assert frame.f_locals.get('reserved') is True
  assert db_a._local is local_a and local_a.conn is a
  assert a is not b and _ordinary_connections.get(a) is lease_a
  assert lease_a.resource_thread is worker_thread
  assert db_a._maintenance_participant.connections.get(a) is lease_a
  del frame
  assert not gates;gates.append(dict(admitted_A=True,same_thread=True))
  db_a._local=threading.local() if outcome=='cache_empty' else local_b
 except BaseException as error:observer.invalid.append('yield:'+type(error).__name__)

observer.registered={monitor.events.PY_START:observer._start,
 monitor.events.PY_RETURN:observer._return,monitor.events.PY_YIELD:yielded}
try:
 for event,callback in observer.registered.items():
  assert monitor.register_callback(observer.tool,event,callback) is None
 monitor.set_local_events(observer.tool,code,
  monitor.events.PY_START|monitor.events.PY_RETURN|monitor.events.PY_YIELD)
 assert monitor.get_events(observer.tool)==0
except BaseException:
 observer.close();db_a.close();db_b.close()
 raise

def callback():
 global a,local_a,lease_a
 a=db_a.conn;local_a=db_a._local;lease_a=_ordinary_connections[a]
 assert local_a.conn is a and lease_a.resource_thread is worker_thread
 assert db_a._maintenance_participant.connections[a] is lease_a
 assert a.execute('SELECT 1').fetchone()[0]==1
 return 'rows'

def worker():
 global b,local_b,lease_b,worker_thread,armed
 worker_thread=threading.current_thread()
 b=db_b.conn;local_b=db_b._local;lease_b=_ordinary_connections[b]
 b.execute('BEGIN')
 assert b.in_transaction and lease_b.resource_thread is worker_thread
 try:
  armed=True
  assert participants.run_finite_local_worker(callback)=='rows'
  armed=False
  assert not observer.invalid and len(gates)==1
  facts['A']=resource(a,db_a,lease_a)
  facts['B']=resource(b,db_b,lease_b)
  facts['B_transaction_live']=alive(b) and b.in_transaction
  facts['B_cache_matches']=db_b._local is local_b and local_b.conn is b
  facts['A_cache_replaced']=db_a._local is not local_a
 finally:
  armed=False
  # Original same-thread fixture owners restore/close their own actual caches.
  # This cleanup is not part of the product acceptance snapshots above.
  db_a._local=local_a if local_a is not None else db_a._local
  if a is not None and alive(a):a.rollback()
  db_a.close()
  if alive(b):b.rollback()
  db_b.close()
  if a is not None:assert not alive(a)
  assert not alive(b)
  with storage._lock:
   assert a not in _ordinary_connections and b not in _ordinary_connections
   assert lease_a not in storage._live_leases and lease_b not in storage._live_leases
   assert a not in db_a._maintenance_participant.connections
   assert b not in db_b._maintenance_participant.connections

def observed_worker():
 try:worker()
 except BaseException as error:errors.append(type(error).__name__)

thread=threading.Thread(target=observed_worker,name='finite-subs-close-owner')
try:
 thread.start();thread.join(20)
 assert not thread.is_alive() and not errors,errors
 assert main_a.execute('SELECT 1').fetchone()[0]==1
 assert main_b.execute('SELECT 1').fetchone()[0]==1
finally:
 armed=False
 if thread.is_alive():thread.join(20)
 assert not thread.is_alive(),'actual subscription worker did not retire'
 db_a.close();db_b.close()
 receipt=observer.close()
 assert receipt['complete'] and receipt['original_source_current'],receipt
 assert receipt['global_events']==0 and receipt['hooks_retired_before_inactive'],receipt
 assert monitor.get_tool(observer.tool) is None
 with storage._lock:
  census=dict(ordinary=len(storage._live_leases-set(storage._startups.values())),
   pending=len(storage._pending_acquisitions),core=len(storage._operations),
   raw=len(storage._raw_operations),retiring=len(storage._retiring_holds))
 assert not any(census.values()),census
 facts.update(observer=receipt,census=census,actual_gate=gates,worker_retired=True)
 print(json.dumps(facts,sort_keys=True))
assert not facts['A']['physical_live'],facts
assert not facts['A']['ordinary_registered'] and not facts['A']['lease_live'],facts
assert not facts['A']['core_registered'],facts
assert all(facts['B'].values()) and facts['B_transaction_live'] and facts['B_cache_matches'],facts
print('retired and reopened')
"""


@pytest.mark.parametrize("outcome", ["cache_empty", "cache_foreign"])
def test_subscription_finite_worker_retires_captured_handle_only(tmp_path, outcome):
    _run(tmp_path, "subscriptions", outcome, script=_SCRIPT)
