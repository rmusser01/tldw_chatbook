"""Native initializer followers wait for changes without repeating path probes."""

import pytest

from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT
from Tests.Backup_Recovery.test_home_citation_retirement import _run


_WAIT = (
    _SCRIPT.split("assert config.get_cli_setting")[0]
    + r"""
import sys,threading,time
from contextlib import nullcontext
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import _core_operation
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

case=sys.argv[1]
db=CharactersRAGDB(data/'fixture'/'wait.db',client_id='initialization-wait')
db.close_connection()
path=db.db_path
admitted,proceed,leading,release,waiting=[threading.Event() for _ in range(5)]
errors=[];leader_errors=[];effects=[];checks=[];waits=[]
pause=None
check_code=storage._Acquisition.check.__code__
initializing_code=storage._Acquisition.initializing.__wrapped__.__code__
wait_code=threading.Condition.wait.__code__
old_main,old_thread=sys.getprofile(),threading.getprofile()

def observe(frame,event,arg):
 if event!='call' or threading.current_thread().name!='initialization-follower':return
 if frame.f_code is check_code and frame.f_back.f_code is initializing_code:
  checks.append(time.monotonic())
 if frame.f_code is wait_code and frame.f_locals.get('self') is storage._changed:
  waits.append(frame.f_locals.get('timeout'))
  waiting.set()

def follower():
 try:
  # The nested acquisition carries a genuinely admitted operation. Its original
  # check performs real parent stat/resolve calls, reproducing the costly loop.
  with (nullcontext() if case=='pause' else _core_operation(db)):
   admitted.set();assert proceed.wait(10)
   with storage.acquire_storage(path) as lease:
    lease.execution_context(path)
    effects.append('admitted')
 except BaseException as error:errors.append(error)
 finally:db.close_connection()

def leader():
 attempt=storage._Acquisition()
 try:
  with attempt.initializing(root,path):
   leading.set();assert release.wait(10)
   if case=='leader-error':raise RuntimeError('original leader failed')
 except BaseException as error:leader_errors.append(error)
 finally:attempt.close()

first=threading.Thread(target=follower,name='initialization-follower')
second=threading.Thread(target=leader,name='initialization-leader')
threading.setprofile_all_threads(observe)
try:
 first.start();assert admitted.wait(5),errors
 checks.clear();waits.clear();waiting.clear()
 second.start();assert leading.wait(5),leader_errors
 proceed.set();assert waiting.wait(5),errors
 # Acquiring the same condition here witnesses the follower's actual release
 # into wait, rather than merely its entry into the profiler callback.
 with storage._changed:
  before=len(checks)
  assert before>=1 and not effects
 if case=='quiet':
  time.sleep(.25)
  with storage._changed:
   print('original initializing checks before/after:',before,len(checks),'wait timeouts:',waits,flush=True)
   assert len(checks)==before,'parked follower repeated native path validation without notification'
 elif case=='spurious':
  with storage._changed:storage._changed.notify_all()
  deadline=time.monotonic()+3
  while len(checks)==before:
   assert time.monotonic()<deadline
   time.sleep(.005)
  with storage._changed:
   assert not effects and first.is_alive()
 elif case=='pause':
  pause=storage._begin_local_pause()
  first.join(3)
  assert not first.is_alive(),'pause did not wake the parked acquisition'
  assert errors and errors[0].args==('storage_locally_paused',),errors
  assert not effects and not release.is_set()
 elif case=='path-drift':
  db.db_path=path.with_name('other.db')
 elif case=='source-drift':
  os.environ['TLDW_CONFIG_PATH']=str(selected.with_name('other.toml'))
 release.set()
 first.join(5);second.join(5)
 assert not first.is_alive() and not second.is_alive()
 if case=='leader-error':
  assert len(leader_errors)==1 and leader_errors[0].args==('original leader failed',)
 else:assert not leader_errors,leader_errors
 if case in {'pause','path-drift','source-drift'}:
  assert not effects and len(errors)==1,errors
  assert isinstance(errors[0],bootstrap.RecoveryRequired),errors
  if case=='path-drift':assert errors[0].args==('operation_path_outside_scope',),errors
  if case=='source-drift':assert errors[0].args==('execution_selection_changed',),errors
 else:assert not errors and effects==['admitted'],errors
finally:
 proceed.set();release.set()
 if first.ident is not None:first.join(10)
 if second.ident is not None:second.join(10)
 db.db_path=path
 os.environ['TLDW_CONFIG_PATH']=str(selected)
 if pause is not None:pause.resume()
 threading.setprofile_all_threads(old_thread);sys.setprofile(old_main)
 db.close_connection();storage._shutdown()
assert not first.is_alive() and not second.is_alive()
assert not storage._pending_acquisitions and not storage._operations
assert not storage._raw_operations and not storage._live_leases
assert not storage._holds and not storage._retiring_holds
print('retired and reopened')
"""
)


@pytest.mark.parametrize(
    "case", ["quiet", "spurious", "pause", "leader-error", "path-drift", "source-drift"]
)
def test_original_native_initialization_wait_preserves_checks_and_retirement(
    tmp_path, case
):
    _run(tmp_path, case, "initialization", script=_WAIT, timeout=40)


_RECURSIVE = (
    _SCRIPT.split("assert config.get_cli_setting")[0]
    + r"""
from tldw_chatbook.Backup_Recovery import storage_admission as storage
first=storage._Acquisition();second=storage._Acquisition()
try:
 with first.initializing(root,selected):
  try:
   with second.initializing(root,selected):raise AssertionError('recursive body entered')
  except bootstrap.RecoveryRequired as error:
   assert error.args==('recursive_authority_initialization',)
  assert first.initializing_root==root
  assert second.initializing_root is None
 assert first.initializing_root is None
finally:
 second.close();first.close();storage._shutdown()
assert not storage._pending_acquisitions and not storage._live_leases
print('retired and reopened')
"""
)


def test_same_thread_original_initializer_recursion_still_refuses(tmp_path):
    _run(tmp_path, "recursive", "initialization", script=_RECURSIVE, timeout=30)
