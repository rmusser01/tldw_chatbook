"""Checked config operations preserve the existing rebuild/file lock order."""

import pytest

from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_BASE = (
    _SCRIPT.split("assert config.get_cli_setting")[0]
    + r"""
import sys,threading
from tldw_chatbook.Backup_Recovery import config_participants,raw_participants as raw,storage_admission as storage
case=sys.argv[1]
config.get_runtime_config_snapshot()
config.runtime_capture_policy()
original_bytes=selected.read_bytes()
file_lock=config._config_file_lock()
rebuild_lock=config._settings_rebuild_lock()
"""
)

_CONCURRENT = (
    _BASE
    + r"""
key='TLDW_CONSOLE_TRACE_NORMALIZED_WRITES'
os.environ[key]='false' if os.environ.get(key)!='false' else 'true'
original_file_getter=config._config_file_lock
original_rebuild_getter=config._settings_rebuild_lock
main_id=threading.get_ident()
at_file=threading.Event();finished=threading.Event()
results=[];errors=[];armed=True
def observed_file_getter():
 if threading.current_thread().name=='runtime-policy':
  assert rebuild_lock._is_owned()
  at_file.set()
 return original_file_getter()
def read_policy():
 try:results.append(config.runtime_capture_policy())
 except BaseException as error:errors.append(error)
worker=threading.Thread(target=read_policy,name='runtime-policy',daemon=True)
def observed_rebuild_getter():
 global armed
 if threading.get_ident()==main_id and armed:
  armed=False
  worker.start()
  assert at_file.wait(5),'public policy did not reach the file lock'
 return original_rebuild_getter()
def watchdog():
 if not finished.wait(10):
  print('native_config_rebuild_file_lock_cycle',flush=True)
  os._exit(9)
config._config_file_lock=observed_file_getter
config._settings_rebuild_lock=observed_rebuild_getter
threading.Thread(target=watchdog,daemon=True).start()
try:
 if case=='runtime':
  snapshot=config.get_runtime_config_snapshot()
  assert snapshot.generation==config.get_runtime_config_generation()
 else:
  snapshot=config.read_cli_config_snapshot()
  assert snapshot.path==selected and snapshot.serialized==original_bytes.decode()
 worker.join(5)
 assert not worker.is_alive() and not errors
 assert len(results)==1 and not armed
 assert selected.read_bytes()==original_bytes
 assert not file_lock._is_owned() and not rebuild_lock._is_owned()
 assert not storage._pending_acquisitions and not storage._raw_operations
 assert raw._runtime_operation() is None
finally:
 finished.set()
 config._config_file_lock=original_file_getter
 config._settings_rebuild_lock=original_rebuild_getter
 storage._shutdown()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("case", ["runtime", "serialized"])
def test_native_config_snapshot_and_public_capture_policy_finish_together(
    tmp_path, case
):
    _run(tmp_path, case, "config-order", script=_CONCURRENT, timeout=40)


_WAIT = (
    _BASE
    + r"""
held=threading.Event();release=threading.Event();waiting=threading.Event();done=threading.Event()
errors=[];entered=[];pause=None
blocked=rebuild_lock if case=='rebuild' else file_lock
def holder():
 with blocked:
  held.set()
  assert release.wait(10)
owner=threading.Thread(target=holder,daemon=True)
original_check=storage._Acquisition.check
def observed_check(self,*args,**kwargs):
 result=original_check(self,*args,**kwargs)
 if threading.current_thread().name=='config-waiter':waiting.set()
 return result
storage._Acquisition.check=observed_check
failure_before=config._CONFIG_PERSISTENCE_ERROR
def acquire():
 try:
  with config_participants.operation(config):entered.append(True)
 except BaseException as error:errors.append(error)
 finally:done.set()
worker=threading.Thread(target=acquire,name='config-waiter',daemon=True)
try:
 owner.start();assert held.wait(3)
 worker.start();assert waiting.wait(3)
 pause=storage._begin_local_pause()
 assert done.wait(3),'config lock wait ignored native pause'
 worker.join(3)
 assert not entered and not worker.is_alive()
 assert len(errors)==1 and type(errors[0]) is bootstrap.RecoveryRequired
 assert errors[0].args==('storage_locally_paused',)
 assert config._CONFIG_PERSISTENCE_ERROR==failure_before
 assert not storage._pending_acquisitions and not storage._raw_operations
 # In the FILE case the worker already acquired REBUILD; cancellation must
 # release that partial acquisition while the original FILE holder stays live.
 if case=='file':
  assert rebuild_lock.acquire(timeout=.5),'partial rebuild lock leaked'
  rebuild_lock.release()
finally:
 release.set()
 owner.join(3);worker.join(3)
 if pause is not None:pause.resume()
 storage._Acquisition.check=original_check
assert config.get_cli_setting('general','users_name')=='fixture'
assert selected.read_bytes()==original_bytes
storage._shutdown()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("case", ["rebuild", "file"])
def test_native_pause_cancels_each_config_lock_wait_without_leaking_partial_locks(
    tmp_path, case
):
    _run(tmp_path, case, "config-order", script=_WAIT, timeout=40)


_CORE = (
    _BASE
    + r"""
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
db=CharactersRAGDB(data/'fixture'/'core.db',client_id='fixture')
db.add_note('Preserved core','Existing native content')
failure=ValueError('synthetic config body failure')
try:
 with storage._repository_operation(db._maintenance_participant) as core:
  assert storage._operation_local.operation is core
  try:
   with config_participants.operation(config):
    assert storage._operation_local.operation is None
    assert rebuild_lock._is_owned() and file_lock._is_owned()
    assert config.get_cli_setting('general','users_name')=='fixture'
    if case=='failure':raise failure
  except ValueError as error:
   assert case=='failure' and error is failure
  else:assert case=='success'
  assert storage._operation_local.operation is core
  storage._check_operation(core,core.path)
  assert not rebuild_lock._is_owned() and not file_lock._is_owned()
  assert db.get_note_by_title('Preserved core')['content']=='Existing native content'
 assert storage._operation_local.operation is None
 assert not storage._pending_acquisitions and not storage._raw_operations
 assert (config._CONFIG_PERSISTENCE_ERROR is not None)==(case=='failure')
 assert selected.read_bytes()==original_bytes
finally:
 db.close_connection()
 storage._shutdown()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("case", ["success", "failure"])
def test_native_config_lock_cleanup_restores_the_enclosing_repository_operation(
    tmp_path, case
):
    _run(tmp_path, case, "config-order", script=_CORE, timeout=40)
