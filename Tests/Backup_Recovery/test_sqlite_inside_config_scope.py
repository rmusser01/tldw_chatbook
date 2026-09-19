"""Independent SQLite reopening must not borrow a config source's helpers."""

import pytest

from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_REOPEN = _SCRIPT.split("assert config.get_cli_setting")[0] + (  # nosec B608 - fixed Python child script, not interpolated SQL.
    r'''
import asyncio,sqlite3,sys,time
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB import private_sqlite
from tldw_chatbook.Backup_Recovery import config_participants,raw_participants as raw,storage_admission as storage
case=sys.argv[1]
db=AgentRunsDB(data/'fixture'/'agent-runs.db',client_id='fixture')
assert db.count_subagents_by_conversation(['fixture-conversation'])=={}
old=db._held_connection()
if case!='cached':
 db.close()
 try:old.execute('SELECT 1')
 except sqlite3.ProgrammingError:pass
 else:raise AssertionError('fixture must positively close the old connection')
native_connect=sqlite3.connect
original=asyncio.CancelledError() if case in {'cancel','cancel_selector'} else OSError('synthetic connector failure')
opened=[];pauses=[];outer=None;native_calls=0
custom_calls=[]
class CustomConnection(sqlite3.Connection):
 def __init__(self,*args,**kwargs):
  custom_calls.append(raw._runtime_operation())
  super().__init__(*args,**kwargs)

def observe_connect(*args,**kwargs):
 global native_calls
 native_calls+=1
 assert outer in raw._states and outer in storage._raw_operations
 assert raw._runtime_operation() is None,'SQLite preflight inherited config discovery'
 if case in {'error_selector','cancel_selector'}:
  os.environ['TLDW_CONFIG_PATH']=str(selected.with_name('other.toml'))
 if case in {'error','cancel','error_selector','cancel_selector'}:raise original
 if case=='pause_during':
  pauses.append(storage._begin_local_pause())
  assert not pauses[-1].drain(time.monotonic())
 connection=native_connect(*args,**kwargs)
 opened.append(connection)
 if case in {'selector_after_open','failed_close'}:
  os.environ['TLDW_CONFIG_PATH']=str(selected.with_name('other.toml'))
 if case=='failed_close':
  def refuse_close(self):
   self._admission_close_attempted=True
   raise original
  type(connection).close=refuse_close
 return connection

sqlite3.connect=observe_connect
try:
 with config_participants.operation(config) as outer:
  baseline=set(storage._live_leases)
  assert config.get_cli_setting('general','users_name')=='fixture'
  if case=='pause_before':pauses.append(storage._begin_local_pause())
  failed=None
  try:
   if case=='custom':
    private_sqlite.connect_private_sqlite('db.base',db.db_path_str,factory=CustomConnection)
   elif case=='getter':
    connection=db._held_connection()
    assert connection.execute('SELECT count(*) FROM agent_runs').fetchone()[0]==0
   else:assert db.count_subagents_by_conversation(['fixture-conversation'])=={}
  except BaseException as error:
   failed=error
  finally:os.environ['TLDW_CONFIG_PATH']=str(selected)
  assert raw._runtime_operation() is outer
  assert outer in raw._states and outer in storage._raw_operations
  if case in {'error','cancel','error_selector','cancel_selector'}:
   assert failed is original,(type(failed).__name__,type(original).__name__)
   assert not opened and set(storage._live_leases)==baseline
   if case in {'error_selector','cancel_selector'}:
    assert 'private_sqlite_outer_scope_revalidation_failed' in failed.__notes__
  elif case in {'selector_after_open','failed_close'}:
   assert isinstance(failed,bootstrap.RecoveryRequired),type(failed).__name__
   assert len(opened)==1
   if case=='failed_close':
    assert opened[0].execute('SELECT 1').fetchone()==(1,)
    retained=set(storage._live_leases)-baseline
    assert len(retained)==1 and next(iter(retained)).resource_close_failed
    assert 'private_sqlite_outer_scope_close_failed' in failed.__notes__
   else:
    try:opened[0].execute('SELECT 1')
    except sqlite3.ProgrammingError:pass
    else:raise AssertionError('failed outer revalidation leaked the new SQLite handle')
    assert set(storage._live_leases)==baseline
  elif case=='custom':
   assert type(failed) is RuntimeError and failed.args==('raw_source_helper_not_supported',)
   assert not custom_calls and native_calls==0
  elif case=='pause_before':
   assert type(failed) is bootstrap.RecoveryRequired and failed.args==('storage_locally_paused',)
   assert native_calls==0 and not opened
  else:
   if failed is not None:raise failed
   if case=='cached':assert native_calls==0 and db._held_connection() is old
   else:
    assert native_calls==1 and len(opened)==1
    assert private_sqlite._ordinary_connections[opened[0]] in storage._live_leases
  assert config.get_cli_setting('general','users_name')=='fixture'
  assert raw._runtime_operation() is outer
  if case not in {'pause_before','pause_during'}:
   assert config.save_setting_to_cli_config('general','users_name','updated')
   assert config.get_cli_setting('general','users_name')=='updated'
finally:
 sqlite3.connect=native_connect
 os.environ['TLDW_CONFIG_PATH']=str(selected)
 db.close()
 for pause in reversed(pauses):pause.resume()
assert raw._runtime_operation() is None and not storage._raw_operations
assert not storage._operations and not storage._pending_acquisitions
if case=='failed_close':
 pause=storage._begin_local_pause()
 assert not pause.drain(time.monotonic())
 assert opened[0].execute('SELECT 1').fetchone()==(1,)
 assert retained <= set(storage._live_leases)
 # The uncertain handle and its native admission remain until child exit.
 print('retired and reopened')
 sys.exit(0)
assert db.count_subagents_by_conversation(['fixture-conversation'])=={}
db.close()
if case=='custom':
 connection=private_sqlite.connect_private_sqlite('db.base',db.db_path_str,factory=CustomConnection)
 try:
  assert custom_calls==[None]
  assert connection.execute('SELECT count(*) FROM agent_runs').fetchone()[0]==0
 finally:connection.close()
print('retired and reopened')
'''
)


@pytest.mark.parametrize("case", [
    "cached", "reopened", "getter", "pause_before", "pause_during",
    "error", "cancel", "error_selector", "cancel_selector",
    "selector_after_open", "failed_close", "custom",
])
def test_native_sqlite_reopen_has_independent_admission_inside_config(tmp_path, case):
    _run(tmp_path, case, "sqlite-config-scope", script=_REOPEN, timeout=40)
