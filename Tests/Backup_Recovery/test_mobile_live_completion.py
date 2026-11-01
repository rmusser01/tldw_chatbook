"""Ordinary live completion reads retain native custody and one allowance."""

import json
import subprocess
import sys

import pytest

from Tests.Web_Server.test_mobile_served_console_browser import (
    REPOSITORY,
    _prepare_profile,
)

_WAL_RACE = r"""
import json,sys
from pathlib import Path
from Tests.real_profile_guard import install as guard,take_violations
guard()
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.config import get_atomic_config_snapshot
from tldw_chatbook.Backup_Recovery.profile_paths import canonical_console_paths,legacy_mobile_dir
from tldw_chatbook.Backup_Recovery.mobile_consolidation import CanonicalConsoleStores,require_mobile_cutover
from tldw_chatbook.Backup_Recovery import storage_admission as storage,sqlite_validation
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
values=get_atomic_config_snapshot().values
stores=CanonicalConsoleStores(*canonical_console_paths(values))
writer=CharactersRAGDB(stores.chat_path,'ordinary-wal-race')
commits=[]
def commit(label):
 note=writer.add_note('Exact native WAL write '+label,'EXACT_CONCURRENT_NOTE_'+str(len(commits)))
 assert note
 commits.append({'id':note,'phase':label})
original_state=storage._PreviewScope._source_state
original_version=sqlite_validation._version
def source_state(scope,source):
 result=original_state(scope,source)
 if Path(source)==stores.chat_path:commit('after-copy-stat')
 return result
def version(connection,owner):
 result=original_version(connection,owner)
 if owner=='db.chachanotes.primary':commit('during-read-transaction')
 return result
storage._PreviewScope._source_state=source_state
sqlite_validation._version=version
try:
 commit('before-gate')
 assert require_mobile_cutover(legacy_mobile_dir(values),stores) is None
 assert len(commits)>=2
 assert any(row['phase']=='during-read-transaction' for row in commits)
 assert all(writer.get_note_by_id(row['id']) for row in commits)
finally:
 storage._PreviewScope._source_state=original_state
 sqlite_validation._version=original_version
 writer.close_connection()
Path(sys.argv[1]).write_text(json.dumps({'native_wal_commits':len(commits),'commits':commits,'gate':'accepted'}))
assert not blocked_attempts() and not take_violations()
"""


@pytest.fixture(scope="module")
def completion_profile(tmp_path_factory):
    return _prepare_profile(tmp_path_factory.mktemp("completion-profile"))


def test_completed_cutover_allows_native_wal_commits(completion_profile):
    root, environment = completion_profile
    with (root / "wal-race.log").open("w") as output:
        result = subprocess.run(
            [sys.executable, "-c", _WAL_RACE, str(root / "wal-race.json")],
            cwd=REPOSITORY,
            env=environment,
            stdout=output,
            stderr=output,
            timeout=60,
            check=False,
        )
    assert result.returncode == 0, (
        str(root) + "\n" + (root / "wal-race.log").read_text()[-6000:]
    )
    assert json.loads((root / "wal-race.json").read_text())["native_wal_commits"] >= 2


_SCOPE_CONTROLS = r"""
import json,sys,sqlite3,threading
from pathlib import Path
from Tests.real_profile_guard import install as guard,take_violations
guard()
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.config import get_atomic_config_snapshot
from tldw_chatbook.Backup_Recovery.profile_paths import canonical_console_paths
from tldw_chatbook.Backup_Recovery import mobile_preview_budget as accounting,storage_admission as storage
from tldw_chatbook.Backup_Recovery.mobile_consolidation_recovery import _open_completion_read
from tldw_chatbook.Backup_Recovery.sqlite_validation import _check,_installed_owner,_restrict_connection
from tldw_chatbook.Chat.console_trace_read_budget import _mobile_cutover_budget,_TraceReadLimit
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite,_ordinary_connections
path=canonical_console_paths(get_atomic_config_snapshot().values)[0]
budget=_mobile_cutover_budget();custody=storage._CaptureFileDescriptors(None)
start=budget._remaining
outcomes=[]
def refuses(callback,code):
 try:callback()
 except (ValueError,TypeError,sqlite3.ProgrammingError) as error:
  assert code in str(error),(code,str(error))
 else:raise AssertionError('unexpectedly accepted '+code)
with accounting._completion_observation(budget,custody) as scope:
 assert accounting.current() is scope and scope.read_budget is budget and scope.custody is custody
 before_buffer=budget._remaining
 workspace=scope._buffer()
 assert len(workspace)==65536 and budget._remaining==before_buffer-65536-1024
 assert scope._buffer() is workspace and budget._remaining==before_buffer-65536-1024
 from tldw_chatbook.Backup_Recovery.mobile_rollback_io import observation
 import hashlib
 before_hash=hashlib.sha256(path.read_bytes()).hexdigest()
 observed=observation(path,budget,custody,threading.Event())
 assert observed and hashlib.sha256(path.read_bytes()).hexdigest()==before_hash
 with _open_completion_read(path) as connection:
  lease=_ordinary_connections[connection]
  assert accounting.connection_budget(connection) is budget
  try:connection.execute("DELETE FROM notes")
  except sqlite3.OperationalError as error:assert 'readonly' in str(error).lower()
  else:raise AssertionError('ordinary completion wrote rows')
  connection.execute('BEGIN')
  restrictions=_restrict_connection(connection,None)
  owner=_installed_owner('db.chachanotes.primary')
  issues,version=_check(connection,owner,owner.schema_policy(),restrictions)
  assert not issues,(issues,version)
  assert not scope.resources
  assert scope.read_budget is budget and budget._remaining<start
 assert lease not in storage._live_leases and connection not in _ordinary_connections
 assert not scope.connections
 refuses(lambda:accounting.connection_budget(connection),'mobile_completion_connection_changed')
 raw=sqlite3.connect(':memory:')
 try:refuses(lambda:scope.bind(raw,path),'mobile_completion_connection_changed')
 finally:raw.close()
 wrong=connect_private_sqlite('db.chachanotes.read',path,read_only=True,must_exist=True)
 try:refuses(lambda:scope.bind(wrong,path),'mobile_completion_connection_changed')
 finally:wrong.close()
 thread_errors=[]
 def other_thread():
  try:scope.check()
  except ValueError as error:thread_errors.append(str(error))
 worker=threading.Thread(target=other_thread);worker.start();worker.join()
 assert thread_errors==['mobile_completion_scope_inactive']
 for name in ('preview_scope','discovery_scope','capture_scope','maintenance_session'):
  previous=getattr(storage._local,name,None);setattr(storage._local,name,object())
  try:refuses(scope.check,'mobile_completion_scope_inactive')
  finally:setattr(storage._local,name,previous)
 outcomes.append('actual fixed-role schema/read-only/lease/reference-budget and negative provenance')
assert not scope.active and not scope.resources and not scope.connections
refuses(scope.check,'mobile_completion_scope_inactive')
assert accounting.current() is None
# The accounting context must not retire borrowed descriptor custody.
with custody.pinned_directory(path.parent) as fd:
 with accounting._completion_observation(budget,custody):pass
 assert storage.os.fstat(fd).st_ino==path.parent.stat().st_ino
for name in ('preview_scope','discovery_scope','capture_scope','maintenance_session'):
 previous=getattr(storage._local,name,None);setattr(storage._local,name,object())
 try:
  try:
   with accounting._completion_observation(budget,custody):raise AssertionError('nested context accepted')
  except ValueError as error:assert str(error)=='mobile_completion_scope_nested'
 finally:setattr(storage._local,name,previous)
# Scarcity before SQL allocation cannot renew the original allowance.
budget.reserve(budget._remaining-2048)
with accounting._completion_observation(budget,custody) as exhausted:
 assert budget._remaining==0
 try:
  with _open_completion_read(path):raise AssertionError('exhausted scope opened')
 except _TraceReadLimit:pass
 assert not exhausted.connections
custody.retire();assert not custody.fds and not custody.close_failed
assert not blocked_attempts() and not take_violations()
Path(sys.argv[1]).write_text(json.dumps({'outcomes':outcomes,'same_budget':True,'physical_close':True,'borrowed_custody':True,'exhaustion':True}))
"""


def test_completion_accounting_keeps_native_provenance_and_retirement(
    completion_profile,
):
    root, environment = completion_profile
    with (root / "completion-controls.log").open("w") as output:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _SCOPE_CONTROLS,
                str(root / "completion-controls.json"),
            ],
            cwd=REPOSITORY,
            env=environment,
            stdout=output,
            stderr=output,
            timeout=60,
            check=False,
        )
    assert result.returncode == 0, (
        str(root) + "\n" + (root / "completion-controls.log").read_text()[-8000:]
    )


_CLOSE_CONTROLS = r"""
import asyncio,json,sys,sqlite3
from pathlib import Path
from Tests.real_profile_guard import install as guard,take_violations
guard()
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.config import get_atomic_config_snapshot
from tldw_chatbook.Backup_Recovery.profile_paths import canonical_console_paths
from tldw_chatbook.Backup_Recovery import mobile_preview_budget as accounting,storage_admission as storage
from tldw_chatbook.Backup_Recovery.mobile_consolidation_recovery import _open_completion_read
from tldw_chatbook.Chat.console_trace_read_budget import _mobile_cutover_budget
from tldw_chatbook.DB.private_sqlite import _ordinary_connections
path=canonical_console_paths(get_atomic_config_snapshot().values)[0]
custody=storage._CaptureFileDescriptors(None)
mode=sys.argv[1]
try:
 with accounting._completion_observation(_mobile_cutover_budget(),custody) as scope:
  with _open_completion_read(path) as connection:
   lease=_ordinary_connections[connection]
   connection.execute('BEGIN')
   connection.execute('SELECT count(*) FROM sqlite_schema').fetchone()
   if mode=='cancel':raise asyncio.CancelledError('exact-native-cancellation')
   if mode=='raise':
    def failed_close(self):raise OSError('exact-close-failure')
    type(connection).close=failed_close
   else:type(connection).close=lambda self:None
except asyncio.CancelledError as error:
 assert mode=='cancel' and str(error)=='exact-native-cancellation'
 assert connection not in _ordinary_connections and lease not in storage._live_leases
 try:sqlite3.Connection.in_transaction.__get__(connection)
 except sqlite3.ProgrammingError:pass
 else:raise AssertionError('cancel left native SQLite open')
 assert not scope.connections and scope not in accounting._uncertain_completion_scopes
except ValueError as error:
 assert mode in {'raise','noop'} and str(error)=='mobile_completion_connection_not_retired'
 assert scope in accounting._uncertain_completion_scopes and scope.connections[connection] is lease
 assert connection in _ordinary_connections and lease in storage._live_leases
 assert lease.resource_close_failed and sqlite3.Connection.in_transaction.__get__(connection)
else:raise AssertionError('missing expected close/cancel result')
assert not scope.active
custody.retire()
assert not blocked_attempts() and not take_violations()
Path(sys.argv[2]).write_text(json.dumps({'mode':mode,'physical_close':mode=='cancel','native_hold_retained_until_process_exit':mode!='cancel','scope_inactive':True}))
"""


@pytest.mark.parametrize("mode", ["cancel", "raise", "noop"])
def test_completion_close_or_cancellation_keeps_physical_custody(
    completion_profile, mode
):
    root, environment = completion_profile
    with (root / f"completion-close-{mode}.log").open("w") as output:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _CLOSE_CONTROLS,
                mode,
                str(root / f"completion-close-{mode}.json"),
            ],
            cwd=REPOSITORY,
            env=environment,
            stdout=output,
            stderr=output,
            timeout=60,
            check=False,
        )
    assert result.returncode == 0, (
        str(root) + "\n" + (root / f"completion-close-{mode}.log").read_text()[-8000:]
    )
