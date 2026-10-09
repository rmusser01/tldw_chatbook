"""Real scope-return races preserve exact publication and retirement."""

import os
import pytest
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import hashlib,json,os,sqlite3,sys,threading
from pathlib import Path
from types import CodeType
from Tests import network_guard,real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install();real_profile_guard.install()
from loguru import logger
logger.remove()
route,fault=sys.argv[1:]
assert route=="scope_publication"
selector=Path(os.environ["TLDW_CONFIG_PATH"]).absolute()
data=Path(os.environ["XDG_DATA_HOME"]).absolute()
selector.write_text('[general]'+chr(10)+'users_name="scope-fence"'+chr(10)+'[paths]'+chr(10)+'data_dir="'+data.as_posix()+'"'+chr(10),encoding="utf-8")
selector.chmod(0o600)

def shape(code):
 return (code.co_code,code.co_exceptiontable,code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars,tuple(shape(x) if type(x) is CodeType else x for x in code.co_consts))

def child(code,name):
 for part in name.split("."):
  code=next(x for x in code.co_consts if type(x) is CodeType and x.co_name==part)
 return code

def main():
 from tldw_chatbook import config
 from tldw_chatbook.Backup_Recovery import bootstrap,storage_admission as storage,raw_participants as raw,participants
 from tldw_chatbook.Backup_Recovery.control_records import admission_authority,bind_profile
 from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
 db=AgentRunsDB(config.get_user_data_dir()/"agent_runs.db",client_id="scope-publication",reconcile_on_init=False)
 with db.connection() as connection:
  assert connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type=?",("table",)).fetchone()[0]>0
 db.close()
 try:sqlite3.Connection.in_transaction.__get__(connection)
 except sqlite3.ProgrammingError:pass
 else:raise AssertionError("original seeded SQL object still open")
 storage._shutdown()
 root=bootstrap.default_bootstrap_root()
 authority=admission_authority(root);authority.register("profile",(selector.parent,data));bind_profile(root,selector,("profile",),authority.control_root)
 holder=storage.acquire_storage(data/"independent.sqlite")
 saved_names=storage._holds[holder._key].names
 if fault=="continuation_retired":
  with selector.open("a",encoding="utf-8") as output:output.write("# changed fingerprint under original live owner"+chr(10))
 other=selector.with_name("other.toml");other.write_bytes(selector.read_bytes());other.chmod(0o600)
 source=Path(storage.__file__);before=source.read_bytes();module=sys.modules[storage.__name__]
 codes=(storage._scope.__code__,storage._acquire_storage.__code__,storage._ScopeProof.check.__code__)
 compiled=compile(before,str(source),"exec",dont_inherit=True)
 for code,name in zip(codes,("_scope","_acquire_storage","_ScopeProof.check"),strict=True):assert shape(code)==shape(child(compiled,name))
 originals=(storage._scope,storage._acquire_storage,storage._ScopeProof.check)
 defaults=tuple((f.__defaults__,f.__kwdefaults__,f.__closure__) for f in originals)
 entered,release,done=threading.Event(),threading.Event(),threading.Event()
 capture={};returned=[];errors=[];actor=None;ordinal=0;pause=None
 participant=participants._repository_participant(db)
 def observe(code,offset,value):
  nonlocal ordinal
  if threading.current_thread() is not actor:return
  frame=sys._getframe(1)
  assert frame.f_code is code and frame.f_globals is vars(storage)
  parent=frame.f_back
  while parent is not None and parent.f_code is not codes[1]:parent=parent.f_back
  assert parent is not None
  attempt=parent.f_locals["attempt"]
  if fault=="operation_revoked" and attempt.operation is None:return
  ordinal+=1
  if ordinal!=1:return
  assert attempt in storage._pending_acquisitions and attempt.thread is actor and attempt.task is None
  assert parent.f_locals["root"]==root and parent.f_locals["selector"]==selector
  assert value==("profile",)
  proof=frame.f_locals["proof"]
  assert type(proof) is storage._ScopeProof and proof.thread is actor
  assert proof.continuation==(fault=="continuation_retired")
  assert not storage._lock._is_owned(),"scope return still owns global coordinator"
  capture.update(attempt=attempt,operation=attempt.operation,continuation=proof.continuation)
  entered.set()
  assert release.wait(10),"original scope return not released"
 def read():
  lease=None
  try:
   if fault=="operation_revoked":
    with participant.operation():
     lease=storage.acquire_storage(db.db_path);returned.append(True);lease.close();lease=None
   else:
    lease=storage.acquire_storage(db.db_path);returned.append(True)
    assert lease.execution_context(db.db_path)[1]==("profile",)
    lease.close();lease=None
  except bootstrap.RecoveryRequired as error:errors.append(str(error))
  finally:
   if lease is not None:lease.close()
   db.close();done.set()
 monitor=sys.monitoring;tools=tuple(monitor.get_tool(i) for i in range(6));tool=next(i for i in range(6) if monitor.get_tool(i) is None)
 monitor.use_tool_id(tool,"scope-return-exact-publication")
 monitor.register_callback(tool,monitor.events.PY_RETURN,observe);monitor.set_local_events(tool,codes[0],monitor.events.PY_RETURN)
 try:
  actor=threading.Thread(target=read,name="scope-publication-original-reader");actor.start();assert entered.wait(10),"original first scope did not return"
  if fault=="cancel":capture["attempt"].cancel.set()
  elif fault=="pause":pause=storage._begin_local_pause()
  elif fault=="selection":os.environ["TLDW_CONFIG_PATH"]=str(other)
  elif fault in {"continuation_retired","saved_binding_retired"}:holder.close()
  elif fault=="operation_revoked":
   with storage._lock:storage._operations.remove(capture["operation"])
  else:raise AssertionError("unknown test fault")
 finally:
  release.set()
  if actor is not None:actor.join(20);assert not actor.is_alive() and done.is_set()
  os.environ["TLDW_CONFIG_PATH"]=str(selector)
  if pause is not None:pause.resume()
  monitor.set_local_events(tool,codes[0],0);monitor.register_callback(tool,monitor.events.PY_RETURN,None);monitor.free_tool_id(tool)
  holder.close();db.close();storage._shutdown()
 assert source.read_bytes()==before and sys.modules[storage.__name__] is module
 assert (storage._scope,storage._acquire_storage,storage._ScopeProof.check)==originals
 for f,code,metadata in zip(originals,codes,defaults,strict=True):assert f.__code__ is code and f.__globals__ is vars(storage) and (f.__defaults__,f.__kwdefaults__,f.__closure__)==metadata
 assert tuple(monitor.get_tool(i) for i in range(6))==tools and monitor.get_events(tool)==0
 with storage._lock:
  census=(len(storage._live_leases),len(storage._pending_acquisitions),len(storage._operations),len(storage._raw_operations),len(raw._states),len(storage._retiring_holds))
 assert not any(census),census
 assert not network_guard.blocked_attempts()
 if fault=="saved_binding_retired":assert returned==[True] and errors==[]
 else:
  assert not returned,"obsolete proof published a token"
  expected={"cancel":"storage_locally_paused","pause":"storage_locally_paused","selection":"execution_selection_changed","continuation_retired":"storage_scope_changed","operation_revoked":"operation_provenance_invalid"}[fault]
  assert errors==[expected],errors
 print("retired and reopened")
with user_fixture_default_owner():main()
"""

pytestmark = pytest.mark.skipif(
    os.name != "nt", reason="actual Windows scope-publication ownership"
)


@pytest.mark.parametrize(
    "fault",
    [
        "cancel",
        "pause",
        "selection",
        "continuation_retired",
        "saved_binding_retired",
        "operation_revoked",
    ],
)
def test_original_scope_return_keeps_exact_publication_fence(tmp_path, fault):
    _run(tmp_path, "scope_publication", fault, script=_SCRIPT)
