"""Evidence-only native hypothesis, not a savings assertion or implementation.

Root runs each route in a separate original private child. The original generic
worker guard remains installed; measured phases start only inside its accepted
body. Cold means the existing original cache-invalidation function was invoked
before this getter bundle; it is explicitly not a whole-Send baseline.
"""

import json

import pytest

from Tests.Backup_Recovery.test_activation_agents import _SCRIPT as _AGENT_SETUP
from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SCRIPT = (
    _AGENT_SETUP.split("effects=[]")[0]
    + r"""
import hashlib, inspect, json, sqlite3, sys, threading, time
from contextlib import nullcontext
from dataclasses import asdict
from types import CodeType, FunctionType
from tldw_chatbook.Agents.activation import worker_guard
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits
from tldw_chatbook.Agents import activation, automatic_work_budget, run_log
from tldw_chatbook.Backup_Recovery import config_participants, raw_participants, participants
from tldw_chatbook.DB import automatic_work, AgentRuns_DB, private_sqlite
from tldw_chatbook.Utils import windows_files

assert route in {'ungrouped','grouped'} and state=='approved'
service=AgentService(db,ToolCatalogRegistry(),chat_call=lambda **kwargs:None)
original_from_settings=AutomaticWorkLimits.from_settings.__func__
original_operation=config_participants.operation
operation_body=original_operation.__wrapped__
native_open=inspect.getattr_static(windows_files._Native,'open_handle')
get_setting=config.get_cli_setting
setting=run_log._setting
aliases={
 'from_settings':original_from_settings,'operation':original_operation,
 'native_open':native_open,'get_setting':get_setting,'setting':setting,
 'load_cli':config.load_cli_config_and_ensure_existence,
 'invalidation':config._invalidate_config_caches,'worker_guard':worker_guard,
 'admission_factory':private_sqlite._with_storage_admission,
 'native_transaction_descriptor':sqlite3.Connection.in_transaction,
}
for body,module in ((original_from_settings,automatic_work_budget),
                    (operation_body,config_participants),(native_open,windows_files),
                    (get_setting,config),(setting,run_log)):
 assert type(body) is FunctionType and body.__globals__ is vars(module)
 assert body.__module__==module.__name__
modules=(config,config_participants,raw_participants,storage,activation,
         automatic_work_budget,run_log,automatic_work,AgentRuns_DB,windows_files,
         private_sqlite,participants)
project=Path(config.__file__).resolve().parents[1]
paths={module.__name__:Path(module.__file__).resolve() for module in modules}
assert all(path.is_relative_to(project) for path in paths.values())
before={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
code_anchors={name:value.__code__ for name,value in aliases.items()
              if type(value) is FunctionType}
codes={native_open.__code__:'native_open',get_setting.__code__:'get_cli_setting',
       setting.__code__:'setting',operation_body.__code__:'config_operation'}
allowed_keys={f'max_autowake_{name}' for name in asdict(AutomaticWorkLimits())}
counts={}; custody={}; phases={}; failures=[]; connections=[]
def exact_admission_child(parent,name):
 children=[code for code in parent.co_consts
           if type(code) is CodeType and code.co_name==name]
 assert len(children)==1,(parent.co_name,name,len(children))
 return children[0]
admission_body_code=exact_admission_child(
 private_sqlite._with_storage_admission.__code__,'admitted')
admitted_class_code=exact_admission_child(admission_body_code,'AdmittedConnection')
admitted_close_code=exact_admission_child(admitted_class_code,'close')
phase='outside'; worker=None
monitor=sys.monitoring
tool=next((index for index in range(6) if monitor.get_tool(index) is None),None)
assert tool is not None
def observe(code,offset):
 if threading.current_thread() is not worker or phase=='outside': return
 label=codes[code]
 counts[phase][label]=counts[phase].get(label,0)+1
 if label=='setting':
  frame=sys._getframe(1)
  assert frame.f_code is code and frame.f_globals is vars(run_log)
  assert frame.f_locals['key'] in allowed_keys
  active=getattr(raw_participants._local,'operation',None)
  if active is not None:
   observed=raw_participants._live_state(active,None,False)
   assert observed.source is config and observed.thread is worker
  custody[phase].append(active is not None)
def census():
 with storage._lock:
  startup=set(storage._startups.values())
  return {'ordinary':len(storage._live_leases-startup),
          'pending':len(storage._pending_acquisitions),
          'operations':len(storage._operations),'raw':len(storage._raw_operations),
          'states':len(raw_participants._states),'retiring':len(storage._retiring_holds)}
def actual_limits_bundle():
 global phase
 actor=threading.current_thread()
 assert actor is worker
 for name in ('warm','invalidated_cold'):
  # The real worker admission already observes a warm cache. The next phase
  # deliberately invokes the original normal invalidation function, not an
  # assignment to cache/identity fields or a fake reader.
  if name=='invalidated_cold': config._invalidate_config_caches()
  initial_identity=config.current_config_identity()
  environment_before=os.environ.copy()
  expected_cli_calls=sum(run_log._env_override(key) is None for key in allowed_keys)
  phase=name; counts[name]={}; custody[name]=[]
  started=time.perf_counter()
  try:
   with (config_participants.operation(config) if route=='grouped' else nullcontext()):
    limits=AutomaticWorkLimits.from_settings()
  finally:
   elapsed=time.perf_counter()-started
   phase='outside'
  assert config.current_config_identity()==initial_identity
  assert os.environ.copy()==environment_before
  assert counts[name].get('setting')==6
  assert counts[name].get('get_cli_setting',0)==expected_cli_calls
  phases[name]={'limits':asdict(limits),'counts':dict(counts[name]),
                'getters_with_exact_live_config_custody':sum(custody[name]),
                'expected_cli_getter_count':expected_cli_calls,
                'environment_override_count':6-expected_cli_calls,
                'elapsed_seconds_diagnostic_only':elapsed}
 # The native ledger is real, with unchanged original worker guard and current
 # explicit limits. Do not count or conflate its SQL with the getter phase.
 ledger=db.automatic_work
 chain=ledger.create_chain('limits-measurement',root_submission_id='measured',limits=limits)
 assert ledger.snapshot(chain).limits==limits
 connection=db._thread_local.conn
 concrete=type(connection)
 assert len(concrete.__bases__)==1 and concrete.__bases__[0] is sqlite3.Connection
 closer=inspect.getattr_static(concrete,'close')
 assert type(closer) is FunctionType and closer.__code__ is admitted_close_code
 assert closer.__globals__ is vars(private_sqlite)
 assert concrete.__module__==private_sqlite.__name__
 # This subclass IS the native handle; no generic proxy unwrap is used.
 assert sqlite3.Connection.in_transaction.__get__(connection) is False
 with storage._lock:
  lease=private_sqlite._ordinary_connections.get(connection)
  participant=db._maintenance_participant
  assert lease in storage._live_leases and lease.resource_participant is participant
  assert participant in participants._installed_repositories
  assert participant.repository() is db and participant.owner_id=='db.agent_runs'
  assert participant.path==db.db_path and lease.resource_path==participant.path
  assert lease.resource_thread is worker and participant.connections[connection] is lease
 connections.append((connection,lease,participant))
def invoke():
 try: worker_guard(service)(actual_limits_bundle)()
 except BaseException as error: failures.append(error)

try:
 monitor.use_tool_id(tool,'automatic-work-limits-evidence')
 monitor.register_callback(tool,monitor.events.PY_START,observe)
 for code in codes: monitor.set_local_events(tool,code,monitor.events.PY_START)
 assert monitor.get_events(tool)==0
 worker=threading.Thread(target=invoke,name='original-limits-control')
 worker.start(); worker.join(40)
 assert not worker.is_alive()
 if failures: raise failures[0]
finally:
 for code in codes: monitor.set_local_events(tool,code,0)
 monitor.register_callback(tool,monitor.events.PY_START,None)
 assert monitor.get_events(tool)==0
 monitor.free_tool_id(tool)
 assert monitor.get_tool(tool) is None
 assert not worker or not worker.is_alive()
assert len(connections)==1
for connection,lease,participant in connections:
 try: sqlite3.Connection.in_transaction.__get__(connection)
 except sqlite3.ProgrammingError: pass
 else: raise AssertionError('original worker retained its native SQLite handle')
 with storage._lock:
  assert lease not in storage._live_leases
  assert connection not in private_sqlite._ordinary_connections
  assert connection not in participant.connections
assert phases['warm']['limits']==phases['invalidated_cold']['limits']
if route=='grouped':
 assert all(phases[name]['getters_with_exact_live_config_custody']==6 for name in phases)
after={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
assert before==after
current={'from_settings':AutomaticWorkLimits.from_settings.__func__,
 'operation':config_participants.operation,
 'native_open':inspect.getattr_static(windows_files._Native,'open_handle'),
 'get_setting':config.get_cli_setting,'setting':run_log._setting,
 'load_cli':config.load_cli_config_and_ensure_existence,
 'invalidation':config._invalidate_config_caches,'worker_guard':activation.worker_guard,
 'admission_factory':private_sqlite._with_storage_admission,
 'native_transaction_descriptor':sqlite3.Connection.in_transaction}
assert all(current[name] is original for name,original in aliases.items())
assert all(current[name].__code__ is code for name,code in code_anchors.items())
db.close()
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None: startup.close()
final=census(); assert not any(final.values()), final
assert not blocked_attempts()
receipt={'route':route,'diagnostic_only':True,'phases':phases,'final_census':final,
         'source_before':before,'source_after':after,'global_events':0,
         'tool_retired':monitor.get_tool(tool) is None,
         'worker_sqlite_physically_closed':True,'original_callbacks_unchanged':True,
         'worker_connection_kind':'registered_native_admitted_subclass'}
(base/'limits-measurement.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["ungrouped", "grouped"])
def test_original_automatic_limits_native_grouping_hypothesis(tmp_path, route):
    _run(tmp_path, route, "approved", script=_SCRIPT, timeout=60)
    receipt = json.loads((tmp_path / "limits-measurement.json").read_text("utf-8"))
    assert receipt["route"] == route and receipt["diagnostic_only"]
    assert receipt["source_before"] == receipt["source_after"]
    assert not any(receipt["final_census"].values())
    assert receipt["global_events"] == 0 and receipt["tool_retired"]
    assert receipt["worker_sqlite_physically_closed"]
    print(json.dumps({"route": route, "phases": receipt["phases"]}, sort_keys=True))
