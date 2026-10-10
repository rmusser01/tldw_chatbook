"""Workspace-only additions to the unchanged finite Library worker controls."""

import pytest

from Tests.Backup_Recovery.test_finite_library_workers import _SCRIPT as _LIBRARY_SCRIPT
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_WORKSPACE_BRANCH = r"""if route in {'workspace_first_factory','workspace_cached_factory'}:
 from tldw_chatbook.Tools import workspace_file_roots as roots
 from tldw_chatbook.Workspaces import registry_service
 from tldw_chatbook.DB import Workspace_DB
 assert roots._registry_factory is roots._default_registry_factory
 assert roots._default_registry_instance is None
 db=None
 def get():
  global db
  registry=roots._default_registry_factory()
  assert type(registry) is registry_service.LocalWorkspaceRegistryService
  db=registry.db
  assert type(db) is Workspace_DB.WorkspaceDB
  registry.get_active_workspace()
  return db._held_connection()
 def close():
  if db is not None:db.close()
elif route=='workspaces':
 from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
 cls=type('CustomDB',(WorkspaceDB,),{}) if outcome=='custom' else WorkspaceDB
 db=cls(':memory:' if outcome=='memory' else root/'workspaces.db',client_id='test')
 get,close=db._held_connection,db.close
elif route=='media':"""

_PHYSICAL_SETUP = r"""
from types import CodeType,FunctionType
from tldw_chatbook.DB import private_sqlite
from tldw_chatbook.Backup_Recovery import storage_admission as storage
import json
def nested_codes(code):
 yield code
 for value in code.co_consts:
  if type(value) is CodeType:yield from nested_codes(value)
original_factory=private_sqlite._with_storage_admission
factory_code=original_factory.__code__
factory_globals=original_factory.__globals__
original_codes=tuple(nested_codes(factory_code))
def native_type_current(native_type):
 if type(native_type) is not type:return False
 if native_type.__bases__!=(sqlite3.Connection,):return False
 if native_type.__mro__!=(native_type,sqlite3.Connection,object):return False
 for name in ('__new__','close','__del__'):
  function=vars(native_type).get(name)
  if isinstance(function,staticmethod):function=function.__func__
  if type(function) is not FunctionType:return False
  if function.__globals__ is not factory_globals:return False
  if not any(function.__code__ is code for code in original_codes if code.co_name==name and ".AdmittedConnection." in code.co_qualname):return False
 return True
def held_identity(connection):
 if outcome=='custom':
  assert route=='workspaces'
  assert type(db) is cls and cls is not WorkspaceDB
  assert cls.__bases__==(WorkspaceDB,)
  assert '_maintenance_participant' not in vars(db)
  with storage._lock:
   lease=private_sqlite._ordinary_connections.get(connection)
   assert type(lease) is storage.StorageLease
   assert lease in storage._live_leases
   assert vars(lease).get('resource_participant') is None
   assert lease.resource_thread is threading.current_thread()
   assert lease.resource_policy is private_sqlite._SQLITE_OWNER_POLICIES['db.base']
   assert lease.resource_path==db.db_path
  native_type=type(connection)
  assert native_type.__module__==private_sqlite.__name__
  assert native_type.__qualname__.endswith('.AdmittedConnection')
  assert native_type_current(native_type)
  close_function=vars(native_type)['close']
  cells=dict(zip(close_function.__code__.co_freevars,close_function.__closure__))
  assert cells['lease'].cell_contents is lease
  assert cells['capture_lease'].cell_contents is False
  assert cells['constructing'].cell_contents is False
  assert sqlite3.Connection.in_transaction.__get__(connection) is False
  generic_identity[id(connection)]=lease
  return None
 if db.is_memory_db:return None
 with storage._lock:
  participant=db._maintenance_participant
  lease=participant.connections.get(connection)
  assert lease is not None
  assert participant.repository() is db
  assert participant.owner_id=='db.workspaces'
  assert private_sqlite._ordinary_connections.get(connection) is lease
  assert lease in storage._live_leases
  assert lease.resource_participant is participant
  assert lease.resource_thread is threading.current_thread()
 native_type=type(connection)
 assert native_type.__module__==private_sqlite.__name__
 assert native_type.__qualname__.endswith('.AdmittedConnection')
 assert native_type_current(native_type)
 close_function=vars(native_type)['close']
 cells=dict(zip(close_function.__code__.co_freevars,close_function.__closure__))
 assert cells['lease'].cell_contents is lease
 assert cells['capture_lease'].cell_contents is False
 assert cells['constructing'].cell_contents is False
 assert sqlite3.Connection.in_transaction.__get__(connection) is False
 return participant,lease,native_type
generic_identity={}
physical=[]
"""

_MAIN_FAILURE_CLEANUP = r"""try:
 asyncio.run(main())
except BaseException as primary:
 try:close()
 except BaseException as cleanup_error:primary.add_note('Main creator cleanup failed: '+type(cleanup_error).__name__)
 raise"""


_PHYSICAL_ORACLE = r"""
  participant,lease,native_type=physical[-1] if physical[-1] is not None else (None,None,type(connection))
  preserved=outcome in {'borrowed','memory','custom','transaction','operation'}
  if participant is not None:
   with storage._lock:
    registered=participant.connections.get(connection) is lease
    ordinary=private_sqlite._ordinary_connections.get(connection) is lease
    live=lease in storage._live_leases
   assert registered is preserved,(outcome,'core-registration',registered)
   assert ordinary is preserved,(outcome,'ordinary-registration',ordinary)
   assert live is preserved,(outcome,'live-lease',live)
  assert getattr(db._thread_local,'conn',None) is (connection if preserved else None)
  if preserved:
   assert connection.execute('SELECT 1').fetchone()[0]==1
  else:
   try:sqlite3.Connection.in_transaction.__get__(connection)
   except sqlite3.ProgrammingError as error:assert 'closed' in str(error).lower()
   else:raise AssertionError('actual native handle remains open')
  assert original_factory is private_sqlite._with_storage_admission
  assert original_factory.__code__ is factory_code
  assert original_factory.__globals__ is factory_globals
  print(json.dumps({'route':route,'outcome':outcome,'database_id':id(db),'participant_id':id(participant) if participant is not None else None,'lease_id':id(lease) if lease is not None else None,'native_id':id(connection),'native_type':native_type.__qualname__,'thread_id':threading.get_ident(),'preserved':preserved,'ordinary_registered':ordinary if participant is not None else None,'core_registered':registered if participant is not None else None,'lease_live':live if participant is not None else None}),flush=True)
"""

_BEFORE_NEGATIVE_ORACLE = r"""
  if outcome=='custom':
   generic=generic_identity[id(connection)]
   with storage._lock:
    assert private_sqlite._ordinary_connections.get(connection) is generic
    assert generic in storage._live_leases
    assert vars(generic).get('resource_participant') is None
    assert generic.resource_thread is threading.current_thread()
   assert '_maintenance_participant' not in vars(db)
   assert getattr(db._thread_local,'conn',None) is connection
   assert native_type_current(type(connection))
   assert sqlite3.Connection.in_transaction.__get__(connection) is False
   print(json.dumps({'phase':'after_original_finite_return_before_negative_oracle','route':route,'outcome':outcome,'database_id':id(db),'generic_lease_id':id(generic),'native_id':id(connection),'thread_id':threading.get_ident(),'repository_participant_present':False,'ordinary_registered':True,'generic_lease_live':True,'native_status':'open','cache_same':True,'policy':'db.base'}),flush=True)
  identity=physical[-1]
  if identity is not None:
   participant,lease,native_type=identity
   with storage._lock:
    registered=participant.connections.get(connection) is lease
    ordinary=private_sqlite._ordinary_connections.get(connection) is lease
    live=lease in storage._live_leases
    operation_count=sum(operation.participant is participant and operation.thread is threading.current_thread() for operation in storage._operations)
   try:
    transaction=sqlite3.Connection.in_transaction.__get__(connection)
    native_status='open'
   except sqlite3.ProgrammingError as error:
    transaction=None
    native_status='closed' if 'closed' in str(error).lower() else 'unknown'
   print(json.dumps({'phase':'after_original_finite_return_before_negative_oracle','route':route,'outcome':outcome,'database_id':id(db),'participant_id':id(participant),'lease_id':id(lease),'native_id':id(connection),'thread_id':threading.get_ident(),'core_registered':registered,'ordinary_registered':ordinary,'lease_live':live,'operation_count':operation_count,'native_status':native_status,'in_transaction':transaction,'cache_same':getattr(db._thread_local,'conn',None) is connection}),flush=True)
"""


def _derive_workspace_script() -> str:
    script = _LIBRARY_SCRIPT.replace("if route=='media':", _WORKSPACE_BRANCH, 1)
    script = script.replace(
        "main_connection=get()\nmain_connection.execute('SELECT 1')",
        "main_connection=None if route=='workspace_first_factory' else get()\n"
        "if main_connection is not None:main_connection.execute('SELECT 1')",
        1,
    )
    script = script.replace("observed=[]", _PHYSICAL_SETUP + "\nobserved=[]", 1)
    script = script.replace(
        " connection.execute('SELECT 1')",
        " connection.execute('SELECT 1')\n physical.append(held_identity(connection))",
        1,
    )
    script = script.replace(
        "  assert get().execute('SELECT 1').fetchone()[0]==1",
        _PHYSICAL_ORACLE + "\n  assert get().execute('SELECT 1').fetchone()[0]==1",
        1,
    )
    script = script.replace(
        "  connection=observed[-1]",
        "  connection=observed[-1]\n" + _BEFORE_NEGATIVE_ORACLE,
        1,
    )
    script = script.replace(
        "assert main_connection.execute('SELECT 1').fetchone()[0]==1",
        "if main_connection is not None:assert main_connection.execute('SELECT 1').fetchone()[0]==1",
        1,
    )
    script = script.replace("asyncio.run(main())", _MAIN_FAILURE_CLEANUP, 1)
    return script


_WORKSPACE_SCRIPT = _derive_workspace_script()

_PROBE_SCRIPT = (
    r"""
import asyncio,contextlib,sqlite3,sys,threading
from pathlib import Path
from tldw_chatbook import config
from tldw_chatbook.Chat import console_agent_bridge as bridge_source
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Tools import workspace_file_roots as roots
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Agents import run_log
from tldw_chatbook.UI.Console_Modules import run_log_probe_scope as scope
route,outcome=sys.argv[1:]
root=Path.home()
runs=AgentRunsDB(root/'runs.sqlite',client_id='finite-probe')
run_id=runs.create_run(conversation_id='conversation',agent_kind='primary')
bridge=bridge_source.ConsoleAgentBridge(agent_runs_db=runs,store=None,provider_gateway=None)
db=None
owners=[]
restores=[]
def own_workspace():
 database=WorkspaceDB(root/('custom-'+str(len(owners))+'.sqlite'),client_id='custom-callback')
 owners.append(database)
 database._held_connection().execute('SELECT 1')
 return database
original_resolve=bridge_source.ConsoleAgentBridge.resolve_run_log_target
def custom_resolve(self,conversation_id,drill_id):
 own_workspace()
 return original_resolve(self,conversation_id,drill_id)
if route=='instance_callback':bridge.resolve_run_log_target=custom_resolve.__get__(bridge)
elif route=='bridge_subclass':
 class CustomBridge(bridge_source.ConsoleAgentBridge):
  def resolve_run_log_target(self,*args):return custom_resolve(self,*args)
 bridge=CustomBridge(agent_runs_db=runs,store=None,provider_gateway=None)
elif route=='class_callback':
 bridge_source.ConsoleAgentBridge.resolve_run_log_target=custom_resolve
 restores.append(lambda:setattr(bridge_source.ConsoleAgentBridge,'resolve_run_log_target',original_resolve))
elif route=='metadata_callback':
 original_metadata=runs.get_run_metadata
 def metadata(run_id):
  own_workspace()
  return original_metadata(run_id)
 runs.get_run_metadata=metadata
elif route=='authority_callback':
 @contextlib.contextmanager
 def authority_scope():
  own_workspace()
  yield root
 bridge._remember_run_log_authority(run_id,root,session_id='custom-authority',access_scope=authority_scope)
elif route=='registry_factory':
 original_factory_callback=roots._registry_factory
 def registry_factory():return LocalWorkspaceRegistryService(own_workspace())
 roots._registry_factory=registry_factory
 restores.append(lambda:setattr(roots,'_registry_factory',original_factory_callback))
elif route=='registry_callback':
 registry=roots._default_registry_factory()
 def active():
  own_workspace()
  return None
 registry.get_active_workspace=active
 restores.append(lambda:delattr(registry,'get_active_workspace'))
elif route=='resolver_callback':
 original_locator=run_log.resolve_existing_log_dir
 def locator(*args,**kwargs):
  own_workspace()
  return None
 run_log.resolve_existing_log_dir=locator
 restores.append(lambda:setattr(run_log,'resolve_existing_log_dir',original_locator))
elif route=='stock_cached':
 registry=roots._default_registry_factory()
 main_workspace=registry.db
 main_connection=main_workspace._held_connection()
else:assert route=='stock_first'
"""
    + _PHYSICAL_SETUP
    + r"""
def read():
 global db
 target=bridge.resolve_run_log_target('conversation',None)
 assert target==run_id
 available=bridge.run_log_available(target)
 registry=roots._default_registry_instance
 if registry is not None and registry.db not in owners:
  owners.append(registry.db)
 for database in owners:
  connection=getattr(database._thread_local,'conn',None)
  if connection is None:continue
  db=database
  physical.append((database,connection,held_identity(connection)))
 assert physical,'original callback opened no Workspace handle'
 if outcome=='error':raise ValueError('original callback error')
 return available
def worker():
 stock=route in {'stock_first','stock_cached'}
 assert scope._stock_default_probe(bridge) is stock,route
 try:
  if outcome=='error':
   try:scope.finite_stock_probe(bridge,read)
   except ValueError as error:assert str(error)=='original callback error'
   else:raise AssertionError('original error swallowed')
  else:assert scope.finite_stock_probe(bridge,read) is False
  for database,connection,identity in physical:
   participant,lease,native_type=identity
   with storage._lock:
    registered=participant.connections.get(connection) is lease
    ordinary=private_sqlite._ordinary_connections.get(connection) is lease
    live=lease in storage._live_leases
   assert registered is (not stock),(route,'core registration',registered)
   assert ordinary is (not stock),(route,'ordinary registration',ordinary)
   assert live is (not stock),(route,'live lease',live)
   assert getattr(database._thread_local,'conn',None) is (None if stock else connection)
   if stock:
    try:sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError as error:assert 'closed' in str(error).lower()
    else:raise AssertionError('stock native remains open')
   else:assert connection.execute('SELECT 1').fetchone()[0]==1
   print(json.dumps({'route':route,'outcome':outcome,'stock_scope':stock,'database_id':id(database),'participant_id':id(participant),'lease_id':id(lease),'native_id':id(connection),'thread_id':threading.get_ident(),'core_registered':registered,'ordinary_registered':ordinary,'lease_live':live,'native_type':native_type.__qualname__}),flush=True)
 finally:
  for database in owners:database.close()
def retire_main_creators():
 for restore in reversed(restores):restore()
 registry=roots._default_registry_instance
 if registry is not None:registry.db.close()
 runs.close()
try:
 asyncio.run(asyncio.to_thread(worker))
 if route=='stock_cached':assert main_connection.execute('SELECT 1').fetchone()[0]==1
except BaseException as primary:
 try:retire_main_creators()
 except BaseException as cleanup_error:primary.add_note('Main creators/restore cleanup failed: '+type(cleanup_error).__name__)
 raise
else:retire_main_creators()
print('retired and reopened')
"""
)


@pytest.mark.parametrize(
    "outcome",
    ["success", "error", "borrowed", "transaction", "operation", "reopen", "cancel"],
)
def test_finite_workspace_worker_actual_native_lifetime(tmp_path, outcome):
    _run(tmp_path, "workspaces", outcome, script=_WORKSPACE_SCRIPT)


@pytest.mark.parametrize("outcome", ["memory", "custom"])
def test_finite_workspace_worker_preserves_unowned_native_lifetime(tmp_path, outcome):
    _run(tmp_path, "workspaces", outcome, script=_WORKSPACE_SCRIPT)


@pytest.mark.parametrize(
    "route", ["workspace_first_factory", "workspace_cached_factory"]
)
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_finite_workspace_default_factory_actual_native_lifetime(
    tmp_path, route, outcome
):
    _run(tmp_path, route, outcome, script=_WORKSPACE_SCRIPT)


@pytest.mark.parametrize("route", ["stock_first", "stock_cached"])
@pytest.mark.parametrize("outcome", ["success", "error"])
def test_stock_probe_default_resolver_actual_native_lifetime(tmp_path, route, outcome):
    _run(tmp_path, route, outcome, script=_PROBE_SCRIPT)


@pytest.mark.parametrize(
    "route",
    [
        "instance_callback",
        "bridge_subclass",
        "class_callback",
        "metadata_callback",
        "authority_callback",
        "registry_factory",
        "registry_callback",
        "resolver_callback",
    ],
)
def test_custom_probe_callback_keeps_new_stock_workspace_lifetime(tmp_path, route):
    _run(tmp_path, route, "success", script=_PROBE_SCRIPT)
