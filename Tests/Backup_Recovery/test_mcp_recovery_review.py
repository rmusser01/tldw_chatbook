"""Actual restored MCP authorization stays historical until fresh root review."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SETUP = r"""
import asyncio,hashlib,json,os,sys,zipfile
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install();os.umask(0o077)
from Tests.Backup_Recovery.test_archive_reader import manifest
from tldw_chatbook.Backup_Recovery import archive_reader,bootstrap
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,select_profile
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.activation import ActivationStore
base=Path.home();source=base/'source';source.mkdir(mode=0o700)
selector=Path(os.environ['TLDW_CONFIG_PATH']);selector.write_text('[general]\nusers_name="Local"\n[paths]\ndata_dir="'+str(source)+'"\n')

user=source/'Local';user.mkdir(mode=0o700)
owners_by_name={'local_mcp_store.json':'mcp.local','mcp_permissions.json':'mcp.permissions','unified_mcp_context.json':'mcp.context','mcp_server_targets.json':'mcp.targets'}
payloads={'local_mcp_store.json':{'profiles':[{'profile_id':'demo','command':'disposable-sentinel','args':[]}],'governance_rules':[{'rule_id':'old','capability_id':'notes.create.local','decision':'allow'}],'approval_requests':[{'request_id':'old-approved','action_name':'tool.execute','resolved_action_id':'notes.create.local','status':'approved'}]},'mcp_permissions.json':{'schema_version':1,'kill_switch':False,'profiles':{'default':{'global_default':'allow','servers':{}}}},'unified_mcp_context.json':{'selected_source':'server'},'mcp_server_targets.json':{'targets':[]}}
for name,value in payloads.items():(user/name).write_text(json.dumps(value))
before={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
doc=manifest();doc['owners']=[dict(owner_id=o,schema_version=1,capabilities=[]) for o in ('config','mcp.local','mcp.permissions','mcp.context','mcp.targets')]
doc['directories']=[dict(logical_id='config-root',root_id='config-root',parent_id=None,relative_path='',synthetic=True,metadata=dict(version=1,mode=448,mtime_ns=0)),dict(logical_id='data-root',root_id='data-root',parent_id=None,relative_path='',metadata=dict(version=1,mode=448,mtime_ns=0))]
parents={source:'data-root'}
for path in sorted((p for p in source.rglob('*') if p.is_dir()),key=lambda p:len(p.parts)):
 ident='dir-'+str(len(parents));parents[path]=ident
 doc['directories'].append(dict(logical_id=ident,root_id='data-root',parent_id=parents[path.parent],relative_path=path.relative_to(source).as_posix(),metadata=dict(version=1,mode=448,mtime_ns=0)))
payloads={'profile:profile:config':selector.read_bytes()};doc['files']=[]
for ident,data,root,parent,relative,owner in [('profile:profile:config',selector.read_bytes(),'config-root','config-root','config.toml','config')]+[(f'file-{i}',p.read_bytes(),'data-root',parents[p.parent],p.relative_to(source).as_posix(),owners_by_name[p.name]) for i,p in enumerate(sorted(p for p in source.rglob('*') if p.is_file()))]:
 payloads[ident]=data;doc['files'].append(dict(logical_id=ident,root_id=root,parent_id=parent,relative_path=relative,owner_id=owner,payload='payload/'+str(len(doc['files'])),size=len(data),sha256=hashlib.sha256(data).hexdigest()))
doc['dependency_groups']=[dict(group_id='main',members=list(payloads),complete=True)]
archive_path=base/'mcp.zip'
with zipfile.ZipFile(archive_path,'w') as zipped:
 zipped.writestr('manifest.json',json.dumps(doc))
 for row in doc['files']:zipped.writestr(row['payload'],payloads[row['logical_id']])
archive=archive_reader.acquire(archive_path,base/'acquired',ArchiveLimits(),None,Event())
destination=base/'restored';destination.mkdir(mode=0o700)
plan=plan_restore(archive,mode='isolated',destinations={'config-root':destination/'config','data-root':destination/'data','profile:profile:paths.data_dir':destination/'data'},target=None,profile_names={'profile':'Local'})
from tldw_chatbook.Backup_Recovery import storage_admission as storage
startup=storage._startups.pop((os.getpid(),str(bootstrap.default_bootstrap_root())),None)
if startup is not None:startup.close()
control=base/'control';profile=restore_isolated(archive,plan,control,Event());select_profile(profile,control)

from tldw_chatbook.MCP.local_store import LocalMCPStore,LocalMCPStoreState,LocalExternalMCPProfile,LocalGovernanceRule,LocalApprovalRequest
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore
from tldw_chatbook.MCP.unified_control_models import UnifiedMCPContext
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.client import MCPClient
from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from types import SimpleNamespace

user=destination/'data'/'Local';history={p.name:p.read_bytes() for p in user.iterdir() if p.is_file()}
from tldw_chatbook import config
from tldw_chatbook.MCP.activation import execution
def plane():
 local=LocalMCPControlService(store=LocalMCPStore(user/'local_mcp_store.json'),client=MCPClient(),manifest_provider=lambda:{})
 return UnifiedMCPControlPlaneService(target_store=ConfiguredServerTargetStore(user/'mcp_server_targets.json'),context_store=UnifiedMCPContextStore(user/'unified_mcp_context.json'),local_service=local,server_service=SimpleNamespace())
_,profiles,_=bootstrap._control_records(bootstrap.default_bootstrap_root());witness=next(p['activation'] for p in profiles if p['selector']==str(destination/'config'/'config.toml'))
activation=ActivationStore(Path(witness['store_root']))
assert not activation.allowed(witness['generation'],'config')
"""

_APPROVED_SETUP = (
    _SETUP
    + r"""
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
plane=plane();plane.approve_recovery_review(plane.capture_recovery_review())
assert not activation.allowed(witness['generation'],'config')
local=plane.local_service;local_store=local.store;client=local.client;delegate=local.runtime_delegate
root=bootstrap.default_bootstrap_root();authority=admission_authority(root)
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None:startup.close()
route,state=sys.argv[1:]
"""
)


@pytest.mark.parametrize("corrupt", [False, True])
def test_actual_restored_permissions_are_passive_and_owner_flag_is_insufficient(
    tmp_path, corrupt
):
    script = (
        _SETUP
        + "corrupt="
        + repr(corrupt)
        + r"""
if corrupt:
 (user/'mcp_permissions.json').write_bytes(b'{invalid retained permissions')
 (user/'mcp_permissions.json.bak').write_bytes(b'older retained evidence')
 history={p.name:p.read_bytes() for p in user.iterdir() if p.is_file()}
service=plane()
assert service.permission_store.get_global_default()=='ask'
assert service.selected_source=='local'
assert not service.local_service.store.list_governance_rules()
for owner in ('mcp.local','mcp.permissions','mcp.context','mcp.targets'):activation.approve(witness['generation'],owner)
try:
 with execution(service):raise AssertionError('owner flags reopened historical authorization')
except PermissionError:pass
assert all((user/name).read_bytes()==value for name,value in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", "passive", script=script)


def test_actual_review_creates_fresh_existing_format_policy_and_only_mcp_approval(
    tmp_path,
):
    script = (
        _SETUP
        + r"""
service=plane()
review=service.capture_recovery_review()
assert review.workspace
assert not service.local_service.store.path.parent.exists()
service.approve_recovery_review(review)
assert service.permission_store.get_global_default()=='ask'
assert not service.local_service.store.list_governance_rules()
assert not service.local_service.store.list_approval_requests()
assert [p.profile_id for p in service.local_service.store.list_profiles()]==['demo']
assert service.selected_source=='local'
assert not activation.allowed(witness['generation'],'config')
with execution(service):pass
assert all(activation.allowed(witness['generation'],o) for o in ('mcp.local','mcp.permissions','mcp.context','mcp.targets'))
assert not activation.allowed(witness['generation'],'skills')
assert all((user/name).read_bytes()==value for name,value in history.items())
service.permission_store.set_global_default('deny')
assert plane().permission_store.get_global_default()=='deny'
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", "approve", script=script)


def test_actual_fresh_permission_store_keeps_native_persistence_binding(tmp_path):
    script = (
        _SETUP
        + r"""
from tldw_chatbook.Backup_Recovery.mcp_source_participants import binding
service=plane();service.approve_recovery_review(service.capture_recovery_review())
assert all(binding(store)[2] for store in (service.permission_store,service.local_service.store,service.context_store))
service.permission_store.set_global_default('deny')
assert service.permission_store.get_global_default()=='deny'
assert all((user/name).read_bytes()==value for name,value in history.items())
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", "bound", script=script)


@pytest.mark.parametrize("change", ["definition", "workspace", "foreign_root"])
def test_actual_review_refuses_changed_sources_before_approval(tmp_path, change):
    script = (
        _SETUP
        + "change="
        + repr(change)
        + r"""
service=plane();review=service.capture_recovery_review()
fresh=service.local_service.store.path.parent
if change=='definition':(user/'local_mcp_store.json').write_text('{}')
elif change=='workspace':
 selector=bootstrap.effective_config_path()
 selector.write_text(selector.read_text()+'\n[console]\nworkspace_root="'+str(base)+'"\n')
else:
 fresh.mkdir(mode=0o700);(fresh/'foreign').write_bytes(b'foreign evidence')
try:service.approve_recovery_review(review)
except (ValueError,FileExistsError):pass
else:raise AssertionError('changed review accepted')
assert not any(activation.allowed(witness['generation'],o) for o in owners_by_name.values())
if change=='foreign_root':assert (fresh/'foreign').read_bytes()==b'foreign evidence'
else:assert not fresh.exists()
assert (user/'mcp_permissions.json').read_bytes()==history['mcp_permissions.json']
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", change, script=script)


@pytest.mark.parametrize("interruption", ["binding", "owner_flag"])
def test_actual_partial_own_setup_retries_without_historical_grants(
    tmp_path, interruption
):
    script = (
        _SETUP
        + "interruption="
        + repr(interruption)
        + r"""
from tldw_chatbook.Backup_Recovery import activation as activation_module
service=plane();review=service.capture_recovery_review()
if interruption=='binding':
 original=activation_module._write
 def stop(parent,name,record):
  original(parent,name,record)
  if name.startswith('mcp-root-'):raise InterruptedError('after durable binding')
 activation_module._write=stop
else:
 original=ActivationStore.approve
 def stop(self,generation,owner):
  original(self,generation,owner)
  raise InterruptedError('after owner flag')
 ActivationStore.approve=stop
try:service.approve_recovery_review(review)
except InterruptedError:pass
else:raise AssertionError('fault did not interrupt')
if interruption=='binding':activation_module._write=original
else:ActivationStore.approve=original
try:
 with execution(service):raise AssertionError('partial setup executed')
except PermissionError:pass
service.approve_recovery_review(service.capture_recovery_review())
with execution(service):pass
assert service.permission_store.get_global_default()=='ask'
assert not service.local_service.store.list_governance_rules()
assert all((user/name).read_bytes()==value for name,value in history.items())
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", interruption, script=script)


@pytest.mark.parametrize("damage", ["missing", "corrupt", "foreign"])
def test_actual_owner_flags_cannot_reopen_damaged_root_binding(tmp_path, damage):
    script = (
        _SETUP
        + "damage="
        + repr(damage)
        + r"""
from tldw_chatbook.MCP.recovery_activation import _record_name
service=plane();service.approve_recovery_review(service.capture_recovery_review())
record=activation._generation(witness['generation'])/_record_name('mcp.permissions',user/'mcp_permissions.json')
if damage=='missing':record.unlink()
elif damage=='corrupt':record.write_bytes(b'{')
else:
 data=json.loads(record.read_bytes());data['scope']='foreign';record.write_text(json.dumps(data))
try:
 with execution(service):raise AssertionError('damaged binding executed')
except PermissionError:pass
try:service.permission_store.set_global_default('allow')
except (ValueError,OSError):pass
else:raise AssertionError('damaged binding wrote policy')
assert all((user/name).read_bytes()==value for name,value in history.items())
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", damage, script=script)


def _rebackup_script():
    warm = r"""
import asyncio,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for module in ('sounddevice','pyaudio'):sys.modules[module]=None
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
select_profile(sys.argv[1],Path(sys.argv[2]))
from tldw_chatbook.app import TldwCli
async def warm_owners():
 app=TldwCli()
 try:
  assert Path(app.chachanotes_db.db_path).is_relative_to(Path(sys.argv[3]))
 finally:
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(warm_owners())
assert not blocked_attempts()
"""
    warmup = r"""
import subprocess
with (base/'mcp-warmup.log').open('w') as output:
 result=subprocess.run([sys.executable,'-c',WARM,profile,str(control),str(destination/'data')],stdout=output,stderr=subprocess.STDOUT,timeout=30)
assert result.returncode==0,(base/'mcp-warmup.log').read_text()[-6000:]
assert all((user/name).read_bytes()==value for name,value in history.items())
"""
    script = (
        _SETUP
        + "\nWARM="
        + repr(warm)
        + "\n"
        + warmup
        + _APPROVED_SETUP[len(_SETUP) :]
        + r"""
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture,capture
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
options={'allow_partial':True,'staging_parent':base}
plane.permission_store.set_global_default('deny')
fresh={store.path:store.path.read_bytes() for store in (plane.permission_store,local_store,plane.context_store)}
review=preview_capture((bootstrap.effective_config_path(),),options=options)
assert set(fresh)<=set(item.path for item in review.items if item.status=='included'),'fresh MCP files absent from installed inventory'
output=base/'rebackup.tldw-backup.zip'
result=capture((bootstrap.effective_config_path(),),review.scope_digest,output,options=options,cancel=Event())
sealed=write_archive(result,output,password=None,cancel=Event())
acquired=archive_reader.acquire(sealed.path,base/'rebackup-readback',ArchiveLimits(),None,Event())
assert acquired.manifest_bytes==sealed.manifest_bytes
assert json.loads(acquired.manifest_bytes)==archive_reader._manifest(result.manifest_bytes,ArchiveLimits(),False).model_dump(mode='json')
doc=json.loads(result.manifest_bytes)
saved={row['logical_id']:row for row in doc['files'] if row['owner_id'].startswith('mcp.')}
for item in review.items:
 if item.path in fresh or item.path is not None and item.path.parent==user and item.path.name in history:
  row=saved[item.logical_id]
  expected=fresh.get(item.path,history.get(item.path.name))
  assert (result.root/row['payload']).read_bytes()==expected
  with zipfile.ZipFile(acquired.path) as zipped:assert zipped.read(row['payload'])==expected
assert all(path.read_bytes()==value for path,value in fresh.items())
assert all((user/name).read_bytes()==value for name,value in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    return script


def test_actual_rebackup_retains_fresh_and_historical_mcp_files(tmp_path):
    _run(tmp_path, "mcp", "rebackup", script=_rebackup_script(), timeout=60)


def _next_restore_script():
    child = r"""
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import ActivationStore
select_profile(sys.argv[1],Path(sys.argv[2]))
from tldw_chatbook import config
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.client import MCPClient
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService
user=Path(sys.argv[3]);old=user/sys.argv[4]
history={p:p.read_bytes() for p in (*old.iterdir(),*(user/n for n in ('local_mcp_store.json','mcp_permissions.json','unified_mcp_context.json','mcp_server_targets.json'))) if p.is_file()}
service=UnifiedMCPControlPlaneService(target_store=ConfiguredServerTargetStore(user/'mcp_server_targets.json'),context_store=UnifiedMCPContextStore(user/'unified_mcp_context.json'),local_service=LocalMCPControlService(store=LocalMCPStore(user/'local_mcp_store.json'),client=MCPClient(),manifest_provider=lambda:{}),server_service=SimpleNamespace())
_,profiles,_=bootstrap._control_records(bootstrap.default_bootstrap_root())
witness=next(row['activation'] for row in profiles if row['selector']==str(bootstrap.effective_config_path()))
activation=ActivationStore(Path(witness['store_root']))
assert not any(activation.allowed(witness['generation'],owner) for owner in ('mcp.local','mcp.permissions','mcp.context','mcp.targets'))
assert service.permission_store.get_global_default()=='ask' and service.selected_source=='local'
assert service.local_service.store.path.parent!=old
manifest_path=Path(witness['store_root']).parent/('operation-'+bootstrap._key(witness['operation_id']))/'verified-manifest.json'
manifest_bytes=manifest_path.read_bytes()
manifest_path.write_bytes(b'corrupt retained manifest')
try:service.capture_recovery_review()
except ValueError as error:assert str(error)=='mcp_recovery_mapping_unverified'
else:raise AssertionError('corrupt local mapping accepted')
assert not service.local_service.store.path.exists()
assert not any(activation.allowed(witness['generation'],owner) for owner in ('mcp.local','mcp.permissions','mcp.context','mcp.targets'))
manifest_path.write_bytes(manifest_bytes)
stale=service.capture_recovery_review()
imported=old/'local_mcp_store.json';original_bytes=imported.read_bytes()
changed=json.loads(original_bytes);changed['profiles'][0]['command']='unreviewed-change'
imported.write_text(json.dumps(changed))
try:service.approve_recovery_review(stale)
except ValueError as error:assert str(error)=='mcp_recovery_review_changed'
else:raise AssertionError('changed imported definitions approved')
assert not service.local_service.store.path.exists()
imported.write_bytes(original_bytes)
service.approve_recovery_review(service.capture_recovery_review())
assert {p.profile_id for p in service.local_service.store.list_profiles()}=={'demo','post-review'},'latest imported definitions omitted'
assert not service.local_service.store.list_governance_rules() and not service.local_service.store.list_approval_requests()
assert service.permission_store.get_global_default()=='ask' and service.selected_source=='local'
assert all(path.read_bytes()==value for path,value in history.items())
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
review=preview_capture((bootstrap.effective_config_path(),),options={'allow_partial':True,'staging_parent':Path.home()})
assert set(history)<=set(item.path for item in review.items if item.status=='included')
assert not any(item.owner=='unknown' and item.path in (old,service.local_service.store.path.parent) for item in review.items)
assert not blocked_attempts()
print('NEXT_MCP_DEFINITIONS_REVIEWED',flush=True)
"""
    script = (
        _rebackup_script()
        .replace(
            "plane.permission_store.set_global_default('deny')",
            "plane.permission_store.set_global_default('deny')\nlocal_store.save_profile(LocalExternalMCPProfile(profile_id='post-review',command='edited-sentinel',args=[]))",
        )
        .replace(
            "assert (result.root/row['payload']).read_bytes()==expected\n  with zipfile.ZipFile(acquired.path) as zipped:assert zipped.read(row['payload'])==expected",
            "captured=(result.root/row['payload']).read_bytes()\n  if item.path==local_store.path:assert json.loads(captured)==json.loads(expected)\n  else:assert captured==expected\n  with zipfile.ZipFile(acquired.path) as zipped:assert zipped.read(row['payload'])==captured",
        )
        + "\nNEXT="
        + repr(child)
        + "\n"
        + r"""
document=archive_reader.verify_sealed(acquired)
assert len(document.profile_ids)==1
profile_key=document.profile_ids[0];next_root=base/'next';next_root.mkdir(mode=0o700)
next_user=next_root/'data'/'Local';mapping={}
ordinary={'chat.attachments','db.chachanotes.primary','db.evals','db.library_collections','db.media.primary','db.prompts.primary','db.scheduled_tasks','db.subscriptions','db.workspaces','kanban.local','mcp.local','mcp.permissions','mcp.context','mcp.targets','notes.sync_bindings','notifications.client','quiz.local','research.local','runtime.event_state','runtime.sync_state','study.local','writing.local'}
for row in document.directories:
 if row.parent_id is not None:continue
 if not row.synthetic:
  assert row.logical_id.endswith(':chat.dictionaries')
  mapping[row.logical_id]=next_user/'chat_dicts'
  continue
 members=[f for f in document.files if f.root_id==row.logical_id]
 assert len(members)==1
 member=members[0]
 if member.owner_id in ('config','runtime.source_state'):target=next_root/'config'
 elif member.owner_id=='eval.definitions':target=next_root/'retained-eval'
 else:
  assert member.owner_id in ordinary
  target=next_user
  if ':fresh' in member.logical_id:target=next_user/local_store.path.parent.name
 mapping[row.logical_id]=target
mapping['profile:'+profile_key+':paths.data_dir']=next_root/'data'
next_plan=plan_restore(acquired,mode='isolated',destinations=mapping,target=None,profile_names={profile_key:'Local'})
next_control=base/'next-control'
next_profile=restore_isolated(acquired,next_plan,next_control,Event())
with (base/'mcp-next-generation.log').open('w') as output:
 completed=subprocess.run([sys.executable,'-c',NEXT,next_profile,str(next_control),str(next_user),local_store.path.parent.name],stdout=output,stderr=subprocess.STDOUT,timeout=30)
assert completed.returncode==0,(base/'mcp-next-generation.log').read_text()[-8000:]
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    return script


def test_actual_next_restore_reviews_edited_fresh_definitions_and_retains_history(
    tmp_path,
):
    _run(tmp_path, "mcp", "next-generation", script=_next_restore_script(), timeout=60)


def test_actual_captured_retained_mcp_destination_rejects_foreign_mapping(tmp_path):
    script = (
        _next_restore_script().split("next_profile=restore_isolated(")[0]
        + r"""
from dataclasses import replace
from tldw_chatbook.MCP.recovery import recovery_adapters
owner=next(row for row in recovery_adapters() if row.owner_id=='mcp.context')
payload=next(row for row in document.files if row.logical_id.endswith(':mcp.context:fresh'))
canonical=next_user/owner.leaf
config_target=next_root/'config'/'config.toml'
owner.validate_retained_destination(profile_key,payload,document,next_plan,canonical,config_target)
for damage in ('owner','profile','leaf','alias','topology','family'):
 row=payload;doc=document;plan=next_plan
 if damage=='owner':
  producer=tuple(p.model_copy(update={'owner_id':'mcp.local'}) if p.logical_id==row.logical_id else p for p in doc.producer_inventory)
  doc=doc.model_copy(update={'producer_inventory':producer})
 elif damage=='profile':row=row.model_copy(update={'logical_id':'profile:foreign:mcp.context:fresh'})
 elif damage=='family':row=row.model_copy(update={'logical_id':row.logical_id+'-foreign'})
 elif damage=='leaf':row=row.model_copy(update={'relative_path':'foreign.json'})
 else:
  target=canonical if damage=='alias' else dict(plan.restore)[row.logical_id].parent/'nested'/owner.leaf
  plan=replace(plan,restore=tuple((key,target if key==row.logical_id else path) for key,path in plan.restore))
  if damage=='alias':plan=replace(plan,destinations=tuple((key,target.parent if key==row.root_id else path) for key,path in plan.destinations))
 try:owner.validate_retained_destination(profile_key,row,doc,plan,canonical,config_target)
 except ValueError as error:assert str(error)=='owner_relocation_unverified:mcp.context'
 else:raise AssertionError('accepted foreign retained mapping:'+damage)
assert not next_control.exists() and not next_user.exists()
assert all((user/name).read_bytes()==value for name,value in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", "retained-mapping-negatives", script=script, timeout=60)


@pytest.mark.parametrize("damage", ["none", "changed", "foreign"])
def test_actual_review_retries_interruption_after_fresh_files_before_bindings(
    tmp_path, damage
):
    script = (
        _SETUP
        + "damage="
        + repr(damage)
        + r"""
from tldw_chatbook.Backup_Recovery import native_files
service=plane();review=service.capture_recovery_review()
fresh=service.local_service.store.path.parent
flush=native_files._flush_private_tree
def interrupted(parent,*args):
 result=flush(parent,*args)
 if (os.fstat(parent).st_dev,os.fstat(parent).st_ino)==(fresh.stat().st_dev,fresh.stat().st_ino):
  raise InterruptedError('after actual fresh-file flush')
 return result
native_files._flush_private_tree=interrupted
try:service.approve_recovery_review(review)
except InterruptedError:pass
else:raise AssertionError('fresh-file interruption absent')
finally:native_files._flush_private_tree=flush
assert set(p.name for p in fresh.iterdir())=={'local_mcp_store.json','mcp_permissions.json','unified_mcp_context.json'}
assert not list(activation._generation(witness['generation']).glob('mcp-root-*'))
assert not any(activation.allowed(witness['generation'],owner) for owner in owners_by_name.values())
assert service.permission_store.get_global_default()=='ask' and service.selected_source=='local'
if damage!='none':
 target=fresh/('local_mcp_store.json' if damage=='changed' else 'foreign.json')
 target.write_bytes(b'preserved foreign or changed evidence')
 try:service.approve_recovery_review(service.capture_recovery_review())
 except ValueError as error:assert str(error)=='mcp_recovery_fresh_policy_changed'
 else:raise AssertionError('changed partial setup accepted')
 assert target.read_bytes()==b'preserved foreign or changed evidence'
 assert not list(activation._generation(witness['generation']).glob('mcp-root-*'))
 assert not any(activation.allowed(witness['generation'],owner) for owner in owners_by_name.values())
 assert all((user/name).read_bytes()==value for name,value in history.items())
 assert not blocked_attempts()
 print('retired and reopened');sys.exit(0)
service.approve_recovery_review(service.capture_recovery_review())
assert {p.profile_id for p in service.local_service.store.list_profiles()}=={'demo'}
assert service.permission_store.get_global_default()=='ask'
assert all((user/name).read_bytes()==value for name,value in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", "prebinding-interruption", script=script)


@pytest.mark.parametrize("damage", ["foreign_file", "link", "directory", "binding"])
def test_actual_rebackup_refuses_foreign_container_or_binding(tmp_path, damage):
    script = (
        _APPROVED_SETUP
        + "damage="
        + repr(damage)
        + r"""
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
fresh=local_store.path.parent
if damage=='foreign_file':(fresh/'foreign.json').write_bytes(b'foreign retained bytes')
elif damage=='link':(fresh/'mcp_permissions.json.bak').symlink_to(user/'mcp_permissions.json')
elif damage=='directory':(fresh/'mcp_permissions.json.bak').mkdir()
else:
 from tldw_chatbook.MCP.recovery_activation import _record_name
 record=activation._generation(witness['generation'])/_record_name('mcp.local',user/'local_mcp_store.json')
 record.write_text('{malformed binding')
before={p.name:p.lstat() for p in fresh.iterdir()}
review=preview_capture((bootstrap.effective_config_path(),),options={'allow_partial':True,'staging_parent':base})
assert not review.complete and 'config_discovery_failure' in review.issues
assert before=={p.name:p.lstat() for p in fresh.iterdir()}
assert all((user/name).read_bytes()==value for name,value in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", "container-" + damage, script=script)


def test_actual_inactive_inspection_is_read_only_and_mixed_batch_stays_denied(tmp_path):
    script = (
        _SETUP
        + r"""
service=plane();fresh=service.local_service.store.path.parent
async def check():
 result=await service.run_action('runtime.request',{'method':'tools/list'})
 assert result['result']=={'tools':[]}
 result=await service.run_action('runtime.batch',{'requests':[{'method':'tools/list'}]})
 assert result['results'][0]['ok']
 try:await service.run_action('runtime.batch',{'requests':[{'method':'tools/list'},{'method':'resources/read','params':{'uri':'note://x'}}]})
 except PermissionError:pass
 else:raise AssertionError('mixed effect bypassed owner review')
asyncio.run(check())
assert not fresh.exists()
assert all((user/name).read_bytes()==value for name,value in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", "inspection", script=script)


def test_actual_unbound_fresh_root_never_selects_foreign_context(tmp_path):
    script = (
        _SETUP
        + r"""
service=plane();fresh=service.local_service.store.path.parent
fresh.mkdir(mode=0o700)
(fresh/'unified_mcp_context.json').write_text(json.dumps({'selected_source':'server'}))
(fresh/'mcp_permissions.json').write_text(json.dumps({'schema_version':1,'profiles':{'default':{'global_default':'allow'}}}))
try:
 service=plane()
 assert service.selected_source=='local','foreign unbound context selected remote mode'
 assert service.permission_store.get_global_default()=='ask'
except PermissionError:pass
assert (fresh/'unified_mcp_context.json').exists()
assert all((user/name).read_bytes()==value for name,value in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", "foreign-display", script=script)


def test_actual_restored_target_bootstrap_and_direct_save_preserve_history(tmp_path):
    script = (
        _SETUP
        + r"""
service=plane();targets=service.target_store
legacy={'tldw_api':{'base_url':'http://127.0.0.1:8000','auth_mode':'api_key'}}
assert targets.list_targets()==[]
assert targets.bootstrap_from_legacy_config(legacy) is False
assert targets.upsert_legacy_config_target(legacy) is None
try:targets.save_targets([])
except PermissionError:pass
else:raise AssertionError('unreviewed target direct save accepted')
assert all((user/name).read_bytes()==value for name,value in history.items())
service.approve_recovery_review(service.capture_recovery_review())
targets.save_targets([])
assert targets.list_targets()==[]
assert not activation.allowed(witness['generation'],'skills')
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", "target-bootstrap", script=script)


@pytest.mark.parametrize("damage", ["intact", "foreign"])
def test_actual_shared_witnesses_retry_only_their_partial_bindings(tmp_path, damage):
    script = (
        _SETUP.replace(
            "profile=restore_isolated(archive,plan,control,Event());select_profile(profile,control)",
            r"""profile=restore_isolated(archive,plan,control,Event())
second=base/'second';second.mkdir(mode=0o700)
second_plan=plan_restore(archive,mode='isolated',destinations={'config-root':second/'config','data-root':second/'data','profile:profile:paths.data_dir':second/'data'},target=None,profile_names={'profile':'Local'})
second_profile=restore_isolated(archive,second_plan,base/'second-control',Event())
from tldw_chatbook.Backup_Recovery.control_records import admission_authority,bind_profile
root=bootstrap.default_bootstrap_root();authority=admission_authority(root)
_,published,_=bootstrap._control_records(root)
names=tuple(sorted({n for row in published for n in row['namespaces']}))
authority.register('shared-mcp-client',(selector.parent,source))
bind_profile(root,selector,('shared-mcp-client',*names),root/'admission')
""",
        )
        + "damage="
        + repr(damage)
        + r"""
from tldw_chatbook.Backup_Recovery import activation as activation_module
from tldw_chatbook.MCP.recovery_activation import observed,_record_name
service=plane()
with observed(user/'local_mcp_store.json') as witnesses:assert len(witnesses)==2
assert all(not ActivationStore(Path(item['store_root'])).allowed(item['generation'],'config') for item in witnesses)
review=service.capture_recovery_review();original=activation_module._write;written=[]
def stop(parent,name,record):
 if name.startswith('mcp-root-'):
  written.append((name,record))
  if len(written)==2:raise InterruptedError('second binding')
 return original(parent,name,record)
activation_module._write=stop
try:service.approve_recovery_review(review)
except InterruptedError:pass
else:raise AssertionError('second binding did not interrupt')
activation_module._write=original
assert len(written)==2
assert all(not ActivationStore(Path(item['store_root'])).allowed(item['generation'],'mcp.local') for item in witnesses)
if damage=='foreign':
 for item in witnesses:
  path=ActivationStore(Path(item['store_root']))._generation(item['generation'])/written[0][0]
  if path.exists():
   value=json.loads(path.read_bytes());value['scope']='foreign';path.write_text(json.dumps(value))
 try:service.approve_recovery_review(service.capture_recovery_review())
 except ValueError:pass
 else:raise AssertionError('foreign binding resumed')
else:
 service.approve_recovery_review(service.capture_recovery_review())
 with execution(service):pass
 assert service.permission_store.get_global_default()=='ask'
assert all((user/name).read_bytes()==value for name,value in history.items())
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", damage, script=script)


@pytest.mark.parametrize("route", ["nested", "worker_cancel"])
def test_actual_review_retains_accepted_native_lifetime(tmp_path, route):
    from Tests.Backup_Recovery.test_activation_mcp import _RETENTION

    setup = (
        _SETUP
        + r"""
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
plane=plane();plane.approve_recovery_review(plane.capture_recovery_review())
local=plane.local_service;local_store=local.store;client=local.client;delegate=local.runtime_delegate
root=bootstrap.default_bootstrap_root();authority=admission_authority(root)
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None:startup.close()
route=sys.argv[1]
"""
    )
    body = _RETENTION[_RETENTION.index("import threading\n") :].replace(
        "('profile',)", "tuple(witness['namespaces'])"
    )
    _run(tmp_path, route, "fresh", script=setup + body)
