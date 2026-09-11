"""Actual restored Skills bytes require a fresh local trust root and review."""

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
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import SkillTrustStore,FileSkillTrustGenerationMarkerStore
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
base=Path.home();source=base/'source';source.mkdir(mode=0o700)
selector=Path(os.environ['TLDW_CONFIG_PATH']);selector.write_text('[general]\nusers_name="Local"\n[paths]\ndata_dir="'+str(source)+'"\n')
store_root=source/'Local'/'skills';old=store_root/'trust'
def service(root):
 trust_root=root/'trust'
 return SkillTrustService(skills_dir=root/'skills',trust_store=SkillTrustStore(trust_root,FileSkillTrustGenerationMarkerStore(trust_root/'generation_marker.json',store_dir=trust_root)))
trust=service(store_root);trust.unlock_with_passphrase('old-passphrase',salt=b'6'*32)
local=LocalSkillsService(store_dir=store_root,trust_service=trust)
asyncio.run(local.create_skill(name='demo',content='---\nname: demo\ndescription: test\n---\n# Demo\n',supporting_files={'scripts/demo.py':"print('real-reviewed-script')\n"}))
trust.bootstrap_trust();trust.grant_script_execution('demo')
before={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
doc=manifest();doc['owners']=[dict(owner_id=o,schema_version=1,capabilities=[]) for o in ('config','skills')]
doc['directories']=[dict(logical_id='config-root',root_id='config-root',parent_id=None,relative_path='',synthetic=True,metadata=dict(version=1,mode=448,mtime_ns=0)),dict(logical_id='data-root',root_id='data-root',parent_id=None,relative_path='',metadata=dict(version=1,mode=448,mtime_ns=0))]
parents={source:'data-root'}
for path in sorted((p for p in source.rglob('*') if p.is_dir()),key=lambda p:len(p.parts)):
 ident='dir-'+str(len(parents));parents[path]=ident
 doc['directories'].append(dict(logical_id=ident,root_id='data-root',parent_id=parents[path.parent],relative_path=path.relative_to(source).as_posix(),metadata=dict(version=1,mode=448,mtime_ns=0)))
payloads={'profile:profile:config':selector.read_bytes()};doc['files']=[]
for ident,data,root,parent,relative,owner in [('profile:profile:config',selector.read_bytes(),'config-root','config-root','config.toml','config')]+[(f'file-{i}',p.read_bytes(),'data-root',parents[p.parent],p.relative_to(source).as_posix(),'skills') for i,p in enumerate(sorted(p for p in source.rglob('*') if p.is_file()))]:
 payloads[ident]=data;doc['files'].append(dict(logical_id=ident,root_id=root,parent_id=parent,relative_path=relative,owner_id=owner,payload='payload/'+str(len(doc['files'])),size=len(data),sha256=hashlib.sha256(data).hexdigest()))
doc['dependency_groups']=[dict(group_id='main',members=list(payloads),complete=True)]
archive_path=base/'skills.zip'
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
restored=destination/'data'/'Local'/'skills';historical=restored/'trust'
history={p.relative_to(historical).as_posix():p.read_bytes() for p in historical.rglob('*') if p.is_file()}
trust=service(restored);local=LocalSkillsService(store_dir=restored,trust_service=trust)
_,profiles,_=bootstrap._control_records(bootstrap.default_bootstrap_root());witness=next(p['activation'] for p in profiles if p['selector']==str(destination/'config'/'config.toml'))
activation=ActivationStore(Path(witness['store_root']));activation.approve(witness['generation'],'config')
"""

_MANUAL = (
    _SETUP
    + r"""
activation.approve(witness['generation'],'skills')
assert not trust.script_execution_granted('demo'),'imported grant became live with owner flag'
assert history=={p.relative_to(historical).as_posix():p.read_bytes() for p in historical.rglob('*') if p.is_file()}
assert before=={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_actual_archived_script_grant_does_not_supply_fresh_root_approval(tmp_path):
    _run(tmp_path, "skills", "manual", script=_MANUAL)


_REVIEW = (
    _SETUP
    + r"""
review=trust.capture_recovery_review()
assert tuple(item.skill_name for item in review.skills)==('demo',)
assert review.historical_script_grants==('demo',)
assert not trust.trust_store.store_dir.exists()
if sys.argv[2]=='changed':
 (restored/'skills'/'demo'/'SKILL.md').write_text('changed after review')
 try:trust.trust_reviewed_recovery(review,'fresh-passphrase')
 except ValueError:pass
 else:raise AssertionError('stale root review was accepted')
 assert not trust.trust_store.store_dir.exists()
else:
 trust.trust_reviewed_recovery(review,'fresh-passphrase')
 assert trust.trust_store.store_dir.parent==historical
 assert trust.trust_store.store_dir.name=='recovery-'+witness['generation']
 assert trust.reduced_rollback_protection and trust.key_cache is None
 assert activation.allowed(witness['generation'],'skills')
 assert not activation.allowed(witness['generation'],'mcp.local')
 trust.ensure_skill_trusted('demo')
 assert not trust.script_execution_granted('demo')
 trust.grant_script_execution('demo')
 assert trust.script_execution_granted('demo')
assert all((historical/name).read_bytes()==data for name,data in history.items())
assert before=={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_actual_fresh_trust_root_preserves_imported_permissions(tmp_path):
    _run(tmp_path, "skills", "approve", script=_REVIEW)


def test_changed_root_review_cannot_create_live_trust(tmp_path):
    _run(tmp_path, "skills", "changed", script=_REVIEW)


def _retention_script():
    from Tests.Backup_Recovery.test_activation_skills import _RETENTION

    body = _RETENTION[_RETENTION.index("import threading\n") :]
    body = body.replace("('profile',)", "tuple(witness['namespaces'])")
    body = body.replace("real-script-effect", "real-reviewed-script")
    return (
        _SETUP
        + r"""
from tldw_chatbook.Backup_Recovery.control_records import admission_authority,register_pending
from tldw_chatbook.Backup_Recovery.activation import bind_activation
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
trust.trust_reviewed_recovery(trust.capture_recovery_review(),'fresh-passphrase')
trust.grant_script_execution('demo')
root=bootstrap.default_bootstrap_root();authority=admission_authority(root)
selector=destination/'config'/'config.toml';owners=tuple(witness['owners']);route=sys.argv[1]
# The real script owner lazily reads config. Settle that ordinary startup
# before testing the narrower accepted worker lifetime.
from tldw_chatbook import config
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None:startup.close()
"""
        + body
    )


def test_fresh_root_keeps_nested_accepted_native_admission(tmp_path):
    _run(tmp_path, "nested", "fresh", script=_retention_script())


_INVENTORY = _REVIEW.replace(
    "assert all((historical/name)",
    r"""
import tomllib
from tldw_chatbook.Skills_Interop.recovery import recovery_adapters
config_doc=tomllib.loads((destination/'config'/'config.toml').read_text())
from tldw_chatbook.Backup_Recovery.models import DiscoveryContext,DISCOVERY_CONTEXT_KEY
config_doc[DISCOVERY_CONTEXT_KEY]=DiscoveryContext(destination/'config'/'config.toml',profile)
items=recovery_adapters()[0].discover(config_doc)
included={item.path for item in items if item.status=='included'}
assert {p for p in trust.trust_store.store_dir.rglob('*') if p.is_file()}<=included
assert {historical/name for name in history}<=included
assert not any(item.status in ('unsupported','missing_required','unavailable') for item in items),items
assert all((historical/name)""",
)


def test_existing_skills_inventory_retains_new_and_historical_trust_bytes(tmp_path):
    _run(tmp_path, "skills", "inventory", script=_INVENTORY)


_REOPEN = r"""
import sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
profile,control=sys.argv[1:];control=Path(control);select_profile(profile,control)
_,data=ProfileCatalog(control).resolve(profile);root=data/'Local'/'skills';old=root/'trust'
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import SkillTrustStore,FileSkillTrustGenerationMarkerStore
class ForbiddenCache:
 def load_keys(self,**kwargs):raise AssertionError('old cache probed')
 def clear(self):raise AssertionError('old cache cleared')
trust=SkillTrustService(skills_dir=root/'skills',trust_store=SkillTrustStore(old,FileSkillTrustGenerationMarkerStore(old/'generation_marker.json',old)),key_cache=ForbiddenCache())
assert trust.key_cache is None
assert trust.trust_store.store_dir!=old
assert trust.trust_posture()=='locked'
trust.unlock_with_passphrase('fresh-passphrase');trust.ensure_skill_trusted('demo')
assert trust.script_execution_granted('demo')
assert not blocked_attempts()
print('retired and reopened')
"""


def test_fresh_process_uses_only_new_trust_root_and_explicit_grants(tmp_path):
    script = (
        _REVIEW
        + "\nimport subprocess\nchild=subprocess.run([sys.executable,'-c',"
        + repr(_REOPEN)
        + ",profile,str(control)],capture_output=True,text=True,timeout=20)\nassert child.returncode==0,child.stderr\n"
    )
    _run(tmp_path, "skills", "approve", script=script)


@pytest.mark.parametrize("action", ["passive", "setup", "changed-parent"])
def test_absent_restored_skills_store_construction_stays_passive(tmp_path, action):
    from Tests.Backup_Recovery.test_notes_recovery_review import _SETUP as isolated

    script = (
        isolated.split("from tldw_chatbook.DB.")[0]
        + r"""
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import SkillTrustStore,FileSkillTrustGenerationMarkerStore
root=data/'Local'/'skills';old=root/'trust'
class ForbiddenCache:
 def load_keys(self,**kwargs):raise AssertionError('old cache probe')
trust=SkillTrustService(skills_dir=root/'skills',trust_store=SkillTrustStore(old,FileSkillTrustGenerationMarkerStore(old/'generation_marker.json',old)),key_cache=ForbiddenCache())
assert not root.exists()
assert trust.trust_posture()=='locked'
assert trust.key_cache is None
assert not root.exists()
if sys.argv[2]!='passive':
 review=trust.capture_recovery_review()
 if sys.argv[2]=='changed-parent':
  (data/'Local').mkdir(mode=0o700)
  try:trust.trust_reviewed_recovery(review,'fresh-passphrase')
  except ValueError:pass
  else:raise AssertionError('changed reviewed absent parent accepted')
  assert not old.exists()
 else:
  trust.trust_reviewed_recovery(review,'fresh-passphrase')
  assert trust.trust_store.store_dir.is_dir()
  trust.unlock_with_passphrase('fresh-passphrase')
  assert trust._load_valid_manifest()['skills']=={}
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "skills", action, script=script)


_APP = (
    _SETUP
    + r"""
import tldw_chatbook.app as app_module
calls=[]
def forbidden(*args,**kwargs):
 calls.append('old keyring factory');raise AssertionError('inactive Skills probed keyring backend')
app_module.build_skill_trust_marker_store_with_fallback=forbidden
app_module.build_default_skill_trust_key_cache=forbidden
async def check():
 app=app_module.TldwCli()
 try:
  current=app.local_skill_trust_service
  assert current.key_cache is None and current.reduced_rollback_protection
  assert current.trust_store.store_dir.parent==historical
  assert not current.trust_store.store_dir.exists()
  assert current.trust_posture()=='locked'
  assert not calls
  assert all((historical/name).read_bytes()==data for name,data in history.items())
 finally:await app._close_owned_tts_resources()
asyncio.run(check())
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_actual_app_uses_passive_recovery_root_without_keyring_factories(tmp_path):
    _run(tmp_path, "skills", "app", script=_APP, timeout=60)


@pytest.mark.parametrize("action", ["revoke", "keyring"])
def test_preexisting_service_cannot_mutate_imported_permissions(tmp_path, action):
    from Tests.Backup_Recovery.test_activation_skills import _SCRIPT

    script = (
        _SCRIPT.split("probes=[]")[0]
        + r"""
path=trust_root/'skill_script_grants.json';before=path.read_bytes()
try:
 if sys.argv[1]=='revoke':trust.revoke_script_execution('demo')
 else:trust.enable_keyring_convenience()
except ValueError as exc:assert str(exc)=='skills_recovery_root_review_required'
else:raise AssertionError('historical grant sidecar mutated')
assert path.read_bytes()==before
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, action, "inactive", script=script)


_FAULT = (
    _SETUP
    + r"""
from tldw_chatbook.Backup_Recovery import activation as activation_module
review=trust.capture_recovery_review()
phase=sys.argv[2]
if phase=='owner':
 original=ActivationStore.approve
 def fail(self,*args):raise RuntimeError('owner interruption')
 ActivationStore.approve=fail
else:
 original=activation_module._write
 def fail(*args):raise RuntimeError('binding interruption')
 activation_module._write=fail
try:trust.trust_reviewed_recovery(review,'fresh-passphrase')
except RuntimeError:pass
else:raise AssertionError('fault did not occur')
assert not activation.allowed(witness['generation'],'skills')
assert not trust.script_execution_granted('demo')
created={p.relative_to(historical).as_posix():p.read_bytes() for p in historical.rglob('*') if p.is_file()}
if phase=='owner':
 ActivationStore.approve=original
 trust.trust_reviewed_recovery(review,'fresh-passphrase')
 trust.ensure_skill_trusted('demo')
 assert not trust.script_execution_granted('demo')
else:
 activation_module._write=original
 try:trust.trust_reviewed_recovery(review,'fresh-passphrase')
 except FileExistsError:pass
 else:raise AssertionError('unbound interrupted root reused')
assert created=={p.relative_to(historical).as_posix():p.read_bytes() for p in historical.rglob('*') if p.is_file()}
assert all((historical/name).read_bytes()==data for name,data in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("phase", ["binding", "owner"])
def test_actual_fresh_root_interruption_preserves_evidence(tmp_path, phase):
    _run(tmp_path, "skills", phase, script=_FAULT)


@pytest.mark.parametrize("fault", ["missing", "corrupt", "replaced"])
def test_local_binding_or_root_damage_cannot_reuse_approval(tmp_path, fault):
    script = (
        _REVIEW
        + r"""
from tldw_chatbook.Skills_Interop.recovery_activation import _selection,_name
binding,_=_selection(trust,(witness,))
record=activation._generation(witness['generation'])/_name(binding)
if sys.argv[2]=='missing':record.unlink()
elif sys.argv[2]=='corrupt':record.write_bytes(b'{')
else:
 root=trust.trust_store.store_dir
 root.rename(root.with_name(root.name+'-retained'))
 root.mkdir(mode=0o700)
assert not trust.script_execution_granted('demo')
assert all((historical/name).read_bytes()==data for name,data in history.items())
"""
    )
    _run(tmp_path, "skills", fault, script=script)


@pytest.mark.parametrize("fault", ["link", "unreadable"])
def test_incomplete_supported_bundle_cannot_get_positive_review(tmp_path, fault):
    script = (
        _SETUP
        + r"""
path=restored/'skills'/'demo'/'scripts'/'demo.py'
if sys.argv[2]=='link':
 path.unlink();path.symlink_to(source/'Local'/'skills'/'skills'/'demo'/'scripts'/'demo.py')
else:path.chmod(0)
try:trust.capture_recovery_review()
except ValueError:pass
else:raise AssertionError('incomplete bundle got positive review')
assert not trust.trust_store.store_dir.exists()
assert all((historical/name).read_bytes()==data for name,data in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "skills", fault, script=script)


def test_actual_fresh_setup_never_probes_old_keyring_accounts(tmp_path):
    script = (
        _SETUP
        + r"""
from Tests.Skills.test_skill_trust_service import FakeSecureKeyring
from tldw_chatbook.Skills_Interop.skill_trust_store import KeyringSkillTrustGenerationMarkerStore,KeyringSkillTrustKeyCache
backend=FakeSecureKeyring();backend.values={('old-service','old-account'):'retained historical keys'}
marker=KeyringSkillTrustGenerationMarkerStore(keyring_backend=backend,account_scope='old-account')
cache=KeyringSkillTrustKeyCache(keyring_backend=backend,account_scope='old-account')
calls=[]
def forbidden(*args,**kwargs):calls.append('old backend');raise AssertionError('old backend touched')
backend.get_password=forbidden;backend.set_password=forbidden;backend.delete_password=forbidden
trust=SkillTrustService(skills_dir=restored/'skills',trust_store=SkillTrustStore(historical,marker),key_cache=cache)
assert trust.trust_posture()=='locked'
trust.trust_reviewed_recovery(trust.capture_recovery_review(),'fresh-passphrase')
trust.ensure_skill_trusted('demo')
assert not trust.script_execution_granted('demo')
assert not calls
assert backend.values=={('old-service','old-account'):'retained historical keys'}
assert all((historical/name).read_bytes()==data for name,data in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "skills", "keyring", script=script)


def test_fresh_process_skills_selection_has_no_runtime_owner_imports(tmp_path):
    guard = r"""
import builtins
original_import=builtins.__import__
forbidden=('tldw_chatbook.config','tldw_chatbook.app','tldw_chatbook.RAG_Search','chromadb','torch','transformers','sentence_transformers')
def passive_import(name,globals=None,locals=None,fromlist=(),level=0):
 if level:
  import importlib.util
  name=importlib.util.resolve_name('.'*level+name,(globals or {}).get('__package__',''))
 attempted=(name,)+tuple(name+'.'+part for part in (fromlist or ()))
 if any(candidate==prefix or candidate.startswith(prefix+'.') for candidate in attempted for prefix in forbidden):
  raise AssertionError('runtime owner import attempted: '+name)
 return original_import(name,globals,locals,fromlist,0)
builtins.__import__=passive_import
"""
    child = _REOPEN.replace(
        "from tldw_chatbook.Skills_Interop.skill_trust_service",
        guard + "\nfrom tldw_chatbook.Skills_Interop.skill_trust_service",
    )
    script = (
        _REVIEW
        + "\nimport subprocess\nchild=subprocess.run([sys.executable,'-c',"
        + repr(child)
        + ",profile,str(control)],capture_output=True,text=True,timeout=20)\nassert child.returncode==0,child.stderr\n"
    )
    _run(tmp_path, "skills", "approve", script=script)


def test_service_from_prior_generation_cannot_bootstrap_its_historical_store(tmp_path):
    script = _retention_script().replace(
        "    assert not trust.script_execution_granted('demo')\nelif route == 'pid':",
        r"""    assert not trust.script_execution_granted('demo')
    retained={p.relative_to(historical).as_posix():p.read_bytes() for p in historical.rglob('*') if p.is_file()}
    try:trust.trust_reviewed_recovery(trust.capture_recovery_review(),'different-passphrase')
    except ValueError as exc:assert str(exc)=='skills_recovery_root_changed'
    else:raise AssertionError('stale service wrote its prior generation trust root')
    assert retained=={p.relative_to(historical).as_posix():p.read_bytes() for p in historical.rglob('*') if p.is_file()}
elif route == 'pid':""",
    )
    _run(tmp_path, "generation", "fresh", script=script)


_SHARED_SETUP = (
    _SETUP.replace(
        "control=base/'control';profile=restore_isolated(archive,plan,control,Event());select_profile(profile,control)",
        "control=base/'control';profile=restore_isolated(archive,plan,control,Event())\n"
        + "second=base/'second';second.mkdir(mode=0o700)\nsecond_plan=plan_restore(archive,mode='isolated',destinations={'config-root':second/'config','data-root':second/'data','profile:profile:paths.data_dir':second/'data'},target=None,profile_names={'profile':'Local'})\nsecond_control=base/'second-control'\nsecond_profile=restore_isolated(archive,second_plan,second_control,Event())\n"
        + r"""
from tldw_chatbook.Backup_Recovery.control_records import admission_authority,bind_profile
root=bootstrap.default_bootstrap_root();authority=admission_authority(root)
_,published,_=bootstrap._control_records(root)
names=tuple(sorted({n for row in published for n in row['namespaces']}))
authority.register('shared-skills-client',(selector,))
bind_profile(root,selector,('shared-skills-client',*names),root/'admission')
""",
    )
    + r"""
# The selected bundles and independently located trust root belong to two
# actually published isolated profiles. Existing bind_profile enrolls their
# exact native namespace union for this ordinary local shared-source client.
second_history=second/'data'/'Local'/'skills'/'trust'
second_before={p.relative_to(second_history).as_posix():p.read_bytes() for p in second_history.rglob('*') if p.is_file()}
trust=SkillTrustService(skills_dir=restored/'skills',trust_store=SkillTrustStore(second_history,FileSkillTrustGenerationMarkerStore(second_history/'generation_marker.json',second_history)))
from tldw_chatbook.Skills_Interop.recovery_activation import observed,_sources
with observed(_sources(trust)) as witnesses:assert len(witnesses)==2,witnesses
for item in witnesses:ActivationStore(Path(item['store_root'])).approve(item['generation'],'config')
"""
)


@pytest.mark.parametrize("damage", ["intact", "foreign"])
def test_two_actual_witnesses_resume_interrupted_root_binding(tmp_path, damage):
    script = (
        _SHARED_SETUP
        + r"""
from tldw_chatbook.Backup_Recovery import activation as activation_module
review=trust.capture_recovery_review();original=activation_module._write;writes=[]
def fail_second(parent,name,value):
 if name.startswith('skills-root-'):
  writes.append(name)
  if len(writes)==2:raise RuntimeError('second root binding interrupted')
 return original(parent,name,value)
activation_module._write=fail_second
try:trust.trust_reviewed_recovery(review,'fresh-passphrase')
except RuntimeError:pass
else:raise AssertionError('second binding fault did not occur')
assert len(writes)==2
assert all(not ActivationStore(Path(item['store_root'])).allowed(item['generation'],'skills') for item in witnesses)
assert not trust.script_execution_granted('demo')
activation_module._write=original
if sys.argv[2]=='foreign':
 from tldw_chatbook.Skills_Interop.recovery_activation import _name
 item=witnesses[0];record=ActivationStore(Path(item['store_root']))._generation(item['generation'])/_name(review.binding)
 row=json.loads(record.read_bytes());row['root']=str(base/'foreign-root');record.write_text(json.dumps(row))
 try:trust.trust_reviewed_recovery(trust.capture_recovery_review(),'fresh-passphrase')
 except ValueError as exc:assert str(exc)=='skills_recovery_binding_changed'
 else:raise AssertionError('foreign partial binding was accepted')
 assert all(not ActivationStore(Path(item['store_root'])).allowed(item['generation'],'skills') for item in witnesses)
 assert all((second_history/name).read_bytes()==data for name,data in second_before.items())
 print('retired and reopened');sys.exit(0)
trust.trust_reviewed_recovery(trust.capture_recovery_review(),'fresh-passphrase')
trust.ensure_skill_trusted('demo')
assert not trust.script_execution_granted('demo')
assert all(ActivationStore(Path(item['store_root'])).allowed(item['generation'],'skills') for item in witnesses)
assert all((historical/name).read_bytes()==data for name,data in history.items())
assert all((second_history/name).read_bytes()==data for name,data in second_before.items())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "skills", damage, script=script)
