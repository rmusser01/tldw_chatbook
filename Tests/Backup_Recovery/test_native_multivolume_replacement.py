"""Actual coherent capture and replacement staging span two native devices."""

import json
import os
import subprocess  # nosec B404: fixed private child programs.
import sys
import tempfile
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_complete_roundtrip import _run_profile_child
from Tests.ProductionApp.test_backup_restore_end_to_end import (
    native_package,  # noqa: F401
)

_PRIVATE = r"""
import asyncio,hashlib,json,os,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
fixture=Path(os.environ['MULTIVOLUME_FIXTURE'])
image=Path(os.environ['MULTIVOLUME_IMAGE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
if os.environ.get('MULTIVOLUME_INSTALLED'):
 import tldw_chatbook
 assert Path(tldw_chatbook.__file__).resolve().is_relative_to(Path(os.environ['MULTIVOLUME_INSTALLED']))
from tldw_chatbook.Backup_Recovery.native_files import pinned_directory,native_identity
from tldw_chatbook.Backup_Recovery.qualification import qualified_for
with pinned_directory(image) as fd:
 identity=native_identity(fd);image_device=os.fstat(fd).st_dev
assert identity=={'os':'Darwin','release':'25.5.0','arch':'arm64','python':'3.12.11','filesystem':'apfs','flags':76583448},identity
host_device=fixture.stat().st_dev
assert image_device!=host_device
assert qualified_for('publish_new',image)[0] and not qualified_for('admission',image)[0]
"""

# Assemble a fixed Python program, not a SQL expression. Its queries below are
# literal SELECT statements with bound values.
_CAPTURE = "".join(  # noqa: FLY002 - preserve literal braces in the child program
    (
        _PRIVATE,
        r"""
import sqlite3,zipfile
from contextlib import closing
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture,capture
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery import storage_admission
async def main():
 app=TldwCli();monitoring=asyncio.create_task(monitor_app(app))
 cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(50,cancel.set)
 try:
  db=app.chachanotes_db;research=app.local_research_service
  assert research.db_path==image/'research.db'
  note=db.add_note('Two-device note','Archived host value')
  session=research.create_session(title='Two-device research',query='Archived image value')
  old_note=db.get_note_by_id(note)
  options={'staging_parent':fixture};preview=preview_capture((selector,),options=options)
  (fixture/'capture-preview.json').write_text(json.dumps({'complete':preview.complete,'issues':preview.issues,'blocking':[{'owner':item.owner,'id':item.logical_id,'path':str(item.path),'status':item.status} for item in preview.items if item.owner=='unknown' or item.status in {'unsupported','unavailable','missing_required'}]},indent=2))
  assert preview.complete,preview.issues
  destination=fixture/'source.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  doc=json.loads(captured.manifest_bytes)
  assert doc['consistency']=='coherent'
  owners={row['owner_id']:row for row in doc['files']}
  assert {'config','db.chachanotes.primary','research.local'}<=owners.keys()
  source_items={item.logical_id:item for item in captured.inventory.items}
  assert source_items[owners['research.local']['logical_id']].path==research.db_path
  assert research.db_path.stat().st_dev==image_device
  assert source_items[owners['db.chachanotes.primary']['logical_id']].path.stat().st_dev==host_device
  assert db.update_note(note,{'content':'Current host value'},expected_version=old_note['version'])
  current_session=research.update_session(session['id'],query='Current image value')
  assert db.get_note_by_id(note)['content']=='Current host value' and current_session['query']=='Current image value'
  await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  acquired=acquire(destination,fixture/'capture-readback',ArchiveLimits(),None,cancel)
  assert verify_sealed(acquired).consistency=='coherent'
  # These copies are inert native readback, not owner constructor/activation.
  with zipfile.ZipFile(acquired.path) as archive:
   for owner,query,key,wanted in (('db.chachanotes.primary','SELECT content FROM notes WHERE id=?',note,'Archived host value'),('research.local','SELECT query FROM research_sessions WHERE id=?',session['id'],'Archived image value')):
    copied=fixture/('readback-'+owner+'.sqlite');copied.write_bytes(archive.read(owners[owner]['payload']))
    with closing(sqlite3.connect(copied.as_uri()+'?mode=ro',uri=True)) as connection:
     assert connection.execute(query,(key,)).fetchone()==(wanted,)
  (fixture/'seed.json').write_text(json.dumps({'archive':str(destination),'archive_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'note':note,'research':session['id'],'selector':str(selector),'config_hex':selector.read_bytes().hex(),'research_path':str(research.db_path),'host_device':host_device,'image_device':image_device,'identity':identity,'archived_values':['Archived host value','Archived image value'],'current_values':['Current host value','Current image value']},indent=2))
  assert not blocked_attempts(),blocked_attempts()
 finally:
  cancel.set();watchdog.cancel();monitoring.cancel();await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
print('COHERENT_TWO_DEVICE_CAPTURE_AND_CURRENT_EDITS',flush=True)
""",
    )
)

_STAGE = (
    _PRIVATE
    + r"""
import tomllib
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
from tldw_chatbook.Backup_Recovery.profile_paths import data_base
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore,recheck_targets
from tldw_chatbook.Backup_Recovery.staging import stage_restore
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.plan_records import load_plan
seed=json.loads((fixture/'seed.json').read_text())
assert selector.read_bytes().hex()==seed['config_hex']
assert hashlib.sha256(Path(seed['archive']).read_bytes()).hexdigest()==seed['archive_sha256']
reviewed=os.environ.get('MULTIVOLUME_REVIEWED')=='1'
archive=acquire(Path(seed['archive']),fixture/('reviewed-acquired' if reviewed else 'acquired'),ArchiveLimits(),None,threading.Event())
doc=verify_sealed(archive);assert doc.consistency=='coherent' and len(doc.profile_ids)==1
target=preview_capture((selector,),options={})
(fixture/'target-preview.json').write_text(json.dumps({'complete':target.complete,'issues':target.issues,'blocking':[{'owner':item.owner,'id':item.logical_id,'path':str(item.path),'status':item.status} for item in target.items if item.owner=='unknown' or item.status in {'unsupported','unavailable','missing_required'}]},indent=2))
assert target.complete,target.issues
actual={item.logical_id:item for item in target.items}
source_profile=doc.profile_ids[0]
current=next(item for item in target.items if item.owner=='config' and item.path==selector)
current_profile=current.logical_id.split(':')[1]
def current_key(key):
 prefix='profile:'+source_profile+':'
 return 'profile:'+current_profile+':'+key[len(prefix):] if key.startswith(prefix) else key
manual=fixture/'inactive';manual.mkdir(mode=0o700,exist_ok=reviewed)
roots={row.logical_id:row for row in doc.directories if row.parent_id is None}
mapping={};deferred={'agents.history','eval.definitions','tts.voices','persona.visual_identity_builtin'}
for index,(key,root) in enumerate(roots.items()):
 members=[row for row in doc.files if row.root_id==key]
 if {row.owner_id for row in members}&deferred:
  mapping[key]=manual/('reviewed-inactive-'+str(index));continue
 if current_key(key) in actual and actual[current_key(key)].path is not None:
  mapping[key]=actual[current_key(key)].path;continue
 choices=set()
 for row in members:
  path=actual[current_key(row.logical_id)].path
  for part in Path(row.relative_path).parts:path=path.parent
  choices.add(path)
 assert len(choices)==1,(key,choices)
 mapping[key]=choices.pop()
config=tomllib.loads(selector.read_text());name=config.get('general',{}).get('users_name','default_user')
mapping['profile:'+source_profile+':paths.data_dir']=data_base(config)
safety=tuple(item.logical_id for item in target.items if item.owner=='persona.visual_identity_builtin' and item.status in {'included','included_directory'})
before={str(item.path):hashlib.sha256(item.path.read_bytes()).hexdigest() for item in target.items if item.status=='included' and item.path is not None}
(fixture/'reviewed-mapping.json').write_text(json.dumps({'destinations':{key:str(path) for key,path in mapping.items()},'safety_scope':safety,'profile_name':name},indent=2))
try:
 omissions=tuple(json.loads((fixture/'credential-review.json').read_text())['issues']) if reviewed else ()
 plan=plan_restore(archive,mode='replace',destinations=mapping,target=target,profile_names={source_profile:name},safety_scope=safety,acknowledged_credential_issues=omissions)
 (fixture/'control').mkdir(mode=0o700,exist_ok=reviewed)
 journal=Journal(fixture/'control','two-device-reviewed-stage' if reviewed else 'two-device-stage')
 candidate=stage_restore(archive,plan,fixture/('reviewed-work' if reviewed else 'work'),threading.Event(),journal=journal)
except Exception as error:
 (fixture/'stage-refusal.json').write_text(json.dumps({'type':type(error).__name__,'args':error.args},default=str));raise
assert load_plan(journal)==plan
recheck_targets(plan)
descriptor=json.loads((candidate/'candidate.json').read_text())
assert {Path(path).stat().st_dev for path in descriptor['private_roots']}=={host_device,image_device}
for row in descriptor['artifacts']:
 destination=Path(row['destination']);parent=destination.parent
 while not parent.exists():parent=parent.parent
 assert Path(row['candidate']).stat().st_dev==parent.stat().st_dev
research=next(row for row in descriptor['artifacts'] if row['destination']==seed['research_path'])
assert Path(research['candidate']).stat().st_dev==image_device and research['publication_unit']
assert any(Path(row['candidate']).stat().st_dev==host_device and row['publication_unit'] for row in descriptor['artifacts'])
assert {path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in before}==before
assert hashlib.sha256(Path(seed['archive']).read_bytes()).hexdigest()==seed['archive_sha256']
(fixture/'stage-evidence.json').write_text(json.dumps({'candidate':str(candidate),'operation':journal.operation_id,'archive_sha256':seed['archive_sha256'],'target_fingerprint':plan.target_fingerprint,'private_roots':descriptor['private_roots'],'artifacts':descriptor['artifacts'],'credential_issues':descriptor['credential_issues'],'host_device':host_device,'image_device':image_device,'image_identity':identity,'original_files_unchanged':len(before),'restoration_validated':False,'replacement_executed':False,'blocked_network_attempts':len(blocked_attempts())},indent=2))
assert not blocked_attempts(),blocked_attempts()
print('ACTUAL_TWO_DEVICE_REPLACEMENT_CANDIDATE_STAGED',flush=True)
"""
)


def _two_device_fixture(tmp_path, installed=None):
    supplied = os.environ.get("TLDW_TEST_APFS_MOUNT")
    if not supplied:
        pytest.skip("requires the explicitly mounted disposable APFS fixture")
    root = tmp_path.resolve()
    mount = Path(supplied).resolve(strict=True)
    assert mount.is_dir() and mount.stat().st_dev != root.stat().st_dev
    image = Path(tempfile.mkdtemp(prefix="chatbook-replacement-", dir=mount))
    for name in ("home", "xdg-config", "xdg-data", "cache", "tmp", "profile"):
        (root / name).mkdir(mode=0o700)
    profile = root / "profile"
    (profile / "custom").mkdir(mode=0o700)
    (profile / "data").mkdir(mode=0o700)
    selector = profile / "config.toml"
    locations = {
        "chachanotes_db_path": profile / "custom" / "notes.db",
        "media_db_path": profile / "custom" / "media.db",
        "research_db_path": image / "research.db",
        "prompts_db_path": profile / "custom" / "prompts.db",
    }
    selector.write_text(
        '[general]\nusers_name="default_user"\ndefault_tab="settings"\n'
        "[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n"
        f"[paths]\ndata_dir={json.dumps(str(profile / 'data'))}\n[database]\n"
        + "".join(f"{key}={json.dumps(str(path))}\n" for key, path in locations.items())
    )
    selector.chmod(0o600)
    environment = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "GOMODCACHE", "GOCACHE", "GOPROXY")
        if key in os.environ
    }
    environment.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        XDG_CONFIG_HOME=str(root / "xdg-config"),
        XDG_DATA_HOME=str(root / "xdg-data"),
        XDG_CACHE_HOME=str(root / "cache"),
        TMPDIR=str(root / "tmp"),
        TLDW_CONFIG_PATH=str(selector),
        MULTIVOLUME_FIXTURE=str(root),
        MULTIVOLUME_IMAGE=str(image),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    if installed is not None:
        environment["MULTIVOLUME_INSTALLED"] = str(installed)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(installed), environment["PYTHONPATH"])
        )
    (root / "fixture.json").write_text(
        json.dumps({"image": str(image), "selector": str(selector)})
    )
    _run_profile_child(root, "capture", _CAPTURE, environment)
    _run_profile_child(root, "stage", _STAGE, environment)
    evidence = json.loads((root / "stage-evidence.json").read_text())
    assert evidence["host_device"] != evidence["image_device"]
    assert (
        evidence["original_files_unchanged"]
        and evidence["blocked_network_attempts"] == 0
    )
    return root, environment


def test_actual_two_device_capture_plans_and_stages_replacement(tmp_path):
    _two_device_fixture(tmp_path)


_INTERRUPT = "".join(  # noqa: FLY002 - fixed child source with literal braces.
    (
        _PRIVATE,
        r"""
import sqlite3,zipfile
from contextlib import closing
from tldw_chatbook.Backup_Recovery import publication,replacement
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.plan_records import load_plan
seed=json.loads((fixture/'seed.json').read_text())
staged=json.loads((fixture/'stage-evidence.json').read_text())
plan=load_plan(Journal(fixture/'control',staged['operation']))
original_capture=replacement.capture_verify_rollback
safety_checked=False
def capture_checked(*args,**kwargs):
 global safety_checked
 try:path=original_capture(*args,**kwargs)
 except replacement.RollbackCredentialReviewRequired as error:
  from collections import Counter
  assert not os.environ.get('MULTIVOLUME_REVIEWED')
  material=next(kwargs['work_root'].glob('originals-*/credential-recovery.json'))
  records=json.loads(material.read_text())['records']
  assert len(records)==17 and all(row['status']=='unreadable' and row['remappable'] is False for row in records)
  assert Counter(row['kind'] for row in records)=={'citation':5,'generation':8,'server':4}
  assert {row['purpose'] for row in records if row['kind']=='server'}=={'api_key','bearer_token','access_token','refresh_token'}
  assert set(error.issues)=={'credential_unreadable:'+row['id'] for row in records}
  (fixture/'credential-review.json').write_text(json.dumps({'issues':error.issues,'records':records},indent=2))
  raise
 # Independent decryption happens before the publisher receives this result.
 archive=acquire(path,fixture/'safety-readback',ArchiveLimits(),b'two-device safety',threading.Event())
 manifest=verify_sealed(archive)
 assert archive.encrypted_source is not None and manifest.consistency=='coherent'
 values={}
 with zipfile.ZipFile(archive.path) as contents:
  for owner,query,key,wanted in (('db.chachanotes.primary','SELECT content FROM notes WHERE id=?',seed['note'],'Current host value'),('research.local','SELECT query FROM research_sessions WHERE id=?',seed['research'],'Current image value')):
   row=next(row for row in manifest.files if row.owner_id==owner)
   copied=fixture/('safety-'+owner+'.sqlite');copied.write_bytes(contents.read(row.payload))
   with closing(sqlite3.connect(copied.as_uri()+'?mode=ro',uri=True)) as connection:
    value=connection.execute(query,(key,)).fetchone()
   assert value==(wanted,),value
   values[owner]={'logical_id':row.logical_id,'value':value[0],'payload_sha256':hashlib.sha256(copied.read_bytes()).hexdigest()}
 (fixture/'safety-evidence.json').write_text(json.dumps({'ciphertext':str(path),'ciphertext_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sealed_digest':archive.digest,'values':values,'checked_before_first_move':True},indent=2))
 safety_checked=True
 return path
replacement.capture_verify_rollback=capture_checked
original_begin=publication._begin_move
original_complete=publication._complete_move
original_publish=publication.publish_new
completed_devices=[]
def record_exit(journal,parent,intent,boundary):
 records=journal._records(parent)
 assert any(row.event=='rollback_verified' for row in records)
 (fixture/'interruption.json').write_text(json.dumps({'operation':journal.operation_id,'boundary':boundary,'intent':intent.model_dump(),'completed_devices':completed_devices,'records':[row.model_dump() for row in records]},indent=2))
 assert not blocked_attempts(),blocked_attempts()
 os._exit(91)
active=None
def begin(journal,parent,item,step):
 global active
 assert safety_checked
 assert any(row.event=='rollback_verified' for row in journal._records(parent))
 intent=original_begin(journal,parent,item,step)
 active=(journal,parent,intent)
 if os.environ['MULTIVOLUME_BOUNDARY']=='between_devices' and completed_devices and intent.source.device!=completed_devices[-1]:
  record_exit(journal,parent,intent,'between_devices')
 return intent
def complete(journal,parent,prepared,intent,*,moved):
 original_complete(journal,parent,prepared,intent,moved=moved)
 if moved:completed_devices.append(intent.source.device)
def publish(source,destination,**kwargs):
 original_publish(source,destination,**kwargs)
 if os.environ['MULTIVOLUME_BOUNDARY']=='image_native' and destination==Path(seed['research_path']):
  journal,parent,intent=active
  assert intent.step=='publish' and intent.source.device==image_device
  assert not source.exists() and destination.exists()
  record_exit(journal,parent,intent,'image_native')
publication._begin_move=begin
publication._complete_move=complete
publication.publish_new=publish
try:replacement.replace(plan,Path(staged['candidate']),control_root=fixture/'control',rollback_password=b'two-device safety',cancel=threading.Event())
except replacement.RollbackCredentialReviewRequired:
 from tldw_chatbook.Backup_Recovery import bootstrap
 root=bootstrap.default_bootstrap_root();pending=bootstrap._records(root)[0]
 assert len(pending)==1
 operation=pending[0]['operation_id'];journal=Journal(fixture/'control',operation)
 with journal._locked(exclusive=False) as parent:records=journal._records(parent)
 assert [row.event for row in records]==['candidate_staged','prepared']
 assert not bootstrap.startup_permission(selector,root)[0]
 assert replacement.recover_replacement(operation,control_root=fixture/'control',action='abort',rollback_password=None,cancel=threading.Event())=='aborted'
 assert bootstrap.startup_permission(selector,root)[0]
 with journal._locked(exclusive=False) as parent:after=journal._records(parent)
 assert after[-1].event=='prepublication_aborted'
 (fixture/'review-abort.json').write_text(json.dumps({'operation':operation,'before':[row.model_dump() for row in records],'after':[row.model_dump() for row in after]},indent=2))
 assert not blocked_attempts(),blocked_attempts()
 sys.exit(92)
raise AssertionError('native interruption boundary was not reached')
""",
    )
)

_RECOVER = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery import bootstrap,replacement,publication
from tldw_chatbook.Backup_Recovery.journal import Journal,_Prepared,_MoveIntent,_evidence_digest
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
seed=json.loads((fixture/'seed.json').read_text())
interrupted=json.loads((fixture/'interruption.json').read_text())
safety=json.loads((fixture/'safety-evidence.json').read_text())
operation=interrupted['operation'];root=bootstrap.default_bootstrap_root()
pending,profiles=bootstrap._records(root)
assert any(row['operation_id']==operation for row in pending)
assert not bootstrap.startup_permission(selector,root)[0]
journal=Journal(fixture/'control',operation)
with journal._locked(exclusive=False) as parent:records=journal._records(parent)
prepared=_Prepared.model_validate(next(row.evidence for row in records if row.event=='prepared'))
verified=next(row.evidence for row in records if row.event=='rollback_verified')
assert verified['ciphertext']['path']==safety['ciphertext']
assert verified['sealed_digest']==safety['sealed_digest']
assert hashlib.sha256(Path(safety['ciphertext']).read_bytes()).hexdigest()==safety['ciphertext_sha256']
for owner,value in safety['values'].items():
 group=next(row for row in verified['sqlite_groups'] if row['owner_id']==owner)
 assert group['logical_id']==value['logical_id'] and group['payload_digest']==value['payload_sha256']
 assert group['artifacts'] and set(group['artifacts'])<=set(verified['coverage'])
 expected=Path(seed['research_path']) if owner=='research.local' else selector.parent/'custom'/'notes.db'
 assert Path(group['source']['path'])==expected
 assert group['source']['device']==(image_device if owner=='research.local' else host_device)
first_move=next(index for index,row in enumerate(records) if row.event=='move_intended')
assert next(index for index,row in enumerate(records) if row.event=='rollback_verified')<first_move
intent=_MoveIntent.model_validate(interrupted['intent'])
assert not any(row.event=='move_observed' and row.evidence['intent_digest']==_evidence_digest(intent.model_dump()) for row in records)
assert publication._move_position(intent)==('after' if interrupted['boundary']=='image_native' else 'before')
if interrupted['boundary']=='between_devices':
 assert interrupted['completed_devices'] and intent.source.device!=interrupted['completed_devices'][-1]
assert {item.previous.device for item in prepared.artifacts if item.previous is not None}=={host_device,image_device}
for item in prepared.artifacts:
 if item.previous is not None:
  assert Path(item.retained).parent.stat().st_dev==item.previous.device
before=[row.model_dump() for row in records]
action=os.environ['MULTIVOLUME_ACTION']
outcome=replacement.recover_replacement(operation,control_root=fixture/'control',action=action,rollback_password=b'two-device safety',cancel=threading.Event())
assert outcome==('committed' if action=='finish' else 'rolled_back'),outcome
with journal._locked(exclusive=False) as parent:after=journal._records(parent)
assert after[-1].event==outcome
assert not any(row['operation_id']==operation for row in bootstrap._records(root)[0])
assert bootstrap.startup_permission(selector,root)[0]
if action=='rollback':
 for item in prepared.artifacts:
  if item.previous is not None:
   assert publication._matches(item.previous,item.target)
   if item.previous_metadata is not None:
    assert publication._matches(item.previous_metadata,item.target,metadata=True)
assert hashlib.sha256(Path(safety['ciphertext']).read_bytes()).hexdigest()==safety['ciphertext_sha256']
assert hashlib.sha256(Path(seed['archive']).read_bytes()).hexdigest()==seed['archive_sha256']
(fixture/'recovery-evidence.json').write_text(json.dumps({'operation':operation,'action':action,'outcome':outcome,'before':before,'after':[row.model_dump() for row in after],'fence_cleared':True,'safety_sha256':safety['ciphertext_sha256']},indent=2))
assert not blocked_attempts(),blocked_attempts()
print('ACTUAL_TWO_DEVICE_RECOVERY',outcome,flush=True)
"""
)

_READ_RECOVERED = (
    _PRIVATE
    + r"""
from tldw_chatbook.cli import main_cli_runner
sys.argv=['tldw-chatbook','--help']
try:main_cli_runner()
except SystemExit as error:assert error.code in (None,0),error
from tldw_chatbook.app import TldwCli
seed=json.loads((fixture/'seed.json').read_text())
async def main():
 app=TldwCli()
 try:
  from tldw_chatbook.config import get_cli_config_path
  assert get_cli_config_path()==selector
  assert app.chachanotes_db.db_path==selector.parent/'custom'/'notes.db'
  values=[app.chachanotes_db.get_note_by_id(seed['note'])['content'],app.local_research_service.get_session(seed['research'])['query']]
  assert values==seed['archived_values' if os.environ['MULTIVOLUME_ACTION']=='finish' else 'current_values'],values
  assert app.local_research_service.db_path==Path(seed['research_path'])
  assert app.local_research_service.db_path.stat().st_dev==image_device
  assert app.chachanotes_db.db_path.stat().st_dev==host_device
  (fixture/'native-readback.json').write_text(json.dumps({'values':values,'note':seed['note'],'research':seed['research'],'blocked_network_attempts':len(blocked_attempts())},indent=2))
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
print('FRESH_TWO_DEVICE_NATIVE_READBACK',flush=True)
"""
)


@pytest.mark.parametrize("action", ["finish", "rollback"])
@pytest.mark.parametrize("boundary", ["between_devices", "image_native"])
def test_actual_two_device_interruption_recovers(
    tmp_path,
    native_package,  # noqa: F811 - imported pytest fixture.
    boundary,
    action,
):
    root, environment = _two_device_fixture(tmp_path, native_package)
    environment.update(MULTIVOLUME_BOUNDARY=boundary, MULTIVOLUME_ACTION=action)
    script = root / "interrupt.py"
    script.write_text(_INTERRUPT)
    with (root / "interrupt.log").open("w") as output:
        result = subprocess.run(  # nosec B603: fixed interpreter and private script.
            [sys.executable, str(script)],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=35,
            check=False,
        )
    assert result.returncode == 92, (root / "interrupt.log").read_text()
    (root / "stage-before-review.json").write_bytes(
        (root / "stage-evidence.json").read_bytes()
    )
    environment["MULTIVOLUME_REVIEWED"] = "1"
    _run_profile_child(root, "reviewed-stage", _STAGE, environment)
    with (root / "reviewed-interrupt.log").open("w") as output:
        # The full profile kept moving through 34.696s of the old 35s cap.
        # Allow its remaining native moves; Admission's own deadline is unchanged.
        result = subprocess.run(  # nosec B603: fixed interpreter and private script.
            [sys.executable, str(script)],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=90,
            check=False,
        )
    assert result.returncode == 91, (root / "reviewed-interrupt.log").read_text()
    # Rollback still made native progress at 88.385s of the original 90s cap.
    _run_profile_child(
        root,
        "recover",
        _RECOVER,
        environment,
        timeout=180 if action == "rollback" else 90,
    )
    _run_profile_child(root, "native-readback", _READ_RECOVERED, environment)
