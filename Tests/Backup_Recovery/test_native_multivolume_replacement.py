"""Actual coherent capture and replacement staging span two native devices."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_complete_roundtrip import _run_profile_child

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
archive=acquire(Path(seed['archive']),fixture/'acquired',ArchiveLimits(),None,threading.Event())
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
manual=fixture/'inactive';manual.mkdir(mode=0o700)
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
 plan=plan_restore(archive,mode='replace',destinations=mapping,target=target,profile_names={source_profile:name},safety_scope=safety)
 (fixture/'control').mkdir(mode=0o700)
 journal=Journal(fixture/'control','two-device-stage')
 candidate=stage_restore(archive,plan,fixture/'work',threading.Event(),journal=journal)
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


def test_actual_two_device_capture_plans_and_stages_replacement(tmp_path):
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
