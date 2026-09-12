"""Actual F9 shutdown, fresh recovery, explicit safety review and replacement."""

import json
import os
import subprocess
import sys
from pathlib import Path

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414 - installed product fixture
)
from Tests.Backup_Recovery.test_home_citation_retirement import _run

# The seed uses real native stores and proves public Complete capture, captured
# note bytes, and resumed ordinary writes before archive packaging.
_SEED = r"""
import asyncio,json,os,sqlite3,sys,threading
from contextlib import closing
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
import tldw_chatbook
installed=Path(os.environ['TLDW_TEST_INSTALLED_PACKAGE'])
assert Path(tldw_chatbook.__file__).resolve()==installed/'tldw_chatbook'/'__init__.py'
selector=Path(os.environ['TLDW_CONFIG_PATH']);selector.write_text('[general]\nusers_name="default_user"\ndefault_tab="settings"\n[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n');selector.chmod(0o600)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService,default_control_root
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery import archive_writer,archive_reader,storage_admission
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
def inventory_diagnostic(inventory):
 blocking=sorted((item.owner,item.logical_id,item.status,item.dependencies,item.metadata.relative_path if item.metadata else None) for item in inventory.items if item.status in {'unsupported','unavailable','missing_required'})
 observed=[]
 for item in inventory.items:
  if item.path is None or item.status in {'unused','intentionally_excluded','intentionally_deleted'}:continue
  try:observed.append((item,item.path.resolve(strict=True)))
  except (OSError,RuntimeError):continue
 ancestor_pairs=sorted((parent.owner,parent.logical_id,child.owner,child.logical_id) for child,child_path in observed for parent,parent_path in observed if parent_path in child_path.parents and parent.owner!=child.owner)
 return {'issues':inventory.issues,'blocking':blocking,'ancestor_pairs':ancestor_pairs}
from Tests.Backup_Recovery.thread_diagnostics import observe_threads,observe_recovery_failures
async def main():
 home=Path.home()
 stop_diagnostics=observe_threads(home/'seed-stacks.log',interval=60)
 stop_failures=observe_recovery_failures(home/'recovery-failures.log')
 app=TldwCli();selector=Path(os.environ['TLDW_CONFIG_PATH'])
 service=RecoveryService(default_control_root())
 from Tests.Backup_Recovery.test_restore_plan import sealed
 warm=sealed(home)
 inspected=service.start_inspection(warm.path,password=None)
 assert (await asyncio.to_thread(service.wait,inspected,timeout=15))['state']=='succeeded'
 entered,release=threading.Event(),threading.Event()
 writer=archive_writer.write_archive
 def held_writer(*args,**kwargs):
  entered.set();assert release.wait(15);return writer(*args,**kwargs)
 archive_writer.write_archive=held_writer
 monitoring=asyncio.create_task(monitor_app(app))
 try:
  note=app.chachanotes_db.add_note('Before backup','Captured native value.')
  options={'staging_parent':home}
  destination=home/'service.tldw-backup.zip'
  details=await asyncio.to_thread(service.preview_backup_details,(selector,),options=options,destination=destination)
  preview=details['inventory']
  assert details['capacity'] and all(row['sufficient'] for row in details['capacity'])
  assert preview.complete,inventory_diagnostic(preview)
  operation=service.start_backup((selector,),preview.scope_digest,destination,options=options,password=None)
  async with asyncio.timeout(150 if sys.platform=='win32' else 65):
   while not entered.is_set():
    assert service.status(operation)['state']=='running',dict(service.status(operation))
    await asyncio.sleep(.01)
  assert service.status(operation)['phase']=='packaging' and not destination.exists()
  for _ in range(500):
   if storage_admission._pause is None and app._backup_runtime_maintenance is None:break
   await asyncio.sleep(.01)
  assert storage_admission._pause is None
  after=app.chachanotes_db.add_note('After capture','Ordinary writer resumed.')
  release.set()
  result=await asyncio.to_thread(service.wait,operation,timeout=30)
  assert result['state']=='succeeded' and result['phase']=='archive_verified',dict(result)
  assert result['result']['complete'] and result['result']['path']==str(destination)
  assert not result['result'].get('restoration_validated',False)
  acquired=archive_reader.acquire(destination,home/'readback',ArchiveLimits(),None,threading.Event())
  manifest=json.loads(acquired.manifest_bytes)
  member=next(row for row in manifest['files'] if row['owner_id']=='db.chachanotes.primary')
  import zipfile
  payload=home/'readback.sqlite'
  with zipfile.ZipFile(acquired.path) as archive:payload.write_bytes(archive.read(member['payload']))
  with closing(sqlite3.connect(payload)) as db:
   assert db.execute('SELECT content FROM notes WHERE id=?',(note,)).fetchone()[0]=='Captured native value.'
   assert db.execute('SELECT 1 FROM notes WHERE id=?',(after,)).fetchone() is None
  assert not blocked_attempts()
 finally:
  release.set();archive_writer.write_archive=writer
  await asyncio.to_thread(service.close)
  monitoring.cancel();await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
  stop_diagnostics();stop_failures()
asyncio.run(main())
print('retired and reopened')
"""

# This terminal driver runs before the fixed production fresh-process entry.
# It chooses only displayed controls; all plans and native effects are real.
_MINIMAL_DRIVER = r"""
import asyncio,json,os,sys
from pathlib import Path
saved=json.loads((Path.home()/'probe-mapping.json').read_text())
sys.path.append(saved['test_root'])
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
import tldw_chatbook
assert Path(tldw_chatbook.__file__).resolve()==Path(saved['package'])/'tldw_chatbook'/'__init__.py'
from textual.app import App
from textual.widgets import Input,Button,Select,Static
assert 'tldw_chatbook.app' not in sys.modules
assert 'tldw_chatbook.config' not in sys.modules
print('FRESH_MINIMAL_ENTRY',flush=True)
def headless(app,*args,**kwargs):
 async def mounted():
  async with app.run_test(size=(120,42)) as pilot:
   screen=app.screen
   assert screen._inspection_id is None and screen._restore_plan is None
   assert screen.query_one('#backup-restore-mode',Select).value=='replace'
   print('FRESH_MINIMAL_NO_INSPECTION',flush=True)
   screen.query_one('#backup-inspect',Button).focus();await pilot.press('enter')
   async with asyncio.timeout(20):
    while not screen.query_one('#backup-restore-form').display:await asyncio.sleep(.03)
   print('FRESH_ARCHIVE_INSPECTED',flush=True)
   for index,slot in enumerate(screen._inspection_summary['destination_slots']):
    screen.query_one(f'#backup-root-{index}',Input).value=saved['mapping'][slot['logical_id']]
   screen.query_one('#backup-profile-name-0',Input).value=saved['name']
   screen.query_one('#backup-rollback-password',Input).value='test-only-new-safety-password'
   screen.query_one('#backup-rollback-confirm',Input).value='test-only-new-safety-password'
   await pilot.pause()
   screen.query_one('#backup-review-restore',Button).focus();await pilot.press('enter')
   async with asyncio.timeout(25):
    while screen.query_one('#backup-start-restore',Button).disabled:
     text=str(screen.query_one('#backup-restore-preview',Static).render())
     if 'refused' in text:
      print('CONCRETE_UI_REVIEW_BLOCKER',text,flush=True)
      (Path.home()/'probe-result.json').write_text(json.dumps({'checkpoint':'preview','refusal':text}))
      return
     await asyncio.sleep(.03)
   safety_keys=set(saved['builtin_safety'])
   assert safety_keys
   boxes=list(screen.query('.backup-safety-member'))
   by_key={box.name:box for box in boxes}
   assert safety_keys <= by_key.keys(),safety_keys-by_key.keys()
   assert not any(box.value for box in boxes)
   for key in safety_keys:by_key[key].value=True
   await pilot.pause()
   assert screen.query_one('#backup-start-restore',Button).disabled
   print('EXPLICIT_BUILTIN_SAFETY_SELECTION',len(safety_keys),flush=True)
   screen.query_one('#backup-review-restore',Button).focus();await pilot.press('enter')
   async with asyncio.timeout(25):
    while screen.query_one('#backup-start-restore',Button).disabled:
     text=str(screen.query_one('#backup-restore-preview',Static).render())
     if 'refused' in text:
      print('CONCRETE_SAFETY_REVIEW_BLOCKER',text,flush=True)
      (Path.home()/'probe-result.json').write_text(json.dumps({'checkpoint':'safety_review','refusal':text}))
      return
     await asyncio.sleep(.03)
   assert set(screen._restore_plan.safety_scope)==safety_keys
   print('FRESH_REPLACEMENT_REVIEWED',flush=True)
   screen.query_one('#backup-start-restore',Button).focus();await pilot.press('enter')
   current=app.recovery_service.current()
   if current['kind']!='restore':
    text=str(screen.query_one('#backup-message',Static).render())
    print('CONCRETE_UI_START_BLOCKER',text,flush=True)
    (Path.home()/'probe-result.json').write_text(json.dumps({'checkpoint':'start','refusal':text}))
    return
   state=await asyncio.to_thread(app.recovery_service.wait,current['operation_id'],timeout=55)
   print('FRESH_REPLACEMENT_TERMINAL',dict(state),flush=True)
   assert state['state']=='recovery_required',dict(state)
   expected=tuple(state['review_issues'])
   assert expected and all(code.startswith('credential_') for code in expected)
   async with asyncio.timeout(10):
    while len(list(screen.query('.backup-acknowledge-restore-credential')))!=len(expected):await asyncio.sleep(.03)
   boxes=list(screen.query('.backup-acknowledge-restore-credential'))
   assert {box.name for box in boxes}==set(expected) and not any(box.value for box in boxes)
   print('OMISSIONS_VISIBLE_UNCHECKED',len(boxes),flush=True)
   screen.query_one('#backup-open-copies',Button).focus();await pilot.press('enter')
   async with asyncio.timeout(15):
    while not list(screen.query('.backup-recover-abort')):await asyncio.sleep(.03)
   screen.query_one('.backup-recover-abort',Button).focus();await pilot.press('enter')
   aborted=await asyncio.to_thread(app.recovery_service.wait,app.recovery_service.current()['operation_id'],timeout=20)
   assert aborted['result']['aborted'],dict(aborted)
   print('ACTUAL_ABORTED_UNTOUCHED',flush=True)
   screen.query_one('#backup-open-inspect',Button).focus();await pilot.press('enter')
   for box in boxes:box.value=True
   screen.query_one('#backup-rollback-password',Input).value='test-only-new-safety-password'
   screen.query_one('#backup-rollback-confirm',Input).value='test-only-new-safety-password'
   await pilot.pause()
   screen.query_one('#backup-review-restore',Button).focus();await pilot.press('enter')
   async with asyncio.timeout(25):
    while screen.query_one('#backup-start-restore',Button).disabled:
     text=str(screen.query_one('#backup-restore-preview',Static).render())
     if 'refused' in text:
      print('CONCRETE_NEW_REVIEW_BLOCKER',text,flush=True)
      (Path.home()/'probe-result.json').write_text(json.dumps({'checkpoint':'second_preview','refusal':text}))
      return
     await asyncio.sleep(.03)
   assert set(screen._restore_plan.acknowledged_credential_issues)==set(expected)
   assert set(screen._restore_plan.safety_scope)==safety_keys
   print('ACTUAL_SECOND_REVIEW_ACKNOWLEDGED',flush=True)
   screen.query_one('#backup-start-restore',Button).focus();await pilot.press('enter')
   state=await asyncio.to_thread(app.recovery_service.wait,app.recovery_service.current()['operation_id'],timeout=90)
   print('SECOND_REPLACEMENT_TERMINAL',dict(state),flush=True)
   result={'state':state['state'],'phase':state['phase'],'issues':list(state['issues']),'review_issues':list(state['review_issues']),'result':dict(state['result'])}
   (Path.home()/'probe-result.json').write_text(json.dumps(result,default=str))
  assert not blocked_attempts(),blocked_attempts()
 asyncio.run(mounted())
App.run=headless
"""

_NORMAL = r"""
import builtins,json,os,sys,tomllib
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
import tldw_chatbook
installed=Path(os.environ['TLDW_TEST_INSTALLED_PACKAGE'])
assert Path(tldw_chatbook.__file__).resolve()==installed/'tldw_chatbook'/'__init__.py'
from tldw_chatbook.Backup_Recovery import archive_reader,launcher
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService,default_control_root
home=Path.home();selector=Path(os.environ['TLDW_CONFIG_PATH'])
manual=home/'selected-manual-parent';manual.mkdir(mode=0o700)
from tldw_chatbook.Backup_Recovery import bootstrap
assert not bootstrap._records(bootstrap.default_bootstrap_root())[1]
service=RecoveryService(default_control_root())
inspection=service.start_inspection(home/'service.tldw-backup.zip',password=None)
assert service.wait(inspection)['state']=='succeeded'
archive=service.inspection(inspection);doc=archive_reader.verify_sealed(archive)
target=service.preview_backup((selector,),options={})
assert target.complete,target.issues
actual={item.logical_id:item for item in target.items}
source_profile=doc.profile_ids[0]
current_config=next(item for item in target.items if item.owner=='config' and item.path==selector)
current_profile=current_config.logical_id.split(':')[1]
def current_key(key):
 prefix='profile:'+source_profile+':'
 return 'profile:'+current_profile+':'+key[len(prefix):] if key.startswith(prefix) else key
roots={row.logical_id:row for row in doc.directories if row.parent_id is None}
mapping={};deferred={'agents.history','eval.definitions','tts.voices','persona.visual_identity_builtin'}
for index,(key,root) in enumerate(roots.items()):
 members=[row for row in doc.files if row.root_id==key]
 owners={row.owner_id for row in members}
 if owners & deferred:
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
profile=doc.profile_ids[0]
config=tomllib.loads(selector.read_text())
name=config.get('general',{}).get('users_name','default_user')
from tldw_chatbook.Backup_Recovery.profile_paths import data_base
mapping[f'profile:{profile}:paths.data_dir']=data_base(config)
print('TARGET_MAPPING',len(mapping),flush=True)
service.close()

import asyncio
for module in ('sounddevice','pyaudio'):sys.modules[module]=None
from textual.widgets import Input,Button,Select
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import recovery_restart
from tldw_chatbook.Utils import terminal_utils
(home/'probe-mapping.json').write_text(json.dumps({'mapping':{key:str(path) for key,path in mapping.items()},'name':name,'package':str(installed),'test_root':os.environ['TLDW_TEST_ROOT'],'builtin_safety':[item.logical_id for item in target.items if item.owner=='persona.visual_identity_builtin' and item.status in {'included','included_directory'}]}))
original_run=TldwCli.run
async def drive(pilot):
 app=pilot.app
 async with asyncio.timeout(20):
  while not app._ui_ready:await asyncio.sleep(.03)
 await pilot.press('f9');await pilot.pause()
 app.screen.query_one('#settings-backup-restore',Button).focus();await pilot.press('enter');await pilot.pause()
 await pilot.click('#backup-open-inspect')
 screen=app.screen
 screen.query_one('#backup-source',Input).value=str(home/'service.tldw-backup.zip')
 await pilot.click('#backup-inspect')
 async with asyncio.timeout(20):
  while screen._inspection_id is None:await asyncio.sleep(.03)
 screen.query_one('#backup-restore-mode',Select).value='replace'
 screen.query_one('#backup-target-config',Input).value=str(selector)
 await pilot.pause()
 assert screen.query_one('#backup-review-restore',Button).disabled
 print('NORMAL_F9_RESTART_REQUESTED',flush=True)
 screen.query_one('#backup-restart',Button).focus();await pilot.press('enter')
 async with asyncio.timeout(20):
  while app.is_running:await asyncio.sleep(.03)
def headless(app,*args,**kwargs):
 return original_run(app,headless=True,size=(120,42),auto_pilot=drive)
TldwCli.run=headless
terminal_utils.warm_up_image_protocol=lambda:None
recovery_restart._ENTRY=DRIVER+recovery_restart._ENTRY
sys.argv=['tldw-chatbook']
from tldw_chatbook.cli import main_cli_runner
main_cli_runner()
raise AssertionError('real exec expected')
"""

_CHILD = "DRIVER=" + repr(_MINIMAL_DRIVER) + "\n" + _NORMAL


def test_full_f9_replacement_after_explicit_safety_and_credential_review(
    tmp_path: Path, native_package: Path
):
    _run(
        tmp_path,
        "service",
        "backup",
        script=_SEED,
        timeout=300 if sys.platform == "win32" else 110,
        installed_package=native_package,
    )
    test_root = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        USERPROFILE=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(tmp_path / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONNOUSERSITE="1",
        PYTHONPATH=os.pathsep.join((str(native_package), str(test_root))),
        TLDW_TEST_INSTALLED_PACKAGE=str(native_package),
        TLDW_TEST_ROOT=str(test_root),
    )
    log = tmp_path / "f9-child.log"
    with log.open("w") as output:
        result = subprocess.run(
            [sys.executable, "-c", _CHILD],
            cwd=tmp_path,
            env=env,
            stdout=output,
            stderr=output,
            text=True,
            # Includes normal-app restart, omission review, and native replacement.
            # The accepted second replacement alone exceeded the old 55s observer.
            timeout=150,
            check=False,
        )
    assert result.returncode == 0, log.read_text()[-8000:]
    state = json.loads((tmp_path / "home" / "probe-result.json").read_text())
    assert (
        state.get("state"),
        state.get("phase"),
        state.get("result", {}).get("restoration_validated"),
    ) == ("succeeded", "restoration_validated", True), state
