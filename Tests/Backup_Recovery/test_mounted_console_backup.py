"""A mounted ordinary Console can capture, verify, and resume native storage."""

import sys

import pytest

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414
)
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import asyncio,json,os,sys,threading,time,zipfile
from pathlib import Path
from contextlib import ExitStack
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
selector=Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\nusers_name="default_user"\ndefault_tab="chat"\n[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[console.rail_state.test]\nwidth=28\n')
selector.chmod(0o600)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService,default_control_root
from tldw_chatbook.Backup_Recovery import archive_reader,storage_admission
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from Tests.Backup_Recovery.thread_diagnostics import observe_threads,observe_recovery_failures,observe_startup_refusals,observe_capture_review,observe_runtime_settlement,stop_observer,_write
from Tests.Backup_Recovery.later_failure_diagnostics import record_failure
from Tests.Backup_Recovery.admission_diagnostics import observe_admission
from Tests.Backup_Recovery.loop_diagnostics import observe_loop_profile
from Tests.Backup_Recovery.initial_screen_observation import initial_screen_observation
phase_start=time.monotonic();phases=[]
def phase(name):
 phases.append({'phase':name,'elapsed':round(time.monotonic()-phase_start,3)})
 try:_write(Path.home()/'mounted-entry-phases.json.log',phases)
 except OSError:pass  # Observation failures must not change the native outcome.
diagnostics=ExitStack()
diagnostics.callback(observe_threads(Path.home()/'mounted-stacks.log',interval=30))
diagnostics.callback(observe_recovery_failures(Path.home()/'mounted-recovery-failures.log'))
diagnostics.callback(observe_startup_refusals(Path.home()/'mounted-startup-refusals.log'))
diagnostics.callback(observe_capture_review(Path.home()/'mounted-capture-review.log'))
diagnostics.callback(observe_runtime_settlement(Path.home()/'mounted-runtime-settlement.log'))
diagnostics.callback(stop_observer,observe_admission(Path.home()/'mounted-admission-timing.log',native_calls=False))
import tldw_chatbook
assert Path(tldw_chatbook.__file__).resolve()==Path(os.environ['TLDW_TEST_INSTALLED_PACKAGE'])/'tldw_chatbook'/'__init__.py'
async def main():
 phase('construct_begin')
 app=TldwCli()
 phase('construct_complete')
 service=RecoveryService(default_control_root())
 async def no_local_discovery(_config):return ()
 app.console_local_server_discovery=no_local_discovery
 try:
  phase('mount_begin')
  async with initial_screen_observation(app),app.run_test(size=(120,42)) as pilot:
   phase('mount_yield')
   async with asyncio.timeout(90):
    while not app._ui_ready:await asyncio.sleep(.05)
   async with asyncio.timeout(90):
    while app._boot_worker_gate is None or not app._boot_worker_gate.is_drained:await asyncio.sleep(.05)
   await asyncio.sleep(3)
   before=app.chachanotes_db.add_note('Mounted backup','Saved before mounted capture')
   if sys.argv[1]=='library':
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    app.post_message(NavigateToScreen('library', {'mode':'notes'}))
    async with asyncio.timeout(90):
     while not isinstance(app.screen,LibraryScreen) or not app.screen._library_loaded:await asyncio.sleep(.05)
    print('MOUNTED_LIBRARY_LOADED',flush=True)
   if sys.argv[1] in {'settings','library'}:
    print('MOUNTED_BEFORE_SETTINGS',flush=True)
    await pilot.press('f4')
    print('MOUNTED_AFTER_SETTINGS',flush=True)
    await asyncio.sleep(1)
   destination=Path.home()/'mounted.tldw-backup.zip'
   options={'staging_parent':Path.home()}
   print('MOUNTED_BEFORE_PREVIEW',flush=True)
   inventory=await asyncio.to_thread(service.preview_backup,(selector,),options=options)
   print('MOUNTED_AFTER_PREVIEW',inventory.complete,inventory.issues,flush=True)
   assert inventory.complete,inventory.issues
   print('MOUNTED_BEFORE_START',flush=True)
   operation=service.start_backup((selector,),inventory.scope_digest,destination,options=options,password=None)
   print('MOUNTED_BEFORE_WAIT',flush=True)
   stop_loop=observe_loop_profile(Path.home()/'mounted-loop-profile.log',delay=2,duration=5)
   try:result=await asyncio.to_thread(service.wait,operation,timeout=240)
   except BaseException as error:
    record_failure(Path.home()/'mounted-wait-failure.json.log',error=error)
    raise
   finally:
    stop_observer(stop_loop)
    state=service.status(operation)
    print('MOUNTED_AFTER_WAIT',state['state'],state['phase'],tuple(state['issues']),flush=True)
   assert result['state']=='succeeded',dict(result)
   assert result['result']['complete']
   async with asyncio.timeout(60):
    while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.05)
   after=app.chachanotes_db.add_note('Resumed','Saved after mounted capture')
   assert app.chachanotes_db.get_note_by_id(before)['content']=='Saved before mounted capture'
   assert app.chachanotes_db.get_note_by_id(after)['content']=='Saved after mounted capture'
   acquired=await asyncio.to_thread(archive_reader.acquire,destination,Path.home()/'readback',ArchiveLimits(),None,threading.Event())
   doc=json.loads(acquired.manifest_bytes)
   assert doc['consistency']=='coherent'
   member=next(row for row in doc['files'] if row['owner_id']=='db.chachanotes.primary')
   with zipfile.ZipFile(acquired.path) as archive:
    saved=archive.read(member['payload'])
   assert b'Saved before mounted capture' in saved
   assert b'Saved after mounted capture' not in saved
 except BaseException as error:
  record_failure(Path.home()/'mounted-body-failure.json.log',error=error)
  raise
 finally:
  phase('service_close_begin')
  await asyncio.to_thread(service.close)
  phase('service_close_complete')
 assert not blocked_attempts(),blocked_attempts()
try:asyncio.run(main())
except BaseException as error:
 record_failure(Path.home()/'mounted-child-failure.json.log',error=error)
 raise
finally:stop_observer(diagnostics.close)
print('retired and reopened')
'''


@pytest.mark.parametrize("screen", ["console", "settings", "library"])
def test_mounted_console_complete_capture_and_resumed_writes(tmp_path, native_package, screen):
    _run(tmp_path, screen, "mounted", script=_SCRIPT,
         timeout=420 if sys.platform == "win32" else 180,
         installed_package=native_package)
