"""A mounted ordinary Console can capture, verify, and resume native storage."""

import sys

import pytest

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414
)
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import asyncio,json,os,sys,threading,zipfile
from pathlib import Path
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
import tldw_chatbook
assert Path(tldw_chatbook.__file__).resolve()==Path(os.environ['TLDW_TEST_INSTALLED_PACKAGE'])/'tldw_chatbook'/'__init__.py'
async def main():
 app=TldwCli()
 service=RecoveryService(default_control_root())
 async def no_local_discovery(_config):return ()
 app.console_local_server_discovery=no_local_discovery
 try:
  async with app.run_test(size=(120,42)) as pilot:
   async with asyncio.timeout(90):
    while not app._ui_ready:await asyncio.sleep(.05)
   async with asyncio.timeout(90):
    while app._boot_worker_gate is None or not app._boot_worker_gate.is_drained:await asyncio.sleep(.05)
   await asyncio.sleep(3)
   before=app.chachanotes_db.add_note('Mounted backup','Saved before mounted capture')
   if sys.argv[1]=='settings':
    await pilot.press('f4')
    await asyncio.sleep(1)
   destination=Path.home()/'mounted.tldw-backup.zip'
   options={'staging_parent':Path.home()}
   inventory=await asyncio.to_thread(service.preview_backup,(selector,),options=options)
   assert inventory.complete,inventory.issues
   operation=service.start_backup((selector,),inventory.scope_digest,destination,options=options,password=None)
   result=await asyncio.to_thread(service.wait,operation,timeout=240)
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
 finally:
  await asyncio.to_thread(service.close)
 assert not blocked_attempts(),blocked_attempts()
asyncio.run(main())
print('retired and reopened')
'''


@pytest.mark.parametrize("screen", ["console", "settings"])
def test_mounted_console_complete_capture_and_resumed_writes(tmp_path, native_package, screen):
    _run(tmp_path, screen, "mounted", script=_SCRIPT,
         timeout=420 if sys.platform == "win32" else 180,
         installed_package=native_package)
