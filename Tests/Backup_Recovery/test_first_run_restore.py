"""A real first-launch Welcome screen can restore an installed native archive."""

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414
)
from Tests.Backup_Recovery.test_f9_replacement_workflow import _SEED
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_NEWCOMER = r"""
import asyncio,json,os,sqlite3,sys,threading
from contextlib import closing
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
selector=Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\nusers_name="default_user"\ndefault_tab="chat"\n[splash_screen]\nenabled=false\n[model_catalog]\nrefresh_consent_recorded=true\nauto_refresh_enabled=false\n')
selector.chmod(0o600)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard
from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen
from tldw_chatbook.Backup_Recovery import storage_admission as storage,participants
from textual.widgets import Button,Input,Static
from Tests.Backup_Recovery.thread_diagnostics import observe_threads,observe_recovery_failures
import tldw_chatbook
assert Path(tldw_chatbook.__file__).resolve()==Path(os.environ['TLDW_TEST_INSTALLED_PACKAGE'])/'tldw_chatbook'/'__init__.py'

def record(label,value):
 with (Path.home()/'first-run-restore-state.jsonl').open('a') as output:output.write(json.dumps({'label':label,'value':value},default=str)+'\n')
# One bounded ownership snapshot; unlike registration stack tracing, this does
# not add work to startup or alter worker acquisition timing.
original_drain=storage._LocalPause.drain
recorded=False
def drain(pause,deadline):
 global recorded
 result=original_drain(pause,deadline)
 if not result and not recorded:
  recorded=True
  with storage._lock:
   record('outstanding',{'pending':len(storage._pending_acquisitions),'operations':len(storage._operations),'raw':len(storage._raw_operations),'leases':[{'owner':participant.owner_id,'thread':lease.resource_thread.name,'startup':lease in storage._startups.values()} for participant in tuple(participants._installed_repositories) for lease in tuple(participant.connections.values())]})
 return result
storage._LocalPause.drain=drain
async def main():
 app=TldwCli()
 async def no_local_discovery(_config):return ()
 app.console_local_server_discovery=no_local_discovery
 stop_stacks=observe_threads(Path.home()/'first-run-stacks.log',interval=30)
 stop_failures=observe_recovery_failures(Path.home()/'first-run-recovery-failures.log')
 try:
  async with app.run_test(size=(120,42)) as pilot:
   async with asyncio.timeout(90):
    while not isinstance(app.screen,FirstRunSetupWizard):await asyncio.sleep(.05)
   wizard=app.screen
   assert app._first_run_startup_action_scheduled
   record('first_launch_wizard',True)
   await pilot.pause()
   button=wizard.query_one('#setup-backup-restore',Button)
   button.scroll_visible(animate=False);await pilot.pause()
   assert await pilot.click(button)
   async with asyncio.timeout(30):
    while not isinstance(app.screen,BackupRestoreScreen):await asyncio.sleep(.05)
   screen=app.screen
   record('recovery_screen',True)
   assert wizard in app.screen_stack
   screen.query_one('#backup-open-inspect',Button).focus();await pilot.press('enter')
   screen.query_one('#backup-source',Input).value=sys.argv[1]
   screen.query_one('#backup-inspect',Button).focus();await pilot.press('enter')
   async with asyncio.timeout(60):
    while not screen.query_one('#backup-restore-form').display:await asyncio.sleep(.03)
   parent=Path.home()/'restore-locations';parent.mkdir(mode=0o700)
   destination=parent/'restored'
   slots=screen._inspection_summary['destination_slots']
   profile_slots=[(index,slot) for index,slot in enumerate(slots) if slot['kind']=='profile_base']
   assert len(profile_slots)==1,slots
   index,slot=profile_slots[0]
   screen.query_one(f'#backup-root-{index}',Input).value=str(destination)
   screen.query_one(f'#backup-profile-name-{index}',Input).value='Restored first launch'
   await pilot.pause()
   screen.query_one('#backup-review-restore',Button).focus();await pilot.press('enter')
   async with asyncio.timeout(45):
    while screen.query_one('#backup-start-restore',Button).disabled:
     text=str(screen.query_one('#backup-restore-preview',Static).render())
     assert 'refused' not in text.lower(),text
     await asyncio.sleep(.03)
   plan=screen._restore_plan
   note_path=next(path for logical_id,path in plan.restore if 'db.chachanotes.primary' in logical_id)
   screen.query_one('#backup-start-restore',Button).focus();await pilot.press('enter')
   async with asyncio.timeout(90):
    while True:
     state=app.recovery_service.current()
     if state and state['kind']=='restore' and state['state']!='running':break
     await asyncio.sleep(.05)
   record('terminal',{'state':state['state'],'phase':state['phase'],'issues':state['issues'],'monitor_error':app._backup_maintenance_error})
   assert state['state']=='succeeded',dict(state)
   assert state['result']['restoration_validated']
   with closing(sqlite3.connect(note_path)) as database:
    assert database.execute('SELECT count(*) FROM notes WHERE content=?',('Captured native value.',)).fetchone()[0]==1
   assert wizard in app.screen_stack
   assert not blocked_attempts(),blocked_attempts()
 finally:
  if getattr(app,'_recovery_service',None) is not None:await asyncio.to_thread(app.recovery_service.close)
  storage._LocalPause.drain=original_drain
  stop_failures();stop_stacks()
asyncio.run(main())
print('retired and reopened')
"""


def test_first_launch_welcome_restores_isolated_native_profile(
    tmp_path, native_package
):
    seed = tmp_path / "seed"
    seed.mkdir(mode=0o700)
    _run(
        seed,
        "seed",
        "archive",
        script=_SEED,
        timeout=180,
        installed_package=native_package,
    )
    newcomer = tmp_path / "newcomer"
    newcomer.mkdir(mode=0o700)
    _run(
        newcomer,
        str(seed / "home/service.tldw-backup.zip"),
        "restore",
        script=_NEWCOMER,
        timeout=180,
        installed_package=native_package,
    )
