"""First-run note creation must leave the installed app ready for live backup."""

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414
)
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_FIRST_NOTE = r'''
import asyncio,json,os,sqlite3,sys,threading,zipfile
from contextlib import closing
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
assert not Path(os.environ['TLDW_CONFIG_PATH']).exists()
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard,SetupWizardContainer
from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen
from tldw_chatbook.Backup_Recovery import storage_admission as storage,participants
from textual.widgets import Button,Input,Select,Static
from Tests.Backup_Recovery.thread_diagnostics import observe_threads,observe_recovery_failures
import tldw_chatbook
assert Path(tldw_chatbook.__file__).resolve()==Path(os.environ['TLDW_TEST_INSTALLED_PACKAGE'])/'tldw_chatbook'/'__init__.py'
def record(label,value):
 with (Path.home()/'first-note-state.jsonl').open('a') as output:output.write(json.dumps({'label':label,'value':value})+'\n')
async def until(predicate,seconds=30):
 async with asyncio.timeout(seconds):
  while not predicate():await asyncio.sleep(.05)
async def main():
 app=TldwCli()
 async def no_local_discovery(_config):return ()
 app.console_local_server_discovery=no_local_discovery
 stop_stacks=observe_threads(Path.home()/'first-note-stacks.json',interval=30)
 stop_failures=observe_recovery_failures(Path.home()/'first-note-failures.json')
 try:
  async with app.run_test(size=(160,50)) as pilot:
   await until(lambda:isinstance(app.screen,FirstRunSetupWizard),90)
   wizard=app.screen
   container=wizard.query_one(SetupWizardContainer)
   for step_id in ('provider','model','voice','protect-keys','summary'):
    await pilot.press('ctrl+n')
    await until(lambda:container.steps[container.current_step].config.id==step_id)
   button=wizard.query_one('#setup-exit-library-notes',Button)
   button.scroll_visible(animate=False);await pilot.pause()
   button.focus();await pilot.press('enter')
   await until(lambda:bool(app.screen.query('#library-notes-create-blank')))
   button=app.screen.query_one('#library-notes-create-blank',Button)
   button.scroll_visible(animate=False);await pilot.pause()
   button.focus();await pilot.press('enter')
   await until(lambda:bool(app.screen.query('#library-note-title')))
   app.screen.query_one('#library-note-title',Input).focus()
   await pilot.press('home','ctrl+k',*'First note live backup','tab',*'First note native value.','escape')
   await until(lambda:not app.screen.query('#library-note-body'))
   # A real user can spend longer than the Home cache TTL in setup.
   # Exercise that same refresh deterministically before leaving Library.
   adapter=app.home_active_work_adapter
   adapter._active_work_cache_at=0
   assert await adapter.refresh_active_work_cache_async()
   with storage._lock:
    retained=[{'owner':p.owner_id,'thread':l.resource_thread.name} for p in tuple(participants._installed_repositories) for l in tuple(p.connections.values()) if p.owner_id=='notifications.client' and l.resource_thread is not threading.main_thread()]
   record('post_home_refresh',retained)
   assert not retained,retained
   if sys.argv[2]=='handoff':
    from tldw_chatbook.Widgets.Library.library_note_work_pane import LibraryNoteWorkPane
    from tldw_chatbook.Backup_Recovery.runtime_maintenance import RuntimeMaintenance
    pane=app.screen.query_one(LibraryNoteWorkPane)
    assert pane.mode=='list' and pane.query('#library-note-work-empty')
    saved=app.chachanotes_db.get_note_by_title('First note live backup')
    assert saved and saved['content']=='First note native value.'
    assert not RuntimeMaintenance(app).unsaved_editors()
    # Match the CLI's advertised restart capability; use the real guarded
    # shutdown, stopping at its request receipt before the external CLI exec.
    app._recovery_restart_available=True
    await pilot.press('ctrl+p',*'backup','enter')
    await until(lambda:isinstance(app.screen,BackupRestoreScreen))
    screen=app.screen
    screen.query_one('#backup-open-inspect',Button).focus();await pilot.press('enter')
    from Tests.Backup_Recovery.test_restore_plan import sealed
    archive=sealed(Path.home())
    screen.query_one('#backup-source',Input).value=str(archive.path)
    screen.query_one('#backup-inspect',Button).focus();await pilot.press('enter')
    await until(lambda:screen._inspection_id is not None,60)
    screen.query_one('#backup-restore-mode',Select).value='replace'
    await pilot.pause()
    button=screen.query_one('#backup-restart',Button)
    button.scroll_visible(animate=False);await pilot.pause()
    button.focus();await pilot.press('enter')
    await until(lambda:getattr(app,'_recovery_restart_request',None) is not None,60)
    assert app._shutting_down
    assert app._recovery_restart_request.target_config==Path(os.environ['TLDW_CONFIG_PATH'])
    record('saved_note_recovery_handoff',True)
    return
   await pilot.press('f4','ctrl+p',*'backup','enter')
   await until(lambda:isinstance(app.screen,BackupRestoreScreen))
   screen=app.screen
   screen.query_one('#backup-open-create',Button).focus();await pilot.press('enter')
   screen.query_one('#backup-destination',Input).value=str(Path.home()/'first-note.tldw-backup.zip')
   screen.query_one('#backup-review',Button).focus();await pilot.press('enter')
   await until(lambda:'Complete coverage' in str(screen.query_one('#backup-coverage',Static).render()),60)
   record('complete_review',True)
   screen.query_one('#backup-create',Button).focus();await pilot.press('enter')
   await until(lambda:(state:=app.recovery_service.current()) is not None and state['kind']=='backup' and state['state']!='running',100)
   state=app.recovery_service.current()
   with storage._lock:
    record('terminal',{'state':state['state'],'phase':state['phase'],'issues':list(state['issues']),'pending':len(storage._pending_acquisitions),'operations':len(storage._operations),'raw':len(storage._raw_operations),'leases':[{'owner':p.owner_id,'thread':l.resource_thread.name} for p in tuple(participants._installed_repositories) for l in tuple(p.connections.values())]})
   assert state['state']=='succeeded',dict(state)
   archive_path=Path.home()/'first-note.tldw-backup.zip'
   with zipfile.ZipFile(archive_path) as archived:
    manifest=json.loads(archived.read('manifest.json'))
    row=next(row for row in manifest['files'] if row['owner_id']=='db.chachanotes.primary')
    database_copy=Path.home()/'archived-notes.db'
    database_copy.write_bytes(archived.read(row['payload']))
   with closing(sqlite3.connect(f'file:{database_copy}?mode=ro',uri=True)) as database:
    saved=database.execute('SELECT id,title,content FROM Notes WHERE title=?',('First note live backup',)).fetchone()
   assert saved and saved[2]=='First note native value.',saved
   assert app.chachanotes_db.get_note_by_id(saved[0])['content']==saved[2]
   record('saved_note_readback',True)
   # The setup model step discovers local providers; the network guard
   # blocks those probes, as it does all other network access.
   record('blocked_network_attempt_count',len(blocked_attempts()))
 finally:
  if getattr(app,'_recovery_service',None) is not None:await asyncio.to_thread(app.recovery_service.close)
  stop_failures();stop_stacks()
asyncio.run(main())
print('retired and reopened')
'''


def test_first_run_note_creation_can_capture_live_backup(tmp_path, native_package):
    _run(tmp_path, 'note', 'backup', script=_FIRST_NOTE, timeout=180,
         installed_package=native_package)


def test_saved_closed_library_note_can_handoff_to_recovery(tmp_path, native_package):
    _run(tmp_path, 'note', 'handoff', script=_FIRST_NOTE, timeout=180,
         installed_package=native_package)
