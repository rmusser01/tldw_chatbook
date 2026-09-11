"""Mounted Notes controls must consume actual restored-owner pairing reviews."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_notes_recovery_review import _SETUP

_UI_BODY = r"""
from types import SimpleNamespace
from textual.app import App
from textual.widgets import Button, Input, Static
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import LibraryFileNotesWorkspace
route, action = sys.argv[1:]
(folder/'[bold]literal.md').write_text('literal name')
if action=='issues':
 if route=='sync':(folder/'alias.md').symlink_to(folder/'one.md')
 else:
  unreadable=folder/'unreadable.md';unreadable.write_text('owned inaccessible bytes');unreadable.chmod(0)
if action=='history':
 from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
 repository=LocalNoteFolderRepository(db)
 historical=db.add_note('Historical membership','kept content')
 placement=repository.create_folder(name='Historical',parent_id=None)
 repository.reconcile_managed(owner_id='captured-device-claim',desired=((placement.folder_id,historical),))
class SyncScreen(LibraryScreen):
 def compose(self):
  yield LibraryNotesCanvas(mode='sync',sync_panel_state=self._build_library_notes_sync_state())
 def on_mount(self):pass
 async def on_unmount(self):pass
class Host(App):
 def on_mount(self):
  if route=='sync':
   self.subject=SyncScreen(SimpleNamespace(app_config={},notes_service=notes,chachanotes_db=db,notes_user_id='test',notify=self.notify))
   self.subject._library_notes_view='sync'
   self.subject._library_notes_sync_folder_text=str(folder)
   self.subject._library_notes_sync_config_loaded=True
   self.push_screen(self.subject)
  else:
   self.subject=LibraryFileNotesWorkspace(root=folder,replica=replica,poll_interval=3600)
   self.mount(self.subject)
async def wait_for(predicate):
 for _ in range(250):
  if predicate():return
  await asyncio.sleep(.02)
 raise AssertionError('bounded UI checkpoint did not settle')
def data_state():
 return (tuple(sync.get_sync_history()),tuple(replica.list_active_files(files.root_key)),(folder/'one.md').read_bytes())
async def run():
 host=Host()
 async with host.run_test(size=(110,45)) as pilot:
  await wait_for(lambda: hasattr(host,'subject') and (route=='sync' or host.subject.initialized))
  subject=host.subject
  before=data_state()
  selector='#library-notes-sync-review' if route=='sync' else '#file-notes-recovery-review'
  assert subject.query(selector), 'actual owner UI has no recovered pairing review control'
  subject.query_one(selector,Button).press()
  await wait_for(lambda:bool(host.screen.query('#notes-recovery-entries')))
  await pilot.pause()
  rows=host.screen.query_one('#notes-recovery-entries',Static)
  assert 'one.md' in str(rows.content) and 'disk_only' in str(rows.content)
  assert '[bold]literal.md' in str(rows.content)
  assert data_state()==before,'preview changed live notes or replica'
  assert not store.allowed(witness['generation'],'notes.sync_bindings')
  assert not store.allowed(witness['generation'],'notes.file_notes')
  if action=='history':
   assert 'captured-device-claim' in str(host.screen.query_one('#notes-recovery-history',Static).content)
  if action=='issues':
   assert host.screen.query_one('#notes-recovery-approve',Button).disabled
   assert 'None' not in str(host.screen.query_one('#notes-recovery-issues',Static).content)
   await pilot.press('escape')
  elif action in ('root','user','service','navigation'):
   if action=='root':
    other=folder.parent/'other';other.mkdir()
    if route=='sync':subject.query_one('#library-notes-sync-folder',Input).value=str(other)
    else:assert await subject.set_root(other,persist=False)
   elif action=='user':subject.app_instance.notes_user_id='different-user'
   elif action=='service':subject.app_instance.notes_service=NotesInteropService(data,installation_client_id(),global_db_to_use=db)
   elif route=='sync':subject._supersede_library_notes_navigation()
   else:await subject.remove()
   await wait_for(lambda:host.screen.query_one('#notes-recovery-approve',Button).disabled)
   assert 'Selection changed' in str(host.screen.query_one('#notes-recovery-status',Static).content)
   await pilot.press('escape')
  elif action=='changed':
   (folder/'one.md').write_text('changed after dry-run')
   before=data_state()
   host.screen.query_one('#notes-recovery-approve',Button).press()
   await wait_for(lambda:bool(host.screen.query('#confirm-button')))
   host.screen.query_one('#confirm-button',Button).press()
   await wait_for(lambda: bool(host.screen.query('#notes-recovery-status')) and 'not approved' in str(host.screen.query_one('#notes-recovery-status',Static).content))
   assert 'notes_pairing_review_changed' in str(host.screen.query_one('#notes-recovery-status',Static).content)
  elif action=='cancel':
   await pilot.press('escape')
   await pilot.pause()
  else:
   host.screen.query_one('#notes-recovery-approve',Button).press()
   await wait_for(lambda: bool(host.screen.query('#confirm-button')))
   host.screen.query_one('#confirm-button',Button).press()
   owner='notes.sync_bindings' if route=='sync' else 'notes.file_notes'
   await wait_for(lambda:store.allowed(witness['generation'],owner))
   await wait_for(lambda:bool(host.screen.query('#notes-recovery-status')) and 'Pairing approved.' in str(host.screen.query_one('#notes-recovery-status',Static).content))
   assert not store.allowed(witness['generation'],'notes.file_notes' if route=='sync' else 'notes.sync_bindings')
  if action not in ('approve','history'):
   assert not store.allowed(witness['generation'],'notes.sync_bindings')
   assert not store.allowed(witness['generation'],'notes.file_notes')
  assert data_state()==before,'review/approval implicitly performed sync or reconcile'
  if action=='history':
   row=db.get_connection().execute('SELECT owner_id,owner_active,deleted FROM note_folder_memberships').fetchone()
   assert tuple(row)==('captured-device-claim',0,0)
  if action=='approve':
   await pilot.press('escape')
   if route=='sync':
    _,progress=await sync.sync_folder(folder,'test',SyncDirection.DISK_TO_DB)
    assert progress.created_notes
   else:
    await pilot.pause()
    subject.query_one('#file-notes-maintenance-toggle',Button).press()
    await pilot.pause()
    button=subject.query_one('#file-notes-refresh',Button)
    assert button.display
    assert not button.disabled,(subject._path_transitioning,subject._active)
    button.press()
    try:await wait_for(lambda:replica.get_bytes(subject._service.root_key,'one.md')==(folder/'one.md').read_bytes())
    except AssertionError:raise AssertionError((subject._runtime_warning,subject._action_detail,subject._path_transitioning,subject._active,subject._service.root_key))
  if route=='file_notes':await subject.shutdown()
asyncio.run(run())
db.close_connection();replica.close()
assert not blocked_attempts()
print('retired and reopened')
"""
_UI = f"{_SETUP}\n{_UI_BODY}"


@pytest.mark.parametrize("route", ["sync", "file_notes"])
@pytest.mark.parametrize("action", ["cancel", "approve"])
def test_actual_restored_owner_review_is_explicit_and_does_not_sync(
    tmp_path, route, action
):
    _run(tmp_path, route, action, script=_UI)


@pytest.mark.parametrize(
    "route,action",
    [
        ("sync", "issues"),
        ("file_notes", "issues"),
        ("sync", "changed"),
        ("file_notes", "changed"),
        ("sync", "root"),
        ("file_notes", "root"),
        ("sync", "user"),
        ("sync", "service"),
        ("sync", "navigation"),
        ("file_notes", "navigation"),
        ("sync", "history"),
    ],
)
def test_actual_pairing_ui_refuses_incomplete_or_stale_review(tmp_path, route, action):
    _run(tmp_path, route, action, script=_UI)


_CANCEL_UI = "\n".join(
    (
        _UI.split("async def run():", 1)[0],
        r"""
import sqlite3,threading
from tldw_chatbook.Notes.sync_engine import NotesSyncEngine
from tldw_chatbook.Backup_Recovery import storage_admission as storage
entered=threading.Event();release=threading.Event();observed=[]
async def run():
 host=Host()
 async with host.run_test(size=(110,45)) as pilot:
  await wait_for(lambda:hasattr(host,'subject') and (route=='sync' or host.subject.initialized))
  subject=host.subject
  main_connection=db.get_connection()
  if route=='sync':
   original=NotesSyncEngine._get_synced_notes_for_root
   def paused(self,*args,**kwargs):
    result=original(self,*args,**kwargs)
    observed.append((self.db.get_connection(),threading.current_thread()))
    entered.set()
    assert release.wait(10),'worker release not reached'
    return result
   NotesSyncEngine._get_synced_notes_for_root=paused
  else:
   service=subject._service
   original=service._load_file
   def paused(*args,**kwargs):
    result=original(*args,**kwargs)
    if not entered.is_set():
     observed.append((None,threading.current_thread()))
     entered.set()
     assert release.wait(10),'worker release not reached'
    return result
   service._load_file=paused
  selector='#library-notes-sync-review' if route=='sync' else '#file-notes-recovery-review'
  subject.query_one(selector,Button).press()
  await wait_for(entered.is_set)
  group='notes-pairing-review' if route=='sync' else 'file-notes-pairing-review'
  for worker in subject.workers:
   if worker.group==group:worker.cancel()
  await asyncio.sleep(.05)
  if route=='sync':
   assert subject._library_notes_sync_active_token is not None
   assert subject._library_notes_pairing_task and not subject._library_notes_pairing_task.done()
   with storage._lock:
    assert any(lease.resource_thread is observed[0][1] for lease in storage._live_leases)
  else:
   assert subject._path_transitioning
   assert subject._session_owner._maintenance_operations.get(observed[0][1])
   assert not await subject.set_root(folder.parent/'changed',persist=False)
  release.set()
  await wait_for(lambda:subject._library_notes_pairing_task is None if route=='sync' else subject._pairing_review_task is None)
  assert not host.screen.query('#notes-recovery-entries'),'cancelled waiter reattached review'
  assert not store.allowed(witness['generation'],'notes.sync_bindings')
  assert not store.allowed(witness['generation'],'notes.file_notes')
  assert main_connection.execute('SELECT 1').fetchone()[0]==1
  if route=='sync':
   assert subject._library_notes_sync_active_token is None
   with pytest_raises_closed():observed[0][0].execute('SELECT 1')
   NotesSyncEngine._get_synced_notes_for_root=original
  else:
   assert not subject._path_transitioning
   service._load_file=original
   await subject.shutdown()
from contextlib import contextmanager
@contextmanager
def pytest_raises_closed():
 try:yield
 except sqlite3.ProgrammingError as error:assert 'closed' in str(error)
 else:raise AssertionError('new native worker connection retained')
try:asyncio.run(run())
finally:release.set()
db.close_connection();replica.close()
assert not blocked_attempts()
print('retired and reopened')
""",
    )
)


@pytest.mark.parametrize("route", ["sync", "file_notes"])
def test_cancelled_review_waiter_retains_actual_native_operation(tmp_path, route):
    _run(tmp_path, route, "cancel_worker", script=_CANCEL_UI)


_CANCEL_APPROVAL = (
    _CANCEL_UI.replace(
        "entered=threading.Event();release=threading.Event();observed=[]",
        "entered=threading.Event();release=threading.Event();observed=[];enabled=False",
    )
    .replace(
        "    observed.append((self.db.get_connection(),threading.current_thread()))",
        "    if not enabled:return result\n"
        "    observed.append((self.db.get_connection(),threading.current_thread()))",
    )
    .replace(
        "    if not entered.is_set():",
        "    if enabled and not entered.is_set():",
    )
    .replace(
        "async def run():\n host=Host()",
        "async def run():\n global enabled\n host=Host()",
    )
    .replace(
        "  await wait_for(entered.is_set)",
        "  await wait_for(lambda:bool(host.screen.query('#notes-recovery-approve')))\n"
        "  host.screen.query_one('#notes-recovery-approve',Button).press()\n"
        "  await wait_for(lambda:bool(host.screen.query('#confirm-button')))\n"
        "  enabled=True\n"
        "  host.screen.query_one('#confirm-button',Button).press()\n"
        "  await wait_for(entered.is_set)",
    )
    .replace(
        "group='notes-pairing-review' if route=='sync' else 'file-notes-pairing-review'",
        "group='notes-pairing-confirm'",
    )
    .replace(
        "  assert not host.screen.query('#notes-recovery-entries'),'cancelled waiter reattached review'",
        "  assert store.allowed(witness['generation'],'notes.sync_bindings' if route=='sync' else 'notes.file_notes')",
    )
    .replace(
        "  assert not store.allowed(witness['generation'],'notes.sync_bindings')\n"
        "  assert not store.allowed(witness['generation'],'notes.file_notes')",
        "  assert not store.allowed(witness['generation'],'notes.file_notes' if route=='sync' else 'notes.sync_bindings')\n"
        "  assert not replica.list_active_files(files.root_key)\n"
        "  assert not sync.get_sync_history()",
    )
)


@pytest.mark.parametrize("route", ["sync", "file_notes"])
def test_cancelled_approved_waiter_does_not_release_native_pairing_early(
    tmp_path, route
):
    _run(tmp_path, route, "cancel_approval", script=_CANCEL_APPROVAL)


_DB_FAILURE_UI = "\n".join(
    (
        _UI.split("async def run():", 1)[0],
        r"""
async def run():
 host=Host()
 async with host.run_test(size=(110,45)) as pilot:
  await wait_for(lambda:hasattr(host,'subject') and (route=='sync' or host.subject.initialized))
  subject=host.subject
  selector='#library-notes-sync-review' if route=='sync' else '#file-notes-recovery-review'
  if action=='approve_error':
   subject.query_one(selector,Button).press()
   await wait_for(lambda:bool(host.screen.query('#notes-recovery-approve')))
  if route=='sync':
   with db.transaction() as connection:connection.execute('ALTER TABLE notes RENAME TO unavailable_notes')
  else:
   with replica._transaction() as cursor:cursor.execute('ALTER TABLE files RENAME TO unavailable_files')
  if action=='approve_error':
   host.screen.query_one('#notes-recovery-approve',Button).press()
   await wait_for(lambda:bool(host.screen.query('#confirm-button')))
   host.screen.query_one('#confirm-button',Button).press()
   await wait_for(lambda:bool(host.screen.query('#notes-recovery-status')) and 'not approved' in str(host.screen.query_one('#notes-recovery-status',Static).content))
  else:
   subject.query_one(selector,Button).press()
   if route=='sync':await wait_for(lambda:any('Pairing review unavailable' in n.message for n in host._notifications))
   else:await wait_for(lambda:'Pairing review unavailable' in subject._action_detail)
  assert not store.allowed(witness['generation'],'notes.sync_bindings')
  assert not store.allowed(witness['generation'],'notes.file_notes')
  if route=='file_notes':await subject.shutdown()
asyncio.run(run())
db.close_connection();replica.close()
assert not blocked_attempts()
print('retired and reopened')
""",
    )
)


@pytest.mark.parametrize("route", ["sync", "file_notes"])
@pytest.mark.parametrize("action", ["preview_error", "approve_error"])
def test_damaged_owner_database_is_reported_without_crashing_ui(
    tmp_path, route, action
):
    _run(tmp_path, route, action, script=_DB_FAILURE_UI)
