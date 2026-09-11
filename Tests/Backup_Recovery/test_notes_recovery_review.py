"""Local owner review must precede restored Notes pairing and sync claims."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SETUP = r"""
import asyncio,os,sys
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,select_profile
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import ActivationStore
base=Path.home();source=base/'source';source.mkdir(mode=0o700)
def config_manifest(doc):
 doc['owners'][0]['owner_id']='config'
 doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
 doc['dependency_groups'][0]['members']=['profile:profile:config']
archive=sealed(source,mutate=config_manifest,data=b'[general]\nusers_name="original"\n')
dest=base/'dest';dest.mkdir(mode=0o700)
plan=plan_restore(archive,mode='isolated',destinations={'root':dest/'config','profile:profile:paths.data_dir':dest/'data'},target=None,profile_names={'profile':'recovered'})
control=base/'control';profile=restore_isolated(archive,plan,control,Event())
select_profile(profile,control)
config,data=ProfileCatalog(control).resolve(profile)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.sync_service import NotesSyncService
from tldw_chatbook.Notes.sync_engine import SyncDirection
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_service import FileNotesService
from tldw_chatbook.Backup_Recovery.isolated_restore import installation_client_id
folder=data/'external';folder.mkdir()
(folder/'one.md').write_text('# Local file\nactual content\n')
db=CharactersRAGDB(data/'notes.db',installation_client_id())
notes=NotesInteropService(data,installation_client_id(),global_db_to_use=db)
sync=NotesSyncService(notes,db)
replica=FileNotesReplica(data/'replica.db');files=FileNotesService(folder,replica)
_,profiles,_=bootstrap._control_records(bootstrap.default_bootstrap_root())
witness=next(p['activation'] for p in profiles if p['selector']==str(config))
store=ActivationStore(Path(witness['store_root']))
store.approve(witness['generation'],'config')
store.approve(witness['generation'],'db.chachanotes.primary')
"""

_MANUAL = (
    _SETUP
    + r"""
owner='notes.sync_bindings' if sys.argv[1]=='sync' else 'notes.file_notes'
store.approve(witness['generation'],owner)
before=(folder/'one.md').read_bytes()
if sys.argv[1]=='sync':
 try:asyncio.run(sync.sync_folder(folder,'test',SyncDirection.DISK_TO_DB))
 except PermissionError:pass
 else:raise AssertionError('global owner flag reused historical root authority')
 assert not sync.get_sync_history()
else:
 result=files.reconcile()
 assert result.replica_warning and not replica.list_active_files(files.root_key),'global flag refreshed replica without fresh pairing'
assert (folder/'one.md').read_bytes()==before
assert not blocked_attempts()
db.close_connection();replica.close()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["sync", "file_notes"])
def test_owner_flag_alone_cannot_authorize_restored_pairing(tmp_path, route):
    _run(tmp_path, route, "manual", script=_MANUAL)


_REVIEW = (
    _SETUP
    + r"""
route,action=sys.argv[1:]
owner='notes.sync_bindings' if route=='sync' else 'notes.file_notes'
def preview():
 return sync.preview_recovery(folder,'test') if route=='sync' else files.preview_recovery()
def approve(review):
 return sync.approve_recovery(review,'test') if route=='sync' else files.approve_recovery(review)
def database_state():
 return (db.get_connection().total_changes,tuple(sync.get_sync_history()),tuple(replica.list_active_files(files.root_key)))
before=database_state();disk=(folder/'one.md').read_bytes()
review=preview()
assert review.entries==(('one.md','disk_only'),),review
assert not review.issues
assert database_state()==before and (folder/'one.md').read_bytes()==disk
assert not store.allowed(witness['generation'],owner)
if action=='changed':
 (folder/'one.md').write_text('changed after preview')
 try:approve(review)
 except ValueError as error:assert error.args==('notes_pairing_review_changed',),error
 else:raise AssertionError('stale review approved')
 assert not store.allowed(witness['generation'],owner)
elif action=='incomplete':
 if route=='sync':(folder/'alias.md').symlink_to(folder/'one.md')
 else:
  unreadable=folder/'unreadable.md';unreadable.write_text('owned inaccessible bytes');unreadable.chmod(0)
 changed=preview()
 assert changed.issues
 try:approve(changed)
 except ValueError:pass
 else:raise AssertionError('incomplete dry-run approved')
 assert not store.allowed(witness['generation'],owner)
else:
 approve(review)
 assert store.allowed(witness['generation'],owner)
 assert not store.allowed(witness['generation'],'sync.state')
 if route=='sync':
  session,progress=asyncio.run(sync.sync_folder(folder,'test',SyncDirection.DISK_TO_DB))
  assert progress.created_notes and ':' in session
  assert notes.get_note_by_id('test',progress.created_notes[0])['content']==disk.decode()
 else:
  assert files.reconcile().created==('one.md',)
  assert replica.get_bytes(files.root_key,'one.md')==disk
 assert (folder/'one.md').read_bytes()==disk
assert not blocked_attempts()
db.close_connection();replica.close()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["sync", "file_notes"])
@pytest.mark.parametrize("action", ["approve", "changed", "incomplete"])
def test_complete_owner_dry_run_and_explicit_current_approval(tmp_path, route, action):
    _run(tmp_path, route, action, script=_REVIEW)


_HISTORY = (
    _SETUP
    + r"""
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
repo=LocalNoteFolderRepository(db)
note=db.add_note('Historical membership','kept content')
placement=repo.create_folder(name='Historical',parent_id=None)
repo.reconcile_managed(owner_id='captured-device-claim',desired=((placement.folder_id,note),))
with db.transaction() as connection:
 connection.execute("INSERT INTO sync_sessions(session_id,sync_root_folder,sync_direction,conflict_resolution,status,client_id) VALUES(?,?,?,?,?,?)",('captured-session',str(folder),'disk_to_db','ask','running','old-device'))
 conflict=connection.execute("INSERT INTO sync_conflicts(session_id,file_path,conflict_type) VALUES(?,?,?)",('captured-session',str(folder/'one.md'),'both_changed')).lastrowid
before=tuple(tuple(row) for row in db.get_connection().execute('SELECT * FROM note_folder_memberships'))
review=sync.preview_recovery(folder,'test')
assert review.historical_owners==('captured-device-claim',)
assert before==tuple(tuple(row) for row in db.get_connection().execute('SELECT * FROM note_folder_memberships'))
sync.approve_recovery(review,'test')
assert not sync.resolve_conflict(conflict,'use_disk','test'),'captured running session became live'
assert db.get_connection().execute('SELECT resolution FROM sync_conflicts WHERE id=?',(conflict,)).fetchone()[0] is None
row=db.get_connection().execute('SELECT owner_id,ownership,owner_active,deleted FROM note_folder_memberships').fetchone()
assert tuple(row)==('captured-device-claim','managed',0,0),tuple(row)
assert repo.list_restore_reviews()[0].owner_id=='captured-device-claim'
assert db.get_note_by_id(note)['content']=='kept content'
assert not blocked_attempts()
db.close_connection();replica.close()
print('retired and reopened')
"""
)


def test_fresh_pairing_does_not_resume_historical_conflicts_or_memberships(tmp_path):
    _run(tmp_path, "sync", "history", script=_HISTORY)


_IMPORT = r"""
import builtins,sqlite3,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
def denied(*args,**kwargs):raise AssertionError('generation import opened a database')
sqlite3.connect=denied
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if any(name==blocked or name.startswith(blocked+'.') for blocked in ('tldw_chatbook.config','tldw_chatbook.app','tldw_chatbook.RAG_Search','chromadb','sentence_transformers','transformers')):
  raise AssertionError('passive reader attempted runtime import: '+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
from tldw_chatbook.Backup_Recovery.generation_witnesses import _witnesses
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
source=Path.home()/'actual';source.mkdir(mode=0o700)
with acquire_storage(source) as lease:assert _witnesses(source,lease)==[]
assert not blocked_attempts()
assert 'tldw_chatbook.config' not in sys.modules
assert 'tldw_chatbook.RAG_Search' not in sys.modules
print('retired and reopened')
"""


def test_existing_admitted_generation_reader_import_is_inert(tmp_path):
    _run(tmp_path, "import", "inert", script=_IMPORT)


_REOPEN = r"""
import asyncio,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,installation_client_id
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
profile,control,route=sys.argv[1:]
control=Path(control);select_profile(profile,control)
_,data=ProfileCatalog(control).resolve(profile)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.sync_service import NotesSyncService
from tldw_chatbook.Notes.sync_engine import SyncDirection
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_service import FileNotesService
folder=data/'external'
db=CharactersRAGDB(data/'notes.db',installation_client_id())
notes=NotesInteropService(data,installation_client_id(),global_db_to_use=db)
sync=NotesSyncService(notes,db)
replica=FileNotesReplica(data/'replica.db');files=FileNotesService(folder,replica)
(folder/'fresh.md').write_text('Created after a fresh launch')
if route=='sync':
 session,progress=asyncio.run(sync.sync_folder(folder,'test',SyncDirection.DISK_TO_DB))
 assert ':' in session and progress.created_notes
else:
 assert files.reconcile().created==('fresh.md',)
assert not blocked_attempts()
db.close_connection();replica.close()
print('fresh process consumed local owner claim')
"""


@pytest.mark.parametrize("route", ["sync", "file_notes"])
def test_owner_claim_reopens_in_fresh_process(tmp_path, route):
    script = _REVIEW.replace(
        "print('retired and reopened')",
        "import subprocess\n"
        "result=subprocess.run([sys.executable,'-c',CHILD,profile,str(control),route],"
        "env=os.environ.copy(),capture_output=True,text=True,timeout=25)\n"
        "assert result.returncode==0,result.stderr[-6000:]\n"
        "assert 'fresh process consumed local owner claim' in result.stdout\n"
        "print('retired and reopened')",
    )
    _run(tmp_path, route, "approve", script="CHILD=" + repr(_REOPEN) + "\n" + script)


from Tests.Backup_Recovery.test_activation_notes import _CANCEL_SCRIPT

_CANCEL = (
    _SETUP
    + r"""
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
authority=admission_authority(bootstrap.default_bootstrap_root())
files.approve_recovery(files.preview_recovery())
db.close_connection();replica.close()
from tldw_chatbook.Backup_Recovery import storage_admission as storage
startup=storage._startups.pop((os.getpid(),str(bootstrap.default_bootstrap_root())),None)
if startup is not None:startup.close()
import threading
"""
    + _CANCEL_SCRIPT.split("import threading\n", 1)[1].replace(
        "authority.maintenance(('profile',)",
        "authority.maintenance(tuple(witness['namespaces'])",
    )
)


def test_actual_reviewed_cancelled_waiter_keeps_native_reconcile_admitted(tmp_path):
    _run(tmp_path, "cancel", "approved", script=_CANCEL)


_IDENTITY = (
    _SETUP
    + r"""
from tldw_chatbook.Backup_Recovery.journal import Journal
journal=Journal(control,witness['operation_id'])
journal.root.rename(journal.root.with_name('retained-original-journal'))
try:files.preview_recovery()
except (ValueError,PermissionError):pass
else:raise AssertionError('cached installation identity bypassed lost committed evidence')
assert not store.allowed(witness['generation'],'notes.file_notes')
db.close_connection();replica.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_cached_installation_id_requires_current_committed_identity(tmp_path):
    _run(tmp_path, "file_notes", "identity", script=_IDENTITY)


_DUPLICATE = (
    _SETUP
    + r"""
for title in ('first','second'):
 note=db.add_note(title,'database content')
 with db.transaction() as connection:
  connection.execute('UPDATE notes SET sync_root_folder=?,relative_file_path_on_disk=?,is_externally_synced=1,last_synced_disk_file_hash=? WHERE id=?',(str(folder),'one.md','0'*64,note))
review=sync.preview_recovery(folder,'test')
assert 'duplicate_sync_path' in review.issues,review
try:sync.approve_recovery(review,'test')
except ValueError:pass
else:raise AssertionError('ambiguous DB claims approved')
assert not store.allowed(witness['generation'],'notes.sync_bindings')
db.close_connection();replica.close()
print('retired and reopened')
"""
)


def test_duplicate_database_path_cannot_be_a_complete_dry_run(tmp_path):
    _run(tmp_path, "sync", "duplicate", script=_DUPLICATE)


_MATRIX = (
    _SETUP
    + r"""
(folder/'two.md').write_text('both changed to same bytes')
for name,content in (('two.md','both changed to same bytes'),('missing.md','stored note')):
 note=db.add_note(name,content)
 with db.transaction() as connection:
  connection.execute('UPDATE notes SET sync_root_folder=?,relative_file_path_on_disk=?,is_externally_synced=1,last_synced_disk_file_hash=? WHERE id=?',(str(folder),name,'0'*64,note))
review=sync.preview_recovery(folder,'test')
assert review.entries==( ('missing.md','deleted_on_disk'),('one.md','disk_only'),('two.md','both_changed') ),review
sync.approve_recovery(review,'test')
_,progress=asyncio.run(sync.sync_folder(folder,'test'))
assert sorted(conflict.conflict_type for conflict in progress.conflicts)==['both_changed','deleted_on_disk']
assert not (folder/'missing.md').exists()
assert (folder/'two.md').read_text()=='both changed to same bytes'
db.close_connection();replica.close()
print('retired and reopened')
"""
)


def test_dry_run_reports_actual_bidirectional_conflicts(tmp_path):
    _run(tmp_path, "sync", "matrix", script=_MATRIX)


_IMPORTED_SEED = r"""
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
repository=LocalNoteFolderRepository(notes)
folder=repository.create_folder(name='Captured placement',parent_id=None)
repository.reconcile_managed(owner_id='captured-machine',desired=((folder.folder_id,note),))
with notes.transaction() as connection:
 connection.execute("INSERT INTO sync_sessions(session_id,sync_root_folder,sync_direction,conflict_resolution,status,client_id) VALUES(?,?,?,?,?,?)",('old-running-session','/old/machine/notes','bidirectional','ask','running','captured-machine'))
 connection.execute("INSERT INTO sync_conflicts(session_id,file_path,conflict_type) VALUES(?,?,?)",('old-running-session','old.md','both_changed'))
"""

_IMPORTED_READ = r"""
import asyncio,json,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,installation_client_id
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import ActivationStore
profile,control=sys.argv[1:];control=Path(control)
select_profile(profile,control);selector,data=ProfileCatalog(control).resolve(profile)
from tldw_chatbook import config
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.sync_service import NotesSyncService
from tldw_chatbook.Notes.sync_engine import SyncDirection
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
notes=CharactersRAGDB(config.get_chachanotes_db_path(),installation_client_id())
service=NotesSyncService(NotesInteropService(data,installation_client_id(),global_db_to_use=notes),notes)
assert service.get_sync_history()[0]['session_id']=='old-running-session'
conflict=service.get_conflicts_for_session('old-running-session')[0]['id']
folder=data/'new-local-notes';folder.mkdir();(folder/'new.md').write_text('Fresh local explicit sync')
_,profiles,_=bootstrap._control_records(bootstrap.default_bootstrap_root())
witness=next(p['activation'] for p in profiles if p['selector']==str(selector))
store=ActivationStore(Path(witness['store_root']))
for owner in ('config','db.chachanotes.primary'):store.approve(witness['generation'],owner)
review=service.preview_recovery(folder,'test')
assert review.historical_owners==('captured-machine',),review
assert not service.resolve_conflict(conflict,'use_disk','test')
service.approve_recovery(review,'test')
assert not service.resolve_conflict(conflict,'use_disk','test')
assert LocalNoteFolderRepository(notes).list_restore_reviews()[0].owner_id=='captured-machine'
session,progress=asyncio.run(service.sync_folder(folder,'test',SyncDirection.DISK_TO_DB))
assert ':' in session and progress.created_notes
row=notes.get_connection().execute('SELECT status,client_id FROM sync_sessions WHERE session_id=?',('old-running-session',)).fetchone()
assert tuple(row)==('running','captured-machine')
notes.close();assert not blocked_attempts()
print('actual archived Notes history remains inert after owner review')
"""


def test_actual_archived_notes_history_is_preserved_through_owner_review(tmp_path):
    import json
    import os
    import subprocess
    import sys

    from Tests.Backup_Recovery.test_isolated_restore import _DATA_RESTORE, _SEED

    seed = _SEED.replace(
        "notes.close();media.close()", _IMPORTED_SEED + "\nnotes.close();media.close()"
    )
    _run(
        tmp_path, "data", "isolated", script="SEED=" + repr(seed) + "\n" + _DATA_RESTORE
    )
    descriptor = json.loads((tmp_path / "home/launch.json").read_text())
    environment = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(tmp_path / "config/config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _IMPORTED_READ,
            descriptor["profile"],
            descriptor["control"],
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr[-6000:] + result.stdout[-1000:]
    assert (
        "actual archived Notes history remains inert after owner review"
        in result.stdout
    )


_BOUNDARY = (
    _SETUP
    + r"""
import json
files.approve_recovery(files.preview_recovery())
if sys.argv[1]=='other_root':
 other=data/'other-root';other.mkdir();(other/'other.md').write_text('Other root bytes')
 checked=FileNotesService(other,replica)
elif sys.argv[1]=='pending_history':
 files._pending_replica_moves['one.md']='other.md'
 review=files.preview_recovery()
 assert 'notes_pairing_pending_history' in review.issues
 try:files.approve_recovery(review)
 except ValueError:pass
 else:raise AssertionError('pending prior filesystem intent approved')
 checked=None
else:
 record=next(store._generation(witness['generation']).glob('notes-pairing-*.json'))
 if sys.argv[1]=='missing':record.rename(record.with_suffix('.retained'))
 else:
  value=json.loads(record.read_text());value['generation']='different-generation'
  record.write_text(json.dumps(value))
 checked=files
if checked is not None:
 result=checked.reconcile()
 assert result.replica_warning and not replica.list_active_files(checked.root_key)
assert (folder/'one.md').read_text()=='# Local file\nactual content\n'
db.close_connection();replica.close();assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize(
    "case", ["other_root", "missing", "generation", "pending_history"]
)
def test_owner_pairing_cannot_expand_or_reuse_changed_local_evidence(tmp_path, case):
    _run(tmp_path, case, "boundary", script=_BOUNDARY)


_RAW_FILES = (
    _SETUP
    + r"""
(folder/'raw.md').write_bytes(b'\xffretained raw file bytes')
(folder/'.git').mkdir();(folder/'.git'/'hidden.md').write_bytes(b'not File Notes data')
(folder/'alias.md').symlink_to(folder/'one.md')
review=files.preview_recovery()
assert review.entries==( ('one.md','disk_only'),('raw.md','disk_only') ),review
assert not review.issues
files.approve_recovery(review)
result=files.reconcile()
assert result.created==('one.md','raw.md'),result
assert replica.get_bytes(files.root_key,'raw.md')==b'\xffretained raw file bytes'
assert not replica.get_bytes(files.root_key,'.git/hidden.md')
assert not replica.get_bytes(files.root_key,'alias.md')
db.close_connection();replica.close();assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_file_notes_review_uses_owned_raw_members_and_git_exclusions(tmp_path):
    _run(tmp_path, "file_notes", "raw", script=_RAW_FILES)


_SPLIT = (
    _SETUP
    + r"""
other=CharactersRAGDB(data/'different-content.db',installation_client_id())
notes.unified_db_template=other
review=sync.preview_recovery(folder,'test')
assert 'notes_pairing_database_mismatch' in review.issues,review
try:sync.approve_recovery(review,'test')
except ValueError:pass
else:raise AssertionError('uncompared distinct content database approved')
assert not store.allowed(witness['generation'],'notes.sync_bindings')
other.close();db.close_connection();replica.close()
print('retired and reopened')
"""
)


def test_separate_content_database_does_not_receive_incomplete_pairing_approval(
    tmp_path,
):
    _run(tmp_path, "sync", "split", script=_SPLIT)


_LATER_GENERATION = r"""
import os,sys
from pathlib import Path
from threading import Event
from pytest import MonkeyPatch
from Tests.network_guard import install,blocked_attempts
install()
from dataclasses import replace
from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
from tldw_chatbook.Backup_Recovery import bootstrap,recovery_copies,replacement,credentials,crypto
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
from tldw_chatbook.Backup_Recovery.activation import ActivationStore
from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
base=Path.home()/'case';base.mkdir(mode=0o700)
with MonkeyPatch.context() as patch, replacement_case(base,patch,prepared=False) as case:
 credential_store=KeyringServerCredentialStore(keyring_backend=FakeKeyring())
 patch.setattr(credentials,'_credential_store',lambda:credential_store)
 patch.setattr(crypto,'_package_resource_root',lambda:Path(sys.argv[1]))
 operation=replacement.replace(replace(case[1],acknowledged_credential_issues=('credential_format_unreadable',)),case[0],control_root=base/'control',rollback_password=b'original',cancel=Event())
 selector=case[-1]
 (selector.parent/'data').mkdir(mode=0o700)
 folder=selector.parent/'manual-notes';folder.mkdir(mode=0o700)
 (folder/'note.md').write_text('Unchanged live File Notes bytes')
 _,profiles,_=bootstrap._control_records(bootstrap.default_bootstrap_root())
 prior=next(p['activation'] for p in profiles if p['selector']==str(selector))
 store=ActivationStore(Path(prior['store_root']));store.approve(prior['generation'],'config')
 from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
 from tldw_chatbook.Notes.file_notes_service import FileNotesService
 replica=FileNotesReplica(selector.parent/'replica.db');files=FileNotesService(folder,replica)
 files.approve_recovery(files.preview_recovery())
 old_claims=[(p.name,p.read_bytes()) for p in store._generation(prior['generation']).glob('notes-pairing-*.json')]
 assert old_claims
 replica.close()
 from tldw_chatbook.Backup_Recovery import storage_admission as storage
 startup=storage._startups.pop((os.getpid(),str(bootstrap.default_bootstrap_root())),None)
 if startup is not None:startup.close()
 import sqlite3
 with sqlite3.connect(case[-2]) as connection:
  connection.execute("UPDATE research_runs SET query='post-restore edit'");connection.commit()
 control=base/'control'
 preview=preview_rollback(operation,control_root=control,old_password=b'original',target=case[1].target,cancel=Event())
 rolled=recovery_copies.rollback(operation,control_root=control,old_password=b'original',new_password=b'new',cancel=Event(),approved_plan=preview)
 assert rolled!=operation
 _,profiles,_=bootstrap._control_records(bootstrap.default_bootstrap_root())
 current=next(p['activation'] for p in profiles if p['selector']==str(selector))
 assert current['generation']!=prior['generation']
 current_store=ActivationStore(Path(current['store_root']))
 assert not current_store.allowed(current['generation'],'notes.file_notes')
 for name,data in old_claims:
  path=current_store._generation(current['generation'])/name;path.write_bytes(data);path.chmod(0o600)
 for owner in ('config','notes.file_notes'):current_store.approve(current['generation'],owner)
 observed=files.reconcile()
 assert observed.replica_warning and not replica.list_active_files(files.root_key)
 assert (folder/'note.md').read_text()=='Unchanged live File Notes bytes'
 replica.close()
assert not blocked_attempts()
print('retired and reopened')
"""


def test_actual_later_rollback_cannot_reuse_prior_owner_claim(
    tmp_path, helper_resource_root
):
    _run(tmp_path, str(helper_resource_root), "later", script=_LATER_GENERATION)


_REJECTED_CLAIM = (
    _SETUP
    + r"""
import json
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
sync.approve_recovery(sync.preview_recovery(folder,'test'),'test')
repo=LocalNoteFolderRepository(db)
note=db.add_note('Historical membership','unchanged')
placement=repo.create_folder(name='Historical',parent_id=None)
repo.reconcile_managed(owner_id='captured-device',desired=((placement.folder_id,note),))
claim=next(store._generation(witness['generation']).glob('notes-pairing-*.json'))
record=json.loads(claim.read_bytes());record['generation']='foreign-generation';claim.write_text(json.dumps(record))
review=sync.preview_recovery(folder,'test')
before=tuple(tuple(row) for row in db.get_connection().execute('SELECT * FROM note_folder_memberships'))
try:sync.approve_recovery(review,'test')
except ValueError:pass
else:raise AssertionError('foreign local claim accepted')
after=tuple(tuple(row) for row in db.get_connection().execute('SELECT * FROM note_folder_memberships'))
assert before==after,'rejected local claim mutated historical membership'
db.close_connection();replica.close();assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_rejected_local_claim_does_not_change_managed_memberships(tmp_path):
    _run(tmp_path, "sync", "rejected_claim", script=_REJECTED_CLAIM)
