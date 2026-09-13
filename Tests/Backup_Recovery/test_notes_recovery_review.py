"""Local owner review must precede restored Notes pairing and sync claims."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SETUP = r"""
import asyncio, os, sys
from pathlib import Path
from threading import Event
from Tests.network_guard import install, blocked_attempts
install()
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated, select_profile
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import ActivationStore
base = Path.home()
source = base / 'source'
source.mkdir(mode=448)

def config_manifest(doc):
    doc['owners'][0]['owner_id'] = 'config'
    doc['files'][0].update(owner_id='config', logical_id='profile:profile:config', relative_path='config.toml')
    doc['dependency_groups'][0]['members'] = ['profile:profile:config']
archive = sealed(source, mutate=config_manifest, data=b'[general]\nusers_name="original"\n')
dest = base / 'dest'
dest.mkdir(mode=448)
plan = plan_restore(archive, mode='isolated', destinations={'root': dest / 'config', 'profile:profile:paths.data_dir': dest / 'data'}, target=None, profile_names={'profile': 'recovered'})
control = base / 'control'
profile = restore_isolated(archive, plan, control, Event())
select_profile(profile, control)
config, data = ProfileCatalog(control).resolve(profile)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_service import FileNotesService
from tldw_chatbook.Backup_Recovery.isolated_restore import installation_client_id
folder = data / 'external'
folder.mkdir()
(folder / 'one.md').write_text('# Local file\nactual content\n')
db = CharactersRAGDB(data / 'notes.db', installation_client_id())
replica = FileNotesReplica(data / 'replica.db')
files = FileNotesService(folder, replica)
_, profiles, _ = bootstrap._control_records(bootstrap.default_bootstrap_root())
witness = next((p['activation'] for p in profiles if p['selector'] == str(config)))
store = ActivationStore(Path(witness['store_root']))

def assert_unrelated_owners_inactive():
    for other in ('config', 'db.chachanotes.primary', 'skills', 'runtime.sync_state', 'db.scheduled_tasks'):
        assert other in witness['owners'], other
        assert not store.allowed(witness['generation'], other), other
assert_unrelated_owners_inactive()
"""

_MANUAL = (
    _SETUP
    + r"""
owner = 'notes.file_notes'
store.approve(witness['generation'], owner)
before = (folder / 'one.md').read_bytes()
result = files.reconcile()
assert result.replica_warning and (not replica.list_active_files(files.root_key)), 'global flag refreshed replica without fresh pairing'
assert (folder / 'one.md').read_bytes() == before
assert not blocked_attempts()
db.close_connection()
replica.close()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["file_notes"])
def test_owner_flag_alone_cannot_authorize_restored_pairing(tmp_path, route):
    _run(tmp_path, route, "manual", script=_MANUAL)


_REVIEW = (
    _SETUP
    + r"""
route, action = sys.argv[1:]
owner = 'notes.file_notes'

def preview():
    return files.preview_recovery()

def approve(review):
    return files.approve_recovery(review)

def database_state():
    return (db.get_connection().total_changes, tuple(tuple(row) for row in db.get_connection().execute('SELECT * FROM sync_sessions')), tuple(replica.list_active_files(files.root_key)))
before = database_state()
disk = (folder / 'one.md').read_bytes()
review = preview()
assert review.entries == (('one.md', 'disk_only'),), review
assert not review.issues
assert database_state() == before and (folder / 'one.md').read_bytes() == disk
assert not store.allowed(witness['generation'], owner)
if action == 'changed':
    (folder / 'one.md').write_text('changed after preview')
    try:
        approve(review)
    except ValueError as error:
        assert error.args == ('notes_pairing_review_changed',), error
    else:
        raise AssertionError('stale review approved')
    assert not store.allowed(witness['generation'], owner)
elif action == 'incomplete':
    unreadable = folder / 'unreadable.md'
    unreadable.write_text('owned inaccessible bytes')
    unreadable.chmod(0)
    changed = preview()
    assert changed.issues
    try:
        approve(changed)
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete dry-run approved')
    assert not store.allowed(witness['generation'], owner)
else:
    approve(review)
    assert store.allowed(witness['generation'], owner)
    other_notes = 'notes.sync_bindings'
    assert not store.allowed(witness['generation'], other_notes)
    assert files.reconcile().created == ('one.md',)
    assert replica.get_bytes(files.root_key, 'one.md') == disk
    assert (folder / 'one.md').read_bytes() == disk
assert_unrelated_owners_inactive()
assert not blocked_attempts()
db.close_connection()
replica.close()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["file_notes"])
@pytest.mark.parametrize("action", ["approve", "changed", "incomplete"])
def test_complete_owner_dry_run_and_explicit_current_approval(tmp_path, route, action):
    _run(tmp_path, route, action, script=_REVIEW)


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
import asyncio, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile, installation_client_id
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
profile, control, route = sys.argv[1:]
control = Path(control)
select_profile(profile, control)
_, data = ProfileCatalog(control).resolve(profile)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_service import FileNotesService
folder = data / 'external'
db = CharactersRAGDB(data / 'notes.db', installation_client_id())
replica = FileNotesReplica(data / 'replica.db')
files = FileNotesService(folder, replica)
(folder / 'fresh.md').write_text('Created after a fresh launch')
assert files.reconcile().created == ('fresh.md',)
assert not blocked_attempts()
db.close_connection()
replica.close()
print('fresh process consumed local owner claim')
"""


@pytest.mark.parametrize("route", ["file_notes"])
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
import asyncio,sys
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
from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore
from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner
from tldw_chatbook.Notes.notes_sync_legacy import snapshot_legacy_notes_sync,plan_legacy_notes_sync_migration,persist_legacy_notes_sync_migration
notes=CharactersRAGDB(config.get_chachanotes_db_path(),installation_client_id())
def history():
 connection=notes.get_connection()
 return tuple(tuple(tuple(row) for row in connection.execute('SELECT * FROM '+table)) for table in ('notes','sync_sessions','sync_conflicts','note_folder_memberships'))
before=history()
assert notes.get_connection().execute('SELECT session_id FROM sync_sessions').fetchone()[0]=='old-running-session'
folder=data/'new-local-notes';folder.mkdir();(folder/'new.md').write_text('Fresh local content remains untouched')
disk=(folder/'new.md').read_bytes()
_,profiles,_=bootstrap._control_records(bootstrap.default_bootstrap_root())
witness=next(p['activation'] for p in profiles if p['selector']==str(selector))
activation=ActivationStore(Path(witness['store_root']))
device_path=config.get_user_data_dir()/'tldw_chatbook_notes_sync_state.db'
assert not device_path.exists(),'restoration copied device-local authority'
device=NotesDeviceStateStore(device_path)
def migrate():
 try:
  snapshot=snapshot_legacy_notes_sync(notes.get_connection(),{},note_scope_id='recovered-notes')
  return persist_legacy_notes_sync_migration(device,plan_legacy_notes_sync_migration(snapshot))
 finally:notes.close_connection()
def denied(*args,**kwargs):raise AssertionError('historical metadata started live device sync')
runtime=NotesSyncRuntimeOwner(store=device,migrate_legacy=migrate,coordinator=denied,adapter=object(),watcher_factory=denied,cutover_admitted=False,profile_process_is_sole=True)
async def check():
 try:
  await runtime.start()
  state=runtime.snapshot()
  assert state.status=='awaiting_cutover',state
  assert not state.roots,state
  assert device.get_setting('cutover_marker') is None
  assert not device.list_incomplete_operations()
 finally:await runtime.shutdown()
asyncio.run(check())
assert history()==before,'starting the current runtime changed restored historical content'
assert (folder/'new.md').read_bytes()==disk
for owner in ('config','db.chachanotes.primary','notes.sync_bindings','notes.file_notes'):
 assert not activation.allowed(witness['generation'],owner),owner
notes.close();device.close();assert not blocked_attempts()
print('actual archived Notes history remains inert without device authority')
"""


def test_actual_archived_notes_history_stays_inert_without_device_authority(tmp_path):
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
        "actual archived Notes history remains inert without device authority"
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
 store=ActivationStore(Path(prior['store_root']))
 assert not store.allowed(prior['generation'],'config')
 from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
 from tldw_chatbook.Notes.file_notes_service import FileNotesService
 replica=FileNotesReplica(selector.parent/'replica.db');files=FileNotesService(folder,replica)
 files.approve_recovery(files.preview_recovery())
 assert not store.allowed(prior['generation'],'config')
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
 current_store.approve(current['generation'],'notes.file_notes')
 assert not current_store.allowed(current['generation'],'config')
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


def test_retired_sync_pairing_cannot_issue_new_device_authority(tmp_path):
    from tldw_chatbook.Notes.recovery_review import review_pairing

    with pytest.raises(ValueError, match="notes_pairing_owner_unsupported"):
        review_pairing("notes.sync_bindings", object(), tmp_path, "test")
