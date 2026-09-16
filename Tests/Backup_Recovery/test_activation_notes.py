"""Restored Notes execution stays inactive while local files remain readable."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, os, sys, types
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
sys.modules.setdefault('parakeet_mlx', types.ModuleType('parakeet_mlx'))
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import bind_activation, ActivationStore
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
route, state = sys.argv[1:]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
data = selector.parent.parent / 'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="' + str(data) + '"\n')
selector.chmod(384)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_service import FileNotesService
folder = data / 'external-notes'
folder.mkdir()
note = folder / 'one.md'
note.write_bytes(b'# Preserved\nordinary local content\n')
db = CharactersRAGDB(data / 'notes.db', 'fixture')
replica = FileNotesReplica(data / 'replica.db')
files = FileNotesService(folder, replica)
db.close_connection()
replica.close()
root = bootstrap.default_bootstrap_root()
startup = storage._startups.pop((os.getpid(), str(root)), None)
if startup is not None:
    startup.close()
authority = admission_authority(root)
authority.register('profile', (selector.parent, data))
control = selector.parent.parent / 'operation'
control.mkdir(mode=448)
owners = ('config', 'notes.sync_bindings', 'db.chachanotes.primary', 'notes.file_notes')
if state not in ('ordinary', 'unqualified'):
    register_pending(root, 'restore', ('profile',), control, (selector,))
    with authority.maintenance(('profile',), 2) as session:
        bind_activation(root, 'restore', selector, 'generation', owners, session=session)
    (root / ('pending-' + bootstrap._key('restore') + '.json')).unlink()
    for owner in owners:
        if state in ('owner_flag_only', 'shared') or (state == 'config_only' and owner == 'config'):
            ActivationStore(control / 'activation').approve('generation', owner)
if state == 'shared':
    other = selector.parent.parent / 'other.toml'
    other.write_text('[general]\n')
    os.environ['TLDW_CONFIG_PATH'] = str(other)
if state == 'unqualified':
    storage.qualified_for = lambda *args: (False, 'native_unqualified')
denied = state not in ('ordinary', 'unqualified')
events = []
from tldw_chatbook.DB import private_sqlite_process
helper = str(Path(private_sqlite_process.__file__).with_name('private_sqlite_helper_entry.py').resolve())

def process_sentry(event, args):
    if event == 'subprocess.Popen' and args[1] == [sys.executable, '-I', '-S', helper]:
        assert args[3]['_TLDW_PRIVATE_SQLITE_PARENT_PID'] == str(os.getpid())
        return
    if event in ('subprocess.Popen', 'os.posix_spawn', 'os.system'):
        events.append(event)
        raise AssertionError('unexpected Notes process')
sys.addaudithook(process_sentry)
before = note.read_bytes()
if route in ('scan', 'open', 'reconcile'):
    result = files.open_file('one.md') if route == 'open' else getattr(files, route)()
    if route == 'open':
        assert result.body == before.decode()
    else:
        assert [entry.relative_path for entry in result.entries] == ['one.md']
    if denied:
        assert result.replica_warning and 'activation' in result.replica_warning.lower()
        assert not replica.list_active_files(files.root_key), 'inspection refreshed replica'
        if route == 'reconcile':
            assert not result.created and (not result.modified) and (not result.deleted)
    else:
        assert len(replica.list_active_files(files.root_key)) == 1
assert note.read_bytes() == before
assert not events and (not blocked_attempts())
db.close_connection()
replica.close()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["scan", "open", "reconcile"])
@pytest.mark.parametrize(
    "state",
    ["inactive", "ordinary", "owner_flag_only", "config_only", "shared", "unqualified"],
)
def test_notes_execution_and_safe_inspection(tmp_path, route, state):
    _run(tmp_path, route, state, script=_SCRIPT)


@pytest.mark.parametrize("route", ["scan"])
def test_ordinary_folder_does_not_authorize_another_restored_database(tmp_path, route):
    script = _SCRIPT.replace(
        "folder = data / 'external-notes'",
        "folder = selector.parent.parent / 'ordinary-notes'",
    )
    _run(tmp_path, route, "shared", script=script)


_CANCEL_SCRIPT = (
    _SCRIPT.split("denied =")[0]
    + r"""
import threading
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
entered, release, finished = threading.Event(), threading.Event(), threading.Event()
original = files._walk_candidates
def blocked():
    entered.set()
    assert release.wait(4)
    return original()
files._walk_candidates = blocked
errors = []
def native():
    try:
        assert files.reconcile().created == ('one.md',)
    except BaseException as error:
        errors.append(error)
    finally:
        replica.close()
        finished.set()
async def main():
    waiter = asyncio.create_task(asyncio.to_thread(native))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        waiter.cancel()
        try:
            await waiter
        except asyncio.CancelledError:
            pass
        assert not finished.is_set()
        try:
            with authority.maintenance(('profile',), .03):
                raise AssertionError('cancelled waiter released native reconciliation')
        except AdmissionTimeout:
            pass
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 3)
asyncio.run(main())
assert not errors, errors
with authority.maintenance(('profile',), 1):
    pass
assert len(replica.list_active_files(files.root_key)) == 1
replica.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_cancelled_reconcile_waiter_keeps_native_execution_admitted(tmp_path):
    from Tests.Backup_Recovery.test_notes_recovery_review import _CANCEL

    _run(tmp_path, "cancel", "approved", script=_CANCEL)
