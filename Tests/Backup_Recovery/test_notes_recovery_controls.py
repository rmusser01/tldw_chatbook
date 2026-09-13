"""Mounted Notes controls must consume actual restored-owner pairing reviews."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_notes_recovery_review import _SETUP

_UI = (
    _SETUP
    + r"""
from textual.app import App
from textual.widgets import Button, Static
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import LibraryFileNotesWorkspace
route, action = sys.argv[1:]
(folder / '[bold]literal.md').write_text('literal name')
if action == 'issues':
    unreadable = folder / 'unreadable.md'
    unreadable.write_text('owned inaccessible bytes')
    unreadable.chmod(0)

class Host(App):

    def on_mount(self):
        self.subject = LibraryFileNotesWorkspace(root=folder, replica=replica, poll_interval=3600)
        self.mount(self.subject)

async def wait_for(predicate):
    for _ in range(250):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError('bounded UI checkpoint did not settle')

def data_state():
    return (tuple((tuple(row) for row in db.get_connection().execute('SELECT * FROM sync_sessions'))), tuple(replica.list_active_files(files.root_key)), (folder / 'one.md').read_bytes())

async def run():
    host = Host()
    async with host.run_test(size=(110, 45)) as pilot:
        await wait_for(lambda: hasattr(host, 'subject') and host.subject.initialized)
        subject = host.subject
        before = data_state()
        selector = '#file-notes-recovery-review'
        assert subject.query(selector), 'actual owner UI has no recovered pairing review control'
        subject.query_one(selector, Button).press()
        await wait_for(lambda: bool(host.screen.query('#notes-recovery-entries')))
        await pilot.pause()
        rows = host.screen.query_one('#notes-recovery-entries', Static)
        assert 'one.md' in str(rows.content) and 'disk_only' in str(rows.content)
        assert '[bold]literal.md' in str(rows.content)
        assert data_state() == before, 'preview changed live notes or replica'
        assert not store.allowed(witness['generation'], 'notes.sync_bindings')
        assert not store.allowed(witness['generation'], 'notes.file_notes')
        if action == 'issues':
            assert host.screen.query_one('#notes-recovery-approve', Button).disabled
            assert 'None' not in str(host.screen.query_one('#notes-recovery-issues', Static).content)
            await pilot.press('escape')
        elif action in ('root', 'navigation'):
            if action == 'root':
                other = folder.parent / 'other'
                other.mkdir()
                assert await subject.set_root(other, persist=False)
            else:
                await subject.remove()
            await wait_for(lambda: host.screen.query_one('#notes-recovery-approve', Button).disabled)
            assert 'Selection changed' in str(host.screen.query_one('#notes-recovery-status', Static).content)
            await pilot.press('escape')
        elif action == 'changed':
            (folder / 'one.md').write_text('changed after dry-run')
            before = data_state()
            host.screen.query_one('#notes-recovery-approve', Button).press()
            await wait_for(lambda: bool(host.screen.query('#confirm-button')))
            host.screen.query_one('#confirm-button', Button).press()
            await wait_for(lambda: bool(host.screen.query('#notes-recovery-status')) and 'not approved' in str(host.screen.query_one('#notes-recovery-status', Static).content))
            assert 'notes_pairing_review_changed' in str(host.screen.query_one('#notes-recovery-status', Static).content)
        elif action == 'cancel':
            await pilot.press('escape')
            await pilot.pause()
        else:
            host.screen.query_one('#notes-recovery-approve', Button).press()
            await wait_for(lambda: bool(host.screen.query('#confirm-button')))
            host.screen.query_one('#confirm-button', Button).press()
            owner = 'notes.file_notes'
            await wait_for(lambda: store.allowed(witness['generation'], owner))
            await wait_for(lambda: bool(host.screen.query('#notes-recovery-status')) and 'Pairing approved.' in str(host.screen.query_one('#notes-recovery-status', Static).content))
            assert not store.allowed(witness['generation'], 'notes.sync_bindings')
        if action not in ('approve',):
            assert not store.allowed(witness['generation'], 'notes.sync_bindings')
            assert not store.allowed(witness['generation'], 'notes.file_notes')
        assert data_state() == before, 'review/approval implicitly performed sync or reconcile'
        if action == 'approve':
            await pilot.press('escape')
            await pilot.pause()
            subject.query_one('#file-notes-maintenance-toggle', Button).press()
            await pilot.pause()
            button = subject.query_one('#file-notes-refresh', Button)
            assert button.display
            assert not button.disabled, (subject._path_transitioning, subject._active)
            button.press()
            try:
                await wait_for(lambda: replica.get_bytes(subject._service.root_key, 'one.md') == (folder / 'one.md').read_bytes())
            except AssertionError:
                raise AssertionError((subject._runtime_warning, subject._action_detail, subject._path_transitioning, subject._active, subject._service.root_key))
        await subject.shutdown()
asyncio.run(run())
db.close_connection()
replica.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["file_notes"])
@pytest.mark.parametrize("action", ["cancel", "approve"])
def test_actual_restored_owner_review_is_explicit_and_does_not_sync(
    tmp_path, route, action
):
    _run(tmp_path, route, action, script=_UI)


@pytest.mark.parametrize(
    "route,action",
    [
        ("file_notes", "issues"),
        ("file_notes", "changed"),
        ("file_notes", "root"),
        ("file_notes", "navigation"),
    ],
)
def test_actual_pairing_ui_refuses_incomplete_or_stale_review(tmp_path, route, action):
    _run(tmp_path, route, action, script=_UI)


_CANCEL_UI = (
    _UI.split("async def run():", 1)[0]
    + r"""
import threading
entered = threading.Event()
release = threading.Event()
observed = []

async def run():
    host = Host()
    async with host.run_test(size=(110, 45)) as pilot:
        await wait_for(lambda: hasattr(host, 'subject') and host.subject.initialized)
        subject = host.subject
        main_connection = db.get_connection()
        service = subject._service
        original = service._load_file

        def paused(*args, **kwargs):
            result = original(*args, **kwargs)
            if not entered.is_set():
                observed.append((None, threading.current_thread()))
                entered.set()
                assert release.wait(10), 'worker release not reached'
            return result
        service._load_file = paused
        selector = '#file-notes-recovery-review'
        subject.query_one(selector, Button).press()
        await wait_for(entered.is_set)
        group = 'file-notes-pairing-review'
        for worker in subject.workers:
            if worker.group == group:
                worker.cancel()
        await asyncio.sleep(0.05)
        assert subject._path_transitioning
        assert subject._session_owner._maintenance_operations.get(observed[0][1])
        assert not await subject.set_root(folder.parent / 'changed', persist=False)
        release.set()
        await wait_for(lambda: subject._pairing_review_task is None)
        assert not host.screen.query('#notes-recovery-entries'), 'cancelled waiter reattached review'
        assert not store.allowed(witness['generation'], 'notes.sync_bindings')
        assert not store.allowed(witness['generation'], 'notes.file_notes')
        assert main_connection.execute('SELECT 1').fetchone()[0] == 1
        assert not subject._path_transitioning
        service._load_file = original
        await subject.shutdown()
try:
    asyncio.run(run())
finally:
    release.set()
db.close_connection()
replica.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["file_notes"])
def test_cancelled_review_waiter_retains_actual_native_operation(tmp_path, route):
    _run(tmp_path, route, "cancel_worker", script=_CANCEL_UI)


_CANCEL_APPROVAL = (
    _UI.split("async def run():", 1)[0]
    + r"""
import threading
entered = threading.Event()
release = threading.Event()
observed = []
enabled = False

async def run():
    global enabled
    host = Host()
    async with host.run_test(size=(110, 45)) as pilot:
        await wait_for(lambda: hasattr(host, 'subject') and host.subject.initialized)
        subject = host.subject
        main_connection = db.get_connection()
        service = subject._service
        original = service._load_file

        def paused(*args, **kwargs):
            result = original(*args, **kwargs)
            if enabled and (not entered.is_set()):
                observed.append((None, threading.current_thread()))
                entered.set()
                assert release.wait(10), 'worker release not reached'
            return result
        service._load_file = paused
        selector = '#file-notes-recovery-review'
        subject.query_one(selector, Button).press()
        await wait_for(lambda: bool(host.screen.query('#notes-recovery-approve')))
        host.screen.query_one('#notes-recovery-approve', Button).press()
        await wait_for(lambda: bool(host.screen.query('#confirm-button')))
        enabled = True
        host.screen.query_one('#confirm-button', Button).press()
        await wait_for(entered.is_set)
        group = 'notes-pairing-confirm'
        for worker in subject.workers:
            if worker.group == group:
                worker.cancel()
        await asyncio.sleep(0.05)
        assert subject._path_transitioning
        assert subject._session_owner._maintenance_operations.get(observed[0][1])
        assert not await subject.set_root(folder.parent / 'changed', persist=False)
        release.set()
        await wait_for(lambda: subject._pairing_review_task is None)
        assert store.allowed(witness['generation'], 'notes.file_notes')
        assert not store.allowed(witness['generation'], 'notes.sync_bindings')
        assert not replica.list_active_files(files.root_key)
        assert main_connection.execute('SELECT 1').fetchone()[0] == 1
        assert not subject._path_transitioning
        service._load_file = original
        await subject.shutdown()
try:
    asyncio.run(run())
finally:
    release.set()
db.close_connection()
replica.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["file_notes"])
def test_cancelled_approved_waiter_does_not_release_native_pairing_early(
    tmp_path, route
):
    _run(tmp_path, route, "cancel_approval", script=_CANCEL_APPROVAL)


_DB_FAILURE_UI = (
    _UI.split("async def run():", 1)[0]
    + r"""
async def run():
    host = Host()
    async with host.run_test(size=(110, 45)) as pilot:
        await wait_for(lambda: hasattr(host, 'subject') and host.subject.initialized)
        subject = host.subject
        selector = '#file-notes-recovery-review'
        if action == 'approve_error':
            subject.query_one(selector, Button).press()
            await wait_for(lambda: bool(host.screen.query('#notes-recovery-approve')))
        with replica._transaction() as cursor:
            cursor.execute('ALTER TABLE files RENAME TO unavailable_files')
        if action == 'approve_error':
            host.screen.query_one('#notes-recovery-approve', Button).press()
            await wait_for(lambda: bool(host.screen.query('#confirm-button')))
            host.screen.query_one('#confirm-button', Button).press()
            await wait_for(lambda: bool(host.screen.query('#notes-recovery-status')) and 'not approved' in str(host.screen.query_one('#notes-recovery-status', Static).content))
        else:
            subject.query_one(selector, Button).press()
            await wait_for(lambda: 'Pairing review unavailable' in subject._action_detail)
        assert not store.allowed(witness['generation'], 'notes.sync_bindings')
        assert not store.allowed(witness['generation'], 'notes.file_notes')
        await subject.shutdown()
asyncio.run(run())
db.close_connection()
replica.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["file_notes"])
@pytest.mark.parametrize("action", ["preview_error", "approve_error"])
def test_damaged_owner_database_is_reported_without_crashing_ui(
    tmp_path, route, action
):
    _run(tmp_path, route, action, script=_DB_FAILURE_UI)
