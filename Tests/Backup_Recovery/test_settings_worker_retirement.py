"""Settings reads retire only their own actual pool-thread WorkspaceDB handle."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import asyncio, sqlite3, sys, threading
from pathlib import Path
from types import SimpleNamespace
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
route, outcome = sys.argv[1:]
db = WorkspaceDB(Path.home() / 'workspaces.db')
registry = LocalWorkspaceRegistryService(db)
registry.ensure_default_workspace()
main_connection = db._held_connection()
entered, release, done = threading.Event(), threading.Event(), threading.Event()
observed = []
errors = []
def rows():
    registry.get_active_workspace()
    observed.append(db._held_connection())
    if outcome == 'cancel':
        entered.set()
        assert release.wait(10)
    return (('Workspace', 'Default'),)
def apply(*args):
    if outcome == 'error':
        raise RuntimeError('render failed after actual read')
owner = SimpleNamespace(
    app_instance=SimpleNamespace(workspace_registry_service=registry),
    app=SimpleNamespace(call_from_thread=lambda callback, *args: callback(*args)),
    _server_sync_workspace_handoff_rows=rows, _manual_sync_rows=rows,
    _apply_sync_rows=apply, _apply_manual_sync_rows=apply,
)
function = getattr(SettingsScreen, route).__wrapped__
def worker():
    previous = db._held_connection() if outcome == 'preexisting' else None
    try:
        try:
            function(owner)
        except RuntimeError:
            assert outcome == 'error'
        assert observed
        connection = observed[-1]
        if previous is not None:
            assert connection is previous
            assert connection.execute('SELECT 1').fetchone()[0] == 1
        else:
            try:
                connection.execute('SELECT 1')
            except sqlite3.ProgrammingError as error:
                assert 'closed' in str(error)
            else:
                raise AssertionError('Settings worker retained native WorkspaceDB')
    except BaseException as error:
        errors.append(error)
        raise
    finally:
        db.close()
        done.set()
async def main():
    pending = asyncio.create_task(asyncio.to_thread(worker))
    if outcome == 'cancel':
        async with asyncio.timeout(10):
            while not entered.is_set(): await asyncio.sleep(.01)
        pending.cancel()
        try: await pending
        except asyncio.CancelledError: pass
        assert not done.is_set()
        release.set()
        async with asyncio.timeout(10):
            while not done.is_set(): await asyncio.sleep(.01)
    else:
        await pending
    assert not errors, errors
    assert main_connection.execute('SELECT 1').fetchone()[0] == 1
    db.close()
asyncio.run(main())
assert not blocked_attempts()
print('retired and reopened')
'''


@pytest.mark.parametrize("route", ("_refresh_sync_rows", "_refresh_manual_sync_rows"))
@pytest.mark.parametrize("outcome", ("fresh", "preexisting", "error", "cancel"))
def test_settings_worker_owns_only_new_workspace_connection(tmp_path, route, outcome):
    _run(tmp_path, route, outcome, script=_SCRIPT)
