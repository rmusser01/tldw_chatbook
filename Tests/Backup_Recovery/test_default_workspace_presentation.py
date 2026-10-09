"""Stock Default display policy must not allocate a cold Workspace handle."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio
from collections import Counter
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
from types import SimpleNamespace

from Tests import network_guard, real_profile_guard
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]
assert route == 'default_presentation'
assert outcome in {'view', 'availability', 'public_legacy'}
selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
root = Path(os.environ['XDG_DATA_HOME']).absolute()
selector.write_text('[general]\nusers_name="qualifier"\n[paths]\ndata_dir="'
                    + root.as_posix() + '"\n', encoding='utf-8')
selector.chmod(0o600)

from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import participants, storage_admission as storage
from tldw_chatbook.DB import base_db
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.UI.Console_Modules import workspace as module
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID
from Tests.UI import test_console_workspace_controller as helpers
from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver

database = WorkspaceDB(config.get_user_data_dir() / 'default-presentation.sqlite',
                       client_id='default-presentation')
registry = LocalWorkspaceRegistryService(database)
registry.ensure_default_workspace()
if outcome == 'public_legacy':
    with database.transaction() as connection:
        connection.execute('INSERT INTO workspace_runtime_bindings '
            '(binding_id,workspace_id,binding_kind,label,locator,status,metadata_json,created_at,updated_at) '
            'VALUES (?,?,?,?,?,?,?,?,?)', ('legacy-default', DEFAULT_WORKSPACE_ID,
                'local-filesystem', 'Legacy', str(root), 'ready', '{}',
                '2026-10-05T00:00:00Z', '2026-10-05T00:00:00Z'))
database.close()
assert not database._maintenance_participant.connections
counts, opened = Counter(), []
register = participants._register_core_connection
targets = [(participants, '_register_core_connection', register)]
for owner, names in (
    (LocalWorkspaceRegistryService, ('list_runtime_bindings', '_delete_default_runtime_bindings')),
    (module._ConsoleRegistryDisplayReads, ('list_runtime_bindings',)),
    (module.ConsoleWorkspaceController, ('_request_workspace_files_availability_refresh',
                                        '_refresh_workspace_files_availability_snapshot')),
):
    for name in names:
        targets.append((owner, name, inspect.getattr_static(owner, name)))

@contextmanager
def observe():
    witness = OriginalStorageUnitObserver({}, lambda: True, lambda unit: None)
    for owner, name, function in targets:
        witness._pin(function)
        witness.slots.append((owner, name, function))
    witness.codes = {function.__code__: function.__qualname__ for _, _, function in targets}
    monitor = sys.monitoring
    tool = next(slot for slot in range(5, 0, -1)
                if slot != monitor.DEBUGGER_ID and monitor.get_tool(slot) is None)
    monitor.use_tool_id(tool, 'original-default-presentation')
    witness.tool, witness.active, witness.installed = tool, True, True
    def started(code, offset):
        witness._start(code, offset)
        counts[witness.codes[code]] += 1
    def returned(code, offset, value):
        witness._return(code, offset, value)
        if code is register.__code__:
            frame = witness._frame(code)
            if frame.f_locals.get('repository') is database:
                connection = frame.f_locals['connection']
                assert value is connection
                opened.append((connection, threading.current_thread()))
    witness.registered = {monitor.events.PY_START: started, monitor.events.PY_RETURN: returned}
    try:
        for event, callback in witness.registered.items():
            assert monitor.register_callback(tool, event, callback) is None
        for code in witness.codes:
            monitor.set_local_events(tool, code, monitor.events.PY_START | monitor.events.PY_RETURN)
        yield
    finally:
        receipt = witness.close()
        assert receipt['complete'] and receipt['original_source_current'], receipt
        assert receipt['global_events'] == 0 and receipt['hooks_retired_before_inactive'], receipt
        assert monitor.get_tool(tool) is None

async def run():
    if outcome == 'availability':
        screen = helpers._AsyncWorkerScreen()
        syncs = []
        controller = helpers._workspace_controller(screen=screen,
            app_instance=SimpleNamespace(workspace_registry_service=registry),
            sync_workspace_context=lambda: syncs.append(True))
        with observe():
            controller._request_workspace_files_availability_refresh((DEFAULT_WORKSPACE_ID,))
            assert len(screen.workers) == 1
            await asyncio.wait_for(asyncio.shield(screen.workers[0][0]), 10)
        assert syncs == [True] and not controller._workspace_files_availability_refresh_in_flight
        assert controller._workspace_files_availability_by_id[DEFAULT_WORKSPACE_ID] is False
        assert controller._workspace_files_runtime_bindings_by_id[DEFAULT_WORKSPACE_ID] == ()
    else:
        with observe():
            with base_db.operation_owned_connection(database):
                if outcome == 'view':
                    value = module._ConsoleRegistryDisplayReads(registry).list_runtime_bindings(DEFAULT_WORKSPACE_ID)
                else:
                    value = registry.list_runtime_bindings(DEFAULT_WORKSPACE_ID)
        assert value == ()
        if outcome == 'public_legacy':
            with base_db.operation_owned_connection(database):
                with database.connection() as connection:
                    assert connection.execute('SELECT COUNT(*) FROM workspace_runtime_bindings '
                        'WHERE workspace_id=?', (DEFAULT_WORKSPACE_ID,)).fetchone()[0] == 0

try:
    asyncio.run(run())
    for connection, actor in opened:
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError as error:
            assert 'closed' in str(error).lower()
        else:
            raise AssertionError('original Workspace connection not physically retired')
    participant = database._maintenance_participant
    assert not participant.connections and not participant.retiring_threads
    assert not any(operation.participant is participant for operation in storage._operations)
    assert not any(lease.resource_path == participant.path for lease in storage._live_leases)
    assert not storage._pending_acquisitions and not storage._retiring_holds
    print(json.dumps({'route': outcome, 'original_calls': dict(counts),
        'actual_connections': len(opened), 'physically_retired': True,
        'source_current': True, 'global_events': 0, 'hooks_retired': True}))
    # Only this final work-count oracle is expected to fail on old source.
    if outcome == 'public_legacy':
        assert opened and counts['LocalWorkspaceRegistryService._delete_default_runtime_bindings'] == 1
    else:
        assert not opened, 'Stock Default presentation opened a cold Workspace connection'
finally:
    database.close()
print('retired and reopened')
"""


@pytest.mark.parametrize("outcome", ["view", "availability", "public_legacy"])
def test_stock_default_presentation_preserves_live_cleanup_without_cold_connection(
    tmp_path, outcome
):
    _run(tmp_path, "default_presentation", outcome, script=_SCRIPT)
