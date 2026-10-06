"""Exact Default custom ABI and source/owner refusal companions."""

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
assert outcome in {'fallback', 'refusal'}
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
    from types import MethodType
    from tldw_chatbook.Workspaces import registry_service as source
    checker = module._default_presentation_policy(registry)
    assert checker is not None and checker() is True
    replacement = None
    if outcome == 'refusal':
        replacement = WorkspaceDB(config.get_user_data_dir() / 'default-replacement.sqlite',
                                  client_id='default-replacement')
        replacement.close()
    edges = []
    with observe():
        if outcome == 'fallback':
            calls = []
            registry._default_companion_calls = calls
            def custom(self, workspace_id):
                assert workspace_id == DEFAULT_WORKSPACE_ID
                self._default_companion_calls.append(True)
                return ('custom-default-display',)
            registry.list_runtime_bindings = MethodType(custom, registry)
            try:
                assert module._default_presentation_policy(registry) is None
                value = module._ConsoleRegistryDisplayReads(registry).list_runtime_bindings(DEFAULT_WORKSPACE_ID)
                assert value == ('custom-default-display',) and calls == [True]
                edges.append('instance_custom_abi')
            finally:
                del registry.list_runtime_bindings
            function = LocalWorkspaceRegistryService.list_runtime_bindings
            original_code = function.__code__
            assert custom.__code__.co_freevars == function.__code__.co_freevars == ()
            # Retarget only the public source body; SQL/getter/guard functions
            # are never replaced, and the original body is restored before
            # Base source/retirement acceptance.
            function.__code__ = custom.__code__
            try:
                assert module._default_presentation_policy(registry) is None
                value = module._ConsoleRegistryDisplayReads(registry).list_runtime_bindings(DEFAULT_WORKSPACE_ID)
                assert value == ('custom-default-display',) and calls == [True, True]
                edges.append('preinstalled_body_fallback')
            finally:
                function.__code__ = original_code
        else:
            original_db = registry.db
            registry.db = replacement
            try:
                assert checker() is False
                assert not replacement._maintenance_participant.connections
                edges.append('real_database_owner_changed')
            finally:
                registry.db = original_db
            registry.db = object()
            try:
                assert checker() is False
                assert module._default_presentation_policy(registry) is None
                edges.append('actual_registry_owner_changed')
            finally:
                registry.db = original_db
            prior_identity = config.current_config_identity()
            assert config.save_setting_to_cli_config('general', 'users_name',
                                                     'changed-companion-owner') is True
            current_identity = config.current_config_identity()
            assert current_identity[1] == prior_identity[1]
            assert current_identity[0] != prior_identity[0]
            assert checker() is False
            fresh_checker = module._default_presentation_policy(registry)
            assert fresh_checker is not None and fresh_checker() is True
            edges.append('actual_public_writer_profile_generation_changed')
            selector_record = source._DEFAULT_PRESENTATION_SELECTOR_SOURCE
            policy_record = source._DEFAULT_PRESENTATION_SOURCE
            try:
                source._DEFAULT_PRESENTATION_SELECTOR_SOURCE = (None,) * 8
                assert module._default_presentation_policy(registry) is None
                edges.append('malformed_dispatcher_record')
                source._DEFAULT_PRESENTATION_SELECTOR_SOURCE = selector_record
                source._DEFAULT_PRESENTATION_SOURCE = (None,) * 7
                assert module._default_presentation_policy(registry) is None
                assert checker() is False
                edges.append('malformed_policy_record')
            finally:
                source._DEFAULT_PRESENTATION_SELECTOR_SOURCE = selector_record
                source._DEFAULT_PRESENTATION_SOURCE = policy_record
    assert module._default_presentation_policy(registry)() is True
    assert not opened
    if replacement is not None:
        replacement.close()
    print(json.dumps({'companion_edges': edges, 'native_connections': 0,
        'original_setup_and_guards_retained': True}))
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
finally:
    database.close()
print('retired and reopened')
"""


@pytest.mark.parametrize("outcome", ["fallback", "refusal"])
def test_default_presentation_companion_contracts(tmp_path, outcome):
    _run(tmp_path, "default_presentation", outcome, script=_SCRIPT)
