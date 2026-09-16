"""Actual network sync entrypoints observe independent restored-owner approval."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, os, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import bind_activation, ActivationStore
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
route, state = sys.argv[1:]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
data = selector.parent.parent / 'data'
selector.write_text('[general]\nusers_name="test"\n')
selector.chmod(0o600)
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.Sync_Interop.server_sync_service import ServerSyncService
from tldw_chatbook.Sync_Interop.sync_scope_service import SyncScopeService
from tldw_chatbook.Sync_Interop.local_first_sync_service import LocalFirstSyncService
from tldw_chatbook.Sync_Interop.notes_local_store import InMemoryNotesStore
repo = SyncStateRepository(data / 'sync.db')
scope_args = dict(server_profile_id='server-a', authenticated_principal_id='user-a', workspace_scope='workspace-1')
repo.set_sync_v2_profile_state(**scope_args, profile_mode='local_first',
    device_id='device-1', dataset_id='dataset-1', dataset_cursors={'sync_v2':'7'},
    capabilities={'supported_domains':['notes']}, dry_run_metadata={'dry_run':True})
before = repo.get_sync_v2_profile_state(**scope_args)
repo.close()
root = bootstrap.default_bootstrap_root()
startup = storage._startups.pop((os.getpid(), str(root)), None)
if startup: startup.close()
authority = admission_authority(root)
authority.register('profile', (selector.parent, data))
control = selector.parent.parent / 'operation'
control.mkdir(mode=0o700)
if state not in ('ordinary', 'unqualified'):
    register_pending(root, 'restore', ('profile',), control, (selector,))
    with authority.maintenance(('profile',), 2) as session:
        bind_activation(root, 'restore', selector, 'generation', ('config','runtime.sync_state'), session=session)
    (root / ('pending-' + bootstrap._key('restore') + '.json')).unlink()
    for owner in ('config', 'runtime.sync_state'):
        if state in ('approved', 'shared') or (state == 'config_only' and owner == 'config'):
            ActivationStore(control/'activation').approve('generation', owner)
if state == 'shared':
    other = selector.parent.parent / 'other.toml'
    other.write_text('[general]\n')
    os.environ['TLDW_CONFIG_PATH'] = str(other)
if state == 'unqualified':
    storage.qualified_for = lambda *args: (False, 'native_unqualified')
denied = state not in ('ordinary', 'approved', 'unqualified')
effects = []
class BoundaryReached(Exception): pass
class Provider:
    def build_client(self):
        effects.append('credential-client')
        raise BoundaryReached()
server = ServerSyncService(None, client_provider=Provider(), state_repository=repo)
facade = SyncScopeService(server_service=server, state_repository=repo)
local = LocalFirstSyncService(server_service=server, state_repository=repo,
    local_store=InMemoryNotesStore(), dataset_keys={'dataset-1': b'x'*32})
async def call():
    if route == 'direct': return await server.get_changes(client_id='fixture')
    if route == 'facade': return await facade.get_changes(client_id='fixture')
    if route == 'dry_run': return await facade.prepare_sync_v2_profile_mode(
        profile_mode='local_first', display_name='fixture', **scope_args)
    if route == 'local': return await local.sync_once(**scope_args, domains=['notes'])
    if route == 'push': return await server.push_v2_envelopes(dataset_id='dataset-1', device_id='device-1', envelopes=[])
    if route == 'conflicts': return await server.list_v2_conflicts(dataset_id='dataset-1')
try:
    asyncio.run(call())
except PermissionError:
    assert denied
except BoundaryReached:
    assert not denied
else:
    raise AssertionError('expected inactive refusal or client boundary')
if denied:
    assert not effects, effects
    assert repo.get_sync_v2_profile_state(**scope_args) == before, 'inactive state was mutated'
assert facade.get_sync_v2_profile_summary(**scope_args), 'local status is readable'
assert not blocked_attempts()
repo.close()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route", ["direct", "facade", "dry_run", "local", "push", "conflicts"]
)
@pytest.mark.parametrize(
    "state",
    ["inactive", "ordinary", "approved", "config_only", "shared", "unqualified"],
)
def test_network_sync_checks_actual_source_before_client_or_replay(
    tmp_path, route, state
):
    _run(tmp_path, route, state, script=_SCRIPT)


@pytest.mark.parametrize("target", ["delegate", "outer"])
def test_independently_injected_repository_is_checked(tmp_path, target):
    script = _SCRIPT.replace(
        "server = ServerSyncService(None,",
        "alternate = SyncStateRepository(selector.parent.parent / 'ordinary.db')\n"
        "server = ServerSyncService(None,",
    )
    if target == "delegate":
        script = script.replace(
            "facade = SyncScopeService(server_service=server, state_repository=repo)",
            "facade = SyncScopeService(server_service=server, state_repository=alternate)",
        )
    else:
        script = script.replace(
            "client_provider=Provider(), state_repository=repo)",
            "client_provider=Provider(), state_repository=alternate)",
        )
    script = script.replace(
        "assert not blocked_attempts()",
        "alternate.close()\nassert not blocked_attempts()",
    )
    _run(tmp_path, "facade", "shared", script=script)


_LIFETIME = (
    _SCRIPT.split("effects = []")[0]
    + r"""
import time
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
effects = []
pause = None
class Client:
    async def get_sync_changes(self, **kwargs):
        effects.append('network')
        return {'changes': [], 'latest_change_id': 0}
server = ServerSyncService(Client(), state_repository=repo)
class Policy:
    def require_allowed(self, **kwargs):
        global pause
        pause = storage._begin_local_pause()
facade = SyncScopeService(server_service=server, state_repository=repo, policy_enforcer=Policy())
async def main():
    global pause
    if route == 'nested':
        try:
            await facade.get_changes(client_id='fixture')
            assert effects == ['network']
            assert pause.drain(time.monotonic()+1)
        finally:
            if pause: pause.resume()
    elif route == 'child':
        class ChildClient:
            async def get_sync_changes(self, **kwargs):
                global pause
                pause = storage._begin_local_pause()
                try:
                    child = asyncio.create_task(server.get_changes(client_id='child'))
                    try: await child
                    except RecoveryRequired: pass
                    else: raise AssertionError('copied context bypassed closed admission')
                    assert not pause.drain(time.monotonic())
                    return {'changes': [], 'latest_change_id': 0}
                finally: pause.resume()
        server.client = ChildClient()
        await server.get_changes(client_id='parent')
    else:
        entered, settle = asyncio.Event(), asyncio.Event()
        class RetainedClient:
            async def get_sync_changes(self, **kwargs):
                entered.set()
                try: await asyncio.Event().wait()
                finally: await settle.wait()
        server.client = RetainedClient()
        task = asyncio.create_task(server.get_changes(client_id='fixture'))
        await entered.wait()
        pause = storage._begin_local_pause()
        try:
            task.cancel()
            await asyncio.sleep(0)
            assert not pause.drain(time.monotonic())
            settle.set()
            try: await task
            except asyncio.CancelledError: pass
            else: raise AssertionError('caller cancellation lost')
            assert pause.drain(time.monotonic()+1)
        finally:
            settle.set()
            pause.resume()
asyncio.run(main())
assert not blocked_attempts()
repo.close()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["nested", "child", "cancel"])
def test_actual_sync_call_scope_ownership_and_settlement(tmp_path, route):
    _run(tmp_path, route, "ordinary", script=_LIFETIME)


def test_restored_sync_refuses_without_native_admission(tmp_path):
    script = _SCRIPT.replace(
        "if state == 'unqualified':",
        "if state in ('unqualified', 'restored_unqualified'):",
    )
    _run(tmp_path, "direct", "restored_unqualified", script=script)


def test_inactive_sync_preserves_real_pending_outbox(tmp_path):
    script = _SCRIPT.replace(
        "before = repo.get_sync_v2_profile_state(**scope_args)",
        "from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder\n"
        "builder = SyncEnvelopeBuilder(dataset_id='dataset-1', device_id='device-1', dataset_key=b'x'*32)\n"
        "envelope = builder.build_note_metadata_update(note_id='note-1', status='archived')\n"
        "repo.enqueue_sync_v2_outbox_envelope(**scope_args, dataset_id='dataset-1', envelope=envelope)\n"
        "outbox = repo.list_sync_v2_outbox_entries(**scope_args, dataset_id='dataset-1')\n"
        "assert len(outbox) == 1\n"
        "before = repo.get_sync_v2_profile_state(**scope_args)",
    ).replace(
        "assert not effects, effects",
        "assert not effects, effects\n"
        "    assert repo.list_sync_v2_outbox_entries(**scope_args, dataset_id='dataset-1') == outbox",
    )
    _run(tmp_path, "local", "inactive", script=script)
