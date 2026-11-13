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


_PERSONAL_CONTEXT_SCRIPT = r"""
import os, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import bind_activation, ActivationStore
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
import tldw_chatbook.Personal_Context.recovery as pc
case = sys.argv[1]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
data = selector.parent.parent / 'data'
selector.write_text('[general]\nusers_name="test"\n')
selector.chmod(0o600)
db = data / 'tldw_chatbook_personal_context.db'
root = bootstrap.default_bootstrap_root()
startup = storage._startups.pop((os.getpid(), str(root)), None)
if startup: startup.close()
authority = admission_authority(root)
authority.register('profile', (selector.parent, data))
control = selector.parent.parent / 'operation'
control.mkdir(mode=0o700)
if case == 'ordinary':
    assert pc.sync_allowed(db) and pc.restore_review(db) is None
    print('retired and reopened')
    raise SystemExit(0)
register_pending(root, 'restore', ('profile',), control, (selector,))
with authority.maintenance(('profile',), 2) as session:
    bind_activation(root, 'restore', selector, 'generation',
        ('config', 'personal_context.repository', 'runtime.sync_state'), session=session)
(root / ('pending-' + bootstrap._key('restore') + '.json')).unlink()
# The restore plan's published owners (read from its journal in production).
published = {
    'neither': frozenset({'config'}),
    'sync_state': frozenset({'config', 'runtime.sync_state'}),
    'repository': frozenset({'personal_context.repository'}),
    'unknown': None,
}[case]
pc._published_owners = lambda witness: published
# AC#9: the guard holds on its own, before any review.
assert not pc.sync_allowed(db)
try:
    pc.require_sync_allowed(db)
except ValueError as error:
    assert str(error) == pc.SYNC_REVIEW_REQUIRED
else:
    raise AssertionError('restored Personal Context sync resumed')
store = ActivationStore(control / 'activation')
if case == 'neither':
    assert pc.restore_review(db) == pc.REVIEW_AVAILABLE
    pc.approve_restore_review(db)
    assert store.allowed('generation', 'personal_context.repository')
    assert pc.restore_review(db) is None
else:
    # TASK-40139: the review clears the restored link first, then approves.
    assert pc.restore_review(db) == pc.RESTORED_LINK_STATE
    pc.approve_restore_review(db)
    assert store.allowed('generation', 'personal_context.repository')
# AC#10: never the shared owners, so Personal Context sync stays off.
assert not store.allowed('generation', 'config')
assert not store.allowed('generation', 'runtime.sync_state')
assert not pc.sync_allowed(db)
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "case", ["ordinary", "neither", "sync_state", "repository", "unknown"]
)
def test_restored_personal_context_sync_waits_for_its_own_review(tmp_path, case):
    """TASK-40134 AC#9-10: after any restore Personal Context sync does not load;
    its review approves only ``personal_context.repository`` (TASK-40139: after
    clearing a restored link when the restore published either owner)."""
    _run(tmp_path, case, "unused", script=_PERSONAL_CONTEXT_SCRIPT)


# TASK-40139: a restored profile never resumes its old link, and Notes/Chat
# never act as the old device: the Personal Context review clears the restored
# link, and the sync-identity review clears the restored identity (counts
# confirmed) before it approves `config` and `runtime.sync_state` together.
_RELINK_SCRIPT = r"""
import asyncio, os, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import bind_activation, ActivationStore
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
import tldw_chatbook.Personal_Context.recovery as pc
import tldw_chatbook.Sync_Interop.recovery as sync_recovery
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.Sync_Interop.server_sync_service import ServerSyncService
from tldw_chatbook.Sync_Interop.local_first_sync_service import LocalFirstSyncService
from tldw_chatbook.Sync_Interop.notes_local_store import InMemoryNotesStore
case = sys.argv[1]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
data = selector.parent.parent / 'data'
selector.write_text('[general]\nusers_name="test"\n')
selector.chmod(0o600)
db = data / 'tldw_chatbook_personal_context.db'
state_path = data / 'tldw_chatbook_sync_state.db'
scope = dict(server_profile_id='server-config-1', authenticated_principal_id='user-1')
# The restored sync state, as the old device left it (shared row, no workspace).
repo = SyncStateRepository(state_path)
repo.set_sync_v2_profile_state(**scope, workspace_scope=None, profile_mode='local_first',
    device_id='old-device', dataset_id='dataset-1', dataset_cursors={'sync_v2': '41'},
    capabilities={'supported_domains': ['notes']})
repo.set_remote_pull_cursor(source_authority='server', **scope, workspace_scope=None,
    domain='sync_v2', remote_collection='dataset-1', cursor='41')
repo.set_personal_context_link_state(**scope, state='complete', device_id='old-device',
    dataset_id='dataset-1', authority_id='authority-1', profile_id='profile-server',
    integrity_key_id='key-1', key_record_id='key-record-1', purge_generation=0,
    bootstrap_cursor='sha256:' + 'b' * 64, sync_transport_cursor='41', plan_id='plan-1',
    rebaseline_version=1, attention_code=None)
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
old = SyncEnvelopeBuilder(dataset_id='dataset-1', device_id='old-device', dataset_key=b'x' * 32)
repo.enqueue_sync_v2_outbox_envelope(**scope, workspace_scope=None, dataset_id='dataset-1',
    envelope=old.build_note_metadata_update(note_id='note-1', status='archived'))
repo.close()
root = bootstrap.default_bootstrap_root()
startup = storage._startups.pop((os.getpid(), str(root)), None)
if startup: startup.close()
authority = admission_authority(root)
authority.register('profile', (selector.parent, data))
control = selector.parent.parent / 'operation'
control.mkdir(mode=0o700)
register_pending(root, 'restore', ('profile',), control, (selector,))
with authority.maintenance(('profile',), 2) as session:
    bind_activation(root, 'restore', selector, 'generation',
        ('config', 'personal_context.repository', 'runtime.sync_state'), session=session)
(root / ('pending-' + bootstrap._key('restore') + '.json')).unlink()
published = {'repository': frozenset({'personal_context.repository', 'config'}),
             'sync_state': frozenset({'runtime.sync_state', 'config'})}[case]
pc._published_owners = lambda witness: published
store = ActivationStore(control / 'activation')
def link_state():
    r = SyncStateRepository(state_path)
    try: return r.get_personal_context_link_state(**scope)
    finally: r.close()
def profile_row():
    r = SyncStateRepository(state_path)
    try: return r.get_sync_v2_profile_state(**scope, workspace_scope=None)
    finally: r.close()
sent = []
class BoundaryReached(Exception): pass
def notes_sync():
    r = SyncStateRepository(state_path)
    server = ServerSyncService(None, client_provider=None, state_repository=r)
    async def record(**kwargs):
        sent.append({k: kwargs.get(k) for k in ('device_id', 'cursor', 'last_known_cursor')})
        raise BoundaryReached()
    server.push_v2_envelopes = record
    server.pull_v2_envelopes = record
    local = LocalFirstSyncService(server_service=server, state_repository=r,
        local_store=InMemoryNotesStore(), dataset_keys={'dataset-1': b'x' * 32})
    try:
        asyncio.run(local.sync_once(**scope, workspace_scope=None, domains=['notes']))
    finally:
        r.close()
# AC#1: each kind of restore keeps Personal Context sync off until review.
assert not pc.sync_allowed(db)
assert pc.restore_review(db) == pc.RESTORED_LINK_STATE
# AC#2: the review clears the restored link, then approves only the repository.
pc.approve_restore_review(db)
assert link_state() is None
assert store.allowed('generation', 'personal_context.repository')
assert not store.allowed('generation', 'config')
assert not store.allowed('generation', 'runtime.sync_state')
assert pc.restore_review(db) is None and not pc.sync_allowed(db)
# AC#3: Notes and Chat sync is still refused, before any request.
try:
    notes_sync()
except PermissionError as error:
    assert str(error) == 'sync_activation_required'
else:
    raise AssertionError('Notes and Chat sync resumed after the Personal Context review')
assert not sent
# AC#9: one step, counts first; a stale count refuses and changes nothing.
counts = sync_recovery.sync_identity_review(state_path)
assert counts == {'device_ids': 1, 'datasets': 1, 'pull_cursors': 1, 'remote_heads': 0,
                  'personal_context_links': 0, 'pending_outbox': 1}, counts
try:
    sync_recovery.approve_sync_identity(state_path, shown={**counts, 'pending_outbox': 0})
except ValueError as error:
    assert str(error) == sync_recovery.SYNC_IDENTITY_CHANGED
else:
    raise AssertionError('changed counts were approved')
assert profile_row()['device_id'] == 'old-device'
assert not store.allowed('generation', 'runtime.sync_state')
sync_recovery.approve_sync_identity(state_path, shown=counts)
assert store.allowed('generation', 'config') and store.allowed('generation', 'runtime.sync_state')
assert sync_recovery.sync_identity_review(state_path) is None
row = profile_row()
assert (row['device_id'], row['dataset_id'], row['dataset_cursors']) == (None, None, {})
assert sync_recovery._identity_counts(state_path) == dict.fromkeys(counts, 0)
assert pc.sync_allowed(db)
# AC#4: a fresh first link binds the device the server issues; no conflict.
from Tests.Sync_Interop.test_personal_context_first_link import (
    FakeFirstLinkSync, FakeProfileService, FakeServerSync, FakeWrappingProvider)
from tldw_chatbook.Personal_Context.link_key_custody import InMemoryPersonalContextLinkKeyCustodian
from tldw_chatbook.Personal_Context.link_service import PersonalContextLinkService
r = SyncStateRepository(state_path)
link = PersonalContextLinkService(personal_context_service=FakeProfileService(),
    server_sync_service=FakeServerSync(), state_repository=r,
    wrapping_key_provider=FakeWrappingProvider(),
    key_custodian=InMemoryPersonalContextLinkKeyCustodian(),
    first_link_sync=FakeFirstLinkSync(), display_name='Laptop', **scope)
asyncio.run(link.plan())
r.close()
row = profile_row()
assert (row['device_id'], row['dataset_id']) == ('device-1', 'dataset-1'), row
assert link_state()['device_id'] == 'device-1'
# AC#5/#6: Notes and Chat run on the newly issued device: no old id, cursor or outbox.
try:
    notes_sync()
except BoundaryReached:
    pass
assert sent == [{'device_id': 'device-1', 'cursor': None, 'last_known_cursor': None}], sent
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("case", ["repository", "sync_state"])
def test_restored_link_and_sync_identity_are_cleared_before_review_approves(
    tmp_path, case
):
    """TASK-40139 AC#1-#6, #9, for a restore that published the repository and
    for an automation-only one that published only ``runtime.sync_state``."""
    _run(tmp_path, case, "unused", script=_RELINK_SCRIPT)
