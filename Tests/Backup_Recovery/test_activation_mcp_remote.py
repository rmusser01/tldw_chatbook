"""Restored remote MCP definitions stay inert until their local review."""

import inspect

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, os, sys, types
from pathlib import Path
from types import SimpleNamespace
from Tests.network_guard import install, blocked_attempts
install()
sys.modules.setdefault('parakeet_mlx', types.ModuleType('parakeet_mlx'))
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending

route, state = sys.argv[1:]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
base = selector.parent.parent
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="' + str(base/'data') + '"\n')
selector.chmod(0o600)
from tldw_chatbook import config
data = config.get_user_data_dir()
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore
from tldw_chatbook.MCP.unified_control_models import (
    ConfiguredServerTarget, SectionCapabilityFlags, ServerAccessContext,
    UnifiedMCPContext,
)
from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

paths = {name: data/name for name in ('local','targets','context','permissions')}
for path in paths.values(): path.mkdir(mode=0o700)
local_path = paths['local']/'local_mcp_store.json'
local_path.write_text('{"profiles": []}')
target_store = ConfiguredServerTargetStore(paths['targets']/'mcp_server_targets.json')
target = ConfiguredServerTarget(
    server_id='server-a', label='Server A', base_url='https://blocked.invalid/api',
    auth_reference='imported-secret-reference', is_default=True,
)
target_store.save_targets([target])
context_store = UnifiedMCPContextStore(paths['context']/'unified_mcp_context.json')
access = ServerAccessContext(
    server_id='server-a', selected_scope='personal', selected_section='overview',
    section_capabilities=SectionCapabilityFlags(
        overview=True, inventory=True, catalogs=True, external_servers=True,
        governance=True, advanced=True,
    ),
)
context_store.save(UnifiedMCPContext(
    selected_source='server', selected_active_server_id='server-a',
    selected_scope='personal', selected_section='overview',
    per_server_state={'server-a': access},
))
permission_store = MCPPermissionStore(paths['permissions']/'mcp_permissions.json')
# Imported permission state remains recovery evidence, never activation authority.
permission_store.set_global_default('allow')

root = bootstrap.default_bootstrap_root()
startup = storage._startups.pop((os.getpid(), str(root)), None)
if startup is not None: startup.close()
authority = admission_authority(root)
restored = paths[state] if state in paths else data
authority.register('profile', (selector.parent, restored))
control = base/'operation'; control.mkdir(mode=0o700)
owners = ('config','mcp.local','mcp.targets','mcp.context','mcp.permissions')
if state not in ('ordinary','unqualified'):
    register_pending(root,'restore',('profile',),control,(selector,))
    with authority.maintenance(('profile',),2) as session:
        bind_activation(root,'restore',selector,'generation',owners,session=session)
    (root/('pending-'+bootstrap._key('restore')+'.json')).unlink()
    activation = ActivationStore(control/'activation')
    for owner in owners:
        if state == 'approved' or state == 'config_only' and owner == 'config':
            activation.approve('generation', owner)
    if state == 'missing':
        (activation._generation('generation')/'required.json').unlink()
    if state == 'corrupt':
        (activation._generation('generation')/'required.json').write_bytes(b'{')
if state in paths:
    other = base/'unrelated.toml'
    other.write_text('[general]\n')
    other.chmod(0o600)
    os.environ['TLDW_CONFIG_PATH'] = str(other)
if state == 'unqualified':
    storage.qualified_for = lambda *args: (False, 'native_unqualified')

events = []
class DirectClient:
    async def get_status(self):
        events.append('network')
        return {'status':'ok'}

def client_factory(selected):
    events.append('credential')
    return DirectClient()

direct = ServerUnifiedMCPService(client_factory=client_factory, target_store=target_store)
cache_key = direct._cache_key(
    section='overview', server_id='server-a', selected_scope='personal',
    selected_scope_ref=None,
)
direct._browse_cache[cache_key] = {'cached': True}

class InjectedServer:
    async def resolve_access_context(self, **kwargs):
        events.append('resolve')
        return access
    async def get_overview(self, **kwargs):
        events.append('overview')
        return {'server_id':'server-a'}
    async def set_external_server_secret(self, **kwargs):
        events.append('secret')
        return {'updated': True}

injected = InjectedServer()
local = SimpleNamespace(store=SimpleNamespace(path=local_path))
plane = UnifiedMCPControlPlaneService(
    target_store=target_store, context_store=context_store,
    local_service=local, server_service=injected,
)
plane._permission_store = permission_store

async def run():
    if route == 'direct_factory':
        direct.invalidate_cache()
        return await direct.get_overview(target=target, access_context=access)
    if route == 'direct_cache':
        return await direct.get_overview(target=target, access_context=access)
    if route == 'select_target':
        return await plane.select_server_target('server-a')
    if route == 'select_scope':
        return await plane.select_scope('personal')
    if route == 'select_section':
        return await plane.select_section('overview')
    if route == 'load_section':
        return await plane.load_section('overview')
    if route == 'run_action':
        return await plane.run_action(
            'external_server.secret.set',
            {'server_id':'external-a','secret':'never-resolve-this'},
        )
    if route == 'inspection':
        assert [item.server_id for item in target_store.list_targets()] == ['server-a']
        assert (await plane.load_context()).selected_active_server_id == 'server-a'
        assert plane.runtime_state_override().active_server_id == 'server-a'
        assert isinstance(plane.available_actions(), list)
        return {'inspected': True}
    raise AssertionError(route)

denied = state not in ('ordinary','approved','unqualified')
try:
    result = asyncio.run(run())
except PermissionError as exc:
    if not denied or route == 'inspection': raise
    assert str(exc) == 'mcp_activation_required'
else:
    if denied and route != 'inspection':
        raise AssertionError('inactive remote MCP route was admitted')
    if route == 'direct_cache': assert result == {'cached': True}
    if route == 'inspection': assert result == {'inspected': True}
if denied or route == 'inspection' or route == 'direct_cache':
    assert events == [], events
else:
    assert events, route
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    [
        "direct_factory",
        "direct_cache",
        "select_target",
        "select_scope",
        "select_section",
        "load_section",
        "run_action",
        "inspection",
    ],
)
def test_inactive_remote_mcp_denies_before_client_or_cache(tmp_path, route):
    _run(tmp_path, route, "inactive", script=_SCRIPT)


@pytest.mark.parametrize(
    "state",
    [
        "ordinary",
        "approved",
        "unqualified",
        "config_only",
        "missing",
        "corrupt",
        "local",
        "targets",
        "context",
        "permissions",
    ],
)
def test_remote_mcp_observes_actual_independent_sources(tmp_path, state):
    _run(tmp_path, "load_section", state, script=_SCRIPT)


@pytest.mark.parametrize("state", ["ordinary", "approved", "targets"])
def test_direct_remote_mcp_observes_configured_target_store(tmp_path, state):
    _run(tmp_path, "direct_factory", state, script=_SCRIPT)


_RETENTION = _SCRIPT.split("events = []")[0] + r"""
import threading
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
from tldw_chatbook.MCP.activation import MCPActivationRequired

events = []
entered = asyncio.Event()
release = asyncio.Event()

def assert_held():
    try:
        with authority.maintenance(('profile',), .03):
            raise AssertionError('remote MCP await lost native admission')
    except AdmissionTimeout:
        pass

class WaitingServer:
    async def get_overview(self, **kwargs):
        events.append('entered')
        entered.set()
        await release.wait()
        assert_held()
        events.append('finished')
        return {'server_id':'server-a'}

local = SimpleNamespace(store=SimpleNamespace(path=local_path))
plane = UnifiedMCPControlPlaneService(
    target_store=target_store, context_store=context_store,
    local_service=local, server_service=WaitingServer(),
)
plane._permission_store = permission_store

async def run():
    accepted = asyncio.create_task(plane.load_section('overview'))
    await entered.wait()
    assert_held()
    pause = storage._begin_local_pause()
    try:
        try:
            await plane.load_section('overview')
        except MCPActivationRequired:
            pass
        else:
            raise AssertionError('new remote MCP intake entered during pause')
        assert events == ['entered']
        assert_held()
    finally:
        release.set()
        result = await accepted
        pause.resume()
    assert result['server_id'] == 'server-a'

asyncio.run(run())
assert events == ['entered','finished'], events
with authority.maintenance(('profile',),1): pass
assert not blocked_attempts()
print('retired and reopened')
"""


def test_remote_mcp_accepted_await_retains_lease_and_denies_new_intake(tmp_path):
    _run(tmp_path, "retained", "approved", script=_RETENTION)


def test_remote_mcp_supported_operation_set_has_no_unguarded_async_route():
    from tldw_chatbook.MCP.server_unified_service import (
        _REMOTE_SERVER_OPERATIONS,
        ServerUnifiedMCPService,
    )
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    public_async = {
        name
        for name, value in vars(ServerUnifiedMCPService).items()
        if not name.startswith("_") and inspect.iscoroutinefunction(value)
    }
    assert public_async == set(_REMOTE_SERVER_OPERATIONS)
    assert all(
        getattr(getattr(ServerUnifiedMCPService, name), "_mcp_activation_guarded", False)
        for name in public_async
    )
    assert all(
        getattr(
            getattr(UnifiedMCPControlPlaneService, name),
            "_mcp_activation_guarded",
            False,
        )
        for name in (
            "select_server_target",
            "select_scope",
            "select_section",
            "load_section",
            "run_action",
        )
    )
