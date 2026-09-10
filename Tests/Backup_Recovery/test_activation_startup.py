"""Restored startup execution is inactive while local persisted reads remain usable."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio
import os
from pathlib import Path
import sys
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice', 'pyaudio'):
    sys.modules[name] = None
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending

scenario = sys.argv[1]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
data = selector.parent.parent / 'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="' + str(data) + '"\n')
selector.chmod(0o600)
root = bootstrap.default_bootstrap_root()
authority = admission_authority(root)
authority.register('profile', (selector.parent, data))
control = selector.parent.parent / 'operation'
control.mkdir(mode=0o700)
register_pending(root, 'restore', ('profile',), control, (selector,))
owners = ('db.scheduled_tasks', 'config', 'models.artifacts', 'tts.profile_store',
          'tts.voices', 'db.subscriptions', 'db.media.primary', 'db.agent_runs')
with authority.maintenance(('profile',), 2) as session:
    bind_activation(root, 'restore', selector, 'generation', owners, session=session)
# Simulate the later executor's completed fence; activation requirements survive.
(root / ('pending-' + bootstrap._key('restore') + '.json')).unlink()
process_attempts = []
def prevent_execution(event, arguments):
    if event in ('subprocess.Popen', 'os.posix_spawn', 'os.system'):
        process_attempts.append(event)
        raise PermissionError('test_process_execution_blocked')

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
# Imports perform Python platform.architecture's read-only `file` probe.
# The sentry covers the actual accepted application execution entries below.
sys.addaudithook(prevent_execution)
db = ScheduledTasksDB(data / 'scheduled.db')
key = db.create_reminder_task(owner_id='local', title='Keep history',
    schedule_kind='one_time', next_run_at='2020-01-01T00:00:00+00:00')
before = dict(db.get_reminder_task(key))
effects = []
async def dispatch(row): effects.append('dispatch')
loop = SchedulerLoop(db, {'reminder': dispatch}, poll_interval=.001)
loop.queue.load()
async def main():
    if scenario == 'tick':
        await loop.tick()
    elif scenario == 'manual':
        await loop.run_reminder_now(key)
    elif scenario == 'run':
        worker = asyncio.create_task(loop.run())
        await asyncio.sleep(.04)
        loop.stop()
        await asyncio.wait_for(worker, 2)
    elif scenario == 'sync':
        service = SchedulingService(db)
        async def sync(owner): effects.append('network')
        service.sync_engine.sync_now = sync
        await service.sync_now()
    elif scenario == 'service_manual':
        service = SchedulingService(db, on_queue_changed=lambda: effects.append('reload'))
        await service.run_reminder_now(key, loop)
asyncio.run(main())
assert not effects, effects
assert dict(db.get_reminder_task(key)) == before, 'inactive execution changed persisted history'
assert not blocked_attempts(), blocked_attempts()
assert not process_attempts, process_attempts
db.close()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "scenario", ["tick", "manual", "run", "sync", "service_manual"]
)
def test_restored_scheduler_does_not_dispatch_or_sync(tmp_path, scenario):
    _run(tmp_path, scenario, "inactive", script=_SCRIPT)


_APP_SCRIPT = (
    _SCRIPT.split("from tldw_chatbook.Scheduling.db")[0]
    + r"""
from types import SimpleNamespace, MethodType
from loguru import logger
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.app import TldwCli
import tldw_chatbook.app as module
sys.addaudithook(prevent_execution)

effects = []
async def initialize(): effects.append('speech')
app = SimpleNamespace(loguru_logger=logger, app_config={}, model_catalog_disk_store=object())
app._speech_initialization_allowed = MethodType(TldwCli._speech_initialization_allowed, app)
app._perform_change_review_retention = lambda: None
app.perform_media_cleanup = lambda: None
app.set_timer = lambda *a, **k: effects.append('timer')
app.set_interval = lambda *a, **k: effects.append('interval')
app.call_after_refresh = lambda *a, **k: effects.append('refresh')
async def refresh(**kwargs): effects.append('catalog')
app.local_llm_provider_catalog_service = SimpleNamespace(refresh_stale_configured_providers=refresh)
app._init_providers_models = lambda: None
app.post_message = lambda *a: None
app.notify = lambda *a, **k: None
module.load_settings = lambda: {'model_catalog': {'auto_refresh_enabled': True, 'refresh_consent_recorded': True}}

async def main():
    if scenario == 'speech':
        await TldwCli._run_speech_initialization(app, 'tts', initialize)
    elif scenario == 'timers':
        TldwCli.schedule_media_cleanup(app)
    elif scenario == 'backfill':
        app.subscriptions_db = SimpleNamespace(close=lambda: None)
        module.backfill_subscription_items_fts = lambda db: effects.append('backfill')
        TldwCli._backfill_subscription_items_fts(app)
    elif scenario == 'catalog':
        await TldwCli._refresh_model_catalogs(app)
asyncio.run(main())
assert not effects, effects
assert not blocked_attempts(), blocked_attempts()
assert not process_attempts, process_attempts
print('retired and reopened')
"""
)


@pytest.mark.parametrize("scenario", ["speech", "timers", "backfill", "catalog"])
def test_restored_startup_callbacks_remain_inactive(tmp_path, scenario):
    _run(tmp_path, scenario, "inactive", script=_APP_SCRIPT)


_SCOPE_SCRIPT = (
    _SCRIPT.split("from tldw_chatbook.Scheduling.db")[0]
    + r"""
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, execution_scope
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
sys.addaudithook(prevent_execution)
source = data / 'source'
source.write_bytes(b'preserve')
if scenario in ('ordinary_registered', 'restored_shared', 'approved_shared'):
    other = selector.parent.parent / 'other.toml'
    other.write_text('[general]\n')
    os.environ['TLDW_CONFIG_PATH'] = str(other)
if scenario == 'ordinary_registered':
    for kind in ('profile', 'activation'):
        (root / (kind + '-' + bootstrap._key(str(selector)) + '.json')).unlink()
if scenario in ('approved_shared', 'held'):
    ActivationStore(control / 'activation').approve('generation', 'db.scheduled_tasks')
with execution_scope(('db.scheduled_tasks',), source) as allowed:
    assert allowed == (scenario in ('ordinary_registered', 'held')), scenario
    if allowed:
        with storage._lock:
            leases = tuple(storage._live_leases)
        assert leases
        assert all(lease.execution_scope()[1] for lease in leases)
        try:
            with authority.maintenance(('profile', 'bootstrap.unbound'), .02):
                raise AssertionError('accepted execution lost native participation')
        except AdmissionTimeout:
            pass
assert source.read_bytes() == b'preserve'
lease = storage.acquire_storage(source)
assert lease.execution_scope()[0] == root
lease.close()
try:
    lease.execution_scope()
except bootstrap.RecoveryRequired as error:
    assert str(error) == 'execution_scope_not_admitted'
else:
    raise AssertionError('retired token granted execution scope')
print('retired and reopened')
"""
)


@pytest.mark.parametrize(
    "scenario", ["ordinary_registered", "restored_shared", "approved_shared", "held"]
)
def test_execution_uses_actual_admitted_source_scope(tmp_path, scenario):
    _run(tmp_path, scenario, "scope", script=_SCOPE_SCRIPT)


_REMOTE_SCRIPT = (
    _SCRIPT.split("from tldw_chatbook.Scheduling.db")[0]
    + r"""
if sys.argv[2] == 'ordinary':
    for kind in ('profile', 'activation'):
        (root / (kind + '-' + bootstrap._key(str(selector)) + '.json')).unlink()
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import ServerUnavailableError
from types import SimpleNamespace
sys.addaudithook(prevent_execution)
effects = []
async def remote(*args, **kwargs):
    effects.append('network')
    raise ServerUnavailableError('test offline response')
db = ScheduledTasksDB(data / 'scheduled.db')
client = SimpleNamespace(create_reminder=remote, update_reminder=remote, delete_reminder=remote)
service = SchedulingService(db, server_client=client, runtime_source='server:test')
async def main():
    if scenario == 'create':
        created = await service.create_reminder({'title':'Local edit', 'schedule_kind':'one_time', 'run_at':'2099-01-01T00:00:00+00:00'})
        assert created.title == 'Local edit'
    else:
        key = db.create_reminder_task(owner_id='server:test', title='Keep history',
            schedule_kind='one_time', server_id='remote-id', run_at='2099-01-01T00:00:00+00:00')
        if scenario == 'update':
            result = await service.update_reminder(key, {'title':'Edited locally'})
            assert result.title == 'Edited locally'
        else:
            assert await service.delete_reminder(key)
            assert db.get_reminder_task(key) is None
asyncio.run(main())
assert effects == (['network'] if sys.argv[2] == 'ordinary' else []), effects
assert not blocked_attempts(), blocked_attempts()
assert not process_attempts, process_attempts
db.close()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("scenario", ["create", "update", "delete"])
@pytest.mark.parametrize("state", ["inactive", "ordinary"])
def test_remote_scheduling_edits_keep_local_fallback(tmp_path, scenario, state):
    _run(tmp_path, scenario, state, script=_REMOTE_SCRIPT)


_CANCEL_SCRIPT = (
    _APP_SCRIPT.split("effects = []")[0]
    + r"""
import threading
from tldw_chatbook.Backup_Recovery.activation import ActivationStore
from tldw_chatbook.Backup_Recovery import storage_admission as storage
for owner in owners:
    ActivationStore(control / 'activation').approve('generation', owner)
app = SimpleNamespace()
app._speech_initialization_allowed = MethodType(TldwCli._speech_initialization_allowed, app)
entered, release = threading.Event(), threading.Event()
async def initialize():
    def native():
        entered.set()
        assert release.wait(5)
    await asyncio.to_thread(native)
async def main():
    with storage._lock:
        before = set(storage._live_leases)
    waiter = asyncio.create_task(TldwCli._run_speech_initialization(app, 'tts', initialize))
    try:
        for _ in range(100):
            if entered.is_set(): break
            await asyncio.sleep(.01)
        assert entered.is_set()
        waiter.cancel()
        await asyncio.sleep(.02)
        assert not waiter.done(), 'cancelled wrapper detached initializer'
        with storage._lock:
            retained = set(storage._live_leases) - before
        assert retained
        assert all(lease.execution_scope()[1] == ('profile',) for lease in retained)
    finally:
        release.set()
        await asyncio.gather(waiter, return_exceptions=True)
    with storage._lock:
        assert set(storage._live_leases) == before
asyncio.run(main())
assert not blocked_attempts()
assert not process_attempts, process_attempts
print('retired and reopened')
"""
)


def test_cancelled_speech_waiter_retains_actual_execution_hold(tmp_path):
    _run(tmp_path, "cancel", "approved", script=_CANCEL_SCRIPT)
