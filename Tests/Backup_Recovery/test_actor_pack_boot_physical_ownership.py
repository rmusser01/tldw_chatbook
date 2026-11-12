"""Finite real-worker/native controls; full TldwCli mount remains separate."""

import pytest
from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r'''
import asyncio, contextvars, sqlite3, sys, threading
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem
from pathlib import Path
from textual.app import App
from textual.worker import Worker, WorkerCancelled, WorkerFailed, get_current_worker
from loguru import logger
from Tests import network_guard, real_profile_guard
network_guard.install()
real_profile_guard.install()
from tldw_chatbook.app import TldwCli
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from tldw_chatbook.Character_Chat.local_character_persona_service import LocalCharacterPersonaService
from tldw_chatbook.Utils.boot_worker_policy import StaggeredBootWorkerGate, STAGGERED_BOOT_WORKER_KEYS, MAX_CONCURRENT_STAGGERED_BOOT_WORKERS
from tldw_chatbook.Utils.text_selection_crash_guard import TextualAppGuards
from tldw_chatbook.Chat.console_preparation_reads import preparation_reads_for

route, case = sys.argv[1:]
assert route == 'actor_boot_owned'
assert case in {'owned_lifecycles', 'direct_unmount', 'queued', 'custom_result', 'custom_error'}

class WorkflowReached(Exception): pass

class Host(TextualAppGuards, App):
    # Actual shipping starter/lifecycle bodies on a controlled real Textual App.
    # This is not a complete TldwCli constructor/mount qualification.
    boot_worker_starters = TldwCli.boot_worker_starters
    _start_actor_pack_recovery_worker = TldwCli._start_actor_pack_recovery_worker
    _close_boot_worker_gate = TldwCli._close_boot_worker_gate
    _stop_boot_worker_reconcile = TldwCli._stop_boot_worker_reconcile
    ensure_actor_pack_recovery = TldwCli.ensure_actor_pack_recovery

    def __init__(self, database):
        super().__init__()
        self.chachanotes_db = database
        self.loguru_logger = logger
        self._boot_worker_gate = StaggeredBootWorkerGate(STAGGERED_BOOT_WORKER_KEYS, MAX_CONCURRENT_STAGGERED_BOOT_WORKERS)
        self._boot_worker_reconcile_timer = None
        self._actor_pack_recovery_reads = set()
        self._shutting_down = False
        self.actor_pack_recovery_error = 'already_blocked'
        self.actor_pack_repository = ActorPackRepository(database)
        self.persona_actor_pack_coordinator = PersonaActorPackCoordinator(
            self.actor_pack_repository,
            LocalCharacterPersonaService(database, persona_store_path=Path.home()/'personas.json'))
        self.workflow = []

    async def _shutdown_workflow_session(self):
        self.workflow.append(True)
        # Stop at the first existing lifecycle boundary, after the proposed drain.
        raise WorkflowReached()

# Preserve the actual installed optional helper; no dummy fallback implementation.
if hasattr(TldwCli, '_shutdown_actor_pack_recovery'):
    Host._shutdown_actor_pack_recovery = TldwCli._shutdown_actor_pack_recovery

entered, release = threading.Event(), threading.Event()
observed = {}
main = threading.current_thread()
marker = contextvars.ContextVar('actor_boot_test_marker', default='absent')
custom_result = object()
custom_error = LookupError('actor_boot_control_error')

def capture_work_item():
    frame = sys._getframe(1)
    try:
        while frame is not None:
            if frame.f_code is _WorkItem.run.__code__ and frame.f_globals is _WorkItem.run.__globals__:
                item = frame.f_locals['self']
                assert type(item) is _WorkItem and type(item.future) is Future
                assert item.future.running() and not item.future.done()
                observed['item'], observed['future'] = item, item.future
                return
            frame = frame.f_back
        raise AssertionError('actual_work_item_missing')
    finally:
        del frame

async def settle(predicate):
    deadline = asyncio.get_running_loop().time() + 10
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, 'actor_boot_control_timeout'
        await asyncio.sleep(.005)

async def exercise():
    database = config.get_chachanotes_db_lazy()
    host = Host(database)
    database.close_connection()
    original = host.actor_pack_repository.list_persona_intents
    worker = shutdown = None
    wrapped_future = None
    calls = []

    def query(*args, **kwargs):
        result = original(*args, **kwargs)
        connection = database.get_connection()
        observed['connection'], observed['thread'] = connection, threading.current_thread()
        observed['worker'] = get_current_worker()
        capture_work_item()
        entered.set()
        assert release.wait(10), 'actor_boot_release_timeout'
        return result

    def custom():
        calls.append(True)
        observed['thread'], observed['worker'] = threading.current_thread(), get_current_worker()
        observed['marker'] = marker.get()
        capture_work_item()
        if case == 'custom_error': raise custom_error
        return custom_result

    try:
        if case in {'owned_lifecycles', 'direct_unmount'}:
            host.actor_pack_repository.list_persona_intents = query
        else:
            host.ensure_actor_pack_recovery = custom
        async with host.run_test() as pilot:
            token = marker.set('loop-only')
            try:
                assert host._boot_worker_gate.admit() == ('actor_pack_recovery',)
                worker = host.boot_worker_starters()['actor_pack_recovery']()
                assert type(worker) is Worker and worker.name == 'deferred_actor_pack_recovery'
                assert worker.group == 'actor_pack_recovery' and worker.exit_on_error is False
                if case == 'queued':
                    host._close_boot_worker_gate('queued control')
                    await worker.wait()
                    assert calls == [] and not observed
                    assert not preparation_reads_for(host._actor_pack_recovery_reads)
                elif case.startswith('custom_'):
                    try: await worker.wait()
                    except WorkerFailed:
                        assert case == 'custom_error'
                    assert calls == [True]
                    assert observed['thread'] is not main and observed['worker'] is worker
                    assert observed['marker'] == 'absent'
                    assert observed['future'].done()
                    if case == 'custom_error': assert worker.error is custom_error
                    else: assert worker.result is custom_result
                    assert not preparation_reads_for(host._actor_pack_recovery_reads)
                else:
                    await settle(entered.is_set)
                    assert observed['thread'] is not main and observed['worker'] is worker
                    assert observed['future'].running() and not observed['future'].done()
                    with storage._lock:
                        assert any(lease.resource_thread is observed['thread'] for lease in storage._live_leases)
                    worker.cancel()
                    worker.cancel()
                    entry = TldwCli._shutdown_app_owned_lifecycles if case == 'owned_lifecycles' else TldwCli.on_unmount
                    shutdown = asyncio.create_task(entry(host))
                    await asyncio.sleep(0)
                    shutdown.cancel()
                    await asyncio.sleep(0)
                    shutdown.cancel()
                    await asyncio.sleep(0)
                    assert not host.workflow, 'workflow began before physical boot retirement'
                    assert not shutdown.done() and not observed['future'].done()
                    assert preparation_reads_for(host._actor_pack_recovery_reads)
                    release.set()
                    try: await shutdown
                    except WorkflowReached: pass
                    else: raise AssertionError('original first workflow boundary not reached')
                    assert host.workflow == [True]
                    assert observed['future'].done()
                    assert not preparation_reads_for(host._actor_pack_recovery_reads)
                    try: await worker.wait()
                    except WorkerCancelled: pass
                    try: sqlite3.Connection.in_transaction.__get__(observed['connection'])
                    except sqlite3.ProgrammingError: pass
                    else: raise AssertionError('original operation-owned connection remains open')
                    with storage._lock:
                        assert not [op for op in storage._operations if op.participant is database._maintenance_participant]
                        assert not [lease for lease in storage._live_leases if lease.resource_thread is observed['thread']]
            finally:
                marker.reset(token)
                release.set()
                if 'future' in observed:
                    wrapped_future = asyncio.wrap_future(observed['future'])
                    try: await asyncio.wait_for(asyncio.shield(wrapped_future), 10)
                    except LookupError:
                        assert case == 'custom_error'
                if shutdown is not None:
                    try: await shutdown
                    except (WorkflowReached, asyncio.CancelledError): pass
                if worker is not None:
                    try: await worker.wait()
                    except (WorkerCancelled, WorkerFailed): pass
    finally:
        release.set()
        host.actor_pack_repository.list_persona_intents = original
        # The exact captured native Future must join before fixture-owned DB close.
        if 'future' in observed: assert observed['future'].done()
        database.close_connection()

    reopened = database.get_connection()
    try:
        assert reopened.execute('SELECT 1').fetchone()[0] == 1
    finally:
        database.close_connection()
    print('retired and reopened')

asyncio.run(exercise())
'''


@pytest.mark.parametrize('case', ['owned_lifecycles', 'direct_unmount', 'queued', 'custom_result', 'custom_error'])
def test_actor_pack_boot_retains_physical_outcome(tmp_path, case):
    _run(tmp_path, 'actor_boot_owned', case, script=_SCRIPT, timeout=60)
