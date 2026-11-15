"""Original subscription admission stays finite and off the Textual loop.

Allow 120 seconds for Windows child imports/startup/exit. Native operation,
checkpoint and cancellation deadlines inside each exercise stay independent.
"""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio
import contextvars
import faulthandler
import inspect
import sys
import threading
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem
from contextlib import contextmanager

if sys.argv[2] in ('thread', 'custom_result'):
    def dump_stalled_child(stderr=sys.stderr):
        waiter = threading.Event()
        while not waiter.wait(30):
            faulthandler.dump_traceback(file=stderr, all_threads=True)

    threading.Thread(target=dump_stalled_child,
                     name='subscription-boot-diagnostic', daemon=True).start()

from loguru import logger
from textual.app import App
from textual.worker import Worker, WorkerCancelled, WorkerFailed, WorkerState, get_current_worker

from Tests import network_guard, real_profile_guard
network_guard.install()
real_profile_guard.install()
from tldw_chatbook import config
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import activation, storage_admission as storage
from tldw_chatbook.Chat.console_preparation_reads import preparation_reads_for
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Utils.boot_worker_policy import StaggeredBootWorkerGate
from tldw_chatbook.Utils.text_selection_crash_guard import TextualAppGuards

route, case = sys.argv[1:]
assert route == 'subscriptions_boot_owned'
key = 'subscriptions_fts_backfill'
main = threading.current_thread()
marker = contextvars.ContextVar('subscription_boot_marker', default='absent')
result = object()
body_error = LookupError('subscription_body_error')


class WorkflowReached(Exception):
    pass


class Host(TextualAppGuards, App):
    # Shipping starter and slot ownership on a real minimal Textual App.
    # Lookup at call time lets the original starter supply the causal RED.
    boot_worker_starters = TldwCli.boot_worker_starters
    _start_boot_worker = TldwCli._start_boot_worker
    _admit_staggered_boot_workers = TldwCli._admit_staggered_boot_workers
    _release_boot_worker_slot = TldwCli._release_boot_worker_slot
    _arm_boot_worker_reconcile = TldwCli._arm_boot_worker_reconcile
    _reconcile_boot_worker_slots = TldwCli._reconcile_boot_worker_slots
    _close_boot_worker_gate = TldwCli._close_boot_worker_gate
    _stop_boot_worker_reconcile = TldwCli._stop_boot_worker_reconcile
    _shutdown_actor_pack_recovery = TldwCli._shutdown_actor_pack_recovery

    def _start_subscriptions_fts_backfill_worker(self, path, admission):
        return TldwCli._start_subscriptions_fts_backfill_worker(self, path, admission)

    def __init__(self, database):
        super().__init__()
        self.subscriptions_db = database
        self.loguru_logger = logger
        self._boot_worker_gate = StaggeredBootWorkerGate((key,), 1)
        self._boot_worker_handles = {}
        self._boot_worker_reconcile_timer = None
        self._actor_pack_recovery_reads = set()
        self._shutting_down = False
        self.workflow = []
        self.fatal_errors = []
        self.terminal_entered = asyncio.Event()
        self.terminal_release = asyncio.Event()
        self.terminal_finished = asyncio.Event()

    async def _shutdown_workflow_session(self):
        self.workflow.append(True)
        raise WorkflowReached()

    async def on_worker_state_changed(self, event):
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            held = case == 'stale_terminal' and not self.terminal_entered.is_set()
            if held:
                self.terminal_entered.set()
                await self.terminal_release.wait()
            self._release_boot_worker_slot(event.worker)
            if held:
                self.terminal_finished.set()

    def _handle_exception(self, error):
        if case == 'body_error':
            self.fatal_errors.append(error)
        else:
            super()._handle_exception(error)


@contextmanager
def monitor(callbacks):
    api = sys.monitoring
    slot = next(slot for slot in range(6) if api.get_tool(slot) is None)
    api.use_tool_id(slot, 'subscription-boot-native')
    selected, masks = {}, {}
    for code, event, callback in callbacks:
        selected.setdefault(event, {})[code] = callback
        masks[code] = masks.get(code, 0) | event
    registered = {}
    for event, entries in selected.items():
        def dispatch(code, offset, *arguments, entries=entries):
            entries[code](sys._getframe(1), *arguments)
        registered[event] = dispatch
        assert api.register_callback(slot, event, dispatch) is None
    for code, mask in masks.items():
        api.set_local_events(slot, code, mask)
    try:
        yield
    finally:
        for code in masks:
            api.set_local_events(slot, code, 0)
        for event, callback in registered.items():
            assert api.register_callback(slot, event, None) is callback
        api.free_tool_id(slot)


def capture_work_item(observed):
    frame = sys._getframe(1)
    try:
        while frame is not None:
            if (frame.f_code is _WorkItem.run.__code__
                    and frame.f_globals is _WorkItem.run.__globals__):
                item = frame.f_locals['self']
                assert type(item) is _WorkItem and type(item.future) is Future
                assert item.future.running() and not item.future.done()
                observed['future'] = item.future
                return
            frame = frame.f_back
        raise AssertionError('actual_subscription_work_item_missing')
    finally:
        del frame


async def exercise():
    database = SubscriptionsDB(config.get_subscriptions_db_path())
    host = Host(database)
    calls, gates, observed = [], [], {}
    original_admission = activation.execution_allowed
    original_code = original_admission.__code__
    worker = None
    pause = None

    def body():
        calls.append(threading.current_thread())
        observed['worker'] = get_current_worker()
        observed['marker'] = marker.get()
        capture_work_item(observed)
        if case == 'body_error':
            raise body_error
        return result

    host._backfill_subscription_items_fts = body

    def gate_start(frame):
        if frame.f_locals.get('owners') == ('db.subscriptions',):
            gates.append((threading.current_thread(), frame.f_locals['path']))

    try:
        async with host.run_test() as pilot:
            token = marker.set('loop-only')
            try:
                if case == 'denied':
                    pause = storage._begin_local_pause()
                with monitor(((original_code, sys.monitoring.events.PY_START, gate_start),)):
                    assert host._boot_worker_gate.admit() == (key,)
                    worker = host.boot_worker_starters()[key]()
                    if case == 'queued':
                        host._close_boot_worker_gate('queued subscription control')
                    if worker is not None:
                        try:
                            await worker.wait()
                        except WorkerFailed:
                            assert case == 'body_error'
                if case == 'thread':
                    # Causal RED: the original real gate runs in the starter
                    # on the UI thread, without replacing a product callable.
                    assert gates and all(thread is not main for thread, _ in gates), gates
                assert type(worker) is Worker
                assert worker.name == '_backfill_subscription_items_fts'
                assert worker.group == 'subscriptions-fts-backfill'
                assert worker.exit_on_error is True
                if case == 'queued':
                    assert not gates and not calls
                else:
                    assert len(gates) == 1 and gates[0][1] == database.db_path
                    assert gates[0][0] is not main
                    if case == 'denied':
                        assert not calls and worker.result is None
                    else:
                        assert calls == [gates[0][0]]
                        assert observed['worker'] is worker
                        assert observed['marker'] == 'absent'
                        assert observed['future'].done()
                        if case == 'body_error':
                            assert worker.error is body_error
                            assert len(host.fatal_errors) == 1
                            assert host.fatal_errors[0].error is body_error
                        else:
                            assert worker.result is result
                assert not preparation_reads_for(host._actor_pack_recovery_reads)
                assert activation.execution_allowed is original_admission
                assert original_admission.__code__ is original_code
                if pause is not None:
                    pause.resume()
                    pause = None
            finally:
                marker.reset(token)
                if pause is not None:
                    pause.resume()
                if worker is not None:
                    try:
                        await worker.wait()
                    except (WorkerCancelled, WorkerFailed):
                        pass
                host._stop_boot_worker_reconcile()
    finally:
        database.close()
    assert not network_guard.blocked_attempts()
    print('retired and reopened')


asyncio.run(exercise())
"""


@pytest.mark.parametrize(
    "case", ["thread", "custom_result", "denied", "queued", "body_error"]
)
def test_subscription_boot_admission_and_callback_contract(tmp_path, case):
    _run(tmp_path, "subscriptions_boot_owned", case, script=_SCRIPT, timeout=120)


_COMMON_SCRIPT = _SCRIPT.split("async def exercise():", 1)[0]

_HELD_SCRIPT = (
    _COMMON_SCRIPT
    + r"""
async def exercise():
    from tldw_chatbook.Backup_Recovery import runtime_producer_lifetime as lifetime

    database = SubscriptionsDB(config.get_subscriptions_db_path())
    database.close()
    host = Host(database)
    observed, calls = {}, []
    entered, draining = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    worker = shutdown = None
    original = activation.execution_allowed
    original_code = original.__code__

    def body():
        calls.append(get_current_worker())
        return result

    host._backfill_subscription_items_fts = body

    def scope_yield(frame, allowed):
        if frame.f_locals.get('owners') != ('db.subscriptions',) or observed:
            return
        caller = frame.f_back
        while caller is not None and caller.f_code is not original_code:
            caller = caller.f_back
        if caller is None:
            return
        observed['thread'] = threading.current_thread()
        if observed['thread'] is main:
            return  # The causal thread assertion reports base without blocking it.
        assert allowed is True
        reads = preparation_reads_for(host._actor_pack_recovery_reads)
        assert len(reads) == 1
        observed['read'] = reads[0]
        observed['lease'] = frame.f_locals['lease']
        capture_work_item(observed)
        with storage._lock:
            assert observed['lease'] in storage._live_leases
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(20), 'held original subscription gate was not released'

    def drain_yield(frame, value):
        if (observed.get('read') is not None
                and any(read is observed['read'] for read in frame.f_locals.get('pending', ()))):
            draining.set()

    callbacks = (
        (inspect.unwrap(activation.execution_scope).__code__,
         sys.monitoring.events.PY_YIELD, scope_yield),
        (lifetime.drain_preparation_reads.__code__,
         sys.monitoring.events.PY_YIELD, drain_yield),
    )
    try:
        async with host.run_test() as pilot:
            with monitor(callbacks):
                try:
                    assert host._boot_worker_gate.admit() == (key,)
                    worker = host.boot_worker_starters()[key]()
                    assert observed.get('thread') is not main
                    await asyncio.wait_for(entered.wait(), 10)
                    read = observed['read']
                    assert not calls and not read.retired.done()
                    for _ in range(3):
                        worker.cancel()
                        await asyncio.sleep(0)
                        assert not worker.is_finished and not observed['future'].done()
                        assert read in preparation_reads_for(host._actor_pack_recovery_reads)
                    entry = (TldwCli._shutdown_app_owned_lifecycles
                             if case == 'owned_lifecycles' else TldwCli.on_unmount)
                    shutdown = asyncio.create_task(entry(host))
                    await asyncio.wait_for(draining.wait(), 5)
                    for _ in range(2):
                        shutdown.cancel()
                        await asyncio.sleep(0)
                        assert not shutdown.done() and not host.workflow
                        assert not read.retired.done() and not observed['future'].done()
                        with storage._lock:
                            assert observed['lease'] in storage._live_leases
                    release.set()
                    try:
                        await asyncio.wait_for(shutdown, 10)
                    except WorkflowReached:
                        pass
                    else:
                        raise AssertionError('shutdown did not reach its original next owner')
                    assert host.workflow == [True] and calls == [worker]
                    assert observed['future'].done() and read.retired.done()
                    assert not preparation_reads_for(host._actor_pack_recovery_reads)
                    with storage._lock:
                        assert observed['lease'] not in storage._live_leases
                    try:
                        await worker.wait()
                    except WorkerCancelled:
                        pass
                    else:
                        raise AssertionError('worker cancellation was lost')
                    assert activation.execution_allowed is original
                    assert original.__code__ is original_code
                finally:
                    release.set()
                    if 'future' in observed:
                        await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(observed['future'])), 10)
                    if shutdown is not None:
                        try:
                            await shutdown
                        except (WorkflowReached, asyncio.CancelledError):
                            pass
                    if worker is not None:
                        try:
                            await worker.wait()
                        except (WorkerCancelled, WorkerFailed):
                            pass
                    host._stop_boot_worker_reconcile()
    finally:
        release.set()
        database.close()
    assert not network_guard.blocked_attempts()
    print('retired and reopened')


asyncio.run(exercise())
"""
)


@pytest.mark.parametrize("case", ["owned_lifecycles", "direct_unmount"])
def test_subscription_gate_retains_native_lease_through_app_shutdown(tmp_path, case):
    _run(tmp_path, "subscriptions_boot_owned", case, script=_HELD_SCRIPT, timeout=120)


_ERROR_SCRIPT = (
    _COMMON_SCRIPT
    + r"""
async def exercise():
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    database = SubscriptionsDB(config.get_subscriptions_db_path())
    database.close()
    host = Host(database)
    calls, attempts, warnings = [], [], []
    original_admission = activation.execution_allowed
    error = (RecoveryRequired('storage_locally_paused') if case in ('gate_pause', 'stale_terminal')
             else LookupError('subscription_gate_error'))
    second_entered = asyncio.Event()
    second_release = threading.Event()
    loop = asyncio.get_running_loop()

    class Log:
        def opt(self, **kwargs):
            return self

        def warning(self, message, *args):
            warnings.append(str(message))

        def debug(self, *args):
            pass

    host.loguru_logger = Log()
    host._backfill_subscription_items_fts = lambda: calls.append(True)

    def admission(owners, path):
        attempts.append((threading.current_thread(), owners, path))
        if len(attempts) == 1:
            raise error
        if case == 'stale_terminal':
            loop.call_soon_threadsafe(second_entered.set)
            assert second_release.wait(20), 'replacement subscription gate was not released'
        return False

    activation.execution_allowed = admission
    workers = []
    try:
        async with host.run_test() as pilot:
            try:
                host._admit_staggered_boot_workers()
                first = host._boot_worker_handles.get(key)
                assert type(first) is Worker, 'gate ran before its finite worker existed'
                workers.append(first)
                await first.wait()
                if case == 'stale_terminal':
                    await asyncio.wait_for(host.terminal_entered.wait(), 5)
                else:
                    await pilot.pause()
                assert len(attempts) == 1 and attempts[0][0] is not main
                assert attempts[0][1:] == (('db.subscriptions',), database.db_path)
                assert not calls and not host.fatal_errors and first.error is None
                assert not preparation_reads_for(host._actor_pack_recovery_reads)
                if case in ('gate_pause', 'stale_terminal'):
                    assert host._boot_worker_gate.pending == (key,)
                    assert host._boot_worker_gate.in_flight == ()
                    assert key not in host._boot_worker_handles
                    host._reconcile_boot_worker_slots()
                    second = host._boot_worker_handles.get(key)
                    assert type(second) is Worker and second is not first
                    workers.append(second)
                    if case == 'stale_terminal':
                        await asyncio.wait_for(second_entered.wait(), 10)
                        host.terminal_release.set()
                        await asyncio.wait_for(host.terminal_finished.wait(), 5)
                        assert host._boot_worker_handles.get(key) is second
                        assert host._boot_worker_gate.in_flight == (key,), (
                            'the retired carrier completed its replacement slot')
                        assert not second.is_finished
                        second_release.set()
                    await second.wait()
                    await pilot.pause()
                    assert len(attempts) == 2 and not calls
                else:
                    assert any('failed to start' in message for message in warnings), warnings
                assert host._boot_worker_gate.is_drained
            finally:
                host.terminal_release.set()
                second_release.set()
                for worker in workers:
                    try:
                        await worker.wait()
                    except (WorkerCancelled, WorkerFailed):
                        pass
                host._stop_boot_worker_reconcile()
    finally:
        activation.execution_allowed = original_admission
        database.close()
    assert not network_guard.blocked_attempts()
    print('retired and reopened')


asyncio.run(exercise())
"""
)


@pytest.mark.parametrize("case", ["gate_error", "gate_pause", "stale_terminal"])
def test_subscription_gate_errors_keep_starter_warning_and_retry_policy(tmp_path, case):
    _run(tmp_path, "subscriptions_boot_owned", case, script=_ERROR_SCRIPT, timeout=120)


_EFFECT_SCRIPT = (
    _COMMON_SCRIPT
    + r"""
async def exercise():
    from types import MethodType

    database = SubscriptionsDB(config.get_subscriptions_db_path())
    database.close()
    host = Host(database)
    host._backfill_subscription_items_fts = MethodType(TldwCli._backfill_subscription_items_fts, host)
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    observed, owned_entries, gates = {}, [], []
    pause = worker = None
    body_code = TldwCli._backfill_subscription_items_fts.__code__
    scope_code = inspect.unwrap(activation.execution_scope).__code__

    def gate_return(frame, allowed):
        if frame.f_locals.get('owners') == ('db.subscriptions',):
            gates.append(allowed)

    def scope_start(frame):
        if frame.f_locals.get('owners') != ('db.subscriptions',) or observed:
            return
        caller = frame.f_back
        while caller is not None and caller.f_code is not body_code:
            caller = caller.f_back
        if caller is None:
            return
        # The original body already resolved its path; the original actual
        # execution scope has not entered native admission yet.
        observed['thread'] = threading.current_thread()
        observed['path'] = frame.f_locals['path']
        assert observed['thread'] is not main
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(20), 'actual effect scope was not released'

    def scope_yield(frame, allowed):
        if threading.current_thread() is observed.get('thread'):
            observed['effect_allowed'] = allowed

    def owned_start(frame):
        if frame.f_locals.get('self') is host:
            owned_entries.append(True)

    callbacks = (
        (activation.execution_allowed.__code__, sys.monitoring.events.PY_RETURN, gate_return),
        (scope_code, sys.monitoring.events.PY_START, scope_start),
        (scope_code, sys.monitoring.events.PY_YIELD, scope_yield),
        (TldwCli._backfill_subscription_items_fts_owned.__code__,
         sys.monitoring.events.PY_START, owned_start),
    )
    try:
        async with host.run_test() as pilot:
            with monitor(callbacks):
                try:
                    assert host._boot_worker_gate.admit() == (key,)
                    worker = host.boot_worker_starters()[key]()
                    await asyncio.wait_for(entered.wait(), 10)
                    assert gates == [True] and not owned_entries
                    assert observed['path'] == database.db_path
                    pause = storage._begin_local_pause()
                    release.set()
                    await asyncio.wait_for(worker.wait(), 10)
                    assert observed['effect_allowed'] is False
                    assert not owned_entries and worker.result is None
                    assert not preparation_reads_for(host._actor_pack_recovery_reads)
                finally:
                    release.set()
                    if worker is not None:
                        try:
                            await worker.wait()
                        except (WorkerCancelled, WorkerFailed):
                            pass
                    if pause is not None:
                        pause.resume()
                    host._stop_boot_worker_reconcile()
    finally:
        release.set()
        database.close()
    assert not network_guard.blocked_attempts()
    print('retired and reopened')


asyncio.run(exercise())
"""
)


def test_subscription_body_keeps_fresh_actual_execution_scope(tmp_path):
    _run(
        tmp_path,
        "subscriptions_boot_owned",
        "effect_pause",
        script=_EFFECT_SCRIPT,
        timeout=120,
    )
