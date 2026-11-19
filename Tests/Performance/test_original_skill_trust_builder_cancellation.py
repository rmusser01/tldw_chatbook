"""Original async ensure must retain its actual builder through cancellation."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import Future
from concurrent.futures import thread as executor_source

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]
assert route == 'skill_trust_builder' and outcome in ('normal', 'cancel', 'recancel', 'inner_cancel', 'winner')

def control():
    selected = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    root = Path(os.environ['XDG_DATA_HOME']).absolute()
    selected.write_text('[general]\nusers_name="skill-builder-control"\n[paths]\ndata_dir="'
                        + root.as_posix() + '"\n', encoding='utf-8')
    selected.chmod(0o600)
    from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
    from tldw_chatbook.app_service_wiring import ServiceWiringMixin
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    builder = inspect.getattr_static(ServiceWiringMixin, '_build_local_skill_trust_service')
    ensure = inspect.getattr_static(ServiceWiringMixin, 'ensure_local_skill_trust_service')
    work_run = inspect.getattr_static(executor_source._WorkItem, 'run')
    monitor = OriginalStorageUnitObserver({}, False, lambda _name: None)
    builder_code = monitor._pin(builder)
    monitor._pin(ensure)
    work_code = monitor._pin(work_run)
    monitor.slots.extend(((ServiceWiringMixin, '_build_local_skill_trust_service', builder),
                          (ServiceWiringMixin, 'ensure_local_skill_trust_service', ensure),
                          (executor_source._WorkItem, 'run', work_run)))
    entered = threading.Event()
    release = threading.Event()
    physical_return = threading.Event()
    invalid = []
    captured = []
    task = None
    main = threading.current_thread()
    owner = ServiceWiringMixin()
    owner._local_skill_trust_service = None
    owner._local_skill_trust_service_build_lock = asyncio.Lock()
    # This real composition owner isolates the original existing API. It is
    # not a shipping App setup, a replaced builder or a new permission route.
    assert type(owner) is ServiceWiringMixin

    def counts():
        with storage._lock:
            return {'ordinary': len(storage._live_leases - set(storage._startups.values())),
                    'pending': len(storage._pending_acquisitions),
                    'core': len(storage._operations), 'raw': len(storage._raw_operations),
                    'retiring': len(storage._retiring_holds)}

    baseline = counts()
    facts = {'route': outcome, 'baseline': baseline, 'real_original_mixin_owner': True,
             'shipping_App_or_full_mount_claimed': False, 'guard_replacements': False,
             'hold_bound_s': 10, 'settle_bound_s': 10}

    def returned(actual_code, offset, value):
        if actual_code is not builder_code or entered.is_set():
            return
        frame = parent = item = future = worker = None
        try:
            frame = sys._getframe(1)
            assert frame.f_code is builder_code and frame.f_globals is builder.__globals__
            assert frame.f_locals['self'] is owner
            worker = threading.current_thread()
            assert worker is not main
            parent = frame.f_back
            if parent is not None and parent.f_code is not work_code:
                # The shared finite reader adds one original invocation frame.
                # Prove its handle owns this exact callback and private Future.
                from tldw_chatbook.Chat import console_preparation_reads as reads_source
                invoke_code = next(code for code in reads_source.run_preparation_read.__code__.co_consts
                                   if isinstance(code, type(work_code)) and code.co_name == 'invoke')
                assert parent.f_code is invoke_code
                assert parent.f_globals is vars(reads_source)
                read = parent.f_locals['read']
                assert type(read) is reads_source.ConsolePreparationRead
                assert read.creator is owner and read.callback.__self__ is owner
                assert read.callback.__func__ is builder
                assert read.task is task and not read.retired.done()
                assert type(read._producer) is asyncio.Future and not read._producer.done()
                facts['private_physical_producer'] = True
                parent = parent.f_back
            assert parent is not None and parent.f_code is work_code
            assert parent.f_globals is work_run.__globals__
            item = parent.f_locals['self']
            assert type(item) is executor_source._WorkItem
            future = item.future
            assert type(future) is Future and future.running() and not future.done()
            captured.append((worker, future))
            facts['original_builder_return_on_actual_executor'] = True
            facts['original_native_counts_at_held_return'] = counts()
        except BaseException as error:
            if len(invalid) < 12:
                invalid.append('builder_return:' + type(error).__name__)
            return
        finally:
            del frame, parent, item, future, worker, value
        # Hold only this exact callback. No transient Python frames/results
        # survive into the blocking test-owned gate.
        entered.set()
        try:
            if not release.wait(10):
                invalid.append('original_builder_release_timeout')
        finally:
            physical_return.set()

    async def settle(predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            assert time.monotonic() < deadline, 'original_skill_builder_prerequisite_timeout'
            await asyncio.sleep(.005)

    async def run():
        nonlocal task
        loop = asyncio.get_running_loop()
        observer = None
        try:
            for tool in range(5, 0, -1):
                if tool == sys.monitoring.DEBUGGER_ID:
                    continue
                try:
                    sys.monitoring.use_tool_id(tool, 'original-skill-builder-return')
                except ValueError:
                    continue
                monitor.tool = tool
                break
            assert monitor.tool is not None and sys.monitoring.get_events(monitor.tool) == 0
            event = sys.monitoring.events.PY_RETURN
            assert sys.monitoring.register_callback(monitor.tool, event, returned) is None
            monitor.registered[event] = returned
            monitor.codes[builder_code] = 'actual_original_skill_builder_return'
            assert sys.monitoring.get_local_events(monitor.tool, builder_code) == 0
            sys.monitoring.set_local_events(monitor.tool, builder_code, event)
            monitor.active = monitor.installed = True
            task = asyncio.create_task(ensure(owner))
            assert type(task) is asyncio.Task and task.get_loop() is loop
            assert task.get_coro().cr_code is ensure.__code__
            await settle(entered.is_set)
            assert not invalid and len(captured) == 1
            worker, future = captured[0]
            assert worker.is_alive() and future.running() and not future.done()
            assert owner._local_skill_trust_service_build_lock.locked()
            if outcome == 'winner':
                winner = object()
                owner._local_skill_trust_service = winner
            if outcome not in {'normal', 'winner'}:
                if outcome == 'inner_cancel':
                    inner = task.get_coro().cr_frame.f_locals.get('preparation')
                    if type(inner) is asyncio.Task:
                        assert inner.get_coro().cr_code is asyncio.to_thread.__code__
                        assert not inner.done()
                        facts['cancelled_original_detachable_inner_Task'] = True
                        inner.cancel()
                    else:
                        assert facts.get('private_physical_producer') is True
                        facts['no_detachable_inner_Task'] = True
                        task.cancel()
                else:
                    task.cancel()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                if outcome == 'recancel':
                    task.cancel()
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                # Observe cancellation propagation while the actual callback remains held.
                await asyncio.wait({task}, timeout=0.1)
                facts['cancelled_waiter_done_before_callback_release'] = task.done()
                facts['singleflight_lock_held_before_callback_release'] = owner._local_skill_trust_service_build_lock.locked()
                facts['actual_executor_future_running_before_release'] = future.running() and not future.done()
            release.set()
            await settle(physical_return.is_set)
            # Await the actual original executor Future, even when bare
            # to_thread detached its cancelled waiter. This is fixture cleanup,
            # not evidence that the preceding waiter retained custody.
            result = await asyncio.wait_for(asyncio.wrap_future(future, loop=loop), timeout=10)
            assert future.done() and not future.cancelled() and physical_return.is_set()
            if outcome in {'normal', 'winner'}:
                if outcome == 'winner':
                    assert result is not winner
                    result = winner
                assert await asyncio.wait_for(asyncio.shield(task), timeout=10) is result
                assert owner._local_skill_trust_service is result
                assert await ensure(owner) is result
            else:
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=10)
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError('original_cancelled_waiter_not_cancelled')
                assert task.done() and task.cancelled()
                assert owner._local_skill_trust_service is None
            assert not owner._local_skill_trust_service_build_lock.locked()
            assert counts() == baseline
            facts['actual_callback_and_native_baseline_retired'] = True
            facts['task_original_cancellation_or_result_preserved'] = True
        finally:
            release.set()
            try:
                if captured:
                    await settle(physical_return.is_set)
                    await asyncio.wait_for(asyncio.wrap_future(captured[0][1], loop=loop), timeout=10)
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(asyncio.shield(task), timeout=10)
                    except asyncio.CancelledError:
                        pass
            finally:
                if monitor.tool is not None:
                    observer = monitor.close()
                facts['observer'] = observer
                facts['invalid'] = invalid
                facts['final_counts'] = counts()
                pause = storage._begin_local_pause()
                try:
                    facts['original_global_drain'] = pause.drain(time.monotonic() + 1)
                finally:
                    pause.resume()
                facts['network_refusals'] = len(network_guard.blocked_attempts())
                facts['profile_refusals'] = len(real_profile_guard.take_violations())
                (root.parent / ('original-skill-builder-' + outcome + '.json')).write_text(
                    json.dumps(facts, sort_keys=True), encoding='utf-8')
        assert observer['complete'] and observer['original_source_current']
        assert observer['global_events'] == 0 and observer['hooks_retired_before_inactive']
        assert not observer['invalid'] and not invalid
        assert facts['final_counts'] == baseline
        assert facts['original_global_drain']
        assert facts['network_refusals'] == facts['profile_refusals'] == 0
        if outcome not in {'normal', 'winner'}:
            # Sole causal oracles after original source/native/callback cleanup.
            assert facts['actual_executor_future_running_before_release']
            assert not facts['cancelled_waiter_done_before_callback_release'], facts
            assert facts['singleflight_lock_held_before_callback_release'], facts
        print('retired and reopened')

    asyncio.run(run())

with user_fixture_default_owner():
    control()
"""


@pytest.mark.parametrize(
    "outcome", ("normal", "cancel", "recancel", "inner_cancel", "winner")
)
def test_original_skill_trust_builder_retains_native_callback(tmp_path, outcome):
    _run(tmp_path, "skill_trust_builder", outcome, script=_SCRIPT)
