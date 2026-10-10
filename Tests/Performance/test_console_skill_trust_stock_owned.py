"""Root-only actual App controls for AC29; not executed by the implementer.

Breaks caught: stock Main-thread builder, pre-policy setup, ready/custom adoption,
and App shutdown or repeated cancellation releasing a live original callback.
The original private-profile transport and 10-second callback custody bounds
remain; original whole-startup/Send performance gates remain independent.
"""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio, contextvars, functools, inspect, json, os, sqlite3, sys, threading, time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.thread import _WorkItem
from pathlib import Path
from types import BuiltinMethodType, FunctionType
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, case = sys.argv[1:]
assert route == 'console_skill_stock' and case in {'mount', 'cancel', 'recancel', 'shutdown', 'denied', 'ready', 'custom', 'local_winner', 'app_winner', 'runtime_dispose', 'runtime_replacement', 'app_closing', 'helper_binding', 'helper_body', 'metadata_body', 'send_enter', 'send_button'}

foreign_calls = []
_skill_foreign_calls = foreign_calls

def foreign_helper(*args, **kwargs):
    _skill_foreign_calls.append(True)
    return False


def native_closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        return True
    return False


async def exercise():
    from tldw_chatbook import app as app_source, app_service_wiring as wiring
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Utils.text_selection_crash_guard import FreshContextExecutor
    from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
    from textual.worker import Worker
    from textual.worker_manager import WorkerManager
    from tldw_chatbook.UI.Console_Modules.view_workers import capture_console_view_workers

    if case in {'send_enter', 'send_button'}:
        from Tests.Performance._stock_cold_send_control import prepare_cold_send_workspace, observe_cold_send, assert_cold_send_receipt
    from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
    creators = OriginalPrivateAppCreators(app_source, Path(os.environ['XDG_DATA_HOME']).parent)
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    runtime = app.console_runtime
    database = app.chachanotes_db
    creator = getattr(database._local, 'conn', None)
    assert creator is not None and not native_closed(creator)
    facade = app.skills_scope_service
    local = app.local_skills_service
    assert app._local_skill_trust_service is None and local._trust_service is None
    workspace_id = prepare_cold_send_workspace(app, case) if case in {'send_enter', 'send_button'} else None
    loop, main = asyncio.get_running_loop(), threading.current_thread()
    builder = inspect.getattr_static(wiring.ServiceWiringMixin, '_build_local_skill_trust_service')
    ensure = inspect.getattr_static(wiring.ServiceWiringMixin, 'ensure_local_skill_trust_service')
    monitor = OriginalStorageUnitObserver({}, lambda: False, lambda _name: None)
    workspace_connections = {}
    workspace_code = None
    if case in {'send_enter', 'send_button'}:
        workspace_database = app.workspace_registry_service.db
        workspace_binding = inspect.getattr_static(type(workspace_database), '_get_connection')
        workspace_getter = inspect.unwrap(workspace_binding)
        workspace_code = monitor._pin(workspace_getter)
        monitor._pin(workspace_binding)
        monitor.slots.append((type(workspace_database), '_get_connection', workspace_binding))
    builder_code = monitor._pin(builder)
    ensure_code = monitor._pin(ensure)
    from tldw_chatbook.UI.Console_Modules.skill import ConsoleSkillController
    discovery = inspect.getattr_static(ConsoleSkillController, '_refresh_console_skill_candidates')
    discovery_code = monitor._pin(discovery)
    monitor.slots.append((ConsoleSkillController, '_refresh_console_skill_candidates', discovery))
    from tldw_chatbook.Chat import console_preparation_reads as read_source
    physical_read = read_source.run_preparation_read
    monitor._pin(physical_read)
    monitor.slots.append((read_source, 'run_preparation_read', physical_read))
    invoke_code = next(code for code in physical_read.__code__.co_consts
                       if inspect.iscode(code) and code.co_name == 'invoke')
    fresh_submit = inspect.getattr_static(FreshContextExecutor, 'submit')
    fresh_code = monitor._pin(fresh_submit)
    base_submit = inspect.getattr_static(ThreadPoolExecutor, 'submit')
    monitor._pin(base_submit)
    work_code = monitor._pin(_WorkItem.run)
    monitor.slots.extend(((wiring.ServiceWiringMixin, '_build_local_skill_trust_service', builder),
        (wiring.ServiceWiringMixin, 'ensure_local_skill_trust_service', ensure),
        (FreshContextExecutor, 'submit', fresh_submit), (ThreadPoolExecutor, 'submit', base_submit),
        (_WorkItem, 'run', _WorkItem.run), (app_source, 'TldwCli', type(app))))
    entered, release, hold_retired = threading.Event(), threading.Event(), threading.Event()
    progress = threading.Event()
    captured, submitted, invalid, facts = [], [], [], {'case': case, 'constructor_owned': True,
        'App_runtime_native_execution': True, 'callback_bound_s': 10, 'settle_bound_s': 10,
        'normal_loop_hold_s': .35, 'whole_startup_Send_GREEN_claim': False}
    control_task = None
    held_task = None
    held_control_admitted = loop.create_future()
    discovery_admitted = loop.create_future()
    controller_thread = None
    calls = []

    captured_reads = []
    mutation = None
    replacement = None
    from tldw_chatbook.Widgets import compact_model_bar as metadata
    assert '_skill_foreign_calls' not in vars(wiring)
    assert '_skill_foreign_calls' not in vars(metadata)
    wiring._skill_foreign_calls = metadata._skill_foreign_calls = foreign_calls

    def counts():
        with storage._lock:
            return {'ordinary': len(storage._live_leases - set(storage._startups.values())),
                'pending': len(storage._pending_acquisitions), 'core': len(storage._operations),
                'raw': len(storage._raw_operations), 'retiring': len(storage._retiring_holds)}

    # Explicit custom contracts are installed only in the test profile; stock
    # methods and all storage/policy/admission guards remain original.
    if case == 'denied':
        class DeniedPolicy:
            def require_allowed(self, *, action_id):
                calls.append(action_id)
                from tldw_chatbook.runtime_policy.types import PolicyDeniedError
                raise PolicyDeniedError(action_id=action_id, reason_code='authority_denied',
                    user_message='denied by control', effective_source='local', authority_owner='local')
        policy = DeniedPolicy()
        facade.policy_enforcer = local.policy_enforcer = policy
        app._local_skills_stack_inputs = (policy, app._local_skills_stack_inputs[1])
    if case == 'custom':
        class CustomFacade:
            async def get_context(self, *, mode=None):
                calls.append(mode)
                return {'available_skills': [], 'blocked_skills': []}
        app.skills_scope_service = CustomFacade()
    if case == 'ready':
        winner = await app.ensure_local_skill_trust_service()
        assert winner is app._local_skill_trust_service
        facts['original_ready_service_installed_before_monitor'] = True
    if case in {'local_winner', 'app_winner'}:
        donor = wiring.ServiceWiringMixin()
        donor._local_skill_trust_service = None
        donor._local_skill_trust_service_build_lock = asyncio.Lock()
        winner = await donor.ensure_local_skill_trust_service()
        assert app._local_skill_trust_service is None and local._trust_service is None

    def builder_started(actual_code, offset):
        frame = sys._getframe(1)
        try:
            assert frame.f_code is actual_code
            if actual_code is discovery_code:
                if discovery_admitted.done():
                    return
                screen = runtime.view
                assert screen is not None and frame.f_locals.get('self') is screen._skill
                assert frame.f_globals is discovery.__globals__
                assert threading.current_thread() is main and asyncio.get_running_loop() is loop
                current = asyncio.current_task()
                captured_discovery = capture_console_view_workers(app, screen)
                rows = tuple(row for row in captured_discovery[-1]
                             if row[0].group == 'console-skill-discovery' and row[2] is current)
                assert len(rows) == 1
                worker, node, task, work = rows[0]
                assert node is screen and type(task) is asyncio.Task and task.get_loop() is loop
                assert inspect.iscoroutine(work) and work.cr_code is discovery_code
                assert work.cr_frame is frame and frame.f_locals.get('self') is screen._skill
                discovery_admitted.set_result((captured_discovery, rows[0]))
                facts['original_discovery_admitted_at'] = time.monotonic()
                return
            assert actual_code is builder_code
            if frame.f_locals.get('self') is app:
                facts['builder_native_start_at'] = time.monotonic()
                facts['builder_started_off_ui_loop'] = threading.current_thread() is not main
        finally:
            del frame

    def returned(actual_code, offset, value):
        nonlocal controller_thread, held_task
        frame = parent = item = future = request = executor = None
        try:
            frame = sys._getframe(1)
            assert frame.f_code is actual_code
            if actual_code is workspace_code:
                if frame.f_locals.get('self') is not workspace_database:
                    return
                assert frame.f_globals is workspace_getter.__globals__
                assert isinstance(value, sqlite3.Connection)
                assert len(workspace_connections) < 128
                lead = []
                parent = frame.f_back
                for _ in range(24):
                    if parent is None or parent.f_code is work_code:
                        break
                    lead.append((parent.f_globals.get('__name__'), parent.f_code.co_qualname))
                    parent = parent.f_back
                item = parent.f_locals.get('self') if parent is not None and parent.f_code is work_code else None
                if threading.current_thread() is not main:
                    assert type(item) is _WorkItem and type(item.future) is Future and item.future.running()
                workspace_connections[value] = {'thread': threading.current_thread().name, 'original_getter': True, 'caller_lead': lead, 'executor_item': type(item) is _WorkItem}
                return
            if actual_code is fresh_code:
                executor = frame.f_locals['self']
                request = frame.f_locals['fn']
                if type(executor) is not FreshContextExecutor or type(value) is not Future:
                    return
                if type(request) is not BuiltinMethodType:
                    return
                if type(request.__self__) is not contextvars.Context or request.__name__ != 'run':
                    return
                arguments = frame.f_locals['args']
                if len(arguments) != 1 or type(arguments[0]) is not FunctionType:
                    return
                callback = arguments[0]
                if callback.__code__ is not invoke_code or callback.__globals__ is not vars(read_source):
                    return
                cells = dict(zip(callback.__code__.co_freevars, callback.__closure__))
                read = cells['read'].cell_contents
                if (
                    read.creator is not app
                    or getattr(read.callback, '__self__', None) is not app
                    or getattr(read.callback, '__func__', None) is not builder
                ):
                    return
                assert type(read) is read_source.ConsolePreparationRead
                assert read.callback.__self__ is app and read.callback.__func__ is builder
                assert read in runtime._preparation_reads and not read.retired.done()
                request = callback
                if len(submitted) >= 4:
                    raise AssertionError('setup_submit_identity_overflow')
                submitted.append((request, value))
                facts['builder_executor_submit_return_at'] = time.monotonic()
                return
            if actual_code is not builder_code or frame.f_locals.get('self') is not app:
                return
            if entered.is_set():
                raise AssertionError('duplicate_original_stock_builder')
            thread = threading.current_thread()
            parent = frame.f_back
            for _ in range(12):
                if parent is None or parent.f_code is work_code:
                    break
                parent = parent.f_back
            facts['original_builder_off_ui_loop'] = thread is not main
            if thread is not main:
                assert parent is not None and parent.f_code is work_code
                item = parent.f_locals['self']
                assert type(item) is _WorkItem
                future = item.future
                assert type(future) is Future and future.running() and not future.done()
                if type(item.fn) is BuiltinMethodType:
                    assert type(item.fn.__self__) is contextvars.Context and item.fn.__name__ == 'run'
                    assert len(item.args) == 2 and not item.kwargs
                    context_run, request = item.args
                    assert type(context_run) is BuiltinMethodType
                    assert type(context_run.__self__) is contextvars.Context and context_run.__name__ == 'run'
                    assert type(request) is FunctionType and request.__code__ is invoke_code
                    assert request.__globals__ is vars(read_source)
                    cells = dict(zip(request.__code__.co_freevars, request.__closure__))
                    read = cells['read'].cell_contents
                    assert type(read) is read_source.ConsolePreparationRead and read.creator is app
                    assert read.callback.__self__ is app and read.callback.__func__ is builder
                    assert read in runtime._preparation_reads and not read.retired.done()
                    assert type(read._producer) is asyncio.Future and not read._producer.done()
                    captured_reads.append(read)
                    facts['original_callback_uses_private_physical_producer_and_Runtime_observer'] = True
                else:
                    # The original Main/direct baseline cannot borrow stock
                    # wrapper authority; no source-forcing fallback is installed.
                    raise AssertionError('unqualified_stock_executor_wrapper')
                captured.append((thread, future, request))
            facts['builder_native_counts_at_original_held_return'] = counts()
        except BaseException as error:
            if len(invalid) < 12: invalid.append(type(error).__name__ + ':' + str(error)[:120])
            return
        finally:
            del frame, parent, item, future, request, executor, value
        facts['builder_held_at'] = time.monotonic()
        entered.set()
        if case == 'mount':
            controller_thread = threading.Thread(target=coordinate_normal, name='stock_skill_original_hold', daemon=False)
            controller_thread.start()
        elif case in {'cancel', 'recancel', 'shutdown', 'local_winner', 'app_winner', 'runtime_dispose', 'runtime_replacement', 'app_closing', 'helper_binding', 'helper_body', 'metadata_body', 'send_enter', 'send_button'}:
            def admit_held_control():
                nonlocal held_task
                assert held_task is None
                facts['held_control_admitted_at'] = time.monotonic()
                held_task = asyncio.create_task(held_control())
                held_control_admitted.set_result(held_task)
            loop.call_soon_threadsafe(admit_held_control)
        try:
            if not release.wait(10): invalid.append('original_builder_release_timeout')
        finally:
            hold_retired.set()

    async def settle(predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            assert time.monotonic() < deadline, 'stock_skill_control_prerequisite_timeout'
            await asyncio.sleep(.005)

    async def physical_result(future):
        try:
            value = await asyncio.wait_for(asyncio.wrap_future(future, loop=loop), timeout=10)
        except RuntimeError as error:
            assert case in {'shutdown', 'runtime_dispose', 'runtime_replacement', 'app_closing', 'helper_binding', 'helper_body', 'metadata_body'} and error.args == ('console_skill_trust_source_changed',)
            assert future.exception() is error
            facts['worker_rejected_changed_source_before_loop_publication'] = True
        else:
            assert case not in {'shutdown', 'runtime_dispose', 'runtime_replacement', 'app_closing', 'helper_binding', 'helper_body', 'metadata_body'}, 'worker_missed_changed_source'
            return value

    async def held_control():
        nonlocal control_task, mutation, replacement
        facts['held_control_started_at'] = time.monotonic()
        await settle(entered.is_set)
        assert facts['original_builder_off_ui_loop'] and not invalid and len(captured) == 1
        future = captured[0][1]
        screen = runtime.view
        assert screen is not None
        assert any(request is captured[0][2] and issued is future for request, issued in submitted)
        facts['actual_original_FreshContext_wrapper_correspondence'] = True
        manager = vars(app).get('_workers')
        assert type(manager) is WorkerManager and vars(manager).get('_app') is app
        owned = tuple(w for w in manager if type(w) is Worker and vars(w).get('_node') is app and w.group == 'console-skill-trust-setup')
        assert len(owned) == 1
        setup = owned[0]
        issued = vars(setup).get('_task')
        work = vars(setup).get('_work')
        assert type(issued) is asyncio.Task and issued.get_loop() is loop
        assert inspect.iscoroutine(work) and work.cr_code is ensure_code
        assert len(captured_reads) == 1 and captured_reads[0].task is issued
        assert captured_reads[0] in runtime._preparation_reads and not captured_reads[0].retired.done()
        selected = capture_console_view_workers(app)
        assert any(row[0] is setup and row[1] is app and row[2] is issued and row[3] is work for row in selected[-1])
        assert any(row[0].group == 'console-skill-discovery' for row in selected[-1])
        facts['exact_App_manager_issued_setup_and_discovery_selected'] = True
        assert app._local_skill_trust_service_build_lock.locked() and future.running() and not future.done()
        if case in {'send_enter', 'send_button'}:
            await observe_cold_send(app, screen, case, workspace_id, release, facts)
        if case in {'cancel', 'recancel'}:
            setup.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            if case == 'recancel':
                setup.cancel()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            facts['exact_setup_Task_retained_through_cancellation'] = not issued.done()
            assert not issued.done() and app._local_skill_trust_service_build_lock.locked()
        if case == 'local_winner':
            local.trust_service = winner
            assert local._trust_service is winner and app._local_skill_trust_service is None
        if case == 'app_winner':
            app.local_skill_trust_service = winner
            assert app._local_skill_trust_service is winner and local._trust_service is None
        if case == 'runtime_replacement':
            from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
            replacement = ConsoleRuntime(app)
            assert replacement is not runtime and not replacement._disposed
            app.console_runtime = replacement
            assert captured_reads[0] in runtime._preparation_reads
            assert not replacement._preparation_reads and future.running()
        if case in {'runtime_dispose', 'app_closing'}:
            control_task = asyncio.create_task(
                runtime.dispose() if case == 'runtime_dispose' else app.on_shutdown_request()
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not control_task.done() and future.running() and not future.done()
            if case == 'runtime_dispose':
                assert runtime._disposed
            else:
                assert app._shutting_down
                facts['runtime_shutdown_already_started_at_shutdown_request'] = app._console_runtime_shutdown_task is not None
            late = await asyncio.wait_for(screen._skill._fetch_console_skill_context(), timeout=1)
            assert late == {} and future.running() and not future.done()
            assert captured_reads[0] in runtime._preparation_reads
            facts['original_close_refuses_late_context_and_retains_exact_read'] = True
        if case == 'helper_binding':
            mutation = (wiring, '_console_skill_metadata_current', wiring._console_skill_metadata_current)
            wiring._console_skill_metadata_current = foreign_helper
        elif case in {'helper_body', 'metadata_body'}:
            target = wiring._console_skill_source_current if case == 'helper_body' else metadata._plain_fields
            mutation = (target, '__code__', target.__code__)
            assert target.__closure__ is None and foreign_helper.__closure__ is None
            target.__code__ = foreign_helper.__code__
        if case == 'shutdown':
            control_task = asyncio.create_task(app._shutdown_console_runtime())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            shutdown = app._console_runtime_shutdown_task
            assert type(shutdown) is asyncio.Task and not shutdown.done()
            late = await asyncio.wait_for(
                runtime.view._skill._fetch_console_skill_context(), timeout=1
            )
            assert late == {} and future.running() and not future.done()
            assert len(tuple(w for w in manager if type(w) is Worker and vars(w).get('_node') is app and w.group == 'console-skill-trust-setup')) == 1
            facts['late_stock_context_cannot_admit_setup_after_shutdown_capture'] = True
            control_task.cancel()
            await asyncio.sleep(0)
            control_task.cancel()
            await asyncio.sleep(0)
            facts['original_shutdown_retains_callback_through_repeated_waiter_cancel'] = not shutdown.done()
            assert not shutdown.done() and not issued.done()
            assert app._local_skill_trust_service_build_lock.locked() and future.running() and not future.done()
        release.set()
        await settle(hold_retired.is_set)
        await physical_result(future)
        await settle(issued.done)
        assert future.done() and not future.cancelled()
        read = captured_reads[0]
        assert read._producer.done() and not read._producer.cancelled()
        assert read.retired.done() and not read.retired.cancelled()
        if case in {'shutdown', 'runtime_dispose', 'runtime_replacement', 'app_closing', 'helper_binding', 'helper_body', 'metadata_body'}:
            assert read._producer.exception() is future.exception()
            assert type(read._producer.exception()) is RuntimeError
            assert read._producer.exception().args == ('console_skill_trust_source_changed',)
        assert read not in runtime._preparation_reads
        assert not app._local_skill_trust_service_build_lock.locked()
        if case in {'cancel', 'recancel', 'shutdown', 'runtime_dispose', 'runtime_replacement', 'app_closing', 'helper_binding', 'helper_body', 'metadata_body'}:
            assert app._local_skill_trust_service is None
        if mutation is not None:
            target, name, value = mutation
            setattr(target, name, value)
            mutation = None
            assert not foreign_calls
            facts['changed_helper_refused_without_foreign_execution'] = True
        if case == 'runtime_replacement':
            assert app.console_runtime is replacement and app._local_skill_trust_service is None
            await asyncio.wait_for(runtime.dispose(), timeout=10)
            assert app.console_runtime is replacement and runtime._disposed
            facts['replacement_Runtime_receives_no_stale_setup_or_cleanup'] = True
        if case in {'runtime_dispose', 'app_closing'}:
            await asyncio.wait_for(asyncio.shield(control_task), timeout=10)
        if case == 'local_winner':
            assert local.trust_service is winner
            assert app._local_skill_trust_service is not winner
            facts['fresh_local_ready_winner_preserved_during_original_held_setup'] = True
        if case == 'app_winner':
            assert app._local_skill_trust_service is winner
            facts['fresh_App_ready_winner_preserved_during_original_held_setup'] = True
        if case == 'shutdown':
            await asyncio.wait_for(asyncio.shield(app._console_runtime_shutdown_task), timeout=10)
            assert runtime._disposed
            if control_task is not None:
                try: await control_task
                except asyncio.CancelledError: pass
        facts['exact_original_callback_Future_and_singleflight_physically_retired'] = True
        facts['held_control_finished_at'] = time.monotonic()

    def coordinate_normal():
        if not entered.wait(10):
            invalid.append('normal_builder_entry_timeout')
            release.set()
            return
        loop.call_soon_threadsafe(progress.set)
        # Preserve the existing original shared-loop .35-second hold gate.
        progress.wait(.35)
        facts['actual_loop_progress_before_original_builder_release'] = progress.is_set()
        release.set()

    producer_diagnostic = None
    if case in {'custom', 'denied', 'runtime_dispose', 'runtime_replacement'}:
        from Tests.Performance._stock_core_producer_diagnostic import OriginalCoreProducerDiagnostic
        producer_diagnostic = OriginalCoreProducerDiagnostic(creators)
        producer_diagnostic.install()
    try:
        for tool in range(5, 0, -1):
            if tool == sys.monitoring.DEBUGGER_ID: continue
            try: sys.monitoring.use_tool_id(tool, 'original-stock-skill-setup')
            except ValueError: continue
            monitor.tool = tool
            break
        assert monitor.tool is not None and sys.monitoring.get_events(monitor.tool) == 0
        event = sys.monitoring.events.PY_RETURN
        assert sys.monitoring.register_callback(monitor.tool, event, returned) is None
        monitor.registered[event] = returned
        start_event = sys.monitoring.events.PY_START
        assert sys.monitoring.register_callback(monitor.tool, start_event, builder_started) is None
        monitor.registered[start_event] = builder_started
        monitor.codes = {builder_code: 'original_stock_builder', fresh_code: 'original_FreshContext_submit', discovery_code: 'original_skill_discovery'}
        if workspace_code is not None:
            monitor.codes[workspace_code] = 'original_workspace_connection_birth'
        for code in monitor.codes:
            assert sys.monitoring.get_local_events(monitor.tool, code) == 0
            sys.monitoring.set_local_events(monitor.tool, code, event | (start_event if code in {builder_code, discovery_code} else 0))
        monitor.active = monitor.installed = True
        async with app.run_test(size=(140, 42)) as pilot:
            await settle(lambda: getattr(app, '_initial_screen_pushed', False))
            from Tests.UI.background_signals import await_background_task
            initial = app._initial_screen_setup_task
            producer = inspect.getattr_static(type(app), '_run_no_splash_post_mount_setup')
            assert type(initial) is asyncio.Task and initial.get_loop() is loop
            coroutine = initial.get_coro()
            assert inspect.iscoroutine(coroutine) and coroutine.cr_code is producer.__code__
            if coroutine.cr_frame is not None:
                assert coroutine.cr_frame.f_globals is producer.__globals__
                assert coroutine.cr_frame.f_locals.get('self') is app
            await await_background_task(initial, what='original initial Console setup')
            assert initial.done() and not initial.cancelled() and initial.exception() is None
            facts['exact_original_initial_setup_Task_settled'] = True
            screen = runtime.view
            if case in {'denied', 'ready', 'custom', 'local_winner', 'app_winner'}:
                assert screen is not None
            if case == 'mount':
                await settle(hold_retired.is_set)
                # Permit the original Main baseline to reach its sole causal
                # RED only after unmodified App/source/observer retirement.
                if captured:
                    assert len(captured) == 1
                    await physical_result(captured[0][1])
                    assert any(request is captured[0][2] and issued is captured[0][1] for request, issued in submitted)
                    facts['actual_original_FreshContext_wrapper_correspondence'] = True
                await settle(lambda: app._local_skill_trust_service is not None)
            elif case in {'cancel', 'recancel', 'shutdown', 'local_winner', 'app_winner', 'runtime_dispose', 'runtime_replacement', 'app_closing', 'helper_binding', 'helper_body', 'metadata_body', 'send_enter', 'send_button'}:
                # held_control handles only the exact setup task; normal App
                # context exit still exercises its original creator teardown.
                # Initial-screen settlement precedes actual lazy builder admission.
                facts['outer_wait_started_at'] = time.monotonic()
                try:
                    if held_task is None:
                        # Initial-screen completion is not skill-builder completion.
                        # Observe only actual original discovery producers inside
                        # the unchanged outer 240-second App enclosure. The
                        # physical callback and admitted control still get 10s.
                        captured_discovery, discovery_row = await asyncio.shield(discovery_admitted)
                        worker, node, task, work = discovery_row
                        assert captured_discovery[0] is app and captured_discovery[1] is screen
                        assert captured_discovery[2] is vars(app).get('_workers')
                        assert node is screen and type(task) is asyncio.Task
                        assert task.get_loop() is loop and inspect.iscoroutine(work)
                        assert work.cr_code is discovery_code
                        assert vars(worker).get('_task') is task and vars(worker).get('_work') is work
                        await asyncio.wait(
                            {held_control_admitted, task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        facts['original_discovery_producer_prerequisite_observed'] = True
                finally:
                    facts['outer_wait_finished_at'] = time.monotonic()
                    facts['held_control_bound_at_outer_wait_end'] = held_task is not None
                assert held_task is not None and held_control_admitted.result() is held_task
                if not held_task.done():
                    remaining = 10 - (time.monotonic() - facts['held_control_admitted_at'])
                    assert remaining > 0, 'stock_skill_admitted_control_deadline'
                    await asyncio.wait_for(asyncio.shield(held_task), timeout=remaining)
                else:
                    held_task.result()
                assert facts['held_control_finished_at'] - facts['held_control_admitted_at'] <= 10
                if case in {'local_winner', 'app_winner'}:
                    context = await screen._skill._fetch_console_skill_context()
                    assert isinstance(context, dict) and 'available_skills' in context
                    assert local.trust_service is winner
            elif case == 'denied':
                assert await screen._skill._fetch_console_skill_context() == {}
                assert calls and all(value == 'skills.context.list.local' for value in calls)
                assert not entered.is_set() and app._local_skill_trust_service is None
                facts['actual_denied_policy_builder_calls_zero'] = True
            elif case == 'ready':
                await screen._skill._fetch_console_skill_context()
                assert app._local_skill_trust_service is winner
                assert not entered.is_set()
                facts['actual_ready_service_never_adopted_into_setup'] = True
            else:
                assert await screen._skill._fetch_console_skill_context() == {'available_skills': [], 'blocked_skills': []}
                assert calls and all(value == 'local' for value in calls)
                assert not entered.is_set() and app._local_skill_trust_service is None
                facts['actual_custom_facade_keeps_original_signature_and_result'] = True
        assert runtime._disposed and app.console_runtime is None
        assert replacement is None or replacement._disposed
        assert all(read._producer.done() and not read._producer.cancelled() and read.retired.done() and not read.retired.cancelled() and read not in runtime._preparation_reads for read in captured_reads)
        facts['declared_fixture_original_constructor_creators_retired'] = creators.retire()
        facts['constructor_owner_source_receipt'] = creators.receipt
        assert native_closed(creator) and runtime._disposed and app.console_runtime is None
        facts['actual_original_App_Runtime_and_creator_retired'] = True
    finally:
        release.set()
        try:
            if controller_thread is not None:
                await asyncio.to_thread(controller_thread.join, 10)
                assert not controller_thread.is_alive()
            try:
                if captured:
                    await settle(hold_retired.is_set)
                    await physical_result(captured[0][1])
            finally:
                try:
                    for read in captured_reads:
                        await asyncio.wait_for(asyncio.shield(read.retired), timeout=10)
                finally:
                    if held_task is not None:
                        await asyncio.wait_for(asyncio.shield(held_task), timeout=10)
        finally:
            try:
                if not creators.retired:
                    facts['declared_fixture_original_constructor_creators_retired'] = creators.retire()
                    facts['constructor_owner_source_receipt'] = creators.receipt
            finally:
                if mutation is not None:
                    target, name, value = mutation
                    setattr(target, name, value)
                    mutation = None
                assert wiring._skill_foreign_calls is metadata._skill_foreign_calls is foreign_calls
                del wiring._skill_foreign_calls, metadata._skill_foreign_calls
                if producer_diagnostic is not None:
                    facts['admitted_original_core_producers'] = producer_diagnostic.snapshot()
                    facts['core_producer_diagnostic_source_receipt'] = producer_diagnostic.close()
                receipt = monitor.close()
                facts['observer'] = receipt
                facts['invalid'] = invalid
                facts['policy_or_custom_calls'] = calls
                facts['final_counts'] = counts()
                if workspace_code is not None:
                    with storage._lock:
                        live_connections = tuple(workspace_database._maintenance_participant.connections)
                    facts['workspace_connection_origins'] = [dict(row, remains_registered=connection in live_connections, native_closed=native_closed(connection)) for connection, row in workspace_connections.items()]
                with storage._lock:
                    assert len(storage._pending_acquisitions) <= 16 and len(storage._raw_operations) <= 16
                    assert len(storage._operations) <= 16
                    facts['core_original_storage_actors'] = [
                        {'owner': a.participant.owner_id, 'thread': a.thread.name,
                         'task_type': type(a.task).__name__,
                         'task_code': a.task.get_coro().cr_code.co_qualname if type(a.task) is asyncio.Task and inspect.iscoroutine(a.task.get_coro()) else None}
                        for a in storage._operations]
                    facts['pending_original_storage_actors'] = [
                        {'thread': a.thread.name, 'owner': getattr(getattr(a.operation, 'participant', None), 'owner_id', None),
                         'operation_type': type(a.operation).__name__, 'task_type': type(a.task).__name__}
                        for a in storage._pending_acquisitions]
                    facts['raw_original_storage_actors'] = [
                        {'type': type(a).__name__, 'owner': getattr(a, 'owner_id', None),
                         'thread': getattr(getattr(a, 'thread', None), 'name', None), 'active': getattr(a, 'active', None)}
                        for a in storage._raw_operations]

                pause = storage._begin_local_pause()
                try: facts['original_one_second_global_drain'] = pause.drain(time.monotonic() + 1)
                finally: pause.resume()
                facts['network_refusals'] = len(network_guard.blocked_attempts())
                facts['profile_refusals'] = len(real_profile_guard.take_violations())
                (Path(os.environ['XDG_DATA_HOME']).parent / ('stock-skill-' + case + '.json')).write_text(json.dumps(facts, sort_keys=True), encoding='utf-8')
    assert receipt['complete'] and receipt['original_source_current'] and receipt['hooks_retired_before_inactive']
    assert receipt['global_events'] == 0 and not receipt['invalid'] and not invalid
    assert facts['original_one_second_global_drain']
    assert facts['network_refusals'] == facts['profile_refusals'] == 0
    if case == 'mount':
        assert facts['original_builder_off_ui_loop']
        assert facts['actual_original_FreshContext_wrapper_correspondence']
        assert facts['actual_loop_progress_before_original_builder_release'], 'Original stock skill trust setup blocked shared UI loop'
    if case in {'send_enter', 'send_button'}:
        assert facts['exact_original_callback_Future_and_singleflight_physically_retired']
        assert_cold_send_receipt(facts)
    print('retired and reopened')


with user_fixture_default_owner():
    selected = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    # These controls exercise skill lifetime, not unconfigured-provider
    # loopback discovery. Keep networking blocked and select ordinary fixture
    # provider inputs, as in the original compact-model controls.
    selected.write_text('[general]\nusers_name="skill-stock-control"\n[first_run]\nsetup_completed=true\n[_first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[api_settings.openai]\napi_key="skill-fixture-not-a-real-key"\n[chat_defaults]\nprovider="openai"\nmodel="gpt-4o-mini"\n', encoding='utf-8')
    selected.chmod(0o600)
    asyncio.run(asyncio.wait_for(exercise(), timeout=240))
"""


@pytest.mark.parametrize(
    "case",
    (
        "mount",
        "cancel",
        "recancel",
        "shutdown",
        "denied",
        "ready",
        "custom",
        "local_winner",
        "app_winner",
        "runtime_dispose",
        "runtime_replacement",
        "app_closing",
        "helper_binding",
        "helper_body",
        "metadata_body",
        "send_enter",
        "send_button",
    ),
)
def test_stock_console_skill_setup_has_original_owned_lifetime(tmp_path, case):
    # Keep the exact control script off Windows' bounded process command line.
    launch = (
        "from Tests.Performance.test_console_skill_trust_stock_owned import _SCRIPT; "
        "exec(compile(_SCRIPT, '<stock-skill-control>', 'exec'))"
    )
    _run(tmp_path, "console_skill_stock", case, script=launch, timeout=240)
