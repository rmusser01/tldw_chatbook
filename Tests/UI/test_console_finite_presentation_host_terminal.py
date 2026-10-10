"""Exact supported-host terminal boundary for original finite presentation work."""

import ast
import asyncio
import inspect
import json
from pathlib import Path
import sqlite3
import sys
import threading
from types import CodeType, MethodType

import pytest
from textual.worker import Worker
from textual.worker_manager import WorkerManager

from Tests.private_profile import private_profile_test
from Tests.Performance.console_storage_unit_observer import (
    OriginalStorageUnitObserver,
    _nested,
    _shape,
)
from Tests.UI.test_console_session_tab_close import (
    ProductionConsoleHarness,
    _pending_close_app,
    _mounted_console,
    _settle,
    _SIZE,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["readiness", "history", "context", "hooks"])
@private_profile_test
async def test_original_finite_presentation_callback_retires_before_host_return(
    request, tmp_path, route
):
    from tldw_chatbook.Backup_Recovery import (
        storage_admission as storage,
        raw_participants as raw,
    )
    from tldw_chatbook.Backup_Recovery.config_participants import (
        checked_config_identity,
    )
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.UI.Console_Modules.agent import ConsoleAgentController
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
        ConsoleContextReadSnapshot,
    )

    from tldw_chatbook.UI.Console_Modules.view_workers import (
        capture_console_view_workers,
        drain_console_view_workers,
    )

    # Reuse original setup/harness and source pinning. Never replace a reader,
    # Worker, shutdown method, operation, admission or native close.
    pin = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for function in (
        ConsoleAgentController._load_historical_presentation,
        ConsoleAgentBridge._derive_historical_snapshot,
        ConsoleReadinessConfigProjection._refresh,
        ConsoleReadinessConfigProjection.run,
        ConsoleContextReadSnapshot._refresh,
        ConsoleChatController.context_control_presentation_inputs,
        checked_config_identity,
        ProductionConsoleHarness._shutdown,
        capture_console_view_workers,
        drain_console_view_workers,
        Worker._run,
        Worker._start,
    ):
        pin._pin(function)
    from tldw_chatbook.DB.base_db import run_owned_db_call

    pin._pin(run_owned_db_call)
    pin.slots.append(
        (
            sys.modules[run_owned_db_call.__module__],
            "run_owned_db_call",
            run_owned_db_call,
        )
    )
    invoke_code = next(
        value
        for value in run_owned_db_call.__code__.co_consts
        if type(value) is CodeType and value.co_name == "invoke"
    )
    history_code = ConsoleAgentBridge._derive_historical_snapshot.__code__
    context_code = next(
        value
        for value in ConsoleChatController.context_control_presentation_inputs.__code__.co_consts
        if type(value) is CodeType and value.co_name == "read"
    )
    history_tree = ast.parse(
        Path(inspect.getsourcefile(ConsoleAgentBridge)).read_bytes()
    )
    history_body = next(
        node
        for cls in history_tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "ConsoleAgentBridge"
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_derive_historical_snapshot"
    )
    history_line = next(
        node.lineno
        for node in history_body.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "primary_records"
    )
    controller_tree = ast.parse(
        Path(inspect.getsourcefile(ConsoleChatController)).read_bytes()
    )
    context_body = next(
        node
        for cls in controller_tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "ConsoleChatController"
        for method in cls.body
        if isinstance(method, ast.AsyncFunctionDef)
        and method.name == "context_control_presentation_inputs"
        for node in method.body
        if isinstance(node, ast.FunctionDef) and node.name == "read"
    )
    context_line = next(
        node.lineno
        for node in ast.walk(context_body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "snapshots"
    )
    config = HookPermissions = ConsoleHooksController = _HookRefreshFlight = None
    ChatScreen = None
    hook_codes = {}
    hook_facts = {"frames": {}, "returns": {}, "joins": {}}
    if route == "hooks":
        from tldw_chatbook import config
        from tldw_chatbook.Agents.hook_permissions import HookPermissions
        from tldw_chatbook.UI.Console_Modules.hooks import (
            ConsoleHooksController,
            _HookRefreshFlight,
        )
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

        raw_reader = inspect.unwrap(config._read_raw_cli_config_unlocked)
        hook_codes.update(
            visit=HookPermissions.visit_snapshot.__code__,
            snapshot=HookPermissions.snapshot.__code__,
            raw=raw_reader.__code__,
            refresh=ConsoleHooksController.refresh.__code__,
        )
        for owner, name in (
            (HookPermissions, "visit_snapshot"),
            (HookPermissions, "snapshot"),
            (HookPermissions, "save_configuration"),
            (ConsoleHooksController, "refresh"),
            (ConsoleHooksController, "_owned_result"),
            (ConsoleHooksController, "_refresh_current"),
            (ConsoleHooksController, "_same_reader"),
            (ChatScreen, "_refresh_console_hooks"),
        ):
            descriptor = inspect.getattr_static(owner, name)
            function = (
                descriptor.__func__ if type(descriptor) is staticmethod else descriptor
            )
            pin._pin(function)
            pin.slots.append((owner, name, descriptor))
        pin._pin(config._read_raw_cli_config_unlocked)
        pin._pin(raw_reader)
        pin.slots.extend(
            (
                (
                    config,
                    "_read_raw_cli_config_unlocked",
                    config._read_raw_cli_config_unlocked,
                ),
                (config._read_raw_cli_config_unlocked, "__wrapped__", raw_reader),
            )
        )

    outer_codes = {
        ConsoleAgentController._load_historical_presentation.__code__,
        ConsoleReadinessConfigProjection._refresh.__code__,
        ConsoleContextReadSnapshot._refresh.__code__,
        ConsoleChatController.context_control_presentation_inputs.__code__,
    }
    entered, release = threading.Event(), threading.Event()
    entered_async = asyncio.Event()
    shutdown_entered = asyncio.Event()
    owner_loop = asyncio.get_running_loop()
    finish_body, mounted, host_returned = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    exact, inner, held, failures = {}, [], {}, []
    before_release = {}
    flags = {"readiness_returns": 0, "armed": False, "attributed": False}
    tool = None

    def observed_workers():
        if route == "hooks":
            flight = exact.get("hook_flight")
            return (
                tuple(
                    row
                    for joined, row in hook_facts["joins"].values()
                    if joined is flight
                )
                if flight is not None
                else ()
            )
        return exact.get("workers", ())

    def observe_hook_join(frame, issuer):
        if frame.f_locals.get("self") is not exact.get("hooks"):
            return
        flight = frame.f_locals.get("flight")
        assert type(flight) is _HookRefreshFlight
        assert flight.owner is exact["hook_owner"]
        assert flight.reader.__self__ is exact["hook_owner"]
        assert flight.reader.__func__ is HookPermissions.visit_snapshot
        assert type(issuer) is asyncio.Task and issuer.get_loop() is owner_loop
        manager = vars(exact["host"])["_workers"]
        assert type(manager) is WorkerManager
        assert vars(manager)["_app"] is exact["host"]
        matches = tuple(
            worker
            for worker in manager
            if type(worker) is Worker and vars(worker).get("_task") is issuer
        )
        assert len(matches) == 1  # Task identity, never group cardinality.
        (worker,) = matches
        values = vars(worker)
        work = values["_work"]
        assert values["_node"] is exact["console"]
        assert values["group"] == "console-hook-refresh"
        assert type(work) is MethodType
        assert work.__self__ is exact["console"]
        assert work.__func__ is ChatScreen._refresh_console_hooks
        coroutine = issuer.get_coro()
        assert coroutine.cr_code is Worker._run.__code__
        assert coroutine.cr_frame.f_locals["self"] is worker
        producer = flight.producer
        assert type(producer) is asyncio.Task
        assert producer.get_loop() is owner_loop and not producer.done()
        row = (worker, values["_node"], issuer, work)
        prior = hook_facts["joins"].get(issuer)
        assert prior is None or (
            prior[0] is flight and all(old is new for old, new in zip(prior[1], row))
        )
        hook_facts["joins"][issuer] = (flight, row)

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError as error:
            return "closed" in str(error).lower()
        return False

    def hold(code, line_or_offset, value=None):
        if code is capture_console_view_workers.__code__:
            if (
                flags["attributed"]
                and value[0] is exact.get("host")
                and value[1] is None
            ):
                # Character widgets also drain their own workers on unmount.
                # Only the original whole-host capture owns this exit oracle.
                exact["shutdown_capture"] = value
            return
        if route == "hooks":
            frame = sys._getframe(1)
            try:
                for name, selected_code in hook_codes.items():
                    if (
                        flags["armed"]
                        and code is selected_code
                        and hook_facts["frames"].get(name) == id(frame)
                        and held.get("thread") is threading.current_thread()
                    ):
                        hook_facts["returns"][name] = True
            finally:
                del frame
            return
        if not flags["armed"] or held:
            return
        frame = sys._getframe(1)
        try:
            if route == "readiness":
                if code is not checked_config_identity.__code__:
                    return
                reader_code = exact["projection"].read_current.__code__
                ancestor = frame.f_back
                while ancestor is not None and ancestor.f_code is not reader_code:
                    ancestor = ancestor.f_back
                if (
                    ancestor is None
                    or ancestor.f_locals.get("projection") is not exact["projection"]
                ):
                    return
                flags["readiness_returns"] += 1
                if flags["readiness_returns"] != 2:
                    return
                state = raw._states[frame.f_locals["active"]]
                with storage._lock:
                    assert state.active and len(state.leases) >= 2
                    assert all(lease in storage._live_leases for lease in state.leases)
                held.update(
                    raw_state=state,
                    leases=tuple(state.leases),
                    thread=threading.current_thread(),
                )
            else:
                selected_code, selected_line = (
                    (history_code, history_line)
                    if route == "history"
                    else (context_code, context_line)
                )
                if code is not selected_code or line_or_offset != selected_line:
                    return
                owner = exact["bridge"] if route == "history" else exact["controller"]
                if frame.f_locals.get("self") is not owner:
                    return
                database = exact["runs"] if route == "history" else exact["notes"]
                local = (
                    database._thread_local if route == "history" else database._local
                )
                connection = getattr(local, "conn", None)
                participant = database._maintenance_participant
                operation = getattr(storage._operation_local, "operation", None)
                with storage._lock:
                    lease = participant.connections.get(connection)
                    assert connection is not None and not closed(connection)
                    assert (
                        lease in storage._live_leases
                        and operation in storage._operations
                    )
                    assert (
                        operation.participant is participant
                        and lease.resource_thread is threading.current_thread()
                    )
                if route == "context":
                    # Join the exact native callback function to its actual
                    # source-qualified coroutine issuer, rather than its group.
                    parent = frame.f_back
                    assert parent.f_code is invoke_code
                    assert parent.f_locals.get("database") is database
                    reader = parent.f_locals.get("operation")
                    assert reader.__code__ is context_code
                    held["context_reader"] = reader
                held.update(
                    connection=connection,
                    participant=participant,
                    operation=operation,
                    lease=lease,
                    thread=threading.current_thread(),
                )
            entered.set()
            owner_loop.call_soon_threadsafe(entered_async.set)
            assert release.wait(
                10
            ), "original held callback release exceeded original ten-second boundary"
        except BaseException as error:
            failures.append(type(error).__name__)
            entered.set()
            owner_loop.call_soon_threadsafe(entered_async.set)
        finally:
            del frame

    def hook_started(code, offset):
        if (
            route != "hooks"
            or not flags["armed"]
            or held
            or code is not hook_codes["raw"]
        ):
            return
        frame = sys._getframe(1)
        try:
            ancestors = {}
            ancestor = frame.f_back
            while ancestor is not None:
                for name in ("visit", "snapshot"):
                    if (
                        ancestor.f_code is hook_codes[name]
                        and ancestor.f_locals.get("self") is exact["hook_owner"]
                    ):
                        ancestors[name] = id(ancestor)
                ancestor = ancestor.f_back
            if set(ancestors) != {"visit", "snapshot"}:
                return
            thread = threading.current_thread()
            with storage._lock:
                operations = tuple(
                    operation
                    for operation, state in raw._states.items()
                    if state.source is config and state.thread is thread
                )
                states = tuple(raw._states[operation] for operation in operations)
                leases = tuple(lease for state in states for lease in state.leases)
                assert operations and leases and all(state.active for state in states)
                assert all(
                    state.selected == exact["hook_config_path"] for state in states
                )
                assert all(lease in storage._live_leases for lease in leases)
            hook_facts["frames"].update(ancestors, raw=id(frame))
            held.update(
                operations=operations, states=states, leases=leases, thread=thread
            )
            entered.set()
            owner_loop.call_soon_threadsafe(entered_async.set)
            assert release.wait(
                10
            ), "original held hook callback release exceeded original ten-second boundary"
        except BaseException as error:
            failures.append(type(error).__name__)
            entered.set()
            owner_loop.call_soon_threadsafe(entered_async.set)
        finally:
            del frame

    def yielded(code, offset, value):
        if code is drain_console_view_workers.__code__:
            frame = sys._getframe(1)
            try:
                captured = frame.f_locals.get("captured")
                if flags["attributed"] and captured is exact.get("shutdown_capture"):
                    assert captured[0] is exact["host"]
                    assert captured[2] is exact["manager"]
                    assert captured[3] is owner_loop
                    assert captured[4] is threading.current_thread()
                    required = observed_workers()
                    exact["shutdown_capture_contains_worker"] = bool(required) and all(
                        any(
                            all(
                                actual is expected
                                for actual, expected in zip(row, candidate)
                            )
                            for candidate in captured[5]
                        )
                        for row in required
                    )
                    shutdown_entered.set()
            finally:
                del frame
            return
        if (
            route == "hooks"
            and flags["armed"]
            and not release.is_set()
            and code is hook_codes["refresh"]
        ):
            frame = sys._getframe(1)
            try:
                observe_hook_join(frame, asyncio.current_task())
            except BaseException as error:
                # A failed observation must fail qualification without breaking
                # the original worker or its physical cleanup.
                failures.append("hook_join_" + type(error).__name__)
            finally:
                del frame
            return
        if not flags["armed"] or code not in outer_codes:
            return
        frame = sys._getframe(1)
        try:
            values = frame.f_locals
            owner = values.get("self")
            if owner not in (
                exact.get("agent"),
                exact.get("projection"),
                exact.get("context"),
                exact.get("controller"),
            ):
                return
            if (
                route == "context"
                and code
                is ConsoleChatController.context_control_presentation_inputs.__code__
            ):
                reader = values.get("read")
                issuer = asyncio.current_task()
                assert reader.__code__ is context_code
                assert type(issuer) is asyncio.Task
                assert issuer.get_loop() is owner_loop
                exact.setdefault("context_issuers", {})[reader] = issuer
            task = values.get("worker") or values.get("task")
            if type(task) is asyncio.Task and task not in inner:
                inner.append(task)
        finally:
            del frame

    async def host_owner():
        async with _pending_close_app(request, "chat_create") as app:
            host = ProductionConsoleHarness(app)
            async with host.run_test(size=_SIZE) as pilot:
                console = await _mounted_console(
                    host, pilot, "#console-native-composer"
                )
                controller = console._ensure_console_chat_controller()
                runtime = app.console_runtime
                bridge = console._ensure_console_agent_bridge()
                app._pending_close_owned_resources.adopt_runtime_runs()
                store = controller.store
                projection = ConsoleReadinessConfigProjection.for_screen(console)
                from tldw_chatbook.UI.Screens.chat_screen import (
                    CONSOLE_SETTINGS_ESTIMATE_TTL_SECONDS,
                )

                context = ConsoleContextReadSnapshot.for_screen(
                    console, max_age=CONSOLE_SETTINGS_ESTIMATE_TTL_SECONDS
                )
                # Bootstrap remains unarmed and uses the original full UI pass.
                # The held test starts only after these real producers settle.
                await console._sync_native_console_chat_ui()
                bootstrap_settled = await _settle(
                    pilot,
                    lambda: not projection.pending
                    and projection._settled.is_set()
                    and context.pending_key is None
                    and not context.lock.locked()
                    and (
                        console._agent._console_historical_read is None
                        or not console._agent._console_historical_read["pending"]
                    ),
                )
                if not bootstrap_settled:
                    from types import CoroutineType, GeneratorType

                    thread_stacks = []
                    current_frames = sys._current_frames()
                    current_thread = threading.get_ident()
                    thread_ids = [current_thread] + sorted(
                        thread_id
                        for thread_id in current_frames
                        if thread_id != current_thread
                    )[:31]
                    for thread_id in thread_ids:
                        frame = current_frames.get(thread_id)
                        stack = []
                        for _depth in range(16):
                            if frame is None:
                                break
                            stack.append(
                                {
                                    "co_name": frame.f_code.co_name,
                                    "co_filename": frame.f_code.co_filename,
                                    "lineno": frame.f_lineno,
                                }
                            )
                            frame = frame.f_back
                        thread_stacks.append({"thread_id": thread_id, "frames": stack})
                    del current_frames
                    frame = None
                    history_state = console._agent._console_historical_read
                    manager = vars(host).get("_workers")
                    manager_current = (
                        type(manager) is WorkerManager
                        and vars(manager).get("_app") is host
                    )
                    worker_facts = []
                    if manager_current:
                        for worker in tuple(manager):
                            if type(worker) is not Worker:
                                continue
                            values = vars(worker)
                            group = values.get("group")
                            if (
                                values.get("_node") is not console
                                or type(group) is not str  # noqa: E721 - stock group scalars only.
                                or group
                                not in {
                                    "console-sync",
                                    "console-readiness-config",
                                    "console-agent-history",
                                    "console-context-presentation",
                                    "console-hook-refresh",
                                }
                            ):
                                continue
                            task = values.get("_task")
                            stock_task = type(task) is asyncio.Task
                            await_chain = []
                            if stock_task and task.get_loop() is owner_loop:
                                awaited = task.get_coro()
                                for _depth in range(8):
                                    if awaited is None:
                                        break
                                    if type(awaited) is CoroutineType:
                                        frame = awaited.cr_frame
                                        next_awaited = awaited.cr_await
                                    elif type(awaited) is GeneratorType:
                                        frame = awaited.gi_frame
                                        next_awaited = awaited.gi_yieldfrom
                                    else:
                                        await_chain.append(
                                            {"await_id": id(awaited), "opaque": True}
                                        )
                                        break
                                    await_chain.append(
                                        {
                                            "await_id": id(awaited),
                                            "co_name": None
                                            if frame is None
                                            else frame.f_code.co_name,
                                            "co_filename": None
                                            if frame is None
                                            else frame.f_code.co_filename,
                                            "lineno": None
                                            if frame is None
                                            else frame.f_lineno,
                                        }
                                    )
                                    awaited = next_awaited
                                awaited = next_awaited = frame = None
                            worker_facts.append(
                                {
                                    "group": group,
                                    "worker_id": id(worker),
                                    "node_id": id(console),
                                    "work_id": id(values.get("_work")),
                                    "task_id": None if task is None else id(task),
                                    "stock_task": stock_task,
                                    "task_done": task.done() if stock_task else None,
                                    "task_cancelling": task.cancelling()
                                    if stock_task
                                    else None,
                                    "task_loop_current": task.get_loop() is owner_loop
                                    if stock_task
                                    else None,
                                    "await_chain": await_chain,
                                }
                            )
                            if len(worker_facts) == 16:
                                break
                    print(
                        json.dumps(
                            {
                                "route": route,
                                "stage": "unarmed_joint_bootstrap_settle_failure",
                                "armed": flags["armed"],
                                "readiness_pending": projection.pending,
                                "readiness_settled": projection._settled.is_set(),
                                "context_pending_key_present": context.pending_key
                                is not None,
                                "context_lock_locked": context.lock.locked(),
                                "history_state_present": history_state is not None,
                                "history_pending": None
                                if history_state is None
                                else history_state["pending"],
                                "worker_manager_current": manager_current,
                                "actual_workers": worker_facts,
                                "thread_stacks": thread_stacks,
                            },
                            sort_keys=True,
                        )
                    )
                assert bootstrap_settled
                # A genuinely new selected session changes the original keys;
                # no memo, TTL or pending field is assigned to manufacture cold.
                store.create_session(title="Finite host lifetime", activate=True)
                session = next(
                    item
                    for item in store.sessions()
                    if item.id == store.active_session_id
                )
                if session.ephemeral:
                    store.promote_ephemeral_session(session.id)
                else:
                    store.persist_session_if_needed(session.id)
                # Ordinary durable fixture action gives the context reader a
                # nonempty persisted lineage; no cache or projection is forged.
                from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

                anchor = store.append_message(
                    session.id,
                    role=ConsoleMessageRole.USER,
                    content="finite host context anchor",
                    persist=True,
                )
                assert anchor.persisted_message_id is not None
                assert anchor.id in store.active_path_message_ids(session.id)
                if route == "hooks":
                    # Finish the ordinary indicator read unarmed. Saving the real
                    # selected private section invalidates its prior visit reuse.
                    await console._refresh_console_hooks()
                    hooks = console._hooks
                    hook_owner = hooks._permissions()
                    assert type(hooks) is ConsoleHooksController
                    assert type(hook_owner) is HookPermissions
                    snapshot = await asyncio.to_thread(hook_owner.snapshot)
                    replacement = dict(snapshot.config.section or {})
                    replacement["enabled"] = (
                        replacement.get("enabled", True) is not True
                    )
                    result, saved = await asyncio.to_thread(
                        hook_owner.save_configuration,
                        snapshot,
                        replacement,
                    )
                    assert result.file_replaced
                    schedule_hooks = hooks._on_send_settled
                    assert callable(schedule_hooks)
                    for callback in (schedule_hooks, hooks._permissions):
                        cells = dict(
                            zip(
                                callback.__code__.co_freevars,
                                callback.__closure__ or (),
                            )
                        )
                        assert cells["screen"].cell_contents is console
                    pin._pin(schedule_hooks)
                    pin._pin(hooks._permissions)
                    pin.slots.extend(
                        (
                            (console, "_hooks", hooks),
                            (hooks, "_on_send_settled", schedule_hooks),
                            (hooks, "_permissions", hooks._permissions),
                        )
                    )
                    exact.update(
                        hooks=hooks,
                        hook_owner=hook_owner,
                        hook_config_path=saved.config.config_path,
                        hook_schedule=schedule_hooks,
                    )
                exact.update(
                    host=host,
                    console=console,
                    controller=controller,
                    runtime=runtime,
                    bridge=bridge,
                    runs=bridge._db,
                    notes=app.chachanotes_db,
                    agent=console._agent,
                    projection=projection,
                    context=context,
                    pilot=pilot,
                )
                if route == "context":
                    # Refresh the genuine new selected session's readiness key
                    # before requesting its separate context presentation work.
                    assert await projection.warm()
                # Normal original presentation entry points; no pending/key/TTL assignment.
                flags["armed"] = True
                if route == "readiness":
                    projection.run(lambda: None)
                elif route == "history":
                    console._agent._presentation_historical_snapshot(
                        bridge, session.persisted_conversation_id
                    )
                elif route == "context":
                    context.inputs(controller, session.id)
                else:
                    # The existing wiring callback schedules only the normal
                    # hook-refresh Worker after refresh; no Worker is fabricated.
                    exact["hook_schedule"]()
                mounted.set()
                await asyncio.wait_for(entered_async.wait(), 10)
                await finish_body.wait()
            # This oracle is before Runtime/creator fixture cleanup.
            host_returned.set()
            observed_tasks = tuple(row[2] for row in observed_workers())
            exact["post_host_before_creator_cleanup"] = {
                "held_callback_live": bool(held) and not release.is_set(),
                "exact_task_present": bool(observed_tasks),
                "observed_outer_task_count": len(observed_tasks),
                "outer_task_terminal": (
                    None
                    if not observed_tasks
                    else all(task.done() for task in observed_tasks)
                ),
                "observed_inner_task_count": len(inner),
                "inner_tasks_terminal": (
                    None if not inner else all(task.done() for task in inner)
                ),
            }
            print(
                json.dumps(
                    {
                        "route": route,
                        "stage": "actual_host_return_before_creator_cleanup",
                        "before_release": before_release,
                        "post_host_before_creator_cleanup": exact[
                            "post_host_before_creator_cleanup"
                        ],
                    },
                    sort_keys=True,
                )
            )

    # This known diagnostic module's local callbacks are normal pytest-rewritten
    # code. Qualify their actual original rewrite before local monitoring starts.
    from _pytest.assertion import rewrite

    module = sys.modules[__name__]
    loader = module.__loader__
    assert (
        type(loader) is rewrite.AssertionRewritingHook
        and loader.config is request.config
    )
    for owner, name, function in (
        (rewrite, "_rewrite_test", rewrite._rewrite_test),
        (rewrite, "rewrite_asserts", rewrite.rewrite_asserts),
        (
            rewrite.AssertionRewritingHook,
            "exec_module",
            rewrite.AssertionRewritingHook.exec_module,
        ),
    ):
        pin._pin(function)
        pin.slots.append((owner, name, function))
    path = Path(__file__).resolve()
    data = path.read_bytes()
    _stat, compiled = rewrite._rewrite_test(path, request.config)
    for function in (
        observed_workers,
        observe_hook_join,
        closed,
        hold,
        hook_started,
        yielded,
        host_owner,
    ):
        expected = _nested(compiled, function.__code__)
        assert expected is not None and _shape(expected) == _shape(function.__code__)
        closure = function.__closure__
        pin.pins.append(
            (
                function,
                function.__code__,
                function.__globals__,
                function.__defaults__,
                function.__kwdefaults__,
                tuple((function.__kwdefaults__ or {}).items()),
                closure,
                tuple((cell, cell.cell_contents) for cell in closure or ()),
            )
        )
    import hashlib

    pin.modules[__name__] = (
        module,
        path,
        module.__spec__,
        module.__spec__.origin,
        module.__loader__,
        module.__spec__.loader,
        hashlib.sha256(data).hexdigest(),
    )

    pin.codes[capture_console_view_workers.__code__] = "original-host-capture"
    pin.codes[drain_console_view_workers.__code__] = "original-host-drain"
    pin.codes[history_code] = "history-native"
    pin.codes[context_code] = "context-native"
    pin.codes[checked_config_identity.__code__] = "readiness-native"
    pin.codes.update({code: "outer-callback" for code in outer_codes})
    if route == "hooks":
        pin.codes.update({code: "hook-native" for code in hook_codes.values()})
    try:
        for candidate in range(5, 0, -1):
            if candidate == sys.monitoring.DEBUGGER_ID:
                continue
            try:
                sys.monitoring.use_tool_id(
                    candidate, "finite-presentation-host-boundary"
                )
            except ValueError:
                continue
            tool = pin.tool = candidate
            break
        assert tool is not None
        for event, callback in (
            (sys.monitoring.events.LINE, hold),
            (sys.monitoring.events.PY_RETURN, hold),
            (sys.monitoring.events.PY_YIELD, yielded),
        ):
            assert sys.monitoring.register_callback(tool, event, callback) is None
            pin.registered[event] = callback
        sys.monitoring.set_local_events(
            tool, capture_console_view_workers.__code__, sys.monitoring.events.PY_RETURN
        )
        sys.monitoring.set_local_events(
            tool, drain_console_view_workers.__code__, sys.monitoring.events.PY_YIELD
        )
        sys.monitoring.set_local_events(tool, history_code, sys.monitoring.events.LINE)
        sys.monitoring.set_local_events(tool, context_code, sys.monitoring.events.LINE)
        sys.monitoring.set_local_events(
            tool, checked_config_identity.__code__, sys.monitoring.events.PY_RETURN
        )
        for code in outer_codes:
            sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_YIELD)
        if route == "hooks":
            sys.monitoring.set_local_events(
                tool, hook_codes["refresh"], sys.monitoring.events.PY_YIELD
            )
            event = sys.monitoring.events.PY_START
            assert sys.monitoring.register_callback(tool, event, hook_started) is None
            pin.registered[event] = hook_started
            sys.monitoring.set_local_events(
                tool,
                hook_codes["raw"],
                sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN,
            )
            for name in ("visit", "snapshot"):
                sys.monitoring.set_local_events(
                    tool, hook_codes[name], sys.monitoring.events.PY_RETURN
                )
        pin.active = pin.installed = True
        owner_task = asyncio.Task(host_owner())
        mounted_task = asyncio.Task(mounted.wait())
        try:
            # Bootstrap already has the original mounted/settle guards.
            # Observe its completion or failure under the runner transport.
            await asyncio.wait(
                {mounted_task, owner_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if owner_task.done():
                await owner_task
            assert mounted.is_set()
            await asyncio.wait_for(entered_async.wait(), 10)
            assert held and not failures
            group = {
                "readiness": "console-readiness-config",
                "history": "console-agent-history",
                "context": "console-context-presentation",
                "hooks": "console-hook-refresh",
            }[route]
            manager = vars(exact["host"])["_workers"]
            assert (
                type(manager) is WorkerManager
                and vars(manager)["_app"] is exact["host"]
            )
            context_issuer = None
            if route == "context":
                context_issuer = exact.get("context_issuers", {}).get(
                    held["context_reader"]
                )
                assert type(context_issuer) is asyncio.Task
                assert context_issuer.get_loop() is owner_loop
            if route == "hooks":
                hooks = exact["hooks"]
                flight = hooks._refresh_flight
                assert type(flight) is _HookRefreshFlight
                assert flight.owner is exact["hook_owner"]
                assert flight.accessor is hooks._permissions
                assert flight.publisher is hooks._on_state
                assert flight.reader.__self__ is exact["hook_owner"]
                assert flight.reader.__func__ is HookPermissions.visit_snapshot
                producer = flight.producer
                assert type(producer) is asyncio.Task
                assert producer.get_loop() is asyncio.get_running_loop()
                assert not producer.done()
                inner.append(producer)
                exact.update(hook_flight=flight, hook_producer=producer)
                # Every original refresh Task observed joining this exact flight
                # is required; a cancelled exclusive predecessor still owns it.
                for candidate in tuple(manager):
                    if type(candidate) is not Worker:
                        continue
                    values = vars(candidate)
                    if (
                        values.get("group") != group
                        or values.get("_node") is not exact["console"]
                    ):
                        continue
                    issuer = values.get("_task")
                    assert (
                        type(issuer) is asyncio.Task and issuer.get_loop() is owner_loop
                    )
                    awaited = issuer.get_coro()
                    while awaited is not None:
                        if getattr(awaited, "cr_code", None) is hook_codes["refresh"]:
                            frame = awaited.cr_frame
                            if (
                                frame is not None
                                and frame.f_locals.get("flight") is flight
                            ):
                                observe_hook_join(frame, issuer)
                            break
                        awaited = getattr(awaited, "cr_await", None) or getattr(
                            awaited, "gi_yieldfrom", None
                        )
                rows = observed_workers()
                assert rows, "No original Worker joined the held hook flight"
                assert not failures
            else:
                workers = tuple(
                    worker
                    for worker in manager
                    if type(worker) is Worker
                    and (
                        vars(worker).get("_task") is context_issuer
                        if route == "context"
                        else vars(worker).get("group") == group
                    )
                    and vars(worker).get("_node") is exact["console"]
                )
                assert len(workers) == 1
                (worker,) = workers
                values = vars(worker)
                task, work = values["_task"], values["_work"]
                if route == "context":
                    group = values["group"]
                    assert type(group) is str  # noqa: E721 - exact original Worker group.
                    assert task is context_issuer
                rows = ((worker, values["_node"], task, work),)
            assert all(
                type(task) is asyncio.Task
                and task.get_loop() is owner_loop
                and not task.done()
                for _worker, _node, task, _work in rows
            )
            assert all(worker in manager for worker, _node, _task, _work in rows)
            exact.update(workers=rows, manager=manager)
            flags["attributed"] = True
            before_release.update(
                exact_node_id=id(exact["console"]),
                exact_manager_id=id(manager),
                group=group,
                exact_workers=[
                    {
                        "worker_id": id(worker),
                        "node_id": id(node),
                        "work_id": id(work),
                        "task_id": id(task),
                        "task_loop_current": task.get_loop() is owner_loop,
                        "worker_task_same": vars(worker).get("_task") is task,
                        "worker_work_same": vars(worker).get("_work") is work,
                        "worker_node_same": vars(worker).get("_node") is node,
                        "task_done": task.done(),
                        "task_cancelling": task.cancelling(),
                    }
                    for worker, node, task, work in rows
                ],
                inner_tasks=[
                    {
                        "id": id(item),
                        "done": item.done(),
                        "loop_current": item.get_loop() is asyncio.get_running_loop(),
                    }
                    for item in inner
                ],
                context_has_no_retained_inner_task=(route == "context" and not inner),
                exact_context_reader_id=(
                    id(held["context_reader"]) if route == "context" else None
                ),
                exact_context_issuer_task_id=(
                    id(context_issuer) if route == "context" else None
                ),
                context_callback_issuer_matches_outer_task=(
                    context_issuer is task if route == "context" else None
                ),
            )
            if route == "readiness":
                with storage._lock:
                    before_release.update(
                        raw_scope_active=held["raw_state"].active,
                        leases_live=all(
                            lease in storage._live_leases for lease in held["leases"]
                        ),
                    )
            elif route == "hooks":
                with storage._lock:
                    before_release.update(
                        exact_hook_owner_id=id(exact["hook_owner"]),
                        exact_hook_flight_id=id(exact["hook_flight"]),
                        exact_hook_reader_function_id=id(
                            exact["hook_flight"].reader.__func__
                        ),
                        exact_hook_producer_id=id(exact["hook_producer"]),
                        hook_producer_pending=not exact["hook_producer"].done(),
                        exact_operation_ids=[
                            id(operation) for operation in held["operations"]
                        ],
                        exact_lease_ids=[id(lease) for lease in held["leases"]],
                        raw_scopes_active=all(state.active for state in held["states"]),
                        raw_source_same=all(
                            state.source is config for state in held["states"]
                        ),
                        raw_thread_same=all(
                            state.thread is held["thread"] for state in held["states"]
                        ),
                        operation_live=all(
                            operation in raw._states for operation in held["operations"]
                        ),
                        leases_live=all(
                            lease in storage._live_leases for lease in held["leases"]
                        ),
                        original_hook_callback_returns=dict(hook_facts["returns"]),
                    )
            else:
                with storage._lock:
                    before_release.update(
                        exact_native_id=id(held["connection"]),
                        native_open=not closed(held["connection"]),
                        exact_operation_id=id(held["operation"]),
                        operation_live=held["operation"] in storage._operations,
                        exact_lease_id=id(held["lease"]),
                        lease_live=held["lease"] in storage._live_leases,
                        registration_current=held["connection"]
                        in held["participant"].connections,
                    )
            # Exit is requested by this separate controller Task; run_test enter
            # and exit remain in the single original host_owner Task.
            finish_body.set()
            await asyncio.wait_for(shutdown_entered.wait(), 10)
            before_release["original_shutdown_drain_observed"] = True
            before_release["original_capture_contains_exact_worker"] = exact[
                "shutdown_capture_contains_worker"
            ]
            done, _ = await asyncio.wait({owner_task}, timeout=0.35)
            ended_while_held = host_returned.is_set()
            observed_tasks = tuple(row[2] for row in observed_workers())
            assert observed_tasks
            logical_terminal_while_held = any(task.done() for task in observed_tasks)
            before_release.update(
                host_returned=ended_while_held,
                outer_done_after_exit_request=logical_terminal_while_held,
                observed_outer_tasks_after_exit_request=[
                    {
                        "id": id(task),
                        "done": task.done(),
                        "cancelling": task.cancelling(),
                    }
                    for task in observed_tasks
                ],
            )
            # Preserve the causal boundary even if later creator cleanup fails.
            print(
                json.dumps(
                    {
                        "route": route,
                        "stage": "exit_probe_before_callback_release",
                        "before_release": before_release,
                        "post_host_before_creator_cleanup": exact.get(
                            "post_host_before_creator_cleanup"
                        ),
                        "host_returned_while_exact_callback_held": ended_while_held,
                        "outer_task_terminal_while_exact_callback_held": logical_terminal_while_held,
                    },
                    sort_keys=True,
                )
            )
        finally:
            body_error = sys.exception()
            if body_error is not None:
                # Qualification failures must not throw from the monitoring
                # callbacks during the original host's cleanup.
                flags["attributed"] = False
            finish_body.set()
            release.set()
            mounted_task.cancel()
            await asyncio.gather(mounted_task, return_exceptions=True)
            try:
                await owner_task
            except BaseException as cleanup_error:
                if body_error is None:
                    raise
                if cleanup_error is not body_error:
                    body_error.add_note("finite_presentation_host_owner_cleanup_failed")
            finally:
                if body_error is not None:
                    print(
                        json.dumps(
                            {
                                "route": route,
                                "setup_failure": type(body_error).__name__,
                                "before_release": before_release,
                                "bootstrap_mounted": mounted.is_set(),
                                "exact_callback_observed": bool(held),
                                "exact_task_present": bool(observed_workers()),
                                "actual_host_return_observed": host_returned.is_set(),
                                "post_host_before_creator_cleanup": exact.get(
                                    "post_host_before_creator_cleanup"
                                ),
                            },
                            sort_keys=True,
                        )
                    )
        assert not failures
        required = observed_workers()
        assert required and all(row[2].done() for row in required)
        assert all(item.done() for item in inner)
        captured = exact["shutdown_capture"]
        # Include original joiners observed after initial attribution as well;
        # a late worker cannot silently disappear from the membership oracle.
        assert all(
            any(
                all(actual is expected for actual, expected in zip(row, candidate))
                for candidate in captured[5]
            )
            for row in required
        ), "original host omitted a joined original worker"
        if route == "readiness":
            assert not held["raw_state"].active
            with storage._lock:
                assert all(
                    lease not in storage._live_leases for lease in held["leases"]
                )
        elif route == "hooks":
            assert hook_facts["returns"] == {
                "raw": True,
                "snapshot": True,
                "visit": True,
            }
            assert exact["hook_producer"].done()
            with storage._lock:
                assert all(
                    operation not in raw._states for operation in held["operations"]
                )
                assert all(not state.active for state in held["states"])
                assert all(
                    lease not in storage._live_leases for lease in held["leases"]
                )
        else:
            assert closed(held["connection"])
            with storage._lock:
                assert held["operation"] not in storage._operations
                assert held["lease"] not in storage._live_leases
                assert held["connection"] not in held["participant"].connections
        print(
            json.dumps(
                {
                    "route": route,
                    "before_release": before_release,
                    "post_host_before_creator_cleanup": exact.get(
                        "post_host_before_creator_cleanup"
                    ),
                    "host_returned_while_exact_callback_held": ended_while_held,
                    "outer_task_terminal_while_exact_callback_held": logical_terminal_while_held,
                    "final_exact_native_and_task_retirement": True,
                    "original_hook_callback_returns": (
                        hook_facts["returns"] if route == "hooks" else None
                    ),
                },
                sort_keys=True,
            )
        )
        assert shutdown_entered.is_set()
        assert exact[
            "shutdown_capture_contains_worker"
        ], "original host omitted exact worker"
        assert (
            not logical_terminal_while_held
        ), "original awaiting task detached from its still-live native callback"
        assert (
            not ended_while_held
        ), "original host returned while exact finite native callback was live"
    finally:
        release.set()
        receipt = pin.close()
        (tmp_path / f"console-host-{route}.json").write_text(
            json.dumps(
                {
                    "route": route,
                    "armed": flags["armed"],
                    "before_release": before_release,
                    "post_host_before_creator_cleanup": exact.get(
                        "post_host_before_creator_cleanup"
                    ),
                    "observer": receipt,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        assert (
            receipt["original_source_current"]
            and receipt["hooks_retired_before_inactive"]
        )
