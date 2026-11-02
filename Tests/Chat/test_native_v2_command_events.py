"""Real command events retain native physical and original host custody."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.Agents.test_hooks_v2_execution import command
from Tests.Chat.test_console_hook_preparation_demand import _save_hooks
from Tests.Plugins.conftest import native_console as _native_console
from Tests.Plugins.conftest import native_package as _native_package
from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

hook_file = _hook_file
native_console = _native_console
native_package = _native_package
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
    pytest.mark.skipif(
        not native_workflow_qualified(), reason="native platform required"
    ),
]


async def until(probe):
    async with asyncio.timeout(8):
        while not probe():
            await asyncio.sleep(0.005)


def opaque_busy(runtime):
    with runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as claim:
        return claim is None


@pytest.fixture
async def event_case(native_console, hook_file):
    rig = native_console
    _save_hooks(hook_file, {})
    runtime = ConsoleRuntime(rig.controller.app)
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    runtime.ensure_hook_permissions().v2_configuration()
    tasks = []
    case = SimpleNamespace(rig=rig, runtime=runtime, tasks=tasks)
    try:
        yield case
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.dispose()


async def test_real_command_stays_owned_until_original_lifecycle_publish(
    event_case, monkeypatch
):
    case = event_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (command(name="SessionStart", required=True),),
        lambda *_: True,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = HookSessionLifecycle.initialize

    async def held_initialize(owner, token):
        result = await original(owner, token)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(HookSessionLifecycle, "initialize", held_initialize)
    task = asyncio.create_task(case.runtime.prepare_hooks_v2(case.rig.session.id))
    case.tasks.append(task)
    try:
        await until(entered.is_set)
        assert not engine.processes.records, (
            "actual command must be physically terminal"
        )
        assert opaque_busy(case.runtime), "lifecycle publish escaped native custody"
        counts = case.runtime.execution_capacity.snapshot()
        assert counts.tool_workers == 1 and counts.child_executions == 0
    finally:
        release.set()
    owner = await task
    assert owner.live
    await until(lambda: not opaque_busy(case.runtime))
    assert case.runtime.execution_capacity.snapshot().tool_workers == 0


async def test_direct_planned_handler_has_native_operation_during_real_process(
    event_case, tmp_path
):
    case = event_case
    entered, release = tmp_path / "entered", tmp_path / "release"
    script = (
        "from pathlib import Path;import time;"
        f"Path({str(entered)!r}).touch();"
        f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(.01)'!r})"
    )
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (command(script, name="PreCompact", required=True),),
        lambda *_: True,
    )
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    event = lifecycle_event(
        "PreCompact", case.rig.session.id, data={"reason": "manual", "candidate": []}
    )
    scope = engine.begin_event(event)
    engine.plan_handlers(scope, event)
    task = asyncio.create_task(engine.fire_handler_async(scope, "command", event))
    case.tasks.append(task)
    try:
        await until(entered.exists)
        assert opaque_busy(case.runtime), (
            "actual direct command ran outside native custody"
        )
        assert case.runtime.execution_capacity.snapshot().tool_workers == 1
    finally:
        release.touch()
        scope.close()
    await task
    await until(lambda: not opaque_busy(case.runtime))


async def test_tool_checkpoint_acceptance_keeps_original_command_hold(
    event_case, monkeypatch
):
    import threading

    from Tests.Agents.test_agent_runtime import CALC
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import ToolHookRun
    from tldw_chatbook.Agents.tool_catalog import ToolDefinitionSnapshot

    case = event_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (command(required=True),), lambda *_: True
    )
    definition = ToolDefinitionSnapshot.from_schema(CALC, "builtin", 0, 1)
    run = ToolHookRun(
        engine,
        run_id="actual-host-run",
        session_id=case.rig.session.id,
        turn_id="turn",
        resolve_definition=lambda _: definition,
    )
    entered, release = threading.Event(), threading.Event()
    original = run.checkpoints.accept

    def held_accept(*args):
        entered.set()
        assert release.wait(8)
        return original(*args)

    monkeypatch.setattr(run.checkpoints, "accept", held_accept)

    def consume():
        try:
            call = run.prepare_call(ToolCall("calculator", {}, "call"))
            run.validate_dispatch(call)
        finally:
            run.settle()

    task = asyncio.create_task(asyncio.to_thread(consume))
    case.tasks.append(task)
    try:
        await until(entered.is_set)
        await asyncio.sleep(0.05)
        assert opaque_busy(case.runtime), "tool checkpoint acceptance escaped custody"
        assert case.runtime.execution_capacity.snapshot().tool_workers == 1
    finally:
        release.set()
    await task
    await until(lambda: not opaque_busy(case.runtime))


async def test_compaction_candidate_preparation_keeps_original_command_hold(event_case):
    import threading

    from tldw_chatbook.Agents.hooks_v2.lifecycle import CompactionHooks

    case = event_case
    case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (command(name="PreCompact", required=True),),
        lambda *_: True,
    )
    lifecycle = await case.runtime.prepare_hooks_v2(case.rig.session.id)
    entered, release = threading.Event(), threading.Event()

    def prepare(messages, cap):
        entered.set()
        assert release.wait(8)
        return SimpleNamespace(known_overflow=False)

    hooks = CompactionHooks(
        lifecycle,
        lifecycle.open_scope(),
        reason="manual",
        prepare=prepare,
        current=lambda: True,
        memory_current=lambda _: True,
    )
    task = asyncio.create_task(
        asyncio.to_thread(lambda: asyncio.run(hooks.before((), 100)))
    )
    case.tasks.append(task)
    try:
        await until(entered.is_set)
        await asyncio.sleep(0.05)
        assert opaque_busy(case.runtime), "compaction prepare escaped native custody"
        assert case.runtime.execution_capacity.snapshot().tool_workers == 1
    finally:
        release.set()
    assert await task == ()
    await until(lambda: not opaque_busy(case.runtime))


@pytest.mark.parametrize("entry", ["fire", "individual", "notify"])
async def test_native_reservation_precedes_actual_original_queue_acceptance(
    event_case, monkeypatch, entry
):
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    case = event_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (command(name="Stop"),), lambda *_: True
    )
    event = lifecycle_event("Stop", case.rig.session.id)
    queued = []
    original = asyncio.run_coroutine_threadsafe
    original_soon = engine.loop.call_soon_threadsafe

    def thread_enqueue(coroutine, loop):
        if coroutine.cr_code.co_name in {"fire_async", "_fire", "fire_handler_async"}:
            queued.append(case.runtime.execution_capacity.snapshot().tool_workers)
        return original(coroutine, loop)

    def notify_enqueue(callback, *args, **kwargs):
        if getattr(callback, "__name__", None) == "submit":
            queued.append(case.runtime.execution_capacity.snapshot().tool_workers)
        return original_soon(callback, *args, **kwargs)

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", thread_enqueue)
    monkeypatch.setattr(engine.loop, "call_soon_threadsafe", notify_enqueue)
    if entry == "fire":
        await asyncio.to_thread(engine.fire, event)
    elif entry == "individual":
        scope = engine.begin_event(event)
        try:
            await asyncio.to_thread(engine.fire_handler, scope, "command", event)
        finally:
            scope.close()
    else:
        assert engine.notify(event)
    await until(lambda: not engine.cleanup_pending)
    assert queued and all(count == 1 for count in queued), queued


@pytest.mark.parametrize("begin_first", [False, True])
async def test_session_end_retains_actual_operation_before_capacity_closes(
    event_case, tmp_path, monkeypatch, begin_first
):
    case = event_case
    marker, release = tmp_path / "terminal-entered", tmp_path / "terminal-release"
    code = (
        "from pathlib import Path;import time;"
        f"Path({str(marker)!r}).touch();"
        f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(.01)'!r})"
    )
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (command(code, name="SessionEnd"),), lambda *_: True
    )
    await case.runtime.prepare_hooks_v2(case.rig.session.id)
    capacity = case.runtime.execution_capacity
    at_close = []
    original = capacity.close

    def close():
        at_close.append(capacity.snapshot().tool_workers)
        return original()

    monkeypatch.setattr(capacity, "close", close)
    if begin_first:
        case.runtime.begin_dispose()
    task = asyncio.create_task(case.runtime.dispose())
    case.tasks.append(task)
    try:
        await until(lambda: marker.exists() or task.done())
        assert marker.exists(), dict(engine.notification_failures)
        assert at_close == [1], "terminal operation was not pre-retained"
        assert opaque_busy(case.runtime)
    finally:
        release.touch()
    await task
    assert not engine.cleanup_pending
    assert capacity.snapshot().tool_workers == 0


@pytest.mark.parametrize("close", ["runtime", "session"])
async def test_cancelled_waiters_join_original_process_owner_and_storage(
    event_case, monkeypatch, close
):
    import threading

    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    case = event_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (command(name="Stop"),), lambda *_: True
    )
    entered, release = threading.Event(), threading.Event()
    original = engine.processes.owner.settle_process

    def held_settlement(*args):
        entered.set()
        assert release.wait(12)
        return original(*args)

    monkeypatch.setattr(engine.processes.owner, "settle_process", held_settlement)
    fired = asyncio.create_task(
        engine.fire_async(lifecycle_event("Stop", case.rig.session.id))
    )
    case.tasks.append(fired)
    await until(entered.is_set)
    fired.cancel()
    await asyncio.gather(fired, return_exceptions=True)
    closed = asyncio.create_task(
        case.runtime.dispose()
        if close == "runtime"
        else case.runtime.close_session(
            case.rig.session.id,
            expected_revision=case.rig.controller.lifecycle_impact(
                session_id=case.rig.session.id
            ).revision,
            timeout_seconds=0,
        )
    )
    case.tasks.append(closed)
    await until(
        lambda: (
            case.runtime._disposed
            if close == "runtime"
            else case.rig.session.id in case.runtime._admission_fenced_sessions
        )
    )
    try:
        for _ in range(2):
            closed.cancel()
            await asyncio.sleep(0.025)
        assert not closed.done(), "cancelled waiter abandoned original callback"
        assert opaque_busy(case.runtime)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 1
        assert (
            case.rig.store.persistence.db.get_connection()
            .execute("SELECT 1")
            .fetchone()[0]
            == 1
        )
    finally:
        release.set()
    result = (await asyncio.gather(closed, return_exceptions=True))[0]
    assert isinstance(result, asyncio.CancelledError)
    assert not engine.cleanup_pending
    assert not opaque_busy(case.runtime)


async def test_operation_finish_return_remains_owned_after_counter_reaches_zero(
    event_case, monkeypatch
):
    import threading

    from tldw_chatbook.Agents.execution_capacity import OwnedOperation
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    case = event_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (command(name="Stop"),), lambda *_: True
    )
    entered, release = threading.Event(), threading.Event()
    original = OwnedOperation.finish

    def held_return(operation):
        original(operation)
        entered.set()
        assert release.wait(12)

    monkeypatch.setattr(OwnedOperation, "finish", held_return)
    await engine.fire_async(lifecycle_event("Stop", case.rig.session.id))
    await until(entered.is_set)
    closed = asyncio.create_task(case.runtime.dispose())
    case.tasks.append(closed)
    try:
        await asyncio.sleep(0.03)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 0
        assert opaque_busy(case.runtime), (
            "finish counter was mistaken for callback return"
        )
        assert not closed.done()
    finally:
        release.set()
    await closed
    assert not opaque_busy(case.runtime)


async def test_failed_original_required_process_settlement_retains_native_protection(
    event_case, monkeypatch
):
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    case = event_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (command(name="Stop"),), lambda *_: True
    )
    original = engine.processes.owner.settle_process

    def failed_settlement(*args):
        original(*args)
        raise OSError("required original settlement refused")

    monkeypatch.setattr(engine.processes.owner, "settle_process", failed_settlement)
    outcome = await engine.fire_async(lifecycle_event("Stop", case.rig.session.id))
    assert outcome.outstanding_cleanup
    await engine.close()
    assert opaque_busy(case.runtime)
    counts = case.runtime.execution_capacity.snapshot()
    assert counts.tool_workers == counts.stopping_tool_workers == 1


@pytest.mark.parametrize("resource", ["capacity", "activity"])
async def test_source_replacement_during_native_reserve_prevents_real_command(
    event_case, tmp_path, monkeypatch, resource
):
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner, RuntimeCapacity
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeActivity

    case = event_case
    marker = tmp_path / "must-not-launch"
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (
            command(
                f"from pathlib import Path;Path({str(marker)!r}).touch()", name="Stop"
            ),
        ),
        lambda *_: True,
    )
    original_capacity, original_activity = (
        case.runtime.execution_capacity,
        case.runtime.worktree_activity,
    )
    original = ExecutionOwner.reserve_tool

    def replaced(owner):
        assert not engine._lock._is_owned()
        assert not case.runtime._run_hooks_lock._is_owned()
        operation = original(owner)
        if resource == "capacity":
            case.runtime._execution_capacity = RuntimeCapacity()
        else:
            case.runtime._worktree_activity = WorktreeActivity()
        return operation

    monkeypatch.setattr(ExecutionOwner, "reserve_tool", replaced)
    try:
        outcome = await engine.fire_async(lifecycle_event("Stop", case.rig.session.id))
        assert outcome.failures and not outcome.accepted
        assert not marker.exists()
        assert original_capacity.snapshot().tool_workers == 0
    finally:
        case.runtime._execution_capacity, case.runtime._worktree_activity = (
            original_capacity,
            original_activity,
        )


async def test_disabled_and_no_target_events_leave_native_resources_cold(event_case):
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    case = event_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (command(name="Stop"),), lambda *_: True
    )
    assert (
        case.runtime._execution_capacity is None
        and case.runtime._worktree_activity is None
    )
    assert (
        await engine.fire_async(
            lifecycle_event(
                "SessionStart", case.rig.session.id, data={"reason": "startup"}
            )
        )
    ).succeeded
    engine.enabled = False
    assert not (
        await engine.fire_async(lifecycle_event("Stop", case.rig.session.id))
    ).succeeded
    assert (
        case.runtime._execution_capacity is None
        and case.runtime._worktree_activity is None
    )


@pytest.mark.parametrize("replacement", ["capacity", "session"])
async def test_actual_delivery_publication_lock_rechecks_original_source(
    event_case, tmp_path, monkeypatch, replacement
):
    import inspect
    import sys
    import threading
    import time

    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    case = event_case
    marker = tmp_path / "stale-publication"
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (
            command(
                f"from pathlib import Path;Path({str(marker)!r}).touch()", name="Stop"
            ),
        ),
        lambda *_: True,
    )
    capacity = case.runtime.execution_capacity
    event = lifecycle_event("Stop", case.rig.session.id)
    scope = engine.begin_event(event)
    lines, start = inspect.getsourcelines(HookEngine.fire_handler_async)
    lock_line = start + next(
        index for index, line in enumerate(lines) if line.strip() == "with self._lock:"
    )
    owner_thread = threading.get_ident()
    locked = threading.Event()
    observed = []
    original_revision = case.rig.session.conversation_binding_revision

    def contender():
        with engine._lock:
            locked.set()
            end = time.monotonic() + 8
            while time.monotonic() < end:
                frame = sys._current_frames().get(owner_thread)
                while frame:
                    if (
                        frame.f_code is HookEngine.fire_handler_async.__code__
                        and frame.f_lineno == lock_line
                    ):
                        observed.append(capacity.snapshot().tool_workers)
                        if replacement == "capacity":
                            case.runtime._execution_capacity = RuntimeCapacity()
                        else:
                            case.rig.session.conversation_binding_revision += 1
                        return
                    frame = frame.f_back
                time.sleep(0.001)
            observed.append("actual publication lock not reached")

    thread = threading.Thread(target=contender)
    thread.start()
    assert locked.wait(2)
    try:
        result = await engine.fire_handler_async(scope, "command", event)
        assert observed == [1]
        assert result.failures and not result.accepted
        assert not marker.exists()
    finally:
        thread.join(2)
        scope.close()
        case.runtime._execution_capacity = capacity
        case.rig.session.conversation_binding_revision = original_revision
    await until(lambda: not opaque_busy(case.runtime))


async def test_real_saved_grant_revocation_before_enqueue_refuses_command(
    event_case, hook_file, tmp_path, monkeypatch
):
    import threading

    from Tests.Agents.test_hook_permissions import _approve
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    case = event_case
    marker = tmp_path / "revoked-command"
    definition = command(
        f"from pathlib import Path;Path({str(marker)!r}).touch()", name="Stop"
    )
    _save_hooks(
        hook_file,
        {
            "enabled": True,
            "handler": [
                {
                    "id": definition.id,
                    "type": "command",
                    "event": "Stop",
                    "effects": [],
                    "argv": list(definition.argv),
                }
            ],
        },
    )
    permissions = case.runtime.ensure_hook_permissions()
    assert _approve(permissions).ready
    lifecycle = await case.runtime.prepare_hooks_v2(case.rig.session.id)
    engine = lifecycle.engine
    entered, release = threading.Event(), threading.Event()
    original = ExecutionOwner.reserve_tool

    def held_reserve(owner):
        assert not engine._lock._is_owned()
        operation = original(owner)
        entered.set()
        assert release.wait(8)
        return operation

    monkeypatch.setattr(ExecutionOwner, "reserve_tool", held_reserve)
    task = asyncio.create_task(
        engine.fire_async(lifecycle_event("Stop", case.rig.session.id))
    )
    case.tasks.append(task)
    try:
        await until(entered.is_set)
        review = permissions.snapshot()
        permissions.revoke(review, review.rows[0].entry.key)
    finally:
        release.set()
    result = await task
    assert result.failures and not result.accepted and not marker.exists()
    await until(lambda: not opaque_busy(case.runtime))


async def test_actual_mcp_event_preserves_whole_uncovered_chain_and_independent_command(
    event_case, tmp_path, monkeypatch
):
    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    case = event_case
    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        case.rig.controller.app.unified_mcp_service = service
        tool = HubTool(
            server_key="local:fixture",
            server_label="fixture",
            source="local",
            name="fixture",
            description="",
            input_schema={"type": "object"},
            tags=(),
            stale=False,
            executable=True,
        )
        service.set_tool_state("local:fixture", "fixture", "allow", tool=tool)
        handlers = parse_handlers(
            [
                {
                    "id": "mcp-start",
                    "event": "SessionStart",
                    "type": "mcp_tool",
                    "server": "local:fixture",
                    "tool": "fixture",
                    "required": True,
                    "effects": [],
                    "input": {
                        "payload": {
                            "structuredContent": {"version": 2, "decision": "pass"}
                        }
                    },
                }
            ]
        )
        handlers += (
            command(name="SessionStart", id="sibling", required=True),
            command(required=True),
            command(name="PreCompact", id="independent", required=True),
        )
        engine = case.runtime.ensure_hooks_v2(
            case.rig.session.id, handlers, lambda *_: True
        )
        entries = []
        original = engine.processes.start

        def observed(handler, *args, **kwargs):
            counts = case.runtime._execution_capacity
            entries.append(
                (handler.id, 0 if counts is None else counts.snapshot().tool_workers)
            )
            return original(handler, *args, **kwargs)

        monkeypatch.setattr(engine.processes, "start", observed)
        result = await case.rig.controller.submit_draft(
            "hello", session_id=case.rig.session.id
        )
        assert result.accepted, result
        assert service.execution_log.read_recent(), "actual MCP peer was not invoked"
        assert ("command", 0) in entries and ("sibling", 0) in entries
        assert (
            case.runtime._execution_capacity is None
            or case.runtime.execution_capacity.snapshot().tool_workers == 0
        )
        lifecycle = engine.lifecycle_owner
        await lifecycle.fire(
            lifecycle.event("PreCompact", data={"candidate": [], "reason": "manual"}),
            lifecycle.scope_id,
        )
        assert ("independent", 1) in entries, (
            "private uncovered context escaped its original delivery"
        )
        await until(lambda: not engine.cleanup_pending)


@pytest.mark.parametrize("event_name", ["SessionStart", "UserPromptSubmit"])
async def test_original_manual_submission_uses_reserve_event_labels_do_not(
    event_case, tmp_path, monkeypatch, event_name
):
    from tldw_chatbook.Agents.execution_capacity import (
        RuntimeCapacity,
        WorkOrigin,
        current_execution_owner,
    )
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    case = event_case
    case.runtime._execution_capacity = RuntimeCapacity(
        max_tool_workers=1, reserved_manual_tool_workers=1
    )
    marker = tmp_path / "manual"
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (
            command(
                f"from pathlib import Path;Path({str(marker)!r}).touch()",
                name=event_name,
                required=True,
            ),
        ),
        lambda *_: True,
    )
    labelled = await engine.fire_async(
        lifecycle_event(
            event_name,
            case.rig.session.id,
            initiator="manual",
            data={"reason": "startup"}
            if event_name == "SessionStart"
            else {"prompt": "hello"},
        )
    )
    assert labelled.failures and not marker.exists()
    origins = []
    original = engine.processes.start

    def launch(*args, **kwargs):
        origins.append(current_execution_owner().origin)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine.processes, "start", launch)
    assert (
        await case.rig.controller.submit_draft("hello", session_id=case.rig.session.id)
    ).accepted
    assert marker.exists() and origins == [WorkOrigin.MANUAL]
    await until(lambda: not engine.cleanup_pending)


async def test_unqualified_ordinary_command_keeps_original_execution(
    event_case, tmp_path, monkeypatch
):
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event
    from tldw_chatbook.Chat import console_workflow_runs

    case = event_case
    monkeypatch.setattr(
        console_workflow_runs, "native_workflow_qualified", lambda: False
    )
    marker = tmp_path / "ordinary"
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (
            command(
                f"from pathlib import Path;Path({str(marker)!r}).touch()", name="Stop"
            ),
        ),
        lambda *_: True,
    )
    assert (
        await engine.fire_async(lifecycle_event("Stop", case.rig.session.id))
    ).succeeded
    assert marker.exists() and engine._native_commands is None
    assert (
        case.runtime._execution_capacity is None
        and case.runtime._worktree_activity is None
    )


@pytest.mark.parametrize("required", [False, True])
async def test_busy_event_preserves_original_required_policy_without_launch(
    event_case, tmp_path, required
):
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    case = event_case
    marker = tmp_path / "busy"
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (
            command(
                f"from pathlib import Path;Path({str(marker)!r}).touch()",
                name="SessionStart",
                required=required,
            ),
        ),
        lambda *_: True,
    )
    with case.runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as held:
        assert held is not None
        result = await engine.fire_async(
            lifecycle_event(
                "SessionStart", case.rig.session.id, data={"reason": "startup"}
            )
        )
        assert result.failures and result.allowed is (not required)
        assert (
            not marker.exists()
            and case.runtime.execution_capacity.snapshot().tool_workers == 0
        )


async def test_original_job_cancellation_does_not_stand_for_settlement_callback_return(
    event_case, monkeypatch
):
    import threading

    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    case = event_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (command(name="Stop"),), lambda *_: True
    )
    entered, release = threading.Event(), threading.Event()
    original = engine.processes.owner.settle_process

    def held(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(engine.processes.owner, "settle_process", held)
    fired = asyncio.create_task(
        engine.fire_async(lifecycle_event("Stop", case.rig.session.id))
    )
    case.tasks.append(fired)
    await until(entered.is_set)
    job = next(iter(engine.processes.records.values()))
    for _ in range(2):
        job.task.cancel()
        await asyncio.sleep(0.01)
    fired.cancel()
    await asyncio.gather(fired, return_exceptions=True)
    drain = asyncio.create_task(asyncio.to_thread(engine._native_commands.wait_closed))
    case.tasks.append(drain)
    try:
        await asyncio.sleep(0.05)
        assert not drain.done(), "cancelled job was mistaken for actual callback return"
        assert opaque_busy(case.runtime)
    finally:
        release.set()
    await drain


@pytest.mark.parametrize("change", ["matcher", "definitions", "attachment"])
async def test_registered_event_maximum_cannot_downgrade_or_widen(
    event_case, tmp_path, change
):
    from Tests.Agents.test_hooks_v2_execution import event
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    case = event_case
    marker = tmp_path / "must-not-run"
    handlers = (
        command(
            f"from pathlib import Path;Path({str(marker)!r}).touch()", required=True
        ),
    )
    handlers += parse_handlers(
        [
            {
                "id": "mcp",
                "event": "PreToolUse",
                "type": "mcp_tool",
                "server": "local:fixture",
                "tool": "fixture",
                "effects": [],
                "match": {"provider": ["mcp"]},
            }
        ]
    )
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, handlers, lambda *_: True
    )
    original = event(runtime=case.rig.session.id)
    scope = engine.begin_event(original)
    assert scope._state.native.covered
    with pytest.raises(AttributeError):
        scope._state.native.covered = False
    engine.plan_handlers(scope, original)
    saved_source = engine._native_commands
    try:
        if change == "matcher":
            changed = original.model_copy(
                update={"data": dict(original.data, provider="mcp")}
            )
        elif change == "definitions":
            engine.definitions = tuple(reversed(handlers))
            changed = original
        else:
            engine._native_commands = None
            changed = original
        result = await engine.fire_handler_async(scope, "command", changed)
        assert result.failures and not result.accepted and not marker.exists()
        assert case.runtime._execution_capacity is None
    finally:
        scope.close()
        engine.definitions = handlers
        engine._native_commands = saved_source


async def test_actual_compaction_render_rechecks_source_before_ledger_publication(
    event_case, monkeypatch
):
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
    from tldw_chatbook.Agents.hooks_v2.context import ContextLedger
    from tldw_chatbook.Agents.hooks_v2.lifecycle import CompactionHooks

    case = event_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (
            command(
                'print(\'{"version":2,"decision":"pass","context":[{"text":"candidate","lifetime":"turn"}]}\')',
                name="PreCompact",
                effects=["context"],
                required=True,
            ),
        ),
        lambda *_: True,
    )
    lifecycle = await case.runtime.prepare_hooks_v2(case.rig.session.id)
    capacity = case.runtime.execution_capacity
    rendered, prepared = [], []
    original = ContextLedger.render

    def replace_source(*args):
        assert capacity.snapshot().tool_workers == 1 and opaque_busy(case.runtime)
        rendered.append(args[0].event_id)
        value = original(*args)
        case.runtime._execution_capacity = RuntimeCapacity()
        return value

    monkeypatch.setattr(ContextLedger, "render", staticmethod(replace_source))
    hooks = CompactionHooks(
        lifecycle,
        lifecycle.open_scope(),
        reason="manual",
        prepare=lambda *_: prepared.append(True),
        current=lambda: True,
        memory_current=lambda _: True,
    )
    try:
        with pytest.raises(ValueError, match="source changed"):
            await hooks.before((), 100)
        assert rendered and not prepared
    finally:
        case.runtime._execution_capacity = capacity
    await until(lambda: not engine.cleanup_pending)


async def test_missing_original_session_cannot_issue_command_custody(
    event_case, tmp_path
):
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    case = event_case
    marker = tmp_path / "missing-session"
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (
            command(
                f"from pathlib import Path;Path({str(marker)!r}).touch()", name="Stop"
            ),
        ),
        lambda *_: True,
    )
    original = case.rig.store._sessions.pop(case.rig.session.id)
    try:
        result = await engine.fire_async(lifecycle_event("Stop", case.rig.session.id))
        assert result.failures and not marker.exists()
        assert (
            case.runtime._execution_capacity is None
            and case.runtime._worktree_activity is None
        )
    finally:
        case.rig.store._sessions[original.id] = original


async def test_actual_stop_proposal_consumer_keeps_original_event_hold(
    event_case, monkeypatch
):
    from uuid import uuid4

    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest

    case = event_case
    case.rig.controller.app.local_chat_conversation_service = ChatConversationService(
        case.rig.store.persistence.db
    )
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (
            command(
                'print(\'{"version":2,"decision":"pass","continuation":{"message":"next"}}\')',
                name="Stop",
                effects=["continuation"],
            ),
        ),
        lambda *_: True,
    )
    queue = case.rig.controller.prompt_queue_coordinator
    original = queue.schedule_continuation
    entered, release = asyncio.Event(), asyncio.Event()
    once = False

    async def held(*args):
        nonlocal once
        result = await original(*args)
        if not once:
            once = True
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(queue, "schedule_continuation", held)
    request = ConsoleTurnCustodyRequest(
        turn_id=str(uuid4()),
        session_id=case.rig.session.id,
        draft="hello",
        configuration=case.rig.controller.resolve_turn_configuration_snapshot(
            case.rig.session.id
        ),
    )
    turn = case.runtime.accept_turn(request)
    task = asyncio.create_task(case.runtime.wait_for_turn(turn))
    case.tasks.append(task)
    try:
        await until(entered.is_set)
        await asyncio.sleep(0.05)
        assert not engine.processes.records
        assert opaque_busy(case.runtime), (
            "Stop proposal escaped before original queue acceptance returned"
        )
        assert case.runtime.execution_capacity.snapshot().tool_workers == 1
    finally:
        release.set()
    await task
    await until(lambda: not engine.cleanup_pending)


@pytest.mark.parametrize("scope_kind", ["fabricated", "foreign", "closed"])
async def test_async_host_execution_requires_original_live_engine_scope(
    event_case, tmp_path, scope_kind
):
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    case = event_case
    marker = tmp_path / "invalid-scope"
    handler = command(
        f"from pathlib import Path;Path({str(marker)!r}).touch()", name="Stop"
    )
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (handler,), lambda *_: True
    )
    event = lifecycle_event("Stop", case.rig.session.id)
    foreign = HookEngine((handler,), lambda *_: True, HookBudgetOwner())
    scope = (
        object()
        if scope_kind == "fabricated"
        else (
            foreign.begin_event(event)
            if scope_kind == "foreign"
            else engine.begin_event(event)
        )
    )
    if scope_kind == "closed":
        scope.close()
    try:
        result = await engine.fire_async(event, _execution=scope)
        assert result.failures and not result.accepted and not marker.exists()
        assert (
            case.runtime._execution_capacity is None
            and case.runtime._worktree_activity is None
        )
    finally:
        if scope_kind != "fabricated":
            scope.close()
        await foreign.close()
