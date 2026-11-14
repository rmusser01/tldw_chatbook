"""Original native MCP workers, nested holds and real terminal handoffs."""

import asyncio
import threading

import pytest

from Tests.Agents.test_hooks_v2_execution import command
from Tests.Chat.test_native_v2_command_events import (
    event_case as _event_case,
)
from Tests.Chat.test_native_v2_command_events import (
    hook_file as _hook_file,
)
from Tests.Chat.test_native_v2_command_events import (
    native_console as _native_console,
)
from Tests.Chat.test_native_v2_command_events import (
    native_package as _native_package,
)
from Tests.Chat.test_native_v2_command_events import (
    opaque_busy,
    until,
)
from Tests.MCP.test_typed_tool_results import (
    controlled_service,
    controlled_stdio_client,
)
from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers
from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified
from tldw_chatbook.MCP.hub_tool_catalog import HubTool

event_case = _event_case
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


def mcp_handler(event="SessionStart", *, required=True, id="mcp", error=False):
    return parse_handlers(
        [
            {
                "id": id,
                "event": event,
                "type": "mcp_tool",
                "server": "local:fixture",
                "tool": "fixture",
                "required": required,
                "effects": [],
                "input": {
                    "payload": {
                        "isError": error,
                        "structuredContent": {"version": 2, "decision": "pass"},
                    }
                },
            }
        ]
    )[0]


@pytest.fixture
async def mcp_case(event_case, tmp_path):
    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        event_case.rig.controller.app.unified_mcp_service = service
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
        event_case.service, event_case.client, event_case.tool = service, client, tool
        yield event_case
        await event_case.runtime.close_hooks_v2()


async def test_original_mcp_worker_return_keeps_one_operation_and_opaque_claim(
    mcp_case, monkeypatch
):
    case = mcp_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (mcp_handler(),), lambda *_: True
    )
    entered, release = threading.Event(), threading.Event()
    original = engine.mcp_executor._invoke_worker_bound

    def held(job):
        result = original(job)
        entered.set()
        assert release.wait(8)
        return result

    monkeypatch.setattr(engine.mcp_executor, "_invoke_worker_bound", held)
    task = asyncio.create_task(
        case.rig.controller.submit_draft("hello", session_id=case.rig.session.id)
    )
    case.tasks.append(task)
    try:
        await until(entered.is_set)
        assert case.service.execution_log.read_recent(), "real MCP request did not run"
        assert opaque_busy(case.runtime), "actual MCP worker return escaped custody"
        counts = case.runtime.execution_capacity.snapshot()
        assert counts.tool_workers == 1 and counts.child_executions == 0
    finally:
        release.set()
    assert (await task).accepted
    await until(lambda: not engine.cleanup_pending)
    assert not opaque_busy(case.runtime)


async def test_session_end_nested_workers_consume_original_holds_after_capacity_close(
    mcp_case, monkeypatch, tmp_path
):
    case = mcp_case
    marker = tmp_path / "nested"
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (
            mcp_handler("SessionEnd", required=False),
            command(
                f"from pathlib import Path;Path({str(marker)!r}).touch()", required=True
            ),
            command(name="PostToolUse", id="post", required=True),
        ),
        lambda *_: True,
    )
    assert (
        await case.rig.controller.submit_draft("hello", session_id=case.rig.session.id)
    ).accepted
    entered, release = threading.Event(), threading.Event()
    original = engine.mcp_executor._invoke_worker_bound

    errors = []

    def held(job):
        entered.set()
        assert release.wait(8)
        try:
            return original(job)
        except Exception as error:
            errors.append(repr(error))
            raise

    monkeypatch.setattr(engine.mcp_executor, "_invoke_worker_bound", held)
    engine.lifecycle_owner.seal()
    try:
        await until(entered.is_set)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 3, (
            "terminal event did not retain exact MCP/Pre/Post workers before close"
        )
        case.runtime.execution_capacity.close()
        assert opaque_busy(case.runtime)
    finally:
        release.set()
    await until(lambda: not engine.cleanup_pending)
    assert marker.exists(), (dict(engine.notification_failures), errors)
    assert case.service.execution_log.read_recent(), dict(engine.notification_failures)
    assert case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert not opaque_busy(case.runtime)


@pytest.mark.parametrize("error", [False, True])
async def test_conditional_post_failure_holds_are_used_or_conclusively_discarded(
    mcp_case, monkeypatch, tmp_path, error
):
    case = mcp_case
    marker = tmp_path / "failure"
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (
            mcp_handler(error=error),
            command(name="PostToolUse", id="post", required=True),
            command(
                f"from pathlib import Path;Path({str(marker)!r}).touch()",
                name="PostToolUseFailure",
                id="failure",
                required=True,
            ),
        ),
        lambda *_: True,
    )
    original = engine.mcp_executor._invoke_worker_bound
    counts = []

    def observed(job):
        counts.append(case.runtime.execution_capacity.snapshot().tool_workers)
        return original(job)

    monkeypatch.setattr(engine.mcp_executor, "_invoke_worker_bound", observed)
    result = await case.rig.controller.submit_draft(
        "hello", session_id=case.rig.session.id
    )
    assert result.accepted is not error
    await until(lambda: not engine.cleanup_pending)
    assert counts == [3]
    assert marker.exists() is error
    assert not opaque_busy(case.runtime)
    assert case.runtime.execution_capacity.snapshot().tool_workers == 0


async def test_real_mcp_lifecycle_publish_and_operation_finish_return_are_joined(
    mcp_case, monkeypatch
):
    from tldw_chatbook.Agents.execution_capacity import OwnedOperation
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle

    case = mcp_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (mcp_handler(),), lambda *_: True
    )
    host_entered, host_release = asyncio.Event(), asyncio.Event()
    finish_entered, finish_release = threading.Event(), threading.Event()
    initialize, finish = HookSessionLifecycle.initialize, OwnedOperation.finish

    async def held_initialize(owner, token):
        result = await initialize(owner, token)
        host_entered.set()
        await host_release.wait()
        return result

    def held_finish(operation):
        result = finish(operation)
        finish_entered.set()
        assert finish_release.wait(8)
        return result

    monkeypatch.setattr(HookSessionLifecycle, "initialize", held_initialize)
    monkeypatch.setattr(OwnedOperation, "finish", held_finish)
    task = asyncio.create_task(
        case.rig.controller.submit_draft("hello", session_id=case.rig.session.id)
    )
    case.tasks.append(task)
    try:
        await until(host_entered.is_set)
        assert not engine.mcp_executor._jobs
        assert opaque_busy(case.runtime)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 1
        host_release.set()
        await until(finish_entered.is_set)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 0
        assert opaque_busy(case.runtime), (
            "counter decrement is not finish callback return"
        )
    finally:
        host_release.set()
        finish_release.set()
    assert (await task).accepted
    await until(lambda: not engine.cleanup_pending)
    assert not opaque_busy(case.runtime)


@pytest.mark.parametrize("closure", ["runtime", "session"])
async def test_cancelled_mcp_bridge_and_repeated_disposal_waiters_retain_actual_return(
    mcp_case, monkeypatch, closure
):
    case = mcp_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (mcp_handler(),), lambda *_: True
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = case.service.execute_hub_tool_result

    async def held(*args, **kwargs):
        result = await original(*args, **kwargs)
        entered.set()
        while not release.is_set():
            try:
                await asyncio.shield(release.wait())
            except asyncio.CancelledError:
                pass
        return result

    monkeypatch.setattr(case.service, "execute_hub_tool_result", held)
    pending = asyncio.create_task(
        case.rig.controller.submit_draft("hello", session_id=case.rig.session.id)
    )
    case.tasks.append(pending)
    close = None
    try:
        await until(entered.is_set)
        job = next(iter(engine.mcp_executor._jobs))
        assert job.bridges and not job.bridges[0].completed.is_set()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        close = asyncio.create_task(
            case.runtime.dispose()
            if closure == "runtime"
            else case.runtime.close_session(
                case.rig.session.id,
                expected_revision=case.rig.controller.lifecycle_impact(
                    session_id=case.rig.session.id
                ).revision,
                timeout_seconds=0,
            )
        )
        case.tasks.append(close)
        await until(
            lambda: (
                case.runtime.execution_capacity.snapshot().closed
                if closure == "runtime"
                else case.rig.session.id in case.runtime._admission_fenced_sessions
            )
        )
        close.cancel()
        close.cancel()
        await asyncio.sleep(0.05)
        assert opaque_busy(case.runtime)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 1
        assert not job.bridges[0].completed.is_set()
        assert case.rig.session.id in case.rig.store._sessions
    finally:
        release.set()
    if close is not None:
        await asyncio.gather(close, return_exceptions=True)
    await case.runtime.dispose()
    assert not engine.cleanup_pending
    assert job.bridges[0].completed.is_set()
    assert not opaque_busy(case.runtime)


async def test_configured_runtime_disposal_keeps_original_mcp_session_end_refusal(
    mcp_case, hook_file, monkeypatch
):
    from Tests.Chat.test_console_hook_preparation_demand import _save_hooks

    case = mcp_case
    handler = mcp_handler("SessionEnd", required=False)
    _save_hooks(
        hook_file,
        {
            "enabled": True,
            "handler": [
                {
                    k: v
                    for k, v in handler.model_dump().items()
                    if v is not None and v not in ({}, ())
                }
            ],
        },
    )
    from Tests.Agents.test_hook_permissions import _approve

    permissions = case.runtime.ensure_hook_permissions()
    case.rig.controller._hook_permissions_accessor = lambda: permissions
    assert _approve(permissions).ready
    assert (
        await case.rig.controller.submit_draft("hello", session_id=case.rig.session.id)
    ).accepted
    engine = case.runtime.get_hooks_v2(case.rig.session.id)
    assert engine is not None
    await case.runtime.dispose()
    assert not case.service.execution_log.read_recent()
    assert not engine.cleanup_pending
    assert not opaque_busy(case.runtime)


@pytest.fixture
async def owned_mcp_case(event_case, native_package, tmp_path, monkeypatch, request):
    from contextlib import aclosing

    from Tests.Plugins.test_owned_mcp_tools import shared_connection_case

    directory = tmp_path / "owned"
    directory.mkdir()
    async with aclosing(
        shared_connection_case.__wrapped__(
            native_package, directory, monkeypatch, request
        )
    ) as cases:
        async for owned in cases:
            owned.release.touch()
            assert (await owned.pending).ok
            event_case.owned = owned
            yield event_case
            await event_case.runtime.close_hooks_v2()


async def bind_owned_event(case, *, release, started):
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookContext

    owned = case.owned
    handler = parse_handlers(
        [
            {
                "id": "owned",
                "event": "PreCompact",
                "type": "mcp_tool",
                "server": "local:" + owned.profile_id,
                "tool": "echo",
                "required": True,
                "effects": [],
                "input": {"hold": str(release), "started": str(started)},
            }
        ]
    )[0]
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (handler,), lambda *_: True
    )
    # The shipping host bind seam uses the actual connected provider and registry.
    lifecycle = await case.runtime.prepare_hooks_v2(case.rig.session.id)
    engine.mcp_executor.bind_context(
        MCPHookContext(
            owned.registries["a"],
            lifecycle,
            lifecycle.scope_id,
            "run-a",
            case.rig.session.id,
            frozenset(row.name for row in owned.providers["a"].list_catalog()),
            lifecycle.current,
            lambda _definition: (),
            workspace_id="a",
        )
    )
    return engine, lifecycle.event(
        "PreCompact", data={"reason": "manual", "candidate": []}
    )


async def test_actual_plugin_request_reaper_retains_native_operation_after_bridge_cancel(
    owned_mcp_case, tmp_path
):
    case = owned_mcp_case
    release, started = tmp_path / "late-release", tmp_path / "late-started"
    engine, event = await bind_owned_event(case, release=release, started=started)
    pending = asyncio.create_task(engine.fire_async(event))
    case.tasks.append(pending)
    try:
        await until(started.exists)
        job = next(iter(engine.mcp_executor._jobs))
        assert len(job.requests) == 1
        record = job.requests[0]
        assert not record.completed.is_set()
        assert opaque_busy(case.runtime)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 1
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await until(lambda: job.bridges[0].completed.is_set())
        assert not record.completed.is_set(), (
            "cancelled bridge is not plugin request drain"
        )
        assert opaque_busy(case.runtime)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 1
        assert (await case.owned.invoke("b")).ok
    finally:
        release.touch()
    await until(lambda: not engine.cleanup_pending)
    assert record.completed.is_set()
    assert not opaque_busy(case.runtime)
    assert case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert case.owned.service._thread.is_alive(), (
        "request return must not retire plugin infrastructure"
    )


async def test_actual_plugin_required_settlement_failure_retains_original_protection(
    owned_mcp_case, monkeypatch, tmp_path
):
    case = owned_mcp_case
    release, started = tmp_path / "write-release", tmp_path / "write-started"
    engine, event = await bind_owned_event(case, release=release, started=started)
    pending = asyncio.create_task(engine.fire_async(event))
    case.tasks.append(pending)
    try:
        await until(started.exists)
        job = next(iter(engine.mcp_executor._jobs))
        record = job.requests[0]
        owner = case.owned.service._coordinator.owner
        settle = owner.settle_process
        failed = threading.Event()

        def fail_exact(token, confirmed):
            if token == record.lease_token:
                failed.set()
                raise OSError("injected original required write failure")
            return settle(token, confirmed)

        monkeypatch.setattr(owner, "settle_process", fail_exact)
        release.touch()
        await until(failed.is_set)
        result = await pending
        assert result.failures and result.outstanding_cleanup
        assert not record.completed.is_set()
        assert engine.cleanup_pending and opaque_busy(case.runtime)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 1
        # Actual response arrived; retry the original failed settlement, never invent proof.
        monkeypatch.setattr(owner, "settle_process", settle)
        await case.owned.ownership._settle(record.lease_token)
        await until(lambda: not engine.cleanup_pending)
        assert record.completed.is_set()
        assert not opaque_busy(case.runtime)
    finally:
        release.touch()


async def test_pending_actual_hook_approval_cancel_does_not_dispatch_or_leak(mcp_case):
    case = mcp_case
    case.rig.controller.app.call_from_thread = lambda callback, *args, **kwargs: (
        callback(*args, **kwargs)
    )
    case.service.set_tool_state("local:fixture", "fixture", "ask", tool=case.tool)
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (mcp_handler(),), lambda *_: True
    )
    prompts = []
    case.rig.controller.set_pending_approval = lambda payload: (
        prompts.append(payload) if payload else None
    )
    pending = asyncio.create_task(
        case.rig.controller.submit_draft("hello", session_id=case.rig.session.id)
    )
    case.tasks.append(pending)
    try:
        await until(
            lambda: bool(case.rig.controller._pending_approval_rounds) or pending.done()
        )
        assert case.rig.controller._pending_approval_rounds, (
            "actual hook approval did not reach its host"
        )
        assert opaque_busy(case.runtime)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 1
        assert not case.service.execution_log.read_recent()
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    await until(lambda: not engine.cleanup_pending)
    assert not case.rig.controller._pending_approval_rounds
    assert not opaque_busy(case.runtime)
    counter = await case.client.call_tool_result(
        "fixture", "fixture", {"counter": True}
    )
    assert counter.content[0]["text"] == "1", "cancelled approval dispatched a request"


async def test_full_nested_capacity_refusal_happens_before_any_original_worker(
    mcp_case, monkeypatch
):
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity

    case = mcp_case
    case.runtime._execution_capacity = RuntimeCapacity(
        max_tool_workers=2, reserved_manual_tool_workers=0
    )
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (
            mcp_handler(),
            command(required=True),
            command(name="PostToolUse", id="post", required=True),
        ),
        lambda *_: True,
    )
    entries = []
    original = engine.mcp_executor.start

    def start(*args, **kwargs):
        entries.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine.mcp_executor, "start", start)
    result = await case.rig.controller.submit_draft(
        "hello", session_id=case.rig.session.id
    )
    assert not result.accepted
    assert not entries and not case.service.execution_log.read_recent()
    assert case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert not engine.cleanup_pending and not opaque_busy(case.runtime)


@pytest.mark.parametrize("changed", ["context", "provider", "session"])
async def test_mcp_publication_lock_refuses_changed_original_source_before_start(
    mcp_case, monkeypatch, changed
):
    import inspect
    import sys
    import time
    from dataclasses import replace

    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    case = mcp_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (mcp_handler("PreCompact"),), lambda *_: True
    )
    assert (
        await case.rig.controller.submit_draft("hello", session_id=case.rig.session.id)
    ).accepted
    lifecycle = engine.lifecycle_owner
    event = lifecycle.event("PreCompact", data={"reason": "manual", "candidate": []})
    scope = engine.begin_event(event)
    await asyncio.to_thread(scope._state.native.admit)
    original_context = engine.mcp_executor._runtime_contexts[case.rig.session.id]
    name, provider = original_context.resolve(engine.definitions[0])
    entry = provider._entry_by_llm_name[name]
    revision = case.rig.session.conversation_binding_revision
    lines, first = inspect.getsourcelines(HookEngine.fire_handler_async)
    lock_line = first + next(
        i for i, line in enumerate(lines) if line.strip() == "with self._lock:"
    )
    main_thread = threading.get_ident()
    locked, observed = threading.Event(), []
    entries = []
    start = engine.mcp_executor.start

    def observed_start(*args, **kwargs):
        entries.append(True)
        return start(*args, **kwargs)

    monkeypatch.setattr(engine.mcp_executor, "start", observed_start)

    def contend():
        with engine._lock:
            locked.set()
            until_time = time.monotonic() + 8
            while time.monotonic() < until_time:
                frame = sys._current_frames().get(main_thread)
                while frame:
                    if (
                        frame.f_code is HookEngine.fire_handler_async.__code__
                        and frame.f_lineno == lock_line
                    ):
                        observed.append(
                            case.runtime.execution_capacity.snapshot().tool_workers
                        )
                        if changed == "context":
                            engine.mcp_executor._runtime_contexts[
                                case.rig.session.id
                            ] = replace(original_context)
                        elif changed == "provider":
                            provider._entry_by_llm_name[name] = (entry[0], entry[1])
                        else:
                            case.rig.session.conversation_binding_revision += 1
                        return
                    frame = frame.f_back
                time.sleep(0.001)
            observed.append("publication lock not reached")

    thread = threading.Thread(target=contend)
    thread.start()
    assert locked.wait(2)
    try:
        result = await engine.fire_handler_async(scope, "mcp", event)
        assert observed == [1]
        assert result.failures and not entries
        assert not case.service.execution_log.read_recent()
    finally:
        thread.join(2)
        scope.close()
        engine.mcp_executor._runtime_contexts[case.rig.session.id] = original_context
        provider._entry_by_llm_name[name] = entry
        case.rig.session.conversation_binding_revision = revision
    await until(lambda: not engine.cleanup_pending)
    assert not opaque_busy(case.runtime)


async def test_distinct_actual_nested_mcp_worker_and_command_share_one_original_event(
    mcp_case, tmp_path, monkeypatch
):
    import json
    import sys
    from dataclasses import replace

    from tldw_chatbook.Agents.execution_capacity import current_execution_owner
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile
    from tldw_chatbook.MCP.tool_naming import llm_tool_name

    case = mcp_case
    assert await case.client.connect_to_server(
        "second",
        sys.executable,
        ["-I", str(tmp_path / "peer.py"), json.dumps({"type": "object"})],
    )
    store = case.service.local_service.store
    store.save_profile(
        LocalExternalMCPProfile(profile_id="second", command=sys.executable)
    )
    store.save_discovery_snapshot("second", await case.client.describe_server("second"))
    second = replace(case.tool, server_key="local:second", server_label="second")
    case.service.set_tool_state("local:second", "fixture", "allow", tool=second)
    nested = mcp_handler("PreToolUse", id="nested").model_dump()
    nested.update(
        server="local:second",
        match={"tool_id": [llm_tool_name("local:fixture", "fixture")]},
    )
    nested = parse_handlers([{k: v for k, v in nested.items() if v is not None}])[0]
    leaf = command(required=True).model_copy(
        update={"match": {"tool_id": (llm_tool_name("local:second", "fixture"),)}}
    )
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (mcp_handler(), nested, leaf), lambda *_: True
    )
    entries = []
    original = engine.mcp_executor._invoke_worker_bound
    start = engine.processes.start

    def observed(job):
        owner = current_execution_owner()
        entries.append(
            (
                job.handler.id,
                owner.execution_id,
                case.runtime.execution_capacity.snapshot().tool_workers,
            )
        )
        return original(job)

    def observed_command(handler, *args, **kwargs):
        owner = current_execution_owner()
        entries.append(
            (
                handler.id,
                owner.execution_id,
                case.runtime.execution_capacity.snapshot().tool_workers,
            )
        )
        return start(handler, *args, **kwargs)

    monkeypatch.setattr(engine.mcp_executor, "_invoke_worker_bound", observed)
    monkeypatch.setattr(engine.processes, "start", observed_command)
    try:
        result = await case.rig.controller.submit_draft(
            "hello", session_id=case.rig.session.id
        )
        assert result.accepted, result
        await until(lambda: not engine.cleanup_pending)
        assert [value[0] for value in entries] == ["mcp", "nested", "command"]
        assert len({value[1] for value in entries}) == 1
        assert [value[2] for value in entries] == [3, 3, 3]
        assert len(case.service.execution_log.read_recent()) == 2
        assert not opaque_busy(case.runtime)
    finally:
        await case.client.disconnect_from_server("second")


async def test_source_loss_in_original_mcp_worker_never_launches_nested_legacy_fallback(
    mcp_case, tmp_path, monkeypatch
):
    case = mcp_case
    marker = tmp_path / "unowned-command"
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id,
        (
            mcp_handler(),
            command(
                f"from pathlib import Path;Path({str(marker)!r}).touch()", required=True
            ),
        ),
        lambda *_: True,
    )
    entered, release = threading.Event(), threading.Event()
    original = engine.mcp_executor._invoke_worker_bound
    source = engine._native_commands

    def held(job):
        entered.set()
        assert release.wait(8)
        return original(job)

    monkeypatch.setattr(engine.mcp_executor, "_invoke_worker_bound", held)
    pending = asyncio.create_task(
        case.rig.controller.submit_draft("hello", session_id=case.rig.session.id)
    )
    case.tasks.append(pending)
    try:
        await until(entered.is_set)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 2
        engine._native_commands = None
        release.set()
        result = await pending
        assert not result.accepted
        assert not marker.exists() and not case.service.execution_log.read_recent()
    finally:
        release.set()
        engine._native_commands = source
    await until(lambda: not engine.cleanup_pending)
    assert not opaque_busy(case.runtime)


async def test_owned_provider_source_replacement_during_reserve_refuses_before_worker(
    owned_mcp_case, monkeypatch, tmp_path
):
    from dataclasses import replace

    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner

    case = owned_mcp_case
    release, started = tmp_path / "source-release", tmp_path / "source-started"
    release.touch()
    engine, event = await bind_owned_event(case, release=release, started=started)
    provider = case.owned.providers["a"]
    snapshot = provider.snapshot
    reserved, proceed = threading.Event(), threading.Event()
    original_reserve = ExecutionOwner.reserve_tool
    launches = []
    original_start = engine.mcp_executor.start

    def reserve(owner):
        assert not engine._lock._is_owned()
        operation = original_reserve(owner)
        reserved.set()
        assert proceed.wait(8)
        return operation

    def start(*args, **kwargs):
        launches.append(True)
        return original_start(*args, **kwargs)

    monkeypatch.setattr(ExecutionOwner, "reserve_tool", reserve)
    monkeypatch.setattr(engine.mcp_executor, "start", start)
    pending = asyncio.create_task(engine.fire_async(event))
    case.tasks.append(pending)
    try:
        await until(reserved.is_set)
        provider.snapshot = replace(snapshot)
        proceed.set()
        result = await pending
        assert result.failures and not launches
        assert not started.exists()
    finally:
        proceed.set()
        provider.snapshot = snapshot
    await until(lambda: not engine.cleanup_pending)
    assert not opaque_busy(case.runtime)


async def test_actual_sequential_result_validator_keeps_original_single_operation(
    mcp_case, monkeypatch
):
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner

    case = mcp_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (mcp_handler(),), lambda *_: True
    )
    entered, release = threading.Event(), threading.Event()
    workers = []
    invoke = engine.mcp_executor._invoke_worker_bound

    # Observe inside the actual original worker guard, preserving its producer
    # callback identity so never-issued conditional branches still retire.
    def observed_invoke(job):
        workers.append(
            (current_execution_owner().execution_id, job.native_target.operation)
        )
        result = invoke(job)
        check = job.result_check

        def checked():
            entered.set()
            assert release.wait(8)
            workers.append(
                (current_execution_owner().execution_id, job.native_target.operation)
            )
            return check()

        job.result_check = checked
        return result

    monkeypatch.setattr(engine.mcp_executor, "_invoke_worker_bound", observed_invoke)
    pending = asyncio.create_task(
        case.rig.controller.submit_draft("hello", session_id=case.rig.session.id)
    )
    case.tasks.append(pending)
    try:
        await until(entered.is_set)
        assert opaque_busy(case.runtime)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 1
        assert len(case.service.execution_log.read_recent()) == 1
    finally:
        release.set()
    assert (await pending).accepted
    await until(lambda: not engine.cleanup_pending)
    assert len(workers) == 2 and workers[0] == workers[1]
    assert not opaque_busy(case.runtime)


async def test_revocation_during_actual_nested_capture_discards_without_host_leak(
    mcp_case, monkeypatch
):
    from tldw_chatbook.Agents.hooks_v2.native_commands import _NestedEvent

    case = mcp_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (mcp_handler(), command(required=True)), lambda *_: True
    )
    original = _NestedEvent.check_memory
    revision = case.rig.session.conversation_binding_revision
    captured = []

    def revoke(source, event=None):
        captured.append(source)
        case.rig.session.conversation_binding_revision += 1
        return original(source, event)

    monkeypatch.setattr(_NestedEvent, "check_memory", revoke)
    try:
        result = await case.rig.controller.submit_draft(
            "hello", session_id=case.rig.session.id
        )
        assert not result.accepted and captured
        assert not case.service.execution_log.read_recent()
        await until(lambda: not engine.cleanup_pending)
        assert not opaque_busy(case.runtime)
        assert case.runtime.execution_capacity.snapshot().tool_workers == 0
    finally:
        case.rig.session.conversation_binding_revision = revision


async def test_unqualified_ordinary_mcp_keeps_real_existing_dispatch(
    mcp_case, monkeypatch
):
    from tldw_chatbook.Utils import platform_files

    case = mcp_case
    monkeypatch.setattr(
        platform_files, "supports_native_process_groups", lambda: False
    )
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (mcp_handler(),), lambda *_: True
    )
    result = await case.rig.controller.submit_draft(
        "hello", session_id=case.rig.session.id
    )
    assert result.accepted and len(case.service.execution_log.read_recent()) == 1
    assert engine._native_commands is None
    assert case.runtime._execution_capacity is None
    assert case.runtime._worktree_activity is None


async def test_selected_deny_never_reserves_a_native_mcp_worker(mcp_case, monkeypatch):
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner

    case = mcp_case
    case.service.set_tool_state("local:fixture", "fixture", "deny", tool=case.tool)
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (mcp_handler(),), lambda *_: True
    )
    reserved = []
    original = ExecutionOwner.reserve_tool

    def observe(owner):
        reserved.append(owner)
        return original(owner)

    monkeypatch.setattr(ExecutionOwner, "reserve_tool", observe)
    result = await case.rig.controller.submit_draft(
        "hello", session_id=case.rig.session.id
    )
    assert not result.accepted and not reserved
    counter = await case.client.call_tool_result(
        "fixture", "fixture", {"counter": True}
    )
    assert counter.content[0]["text"] == "1"
    await until(lambda: not engine.cleanup_pending)
    assert not opaque_busy(case.runtime)


@pytest.mark.parametrize("change", ["none", "negative_drain", "provider", "schema"])
async def test_accepted_context_uses_positive_original_drain_without_reviving_dispatch(
    mcp_case, monkeypatch, change
):
    import json

    from tldw_chatbook.Agents.execution_capacity import OwnedOperation

    case = mcp_case
    value = mcp_handler().model_dump()
    value.update(effects=["context"], require_context=True)
    value["input"]["payload"]["structuredContent"]["context"] = [
        {"text": "Original runtime context", "lifetime": "runtime"}
    ]
    handler = parse_handlers([{k: v for k, v in value.items() if v is not None}])[0]
    nested = command(
        "print("
        + repr(
            json.dumps(
                {
                    "version": 2,
                    "decision": "pass",
                    "context": [{"text": "Original turn context", "lifetime": "turn"}],
                }
            )
        )
        + ")",
        effects=["context"],
        required=True,
    )
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (handler, nested), lambda *_: True
    )
    captures = []
    begin = engine.begin_event
    finish = OwnedOperation.finish

    def capture(event, *args, **kwargs):
        scope = begin(event, *args, **kwargs)
        if event.event == "SessionStart":
            captures.append(scope)
        return scope

    def finish_with_negative_proof(operation):
        if (
            change == "negative_drain"
            and captures
            and captures[0]._state.native.custody is not None
            and operation.owner is captures[0]._state.native.custody.owner
        ):
            operation.owner.mark_cleanup_unproven()
        return finish(operation)

    monkeypatch.setattr(engine, "begin_event", capture)
    monkeypatch.setattr(OwnedOperation, "finish", finish_with_negative_proof)
    assert (
        await case.rig.controller.submit_draft("first", session_id=case.rig.session.id)
    ).accepted
    source = captures[0]._state.native
    await until(lambda: source.custody.retired)
    assert source.custody.owner._drain_outcome is (change != "negative_drain")
    # This is the real registration retirement API, not a replacement context.
    engine.mcp_executor.retire_context("session:" + case.rig.session.id)
    selected = source.plans["mcp"].mcp
    provider, entry = selected.provider, selected.entry
    if change == "provider":
        provider._entry_by_llm_name[selected.name] = (entry[0], entry[1])
    if change == "schema":
        entry[0].input_schema["description"] = "changed after accepted result"
    if change == "none":
        source.check_memory()
    else:
        with pytest.raises(PermissionError):
            source.check_memory()
    replay = await engine.fire_async(source.event, _execution=captures[0])
    assert replay.failures and not replay.accepted
    with pytest.raises((RuntimeError, PermissionError)):
        source.admit()
    again = await case.rig.controller.submit_draft(
        "second", session_id=case.rig.session.id
    )
    assert again.accepted is (change == "none")
    assert len(case.service.execution_log.read_recent()) == 1
    assert opaque_busy(case.runtime) is (change == "negative_drain")


@pytest.mark.parametrize("ordering", ["done_first", "reaper_first"])
async def test_original_reaper_and_done_callback_return_the_same_hold_once(
    mcp_case, monkeypatch, ordering
):
    case = mcp_case
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (mcp_handler(),), lambda *_: True
    )
    entered, release = asyncio.Event(), asyncio.Event()
    execute = case.service.execute_hub_tool_result

    async def held(*args, **kwargs):
        result = await execute(*args, **kwargs)
        entered.set()
        while not release.is_set():
            try:
                await asyncio.shield(release.wait())
            except asyncio.CancelledError:
                pass
        return result

    monkeypatch.setattr(case.service, "execute_hub_tool_result", held)
    loop = asyncio.get_running_loop()
    schedule = loop.call_soon
    deferred = []

    def schedule_done(callback, *args, context=None):
        if (
            getattr(callback, "__qualname__", "")
            == "MCPHookExecutor.start.<locals>.returned"
        ):
            deferred.append((callback, args, context))
            return schedule(lambda: None, context=context)
        return schedule(callback, *args, context=context)

    monkeypatch.setattr(loop, "call_soon", schedule_done)

    def return_done():
        callback, args, context = deferred.pop()
        schedule(callback, *args, context=context)

    pending = asyncio.create_task(
        case.rig.controller.submit_draft("hello", session_id=case.rig.session.id)
    )
    case.tasks.append(pending)
    try:
        await until(entered.is_set)
        job = next(iter(engine.mcp_executor._jobs))
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await until(lambda: deferred and job.pending_release and job.task.done())
        assert job.native_target._callbacks == 1
        assert not job.bridges[0].completed.is_set()
        if ordering == "done_first":
            return_done()
            await asyncio.sleep(0)
            assert job.native_target._callbacks == 1 and opaque_busy(case.runtime)
        release.set()
        await until(lambda: not engine.cleanup_pending)
        assert job.bridges[0].completed.is_set()
        assert job.native_target._callbacks == 0
        if ordering == "reaper_first":
            return_done()
            await asyncio.sleep(0)
        assert job.native_target._callbacks == 0
        assert not opaque_busy(case.runtime)
        await case.runtime.close_hooks_v2()
        assert not engine.cleanup_pending
    finally:
        release.set()
        if deferred:
            return_done()


@pytest.mark.parametrize(
    "revocation",
    [
        "hook",
        "mcp",
        "same_path",
        "ask",
        "kill",
        "unrelated",
        "race",
        "disk",
        "pending",
        "session",
        "argument",
    ],
)
async def test_passive_accepted_mcp_context_still_refuses_actual_saved_grant_revocation(
    mcp_case, hook_file, monkeypatch, revocation
):
    from Tests.Agents.test_hook_permissions import _approve
    from Tests.Chat.test_console_hook_preparation_demand import _save_hooks
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookExecutor

    case = mcp_case
    value = mcp_handler().model_dump()
    value.update(effects=["context"], require_context=True)
    value["input"]["payload"]["structuredContent"]["context"] = [
        {"text": "Reviewed runtime context", "lifetime": "runtime"}
    ]
    if revocation in {"session", "argument"}:
        case.service.set_tool_state("local:fixture", "fixture", "ask")
        if revocation == "session":
            case.service.approve_for_session("local:fixture", "fixture")
        else:
            case.service.add_tool_arg_rule(
                "local:fixture", "fixture", args=value["input"], tool=case.tool
            )
    _save_hooks(
        hook_file,
        {
            "enabled": True,
            "handler": [
                {k: v for k, v in value.items() if v is not None and v not in ({}, ())}
            ],
        },
    )
    permissions = case.runtime.ensure_hook_permissions()
    case.rig.controller._hook_permissions_accessor = lambda: permissions
    assert _approve(permissions).ready
    sources = []
    start = MCPHookExecutor.start

    def observed(self, *args, **kwargs):
        job = start(self, *args, **kwargs)
        sources.append(job.native_target.event.source)
        return job

    monkeypatch.setattr(MCPHookExecutor, "start", observed)
    assert (
        await case.rig.controller.submit_draft("first", session_id=case.rig.session.id)
    ).accepted
    source = sources[0]
    await until(lambda: source.custody.retired)
    assert source.custody.owner._drain_outcome is True
    source.issuer.engine.mcp_executor.retire_context("session:" + case.rig.session.id)
    source.check_memory()
    from tldw_chatbook.Agents.mcp_tool_provider import MCPResultPermission

    read = MCPResultPermission.check
    reads = []

    def observed_read(grant):
        assert not source.lifecycle.checkpoints._condition._is_owned()
        reads.append(grant)
        read(grant)
        if revocation == "race":
            # Revoke after the actual provider read but before pure publication.
            case.service.set_tool_state("local:fixture", "fixture", "deny")

    monkeypatch.setattr(MCPResultPermission, "check", observed_read)
    if revocation == "hook":
        review = permissions.snapshot()
        permissions.revoke(review, review.rows[0].entry.key)
    elif revocation == "same_path":
        from tldw_chatbook.MCP.permission_store import MCPPermissionStore

        other = MCPPermissionStore(case.service.permission_store.path)
        assert other is not case.service.permission_store
        other.set_tool_state("local:fixture", "fixture", "deny")
    elif revocation == "kill":
        case.service.set_kill_switch(True)
    elif revocation == "session":
        assert source.plans["mcp"].permission.grant.decision == "approved-session"
        assert case.service.revoke_session_approval("local:fixture", "fixture")
    elif revocation == "argument":
        assert source.plans["mcp"].permission.grant.decision == "allowed"
        rule = case.service.list_tool_arg_rules("local:fixture", "fixture")[0]
        assert case.service.remove_tool_arg_rule(
            "local:fixture", "fixture", rule["rule_id"]
        )
    elif revocation == "pending":
        from tldw_chatbook.MCP.permission_store import MCPPermissionStore

        store = MCPPermissionStore(case.service.permission_store.path)
        publication = store._permission_publication
        complete = type(publication).complete
        entered, release = threading.Event(), threading.Event()
        errors = []

        def held_close(owner):
            if owner is publication:
                entered.set()
                assert release.wait(8)
            complete(owner)

        def revoke():
            try:
                store.set_tool_state("local:fixture", "fixture", "deny")
            except BaseException as error:  # noqa: BLE001 -- asserted after writer joins
                errors.append(error)

        monkeypatch.setattr(type(publication), "complete", held_close)
        writer = threading.Thread(target=revoke)
        writer.start()
        try:
            await until(entered.is_set)
            assert publication.pending and publication.token is None
            with pytest.raises(PermissionError):
                source.check_memory()
        finally:
            release.set()
            await asyncio.to_thread(writer.join, 8)
        assert not writer.is_alive() and not errors
        assert not publication.pending and publication.token is None
    elif revocation == "unrelated":
        case.service.set_tool_state("local:fixture", "other_tool", "deny")
    elif revocation == "disk":
        import json

        path = case.service.permission_store.path
        payload = json.loads(path.read_text())
        payload["profiles"]["default"]["servers"]["local:fixture"]["tools"]["fixture"][
            "state"
        ] = "deny"
        path.write_text(json.dumps(payload))
    elif revocation == "race":
        pass
    else:
        case.service.set_tool_state(
            "local:fixture", "fixture", "ask" if revocation == "ask" else "deny"
        )
    if revocation in {"mcp", "same_path"}:
        assert case.service.gate_tool_test(case.tool).state == "deny"
    if revocation not in {"unrelated", "race", "disk"}:
        with pytest.raises(PermissionError):
            source.check_memory()
    again = await case.rig.controller.submit_draft(
        "second", session_id=case.rig.session.id
    )
    assert again.accepted is (revocation == "unrelated")
    if revocation in {"unrelated", "race", "disk"}:
        assert reads
    assert len(case.service.execution_log.read_recent()) == 1
    assert not opaque_busy(case.runtime)


@pytest.fixture
async def native_owned_result_case(native_package, tmp_path, monkeypatch, request):
    from contextlib import aclosing

    import Tests.Plugins.test_owned_mcp_tools as existing

    # The Console fixture already owns tmp_path/profile; this actual plugin
    # service needs its own fixture directory, as owned_mcp_case does above.
    directory = tmp_path / "owned-result"
    directory.mkdir()
    response = '{"version":2,"decision":"pass","context":[{"text":"owned instructions","lifetime":"turn"}]}'
    monkeypatch.setattr(
        existing,
        "SERVER",
        existing.SERVER.replace(
            'result = {"content": [{"type": "text", "text": "ok"}]}',
            f'result = {{"content": [{{"type": "text", "text": {response!r}}}]}}',
        ),
    )
    async with aclosing(
        existing.shared_connection_case.__wrapped__(
            native_package, directory, monkeypatch, request
        )
    ) as cases:
        async for case in cases:
            yield case


async def test_native_owned_result_rechecks_original_grant_after_positive_retirement(
    event_case, native_owned_result_case
):
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookContext

    case, owned = event_case, native_owned_result_case
    owned.release.touch()
    assert (await owned.pending).ok
    handler = parse_handlers(
        [
            {
                "id": "owned_context",
                "event": "PreCompact",
                "type": "mcp_tool",
                "server": "local:" + owned.profile_id,
                "tool": "echo",
                "required": True,
                "require_context": True,
                "effects": ["context"],
            }
        ]
    )[0]
    engine = case.runtime.ensure_hooks_v2(
        case.rig.session.id, (handler,), lambda *_: True
    )
    lifecycle = await case.runtime.prepare_hooks_v2(case.rig.session.id)
    provider = owned.providers["a"]
    engine.mcp_executor.bind_context(
        MCPHookContext(
            owned.registries["a"],
            lifecycle,
            lifecycle.scope_id,
            "run-a",
            case.rig.session.id,
            frozenset(row.name for row in provider.list_catalog()),
            lifecycle.current,
            lambda _definition: (),
            workspace_id="a",
        )
    )
    before = len(owned.calls.read_text().splitlines())
    owner = lifecycle.open_scope()
    try:
        event = lifecycle.event(
            "PreCompact", data={"reason": "manual", "candidate": []}
        )
        result = await lifecycle.fire(event, owner)
        assert result.succeeded and result.accepted
        source = result._native_sources[0]
        await until(lambda: source.custody.retired)
        assert source.custody.owner._drain_outcome is True
        engine.mcp_executor.retire_context("session:" + case.rig.session.id)
        lifecycle.context.refresh_sources(owner)
        assert lifecycle.context.blocks(owner, "positive")
        tool = source.plans["owned_context"].mcp.entry[0]
        owned.unified.set_tool_state(tool.server_key, tool.name, "deny", tool=tool)
        with pytest.raises(PermissionError):
            source.check_memory()
        with pytest.raises(PermissionError):
            lifecycle.context.refresh_sources(owner)
        with pytest.raises(ValueError):
            lifecycle.context.blocks(owner, "revoked")
        assert len(owned.calls.read_text().splitlines()) == before + 1
        assert not opaque_busy(case.runtime)
        assert owned.service._thread.is_alive()
    finally:
        lifecycle.close_scope(owner)
        await case.runtime.close_hooks_v2()


@pytest.mark.parametrize("mutation", ["unrelated", "mcp_deny", "disk_deny"])
async def test_actual_child_receiver_refreshes_original_passive_mcp_grant(
    mcp_case, hook_file, monkeypatch, tmp_path, mutation
):
    import json

    from Tests.Agents.test_agent_service import provider_reply
    from Tests.Agents.test_fleet_runtime import FLEET_CFG, fence, make_inline_service
    from Tests.Agents.test_hook_permissions import _approve
    from Tests.Chat.test_console_hook_preparation_demand import _save_hooks
    from tldw_chatbook.Agents.activation import worker_guard
    from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookExecutor
    from tldw_chatbook.Agents.mcp_tool_provider import MCPResultPermission
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    case = mcp_case
    value = mcp_handler().model_dump()
    value.update(effects=["context"], require_context=True)
    value["input"]["payload"]["structuredContent"]["context"] = [
        {"text": "Original runtime child instructions", "lifetime": "runtime"}
    ]
    _save_hooks(
        hook_file,
        {
            "enabled": True,
            "handler": [
                {k: v for k, v in value.items() if v is not None and v not in ({}, ())}
            ],
        },
    )
    permissions = case.runtime.ensure_hook_permissions()
    case.rig.controller._hook_permissions_accessor = lambda: permissions
    assert _approve(permissions).ready
    sources = []
    start = MCPHookExecutor.start

    def observed(self, *args, **kwargs):
        job = start(self, *args, **kwargs)
        sources.append(job.native_target.event.source)
        return job

    monkeypatch.setattr(MCPHookExecutor, "start", observed)
    db = AgentRunsDB(":memory:", client_id="native-child-receiver")
    service, _chat = make_inline_service(db, [], monkeypatch)
    calls, changed, received = [], [], []

    def model(**kwargs):
        calls.append(kwargs)
        if not changed:
            assert "Original runtime child instructions" in str(
                kwargs["messages_payload"]
            )
            changed.append(True)
            # The parent input has already been prepared. This real policy edit
            # happens during its model call, before actual child initialization.
            if mutation == "disk_deny":
                path = case.service.permission_store.path
                payload = json.loads(path.read_text())
                payload["profiles"]["default"]["servers"]["local:fixture"]["tools"][
                    "fixture"
                ]["state"] = "deny"
                path.write_text(json.dumps(payload))
            else:
                case.service.set_tool_state(
                    "local:fixture",
                    "other" if mutation == "unrelated" else "fixture",
                    "deny",
                )
            return provider_reply(fence(SPAWN_TOOL_NAME, {"task": "child"}))
        return provider_reply("parent complete")

    service.chat_call = model
    try:
        assert (
            await case.rig.controller.submit_draft(
                "first", session_id=case.rig.session.id
            )
        ).accepted
        source = sources[0]
        await until(lambda: source.custody.retired)
        assert source.custody.owner._drain_outcome is True
        engine, lifecycle = source.issuer.engine, source.lifecycle
        engine.mcp_executor.retire_context("session:" + case.rig.session.id)
        source.check_memory()
        service._hooks_v2_engine = engine
        service._hooks_v2_lifecycle = lifecycle
        service._hooks_v2_session_id = case.rig.session.id
        service._hooks_v2_turn_id = "actual-child-turn"
        check = MCPResultPermission.check

        def checked(grant):
            assert not lifecycle.checkpoints._condition._is_owned()
            return check(grant)

        monkeypatch.setattr(MCPResultPermission, "check", checked)
        blocks = lifecycle.context.blocks

        def observed_blocks(owner, checkpoint):
            rows = blocks(owner, checkpoint)
            if checkpoint == "child":
                received.extend(rows)
            return rows

        monkeypatch.setattr(lifecycle.context, "blocks", observed_blocks)

        def run_parent():
            # SQLite :memory: is thread-local. Initialize and inspect the actual
            # worker database in that same thread; no cross-thread row stand-in.
            db._initialize_schema()
            try:
                run, outcome = service.run_turn(
                    conversation_id="child-conversation",
                    messages=[{"role": "user", "content": "spawn child"}],
                    config=FLEET_CFG,
                    api_endpoint="llama_cpp",
                )
                return (
                    run,
                    outcome,
                    db.count_subagent_runs("child-conversation"),
                    [
                        db.get_run(row["id"])
                        for row in db.list_runs("child-conversation")
                    ],
                )
            finally:
                db.close()

        _run, result, child_count, actual_rows = await asyncio.to_thread(
            worker_guard(service)(run_parent)
        )
        assert changed and calls
        assert child_count == (1 if mutation == "unrelated" else 0)
        if mutation == "unrelated":
            assert "Original runtime child instructions" in str(received)
            assert result.final_text == "parent complete"
            # This exercises the real child receiving boundary, not a complete
            # child model journey. Its later guarded entry adds the parent's
            # now-bound run-log source and preserves the original-source refusal.
            child = next(row for row in actual_rows if row["agent_kind"] == "subagent")
            assert child["status"] == "error"
            assert any(
                step["kind"] == "error"
                and "Command event activation source changed" in step["summary"]
                for step in child["steps"]
            )
        else:
            assert not received
            # The next parent model input also refuses the revoked runtime
            # context, so its final error replaces the earlier spawn result.
            assert result.status == "error"
        assert len(case.service.execution_log.read_recent()) == 1
        assert not opaque_busy(case.runtime)
    finally:
        db.close()
