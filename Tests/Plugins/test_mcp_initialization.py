"""Prospective hook MCP authority uses existing scoped connections and custody."""

import asyncio
import contextlib
import threading
import time

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.Plugins.test_owned_mcp_tools import (
    shared_connection_case as _owned_connection_fixture,
)

shared_connection_case = _owned_connection_fixture


def restriction(**changes):
    from tldw_chatbook.Agents.mcp_tool_provider import MCPInvocationPolicy

    return MCPInvocationPolicy(
        current=changes.pop("current", lambda: True),
        deadline=time.monotonic() + 10,
        cancel_event=threading.Event(),
        allow_approval=True,
        wait_scope=lambda _kind: contextlib.nullcontext(),
        **changes,
    )


@pytest.mark.asyncio
async def test_owned_prospective_call_cannot_attach_using_siblings_connection(
    shared_connection_case,
):
    from tldw_chatbook.Agents.mcp_tool_provider import restrict_mcp_invocation
    from tldw_chatbook.Plugins.mcp_provider import PluginMCPProvider

    case = shared_connection_case
    case.release.touch()
    assert (await case.pending).ok
    snapshot = await case.service.capture_mcp_snapshot(
        case.installation, "a", "provisional"
    )
    provider = PluginMCPProvider(
        plugin_service=case.service,
        ownership=case.ownership,
        snapshot=snapshot,
        service=case.unified,
        main_loop=asyncio.get_running_loop(),
        approval_callback=lambda calls: {
            call.llm_name: "approve_once" for call in calls
        },
    )
    await provider.compose_catalog()
    name = provider.list_catalog()[0].name
    assert case.ownership.is_connected(case.profile_id)
    assert not case.ownership.is_connected(case.profile_id, "provisional:a")
    before = len(case.calls.read_text().splitlines())
    before_tokens = {record.lease_token for record in case.service.live_runs()}

    def invoke():
        with restrict_mcp_invocation(restriction()):
            return provider.invoke(name, {})

    denied = await asyncio.to_thread(invoke)
    assert not denied.ok and denied.dispatch_state == "not_started"
    assert len(case.calls.read_text().splitlines()) == before
    assert {record.lease_token for record in case.service.live_runs()} == before_tokens
    assert not case.ownership.is_connected(case.profile_id, "provisional:a")
    assert (await case.invoke("b")).ok

    # Only the explicit normal setup path can attach this prospective scope.
    await provider.connect("mcp:shared")
    assert case.ownership.is_connected(case.profile_id, "provisional:a")
    assert (await asyncio.to_thread(invoke)).ok


@pytest.mark.asyncio
async def test_restricted_owned_request_exposes_exact_existing_terminal_custody(
    shared_connection_case,
):
    from tldw_chatbook.Agents.mcp_tool_provider import (
        capture_mcp_result,
        restrict_mcp_invocation,
    )

    case = shared_connection_case
    case.release.touch()
    assert (await case.pending).ok
    provider = case.providers["a"]
    name = provider.list_catalog()[0].name
    records = []

    def request_owned(plugins, token):
        matching = [
            record for record in plugins.live_runs() if record.lease_token == token
        ]
        assert len(matching) == 1 and not matching[0].completed.is_set()
        records.extend(matching)

    def invoke():
        with (
            restrict_mcp_invocation(restriction(on_owned_request=request_owned)),
            capture_mcp_result(provider) as capture,
        ):
            result = case.registries["a"].invoke_by_name(name, {})
            raw = capture.consume(result)
            assert result.ok and raw.duplicate_keys_checked
            assert raw.content[0]["text"] == "ok"
            return result

    assert (await asyncio.to_thread(invoke)).dispatch_state == "settled"
    assert len(records) == 1
    assert records[0].completed.is_set()
    assert records[0].lease_token not in {
        row.lease_token for row in case.service.live_runs()
    }
    assert case.ownership.is_connected(case.profile_id, "run-b:b")


@pytest.mark.asyncio
async def test_cancelled_bridge_retains_exact_request_until_real_late_response(
    shared_connection_case, tmp_path
):
    from tldw_chatbook.Agents.mcp_tool_provider import restrict_mcp_invocation

    case = shared_connection_case
    case.release.touch()
    assert (await case.pending).ok
    provider = case.providers["a"]
    name = provider.list_catalog()[0].name
    records, bridges = [], []
    released, started = tmp_path / "late-release", tmp_path / "late-started"

    def request_owned(plugins, token):
        records.extend(
            record for record in plugins.live_runs() if record.lease_token == token
        )

    policy = restriction(on_owned_request=request_owned, on_bridge=bridges.append)

    def invoke():
        with restrict_mcp_invocation(policy):
            return case.registries["a"].invoke_by_name(
                name, {"hold": str(released), "started": str(started)}
            )

    pending = asyncio.create_task(asyncio.to_thread(invoke))
    try:
        for _ in range(1000):
            if started.exists():
                break
            await asyncio.sleep(0.01)
        assert started.exists() and len(records) == len(bridges) == 1
        assert not bridges[0].completed.is_set()
        policy.cancel_event.set()
        bridges[0].future.cancel()
        result = await asyncio.wait_for(pending, 2)
        assert not result.ok and result.dispatch_state == "uncertain"
        for _ in range(100):
            if bridges[0].completed.is_set():
                break
            await asyncio.sleep(0.01)
        assert bridges[0].completed.is_set()
        assert not records[0].completed.is_set()
        assert records[0].lease_token in {
            record.lease_token for record in case.service.live_runs()
        }
        assert (await case.invoke("b")).ok
        released.touch()
        for _ in range(300):
            if records[0].completed.is_set():
                break
            await asyncio.sleep(0.01)
        assert records[0].completed.is_set()
        assert bridges[0].dispatch.state == "settled"
    finally:
        released.touch()
        await asyncio.wait_for(pending, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cancel", "revoke"])
async def test_executor_retains_exact_owned_request_ticket_until_late_completion(
    shared_connection_case, tmp_path, action
):
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookContext
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    case = shared_connection_case
    case.release.touch()
    assert (await case.pending).ok
    release, started = tmp_path / "hook-release", tmp_path / "hook-started"
    provider = case.providers["a"]
    handler = parse_handlers(
        [
            {
                "id": "owned-hook",
                "event": "SessionStart",
                "type": "mcp_tool",
                "server": "local:" + case.profile_id,
                "tool": "echo",
                "required": True,
                "effects": [],
                "timeout_seconds": 5,
                "input": {"hold": str(release), "started": str(started)},
            }
        ]
    )[0]
    assert provider.hook_target(handler.server, handler.tool)
    owner = HookBudgetOwner()
    engine = HookEngine((handler,), lambda *_: True, owner)
    lifecycle = HookSessionLifecycle(engine, "hook-session")
    engine.mcp_executor.bind_context(
        MCPHookContext(
            case.registries["a"],
            lifecycle,
            lifecycle.scope_id,
            "run-a",
            "hook-session",
            frozenset(row.name for row in provider.list_catalog()),
            lambda: True,
            lambda _definition: (),
            workspace_id="a",
        )
    )
    existing = {row.lease_token for row in case.service.live_runs()}
    before = len(case.calls.read_text().splitlines())
    pending = asyncio.create_task(
        engine.fire_async(lifecycle.event("SessionStart", data={"reason": "startup"}))
    )
    try:
        for _ in range(1000):
            if started.exists():
                break
            await asyncio.sleep(0.01)
        assert started.exists(), "hook never reached the controlled peer"
        records = [
            row for row in case.service.live_runs() if row.lease_token not in existing
        ]
        assert len(records) == 1 and not records[0].completed.is_set()
        assert owner.snapshot()["tickets"] == owner.snapshot()["execution"] == 1
        if action == "cancel":
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            await case.disable_a()
        for _ in range(1000):
            if owner.snapshot()["execution"] == 0:
                break
            await asyncio.sleep(0.01)
        assert owner.snapshot()["execution"] == 0
        assert owner.snapshot()["tickets"] == 1 and engine.cleanup_pending
        assert not records[0].completed.is_set()
        assert (await case.invoke("b")).ok
        # Only the peer's actual completion can retire A's exact request custody.
        release.touch()
        for _ in range(300):
            if records[0].completed.is_set() and owner.snapshot()["tickets"] == 0:
                break
            await asyncio.sleep(0.01)
        assert records[0].completed.is_set()
        assert owner.snapshot()["tickets"] == owner.snapshot()["execution"] == 0
        assert not engine.cleanup_pending
        assert len(case.calls.read_text().splitlines()) == before + 2
        if action == "revoke":
            result = await asyncio.wait_for(pending, 2)
            assert not result.accepted and not result.allowed
    finally:
        release.touch()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await engine.close()


@pytest.mark.asyncio
async def test_executor_refuses_actual_changed_owned_mapping_before_dispatch(
    shared_connection_case,
):
    import json

    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookContext
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    case = shared_connection_case
    case.release.touch()
    assert (await case.pending).ok
    original = case.local.store.get_discovery_snapshot(case.profile_id)
    changed = json.loads(json.dumps(original))
    changed["tools"][0]["description"] = "Changed exact mapping"
    handler = parse_handlers(
        [
            {
                "id": "stale",
                "event": "SessionStart",
                "type": "mcp_tool",
                "server": "local:" + case.profile_id,
                "tool": "echo",
                "required": True,
                "effects": [],
            }
        ]
    )[0]
    owner = HookBudgetOwner()
    engine = HookEngine((handler,), lambda *_: True, owner)
    lifecycle = HookSessionLifecycle(engine, "session")
    engine.mcp_executor.bind_context(
        MCPHookContext(
            case.registries["a"],
            lifecycle,
            lifecycle.scope_id,
            "run-a",
            "session",
            frozenset(row.name for row in case.providers["a"].list_catalog()),
            lambda: True,
            lambda _definition: (),
            workspace_id="a",
        )
    )
    before_calls = case.calls.read_text()
    before_tokens = {record.lease_token for record in case.service.live_runs()}
    case.local.store.save_discovery_snapshot(case.profile_id, changed)
    try:
        result = await asyncio.wait_for(
            engine.fire_async(
                lifecycle.event("SessionStart", data={"reason": "startup"})
            ),
            3,
        )
        assert not result.allowed and not result.accepted
        assert case.calls.read_text() == before_calls
        assert {
            record.lease_token for record in case.service.live_runs()
        } == before_tokens
        assert owner.snapshot()["tickets"] == owner.snapshot()["execution"] == 0
    finally:
        case.local.store.save_discovery_snapshot(case.profile_id, original)
        await engine.close()
    assert (await case.invoke("b")).ok


@pytest.fixture
async def owned_result_case(native_package, tmp_path, monkeypatch, request):
    import Tests.Plugins.test_owned_mcp_tools as existing

    # Preserve the actual owned fixture's transport/setup/custody; only the
    # controlled peer's response body changes to a qualified native result.
    response = '{"version":2,"decision":"pass","context":[{"text":"owned instructions","lifetime":"runtime"}]}'
    monkeypatch.setattr(
        existing,
        "SERVER",
        existing.SERVER.replace(
            'result = {"content": [{"type": "text", "text": "ok"}]}',
            f'result = {{"content": [{{"type": "text", "text": {response!r}}}]}}',
        ),
    )
    async with contextlib.aclosing(
        _owned_connection_fixture.__wrapped__(
            native_package, tmp_path, monkeypatch, request
        )
    ) as cases:
        async for case in cases:
            yield case


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ["none", "permission", "disable", "definition", "profile", "kill", "persona"],
)
async def test_owned_result_currentness_after_required_post_join(
    owned_result_case, tmp_path, change
):
    import json

    from Tests.Agents.test_hooks_v2_tool_pipeline import hook_command
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookContext
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    case = owned_result_case
    case.release.touch()
    assert (await case.pending).ok
    provider = case.providers["a"]
    name = provider.list_catalog()[0].name
    tool = provider._entry_by_llm_name[name][0]
    prompts = []
    provider._approval_callback = lambda calls: (
        prompts.extend(calls) or {call.llm_name: "approve_once" for call in calls}
    )
    marker, release = tmp_path / "post-entered", tmp_path / "post-release"
    wait = f"while not Path({str(release)!r}).exists(): time.sleep(.01)"
    post = hook_command(
        "post",
        event="PostToolUse",
        required=True,
        code=f"from pathlib import Path;import time;Path({str(marker)!r}).touch();exec({wait!r})",
    )
    handler = parse_handlers(
        [
            {
                "id": "owned",
                "event": "SessionStart",
                "type": "mcp_tool",
                "server": "local:" + case.profile_id,
                "tool": "echo",
                "required": True,
                "effects": ["context"],
                "require_context": True,
            }
        ]
    )[0]
    owner = HookBudgetOwner()
    engine = HookEngine((handler, post), lambda *_: True, owner)
    lifecycle = HookSessionLifecycle(engine, "session")
    engine.mcp_executor.bind_context(
        MCPHookContext(
            case.registries["a"],
            lifecycle,
            lifecycle.scope_id,
            "run-a",
            "session",
            frozenset([name]),
            lambda: True,
            lambda _definition: (),
            workspace_id="a",
        )
    )
    original = case.local.store.get_discovery_snapshot(case.profile_id)
    original_persona = provider._persona_policy_provider
    original_profile = provider._profile_id
    before = len(case.calls.read_text().splitlines())
    pending = asyncio.create_task(
        engine.fire_async(lifecycle.event("SessionStart", data={"reason": "startup"}))
    )
    try:
        for _ in range(300):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        assert marker.exists() and not pending.done()
        assert len(case.calls.read_text().splitlines()) == before + 1
        assert len(prompts) == 1
        if change == "permission":
            case.unified.set_tool_state(tool.server_key, tool.name, "deny", tool=tool)
        elif change == "disable":
            await case.disable_a()
        elif change == "definition":
            changed = json.loads(json.dumps(original))
            changed["tools"][0]["description"] = "changed during post wait"
            case.local.store.save_discovery_snapshot(case.profile_id, changed)
        elif change == "profile":
            provider._profile_id = lambda: "changed-profile"
        elif change == "kill":
            case.unified.set_kill_switch(True)
        elif change == "persona":

            def broken_persona():
                raise RuntimeError("configured persona unavailable")

            provider._persona_policy_provider = broken_persona
        release.touch()
        outcome = await asyncio.wait_for(pending, 5)
        assert bool(outcome.accepted) is (change == "none"), outcome
        assert outcome.allowed is (change == "none")
        assert len(prompts) == 1
        assert len(case.calls.read_text().splitlines()) == before + 1
        assert owner.snapshot()["tickets"] == owner.snapshot()["execution"] == 0
    finally:
        release.touch()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        provider._persona_policy_provider = original_persona
        provider._profile_id = original_profile
        case.local.store.save_discovery_snapshot(case.profile_id, original)
        case.unified.set_kill_switch(False)
        if change == "permission":
            case.unified.set_tool_state(tool.server_key, tool.name, "ask", tool=tool)
        await engine.close()
    assert (await case.invoke("b")).ok


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cancel", "deadline"])
async def test_result_currentness_worker_retains_same_ticket_until_actual_completion(
    owned_result_case, action
):
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookContext
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    case = owned_result_case
    case.release.touch()
    assert (await case.pending).ok
    provider = case.providers["a"]
    original = provider._result_currentness
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    checks = []

    def capture(*args):
        current = original(*args)

        def check():
            current()
            checks.append(threading.get_ident())
            if len(checks) == 2:
                entered.set()
                try:
                    assert release.wait(20)
                finally:
                    finished.set()

        return check

    provider._result_currentness = capture
    handler = parse_handlers(
        [
            {
                "id": "owned",
                "event": "SessionStart",
                "type": "mcp_tool",
                "server": "local:" + case.profile_id,
                "tool": "echo",
                "required": True,
                "effects": ["context"],
                "timeout_seconds": 5,
            }
        ]
    )[0]
    owner = HookBudgetOwner()
    engine = HookEngine((handler,), lambda *_: True, owner)
    lifecycle = HookSessionLifecycle(engine, "session")
    engine.mcp_executor.bind_context(
        MCPHookContext(
            case.registries["a"],
            lifecycle,
            lifecycle.scope_id,
            "run-a",
            "session",
            frozenset(row.name for row in provider.list_catalog()),
            lambda: True,
            lambda _definition: (),
            workspace_id="a",
        )
    )
    before = len(case.calls.read_text().splitlines())
    pending = asyncio.create_task(
        engine.fire_async(lifecycle.event("SessionStart", data={"reason": "startup"}))
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        job = next(iter(engine.mcp_executor._jobs))
        assert all(record.completed.is_set() for record in job.requests)
        assert len(checks) == 2 and all(
            thread != threading.get_ident() for thread in checks
        )
        if action == "cancel":
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            outcome = await asyncio.wait_for(asyncio.shield(pending), 12)
            assert not outcome.accepted and not outcome.allowed
        for _ in range(200):
            if owner.snapshot()["execution"] == 0:
                break
            await asyncio.sleep(0.01)
        assert owner.snapshot()["execution"] == 0
        assert owner.snapshot()["tickets"] == 1 and engine.cleanup_pending
        assert not job.worker.done() and not finished.is_set()
        assert len(checks) == 2, "deadline/cancellation queued another check"
        assert len(case.calls.read_text().splitlines()) == before + 1
        release.set()
        for _ in range(300):
            if owner.snapshot()["tickets"] == 0:
                break
            await asyncio.sleep(0.01)
        assert finished.is_set() and job.worker.done()
        assert owner.snapshot()["tickets"] == 0 and not engine.cleanup_pending
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await engine.close()
        provider._result_currentness = original
    assert (await case.invoke("b")).ok
