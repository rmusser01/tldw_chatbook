"""Lifecycle effects at real Console admission and runtime custody boundaries."""

from types import SimpleNamespace

import pytest

from Tests.Agents.test_hook_permissions import _approve, _edit
from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.Agents.test_hooks_v2_execution import command
from Tests.Chat.test_console_fleet_wake import _controller_rig
from tldw_chatbook.Agents.run_hooks import HookOutcome
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]
hook_file = _hook_file


@pytest.fixture
async def session_case(tmp_path):
    notifications = []

    class LegacyRecorder:
        async def fire_async(self, event, **kwargs):
            notifications.append((event, kwargs))
            return HookOutcome()

        def notify(self, event, **kwargs):
            notifications.append((event, kwargs))

    rig = _controller_rig(tmp_path, ensure_run_hooks=lambda: LegacyRecorder())
    chacha, app, _runs, store, session, gateway, _bridge, controller = rig
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    accepted = []
    controller.on_submission_accepted = lambda: accepted.append(session.id)
    permit = tmp_path / "permit"
    marker = tmp_path / "initialized"
    hook = command(
        "from pathlib import Path;"
        f"Path({str(marker)!r}).write_text('started');"
        f"raise SystemExit(0 if Path({str(permit)!r}).exists() else 1)",
        name="SessionStart",
        required=True,
    )
    runtime.ensure_hooks_v2(session.id, (hook,), lambda *_: True)
    case = SimpleNamespace(
        runtime=runtime,
        controller=controller,
        session=session,
        gateway=gateway,
        permit=permit,
        marker=marker,
        refuse_required_initialization=lambda: permit.unlink(missing_ok=True),
        allow_required_initialization=lambda: permit.write_text("allow"),
        submit=lambda: controller.submit_draft("hello", session_id=session.id),
        admitted_turn_count=lambda: len(accepted),
        root_stop_count=lambda: sum(event == "Stop" for event, _ in notifications),
    )
    yield case
    await runtime.close_hooks_v2()
    await runtime.dispose()
    _runs.close()
    chacha.close()


@pytest.mark.asyncio
async def test_failed_initialization_never_publishes_a_root_turn(session_case):
    case = session_case
    case.refuse_required_initialization()
    await case.submit()
    assert case.admitted_turn_count() == 0
    assert case.root_stop_count() == 0
    assert case.marker.exists(), "the controlled command did not execute"
    case.allow_required_initialization()
    await case.submit()
    assert case.admitted_turn_count() == 1
    assert case.root_stop_count() == 1


@pytest.mark.asyncio
async def test_valid_console_submission_control(session_case):
    session_case.allow_required_initialization()
    result = await session_case.submit()
    assert result.accepted
    assert session_case.admitted_turn_count() == 1
    assert session_case.root_stop_count() == 1
    assert session_case.gateway.payloads


@pytest.mark.asyncio
async def test_initialization_cancel_releases_slot_and_retains_child_cleanup(
    session_case, tmp_path
):
    import asyncio

    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    entered = tmp_path / "entered"
    release = tmp_path / "release"
    hook = command(
        "from pathlib import Path;import time;"
        f"Path({str(entered)!r}).write_text('entered');"
        f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(0.01)'!r})",
        name="SessionStart",
        required=True,
    )
    engine = case.runtime.ensure_hooks_v2(case.session.id, (hook,), lambda *_: True)
    pending = asyncio.create_task(case.submit())
    try:
        for _ in range(400):
            if entered.exists():
                break
            await asyncio.sleep(0.005)
        assert entered.exists()
        concurrent = await case.submit()
        assert not concurrent.accepted
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert case.admitted_turn_count() == 0
        assert case.root_stop_count() == 0
        assert case.controller.in_flight_run_count() == 0
        release.write_text("allow")
        result = await case.submit()
        assert result.accepted
        assert case.admitted_turn_count() == 1
        assert case.root_stop_count() == 1
    finally:
        release.write_text("release")
        if not pending.done():
            pending.cancel()
        await engine.close()


@pytest.mark.asyncio
async def test_master_off_preserves_required_session_refusal(session_case):
    case = session_case
    case.allow_required_initialization()
    case.runtime.get_hooks_v2(case.session.id).enabled = False
    result = await case.submit()
    assert not result.accepted
    assert case.admitted_turn_count() == 0
    assert case.root_stop_count() == 0
    assert not case.marker.exists()


@pytest.mark.asyncio
async def test_runtime_context_is_live_attributed_and_once_per_send(session_case):
    from tldw_chatbook.Agents.agent_models import PluginContextText

    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    hook = command(
        'print(\'{"version":2,"decision":"pass","context":[{"text":"session material","lifetime":"runtime"}]}\')',
        name="SessionStart",
        required=True,
        effects=["context"],
    )
    case.runtime.ensure_hooks_v2(case.session.id, (hook,), lambda *_: True)
    assert (await case.submit()).accepted
    assert (await case.submit()).accepted
    for payload in case.gateway.payloads:
        blocks = [
            row["content"]
            for row in payload
            if "session material" in str(row.get("content"))
        ]
        assert len(blocks) == 1
        assert isinstance(blocks[0], PluginContextText)
        assert blocks[0].checked_hook_origins()


@pytest.mark.asyncio
async def test_live_viewless_session_end_is_observed_once(session_case, tmp_path):
    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    marker = tmp_path / "end"
    hook = command(
        f"from pathlib import Path;Path({str(marker)!r}).write_text('end')",
        name="SessionEnd",
    )
    case.runtime.ensure_hooks_v2(case.session.id, (hook,), lambda *_: True)
    assert (await case.submit()).accepted
    await case.runtime.close_hooks_v2(case.session.id)
    assert marker.read_text() == "end"
    await case.runtime.close_hooks_v2(case.session.id)


@pytest.mark.asyncio
async def test_scheduled_wake_initializes_without_user_prompt_submit(
    session_case, tmp_path
):
    from Tests.Chat.test_console_fleet_wake import (
        _drain,
        _settle,
        _survivor,
        _terminal_subagent_run,
    )

    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    prompt = tmp_path / "prompt"
    start = command(
        f"from pathlib import Path;Path({str(case.marker)!r}).write_text('start')",
        name="SessionStart",
        required=True,
    )
    user = command(
        f"from pathlib import Path;Path({str(prompt)!r}).write_text('prompt')",
        name="UserPromptSubmit",
        id="user",
    )
    case.runtime.ensure_hooks_v2(case.session.id, (start, user), lambda *_: True)
    _, run_id = _terminal_subagent_run(
        case.controller._agent_bridge.runs_db, case.session.id
    )
    case.controller.fleet_wake.on_fleet_drained(
        _drain(case.session.id, _survivor(run_id, session_id=case.session.id))
    )
    assert await _settle(lambda: case.gateway.payloads)
    assert case.marker.exists()
    assert not prompt.exists()
    assert case.admitted_turn_count() == 0  # Composer publication is manual-only.
    assert case.root_stop_count() == 1


@pytest.mark.asyncio
async def test_revoked_runtime_contribution_blocks_next_actual_send(session_case):
    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    authorized = True
    hook = command(
        'print(\'{"version":2,"decision":"pass","context":[{"text":"revocable runtime","lifetime":"runtime"}]}\')',
        name="SessionStart",
        required=True,
        effects=["context"],
    )
    case.runtime.ensure_hooks_v2(case.session.id, (hook,), lambda *_: authorized)
    assert (await case.submit()).accepted
    first_count = len(case.gateway.payloads)
    authorized = False
    result = await case.submit()
    assert not result.accepted
    assert len(case.gateway.payloads) == first_count
    assert case.root_stop_count() == 1


def test_shared_owner_scope_rejects_stale_claims_and_keeps_dependency_scope():
    from Tests.Agents.test_hooks_v2_post_checkpoints import post_event
    from tldw_chatbook.Agents.hooks_v2.checkpoints import (
        HookCheckpointError,
        HookCheckpointStore,
    )
    from tldw_chatbook.Agents.hooks_v2.engine import HookEventOutcome, HookFailure

    store = HookCheckpointStore()
    store.bind_owner("session")
    store.bind_owner("turn", "session")
    store.bind_owner("run", "turn")
    store.bind_owner("sibling", "session")
    with pytest.raises(HookCheckpointError):
        store.bind_owner("unknown-child", "unknown")
    with pytest.raises(HookCheckpointError):
        store.bind_owner("run", "session")
    with pytest.raises(HookCheckpointError):
        store.bind_owner("session", "run")
    event = post_event()
    token = store.begin(event, (), owner_id="turn", dependency_requirements=("dep",))
    store.accept(
        token, HookEventOutcome(failures=(HookFailure("dep", "failed", False, True),))
    )
    store.assert_next_input_allowed("run", required_handler_ids=())
    store.assert_next_input_allowed("sibling", required_handler_ids=("dep",))
    with pytest.raises(HookCheckpointError, match="failed"):
        store.assert_next_input_allowed("run", required_handler_ids=("dep",))
    with pytest.raises(HookCheckpointError, match="unknown"):
        store.assert_next_input_allowed("run", required_handler_ids=None)
    store.retire_owner("run")
    with pytest.raises(HookCheckpointError):
        store.bind_owner("run", "turn")
    store.retire_owner("turn")
    assert not store._entries
    store.assert_next_input_allowed("sibling")


@pytest.mark.asyncio
async def test_idle_config_replacement_creates_a_new_hook_session(
    session_case, tmp_path, hook_file
):
    from Tests.hooks_v2_process_support import child_argv

    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    marker = tmp_path / "generations"

    def configuration(label):
        code = f"from pathlib import Path;p=Path({str(marker)!r});p.write_text((p.read_text() if p.exists() else '')+{label!r})"
        return {
            "hooks": {
                "handler": [
                    {
                        "id": "start",
                        "event": "SessionStart",
                        "type": "command",
                        "required": True,
                        "effects": [],
                        "argv": child_argv(code),
                    }
                ]
            }
        }

    permissions = case.runtime.ensure_hook_permissions()
    case.controller._hook_permissions_accessor = lambda: permissions
    _edit(hook_file, lambda section: section.update(configuration("one")["hooks"]))
    assert not (await case.submit()).accepted
    assert not marker.exists()
    assert _approve(permissions).ready
    assert (await case.submit()).accepted
    first = case.runtime._hooks_v2_lifecycles[case.session.id]
    assert (await case.submit()).accepted
    assert marker.read_text() == "one"
    # A stale app dictionary cannot approve or activate changed saved hooks.
    case.runtime._app.app_config = configuration("one")
    _edit(hook_file, lambda section: section.update(configuration("two")["hooks"]))
    assert not permissions.snapshot().ready
    assert not first.current()
    assert not (await case.submit()).accepted
    assert marker.read_text() == "one"
    assert _approve(permissions).ready
    assert (await case.submit()).accepted
    second = case.runtime._hooks_v2_lifecycles[case.session.id]
    assert first is not second
    assert not first.live and second.live
    assert not first.context._rows
    assert marker.read_text() == "onetwo"


@pytest.mark.asyncio
async def test_session_seal_and_checkpoint_currentness_have_one_lock_order(
    session_case, monkeypatch
):
    import asyncio
    import threading

    case = session_case
    case.allow_required_initialization()
    assert (await case.submit()).accepted
    runtime = case.runtime
    lifecycle = runtime._hooks_v2_lifecycles[case.session.id]
    scope = lifecycle.open_scope()
    checkpoint_entered = threading.Event()
    sealing_entered = threading.Event()
    original_current = lifecycle.current
    original_lock = runtime._run_hooks_lock
    original_seal = lifecycle.seal
    lock_timeouts = []
    admission_sealed = []

    class BoundedMapLock:
        def __enter__(self):
            # Bound the actual get/seal lock, without acquiring it a second time.
            if not original_lock.acquire(timeout=0.5):
                lock_timeouts.append(True)
                raise RuntimeError("runtime/checkpoint lock inversion")
            return self

        def __exit__(self, *_args):
            original_lock.release()

    def current_under_checkpoint():
        checkpoint_entered.set()
        assert sealing_entered.wait(2), "sealing never reached the owner"
        return original_current()

    def seal_under_runtime():
        admission_sealed.append(lifecycle.engine._sealed_at is not None)
        sealing_entered.set()
        original_seal()

    monkeypatch.setattr(runtime, "_run_hooks_lock", BoundedMapLock())
    monkeypatch.setattr(lifecycle, "current", current_under_checkpoint)
    monkeypatch.setattr(lifecycle, "seal", seal_under_runtime)
    waiter = asyncio.create_task(lifecycle.wait(scope))
    assert await asyncio.to_thread(checkpoint_entered.wait, 2)
    closer = asyncio.create_task(
        asyncio.to_thread(runtime._seal_hooks_v2, case.session.id)
    )
    results = await asyncio.wait_for(
        asyncio.gather(waiter, closer, return_exceptions=True), 4
    )
    assert not lock_timeouts, results
    assert admission_sealed == [True]
    assert closer.exception() is None
    assert not lifecycle.checkpoints.is_current(scope)
    assert (
        not runtime._disposed
    )  # Exercise live session closure, not shutdown short-circuit.
    outcome = await lifecycle.engine.fire_async(
        lifecycle.event("SessionStart", data={"reason": "resume"})
    )
    assert not outcome.allowed


@pytest.mark.asyncio
async def test_absent_v2_control_does_not_treat_invalid_config_as_absence(
    session_case, hook_file
):
    import toml

    case = session_case
    await case.runtime.close_hooks_v2(case.session.id)
    raw = toml.loads(hook_file.read_text())
    raw["hooks"] = {}
    hook_file.write_text(toml.dumps(raw))
    assert await case.runtime.prepare_hooks_v2(case.session.id) is None
    raw["hooks"] = "invalid saved section"
    hook_file.write_text(toml.dumps(raw))
    with pytest.raises(RuntimeError, match="Review enabled hooks"):
        await case.runtime.prepare_hooks_v2(case.session.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True, "approval"])
async def test_actual_console_admission_runs_connected_mcp_initializer(
    tmp_path, nested
):
    import asyncio

    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    rig = _controller_rig(tmp_path)
    chacha, app, _runs, store, session, gateway, _bridge, controller = rig
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        app.unified_mcp_service = service
        from tldw_chatbook.MCP.hub_tool_catalog import HubTool

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
        if nested != "approval":
            service.set_tool_state("local:fixture", "fixture", "allow", tool=tool)
        positive = await service.execute_hub_tool_result(
            "local:fixture",
            "fixture",
            {"payload": {"structuredContent": {"version": 2, "decision": "pass"}}},
        )
        assert positive.dispatch_state == "settled" and not positive.is_error
        handlers = parse_handlers(
            [
                {
                    "id": "native-init",
                    "event": "SessionStart",
                    "type": "mcp_tool",
                    "server": "local:fixture",
                    "tool": "fixture",
                    "required": True,
                    "effects": ["context"],
                    "require_context": True,
                    "input": {
                        "payload": {
                            "structuredContent": {
                                "version": 2,
                                "decision": "pass",
                                "context": [
                                    {
                                        "text": "Console MCP runtime instructions",
                                        "lifetime": "runtime",
                                    }
                                ],
                            }
                        }
                    },
                }
            ]
        )
        if nested:
            import json

            payload = {
                "version": 2,
                "decision": "pass",
                "context": [
                    {
                        "text": "Only the triggering turn receives this",
                        "lifetime": "turn",
                    }
                ],
            }
            handlers += (
                command(
                    "print(" + repr(json.dumps(payload)) + ")",
                    effects=["context"],
                    required=True,
                ),
            )
        engine = runtime.ensure_hooks_v2(session.id, handlers, lambda *_: True)
        approved_tickets = []
        if nested == "approval":
            app.call_from_thread = lambda callback, *args, **kwargs: callback(
                *args, **kwargs
            )

            def decide(payload):
                if payload is None:
                    return
                assert engine.budget_owner.snapshot()["tickets"] == 1
                assert engine.budget_owner.snapshot()["execution"] == 0
                approved_tickets.append(next(iter(engine.mcp_executor._jobs)).ticket)
                controller.resolve_pending_approval(
                    {row["llm_name"]: "approve_once" for row in payload["calls"]},
                    round_id=payload["round_id"],
                )

            view = SimpleNamespace(
                app=app, console_view_hooks=lambda: {"set_pending_approval": decide}
            )
            app.screen = view
            generation = runtime.attach_view(view)
            runtime.finish_view_reconciliation(view, generation)
        try:
            result = await asyncio.wait_for(
                controller.submit_draft("hello", session_id=session.id), 10
            )
            assert result.accepted, result
            assert gateway.payloads
            assert "Console MCP runtime instructions" in str(gateway.payloads)
            assert len(service.execution_log.read_recent()) == 2
            assert engine.budget_owner.snapshot()["tickets"] == 0
            if nested == "approval":
                assert len(approved_tickets) == 1 and approved_tickets[0].released
                assert not controller._pending_approval_rounds
            if nested:
                assert "Only the triggering turn receives this" in str(gateway.payloads)
                gateway.payloads.clear()
                again = await controller.submit_draft(
                    "second input", session_id=session.id
                )
                assert again.accepted
                assert "Only the triggering turn receives this" not in str(
                    gateway.payloads
                )
                assert "Console MCP runtime instructions" in str(gateway.payloads)
                assert len(service.execution_log.read_recent()) == 2
        finally:
            await runtime.close_hooks_v2()
            chacha.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["approve", "late_answer", "timeout", "cancel", "observation"]
)
async def test_hook_approval_uses_actual_retained_round_deadline_and_cancellation(mode):
    import asyncio
    import contextlib
    import threading
    import time

    from Tests.UI.test_console_mcp_approval import _build_controller, _FakeApp, _pending
    from tldw_chatbook.Agents.mcp_tool_provider import (
        MCPInvocationPolicy,
        restrict_mcp_invocation,
    )

    controller, store = _build_controller()
    session = store.ensure_session()
    controller.app = _FakeApp()
    received = []
    policy = MCPInvocationPolicy(
        current=lambda: True,
        deadline=time.monotonic() + 3,
        cancel_event=threading.Event(),
        allow_approval=mode != "observation",
        wait_scope=lambda _kind: contextlib.nullcontext(),
        approval_deadline=lambda: approval_end,
    )
    approval_end = time.monotonic() + (
        0.12 if mode in {"late_answer", "timeout"} else 2
    )

    def show(payload):
        if payload is None:
            return
        received.append(payload)
        if mode in {"approve", "late_answer"}:
            if mode == "late_answer":
                time.sleep(0.18)
            controller.resolve_pending_approval(
                {"mcp__srv__tool": "approve_once"}, round_id=payload["round_id"]
            )
        if mode == "cancel":
            policy.cancel_event.set()

    controller.set_pending_approval = show

    def request():
        with restrict_mcp_invocation(policy):
            return controller.request_mcp_approvals([_pending()], session_id=session.id)

    pending = asyncio.create_task(asyncio.to_thread(request))
    try:
        answer = await asyncio.wait_for(asyncio.shield(pending), 2)
        assert answer["mcp__srv__tool"] == (
            "approve_once"
            if mode == "approve"
            else "timeout"
            if mode in {"late_answer", "timeout"}
            else "deny"
        )
        assert bool(received) is (mode != "observation")
        assert not controller._pending_approval_rounds
        if mode != "approve":
            assert not controller._parked_approval_payloads
    finally:
        policy.cancel_event.set()
        await asyncio.wait_for(pending, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["sibling", "cancel", "overflow"])
async def test_failed_console_initialization_discards_nested_turn_context(
    tmp_path, failure
):
    import asyncio
    import json

    from Tests.Agents.test_hooks_v2_tool_pipeline import hook_command
    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    chacha, app, _runs, store, session, gateway, _bridge, controller = _controller_rig(
        tmp_path
    )
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    release, started = tmp_path / "release-sibling", tmp_path / "sibling-started"
    direct_blocks = [{"text": "Runtime context", "lifetime": "runtime"}]
    nested_blocks = [{"text": "Discarded nested secret", "lifetime": "turn"}]
    if failure == "overflow":
        direct_blocks = [{"text": "r" * 3000, "lifetime": "runtime"}] * 3
        nested_blocks = [{"text": "n" * 3600, "lifetime": "turn"}] * 4
    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        app.unified_mcp_service = service
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
        handlers = list(
            parse_handlers(
                [
                    {
                        "id": "initializer",
                        "event": "SessionStart",
                        "type": "mcp_tool",
                        "required": True,
                        "server": "local:fixture",
                        "tool": "fixture",
                        "effects": ["context"],
                        "input": {
                            "payload": {
                                "structuredContent": {
                                    "version": 2,
                                    "decision": "pass",
                                    "context": direct_blocks,
                                }
                            }
                        },
                    }
                ]
            )
        )
        nested_body = json.dumps(
            {"version": 2, "decision": "pass", "context": nested_blocks}
        )
        handlers.append(
            hook_command(
                "nested-pre",
                effects=["context"],
                required=True,
                code=f"print({nested_body!r})",
            )
        )
        if failure == "overflow":
            handlers.append(
                hook_command(
                    "nested-post",
                    event="PostToolUse",
                    effects=["context"],
                    required=True,
                    code=f"print({nested_body!r})",
                )
            )
        else:
            wait = f"while not Path({str(release)!r}).exists(): time.sleep(.01)"
            code = (
                f"from pathlib import Path;import time;Path({str(started)!r}).touch();"
            )
            code += f"exec({wait!r})" if failure == "cancel" else "raise SystemExit(1)"
            handlers.append(
                hook_command("sibling", event="SessionStart", required=True, code=code)
            )
        engine = runtime.ensure_hooks_v2(session.id, tuple(handlers), lambda *_: True)
        pending = asyncio.create_task(
            controller.submit_draft("first", session_id=session.id)
        )
        try:
            if failure == "cancel":
                for _ in range(400):
                    if started.exists():
                        break
                    await asyncio.sleep(0.01)
                assert started.exists(), "enclosing sibling never started"
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                assert not (await asyncio.wait_for(pending, 6)).accepted
            assert not gateway.payloads
            assert len(service.execution_log.read_recent()) == 1
            release.touch()
            await runtime.close_hooks_v2(session.id)
            runtime.ensure_hooks_v2(
                session.id,
                (command(name="SessionStart", required=True),),
                lambda *_: True,
            )
            result = await controller.submit_draft("replacement", session_id=session.id)
            assert result.accepted and gateway.payloads
            assert "Discarded nested secret" not in str(gateway.payloads)
            assert "Runtime context" not in str(gateway.payloads)
            assert engine.budget_owner.snapshot()["tickets"] == 0
        finally:
            release.touch()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await runtime.close_hooks_v2()
            chacha.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "guard_denies",
        "requires_context",
        "ask",
        "stale",
        "dependency_pending",
        "requirements_changed",
        "cap_exhausted",
        "definition_changed",
    ],
)
async def test_actual_console_teardown_mcp_uses_captured_runtime_and_normal_guards(
    tmp_path, mode
):
    import json
    import time
    from dataclasses import replace

    from Tests.Agents.test_hooks_v2_tool_pipeline import hook_command
    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    chacha, app, _runs, store, session, gateway, _bridge, controller = _controller_rig(
        tmp_path
    )
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        app.unified_mcp_service = service
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
        if mode != "ask":
            service.set_tool_state("local:fixture", "fixture", "allow", tool=tool)
        handlers = [command(name="SessionStart", required=True)]
        handlers.extend(
            parse_handlers(
                [
                    {
                        "id": "teardown",
                        "event": "SessionEnd",
                        "type": "mcp_tool",
                        "server": "local:fixture",
                        "tool": "fixture",
                        "effects": [],
                        "input": {
                            "payload": {
                                "structuredContent": {"version": 2, "decision": "pass"}
                            }
                        },
                    }
                ]
            )
        )
        if mode in {"guard_denies", "requires_context"}:
            output = {
                "version": 2,
                "decision": "deny" if mode == "guard_denies" else "pass",
            }
            if mode == "requires_context":
                output["context"] = [
                    {"text": "No input owner remains", "lifetime": "turn"}
                ]
            handlers.append(
                hook_command(
                    "normal-guard",
                    required=True,
                    effects=["deny"] if mode == "guard_denies" else ["context"],
                    code=f"print({json.dumps(output)!r})",
                )
            )
        engine = runtime.ensure_hooks_v2(session.id, tuple(handlers), lambda *_: True)
        prompts = []
        controller.set_pending_approval = prompts.append
        try:
            assert (
                await controller.submit_draft("hello", session_id=session.id)
            ).accepted
            assert gateway.payloads
            assert not service.execution_log.read_recent()
            view = engine.mcp_executor._runtime_contexts[session.id]
            assert view.input_scope is None and view.run_id == view.lifecycle.scope_id
            assert view.budget_run_id != view.run_id
            if mode == "stale":
                service.set_tool_state("local:fixture", "fixture", "deny", tool=tool)
            if mode == "definition_changed":
                discovery = await client.describe_server("fixture")
                discovery["tools"][0]["description"] = "changed after admission"
                service.local_service.store.save_discovery_snapshot(
                    "fixture", discovery
                )
            if mode == "dependency_pending":
                view.lifecycle.checkpoints.begin(
                    view.lifecycle.event("Stop"),
                    ("initializer",),
                    owner_id=view.parent_scope,
                )
                engine.mcp_executor._runtime_contexts[session.id] = replace(
                    view, required_handler_ids=lambda _definition: ("initializer",)
                )
            if mode == "requirements_changed":
                state = {"ids": ()}
                engine.mcp_executor._runtime_contexts[session.id] = replace(
                    view, required_handler_ids=lambda _definition: state["ids"]
                )
                original = engine.notify_teardown

                def changed(event):
                    state["ids"] = ("new-required-initializer",)
                    return original(event)

                engine.notify_teardown = changed
            if mode == "cap_exhausted":
                from tldw_chatbook.Agents.run_tool_policy import RunToolPolicy

                name, _provider = view.resolve(handlers[1])
                caps = RunToolPolicy({name: 1})
                view.registry.set_run_tool_policy(caps)
                assert caps.check(view.budget_run_id, name)[0]
            started = time.monotonic()
            await runtime.close_hooks_v2(session.id)
            assert time.monotonic() - started < 8.5
            # A real counter call proves whether the peer saw the notification.
            counter = await client.call_tool_result(
                "fixture", "fixture", {"counter": True}
            )
            assert counter.content[0]["text"] == ("2" if mode == "success" else "1"), (
                dict(engine.notification_failures),
                engine.notification_omissions,
                engine.budget_owner.snapshot(),
            )
            assert not prompts
            assert (
                engine.budget_owner.snapshot()["tickets"]
                == engine.budget_owner.snapshot()["execution"]
                == 0
            )
            denied = await engine.fire_async(
                engine.lifecycle_owner.event("SessionStart", data={"reason": "startup"})
            )
            assert not denied.allowed
            await runtime.close_hooks_v2(session.id)
            again = await client.call_tool_result(
                "fixture", "fixture", {"counter": True}
            )
            assert again.content[0]["text"] == ("3" if mode == "success" else "2")
        finally:
            await runtime.close_hooks_v2()
            chacha.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "slow_guard", "closed"])
async def test_actual_console_stop_mcp_notification_keeps_one_second_guard_deadline(
    tmp_path, mode
):
    import asyncio
    import time
    from uuid import uuid4

    from Tests.Agents.test_hooks_v2_tool_pipeline import hook_command
    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    chacha, app, _runs, store, session, gateway, _bridge, controller = _controller_rig(
        tmp_path
    )
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    started, release = asyncio.Event(), asyncio.Event()

    async def blocked_stream(*_args, **_kwargs):
        started.set()
        await release.wait()
        yield "done"

    gateway.stream_chat = blocked_stream
    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        app.unified_mcp_service = service
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
        handlers = [command(name="SessionStart", required=True)]
        handlers.extend(
            parse_handlers(
                [
                    {
                        "id": "interrupt",
                        "event": "Interrupt",
                        "type": "mcp_tool",
                        "server": "local:fixture",
                        "tool": "fixture",
                        "effects": [],
                        "input": {
                            "payload": {
                                "structuredContent": {"version": 2, "decision": "pass"}
                            }
                        },
                    }
                ]
            )
        )
        if mode == "slow_guard":
            handlers.append(
                hook_command(
                    "slow-guard",
                    required=True,
                    effects=["deny"],
                    code='import time;time.sleep(20);print(\'{"version":2,"decision":"pass"}\')',
                )
            )
        engine = runtime.ensure_hooks_v2(session.id, tuple(handlers), lambda *_: True)
        request = ConsoleTurnCustodyRequest(
            turn_id=str(uuid4()),
            session_id=session.id,
            draft="hello",
            configuration=controller.resolve_turn_configuration_snapshot(session.id),
        )
        turn = runtime.accept_turn(request)
        pending = asyncio.create_task(runtime.wait_for_turn(turn))
        try:
            await asyncio.wait_for(started.wait(), 5)
            stamp = time.monotonic()
            assert controller.stop_active_run()
            controller.stop_active_run()
            if mode == "closed":
                runtime._seal_hooks_v2(session.id)
            release.set()
            await asyncio.wait_for(pending, 5)
            for _ in range(400):
                if not engine.cleanup_pending and not any(
                    engine.budget_owner.snapshot().values()
                ):
                    break
                await asyncio.sleep(0.01)
            assert time.monotonic() - stamp < (2.5 if mode == "slow_guard" else 2)
            assert engine.budget_owner.snapshot()["tickets"] == 0
            counter = await client.call_tool_result(
                "fixture", "fixture", {"counter": True}
            )
            assert counter.content[0]["text"] == ("2" if mode == "success" else "1"), (
                dict(engine.notification_failures),
                engine.notification_omissions,
                engine.budget_owner.snapshot(),
            )
        finally:
            release.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await runtime.close_hooks_v2()
            chacha.close()


@pytest.mark.asyncio
async def test_hook_currentness_does_not_read_unrelated_skill_catalog(
    tmp_path, monkeypatch
):
    import tldw_chatbook.Chat.console_chat_controller as module
    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )

    chacha, app, runs, store, session, _gateway, _bridge, controller = _controller_rig(
        tmp_path
    )
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    async with controlled_stdio_client(tmp_path) as client:
        app.unified_mcp_service = await controlled_service(tmp_path, client)
        from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

        teardown = parse_handlers(
            [
                {
                    "id": "teardown",
                    "event": "SessionEnd",
                    "type": "mcp_tool",
                    "server": "local:fixture",
                    "tool": "fixture",
                    "effects": [],
                }
            ]
        )[0]
        engine = runtime.ensure_hooks_v2(
            session.id,
            (command(name="SessionStart", required=True), teardown),
            lambda *_: True,
        )
        try:
            assert (
                await controller.submit_draft("hello", session_id=session.id)
            ).accepted
            view = engine.mcp_executor._runtime_contexts[session.id]

            def unrelated_catalog(*_args, **_kwargs):
                raise AssertionError("hook currentness read unrelated skill catalog")

            monkeypatch.setattr(
                module, "capture_skill_context_maximum", unrelated_catalog
            )
            assert view.current()
        finally:
            await runtime.close_hooks_v2()
            await runtime.dispose()
            runs.close()
            chacha.close()
