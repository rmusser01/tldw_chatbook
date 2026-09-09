"""Runtime continuation uses real native increments and durable shared allowance."""

import asyncio
import threading

import pytest

from Tests.Agents.test_goal_iteration_report import report
from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture

stores = _stores_fixture


def progress(number):
    return {
        "choices": [
            {"message": {"content": report(candidate_draft=f"draft {number}")}}
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


@pytest.mark.asyncio
async def test_successors_stop_at_original_cap_in_one_chain(stores, monkeypatch):
    count = 0

    def provider(**kwargs):
        nonlocal count
        count += 1
        return progress(count)

    goal, _, _, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    try:
        assert hasattr(coordinator, "start"), "runtime continuation API missing"
        saved = await coordinator.start(goal.id)
        assert (
            saved.status == "paused" and saved.pause_reason == "goal_budget_exhausted"
        )
        assert len(calls) == 3
        assert saved.chain_id == goal.chain_id
        assert saved.accounting.used["generation"] == 3
        assert saved.accounting.used["model_call"] == 3
        assert controller.in_flight_run_count() == 0
        await coordinator.start(goal.id)
        assert len(calls) == 3
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control,expected", [("pause", "paused"), ("stop", "recovery_required")]
)
async def test_control_retains_ownership_until_actual_provider_cleanup(
    stores, monkeypatch, control, expected
):
    entered, release = threading.Event(), threading.Event()

    def provider(**kwargs):
        entered.set()
        assert release.wait(5)
        return progress(1)

    goal, store, _session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    assert hasattr(coordinator, "start"), "runtime continuation API missing"
    task = coordinator.start(goal.id)
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        getattr(coordinator.service, control)(goal.id)
        await asyncio.sleep(0.03)
        assert not task.done() and controller.in_flight_run_count() == 1
        alias = store.create_session(workspace_id="workspace")
        alias.persisted_conversation_id = goal.conversation_id
        blocked = await controller.submit_draft("manual alias", session_id=alias.id)
        assert not blocked.accepted
        release.set()
        saved = await task
        assert saved.status == expected, saved
        assert len(calls) == 1 and controller.in_flight_run_count() == 0
        assert saved.iteration_count == 1
    finally:
        release.set()
        await task
        await gateway.aclose()


@pytest.mark.asyncio
async def test_capacity_wait_does_not_prepare_or_consume_generation(
    stores, monkeypatch
):
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus

    goal, store, _, controller, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, lambda **kw: progress(len(calls))
    )
    busy = [store.create_session(workspace_id="workspace") for _ in range(2)]
    for s in busy:
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING), session_id=s.id
        )
    task = co.start(goal.id)
    try:
        await asyncio.sleep(0.1)
        assert not task.done(), "capacity refusal must remain an owned bounded wait"
        snapshot = co.service.get(goal.id)
        assert (
            snapshot.retry_at is not None
            and snapshot.pause_reason == "primary_capacity"
        )
        assert snapshot.iteration_count == 0 and not calls
        assert snapshot.accounting.reserved["generation"] == 0
        for s in busy:
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.IDLE), session_id=s.id
            )
        saved = await asyncio.wait_for(task, 3)
        assert saved.iteration_count == 3 and len(calls) == 3
    finally:
        co.close_admission()
        await task
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,expected,count",
    [
        ("rate", "paused", 3),
        ("permanent", "paused", 1),
        ("generic_rate", "recovery_required", 1),
        ("timeout", "recovery_required", 1),
    ],
)
async def test_adapter_proof_survives_native_route_and_retry_is_bounded(
    stores, monkeypatch, kind, expected, count
):
    import importlib.util

    spec = importlib.util.find_spec("tldw_chatbook.LLM_Calls.provider_outcomes")
    assert spec is not None, "typed adapter pre-effect contract missing"
    from tldw_chatbook.Chat.Chat_Deps import ChatRateLimitError
    from tldw_chatbook.LLM_Calls.provider_outcomes import (
        PreEffectPermanentError,
        PreEffectRateLimitError,
    )

    def provider(**kw):
        raise {
            "rate": PreEffectRateLimitError,
            "permanent": PreEffectPermanentError,
            "generic_rate": ChatRateLimitError,
            "timeout": TimeoutError,
        }[kind]("local gate")

    goal, store, _, controller, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    task = co.start(goal.id)
    try:
        for _ in range(300):
            if calls or task.done():
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.25)
        assert len(calls) == 1, "provider attempts must not storm while waiting"
        if kind == "rate":
            assert not task.done()
            waiting = co.service.get(goal.id)
            assert waiting.retry_at and waiting.accounting.reserved["tokens"] == 0
            # Independent manual generation still has reserved physical capacity.
            gateway._chat_api_call_fn = lambda **kw: progress(99)
            from tldw_chatbook.Chat.console_project_instructions import (
                ProjectInstructionControlState,
            )

            other = store.create_session(
                workspace_id="workspace",
                project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
            )
            result = await controller.submit_draft("manual work", session_id=other.id)
            assert result.accepted
            gateway._chat_api_call_fn = lambda **kw: (calls.append(kw), provider(**kw))[
                1
            ]
        saved = await asyncio.wait_for(task, 8)
        assert saved.status == expected
        assert len(calls) == count
        assert saved.iteration_count == count
        if kind == "rate":
            assert saved.pause_reason in {
                "pre_effect_retry_exhausted",
                "goal_budget_exhausted",
            }
        assert saved.accounting.used["model_call"] == count
        if kind in {"generic_rate", "timeout"}:
            assert saved.accounting.uncertain
        else:
            assert not saved.accounting.uncertain
            assert saved.accounting.used["tokens"] == 0
    finally:
        co.close_admission()
        await task
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("controller_failure", [False, True])
async def test_runtime_disposal_stops_retry_before_draining_controller(
    stores, monkeypatch, controller_failure
):
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    goal, store, _, controller, co, gateway, calls = build_goal_rig(stores, monkeypatch)
    for _ in range(2):
        s = store.create_session(workspace_id="workspace")
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING), session_id=s.id
        )
    task = co.start(goal.id)
    await asyncio.sleep(0.05)
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime._chat_controller = controller
    runtime._provider_gateway = gateway
    if controller_failure:

        async def fail_shutdown():
            raise RuntimeError("injected controller teardown failure")

        monkeypatch.setattr(controller, "shutdown", fail_shutdown)
    gateway_close = gateway.aclose
    close_order = []

    async def close_after_goal():
        close_order.append(task.done())
        await gateway_close()

    monkeypatch.setattr(gateway, "aclose", close_after_goal)
    await runtime.dispose()
    assert close_order == [True], "goal drain must precede gateway close even on error"
    assert task.done(), "runtime disposal must drain the goal scheduler"
    assert co.service.get(goal.id).status == "stopped"
    assert not calls
    with pytest.raises(RuntimeError):
        co.start(goal.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit,expected_calls",
    [("iterations", 2), ("model_calls", 1), ("budget_tokens", 0)],
)
async def test_earliest_launch_allowance_wins(
    stores, monkeypatch, limit, expected_calls
):
    runs, persistence, registry, req = stores
    value = 2 if limit == "iterations" else 1
    req = req.model_copy(
        update={"policy": req.policy.model_copy(update={limit: value})}
    )
    count = 0

    def provider(**kw):
        nonlocal count
        count += 1
        return progress(count)

    goal, _, _, _, co, gateway, calls = build_goal_rig(
        (runs, persistence, registry, req), monkeypatch, provider
    )
    try:
        saved = await co.start(goal.id)
        assert saved.status == "paused"
        assert len(calls) == expected_calls
        before = saved.accounting
        monkeypatch.setenv("TLDW_AGENTS_MAX_GOAL_GENERATIONS", "100")
        monkeypatch.setenv("TLDW_AGENTS_MAX_GOAL_MODEL_CALLS", "100")
        monkeypatch.setenv("TLDW_AGENTS_MAX_GOAL_BUDGET_TOKENS", "999999")
        if saved.checkpoints:
            co.service.resume(goal.id, expected_revision=saved.revision)
        await co.start(goal.id)
        assert len(calls) == expected_calls
        assert co.service.get(goal.id).accounting.limits == before.limits
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_failed_report_and_later_progress_keep_all_accepted_charges(
    stores, monkeypatch
):
    count = 0

    def provider(**kw):
        nonlocal count
        count += 1
        response = progress(count)
        if count == 1:
            response["choices"][0]["message"]["content"] = "malformed report"
        return response

    goal, _, _, _, co, gateway, calls = build_goal_rig(stores, monkeypatch, provider)
    try:
        saved = await co.start(goal.id)
        assert len(calls) == 3 and saved.iteration_count == 3
        assert saved.checkpoints[0].report_error == "malformed_report"
        assert saved.accounting.used["tokens"] == 30
        assert saved.accounting.used["model_call"] == 3
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_pre_effect_proof_does_not_refund_earlier_native_call(
    stores, monkeypatch
):
    import json

    from tldw_chatbook.Agents.goal_models import GoalToolScope
    from tldw_chatbook.LLM_Calls.provider_outcomes import (
        PreEffectPermanentError,
        PreEffectRateLimitError,
    )

    runs, persistence, registry, req = stores
    req = req.model_copy(
        update={
            "tool_scope": GoalToolScope(
                catalog_tools=("builtin:get_current_datetime",),
                runtime_tools=("find_tools",),
            )
        }
    )
    count = 0

    def provider(**kw):
        nonlocal count
        count += 1
        if count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "lookup",
                                    "type": "function",
                                    "function": {
                                        "name": "find_tools",
                                        "arguments": json.dumps({"query": "none"}),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        raise (
            PreEffectRateLimitError("local gate")
            if count == 2
            else PreEffectPermanentError("permanent local gate")
        )

    goal, _, _, _, co, gateway, calls = build_goal_rig(
        (runs, persistence, registry, req), monkeypatch, provider
    )
    try:
        saved = await co.start(goal.id)
        assert saved.status == "paused" and len(calls) == 3
        assert saved.accounting.used["generation"] == 2, [
            (c.report_error, c.decision.reason) for c in saved.checkpoints
        ]
        assert saved.accounting.used["model_call"] == 3
        assert saved.accounting.used["tokens"] == 10
        assert not saved.accounting.uncertain
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_pause_resume_and_capacity_events_do_not_erase_provider_backoff(
    stores, monkeypatch
):
    from tldw_chatbook.LLM_Calls.provider_outcomes import PreEffectRateLimitError

    def provider(**kw):
        raise PreEffectRateLimitError("local gate")

    goal, _, _, _, co, gateway, calls = build_goal_rig(stores, monkeypatch, provider)
    task = co.start(goal.id)
    try:
        while not co.service.get(goal.id).retry_at:
            await asyncio.sleep(0.01)
        waiting = co.service.get(goal.id)
        co.service.pause(goal.id)
        paused = await task
        co.service.resume(goal.id, expected_revision=paused.revision)
        task = co.start(goal.id)
        await asyncio.sleep(0.02)
        co.notify_capacity()
        await asyncio.sleep(0.1)
        assert len(calls) == 1, "explicit resume must retain the saved provider backoff"
        assert co.service.get(goal.id).retry_at == waiting.retry_at
    finally:
        co.close_admission()
        await task
        await gateway.aclose()


@pytest.mark.asyncio
async def test_manual_draft_priority_on_goal_alias_blocks_automatic_admission(
    stores, monkeypatch
):
    goal, store, _, controller, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, lambda **kw: progress(len(calls))
    )
    alias = store.create_session(workspace_id="workspace")
    alias.persisted_conversation_id = goal.conversation_id
    pending = True
    controller.wake_user_priority_probe = lambda session_id: (
        pending and session_id == alias.id
    )
    task = co.start(goal.id)
    try:
        await asyncio.sleep(0.15)
        assert not calls, "a pending manual draft in a conversation alias has priority"
        assert co.service.get(goal.id).iteration_count == 0
        pending = False
        co.notify_capacity()
        saved = await asyncio.wait_for(task, 3)
        assert saved.iteration_count == 3
    finally:
        co.close_admission()
        await task
        await gateway.aclose()


@pytest.mark.asyncio
async def test_goal_and_fleet_share_two_automatic_primaries_even_with_larger_total_cap(
    stores, monkeypatch
):
    from types import SimpleNamespace

    from Tests.Chat.test_automatic_wake_budget import queue_result, result_for
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )

    entered, release = threading.Event(), threading.Event()

    def provider(**kw):
        entered.set()
        assert release.wait(8)
        return progress(1)

    goal, store, _, controller, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "true")
    monkeypatch.setattr(type(controller), "max_parallel_runs", property(lambda _: 5))
    controller.fleet_wake.wire(app=SimpleNamespace(chachanotes_db=stores[1].db))
    task = co.start(goal.id)
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        for index in range(2):
            session = store.create_session(
                workspace_id="workspace",
                project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
            )
            stores[1].db.add_conversation(
                {
                    "id": session.id,
                    "title": "fleet",
                    "workspace_id": "workspace",
                    "runtime_backend": "local",
                    "assistant_kind": "generic",
                    "assistant_id": "console",
                }
            )
            session.persisted_conversation_id = session.id
            rig = (
                stores[1].db,
                None,
                stores[0],
                store,
                session,
                gateway,
                controller._agent_bridge,
                controller,
            )
            chain = co.ledger.create_chain(
                session.id, root_submission_id=f"manual-{index}"
            )
            queue_result(rig, result_for(rig, chain))
        await asyncio.sleep(0.6)
        assert len(calls) == 2, (
            "one goal plus one fleet turn exhausts automatic primary capacity"
        )
        assert controller.in_flight_run_count() == 2
    finally:
        co.close_admission()
        controller.fleet_wake._disposed = True
        release.set()
        await task
        await controller.shutdown()
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["binding", "provider"])
async def test_successor_revalidates_live_authority_and_saves_refusal(
    stores, monkeypatch, change
):
    from dataclasses import replace

    from Tests.Chat.test_automatic_provider_budget import resolution

    goal, _, _, _, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, lambda **kw: progress(len(calls))
    )
    checkpoint = co.service.checkpoint

    def changed_after_settlement(result):
        saved = checkpoint(result)
        if change == "binding":
            stores[2].remove_runtime_binding("binding")
        else:

            async def drift(_selection):
                return replace(resolution(), model="changed-model")

            gateway.resolve_for_send = drift
        return saved

    monkeypatch.setattr(co.service, "checkpoint", changed_after_settlement)
    try:
        saved = await co.start(goal.id)
        assert len(calls) == 1 and saved.iteration_count == 1
        assert saved.status == "paused"
        assert saved.pause_reason == (
            "binding_missing" if change == "binding" else "provider_binding_changed"
        )
        assert saved.accounting.used["generation"] == 1
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_two_no_progress_increments_pause_without_a_third_dispatch(
    stores, monkeypatch
):
    def provider(**kw):
        response = progress(1)
        response["choices"][0]["message"]["content"] = report()
        return response

    goal, _, _, _, co, gateway, calls = build_goal_rig(stores, monkeypatch, provider)
    try:
        saved = await co.start(goal.id)
        assert saved.status == "paused" and saved.pause_reason == "no_progress"
        assert len(calls) == 2 and saved.iteration_count == 2
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_off_loop_stop_marshals_controller_signal_to_runtime_loop(
    stores, monkeypatch
):
    entered, release = threading.Event(), threading.Event()

    def provider(**kw):
        entered.set()
        assert release.wait(5)
        return progress(1)

    goal, _, _, controller, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    loop_thread = threading.get_ident()
    signal_threads = []
    original = controller._signal_stop

    def signal(**kw):
        signal_threads.append(threading.get_ident())
        return original(**kw)

    monkeypatch.setattr(controller, "_signal_stop", signal)
    task = co.start(goal.id)
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        await asyncio.to_thread(co.service.stop, goal.id)
        await asyncio.sleep(0.03)
        assert signal_threads and set(signal_threads) == {loop_thread}
        assert not task.done() and controller.in_flight_run_count() == 1
        release.set()
        saved = await task
        assert saved.status == "recovery_required" and len(calls) == 1
    finally:
        release.set()
        await task
        await gateway.aclose()


@pytest.mark.asyncio
async def test_checkpoint_failure_fences_continuation_without_redispatch(
    stores, monkeypatch
):
    import sqlite3

    goal, _, _, _, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, lambda **kw: progress(1)
    )

    def unavailable(result):
        raise sqlite3.OperationalError("injected checkpoint failure")

    monkeypatch.setattr(co.service, "checkpoint", unavailable)
    try:
        saved = await co.start(goal.id)
        assert saved.status == "recovery_required"
        await co.start(goal.id)
        assert len(calls) == 1 and saved.accounting.used["generation"] == 1
        assert saved.pause_reason == "checkpoint_unavailable"
    finally:
        await gateway.aclose()
