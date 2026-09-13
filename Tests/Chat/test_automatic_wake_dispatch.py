"""Controller-to-provider dispatch uses durable automatic authority on both paths."""

import asyncio
from dataclasses import replace

import pytest

from Tests.Chat.test_automatic_provider_budget import resolution, response
from Tests.Chat.test_automatic_wake_budget import close_rig, queue_result, result_for
from Tests.Chat.test_console_fleet_wake import _controller_rig, _settle
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits
from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway


@pytest.fixture
def rig(tmp_path):
    return _controller_rig(tmp_path)


def real_gateway(rig, *, agent=False, provider=None):
    calls = []

    def generate(**kwargs):
        context = current_automatic_work()
        calls.append((kwargs, context.chain_id if context else None))
        return provider(**kwargs) if provider else response()

    gateway = ConsoleProviderGateway(chat_api_call_fn=generate)

    async def resolve(_selection):
        return resolution()

    gateway.resolve_for_send = resolve
    controller = rig[7]
    controller.provider_gateway = gateway
    if agent:
        from tldw_chatbook.Chat.console_project_instructions import (
            ProjectInstructionControlState,
        )

        rig[
            4
        ].project_instruction_state = ProjectInstructionControlState.legacy_disabled()
        bridge = ConsoleAgentBridge(
            agent_runs_db=rig[2], store=rig[3], provider_gateway=gateway
        )
        controller.update_agent_runtime(bridge=bridge, enabled=True)
    return gateway, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", [False, True])
async def test_controller_enforces_three_generations_and_real_provider_usage(
    rig, agent
):
    db, session = rig[2], rig[4]
    gateway, calls = real_gateway(rig, agent=agent)
    chain = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    try:
        for _ in range(3):
            child = result_for(rig, chain)
            queue_result(rig, child)
            assert await _settle(
                lambda child=child: db.get_run(child)["wake_delivered_at"] is not None
            )
        fourth = result_for(rig, chain)
        queue_result(rig, fourth)
        assert await _settle(
            lambda: (
                db.automatic_work.snapshot(chain).pause_reason == "generation_budget"
            )
        )
        assert len(calls) == 3, [
            (m.role, m.content) for m in rig[3].messages_for_session(session.id)
        ]
        assert {identity for _, identity in calls} == {chain}
        assert all(call["max_tokens"] <= 8192 for call, _ in calls)
        snapshot = db.automatic_work.snapshot(chain)
        assert snapshot.used["model_call"] == 3
        assert snapshot.used["tokens"] == 30
        assert snapshot.reserved["tokens"] == 0
    finally:
        await close_rig(rig)
        await gateway.aclose()


@pytest.mark.asyncio
async def test_manual_send_clears_inherited_automatic_scope_before_helpers(rig):
    from Tests.Chat.test_automatic_provider_budget import context_for

    context = context_for(rig[2])
    gateway, calls = real_gateway(rig)
    controller = rig[7]
    helper_contexts = []
    manual_session = controller.new_session()
    substitute = controller._apply_skill_substitution

    async def observed(*args, **kwargs):
        helper_contexts.append(current_automatic_work())
        return await substitute(*args, **kwargs)

    controller._apply_skill_substitution = observed
    try:
        with context.scope():
            assert (
                await controller.submit_draft("Continue", session_id=manual_session.id)
            ).accepted
            assert current_automatic_work() is context
        assert helper_contexts == [None]
        assert calls and calls[0][1] is None
        assert rig[2].automatic_work.snapshot(context.chain_id).used["model_call"] == 0
    finally:
        await close_rig(rig)
        await gateway.aclose()


@pytest.mark.asyncio
async def test_elapsed_limit_cancels_a_blocked_stream_and_releases_primary_slot(rig):
    controller, db, session, gateway = rig[7], rig[2], rig[4], rig[5]
    gateway.stream_gate = asyncio.Event()
    chain = db.automatic_work.create_chain(
        session.id,
        root_submission_id="manual",
        limits=replace(AutomaticWorkLimits(), wall_seconds=1),
    )
    try:
        child = result_for(rig, chain)
        queue_result(rig, child)
        assert await _settle(lambda: gateway.payloads)
        assert await _settle(
            lambda: not controller.fleet_wake._delivery_tasks, seconds=3
        )
        assert db.automatic_work.snapshot(chain).pause_reason == "wall_budget"
        assert controller.fleet_wake.has_pending(session.id)
        assert controller.in_flight_run_count() == 0
        assert db.get_run(child)["wake_delivered_at"] is None
    finally:
        gateway.stream_gate.set()
        await close_rig(rig)


@pytest.mark.asyncio
async def test_fast_survivor_wakes_before_slow_sibling_finishes(tmp_path):
    from Tests.Chat.test_console_agent_bridge import _join_fleet_threads, _run
    from Tests.Chat.test_console_fleet_wake import _RecordingWakeGateway
    from Tests.Chat.test_individual_fleet_settlement import _bridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    child_gateway, db, store, session, aid, bridge = _bridge(tmp_path)
    gateway = _RecordingWakeGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=False,
    )
    drains = []
    bridge.on_fleet_drained("final-usage-proof", drains.append)
    try:
        assert (
            await asyncio.to_thread(
                _run, bridge, store, session, aid, conversation_id=session.id
            )
        ).status == "done"
        assert await _settle(child_gateway.entered["fast"].is_set)
        assert await _settle(child_gateway.entered["slow"].is_set)
        child_gateway.gates["fast"].set()
        assert await _settle(lambda: gateway.payloads)
        assert "fast answer" in gateway.payloads[0][-1]["content"]
        assert "slow answer" not in gateway.payloads[0][-1]["content"]
        assert drains == []
        assert bridge.has_unsettled_children(session.id)
        child_gateway.gates["slow"].set()
        assert await _settle(lambda: len(gateway.payloads) == 2)
        assert "slow answer" in gateway.payloads[1][-1]["content"]
        assert len(drains) == 1
        assert await _settle(lambda: not controller.fleet_wake.has_pending(session.id))
    finally:
        child_gateway.release_all()
        await asyncio.to_thread(_join_fleet_threads)
        controller._disposed = True
        for task in tuple(controller.fleet_wake._delivery_tasks):
            task.cancel()
        await asyncio.gather(
            *controller.fleet_wake._delivery_tasks, return_exceptions=True
        )
        db.close()


@pytest.mark.asyncio
async def test_lowered_live_allowance_stops_active_wake_without_refreshing_chain(
    rig, monkeypatch
):
    controller, db, session, gateway = rig[7], rig[2], rig[4], rig[5]
    gateway.stream_gate = asyncio.Event()
    chain = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    try:
        child = result_for(rig, chain)
        queue_result(rig, child)
        assert await _settle(lambda: gateway.payloads)
        monkeypatch.setenv("TLDW_AGENTS_MAX_AUTOWAKE_GENERATIONS", "0")
        assert await _settle(
            lambda: not controller.fleet_wake._delivery_tasks, seconds=3
        )
        assert db.automatic_work.snapshot(chain).pause_reason == "generation_budget"
        assert controller.in_flight_run_count() == 0
        monkeypatch.setenv("TLDW_AGENTS_MAX_AUTOWAKE_GENERATIONS", "3")
        controller.fleet_wake.retry_soon()
        await asyncio.sleep(0.35)
        assert len(gateway.payloads) == 1
        assert controller.fleet_wake.has_pending(session.id)
    finally:
        gateway.stream_gate.set()
        await close_rig(rig)


@pytest.mark.asyncio
async def test_automatic_child_and_followup_share_real_provider_allowance(rig):
    import threading

    from Tests.Agents.test_agent_service import SUBAGENT_PROMPT_PREFIX
    from Tests.Chat.test_console_agent_swap import _fence

    child_entered = threading.Event()
    child_release = threading.Event()
    parent_calls = 0

    def provider(**kwargs):
        nonlocal parent_calls
        if SUBAGENT_PROMPT_PREFIX in str(kwargs.get("system_message", "")):
            child_entered.set()
            assert child_release.wait(5)
            text = "automatic child completed"
        else:
            parent_calls += 1
            text = (
                _fence("spawn_subagent", {"task": "Read the saved result"})
                if parent_calls == 1
                else "parent completed"
            )
        data = response()
        data["choices"][0]["message"]["content"] = text
        return data

    gateway, calls = real_gateway(rig, agent=True, provider=provider)
    db, session, controller = rig[2], rig[4], rig[7]
    chain = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    try:
        source = result_for(rig, chain)
        queue_result(rig, source)
        assert await _settle(child_entered.is_set)
        assert await _settle(
            lambda: db.get_run(source)["wake_delivered_at"] is not None
        )
        assert db.automatic_work.snapshot(chain).used["child_launch"] == 1
        child_release.set()
        assert await _settle(
            lambda: db.automatic_work.snapshot(chain).used["generation"] == 2
        )
        assert await _settle(lambda: not controller.fleet_wake._delivery_tasks)
        snapshot = db.automatic_work.snapshot(chain)
        assert len(calls) == snapshot.used["model_call"] == 4
        assert snapshot.used["tokens"] == 40
        assert snapshot.reserved["tokens"] == 0
        assert {identity for _, identity in calls} == {chain}
    finally:
        child_release.set()
        await close_rig(rig)
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("reported_status", ["stuck", "done"])
async def test_stuck_child_does_not_pause_chain_or_block_done_sibling(rig, reported_status):
    from Tests.Chat.test_console_fleet_wake import _drain, _survivor, _terminal_subagent_run
    from tldw_chatbook.Chat.console_fleet_attention import (
        set_fleet_unseen_completion,
        fleet_unseen_conversation_ids,
    )

    from types import SimpleNamespace

    _, app, db, _, session, _, _, controller = rig
    chain = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    _, stuck = _terminal_subagent_run(db, session.id, status="stuck", work_chain_id=chain)
    wake = controller.fleet_wake
    leases = []
    wake.buddy_sink = SimpleNamespace(
        wake=lambda cid, rid, *, active: leases.append((cid, rid, active))
    )
    wake.capture_loop_if_running()
    assert set_fleet_unseen_completion(app, session.id)
    assert session.id in fleet_unseen_conversation_ids(app)
    try:
        wake.on_fleet_drained(_drain(session.id, _survivor(stuck, status=reported_status)))
        assert await _settle(lambda: not wake.has_pending(session.id))
        assert db.automatic_work.snapshot(chain).status == "active"
        assert (session.id, stuck, False) in leases
        assert await _settle(
            lambda: session.id not in fleet_unseen_conversation_ids(app)
        )
        done = result_for(rig, chain)
        queue_result(rig, done)
        assert await _settle(lambda: db.get_run(done)["wake_delivered_at"] is not None)
        assert db.get_run(stuck)["wake_delivered_at"] is None
    finally:
        await close_rig(rig)
