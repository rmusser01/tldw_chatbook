"""Real native provider requests require committed goal authority and fresh history."""

import importlib
import sqlite3

import pytest

from Tests.Chat.test_goal_conversation_provisioning import service
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture

stores = _stores_fixture
from Tests.Chat.test_automatic_provider_budget import resolution, response
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway


def goal_module():
    spec = importlib.util.find_spec("tldw_chatbook.Chat.console_goal_runs")
    assert spec is not None, "native goal coordinator missing"
    return importlib.import_module(spec.name)


def build_goal_rig(stores, monkeypatch, provider=None):
    mod = goal_module()
    runs, persistence, registry, req = stores
    monkeypatch.setenv("TLDW_AGENTS_GOAL_RUNS_ENABLED", "true")
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return provider(**kwargs) if provider else response()

    gateway = ConsoleProviderGateway(chat_api_call_fn=generate)

    async def resolve(_selection):
        return resolution()

    gateway.resolve_for_send = resolve
    req = req.model_copy(update={"provider": mod.goal_provider_ref(resolution())})
    owner = service((runs, persistence, registry, req))
    goal = owner.create(req, launch_id="start")
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(
        session_id=goal.conversation_id,
        workspace_id="workspace",
        project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
    )
    session.persisted_conversation_id = goal.conversation_id
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs, store=store, provider_gateway=gateway
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    coordinator = mod.ConsoleGoalCoordinator(controller, owner)
    controller._goal_coordinator = coordinator
    return goal, store, session, controller, coordinator, gateway, calls


@pytest.mark.asyncio
async def test_goal_native_dispatch_is_automatic_fresh_and_returns_exact_native_outcome(
    stores, monkeypatch
):
    goal, store, session, _controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="OLD PRIVATE TRANSCRIPT"
    )
    try:
        result = await coordinator.dispatch_once(goal.id)
        assert result.native_run_id and result.outcome.status == "done", result
        assert result.outcome.final_text == "finished"
        assert result.goal_id == goal.id and result.ordinal == 1
        assert len(calls) == 1
        assert "OLD PRIVATE TRANSCRIPT" not in str(calls[0])
        assert "Repair fixture" in str(calls[0])
        assert stores[0].get_run(result.native_run_id)["work_chain_id"] == goal.chain_id
        assert stores[0].automatic_work.snapshot(goal.chain_id).used["generation"] == 1
        assert stores[0].goal_runs.get(goal.id).status != "completed"
        rows = store.messages_for_session(session.id)
        assert any(
            getattr(m.metadata, "origin", None) == "goal_iteration" for m in rows
        )
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_completed", [False, True])
async def test_failed_acceptance_never_calls_preparation_or_provider(
    stores, monkeypatch, commit_completed
):
    goal, _store, _session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    helpers = []
    original = controller._apply_skill_substitution

    async def preparation(*args, **kwargs):
        helpers.append(True)
        return await original(*args, **kwargs)

    controller._apply_skill_substitution = preparation

    original_accept = stores[0].automatic_work.accept_goal_iteration

    def fail(*args, **kwargs):
        if commit_completed:
            original_accept(*args, **kwargs)
        raise sqlite3.OperationalError("injected commit failure")

    monkeypatch.setattr(stores[0].automatic_work, "accept_goal_iteration", fail)
    try:
        result = await coordinator.dispatch_once(goal.id)
        assert result.native_run_id is None
        assert calls == [] and helpers == []
        assert controller.in_flight_run_count() == 0
        snapshot = stores[0].automatic_work.snapshot(goal.chain_id)
        assert snapshot.reserved["generation"] == 0
        assert snapshot.used["generation"] == int(commit_completed)
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_cancellation_during_preparation_retains_primary_until_own_cleanup(
    stores, monkeypatch
):
    import asyncio

    goal, _store, _session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = controller._apply_skill_substitution

    async def blocked(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    controller._apply_skill_substitution = blocked
    task = asyncio.create_task(coordinator.dispatch_once(goal.id))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert controller.in_flight_run_count() == 1
        release.set()
        result = await task
        assert calls == []
        assert controller.in_flight_run_count() == 0
        assert result.native_run_id is None or result.outcome.status == "cancelled"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await gateway.aclose()


@pytest.mark.asyncio
async def test_goal_rejects_plain_route_before_acceptance(stores, monkeypatch):
    goal, _store, _session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    controller._agent_runtime_enabled = False
    try:
        result = await coordinator.dispatch_once(goal.id)
        assert calls == []
        assert result.reason_code == "native_goal_required"
        snapshot = stores[0].automatic_work.snapshot(goal.chain_id)
        assert snapshot.used["generation"] == 0 and snapshot.reserved["generation"] == 0
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_both_coordinators_wait_for_one_runtime_audit(stores, monkeypatch):
    import asyncio
    import threading

    goal, _store, _session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    entered, release = threading.Event(), threading.Event()
    recover = stores[0].automatic_work.recover
    audits = []

    def delayed(**kwargs):
        audits.append(kwargs["current_owner_id"])
        entered.set()
        release.wait(3)
        return recover(**kwargs)

    monkeypatch.setattr(stores[0].automatic_work, "recover", delayed)
    controller.fleet_wake.start_recovery()
    task = asyncio.create_task(coordinator.dispatch_once(goal.id))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        await asyncio.sleep(0.02)
        assert calls == [] and not task.done()
        release.set()
        result = await task
        assert result.native_run_id
        assert audits == [coordinator._owner_id]
        assert await controller.fleet_wake.wait_for_recovery()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await gateway.aclose()


@pytest.mark.asyncio
async def test_iteration_budget_has_typed_reason_and_never_becomes_manual(
    stores, monkeypatch
):
    import json

    from tldw_chatbook.Agents.goal_models import GoalToolScope

    runs, persistence, registry, req = stores
    req = req.model_copy(
        update={
            "policy": req.policy.model_copy(update={"iteration_model_turns": 1}),
            "tool_scope": GoalToolScope(
                catalog_tools=("builtin:get_current_datetime",),
                runtime_tools=("find_tools",),
            ),
        }
    )

    def provider(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "find",
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
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }

    goal, _store, _session, _controller, coordinator, gateway, calls = build_goal_rig(
        (runs, persistence, registry, req), monkeypatch, provider
    )
    try:
        result = await coordinator.dispatch_once(goal.id)
        from enum import Enum

        assert isinstance(result.termination_reason, Enum), (
            "termination reason must be typed"
        )
        assert result.termination_reason.value == "model_turn_limit"
        assert len(calls) == 1
        assert runs.automatic_work.snapshot(goal.chain_id).used["model_call"] == 1
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_failed_startup_audit_is_shared_and_never_allows_goal_calls(
    stores, monkeypatch
):
    goal, _store, _session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    audits = []

    def failed(**kwargs):
        audits.append(kwargs["current_owner_id"])
        raise sqlite3.OperationalError("startup failure")

    monkeypatch.setattr(stores[0].automatic_work, "recover", failed)
    controller.fleet_wake.start_recovery()
    try:
        result = await coordinator.dispatch_once(goal.id)
        assert not await controller.fleet_wake.wait_for_recovery()
        assert result.reason_code == "history_unavailable"
        assert calls == []
        assert len(audits) == 1
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_lazy_goal_service_attachment_does_not_revoke_live_fleet_owner(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from Tests.Chat.test_automatic_wake_budget import (
        close_rig,
        queue_result,
        result_for,
    )
    from Tests.Chat.test_console_fleet_wake import _controller_rig, _settle
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    rig = _controller_rig(tmp_path)
    controller, ledger, gateway = rig[7], rig[2].automatic_work, rig[5]
    gateway.stream_gate = asyncio.Event()
    controller.fleet_wake.start_recovery()
    assert await controller.fleet_wake.wait_for_recovery()
    chain = ledger.create_chain(rig[4].id, root_submission_id="manual")
    queue_result(rig, result_for(rig, chain))
    try:
        assert await _settle(lambda: gateway.payloads)
        rig[3].persistence = ChatPersistenceService(rig[0])
        runtime = ConsoleRuntime(SimpleNamespace())
        runtime._chat_controller = controller
        goal_coordinator = runtime.ensure_goal_coordinator()
        assert goal_coordinator._owner_id == controller.fleet_wake._owner_id
        assert (
            ledger.check_active(chain, owner_id=goal_coordinator._owner_id).status
            == "active"
        )
        assert runtime.ensure_goal_coordinator() is goal_coordinator
    finally:
        rig[3].persistence = None
        gateway.stream_gate.set()
        await close_rig(rig)


@pytest.mark.asyncio
async def test_repeated_start_cannot_reprovision_an_accepted_attempt(
    stores, monkeypatch
):
    import asyncio
    import threading

    entered, release = threading.Event(), threading.Event()

    def provider(**kwargs):
        entered.set()
        assert release.wait(3)
        return response()

    goal, _store, _session, _controller, coordinator, gateway, _calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    task = asyncio.create_task(coordinator.dispatch_once(goal.id))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        monkeypatch.setattr(
            coordinator.service.persistence,
            "provision_goal_conversation",
            lambda intent: pytest.fail("accepted launch must not run setup again"),
        )
        again = coordinator.service.create(goal.request, launch_id=goal.launch_id)
        assert again.status == goal.status
        assert again.iteration_count == 1
        assert coordinator._active.context.check().used["generation"] == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await gateway.aclose()


@pytest.mark.asyncio
async def test_goal_authorization_cannot_be_forged_or_moved_to_another_session(
    stores, monkeypatch
):
    import asyncio
    import threading

    from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin

    entered, release = threading.Event(), threading.Event()

    def provider(**kwargs):
        entered.set()
        assert release.wait(3)
        return response()

    goal, store, session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    try:
        with pytest.raises(PermissionError):
            await controller.submit_draft(
                "untrusted",
                session_id=session.id,
                origin=ConsoleSubmissionOrigin.GOAL_ITERATION,
                goal_authorization=object(),
            )
        assert calls == []
        task = asyncio.create_task(coordinator.dispatch_once(goal.id))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            other = store.create_session(workspace_id="workspace")
            other.persisted_conversation_id = goal.conversation_id
            blocked = await controller.submit_draft("manual alias", session_id=other.id)
            assert not blocked.accepted and len(calls) == 1
            with pytest.raises(PermissionError):
                await controller.submit_draft(
                    "retarget",
                    session_id=other.id,
                    origin=ConsoleSubmissionOrigin.GOAL_ITERATION,
                    goal_authorization=coordinator._active,
                )
            assert len(calls) == 1
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
    finally:
        await gateway.aclose()
