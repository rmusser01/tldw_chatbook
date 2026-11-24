"""One startup audit, exact result review, and explicit clean resume."""

import asyncio
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.bootstrap_profile

from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_console_goal_scheduling import progress
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture
from Tests.UI.test_console_shared_context_policy import _screen
from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.console_goal_runs import ConsoleGoalCoordinator
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

stores = _stores_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("drain_times_out", [False, True])
async def test_reopened_goal_chat_can_send_and_close(
    tmp_path, monkeypatch, drain_times_out
):
    database, store, controller, screen, tasks = _screen(tmp_path)
    app = screen.app_instance
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=ChatConversationService(database), server_service=None
    )
    old = store.sessions()[0]
    conversation_id = old.persisted_conversation_id
    runtime = ConsoleRuntime(app)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    goal = SimpleNamespace(
        conversation_id=conversation_id,
        request=SimpleNamespace(
            provider=SimpleNamespace(provider="llama_cpp", model="test-model"),
            binding=SimpleNamespace(workspace_id=old.workspace_id),
        ),
    )
    coordinator = ConsoleGoalCoordinator(
        controller,
        SimpleNamespace(db=SimpleNamespace(automatic_work=None), get=lambda _: goal),
    )
    release_late_callback = asyncio.Event()
    cancel_observed = asyncio.Event()
    late = None

    async def late_callback():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancel_observed.set()
            await release_late_callback.wait()
            with pytest.raises(RuntimeError, match="session is closed"):
                runtime._raise_if_disposed_or_session_fenced(session.id)

    try:
        await runtime.close_session(
            old.id,
            expected_revision=controller.lifecycle_impact(session_id=old.id).revision,
        )
        session = await coordinator.restore_session("goal", app=app)
        # Goals first hydrate using the conversation id as the live id.
        assert session.id == conversation_id
        impact = controller.lifecycle_impact(session_id=session.id)
        if drain_times_out:
            late = asyncio.create_task(late_callback())
            await asyncio.sleep(0)
            shutdown_tasks = controller.session_shutdown_tasks
            monkeypatch.setattr(
                controller,
                "session_shutdown_tasks",
                lambda session_id: (
                    (late, *shutdown_tasks(session_id))
                    if session_id == session.id
                    else shutdown_tasks(session_id)
                ),
            )
        await runtime.close_session(
            session.id, expected_revision=impact.revision, timeout_seconds=0.01
        )
        reopened = await coordinator.restore_session("goal", app=app)
        runtime._raise_if_disposed_or_session_fenced(reopened.id)
        assert reopened.id != session.id
        assert reopened.persisted_conversation_id == conversation_id
        assert await coordinator.restore_session("goal", app=app) is reopened
        assert store.messages_for_session(reopened.id)[0].content == "hello"
        result = await controller.submit_draft("after reopen", session_id=reopened.id)
        assert result.accepted and result.provider_started, result
        if late is not None:
            assert cancel_observed.is_set()
            release_late_callback.set()
            await late
        await runtime.close_session(
            reopened.id,
            expected_revision=controller.lifecycle_impact(
                session_id=reopened.id
            ).revision,
        )
        assert reopened.id not in {open_session.id for open_session in store.sessions()}
    finally:
        release_late_callback.set()
        if late is not None:
            await asyncio.gather(late, return_exceptions=True)
        await runtime.dispose()
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_shared_audit_projects_only_covered_unfinished_goals(stores, monkeypatch):
    goal, _, _, _controller, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, lambda **kw: progress(1)
    )
    try:
        result = await co.dispatch_once(goal.id)
        clean = co.service.checkpoint(result)
        assert clean.status == "ready"
        ledger = stores[0].automatic_work
        ledger.recover(current_owner_id="replacement")
        assert hasattr(ledger, "recovery_result"), "typed startup audit missing"
        late = co.service.create(goal.request, launch_id="post-audit")
        projected = co.service.project_recovery(ledger.recovery_result)
        paused = co.service.get(goal.id)
        assert paused.status == "paused" and paused.pause_reason == "restart_checkpoint"
        assert co.service.get(late.id).status == "ready"
        assert co.service.project_recovery(ledger.recovery_result) == projected
        resumed = co.service.resume(goal.id, expected_revision=paused.revision)
        assert resumed.status == "ready"
        assert resumed.accounting.deadline_at == clean.accounting.deadline_at
        assert resumed.accounting.used == clean.accounting.used
        assert len(calls) == 1
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_interrupted_goal_cannot_be_quality_approved_or_replayed(
    stores, monkeypatch
):
    goal, _, _, _, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, lambda **kw: progress(1)
    )
    try:
        result = await co.dispatch_once(goal.id)
        ledger = stores[0].automatic_work
        ledger.recover(current_owner_id="replacement")
        assert hasattr(co.service, "project_recovery"), "recovery projection missing"
        co.service.project_recovery(ledger.recovery_result)
        interrupted = co.service.get(goal.id)
        assert interrupted.status == "recovery_required"
        with pytest.raises(ValueError, match="result_review_unavailable"):
            co.service.review_result(
                goal.id,
                expected_revision=interrupted.revision,
                checkpoint_id=result.attempt_id,
                artifact_digest="0" * 64,
                accepted=True,
            )
        with pytest.raises(ValueError, match="resume_requires_clean_checkpoint"):
            co.service.resume(goal.id, expected_revision=interrupted.revision)
        from tldw_chatbook.Agents.goal_models import RecoveryResolution

        closed = co.service.resolve_recovery(
            goal.id,
            expected_revision=interrupted.revision,
            resolution=RecoveryResolution.CLOSE_UNCERTAIN,
        )
        assert closed.status == "closed"
        assert closed.accounting.used == interrupted.accounting.used
        assert closed.accounting.reserved == interrupted.accounting.reserved
        with pytest.raises(ValueError, match="settled"):
            co.service.remove_payloads(goal.id)
        assert len(calls) == 1
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_close", [False, True])
async def test_stable_nonactivating_hydration_reads_real_tree(
    stores, monkeypatch, commit_close
):
    goal, store, session, controller, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, lambda **kw: progress(1)
    )
    runtime = stores[0]._goal_test_runtimes[-1]
    try:
        result = await co.dispatch_once(goal.id)
        co.service.checkpoint(result)
        stored_messages = stores[1].db.get_messages_for_conversation(
            goal.conversation_id
        )
        assert stored_messages
        if commit_close:
            await runtime.close_session(
                session.id,
                expected_revision=controller.lifecycle_impact(
                    session_id=session.id
                ).revision,
            )
        else:
            store.close_session(session.id)
        other = store.create_session(workspace_id="workspace")
        app = SimpleNamespace(
            chachanotes_db=stores[1].db,
            chat_conversation_scope_service=ChatConversationScopeService(
                local_service=ChatConversationService(stores[1].db), server_service=None
            ),
        )
        assert hasattr(co, "restore_session"), "stable hydration entry point missing"
        if commit_close:
            paused = co.service.pause(goal.id)
            await co.resume(goal.id, expected_revision=paused.revision, app=app)
            settled = await co._runner
            assert settled.iteration_count > paused.iteration_count, (
                settled.pause_reason
            )

        restored = await co.restore_session(goal.id, app=app)
        assert restored.persisted_conversation_id == goal.conversation_id
        assert (restored.id != session.id) is commit_close
        assert store.active_session_id == other.id
        assert any(
            "draft 1" in m.content for m in store.messages_for_session(restored.id)
        )
        assert await co.restore_session(goal.id, app=app) is restored
        if commit_close:
            with stores[0].connection() as connection:
                next_attempt = connection.execute(
                    "SELECT attempt_id, run_id FROM goal_iterations WHERE goal_id=? AND ordinal=2",
                    (goal.id,),
                ).fetchone()
            assert next_attempt["run_id"]
            attempt = co.ledger.read_goal_attempt(
                next_attempt["attempt_id"], owner_id=co._owner_id
            )
            assert attempt.session_id == restored.id
            assert attempt.conversation_id == goal.conversation_id
            assert len(calls) > 1
        else:
            assert len(calls) == 1
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_start_cannot_demote_shared_audit_interruption(stores, monkeypatch):
    goal, _, _, controller, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, lambda **kw: progress(1)
    )
    try:
        await co.dispatch_once(goal.id)
        controller.fleet_wake._owner_id = "replacement"
        co._owner_id = "replacement"
        # The current runtime audits once at construction; explicitly model
        # the replacement owner audit instead of re-requesting that same audit.
        await controller.fleet_wake.recover()
        saved = await co.start(goal.id)
        assert saved.status == "recovery_required"
        assert len(calls) == 1
    finally:
        await gateway.aclose()
