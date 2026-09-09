"""One startup audit, exact result review, and explicit clean resume."""

import pytest

from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_console_goal_scheduling import progress
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture

stores = _stores_fixture


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
async def test_stable_nonactivating_hydration_reads_real_tree(stores, monkeypatch):
    from types import SimpleNamespace

    from tldw_chatbook.Chat.chat_conversation_scope_service import (
        ChatConversationScopeService,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

    goal, store, session, _, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, lambda **kw: progress(1)
    )
    try:
        result = await co.dispatch_once(goal.id)
        co.service.checkpoint(result)
        stored_messages = stores[1].db.get_messages_for_conversation(
            goal.conversation_id
        )
        assert stored_messages
        store.close_session(session.id)
        other = store.create_session(workspace_id="workspace")
        app = SimpleNamespace(
            chachanotes_db=stores[1].db,
            chat_conversation_scope_service=ChatConversationScopeService(
                local_service=ChatConversationService(stores[1].db), server_service=None
            ),
        )
        assert hasattr(co, "restore_session"), "stable hydration entry point missing"
        restored = await co.restore_session(goal.id, app=app)
        assert restored.id == restored.persisted_conversation_id == goal.conversation_id
        assert store.active_session_id == other.id
        assert any(
            "draft 1" in m.content for m in store.messages_for_session(restored.id)
        )
        assert await co.restore_session(goal.id, app=app) is restored
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
        controller.fleet_wake.start_recovery()
        saved = await co.start(goal.id)
        assert saved.status == "recovery_required"
        assert len(calls) == 1
    finally:
        await gateway.aclose()
