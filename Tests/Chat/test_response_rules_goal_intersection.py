"""Rules and Goal generations retain one current automatic authority owner."""

import pytest

from Tests.Chat.response_rules_fixtures import source
from Tests.Chat.response_rules_store_fixtures import learning
from Tests.Chat.test_automatic_provider_budget import response
from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture
from tldw_chatbook.Chat.response_rules.models import RuleScope

stores = _stores_fixture

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Evidence: finished", "finished"])
async def test_goal_with_active_global_rule_keeps_committed_native_identity(
    stores, monkeypatch, text
):
    def generate(**kwargs):
        result = response()
        result["choices"][0]["message"]["content"] = text
        return result

    goal, chats, session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider=generate
    )
    runtime = controller._hooks_v2_runtime
    rules = runtime.ensure_response_rules()
    # Durable global activation is independent of the current Goal transcript.
    learned = learning(source(profile_id=rules.profile_id, conversation_id=None))
    rules.store.activate(
        learned.rule,
        learned.validation,
        RuleScope("global", rules.profile_id),
        expected_binding_revision=0,
    )
    from tldw_chatbook.Chat.console_trace_maintenance import (
        LegacyTraceMaintenance,
        TraceGarbageCollector,
    )
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode

    native_reply = controller._agent_bridge.run_reply
    collected_handoffs = []

    def collect_before_native_reply(*args, **kwargs):
        # Goal's existing automatic path deliberately uses Capture Off and
        # its genuine SYSTEM handoff, not a durable USER dispatch checkpoint.
        assert kwargs["capture_mode"] is ConsoleTraceCaptureMode.CAPTURE_OFF
        assert kwargs["trace_request"] is None
        database = stores[1].db
        from tldw_chatbook.DB.base_db import operation_owned_connection

        with operation_owned_connection(database):
            rows = (
                database.get_connection()
                .execute(
                    "SELECT id,role,content FROM messages WHERE conversation_id=? AND role='system'",
                    (goal.conversation_id,),
                )
                .fetchall()
            )
            assert rows
            assert (
                database.get_connection()
                .execute(
                    "SELECT COUNT(*) FROM console_dispatch_checkpoints WHERE conversation_id=?",
                    (goal.conversation_id,),
                )
                .fetchone()[0]
                == 0
            )
            assert LegacyTraceMaintenance(database).run_batch().logical_complete
            assert (
                TraceGarbageCollector(database)
                .collect(request_id="goal-before-provider")
                .status
                == "completed"
            )
            for row in rows:
                assert tuple(
                    database.get_connection()
                    .execute(
                        "SELECT id,role,content FROM messages WHERE id=?", (row[0],)
                    )
                    .fetchone()
                ) == tuple(row)
            collected_handoffs.extend(tuple(row) for row in rows)
        return native_reply(*args, **kwargs)

    monkeypatch.setattr(
        controller._agent_bridge, "run_reply", collect_before_native_reply
    )
    observed = []
    assess = rules.evaluator.assess

    async def observed_assessment(*args, **kwargs):
        pinned = rules._goal_inputs[session.id]
        assert args[1].request_text == pinned.request.draft.strip()
        observed.append(pinned)
        return await assess(*args, **kwargs)

    rules.evaluator.assess = observed_assessment
    try:
        result = await coordinator.dispatch_once(goal.id)
        assert result.native_run_id and result.outcome.status == "done", result
        assert stores[0].get_run(result.native_run_id)["work_chain_id"] == goal.chain_id
        state = rules.state(session.id)
        assert state.assessment is not None, state
        assert state.assessment.outcome == (
            "pass" if "Evidence" in text else "violation"
        )
        snapshot = stores[0].automatic_work.snapshot(goal.chain_id)
        assert snapshot.used["generation"] == len(calls)
        assert snapshot.reserved["generation"] == 0
        assert len(calls) == 1 and len(observed) == 1 and collected_handoffs
        assert rules.helpers.unsettled_count == 0
        assert rules._goal_inputs == {} and not rules.acceptance_current(
            state.assessment.source
        )
        with pytest.raises(ValueError, match="rule_goal_authority_unavailable"):
            observed[0].validate(chats)
        assert state.reason == (
            "" if "Evidence" in text else "goal_repair_requires_new_authority"
        )
        with stores[0].connection() as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM goal_iterations WHERE goal_id=?", (goal.id,)
                ).fetchone()[0]
                == 1
            )
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM agent_runs WHERE work_chain_id=?",
                    (goal.chain_id,),
                ).fetchone()[0]
                == 1
            )
        with stores[1].db.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM console_machine_followup_receipts"
                ).fetchone()[0]
                == 0
            )
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["handoff", "stop", "fake"])
async def test_goal_assessment_rechecks_handoff_and_current_capability(
    stores, monkeypatch, change
):
    from dataclasses import replace
    from types import SimpleNamespace

    from tldw_chatbook.Chat.response_rules.evidence import capture_rule_input
    from tldw_chatbook.Chat.response_rules.goal_input import pin_goal_input

    goal, chats, session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    rules = controller._hooks_v2_runtime.ensure_response_rules()
    learned = learning(source(profile_id=rules.profile_id, conversation_id=None))
    rules.store.activate(
        learned.rule,
        learned.validation,
        RuleScope("global", rules.profile_id),
        expected_binding_revision=0,
    )
    assess = rules.evaluator.assess

    async def changed(*args, **kwargs):
        pinned = rules._goal_inputs[session.id]
        if change == "handoff":
            chats.update_message_content(pinned.task_message_id, "Changed handoff")
        elif change == "stop":
            stopped = coordinator.service.stop(goal.id)
            coordinator._apply_control(stopped)
        else:
            fake = SimpleNamespace(**vars(pinned.authorization))
            assert (
                pin_goal_input(controller, chats.active_leaf(session.id), None, fake)
                is None
            )
            with pytest.raises(ValueError, match="rule_goal_authority_unavailable"):
                capture_rule_input(
                    chats, args[0], goal_input=replace(pinned, authorization=fake)
                )
            rules._goal_inputs[session.id] = replace(pinned, authorization=fake)
        return await assess(*args, **kwargs)

    rules.evaluator.assess = changed
    try:
        await coordinator.dispatch_once(goal.id)
        assert len(calls) == 1
        assert rules.state(session.id).assessment is None
        assert rules.helpers.unsettled_count == 0
        assert (
            stores[0].automatic_work.snapshot(goal.chain_id).reserved["generation"] == 0
        )
        with stores[1].db.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM console_machine_followup_receipts"
                ).fetchone()[0]
                == 0
            )
    finally:
        await gateway.aclose()
