"""Native runtime checks over actual Console generation and queue owners."""

import asyncio
from dataclasses import replace

import pytest

from Tests.Chat.response_rules_fixtures import source
from Tests.Chat.response_rules_store_fixtures import learning
from Tests.Chat.test_console_fleet_wake import _controller_rig
from Tests.Chat.test_response_rules_builder import Transport
from Tests.Chat.test_console_provider_gateway import _auxiliary_resolution
from Tests.console_provider_doubles import with_destination
from Tests.conftest import _close_database_instance
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
from tldw_chatbook.Chat.response_rules.models import RuleRuntimeState

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.fixture
async def native(tmp_path):
    db, app, runs, chats, session, gateway, _bridge, controller = _controller_rig(
        tmp_path
    )
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(chats)
    runtime.set_chat_controller(controller)
    rules = runtime.ensure_response_rules()
    transport = Transport()
    gateway.complete_auxiliary = transport.complete_auxiliary

    async def resolve(_selection):
        return with_destination(
            _auxiliary_resolution(
                provider="llama_cpp",
                execution_key="llama_cpp",
                readiness_key="llama_cpp",
                base_url=None,
                api_key=None,
                model="test-model",
            )
        )

    gateway.resolve_for_send = resolve
    try:
        yield runtime, rules, chats, session, gateway, controller
    finally:
        await runtime.dispose()
        runs.close()
        _close_database_instance(db)


def seed(native):
    _runtime, rules, chats, session, _gateway, _controller = native
    user = chats.append_message(
        session.id, role=ConsoleMessageRole.USER, content="Explain result"
    )
    chats.persist_message_if_needed(user.id)
    answer = chats.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Missing proof"
    )
    answer = chats.persist_message_if_needed(answer.id)
    return answer


@pytest.mark.asyncio
async def test_profile_rules_exist_before_console_controller_is_constructed(native):
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    old_owner, _rules, _chats, _session, _gateway, _controller = native
    owner = ConsoleRuntime(app=old_owner._app)
    try:
        rules = owner.ensure_response_rules()
        assert rules is not None
        assert owner._chat_controller is None
        assert not rules.chat_store.sessions()
        # Later real assembly binds the actual execution owner, not a Settings stub.
        controller = ConsoleChatController(
            store=rules.chat_store, provider_gateway=owner.ensure_provider_gateway()
        )
        owner.set_chat_controller(controller)
        assert owner.ensure_response_rules() is rules
        assert rules.controller is controller
    finally:
        await owner.dispose()


def activate(native):
    _runtime, rules, chats, session, _gateway, _controller = native
    answer = seed(native)
    origin = source(
        profile_id=rules.profile_id,
        session_id=session.id,
        conversation_id=session.persisted_conversation_id,
        message_id=answer.persisted_message_id,
        branch_id=answer.id,
        message_version=chats.response_rule_source_version(answer.id),
    )
    result = learning(origin)
    rules.store.activate(
        result.rule,
        result.validation,
        rules.scopes(session.id)[0],
        expected_binding_revision=0,
    )
    return answer


def request(native, text="Explain another result"):
    _runtime, _rules, _chats, session, _gateway, controller = native
    return ConsoleTurnCustodyRequest(
        turn_id="manual-test-turn",
        session_id=session.id,
        draft=text,
        configuration=controller.resolve_turn_configuration_snapshot(session.id),
    )


@pytest.mark.asyncio
async def test_completed_response_checked_before_completion_and_queue_drain(native):
    runtime, rules, chats, session, gateway, controller = native
    activate(native)
    gateway.reply = "Evidence: this is a scripted result"
    events = []
    chats.subscribe_message_completed(lambda _: events.append("completion_released"))
    original = rules.evaluator.assess
    entered, release = asyncio.Event(), asyncio.Event()

    async def checked(*args, **kwargs):
        entered.set()
        await release.wait()
        result = await original(*args, **kwargs)
        events.append("assessment_settled")
        return result

    rules.evaluator.assess = checked
    failures = []
    effect = controller._run_durable_postcommit_effect

    async def observed(*args, **kwargs):
        try:
            return await effect(*args, **kwargs)
        except Exception as exc:
            failures.append((args[1], type(exc).__name__, str(exc)))
            raise

    controller._run_durable_postcommit_effect = observed
    turn = runtime.accept_turn(request(native))
    task = runtime._turn_custody[turn].task
    entered_task = asyncio.create_task(entered.wait())
    async with asyncio.timeout(15):
        done, _ = await asyncio.wait(
            (task, entered_task), return_when=asyncio.FIRST_COMPLETED
        )
    if entered_task not in done:
        entered_task.cancel()
    assert entered_task in done, (task.result(), failures)
    assert events == [] and rules.state(session.id).phase == "checking"
    assert chats.get_message(chats.active_leaf(session.id)).status == "complete"
    release.set()
    result = await runtime.wait_for_turn(turn)
    assert result.accepted
    assert events == ["assessment_settled", "completion_released"]
    assert rules.state(session.id).assessment.outcome == "pass"


@pytest.mark.asyncio
async def test_real_agent_service_reply_uses_the_owned_rule_boundary(native):
    from Tests.Chat.test_console_agent_bridge import _ChunkGateway
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

    runtime, rules, chats, session, gateway, controller = native
    activate(native)
    scripts = _ChunkGateway([["Evidence: completed by the real agent runtime"]])
    gateway.stream_chat = scripts.stream_chat
    bridge = ConsoleAgentBridge(
        agent_runs_db=controller._agent_bridge.runs_db,
        store=chats,
        provider_gateway=gateway,
    )
    controller._agent_bridge = bridge
    controller._agent_runtime_enabled = True
    events = []
    chats.subscribe_message_completed(lambda _token: events.append("completion"))
    assess = rules.evaluator.assess

    async def observed(*args, **kwargs):
        result = await assess(*args, **kwargs)
        events.append("assessment")
        return result

    rules.evaluator.assess = observed
    failures = []
    effect = controller._run_durable_postcommit_effect

    async def observed_effect(*args, **kwargs):
        try:
            return await effect(*args, **kwargs)
        except Exception as exc:
            failures.append((args[1], type(exc).__name__, str(exc)))
            raise

    controller._run_durable_postcommit_effect = observed_effect
    result = await runtime.wait_for_turn(runtime.accept_turn(request(native)))
    assert result.accepted, result.visible_copy
    assert scripts.calls == 1, (
        failures,
        controller.run_state_for(session.id),
        chats.get_message(chats.active_leaf(session.id)),
    )
    assert chats.get_message(chats.active_leaf(session.id)).status == "complete"
    assert rules.state(session.id).assessment.outcome == "pass"
    assert events == ["assessment", "completion"]


@pytest.mark.asyncio
async def test_real_completed_tool_work_is_retained_during_native_repair(
    native, monkeypatch
):
    from Tests.Chat.test_console_agent_bridge import _ChunkGateway, _fence
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Tools.tool_executor import CalculatorTool

    runtime, rules, chats, session, gateway, controller = native
    activate(native)
    executions = []
    execute = CalculatorTool.execute

    async def observed(self, expression):
        executions.append(expression)
        return await execute(self, expression)

    monkeypatch.setattr(CalculatorTool, "execute", observed)
    scripts = _ChunkGateway(
        [
            [_fence("calculator", {"expression": "6*7"})],
            ["Missing proof"],
            ["Evidence: the retained calculator result is 42."],
        ]
    )
    gateway.stream_chat = scripts.stream_chat
    controller._agent_bridge = ConsoleAgentBridge(
        agent_runs_db=controller._agent_bridge.runs_db,
        store=chats,
        provider_gateway=gateway,
    )
    controller._agent_runtime_enabled = True
    result = await runtime.wait_for_turn(
        runtime.accept_turn(request(native, "Calculate 6*7 and explain result"))
    )
    assert result.accepted, result.visible_copy
    assert scripts.calls == 3
    messages = chats.read_only_messages_for_session(session.id)
    assert executions == ["6*7"], [
        m.content for m in messages if m.role is ConsoleMessageRole.TOOL
    ]
    assert sum(m.role is ConsoleMessageRole.TOOL for m in messages) == 1
    assert any(
        m.role is ConsoleMessageRole.ASSISTANT and m.content == "Missing proof"
        for m in messages
    )
    assert rules.state(session.id).assessment.outcome == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize("hooks", [True, False])
@pytest.mark.parametrize("model_cap", [1, 2])
async def test_real_agent_remaining_budget_limits_native_repairs(
    native, monkeypatch, hooks, model_cap
):
    from Tests.Chat.test_console_agent_bridge import _ChunkGateway
    from tldw_chatbook.Chat import console_agent_bridge

    runtime, rules, chats, session, gateway, controller = native
    activate(native)
    engine = runtime.ensure_hooks_v2(session.id, (), lambda *_: True) if hooks else None
    scripts = _ChunkGateway([["Missing proof"]] * 4)
    gateway.stream_chat = scripts.stream_chat
    bridge = console_agent_bridge.ConsoleAgentBridge(
        agent_runs_db=controller._agent_bridge.runs_db,
        store=chats,
        provider_gateway=gateway,
        get_hooks_v2=runtime.get_hooks_v2,
    )
    controller._agent_bridge = bridge
    controller._agent_runtime_enabled = True
    budget = console_agent_bridge.console_run_budget()
    monkeypatch.setattr(
        console_agent_bridge,
        "console_run_budget",
        lambda: replace(budget, max_model_turns=model_cap),
    )
    result = await runtime.wait_for_turn(runtime.accept_turn(request(native)))
    assert result.accepted, result.visible_copy
    assert rules.state(session.id).assessment.outcome == "violation"
    assert scripts.calls == model_cap
    if engine is not None:
        assert not engine.lifecycle_owner.inherited_budgets
        assert not engine.lifecycle_owner.terminal_budgets


@pytest.mark.asyncio
async def test_exhausted_native_repairs_report_the_correction_limit(native):
    runtime, rules, _chats, session, gateway, _controller = native
    activate(native)
    gateway.reply = "Missing proof"
    await runtime.wait_for_turn(runtime.accept_turn(request(native)))
    assert len(gateway.payloads) == 3
    assert rules.state(session.id).reason == "correction_limit"


@pytest.mark.asyncio
async def test_committed_rule_stays_active_if_initial_repair_configuration_is_unavailable(
    native, monkeypatch
):
    _owner, rules, _chats, session, gateway, controller = native
    seed(native)

    def unavailable(*args, **kwargs):
        raise RuntimeError("fixture local storage unavailable")

    monkeypatch.setattr(controller, "resolve_turn_configuration_snapshot", unavailable)
    result = await rules.learn(session.id, "The answer omitted evidence")
    assert rules.store.list_bindings(rules.scopes(session.id)[0])[0].state == "enabled"
    assert result.state == "active"
    assert result.reason == "initial_repair_unavailable"
    assert gateway.payloads == []


@pytest.mark.asyncio
async def test_rule_lookup_failure_does_not_block_ordinary_generation(
    native, monkeypatch
):
    runtime, rules, chats, session, gateway, controller = native
    seed(native)
    gateway.reply = "Evidence: the requested answer"

    def unavailable(_session):
        raise ValueError("too_many_effective_rules")

    monkeypatch.setattr(rules, "effective_rules", unavailable)
    turn = runtime.accept_turn(request(native))
    await runtime._turn_custody[turn].task
    assert gateway.payloads
    assert chats.get_message(chats.active_leaf(session.id)).status == "complete"
    assert rules.state(session.id).reason == "rules_unavailable"


@pytest.mark.asyncio
async def test_disabling_rule_clears_its_live_success_verdict(native):
    runtime, rules, _chats, session, gateway, _controller = native
    activate(native)
    gateway.reply = "Evidence: complete"
    await runtime.wait_for_turn(runtime.accept_turn(request(native)))
    assert rules.state(session.id).assessment.outcome == "pass"
    scope = rules.scopes(session.id)[0]
    binding = rules.store.list_bindings(scope)[0]
    rules.store.set_binding(
        replace(binding, state="disabled"),
        expected_binding_revision=binding.binding_revision,
    )
    await asyncio.sleep(0)
    assert rules.state(session.id).assessment is None
    assert rules.state(session.id).learning is None


@pytest.mark.asyncio
async def test_user_input_during_check_fences_repair(native):
    runtime, rules, chats, session, gateway, controller = native
    activate(native)
    gateway.reply = "Missing proof"
    entered, release = asyncio.Event(), asyncio.Event()
    original = rules.evaluator.assess

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    rules.evaluator.assess = delayed
    turn = runtime.accept_turn(request(native))
    async with asyncio.timeout(15):
        await entered.wait()
    pending = await controller.queue_prompt(
        session.id,
        text="User request wins",
        expected_revision=controller.prompt_queue_registry.snapshot(
            session.id
        ).revision,
    )
    assert pending.applied
    gateway.reply = "Evidence: next response"
    release.set()
    await runtime.wait_for_turn(turn)
    assert len(gateway.payloads) == 2, (
        rules.state(session.id),
        rules.queue.registry.snapshot(session.id),
    )
    assert gateway.payloads[-1][-1]["content"] == "User request wins"
    assert not controller.prompt_queue_coordinator._shared_machine_receipts


@pytest.mark.asyncio
async def test_stop_remains_available_after_generation_while_checking(native):
    runtime, rules, chats, session, gateway, controller = native
    activate(native)
    gateway.reply = "Missing proof"
    entered = asyncio.Event()

    async def held(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    rules.evaluator.assess = held
    turn = runtime.accept_turn(request(native))
    async with asyncio.timeout(15):
        await entered.wait()
    assert controller.is_stop_allowed and controller.stop_active_run()
    await runtime.wait_for_turn(turn)
    assert len(gateway.payloads) == 1
    assert chats.get_message(chats.active_leaf(session.id)).status == "complete"
    assert rules.state(session.id).reason == "stopped"
    assert rules.state(session.id).assessment.state == "cancelled"


@pytest.mark.asyncio
async def test_manual_learning_activates_then_repairs_with_fresh_operation(native):
    runtime, rules, chats, session, gateway, _controller = native
    gateway.reply = "Missing proof"
    turn = runtime.accept_turn(request(native, "Explain result"))
    await runtime.wait_for_turn(turn)
    gateway.reply = "Evidence: I cannot verify the work"
    initial = rules.queue.start_native_repair
    observed = []
    submit = rules.queue._submit_queued

    async def observed_submit(*args, **kwargs):
        result = await submit(*args, **kwargs)
        observed.append(
            ("submitted", result.accepted, result.visible_copy, result.provider_started)
        )
        return result

    rules.queue._submit_queued = observed_submit

    async def observed_initial(request, proposal):
        snap = rules.queue.registry.snapshot(session.id)
        observed.append(
            (
                "before",
                snap.reservation.value,
                snap.expected_context_epoch,
                rules.queue._continuation_admission_current(request),
                rules.current_assessment(proposal.source, proposal.assessment_id)
                is not None,
            )
        )
        admitted = await initial(request, proposal)
        observed.append(
            ("after", admitted, rules.queue.registry.snapshot(session.id).waiting_count)
        )
        return admitted

    rules.queue.start_native_repair = observed_initial
    phases = []
    _controller.response_rules_changed = lambda _session: phases.append(
        rules.state(session.id).phase
    )
    result = await rules.learn(session.id, "The answer omitted evidence")
    assert result.state == "active" and result.validation is not None, result
    assert len(gateway.payloads) == 2, observed
    assert result.rule.origin.operation_id != "manual-test-turn"
    assert rules.state(session.id).assessment.outcome == "pass"
    assert (
        phases.index("drafting") < phases.index("testing") < phases.index("repairing")
    )


@pytest.mark.asyncio
async def test_no_effective_rules_uses_no_evaluator_or_helper(native):
    runtime, rules, _chats, session, _gateway, _controller = native

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("no-rule path invoked an evaluator")

    rules.evaluator.assess = forbidden
    await runtime.wait_for_turn(runtime.accept_turn(request(native)))
    assert rules.helpers.unsettled_count == 0
    assert rules.state(session.id) == RuleRuntimeState("idle", None, None, "")


@pytest.mark.asyncio
async def test_learning_requires_an_eligible_response_and_refuses_live_generation(
    native,
):
    runtime, rules, _chats, session, gateway, _controller = native
    assert (
        await rules.learn(session.id, "Missing evidence")
    ).reason == "no_eligible_response"
    gateway.stream_gate = asyncio.Event()
    turn = runtime.accept_turn(request(native))
    async with asyncio.timeout(15):
        while not gateway.payloads:
            await asyncio.sleep(0.01)
    assert (await rules.learn(session.id, "Missing evidence")).reason == "active_run"
    gateway.stream_gate.set()
    await runtime.wait_for_turn(turn)


@pytest.mark.parametrize(
    "action", ["edit", "delete", "branch", "close", "workspace", "settings", "variant"]
)
@pytest.mark.asyncio
async def test_source_changes_revoke_late_learning_activation(native, action):
    _runtime, rules, chats, session, gateway, _controller = native
    answer = seed(native)
    if action == "variant":
        chats.add_variant(answer.id, "Missing proof variant")
    entered, release = asyncio.Event(), asyncio.Event()
    complete = gateway.complete_auxiliary

    async def held(*args, **kwargs):
        entered.set()
        await release.wait()
        return await complete(*args, **kwargs)

    gateway.complete_auxiliary = held
    task = asyncio.create_task(rules.learn(session.id, "The answer omitted evidence"))
    async with asyncio.timeout(10):
        await entered.wait()
    scope = rules.scopes(session.id)[0]
    if action == "edit":
        chats.update_message_content(answer.id, "Changed by user")
    elif action == "delete":
        chats.delete_message(answer.id)
    elif action == "branch":
        chats.set_active_leaf(session.id, chats.active_path_message_ids(session.id)[0])
    elif action == "close":
        chats.close_session(session.id)
    elif action == "settings":
        from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

        chats.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="changed-model"),
        )
    elif action == "variant":
        chats.select_variant(answer.id, 0)
    else:
        chats.set_workspace_context(
            ConsoleWorkspaceContext(active_workspace_id="new-workspace")
        )
    release.set()
    result = await task
    assert result.state in {"cancelled", "stale"}
    assert rules.store.list_bindings(scope) == ()
    assert rules.helpers.unsettled_count == 0


@pytest.mark.parametrize("route", ["continue", "retry", "regenerate"])
@pytest.mark.asyncio
async def test_other_reply_actions_check_before_completion(native, route):
    _runtime, rules, chats, session, gateway, controller = native
    answer = activate(native)
    gateway.reply = "Evidence: reviewed result"
    events = []
    chats.subscribe_message_completed(lambda _token: events.append("completion"))
    original = rules.evaluator.assess

    async def checked(*args, **kwargs):
        result = await original(*args, **kwargs)
        events.append("assessment")
        return result

    rules.evaluator.assess = checked
    persistence_errors = []
    save_assessment = rules.store.save_assessment

    def inspected_save(assessment, **kwargs):
        try:
            save_assessment(assessment, **kwargs)
        except Exception as exc:
            with rules.store.repository.db.transaction() as cursor:
                row = cursor.execute(
                    "SELECT version,sender,deleted FROM messages WHERE id=?",
                    (assessment.source.message_id,),
                ).fetchone()
            persistence_errors.append(
                (
                    type(exc).__name__,
                    str(exc),
                    assessment.source.message_version,
                    tuple(row) if row else None,
                )
            )
            raise

    rules.store.save_assessment = inspected_save
    if route == "retry":
        chats.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Explain result",
            persist=True,
        )
        failed = chats.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
        )
        chats.append_stream_chunk(failed.id, "Partial answer")
        chats.mark_message_failed(failed.id)
        failed = chats.persist_message_if_needed(failed.id)
        assert failed.persisted_message_id is not None, (failed.status, failed.content)
        result = await controller.retry_message(failed.id)
    elif route == "continue":
        result = await controller.continue_from_message(answer.id)
    else:
        result = await controller.regenerate_message(answer.id)
    assert result.accepted, result.visible_copy
    assert events == ["assessment", "completion"]
    assert rules.state(session.id).assessment is not None, (
        rules.state(session.id).reason,
        persistence_errors,
    )
    assert rules.state(session.id).assessment.outcome == "pass"


@pytest.mark.asyncio
async def test_feedback_edit_reuses_exact_predicate_calibration(native):
    runtime, rules, _chats, session, gateway, _controller = native
    gateway.reply = "Missing proof"
    await runtime.wait_for_turn(runtime.accept_turn(request(native, "Explain result")))
    gateway.reply = "Evidence: I cannot verify the work"
    learned = await rules.learn(session.id, "The answer omitted evidence")
    assert learned.state == "active", learned.reason

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("feedback-only edit repeated predicate calibration")

    rules.evaluator.validate_cases = forbidden
    candidate = learned.rule.candidate.model_copy(
        update={"feedback": "Explain the evidence clearly."}
    )
    edited = await rules.test_edit(
        session.id,
        rules.scopes(session.id)[0],
        learned.rule.rule_id,
        learned.rule.revision,
        candidate,
    )
    assert edited.reason == "validation_reused"
    assert edited.rule.rule_id == learned.rule.rule_id and edited.rule.revision == 2
    assert edited.rule.candidate == candidate
    assert all(case.check.revision == 2 for case in edited.validation.case_results)
    assert rules.store.list_bindings(rules.scopes(session.id)[0])[0].revision == 1
    changed = candidate.model_copy(
        update={"feedback": "Explain evidence and any missing work clearly."}
    )
    again = await rules.test_edit(
        session.id, rules.scopes(session.id)[0], learned.rule.rule_id, 1, changed
    )
    assert again.rule.revision == 3
    assert again.rule.candidate == changed


@pytest.mark.asyncio
async def test_failed_learning_draft_is_retained_without_a_revision(native):
    from tldw_chatbook.Chat.response_rules.models import RuleLearningResult

    _runtime, rules, _chats, session, _gateway, _controller = native
    seed(native)

    async def failed(*_args, **_kwargs):
        return RuleLearningResult("inactive", None, None, {}, "candidate_unavailable")

    rules.builder.learn = failed
    result = await rules.learn(session.id, "The answer omitted evidence")
    assert result.reason == "candidate_unavailable"
    assert rules.store.list_drafts(rules.scopes(session.id)[0]) == (result,)


@pytest.mark.asyncio
async def test_broken_disposable_projection_cannot_fail_the_owned_operation(native):
    _runtime, rules, _chats, session, gateway, controller = native
    seed(native)
    gateway.reply = "Evidence: completed repair"

    def broken(_session_id):
        raise ValueError("detached view")

    controller.response_rules_changed = broken
    result = await rules.learn(session.id, "The answer omitted evidence")
    assert result.state == "active"
    assert rules.store.list_bindings(rules.scopes(session.id)[0])[0].state == "enabled"


@pytest.mark.asyncio
async def test_activation_source_is_fenced_inside_the_write(native):
    _runtime, rules, _chats, session, _gateway, _controller = native
    seed(native)
    activate_write = rules.store.activate

    def changed(*args, **kwargs):
        rules.invalidate(session.id, "source_changed")
        return activate_write(*args, **kwargs)

    rules.store.activate = changed
    result = await rules.learn(session.id, "The answer omitted evidence")
    assert result.state == "inactive"
    assert rules.store.list_bindings(rules.scopes(session.id)[0]) == ()
    assert len(rules.store.list_drafts(rules.scopes(session.id)[0])) == 1


@pytest.mark.asyncio
async def test_helper_usage_retains_original_owner_and_purpose(native):
    import time

    _runtime, rules, _chats, session, _gateway, _controller = native
    seed(native)
    origin = rules._source(session.id)
    lease = rules.helpers.try_acquire(
        origin, purpose="learning", deadline=time.monotonic() + 30
    )
    await lease.run_sync(lambda: "result", lambda _result: None)
    records = _runtime.response_rule_usage(session.id)
    assert len(records) == 1
    assert records[0].source == origin and records[0].purpose == "learning"
    assert records[0].usage is None


@pytest.mark.asyncio
async def test_live_violations_keep_original_task_and_bound_real_repairs(native):
    runtime, rules, _chats, session, gateway, _controller = native
    activate(native)
    gateway.reply = "Missing proof"
    original = rules.evaluator.assess
    captured = []

    async def observed(origin, inputs, *args, **kwargs):
        captured.append(inputs.request_text)
        return await original(origin, inputs, *args, **kwargs)

    rules.evaluator.assess = observed
    turn = runtime.accept_turn(request(native))
    await runtime.wait_for_turn(turn)
    assert len(gateway.payloads) == 3
    assert captured == ["Explain another result"] * 3
    with rules.store.repository.db.transaction() as cursor:
        counters = cursor.execute(
            "SELECT admitted_turns,native_turns FROM console_machine_followup_receipts ORDER BY rowid"
        ).fetchall()
    assert [tuple(row) for row in counters] == [(1, 1), (2, 2)]
    assert rules.state(session.id).assessment.outcome == "violation"


@pytest.mark.asyncio
async def test_direct_continue_violation_uses_the_shared_repair_chain(native):
    _runtime, rules, _chats, session, gateway, controller = native
    answer = activate(native)
    gateway.reply = "Missing proof"
    result = await controller.continue_from_message(answer.id)
    assert result.accepted
    assert len(gateway.payloads) == 3
    assert rules.state(session.id).assessment.outcome == "violation"


@pytest.mark.asyncio
async def test_view_detach_remount_retains_owned_check_and_one_completion(native):
    from types import SimpleNamespace

    runtime, rules, chats, session, gateway, _controller = native
    activate(native)
    gateway.reply = "Evidence: retained answer"
    callbacks, completions = [], []
    view = SimpleNamespace(
        console_view_hooks=lambda: {"response_rules_changed": callbacks.append}
    )
    generation = runtime.attach_view(view)
    entered, release = asyncio.Event(), asyncio.Event()
    assess = rules.evaluator.assess

    async def held(*args, **kwargs):
        entered.set()
        await release.wait()
        return await assess(*args, **kwargs)

    rules.evaluator.assess = held
    chats.subscribe_message_completed(completions.append)
    turn = runtime.accept_turn(request(native))
    async with asyncio.timeout(15):
        await entered.wait()
    retained = rules.state(session.id)
    assert runtime.detach_view(view, generation)
    assert runtime.ensure_response_rules() is rules
    release.set()
    await runtime.wait_for_turn(turn)
    settled = rules.state(session.id)
    assert retained.phase == "checking" and settled.assessment.outcome == "pass"
    replacement = SimpleNamespace(console_view_hooks=lambda: {})
    runtime.attach_view(replacement)
    assert rules.state(session.id) == settled and len(completions) == 1


@pytest.mark.parametrize("terminal", ["stopped", "failed"])
@pytest.mark.asyncio
async def test_partial_and_unsuccessful_primary_generation_is_not_checked(
    native, terminal
):
    runtime, rules, _chats, _session, gateway, controller = native
    activate(native)
    entered, release = asyncio.Event(), asyncio.Event()

    async def stream(_resolution, messages, **_kwargs):
        gateway.payloads.append(messages)
        yield "Partial text"
        entered.set()
        await release.wait()
        raise ValueError("scripted transport failure")

    gateway.stream_chat = stream

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("unfinished generation was checked")

    rules.evaluator.assess = forbidden
    turn = runtime.accept_turn(request(native))
    async with asyncio.timeout(15):
        await entered.wait()
    if terminal == "stopped":
        controller.stop_active_run()
    else:
        release.set()
    await runtime.wait_for_turn(turn)
    assert rules.state(_session.id).assessment is None
    assert rules.helpers.unsettled_count == 0
