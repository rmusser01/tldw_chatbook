"""Controller-to-owner journeys with real permission state and harmless transport."""

import pytest

from Tests.Agents.test_mcp_tool_provider import (
    FakeMCPService,
    _catalog_record,
    _tool_dict,
    _compose,
)
from Tests.Agents import test_mcp_tool_provider as mcp_helpers
from Tests.MCP.test_control_plane_permissions import _service
from Tests.UI.test_console_mcp_approval import _FakeApp
from Tests.console_provider_doubles import persisted_console_store
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_approval_feedback import format_approval_feedback

running_loop = mcp_helpers.running_loop

pytestmark = pytest.mark.bootstrap_profile


def owner_journey(tmp_path, loop, *, profile="default"):
    """Real service/store/cache; only the external MCP transport is a double."""
    service, _ = _service(tmp_path)
    payload = service.permission_store.load()
    payload["profiles"]["default"]["global_default"] = "ask"
    payload["profiles"]["Writer"] = {"servers": {}}
    payload["profiles"]["Reader"] = {"servers": {}}
    service.permission_store.save(payload)
    transport = FakeMCPService(
        catalog_records=[
            _catalog_record(
                "srv",
                [
                    _tool_dict(
                        "read",
                        input_schema={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    )
                ],
            )
        ]
    )
    service.local_external_catalog = transport.local_external_catalog
    service.execute_hub_tool = transport.execute_hub_tool
    service.execute_hub_tool_result = None  # Select the harmless legacy transport seam.
    provider = MCPToolProvider(
        service=service, main_loop=loop, profile_id_provider=lambda: profile
    )
    _compose(provider)
    controller = ConsoleChatController(
        store=persisted_console_store(db_path=tmp_path / "chat.sqlite"),
        provider_gateway=object(),
    )
    controller.app = _FakeApp()
    session = controller.store.ensure_session()
    controller.store.persistence.db.close()
    return controller, session, provider, service, transport


def answer_round(controller, session, pending, decisions):
    """Use the actual admission/host/controller path, with a synchronous UI answer."""

    def show(payload):
        if payload:
            controller.resolve_pending_approval(decisions, round_id=payload["round_id"])

    controller.set_pending_approval = show
    with use_run_id("journey"):
        try:
            return controller.request_mcp_approvals(pending, session_id=session.id)
        finally:
            if not controller.store.persistence.db.is_memory_db:
                controller.store.persistence.db.close()


@pytest.mark.parametrize(
    "choice", ["approve_once", "deny", "approve_session", "always_allow"]
)
def test_controller_decision_reaches_real_owner_and_next_invocation(
    tmp_path,
    running_loop,
    choice,
):
    controller, session, provider, service, transport = owner_journey(
        tmp_path,
        running_loop,
        profile="Writer",
    )
    tool = provider.list_catalog()[0].id
    pending = provider.pending_gate_for(tool, {"query": "first"})
    answers = answer_round(controller, session, [pending], {tool: choice})
    assert dict(answers) == {tool: choice}
    provider.apply_batch_decisions("journey", answers)
    with use_run_id("journey"), use_tool_call_id("first"):
        first = provider.invoke(tool, {"query": "first"})
    assert first.ok is (choice != "deny")
    assert len(transport.execute_calls) == int(choice != "deny")
    hub = provider._entry_by_llm_name[tool][0]
    assert service.is_session_approved(
        hub.server_key, hub.name, profile_id="Writer"
    ) is (choice == "approve_session")
    assert not service.is_session_approved(
        hub.server_key, hub.name, profile_id="Reader"
    )
    assert service.gate_tool_test(hub, profile_id="Writer").state == (
        "allow" if choice == "always_allow" else "ask"
    )
    assert service.gate_tool_test(hub, profile_id="Reader").state == "ask"
    with use_run_id("later"), use_tool_call_id("second"):
        second = provider.invoke(tool, {"query": "second"})
    assert second.ok is (choice in {"approve_session", "always_allow"})
    fact = controller.approval_feedback.snapshot(session.id, "journey")[0]
    assert fact.decision_state == "accepted"
    assert fact.grant_state == (
        "applied" if choice in {"approve_session", "always_allow"} else "unknown"
    )


def test_default_persistent_grant_inherits_but_temporary_grant_does_not(
    tmp_path, running_loop
):
    controller, session, provider, service, _ = owner_journey(tmp_path, running_loop)
    tool = provider.list_catalog()[0].id
    hub = provider._entry_by_llm_name[tool][0]
    service.approve_for_session(hub.server_key, hub.name)
    assert not service.is_session_approved(
        hub.server_key, hub.name, profile_id="Writer"
    )
    row = provider.pending_gate_for(tool, {})
    # Session approval would skip review; revoke through the same real cache owner.
    service.revoke_session_approval(hub.server_key, hub.name)
    row = provider.pending_gate_for(tool, {})
    answers = answer_round(controller, session, [row], {tool: "always_allow"})
    provider.apply_batch_decisions("journey", answers)
    with use_run_id("journey"), use_tool_call_id("c"):
        assert provider.invoke(tool, {}).ok
    assert service.gate_tool_test(hub, profile_id="Writer").state == "allow"
    assert service.gate_tool_test(hub, profile_id="Reader").state == "allow"


def test_failed_remembering_keeps_actual_call_and_feedback_honest(
    tmp_path,
    running_loop,
    monkeypatch,
):
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    controller, session, provider, service, transport = owner_journey(
        tmp_path, running_loop
    )
    tool = provider.list_catalog()[0].id
    row = provider.pending_gate_for(tool, {"query": "only once"})

    def failed(*args, **kwargs):
        raise OSError("synthetic private writer failure")

    monkeypatch.setattr(service.permission_store, "set_tool_state", failed)
    answers = answer_round(controller, session, [row], {tool: "always_allow"})
    provider.apply_batch_decisions("journey", answers)
    with use_run_id("journey"), use_tool_call_id("c"):
        result = provider.invoke(tool, row.arguments)
    assert result.ok and len(transport.execute_calls) == 1
    store = controller.approval_feedback
    fact = store.snapshot(session.id, "journey")[0]
    assert fact.grant_state == "failed"
    assert "Allowed this call" not in format_approval_feedback(fact)
    # The actual successful result supplies the terminal observation, as the bridge does.
    store.publish(ApprovalObservation(fact.identity, "tool_completed", "success"))
    assert (
        "Allowed this call; permission was not remembered"
        in format_approval_feedback(store.snapshot(session.id, "journey")[0])
    )
    with use_run_id("later"), use_tool_call_id("next"):
        assert not provider.invoke(tool, row.arguments).ok
    assert len(transport.execute_calls) == 1


def test_independent_same_tool_rows_withhold_matching_but_preserve_inputs(
    tmp_path, running_loop
):
    from tldw_chatbook.Chat.console_chat_controller import _build_approval_payload

    controller, session, provider, service, transport = owner_journey(
        tmp_path, running_loop
    )
    tool = provider.list_catalog()[0].id
    pending = [
        provider.pending_gate_for(tool, {"query": value}, value)
        for value in ("first", "second")
    ]
    payload = _build_approval_payload("r", session.id, "journey", pending, 0, None)
    assert len(payload["view"].rows) == 2
    for row, original in zip(payload["view"].rows, pending):
        assert "allow_matching" not in row.legal_decisions
        assert row.withheld_scope_copy
        assert row.argument_sets == (original.arguments,)
    assert not transport.execute_calls and not service._session_approvals


def test_user_denial_reason_survives_controller_and_refusal_projection(
    tmp_path, running_loop
):
    from tldw_chatbook.Agents.agent_models import normalize_tool_review
    from tldw_chatbook.Agents.approval_provenance import ApprovalDecisions
    from tldw_chatbook.Chat.console_chat_controller import _review_decision

    controller, session, provider, service, transport = owner_journey(
        tmp_path, running_loop
    )
    tool = provider.list_catalog()[0].id
    reason = "Use the public document instead."
    row = provider.pending_gate_for(tool, {"query": "private"})
    answers = answer_round(
        controller,
        session,
        [row],
        ApprovalDecisions({tool: "deny"}, denial_reasons={tool: reason}),
    )
    provider.apply_batch_decisions("journey", answers)
    with use_run_id("journey"), use_tool_call_id("c"):
        result = provider.invoke(tool, row.arguments)
    assert not result.ok
    denied = normalize_tool_review(_review_decision(row, answers, result.error))
    assert denied.approval_decision == "denied"
    assert reason in denied.verdict and "untrusted text" in denied.verdict
    assert not transport.execute_calls and not service._session_approvals


def test_raw_chat_grant_reaches_runtime_then_disarm_refuses_later_call(tmp_path):
    from Tests.Chat.test_console_raw_shell_revocation import (
        _controller_provider_hook,
        _ImmediateExecutor,
    )
    from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
    from tldw_chatbook.Agents.agent_models import ToolCall

    runtime = RawCliRuntime(lambda: True, executor=_ImmediateExecutor())
    assert runtime.arm().armed
    controller, provider, _ = _controller_provider_hook(tmp_path, runtime)
    session = controller.store.ensure_session()
    call = ToolCall("shell_exec", {"command": "echo harmless fixture"}, "raw")
    pending = provider.pending_gate_for(call)
    answers = answer_round(controller, session, [pending], {"raw": "approve_session"})
    provider.apply_batch_decisions("journey", answers, [pending])
    with use_run_id("journey"), use_tool_call_id("raw"):
        assert provider.invoke("shell_exec", call.args).ok
    assert runtime.model_session_granted(session.id)
    assert not runtime.model_session_granted("different-chat")
    fact = controller.approval_feedback.snapshot(session.id, "journey")[0]
    assert fact.grant_state == "applied" and fact.applied_scope == "raw_shell_session"
    runtime.disarm()
    assert not runtime.model_session_granted(session.id)
    with use_run_id("later"), use_tool_call_id("later"):
        result = provider.invoke("shell_exec", call.args)
    assert not result.ok and result.outcome == "blocked"


@pytest.mark.parametrize("profile", ["default", "Writer"])
@pytest.mark.parametrize("hook_policy", [False, True])
def test_direct_mcp_fallback_captures_owner_and_preserves_temporary_scope(
    tmp_path, running_loop, monkeypatch, profile, hook_policy
):
    import contextlib
    import threading
    import time
    import tldw_chatbook.Agents.mcp_tool_provider as module
    from tldw_chatbook.Chat.approval_presentation import capture_approval_view

    controller, session, provider, service, transport = owner_journey(
        tmp_path, running_loop, profile=profile
    )
    tool_id = provider.list_catalog()[0].id
    seen = []

    def approve(pending):
        view = capture_approval_view(
            pending,
            round_id="fallback",
            session_id=session.id,
            run_id="direct",
            revision=1,
        )
        seen.append(view)
        row = view.rows[0]
        assert row.authority.profile_id == profile
        assert row.authority.location_label != "Unknown location"
        assert "approve_session" in row.legal_decisions
        assert "allow_matching" in row.legal_decisions
        assert "always_allow" in row.legal_decisions
        return {row.verdict_key: "approve_session"}

    provider._approval_callback = approve
    if hook_policy:
        policy = module.MCPInvocationPolicy(
            current=lambda: True,
            deadline=time.monotonic() + 30,
            cancel_event=threading.Event(),
            allow_approval=True,
            wait_scope=lambda kind: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            module, "current_mcp_invocation_policies", lambda: (policy,)
        )
        monkeypatch.setattr(provider, "_check_current_definition", lambda tool: None)
    with use_run_id("direct"), use_tool_call_id("direct-call"):
        result = provider.invoke(tool_id, {"query": "original"})
    assert result.ok and len(seen) == 1
    hub = provider._entry_by_llm_name[tool_id][0]
    assert service.is_session_approved(hub.server_key, hub.name, profile_id=profile)
    assert len(transport.execute_calls) == 1


def test_grouped_real_grant_success_corrects_an_earlier_failed_writer(
    tmp_path, running_loop, monkeypatch
):
    controller, session, provider, service, transport = owner_journey(
        tmp_path, running_loop, profile="Writer"
    )
    tool_id = provider.list_catalog()[0].id
    pending = [
        provider.pending_gate_for(tool_id, {"query": value})
        for value in ("first", "second")
    ]
    answers = answer_round(controller, session, pending, {tool_id: "approve_session"})
    provider.apply_batch_decisions("journey", answers)
    actual_writer = service.approve_for_session
    attempts = []

    def fail_once(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("synthetic remembering failure")
        return actual_writer(*args, **kwargs)

    monkeypatch.setattr(service, "approve_for_session", fail_once)
    for index, row in enumerate(pending):
        with use_run_id("journey"), use_tool_call_id(f"member-{index}"):
            assert provider.invoke(tool_id, row.arguments).ok
    hub = provider._entry_by_llm_name[tool_id][0]
    assert service.is_session_approved(hub.server_key, hub.name, profile_id="Writer")
    assert len(attempts) == 2 and len(transport.execute_calls) == 2
    fact = controller.approval_feedback.snapshot(session.id, "journey")[0]
    assert fact.grant_state == "applied" and fact.applied_scope == "approve_session"
    assert "not remembered" not in format_approval_feedback(fact)
