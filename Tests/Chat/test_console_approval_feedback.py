"""Monotonic approval facts are separate from authorization and UI delivery."""

from dataclasses import replace
import pytest

pytestmark = pytest.mark.bootstrap_profile


def captured(round_id="r", revision=1):
    from tldw_chatbook.Chat.approval_presentation import (
        ApprovalAuthority,
        ApprovalRowView,
        ApprovalBatchView,
    )

    owner = ApprovalAuthority(
        "local", "Writer", "Writer", "Scratch", "profile", "call", "Settings"
    )
    row = ApprovalRowView(
        "c",
        1,
        "Read",
        (),
        owner,
        ("approve_once", "approve_session", "deny"),
        False,
        "",
        (),
    )
    return ApprovalBatchView(round_id, "s", "run", revision, (row,), 1, True, True)


def test_receipt_is_not_settlement_and_missing_owner_is_unknown():
    from tldw_chatbook.Chat.console_approval_feedback import (
        ApprovalFeedbackStore,
        format_approval_feedback,
    )
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    store = ApprovalFeedbackStore()
    store.bind_round(captured())
    identity = store.context_for_call("run", "c")
    store.select_decisions("r", {"c": "approve_session"})
    store.publish(ApprovalObservation(identity, "received", "received"))
    fact = store.snapshot("s", "run")[0]
    assert fact.decision_state == "received" and fact.grant_state == "unknown"
    assert "Saved" not in format_approval_feedback(fact)
    store.publish(ApprovalObservation(identity, "settled", "cancelled"))
    assert "not remembered" not in format_approval_feedback(
        store.snapshot("s", "run")[0]
    )


def test_reordered_feedback_cannot_regress_execution():
    from tldw_chatbook.Chat.console_approval_feedback import (
        ApprovalFeedbackStore,
        format_approval_feedback,
    )
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    store = ApprovalFeedbackStore()
    store.bind_round(captured())
    identity = store.context_for_call("run", "c")
    store.select_decisions("r", {"c": "approve_session"})
    for kind, outcome in [
        ("settled", "accepted"),
        ("tool_completed", "success"),
        ("dispatch_started", "starting"),
        ("received", "received"),
        ("grant", "failed"),
    ]:
        store.publish(ApprovalObservation(identity, kind, outcome))
    fact = store.snapshot("s", "run")[0]
    assert fact.execution_state == "tool_completed"
    assert fact.decision_state == "accepted"
    assert (
        "Allowed this call; permission was not remembered"
        in format_approval_feedback(fact)
    )


def test_old_round_feedback_cannot_change_next_card():
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    store = ApprovalFeedbackStore()
    store.bind_round(captured())
    old = store.context_for_call("run", "c")
    store.bind_round(captured("next", 2))
    assert not store.publish(
        ApprovalObservation(old, "grant", "applied", actual_scope="approve_session")
    )
    assert store.snapshot("s", "run")[0].identity.round_id == "next"
    assert store.snapshot("another", "run") == ()


def test_fast_completion_does_not_require_applying_paint():
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    store = ApprovalFeedbackStore()
    store.bind_round(captured())
    identity = store.context_for_call("run", "c")
    assert store.publish(ApprovalObservation(identity, "tool_completed", "success"))
    assert store.snapshot("s", "run")[0].terminal_outcome == "success"
    store.retire_run("run")
    assert not store.publish(ApprovalObservation(identity, "received", "received"))


def test_controller_releases_event_before_receipt_callback_and_rejects_stale_round():
    import threading
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore
    from tldw_chatbook.Agents.approval_observation import ApprovalObservationContext

    controller = object.__new__(ConsoleChatController)
    controller._approval_state_lock = threading.RLock()
    controller.approval_feedback = ApprovalFeedbackStore()
    controller.approval_feedback.bind_round(captured())
    event = threading.Event()
    state = {
        "event": event,
        "decisions": {},
        "session_id": "s",
        "run_id": "run",
        "names": ["c"],
        "observation_contexts": {
            "c": ApprovalObservationContext(
                controller.approval_feedback.context_for_call("run", "c"),
                lambda event: None,
            )
        },
    }
    from Tests.Chat.console_interrupt_test_bindings import make_interrupt_host

    controller._interrupt_host = make_interrupt_host(controller)
    controller._approval_state_lock = controller._interrupt_host.lock
    controller._pending_approval_rounds = controller._interrupt_host.registries[
        "approval"
    ]
    controller._pending_approval_rounds["r"] = state
    seen = []

    def received(session, run):
        assert event.is_set()
        seen.extend(controller.approval_feedback.snapshot(session, run))
        raise RuntimeError("paint unavailable")

    controller.approval_feedback_changed = received
    assert (
        controller.resolve_pending_approval({"c": "approve_once"}, round_id="stale")
        is None
    )
    assert not event.is_set() and not seen
    assert (
        controller.resolve_pending_approval({"c": "approve_once"}, round_id="r") is None
    )
    assert event.is_set() and seen[0].decision_state == "received"
    assert state["decisions"] == {"c": "approve_once"}


def test_feedback_does_not_delay_fifo_promotion():
    import threading
    from Tests.Chat.test_console_interrupt_rounds import FakeSeamsFull, _payload
    from Tests.Chat.console_interrupt_test_bindings import (
        make_interrupt_host as InterruptRoundHost,
    )
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    store = ApprovalFeedbackStore()
    store.bind_round(captured())
    identity = store.context_for_call("run", "c")
    event = threading.Event()
    event.set()
    host.park_round_payload("approval", "queued", _payload("queued"))
    host.run_round(
        "approval",
        "r",
        _payload("r"),
        {"event": event, "session_id": "sess-A"},
        session_id="sess-A",
        owning_session_id="sess-A",
        deadline=None,
        is_parked=False,
        on_outcome=lambda outcome: store.publish(
            ApprovalObservation(identity, "settled", "accepted")
        ),
    )
    assert seams.mounted["approval"][-1]["round_id"] == "queued"
    assert store.snapshot("s", "run")[0].decision_state == "accepted"


def test_bridge_retains_final_grant_failure_in_session_tool_presentation():
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_tool_activity import ConsoleToolActivity
    from tldw_chatbook.Chat.console_approval_feedback import (
        ApprovalFeedbackStore,
        format_approval_feedback,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleActivityPresentation
    from tldw_chatbook.Agents.agent_models import AgentStep
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    chat = ConsoleChatStore()
    session = chat.ensure_session()
    bridge = ConsoleAgentBridge(agent_runs_db=None, store=chat, provider_gateway=None)
    activity = ConsoleToolActivity(chat, session.id)
    bridge._tool_activity_runs["run"] = activity
    feedback = ApprovalFeedbackStore()
    bridge.approval_feedback_store = feedback
    feedback.bind_round(replace(captured(), session_id=session.id))
    identity = feedback.context_for_call("run", "c")
    activity.observe(
        AgentStep(index=1, kind="tool_proposed", tool_name="read", call_id="c"), 1
    )
    bridge._observe_approval_step(
        session.id,
        "run",
        AgentStep(index=1, kind="tool_execution_started", call_id="c"),
    )
    assert feedback.snapshot(session.id, "run")[0].execution_state == "dispatch_started"
    assert "Running" not in format_approval_feedback(
        feedback.snapshot(session.id, "run")[0]
    )
    feedback.publish(
        ApprovalObservation(identity, "grant", "failed", error_code="writer_failed")
    )
    step = AgentStep(
        index=1,
        kind="tool_result",
        tool_name="read",
        call_id="c",
        tool_outcome="success",
        result="ok",
    )
    bridge._observe_approval_step(session.id, "run", step)
    activity.complete(
        step,
        ConsoleActivityPresentation("tool", "read", "success"),
        "result",
        None,
        record_trajectory=True,
    )
    bridge.project_approval_feedback(session.id, "run")
    activity.finish(False)
    feedback.retire_run("run")
    row = chat.messages_for_session(session.id)[-1]
    fact = row.activity_presentation.approval_feedback
    assert fact.terminal_outcome == "success"
    assert (
        "Allowed this call; permission was not remembered"
        in format_approval_feedback(fact)
    )
    assert feedback.snapshot(session.id, "run") == ()


def test_group_does_not_claim_every_exact_input_saved_or_completed():
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    view = captured()
    view = replace(view, rows=(replace(view.rows[0], call_count=2),))
    store = ApprovalFeedbackStore()
    store.bind_round(view)
    identity = store.context_for_call("run", "c")
    assert not store.publish(
        ApprovalObservation(identity, "grant", "applied", actual_scope="allow_matching")
    )
    assert not store.publish(ApprovalObservation(identity, "tool_completed", "success"))
    assert store.publish(
        ApprovalObservation(identity, "grant", "applied", actual_scope="always_allow")
    )
    fact = store.snapshot("s", "run")[0]
    assert fact.applied_scope == "always_allow" and fact.execution_state == ""


def test_feedback_is_excluded_from_durable_message_and_restore():
    from Tests.Chat.test_console_activity_presentation import _RecordingPersistence
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleActivityPresentation,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedback
    from tldw_chatbook.Agents.approval_observation import ApprovalObservationIdentity

    persistence = _RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    fact = ApprovalFeedback(
        ApprovalObservationIdentity(session.id, "run", "r", 1, "c"),
        grant_state="failed",
    )
    presentation = ConsoleActivityPresentation(
        "tool", "read", "success", approval_feedback=fact
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        persist=True,
        activity_presentation=presentation,
    )
    durable = persistence.created[-1]
    assert "approval_feedback" not in repr(durable)
    assert "activity_presentation" not in durable


@pytest.mark.parametrize("terminal", ["accepted", "cancelled", "timeout"])
def test_host_final_arbitration_wins_after_controller_receipt(terminal):
    import contextlib
    import threading
    import time
    from Tests.UI.test_console_mcp_approval import _build_controller, _FakeApp, _pending
    from tldw_chatbook.Agents.mcp_tool_provider import (
        MCPInvocationPolicy,
        restrict_mcp_invocation,
    )
    from tldw_chatbook.Agents.run_context import use_run_id

    controller, chat = _build_controller()
    session = chat.ensure_session()
    controller.app = _FakeApp()
    policy = MCPInvocationPolicy(
        current=lambda: True,
        deadline=time.monotonic() + 3,
        cancel_event=threading.Event(),
        allow_approval=True,
        wait_scope=lambda kind: contextlib.nullcontext(),
    )
    receipts = []

    def show(payload):
        if payload is None:
            return
        controller.resolve_pending_approval(
            {"mcp__srv__tool": "approve_once"}, round_id=payload["round_id"]
        )
        receipts.extend(controller.approval_feedback.snapshot(session.id, "run"))
        if terminal == "cancelled":
            policy.cancel_event.set()
        if terminal == "timeout":
            # Advance only the sampled final arbitration clock after receipt.
            time.sleep(0.04)

    if terminal == "timeout":
        from dataclasses import replace

        policy = replace(policy, deadline=time.monotonic() + 0.03)
    controller.set_pending_approval = show
    with use_run_id("run"), restrict_mcp_invocation(policy):
        answer = controller.request_mcp_approvals([_pending()], session_id=session.id)
    assert receipts and receipts[0].decision_state == "received"
    assert (
        controller.approval_feedback.snapshot(session.id, "run")[0].decision_state
        == terminal
    )
    assert (
        answer["mcp__srv__tool"]
        == {"accepted": "approve_once", "cancelled": "deny", "timeout": "timeout"}[
            terminal
        ]
    )
    assert not controller._pending_approval_rounds


def test_broken_tool_projection_cannot_change_dispatch_or_teardown():
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore
    from tldw_chatbook.Agents.agent_models import AgentStep

    bridge = ConsoleAgentBridge(
        agent_runs_db=None, store=ConsoleChatStore(), provider_gateway=None
    )
    bridge.approval_feedback_store = ApprovalFeedbackStore()
    bridge.approval_feedback_store.bind_round(captured())

    def broken(facts):
        raise RuntimeError("view unavailable")

    bridge._tool_activity_runs["run"] = SimpleNamespace(session_id="s", feedback=broken)
    bridge._observe_approval_step(
        "s",
        "run",
        AgentStep(index=1, kind="tool_result", call_id="c", tool_outcome="success"),
    )
    assert (
        bridge.approval_feedback_store.snapshot("s", "run")[0].terminal_outcome
        == "success"
    )


@pytest.mark.parametrize(
    "outcome,copy",
    [
        ("failure", "Tool failed"),
        ("blocked", "Tool blocked"),
        ("timeout", "Tool timed out"),
        ("cancelled", "Tool stopped"),
    ],
)
def test_terminal_feedback_uses_actual_result_outcome(outcome, copy):
    from tldw_chatbook.Chat.console_approval_feedback import (
        ApprovalFeedbackStore,
        format_approval_feedback,
    )
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    store = ApprovalFeedbackStore()
    store.bind_round(captured())
    identity = store.context_for_call("run", "c")
    store.publish(ApprovalObservation(identity, "tool_completed", outcome))
    assert format_approval_feedback(store.snapshot("s", "run")[0]) == copy


@pytest.mark.parametrize("late_start", [True, False])
def test_confirmed_start_survives_failed_terminal_and_late_delivery(late_start):
    from tldw_chatbook.Chat.console_approval_feedback import (
        ApprovalFeedbackStore,
        format_approval_feedback,
    )
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    store = ApprovalFeedbackStore()
    store.bind_round(captured())
    identity = store.context_for_call("run", "c")
    events = [("backend_started", "running"), ("tool_completed", "failure")]
    if late_start:
        events.reverse()
    for kind, outcome in events:
        store.publish(ApprovalObservation(identity, kind, outcome))
    store.publish(ApprovalObservation(identity, "grant", "failed"))
    fact = store.snapshot("s", "run")[0]
    assert fact.execution_state == "tool_completed"
    assert (
        "Allowed this call; permission was not remembered"
        in format_approval_feedback(fact)
    )
    assert "Tool failed" in format_approval_feedback(fact)


@pytest.mark.parametrize("failure", ["bind_round", "publish"])
def test_reducer_failure_preserves_controller_verdict_release_and_cleanup(
    monkeypatch, failure
):
    from Tests.UI.test_console_mcp_approval import _build_controller, _FakeApp, _pending

    controller, chat = _build_controller()
    session = chat.ensure_session()
    controller.app = _FakeApp()

    def broken(*args, **kwargs):
        raise RuntimeError("observation unavailable")

    monkeypatch.setattr(controller.approval_feedback, failure, broken)
    released = []

    def show(payload):
        if payload is None:
            return
        state = controller._pending_approval_rounds[payload["round_id"]]
        assert (
            controller.resolve_pending_approval(
                {"mcp__srv__tool": "approve_once"}, round_id=payload["round_id"]
            )
            is None
        )
        released.append(state["event"].is_set())

    controller.set_pending_approval = show
    try:
        answer = controller.request_mcp_approvals([_pending()], session_id=session.id)
        assert answer["mcp__srv__tool"] == "approve_once"
        assert released == [True]
        assert not controller._pending_approval_rounds
    finally:
        controller.begin_shutdown()


@pytest.mark.parametrize("count", [1, 2])
def test_name_fallback_grant_failure_transfers_to_observed_member_rows(count):
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_tool_activity import ConsoleToolActivity
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore
    from tldw_chatbook.Agents.agent_models import AgentStep
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    chat = ConsoleChatStore()
    session = chat.ensure_session()
    bridge = ConsoleAgentBridge(agent_runs_db=None, store=chat, provider_gateway=None)
    activity = ConsoleToolActivity(chat, session.id)
    bridge._tool_activity_runs["run"] = activity
    store = ApprovalFeedbackStore()
    bridge.approval_feedback_store = store
    view = captured()
    store.bind_round(
        replace(
            view,
            session_id=session.id,
            rows=(replace(view.rows[0], verdict_key="read", call_count=count),),
        )
    )
    store.register_aliases("run", {"read": ("read",)}, legacy_keys=frozenset({"read"}))
    identity = store.context_for_call("run", "read")
    store.publish(ApprovalObservation(identity, "grant", "failed"))
    for number in range(count):
        step = AgentStep(
            index=number + 1,
            kind="tool_proposed",
            tool_name="read",
            call_id=f"actual-{number}",
        )
        activity.observe(step, 1)
        bridge._observe_approval_step(
            session.id, "run", replace(step, kind="tool_execution_started")
        )
    bridge.project_approval_feedback(session.id, "run")
    activity.finish(False)
    store.retire_run("run")
    rows = chat.messages_for_session(session.id)
    assert len(rows) == count
    assert all(
        row.activity_presentation.approval_feedback.grant_state == "failed"
        for row in rows
    )


def test_raw_stream_confirms_start_and_retains_name_fallback_grant_failure(tmp_path):
    from Tests.Chat.test_console_raw_shell_progress import (
        _bridge,
        _call_step,
        _tool_markers,
    )
    from tldw_chatbook.Chat.console_tool_activity import ConsoleToolActivity
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore
    from tldw_chatbook.Agents.agent_models import AgentStep, AGENT_KIND_PRIMARY
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation
    from tldw_chatbook.Tools.raw_cli_executor import RawCliStreamEvent

    bridge, chat, session_id = _bridge(tmp_path)
    activity = ConsoleToolActivity(chat, session_id)
    bridge._tool_activity_runs["run"] = activity
    store = ApprovalFeedbackStore()
    bridge.approval_feedback_store = store
    view = captured()
    store.bind_round(
        replace(
            view,
            session_id=session_id,
            rows=(replace(view.rows[0], verdict_key="shell_exec"),),
        )
    )
    store.register_aliases(
        "run", {"shell_exec": ("shell_exec",)}, legacy_keys=frozenset({"shell_exec"})
    )
    identity = store.context_for_call("run", "shell_exec")
    store.publish(ApprovalObservation(identity, "grant", "failed"))
    activity.observe(
        AgentStep(
            index=1, kind="tool_proposed", tool_name="shell_exec", call_id="actual"
        ),
        1,
    )
    bridge._project_raw_shell_step(
        session_id,
        "run",
        _call_step(tmp_path, "actual", "printf sample"),
        AGENT_KIND_PRIMARY,
    )
    bridge.raw_shell_progress_sink(
        "run",
        "actual",
        RawCliStreamEvent(
            stream="stdout", text="sample", total_bytes=6, truncated=False
        ),
    )
    assert store.snapshot(session_id, "run")[0].confirmed_backend_start
    bridge.project_approval_feedback(session_id, "run")
    bridge._clear_raw_shell_progress({"run"})
    store.retire_run("run")
    fact = _tool_markers(chat, session_id)[0].activity_presentation.approval_feedback
    assert fact.grant_state == "failed" and fact.confirmed_backend_start


def test_unique_exact_alias_cannot_attribute_a_different_call():
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore

    store = ApprovalFeedbackStore()
    store.bind_round(captured())
    store.register_aliases("run", {"read": ("c",)})
    assert store.context_for_call("run", "other", fallback_tool_name="read") is None


def test_one_group_member_start_does_not_mark_its_siblings_running():
    from tldw_chatbook.Chat.console_approval_feedback import (
        ApprovalFeedbackStore,
        format_approval_feedback,
    )
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    view = captured()
    store = ApprovalFeedbackStore()
    store.bind_round(replace(view, rows=(replace(view.rows[0], call_count=2),)))
    store.register_aliases("run", {"read": ("c",)}, legacy_keys=frozenset({"c"}))
    first = store.context_for_call("run", "member-a", fallback_tool_name="read")
    second = store.context_for_call("run", "member-b", fallback_tool_name="read")
    assert first == second
    store.publish(ApprovalObservation(first, "backend_started", "running"))
    fact = store.snapshot("s", "run")[0]
    assert set(store.projection_call_keys(fact.identity)) == {"member-a", "member-b"}
    assert (
        not fact.confirmed_backend_start
        and "Running" not in format_approval_feedback(fact)
    )


def test_completed_child_feedback_retires_without_retiring_live_sibling():
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore

    bridge = ConsoleAgentBridge(
        agent_runs_db=None, store=ConsoleChatStore(), provider_gateway=None
    )
    store = ApprovalFeedbackStore()
    bridge.approval_feedback_store = store
    for run in ("child-done", "child-live", "primary"):
        store.bind_round(replace(captured(), run_id=run))
    bridge._on_live_run_terminal("child-done", None)
    assert store.snapshot("s", "child-done") == ()
    bridge._on_live_run_terminal("primary", None)
    assert store.snapshot("s", "child-live")


def test_terminal_cleanup_keeps_primary_projection_until_final_transfer():
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore

    bridge = ConsoleAgentBridge(
        agent_runs_db=None, store=ConsoleChatStore(), provider_gateway=None
    )
    store = ApprovalFeedbackStore()
    bridge.approval_feedback_store = store
    store.bind_round(replace(captured(), run_id="primary"))
    store.bind_round(replace(captured(), run_id="child"))
    bridge._tool_activity_runs["primary"] = object()
    owner = SimpleNamespace(
        _approval_observation_runs={"primary": object(), "child": object()}
    )
    callbacks = []
    bridge._on_live_run_terminal(
        "primary", callbacks.append, observation_owners=(owner,)
    )
    assert store.snapshot("s", "primary") and store.snapshot("s", "child")
    bridge._on_live_run_terminal("child", callbacks.append, observation_owners=(owner,))
    assert not store.snapshot("s", "child")
    assert set(owner._approval_observation_runs) == {"primary"}
    assert callbacks == ["primary", "child"]


def test_captured_legacy_flag_is_required_for_synthesized_name_identity():
    from Tests.UI.test_console_mcp_approval import _build_controller, _FakeApp, _pending
    from tldw_chatbook.Agents.run_context import use_run_id

    controller, chat = _build_controller()
    session = chat.ensure_session()
    controller.app = _FakeApp()
    row = replace(_pending(), call_id="command", legacy_observation_key=True)
    controller.set_pending_approval = (
        lambda payload: controller.resolve_pending_approval(
            {"command": "approve_session"}, round_id=payload["round_id"]
        )
        if payload
        else None
    )
    with use_run_id("run"):
        answers = controller.request_mcp_approvals([row], session_id=session.id)
    assert answers.observation_legacy_keys == frozenset({"command"})
    assert (
        controller.approval_feedback.context_for_call(
            "run", "actual", fallback_tool_name=row.tool_name
        ).call_key
        == "command"
    )


@pytest.mark.parametrize("initial", ["failed", "not_applied"])
def test_grouped_later_applied_fact_corrects_failure_without_regressing_success(
    initial,
):
    from tldw_chatbook.Chat.console_approval_feedback import (
        ApprovalFeedbackStore,
        format_approval_feedback,
    )
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    view = captured()
    store = ApprovalFeedbackStore()
    store.bind_round(
        replace(view, call_count=2, rows=(replace(view.rows[0], call_count=2),))
    )
    identity = store.context_for_call("run", "c")
    store.publish(ApprovalObservation(identity, "settled", "accepted"))
    assert store.publish(
        ApprovalObservation(identity, "grant", initial, actual_scope="approve_session")
    )
    assert store.publish(
        ApprovalObservation(
            identity, "grant", "applied", actual_scope="approve_session"
        )
    )
    assert not store.publish(
        ApprovalObservation(identity, "grant", "failed", actual_scope="approve_session")
    )
    fact = store.snapshot("s", "run")[0]
    assert fact.grant_state == "applied" and fact.applied_scope == "approve_session"
    assert "Until Chatbook exits" in format_approval_feedback(fact)
    assert "not remembered" not in format_approval_feedback(fact)
