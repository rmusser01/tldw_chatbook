"""Review-check and launch-claim cleanup never strands its neighbours.

TASK-35582 review fixes: a review settlement/check cleanup that raises must
not skip the child-settled signal or fleet settling, a failed tool call ends
pending checks only when the call could not be admitted, an unexpected spawn
failure releases its unaccepted claim, and a half-applied attach is retired.
"""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pytest

from Tests.Agents import test_code_review_dispatch
from Tests.Agents.test_code_review_boundary_ledger import ledger_case  # noqa: F401
from Tests.Agents.test_post_tool_dispatch_hook import _Provider
from Tests.Chat import test_console_code_review_authority as native_helpers
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    RunBudget,
    ToolCall,
    ToolResult,
    WorkOrigin,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.code_review_models import ReviewRefused
from tldw_chatbook.Agents.code_review_service import CodeReviewService
from tldw_chatbook.Agents.code_review_tool_provider import CodeReviewToolProvider
from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

dispatch_case = test_code_review_dispatch.dispatch_case
review_case = test_code_review_dispatch.review_case
source_case = test_code_review_dispatch.source_case
native_dispatch = native_helpers.native_dispatch
pytestmark = pytest.mark.bootstrap_profile


# -- check-ending rule, on the real invoke/observation seams -------------------


def _checked_service(
    behave, *, end=None, prepare=None, hook=None, claimed=False, review=None
):
    """A service whose code-review owner is a recording stub (or ``review``)."""
    ended = []
    stub = review or SimpleNamespace(
        end_check_actor=end
        or (lambda run_id, reason, **kw: ended.append((run_id, reason, kw))),
        prepare_check_process=prepare or (lambda call: contextlib.nullcontext()),
        is_claimed_check_call=lambda run_id, call: claimed,
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(_Provider(behave))
    review_owner = ("code_review:code_review_read", CodeReviewToolProvider(stub))
    original = registry.resolve_owner_for_name
    registry.resolve_owner_for_name = lambda name: (
        review_owner if name.startswith("code_review_") else original(name)
    )
    service = AgentService(
        db=SimpleNamespace(), registry=registry, post_tool_dispatch=hook
    )
    config = AgentConfig(
        model="m", system_prompt="s", allowed_tools=("probe",), budget=RunBudget()
    )
    return service, config, ended


def _invoke(service, config, name="probe"):
    invoke = service._make_invoke_tool(config, {"probe"}, run_id="primary")
    return invoke(ToolCall(name=name, args={}, call_id="c1"))


def test_tool_admission_refusal_ends_unstarted_checks_exactly_once():
    service, config, ended = _checked_service(lambda: ToolResult(ok=True))
    capacity = RuntimeCapacity(max_tool_workers=1, reserved_manual_tool_workers=0)
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    held = owner.reserve_tool()
    try:
        with owner.activate():
            result = _invoke(service, config)
    finally:
        held.finish()
    assert result.error == "tool call refused: tool_capacity"
    assert ended == [
        ("primary", "tool call refused: tool_capacity", {"unstarted_only": True})
    ]


@pytest.mark.parametrize(
    "result",
    [
        # Recoverable review-tool errors the verifier guide says to retry.
        ToolResult(ok=False, error="invalid_source_range"),
        ToolResult(ok=False, error="submission_revision_conflict"),
        # An unrelated primary tool failure.
        ToolResult(ok=False, error="it broke"),
    ],
)
def test_recoverable_and_unrelated_failures_keep_pending_checks(result):
    service, config, ended = _checked_service(lambda: result)
    assert not _invoke(service, config).ok
    assert ended == []


def test_check_gate_block_keeps_the_claim_it_protects():
    def gate(call):
        raise ReviewRefused("check_execution_specification_mismatch")

    service, config, ended = _checked_service(lambda: ToolResult(ok=True), prepare=gate)
    result = _invoke(service, config)
    assert result.error == "check execution specification refused"
    assert ended == []


@pytest.mark.parametrize("claimed", [True, False])
def test_approval_denial_ends_checks_only_for_the_claimed_check_process(claimed):
    service, _config, ended = _checked_service(
        lambda: ToolResult(ok=True), claimed=claimed
    )
    denied = ToolResult(ok=False, error="deny", outcome="review_denied")
    service._fire_post_tool_dispatch(
        ToolCall(name="shell_exec", args={}), denied, 0.0, "p"
    )
    assert bool(ended) is claimed


def test_claimed_check_predicate_matches_only_the_bound_process(
    review_case, monkeypatch
):
    """Only the claimed check's own argv-exact process denial ends its wait."""
    from tldw_chatbook.Agents.code_review_checks import (
        PendingCheck,
        ReviewCheckExecution,
    )

    c = review_case
    prepared = c.prepare()
    spec = {
        "command": "python -m pytest -q",
        "shell": "auto",
        "initial_directory": str(c.root),
        "timeout_seconds": 30.0,
    }
    check = PendingCheck("p1", None, None, "verifier", "primary")
    check.execution = ReviewCheckExecution(c.primary, None, check, None, None, spec)
    with c.store.lease(prepared.review_id) as context:
        context.checks["p1"] = check
    ended = []
    monkeypatch.setattr(
        c.primary,
        "end_check_actor",
        lambda run_id, reason, **kw: ended.append((run_id, reason)),
    )
    service, _config, _ = _checked_service(
        lambda: ToolResult(ok=True), review=c.primary
    )
    denied = ToolResult(ok=False, error="deny", outcome="review_denied")
    claimed = ToolCall(name="shell_exec", args=dict(spec))
    unrelated = ToolCall(name="shell_exec", args={**spec, "command": "ls"})
    try:
        assert c.primary.is_claimed_check_call("primary", claimed)
        assert not c.primary.is_claimed_check_call("other-run", claimed)
        assert not c.primary.is_claimed_check_call(
            "primary", ToolCall(name="fs_read", args=dict(spec))
        )
        service._fire_post_tool_dispatch(unrelated, denied, 0.0, "primary")
        assert ended == []
        service._fire_post_tool_dispatch(claimed, denied, 0.0, "primary")
        assert ended == [("primary", "deny")]
        check.execution.started = True
        assert not c.primary.is_claimed_check_call("primary", claimed)
    finally:
        with c.store.lease(prepared.review_id) as context:
            context.checks.pop("p1")


def test_raising_check_cleanup_keeps_the_result_and_post_dispatch_hook():
    seen = []

    def end(*args, **kwargs):
        raise OSError("scratch quota")

    service, _config, _ended = _checked_service(
        lambda: ToolResult(ok=True), end=end, hook=lambda *args: seen.append(args)
    )
    refused = ToolResult(ok=False, error="tool call refused: tool_capacity")
    service._fire_post_tool_dispatch(ToolCall(name="probe", args={}), refused, 0.0, "p")
    assert len(seen) == 1 and seen[0][1] is refused


# -- end-of-turn and child-teardown containment --------------------------------


def test_raising_primary_check_end_still_settles_the_fleet(dispatch_case, monkeypatch):
    c = dispatch_case
    settled = []
    original_end = CodeReviewService.end_check_actor
    original_settle = AgentService._settle_fleet

    def end(self, run_id, reason, **kwargs):
        if reason == "primary_ended":
            raise OSError("scratch quota")
        return original_end(self, run_id, reason, **kwargs)

    def settle(self, *args, **kwargs):
        settled.append(True)
        return original_settle(self, *args, **kwargs)

    monkeypatch.setattr(CodeReviewService, "end_check_actor", end)
    monkeypatch.setattr(AgentService, "_settle_fleet", settle)
    c.run()
    assert settled
    assert c.outcome.status == "done"


def test_review_settlement_failure_still_delivers_child_settled(
    dispatch_case, monkeypatch
):
    c = dispatch_case
    signals = []
    original_init = AgentService.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("on_child_settled", lambda run_id, s: signals.append(run_id))
        original_init(self, *args, **kwargs)

    def shutdown(self, *args, **kwargs):
        raise RuntimeError("call_from_thread after app shutdown")

    monkeypatch.setattr(AgentService, "__init__", init)
    monkeypatch.setattr(CodeReviewService, "settle_review_run", shutdown)
    c.run()
    children = {a.run_id for a, _ in c.calls if a.kind == "subagent"}
    assert children and children <= set(signals)


def test_settle_review_capture_contains_a_missing_run_row(dispatch_case):
    c = dispatch_case.run()
    child = next(a.run_id for a, _ in c.calls if a.kind == "subagent")
    c.runtime.db = SimpleNamespace(get_run=lambda run_id: None)
    c.runtime._settle_review_capture(c.primary, child)


# -- launch claims ---------------------------------------------------------------


def test_unexpected_launch_failure_releases_the_unaccepted_claim(
    dispatch_case, monkeypatch
):
    c = dispatch_case
    original_loop = agent_service.run_agent_loop
    raised = []

    def fail_once(c):
        original = c.primary.launch_task

        def launch(claim):
            if not raised:
                raised.append(claim)
                raise RuntimeError("scripted launch failure")
            return original(claim)

        monkeypatch.setattr(c.primary, "launch_task", launch)

    def loop(config, messages, schemas, deps):
        if config.review_binding is None and not raised:
            with pytest.raises(RuntimeError, match="scripted launch failure"):
                deps.spawn(review_id=c.prepared.review_id, review_slot_id="R1")
            with c.store.lease(c.prepared.review_id) as context:
                assert "R1" not in context.launch_claims
        return original_loop(config, messages, schemas, deps)

    monkeypatch.setattr(agent_service, "run_agent_loop", loop)
    c.run(prepare_hook=fail_once)
    assert raised
    assert [a for a, _ in c.calls if a.kind == "subagent"]
    assert c.outcome.subagents_spawned == 1


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("inline", [False, True])
def test_check_actor_refusal_retires_the_half_applied_attach(
    dispatch_case, monkeypatch, inline, scheduled
):
    c = dispatch_case
    refused = []

    def refuse_once(c):
        original = c.primary.store.bind_check_actor
        if scheduled:
            # A boundary wake / stage acceptance pinned the verifier attempt.
            with c.store.lease(c.prepared.review_id) as context:
                context.scheduled_attempts["V1"] = "scheduled-v1"

        def bind(primary_run_id, review_id, attempt_id):
            if not refused:
                with c.store.lease(review_id) as context:
                    refused.append(context.launch_claims["V1"].run_id)
                raise ReviewRefused("verification_attempt_closed")
            return original(primary_run_id, review_id, attempt_id)

        monkeypatch.setattr(c.primary.store, "bind_check_actor", bind)

    def parent_script(c):
        spawn = c.reply(
            "spawn_subagent",
            {"review_id": c.prepared.review_id, "review_slot_id": "V1"},
        )
        return [spawn, spawn]

    c.run(
        verify=True,
        inline=inline,
        prepare_hook=refuse_once,
        parent_script=parent_script,
    )
    assert c.outcome.status == "done"
    assert refused and refused[0]
    assert c.db.get_run(refused[0])["status"] == "error"
    assert refused[0] not in c.store._tab.runs
    children = [a for a, _ in c.calls if a.kind == "subagent"]
    assert children and refused[0] not in {a.run_id for a in children}
    assert "verification_attempt_closed" in json.dumps(c.calls[1][1])
    if scheduled:
        # The retry relaunched the SAME scheduled attempt, not a stale refusal.
        assert c.store._tab.runs[children[0].run_id].attempt_id == "scheduled-v1"


@pytest.mark.parametrize(
    "case,inline,left",
    [
        # Child setup raises after the slot was counted: no run row, no child.
        ("setup", False, 2),
        ("setup", True, 2),
        # A run row exists and nothing proves the child never ran: stays charged.
        ("run_row", True, 1),
        # The child dispatched model turns, then crashed in post-loop code.
        ("after_model_turn", True, 1),
        # _run_owned returned; the inline cleanup after it raised.
        ("after_return", True, 1),
    ],
)
def test_failed_spawn_returns_its_slot_only_when_no_child_ran(
    dispatch_case, monkeypatch, case, inline, left
):
    c = dispatch_case
    original_loop = agent_service.run_agent_loop
    remaining = []

    def fail_once(owner, name, check=lambda *args: True):
        original = getattr(owner, name)

        def failing(*args, **kwargs):
            if not remaining and check(*args):
                raise RuntimeError("scripted spawn failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, failing)

    if case == "setup":
        fail_once(agent_service, "get_internal_prompt")
    elif case == "run_row":
        fail_once(AgentService, "_fork_review_child")
    elif case == "after_return":
        from tldw_chatbook.Agents.execution_capacity import ExecutionOwner

        finished = set()

        def second_finish(owner):
            seen = owner in finished
            finished.add(owner)
            return owner.child and seen

        fail_once(ExecutionOwner, "finish_root", second_finish)
    else:
        fail_once(
            AgentService, "_persist", lambda self, run_id, *a: run_id != "primary"
        )

    def loop(config, messages, schemas, deps):
        if config.review_binding is None and not remaining:
            before = c.primary.capacity().remaining_children
            with pytest.raises(RuntimeError, match="scripted spawn failure"):
                deps.spawn(review_id=c.prepared.review_id, review_slot_id="R1")
            ran = any(actor.kind == "subagent" for actor, _ in c.calls)
            remaining.append((before, c.primary.capacity().remaining_children, ran))
        return original_loop(config, messages, schemas, deps)

    monkeypatch.setattr(agent_service, "run_agent_loop", loop)
    c.run(inline=inline)
    ran = case in {"after_model_turn", "after_return"}
    assert remaining == [(2, left, ran)]


def test_thread_start_refusal_survives_a_refused_launch_retirement(
    dispatch_case, monkeypatch
):
    import threading

    c = dispatch_case
    original_start = threading.Thread.start
    started = []

    def start(thread):
        if thread.name.startswith("fleet-") and not started:
            started.append(thread)
            raise RuntimeError("scripted fleet start failure")
        return original_start(thread)

    def expire(c):
        def retire(claim, run_id):
            raise ReviewRefused("review_expired")

        monkeypatch.setattr(c.primary, "retire_unstarted_launch", retire)

    monkeypatch.setattr(threading.Thread, "start", start)
    c.run(prepare_hook=expire)
    assert started
    assert c.outcome.status == "done"
    assert "could not start sub-agent" in json.dumps(c.calls[1][1])


def test_wait_notice_cleanup_failure_still_returns_the_refusal(
    dispatch_case, monkeypatch
):
    c = dispatch_case
    original_end = CodeReviewService.end_check_actor
    notices = []

    def notice(self, run_id, child_run_ids):
        if notices:
            return None
        # A notice that cannot fit the live tool-result cap.
        notices.append(SimpleNamespace(review_id="r" * 100_000, proposal_id="p"))
        notices[0].child_run_id = child_run_ids[0] if child_run_ids else ""
        return notices[0]

    def end(self, run_id, reason, **kwargs):
        if reason == "tool_check_notice_transport_unavailable":
            raise OSError("scratch quota")
        return original_end(self, run_id, reason, **kwargs)

    monkeypatch.setattr(CodeReviewService, "pending_check_for_wait", notice)
    monkeypatch.setattr(CodeReviewService, "end_check_actor", end)
    c.run()
    assert notices
    assert c.outcome.status == "done"
    primary = [request for actor, request in c.calls if actor.kind == "primary"]
    assert "check_notice_transport_unavailable" in json.dumps(primary[-1])


def test_thread_start_refusal_keeps_a_scheduled_attempt_launchable(
    dispatch_case, monkeypatch
):
    import threading

    c = dispatch_case
    original_start = threading.Thread.start
    started = []

    def start(thread):
        if thread.name.startswith("fleet-") and not started:
            started.append(thread)
            raise RuntimeError("scripted fleet start failure")
        return original_start(thread)

    def pin(c):
        with c.store.lease(c.prepared.review_id) as context:
            context.scheduled_attempts["R1"] = "scheduled-r1"

    monkeypatch.setattr(threading.Thread, "start", start)
    c.run(replay=True, prepare_hook=pin)
    children = [a for a, _ in c.calls if a.kind == "subagent"]
    assert started and children
    assert c.store._tab.runs[children[0].run_id].attempt_id == "scheduled-r1"


def test_inline_review_child_gets_a_second_settlement_attempt(
    dispatch_case, monkeypatch
):
    c = dispatch_case
    original = CodeReviewService.settle_review_run
    failures = []

    def settle_once_failing(self, run_id, **kwargs):
        if not failures:
            failures.append(run_id)
            raise OSError("transient settlement failure")
        return original(self, run_id, **kwargs)

    monkeypatch.setattr(CodeReviewService, "settle_review_run", settle_once_failing)
    c.run(inline=True)
    assert failures
    with c.store.lease(c.prepared.review_id) as context:
        settled = [
            stream.terminal_status
            for (slot, _attempt), stream in context.streams.items()
            if slot == "R1"
        ]
    assert settled == ["done"]


def test_automatic_budget_refusal_ends_unstarted_checks(monkeypatch):
    from tldw_chatbook.Agents import automatic_work_runtime
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Agents.tool_catalog import ToolExecutionPolicy

    def refuse():
        raise AutomaticWorkRefused("budget_exhausted")

    service, config, ended = _checked_service(lambda: ToolResult(ok=True))
    # code_review_read/submit dispatch on the definitive-after-start path.
    monkeypatch.setattr(
        service.registry,
        "execution_policy_for",
        lambda name: ToolExecutionPolicy.DEFINITIVE_AFTER_START,
    )
    monkeypatch.setattr(
        automatic_work_runtime,
        "current_automatic_work",
        lambda: SimpleNamespace(check=refuse),
    )
    result = _invoke(service, config)
    assert not result.ok and result.dispatch_state == "not_started"
    assert [reason for _run, reason, _kw in ended] == [result.error]


# -- wake-claim predecessor check ------------------------------------------------


def test_predecessor_check_reads_only_the_named_settlement_step(ledger_case):  # noqa: F811
    c = ledger_case
    run_id = c.live_boundary.predecessor_run_ids[0]
    with c.db.transaction() as connection:
        connection.execute(
            "INSERT INTO agent_run_steps (run_id, seq, payload, created_at) "
            "VALUES (?, ?, ?, ?)",
            (run_id, 987654, "{not json", "2026-10-29T00:00:00Z"),
        )
    assert c.claim().state == "prepared"
