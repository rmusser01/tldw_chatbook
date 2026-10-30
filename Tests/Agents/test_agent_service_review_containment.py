"""Review-check and launch-claim cleanup never strands its neighbours.

TASK-35582 review fixes: a review settlement/check cleanup that raises must
not skip the child-settled signal or fleet settling, a failed tool call ends
pending checks only when the call could not be admitted, an unexpected spawn
failure releases its unaccepted claim, and a half-applied attach is retired.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from Tests.Agents import test_code_review_dispatch
from Tests.Agents.test_post_tool_dispatch_hook import _Provider
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
pytestmark = pytest.mark.bootstrap_profile


# -- check-ending rule, on the real invoke/observation seams -------------------


def _checked_service(behave, *, end=None, prepare=None, hook=None):
    """A service whose code-review owner is a recording stub."""
    ended = []
    stub = SimpleNamespace(
        end_check_actor=end
        or (lambda run_id, reason, **kw: ended.append((run_id, reason, kw))),
        prepare_check_process=prepare or (lambda call: contextlib.nullcontext()),
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


@pytest.mark.parametrize("name,ends", [("shell_exec", True), ("probe", False)])
def test_approval_denial_ends_checks_only_for_the_check_process_tool(name, ends):
    service, _config, ended = _checked_service(lambda: ToolResult(ok=True))
    denied = ToolResult(ok=False, error="deny", outcome="review_denied")
    service._fire_post_tool_dispatch(ToolCall(name=name, args={}), denied, 0.0, "p")
    assert bool(ended) is ends


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
