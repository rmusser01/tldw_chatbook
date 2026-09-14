"""Compatibility tests for typed tool-review denial provenance."""

from dataclasses import dataclass

from tldw_chatbook.Agents.agent_models import (
    ToolCall,
    ToolResult,
    ToolReviewDecision,
    normalize_tool_review,
)
from tldw_chatbook.Agents.agent_runtime import _effective_review_decision
from tldw_chatbook.Agents.agent_service import AgentService


def test_legacy_strings_keep_verdict_without_authority():
    for text in ("proceed", "denied by you", "ERROR: permission Off"):
        decision = normalize_tool_review(text)
        assert decision.verdict == text
        assert decision.approval_decision is None


def test_new_tool_result_field_does_not_shift_subclass_positionals():
    @dataclass(frozen=True)
    class ChildResult(ToolResult):
        extra: str = ""

    value = ChildResult(False, "", "failed", None, "child")
    assert value.extra == "child"
    assert value.approval_decision is None
    denied = ToolResult.blocked("no", approval_decision="denied")
    assert denied.outcome == "blocked"
    assert denied.approval_decision == "denied"


def test_effective_review_decision_prefers_call_id_then_name_then_proceed():
    call = ToolCall(name="shared", args={}, call_id="original")
    verdicts = {
        "selected": ToolReviewDecision("call verdict", "denied"),
        "shared": ToolReviewDecision("name verdict", "approved"),
    }

    selected = _effective_review_decision(call, verdicts, call_id="selected")
    assert selected == ToolReviewDecision("call verdict", "denied")

    fallback = _effective_review_decision(call, verdicts, call_id="missing")
    assert fallback == ToolReviewDecision("name verdict", "approved")

    absent = _effective_review_decision(ToolCall("absent", {}), verdicts)
    assert absent == ToolReviewDecision("proceed")


def test_normalize_tool_review_drops_malformed_approval_fact():
    malformed = ToolReviewDecision("denied", "invented")  # type: ignore[arg-type]
    assert normalize_tool_review(malformed) == ToolReviewDecision("denied")


def test_review_observer_accepts_mixed_values_and_receives_string_error():
    observed: list[tuple[ToolCall, ToolResult, float, str]] = []
    service = object.__new__(AgentService)
    service.post_tool_dispatch = lambda *args: observed.append(args)
    calls = [
        ToolCall("legacy", {}, "legacy-id"),
        ToolCall("structured", {}, "structured-id"),
        ToolCall("allowed", {}, "allowed-id"),
    ]
    verdicts = {
        "legacy-id": "legacy refusal",
        "structured-id": ToolReviewDecision("structured refusal", "denied"),
        "allowed-id": ToolReviewDecision("proceed", "approved"),
    }
    wrapped = service._wrap_review_with_observation(lambda _calls: verdicts, "run-1")

    assert wrapped(calls) is verdicts
    assert [(call.name, result.error) for call, result, _, _ in observed] == [
        ("legacy", "legacy refusal"),
        ("structured", "structured refusal"),
    ]
    assert all(isinstance(result.error, str) for _, result, _, _ in observed)
