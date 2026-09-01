"""An observational seam that fires after every tool call completes.

TASK-26010. A pre-dispatch hook existed (`review_tool_calls`,
`before_tool_dispatch`); nothing observed a COMPLETED call. Usage telemetry,
incident capture and verification policies all need that seam, and the one
module that looked like it (`Tools/file_operation_hooks.py`) is dead code
pinned retired by `test_system_a_is_retired.py`.

The hook lives on the service's per-run invoke closure, so sub-agents fire it
with their own run id, and it is strictly observational: a raising hook costs
nothing but its own observation.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    RunBudget,
    ToolCall,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry


class _Provider:
    source = "test"

    def __init__(self, behave):
        self._behave = behave

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id="test:probe", name="probe", one_line_description="d", source="test"
            )
        ]

    def load_schema(self, tool_id):
        return ToolSchema(id=tool_id, name="probe", description="d", parameters={})

    def invoke(self, tool_id, args):
        return self._behave()


def _service(behave, hook, **budget_kw):
    registry = ToolCatalogRegistry()
    registry.register_provider(_Provider(behave))
    service = AgentService(
        db=SimpleNamespace(),
        registry=registry,
        post_tool_dispatch=hook,
    )
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("probe",),
        budget=RunBudget(**budget_kw),
    )
    return service, config


def _invoke(service, config, run_id="run-1"):
    invoke_tool = service._make_invoke_tool(
        config, {"probe"}, lambda: False, run_id=run_id
    )
    return invoke_tool(ToolCall(name="probe", args={}, call_id="c1"))


def test_hook_fires_on_success_with_call_result_and_timing():
    seen = []
    service, config = _service(
        lambda: ToolResult(ok=True, content="fine"),
        lambda call, result, duration, run_id: seen.append(
            (call.name, result.ok, duration, run_id)
        ),
    )

    result = _invoke(service, config)

    assert result.ok is True
    assert len(seen) == 1
    name, ok, duration, run_id = seen[0]
    assert (name, ok, run_id) == ("probe", True, "run-1")
    assert duration >= 0


def test_hook_sees_a_failed_call_distinguishably():
    seen = []
    service, config = _service(
        lambda: ToolResult(ok=False, error="it broke"),
        lambda call, result, duration, run_id: seen.append(result),
    )

    _invoke(service, config)

    assert seen[0].ok is False
    assert "broke" in (seen[0].error or "")


def test_hook_sees_a_gate_denied_call_distinguishably():
    seen = []
    service, config = _service(
        lambda: ToolResult(ok=False, error="denied by the user", outcome="blocked"),
        lambda call, result, duration, run_id: seen.append(result),
    )

    _invoke(service, config)

    assert seen[0].outcome == "blocked"


def test_hook_sees_a_timed_out_call():
    from tldw_chatbook.Agents.agent_service import TOOL_OUTCOME_TIMEOUT

    seen = []

    def slow():
        time.sleep(0.4)
        return ToolResult(ok=True, content="too late")

    service, config = _service(
        slow,
        lambda call, result, duration, run_id: seen.append((result, duration)),
        max_tool_call_seconds=0.05,
    )

    result = _invoke(service, config)

    assert result.outcome == TOOL_OUTCOME_TIMEOUT
    assert seen[0][0].outcome == TOOL_OUTCOME_TIMEOUT
    assert seen[0][1] >= 0.05, "timing must cover the whole bounded call"


def test_a_raising_hook_never_fails_the_call():
    def explode(call, result, duration, run_id):
        raise RuntimeError("HOOK-SENTINEL")

    service, config = _service(lambda: ToolResult(ok=True, content="fine"), explode)

    result = _invoke(service, config)

    assert result.ok is True
    assert result.content == "fine"


def test_run_attribution_follows_the_closure():
    """AC#4: each run's closure carries its own id, so children attribute."""
    seen = []
    service, config = _service(
        lambda: ToolResult(ok=True, content="fine"),
        lambda call, result, duration, run_id: seen.append(run_id),
    )

    _invoke(service, config, run_id="parent-run")
    _invoke(service, config, run_id="child-run")

    assert seen == ["parent-run", "child-run"]


def test_no_hook_means_the_closure_is_not_even_wrapped():
    """AC#5: zero overhead when nothing is registered."""
    registry = ToolCatalogRegistry()
    registry.register_provider(_Provider(lambda: ToolResult(ok=True, content="x")))
    service = AgentService(db=SimpleNamespace(), registry=registry)
    config = AgentConfig(
        model="m", system_prompt="s", allowed_tools=("probe",), budget=RunBudget()
    )

    result = service._make_invoke_tool(config, {"probe"}, lambda: False, run_id="r")(
        ToolCall(name="probe", args={}, call_id="c1")
    )

    assert result.ok is True


def test_review_denied_calls_are_observed_too():
    """A call refused by the review hook never dispatches, but its refusal is
    still a completion the observer should see, distinguishably."""
    seen = []
    registry = ToolCatalogRegistry()
    registry.register_provider(_Provider(lambda: ToolResult(ok=True, content="x")))
    service = AgentService(
        db=SimpleNamespace(),
        registry=registry,
        review_tool_calls=lambda calls, run_id: {c.call_id: "denied" for c in calls},
        post_tool_dispatch=lambda call, result, duration, run_id: seen.append(
            (call.name, result.outcome, run_id)
        ),
    )

    wrapped = service._wrap_review_with_observation(
        lambda calls: {c.call_id: "denied" for c in calls}, "run-9"
    )
    verdicts = wrapped([ToolCall(name="probe", args={}, call_id="c1")])

    assert verdicts == {"c1": "denied"}
    assert seen == [("probe", "review_denied", "run-9")]


def test_review_proceed_verdicts_are_not_observed():
    seen = []
    registry = ToolCatalogRegistry()
    registry.register_provider(_Provider(lambda: ToolResult(ok=True, content="x")))
    service = AgentService(
        db=SimpleNamespace(),
        registry=registry,
        post_tool_dispatch=lambda call, result, duration, run_id: seen.append(1),
    )

    wrapped = service._wrap_review_with_observation(
        lambda calls: {c.call_id: "proceed" for c in calls}, "run-9"
    )
    wrapped([ToolCall(name="probe", args={}, call_id="c1")])

    assert seen == [], "proceed is not a completion; the dispatch will report"
