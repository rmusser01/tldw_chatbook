"""A single tool call must not outlive the run's wall budget.

TASK-25913. `max_wall_seconds` was checked only at the top of the loop, while
`_call_with_timeout` took an absolute per-call bound. The engine default is 300s
but Console raises it to 3600s, so one hung tool could hold a run roughly an hour
past a budget the user set -- the loop simply never got back to its own check.

The clamp is computed once, at dispatch. A human approval wait that happens
afterwards must not shrink a call that is already running (ADR-067).
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents.agent_service import _effective_tool_timeout


def _clock_at(value: float):
    return lambda: value


def test_configured_timeout_wins_when_it_is_smaller():
    seconds, clamped = _effective_tool_timeout(
        configured=30.0, run_started=0.0, wall_budget=600.0, clock=_clock_at(10.0)
    )

    assert seconds == 30.0
    assert clamped is False


def test_remaining_budget_wins_when_it_is_smaller():
    """AC#1: the effective bound is the lesser of the two."""
    seconds, clamped = _effective_tool_timeout(
        configured=3600.0, run_started=0.0, wall_budget=100.0, clock=_clock_at(40.0)
    )

    assert seconds == pytest.approx(60.0)
    assert clamped is True


def test_no_wall_budget_leaves_the_configured_timeout_untouched():
    """AC#4: a run with no wall budget behaves exactly as before."""
    for budget in (0, 0.0, None):
        seconds, clamped = _effective_tool_timeout(
            configured=3600.0,
            run_started=0.0,
            wall_budget=budget,
            clock=_clock_at(9999.0),
        )
        assert seconds == 3600.0
        assert clamped is False


def test_an_exhausted_budget_refuses_to_dispatch_at_all():
    """An exhausted budget must not start the tool.

    The first version returned a 0.05s floor here, reasoning that a falsy value
    would make the caller dispatch unbounded. But a tiny bound still STARTS the
    call on a daemon thread and abandons it 50ms later -- `_call_with_timeout`
    documents that an abandoned worker "may still complete and act for real"
    after a timeout is reported. For a tool that writes files or spends money,
    dispatch-and-abandon is worse than the overrun it replaced. None means
    "do not dispatch", and the caller reports it without running anything.
    """
    seconds, clamped = _effective_tool_timeout(
        configured=3600.0, run_started=0.0, wall_budget=100.0, clock=_clock_at(500.0)
    )

    assert seconds is None
    assert clamped is True


def test_clamped_timeouts_are_reported_distinctly(monkeypatch):
    """AC#2: the cause must be legible, not just 'timed out'."""
    from tldw_chatbook.Agents import agent_service

    slow_called = []

    def never_finishes():
        slow_called.append(1)
        import time

        time.sleep(5)

    ceiling = agent_service._call_with_timeout(
        never_finishes, 0.05, "slow_tool", lambda: False
    )
    clamped = agent_service._call_with_timeout(
        never_finishes,
        0.05,
        "slow_tool",
        lambda: False,
        clamped_by_wall_budget=True,
    )

    assert ceiling.ok is False and clamped.ok is False
    assert ceiling.error != clamped.error
    assert "budget" in clamped.error.lower()
    assert "budget" not in ceiling.error.lower()


def test_a_long_tool_call_cannot_push_a_run_past_its_wall_budget():
    """AC#5, bounded honestly.

    The earlier version of this test re-derived `remaining = budget - elapsed`
    from hand-picked numbers and asserted the arithmetic it had just performed
    -- a tautology. This asserts the property that actually matters: whatever
    bound comes back, running for it cannot end after the budget does.
    """
    wall_budget = 2.0
    for elapsed in (0.0, 0.5, 1.5, 1.999):
        seconds, _clamped = _effective_tool_timeout(
            configured=3600.0,
            run_started=0.0,
            wall_budget=wall_budget,
            clock=_clock_at(elapsed),
        )
        assert seconds is not None
        assert elapsed + seconds <= wall_budget + 1e-6, (
            f"a call dispatched at {elapsed}s could run until "
            f"{elapsed + seconds}s, past the {wall_budget}s budget"
        )


def _slow_work(duration: float):
    """A tool whose work genuinely outlasts a clamped bound."""
    import time

    from tldw_chatbook.Agents.agent_models import ToolResult

    def run():
        time.sleep(duration)
        return ToolResult(ok=True, content="done")

    return run


def test_approval_wait_holds_a_clamped_call_open():
    """AC#3: ADR-067's refcounted pause must survive the clamp.

    The first version of this test let the work finish in about a millisecond,
    comfortably inside the bound -- it passed with `pauses_deadline` returning
    False, so it proved nothing. The work now genuinely outlasts the bound, and
    `test_without_the_pause_the_same_call_is_stopped` is its control: if that
    one ever passes too, this one has gone vacuous again.
    """
    from tldw_chatbook.Agents import agent_service

    result = agent_service._call_with_timeout(
        _slow_work(0.6),
        0.15,                       # genuinely shorter than the work
        "approved_tool",
        lambda: False,
        pauses_deadline=lambda: True,
        clamped_by_wall_budget=True,
    )

    assert result.ok is True, f"the pause did not hold the call open: {result.error}"


def test_without_the_pause_the_same_call_is_stopped():
    """Control for the test above -- proves the bound is real."""
    from tldw_chatbook.Agents import agent_service

    result = agent_service._call_with_timeout(
        _slow_work(0.6),
        0.15,
        "approved_tool",
        lambda: False,
        pauses_deadline=lambda: False,
        clamped_by_wall_budget=True,
    )

    assert result.ok is False
    assert "budget" in result.error.lower()


def test_the_clamp_reaches_a_real_dispatch(monkeypatch):
    """Proves the clamp is WIRED IN, not merely implemented.

    The helper tests above are arithmetic; this drives the real
    `_make_invoke_tool` closure with an injected clock and captures the bound
    `_call_with_timeout` actually received. Without this, every assertion in
    this file would still pass if the computed value were discarded.
    """
    from types import SimpleNamespace

    from tldw_chatbook.Agents import agent_service
    from tldw_chatbook.Agents.agent_models import (
        AgentConfig,
        RunBudget,
        ToolCall,
        ToolCatalogEntry,
        ToolResult,
        ToolSchema,
    )
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    class _Provider:
        source = "test"

        def list_catalog(self):
            return [
                ToolCatalogEntry(
                    id="test:slow",
                    name="slow",
                    one_line_description="d",
                    source="test",
                )
            ]

        def load_schema(self, tool_id):
            return ToolSchema(id=tool_id, name="slow", description="d", parameters={})

        def invoke(self, tool_id, args):
            return ToolResult(ok=True, content="ok")

    registry = ToolCatalogRegistry()
    registry.register_provider(_Provider())

    now = {"t": 0.0}
    service = agent_service.AgentService(
        db=SimpleNamespace(),
        registry=registry,
        clock=lambda: now["t"],
    )

    captured = {}

    def spy_call_with_timeout(fn, seconds, tool_name, *args, **kwargs):
        captured["seconds"] = seconds
        captured["clamped"] = kwargs.get("clamped_by_wall_budget")
        return fn()

    monkeypatch.setattr(agent_service, "_call_with_timeout", spy_call_with_timeout)

    config = AgentConfig(
        model="test-model",
        system_prompt="",
        allowed_tools=["slow"],
        budget=RunBudget(max_wall_seconds=100.0, max_tool_call_seconds=3600.0),
    )
    invoke_tool = service._make_invoke_tool(
        config, {"slow"}, lambda: False, run_id="run-1"
    )

    now["t"] = 40.0                      # 60s of the 100s budget remains
    outcome = invoke_tool(ToolCall(name="slow", args={}, call_id="c1"))

    assert captured, f"dispatch never reached the timeout wrapper: {outcome}"
    assert captured["clamped"] is True
    assert captured["seconds"] == pytest.approx(60.0), (
        "the dispatch used the per-tool ceiling, not the remaining budget"
    )
