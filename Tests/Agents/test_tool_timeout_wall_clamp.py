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


def test_an_already_exhausted_budget_still_bounds_the_call():
    """Must not fall through to an unbounded invoke.

    The caller dispatches unbounded when the timeout is falsy, so an exhausted
    budget has to yield a small POSITIVE value rather than zero.
    """
    seconds, clamped = _effective_tool_timeout(
        configured=3600.0, run_started=0.0, wall_budget=100.0, clock=_clock_at(500.0)
    )

    assert seconds > 0
    assert seconds < 1
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
    """AC#5: the end-to-end property the defect violated."""
    wall_budget = 2.0
    configured_per_tool = 3600.0
    elapsed_when_dispatched = 1.5

    seconds, clamped = _effective_tool_timeout(
        configured=configured_per_tool,
        run_started=0.0,
        wall_budget=wall_budget,
        clock=_clock_at(elapsed_when_dispatched),
    )

    assert clamped is True
    assert elapsed_when_dispatched + seconds <= wall_budget + 0.001


def test_approval_wait_still_pauses_a_clamped_call(monkeypatch):
    """AC#3: ADR-067's refcounted pause must keep working under a clamp.

    The clamp only sets the initial bound. A human decision pending during the
    call re-arms the per-call clock exactly as before, so a slow approver does
    not kill a call that is otherwise progressing.
    """
    import threading

    from tldw_chatbook.Agents import agent_service
    from tldw_chatbook.Agents.agent_models import ToolResult

    release = threading.Event()
    waiting = threading.Event()

    def slow_but_approved():
        waiting.set()
        release.wait(3.0)
        return ToolResult(ok=True, content="done")

    pauses = {"active": True}

    def finish_soon():
        waiting.wait(2.0)
        release.set()

    threading.Thread(target=finish_soon, daemon=True).start()

    result = agent_service._call_with_timeout(
        slow_but_approved,
        0.2,                       # far shorter than the work
        "approved_tool",
        lambda: False,
        pauses_deadline=lambda: pauses["active"],
        clamped_by_wall_budget=True,
    )

    assert result.ok is True, f"pause did not hold the call open: {result.error}"


def test_dispatch_site_applies_the_clamp():
    """Pins the wiring, not just the helper.

    Uses the same source-assertion approach as
    `Tests/Agents/test_mcp_refusal_provenance.py`: the helper is unit-tested
    above, and this catches the failure mode where it exists but nothing calls
    it. Building an AgentService here would need substantially more scaffolding
    than the three lines under test.
    """
    from pathlib import Path

    import tldw_chatbook.Agents.agent_service as agent_service

    source = Path(agent_service.__file__).read_text(encoding="utf-8")

    assert "_effective_tool_timeout(" in source
    assert "run_started = self.clock()" in source
    assert "clamped_by_wall_budget=clamped_by_wall_budget" in source
