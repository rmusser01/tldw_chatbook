"""task-31386: abandon one in-flight tool call and let the turn continue."""

from __future__ import annotations

import threading

from tldw_chatbook.Agents.agent_models import TOOL_OUTCOME_CANCELLED
from tldw_chatbook.Agents.agent_service import (
    _call_with_timeout,
    _clear_tool_call_inflight,
    _mark_tool_call_inflight,
    inflight_tool_call,
    request_tool_call_abandon,
    tool_call_abandon_requested,
)


def test_a_request_with_no_call_in_flight_is_refused_and_never_queued():
    _clear_tool_call_inflight("run-x")
    assert request_tool_call_abandon("run-x") is None
    assert tool_call_abandon_requested("run-x") is False
    _mark_tool_call_inflight("run-x", "c1", "grep_files")
    assert inflight_tool_call("run-x") == ("c1", "grep_files")
    assert tool_call_abandon_requested("run-x") is False  # the refusal left nothing behind
    _clear_tool_call_inflight("run-x")


def test_an_abandon_request_cancels_only_the_wrapped_call_and_is_cleared_after():
    run_id = "run-abandon"
    release = threading.Event()

    def _hung():
        release.wait(10)
        return None

    _mark_tool_call_inflight(run_id, "c1", "slow_tool")
    try:
        assert request_tool_call_abandon(run_id) == "slow_tool"
        run_wide_stop = False
        result = _call_with_timeout(
            _hung,
            seconds=30.0,
            tool_name="slow_tool",
            should_cancel=lambda: run_wide_stop or tool_call_abandon_requested(run_id),
        )
    finally:
        _clear_tool_call_inflight(run_id)
        release.set()
    assert result.ok is False and result.outcome == TOOL_OUTCOME_CANCELLED
    assert "cancelled" in (result.error or "")
    # The run itself was never asked to stop, and the flag did not outlive the call.
    assert run_wide_stop is False
    assert tool_call_abandon_requested(run_id) is False
    assert inflight_tool_call(run_id) is None
