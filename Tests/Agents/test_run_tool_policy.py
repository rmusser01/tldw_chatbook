"""Task 7 (workspace assistant defaults): per-run tool call caps.

`RunToolPolicy` counts invocations per `(run_id, tool name)`; a capped tool
refuses persistently once its cap is reached, uncapped tools are untouched,
and a different run id gets its own fresh counters.
"""

from __future__ import annotations

from tldw_chatbook.Agents.run_tool_policy import (
    PERSONA_POLICY_CALL_CAP_REFUSAL,
    RunToolPolicy,
)


def test_cap_allows_up_to_limit_then_refuses_persistently():
    policy = RunToolPolicy({"web_search": 2})
    assert policy.check("run-1", "web_search") == (True, None)
    assert policy.check("run-1", "web_search") == (True, None)
    ok, refusal = policy.check("run-1", "web_search")
    assert ok is False and refusal == PERSONA_POLICY_CALL_CAP_REFUSAL.format(name="web_search")
    assert policy.check("run-1", "web_search")[0] is False  # stays refused
    assert policy.check("run-2", "web_search")[0] is True  # per-run counters
    assert policy.check("run-1", "fs_read") == (True, None)  # uncapped untouched


def test_no_caps_is_identity():
    policy = RunToolPolicy({})
    assert policy.check("run-1", "anything") == (True, None)


def test_refusal_does_not_advance_the_counter():
    """A refused check must not keep consuming: the cap bounds EXECUTIONS."""
    policy = RunToolPolicy({"t": 1})
    assert policy.check("r", "t") == (True, None)
    for _ in range(3):
        assert policy.check("r", "t")[0] is False
