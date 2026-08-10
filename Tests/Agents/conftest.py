"""Shared Agents-suite fixtures.

`scratch_config` is re-exported from Tests/Internal_Prompts/conftest.py.

`inline_spawns` (PR2a Task 6.5) pins the fleet OFF. Read its docstring
before writing any test that spawns a sub-agent: since Task 6.5 the
shipped default is `max_live_subagents = 3`, so **a test that builds an
`AgentService` without saying which path it wants silently gets the
THREADED one**. That is not a hypothetical -- it is exactly how
`test_agent_service_review_state_scope.py` lost its coverage while staying
green (its six tests describe an INLINE, mid-parent-dispatch interleave in
their docstrings and were running threaded; deleting the whole C1
protection left the file passing).
"""

import pytest

from tldw_chatbook.Agents import agent_service

from Tests.Internal_Prompts.conftest import scratch_config  # noqa: F401


def pin_max_live_subagents(monkeypatch, value):
    """Make `[agents] max_live_subagents` read as `value` for this test.

    Every OTHER `_setting` key (the run-log eviction knobs read by
    `_make_call_model`) keeps returning its own default, so this never
    silently reconfigures unrelated behaviour.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        value: the raw config value to serve. `1` (or anything coercing to
            <= 1) selects the inline path; `> 1` builds a fleet.
    """
    real_key = agent_service.MAX_LIVE_SUBAGENTS_KEY

    def fake_setting(key, default):
        return value if key == real_key else default

    monkeypatch.setattr(agent_service, "_setting", fake_setting)


@pytest.fixture()
def inline_spawns(monkeypatch):
    """Pin the INLINE spawn path: `[agents] max_live_subagents = 1`.

    Use this in any test whose subject is what happens *while a child
    runs* -- nested-run state scoping, record ordering, step ordering --
    because that subject only exists on the inline path, where `spawn`
    runs the child's whole loop synchronously before returning. On the
    threaded default the child is off on its own thread and such a test
    usually still passes while proving nothing.

    A test that wants the fleet should say so just as explicitly (inject a
    `FleetCoordinator`, or `pin_max_live_subagents(monkeypatch, 3)`).
    """
    pin_max_live_subagents(monkeypatch, 1)
    return 1
