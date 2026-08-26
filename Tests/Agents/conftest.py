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


def pin_agent_settings(monkeypatch, **overrides):
    """Serve these `[agents]` keys for this test; all others keep defaults.

    Every OTHER `_setting` key (the run-log eviction knobs read by
    `_make_call_model`) keeps returning its own default, so this never
    silently reconfigures unrelated behaviour -- and, just as importantly,
    never reads the developer's live `config.toml`.

    COMPOSABLE (PR3a-1 Task 2): each call MERGES into whatever a previous
    call pinned, instead of replacing it. Two knobs now matter to the same
    test -- `max_live_subagents` and `subagents_outlive_turn` -- and the
    old replace-the-whole-function shape would have let the second pin
    silently drop the first, handing the test a fleet size it never asked
    for.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        **overrides: `[agents]` key -> raw config value to serve.
    """
    previous = getattr(agent_service._setting, "_pinned_overrides", {})
    merged = dict(previous)
    merged.update(overrides)

    def fake_setting(key, default):
        return merged[key] if key in merged else default

    fake_setting._pinned_overrides = merged
    monkeypatch.setattr(agent_service, "_setting", fake_setting)


def pin_max_live_subagents(monkeypatch, value):
    """Make `[agents] max_live_subagents` read as `value` for this test.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        value: the raw config value to serve. `1` (or anything coercing to
            <= 1) selects the inline path; `> 1` builds a fleet.
    """
    pin_agent_settings(
        monkeypatch, **{agent_service.MAX_LIVE_SUBAGENTS_KEY: value}
    )


def pin_turn_scoped_children(monkeypatch):
    """Pin `[agents] subagents_outlive_turn = false` -- the phase-2 rule.

    PR3a-1 Task 2 made a still-running child outlive its turn BY DEFAULT.
    Use this in any test whose subject is the end-of-turn settle itself
    (waiting for a straggler, cancelling and abandoning a wedged child,
    revoking its approval cards): that behaviour is now what the kill
    switch buys, and it must stay byte-identical under it.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
    """
    pin_agent_settings(
        monkeypatch, **{agent_service.SUBAGENTS_OUTLIVE_TURN_KEY: False}
    )


def join_fleet_children(service, timeout=10.0):
    """Wait for the sub-agents the service's LAST turn started.

    PR3a-1 Task 2 (survival by default): `run_turn` no longer waits for a
    child that is still running, so "the child's run row" is not
    guaranteed to exist -- let alone be terminal -- the instant the turn
    returns. Any test whose assertions are about the CHILD (its row, its
    result, its tool calls) must therefore say so explicitly.

    Joining the threads rather than polling the DB is exact, and STRICTLY
    STRONGER than waiting on the coordinator: `run_child`'s `finally`
    calls `fleet.finish()` BEFORE `db.set_status`, so a finished handle
    does not imply a terminal row (the setup-exception path is where the
    two diverge). Anyone reaching for `all_finished()` as a barrier
    because they want the ROW should reach for this instead.

    A join that times out is asserted, not swallowed: a wedged child would
    otherwise surface much later as a baffling `assert 0 == 1` in whatever
    the caller went on to check.

    Args:
        service: the `AgentService` whose turn just returned.
        timeout: per-thread join budget.
    """
    for handle_id, thread in list(service._fleet_threads.items()):
        thread.join(timeout)
        assert not thread.is_alive(), (
            f"sub-agent {handle_id} did not finish within {timeout}s "
            f"(thread {thread.name} still alive)"
        )


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
