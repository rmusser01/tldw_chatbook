"""One unproven tool cleanup ends its run and keeps only its resource (TASK-40716).

The Console's ordinary tree gives every tool call a source scope. When that
scope's return cannot be proven, the resource behind it may still be live, so
it stays held and is never reused. The run that owns it ends at once with a
reason naming the unproven cleanup, and the user can return its tool slot.
"""

import time
from contextlib import contextmanager
from types import SimpleNamespace

from tldw_chatbook.Agents.agent_models import (
    RUN_STUCK,
    AgentConfig,
    ModelTurn,
    RunTerminationReason,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import run_agent_loop
from tldw_chatbook.Agents.agent_service import _call_with_timeout
from tldw_chatbook.Agents.execution_capacity import (
    RuntimeCapacity,
    WorkOrigin,
    holds_unproven_custody,
)
from Tests.Agents.test_agent_runtime import CALC, fence, make_deps


class _UnprovenSource:
    """A source scope whose return fails after the tool itself returned."""

    def __init__(self):
        self.entered = 0
        self.exited = 0

    @contextmanager
    def activate(self):
        self.entered += 1
        try:
            yield
        finally:
            self.exited += 1
        raise OSError("controlled: original source return unavailable")


class _OrdinaryTree:
    """The members of an ordinary tree that `_call_with_timeout` reads."""

    def __init__(self, owner):
        self.owner = owner
        self.capacity = owner.capacity
        self.members = {owner: None}
        self.source = _UnprovenSource()
        self.retained_scopes = 0

    @contextmanager
    def retain_source_scope(self):
        self.retained_scopes += 1
        try:
            yield
        finally:
            self.retained_scopes -= 1

    def check_local_sources(self):
        return None


def test_unproven_source_return_ends_the_call_and_keeps_only_its_resource():
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    tree = _OrdinaryTree(owner)
    drains = []
    owner.on_drained(drains.append)
    retention = None
    try:
        started = time.monotonic()
        result = _call_with_timeout(
            lambda: ToolResult(ok=True, content="read"),
            30.0,
            "fs_read",
            execution_owner=SimpleNamespace(_ordinary_tree=tree),
            owner=owner,
        )
        elapsed = time.monotonic() - started

        # The call reports at once instead of waiting out its 30 s deadline.
        assert elapsed < 5, elapsed
        assert not result.ok and result.dispatch_state == "uncertain"
        assert result.termination_reason is RunTerminationReason.CLEANUP_UNPROVEN
        assert "cleanup of fs_read could not be proven" in result.error

        (retention,) = capacity.retained()
        assert retention.label == "fs_read" and owner._cleanup_unproven
        # The worker still holds the original scopes: the tree's retained
        # source scope is entered and nothing has unwound it.
        workers = [t for t in _custody_threads() if t.name == "tool-fs_read"]
        assert workers and workers[0].daemon
        assert tree.retained_scopes == 1

        # The slot stays held after the run's root finishes...
        owner.finish_root()
        assert capacity.snapshot().tool_workers == 1
        assert capacity.snapshot().stopping_tool_workers == 1
        assert drains == []

        # ...until the user abandons it. Only the slot returns: the drain is
        # unproven and the resource (scopes, retained record) stays held.
        assert retention.abandon() and not retention.abandon()
        assert capacity.snapshot().tool_workers == 0
        assert drains == [False]
        assert capacity.retained() == (retention,)
        assert tree.retained_scopes == 1 and workers[0].is_alive()
    finally:
        if retention is not None:
            retention.prove()
            for worker in _custody_threads():
                worker.join(5)
    # A late proof lets the original scopes unwind.
    assert tree.retained_scopes == 0
    assert capacity.retained() == ()


def test_cleanup_unproven_result_ends_every_run_not_only_goal_runs():
    calls = []
    unproven = ToolResult(
        ok=False,
        error="the cleanup of calculator could not be proven",
        dispatch_state="uncertain",
        termination_reason=RunTerminationReason.CLEANUP_UNPROVEN,
    )
    outcome = run_agent_loop(
        AgentConfig(model="m", system_prompt="s", allowed_tools=("calculator",)),
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(
            [
                ModelTurn(text=fence("calculator", {"expression": "1+1"})),
                ModelTurn(text=fence("calculator", {"expression": "2+2"})),
                ModelTurn(text="done"),
            ],
            invoke=lambda call: calls.append(call) or unproven,
        ),
    )
    assert len(calls) == 1
    assert outcome.status == RUN_STUCK
    assert outcome.termination_reason is RunTerminationReason.CLEANUP_UNPROVEN
    assert outcome.steps[-1].summary == (
        "Agent stopped: the cleanup of calculator could not be proven."
    )


def _custody_threads():
    import threading

    return [t for t in threading.enumerate() if holds_unproven_custody(t)]
