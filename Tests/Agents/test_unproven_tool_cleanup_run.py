"""One unproven tool cleanup ends its run and keeps only its resource (TASK-40716).

The Console's ordinary tree gives every tool call a source scope. When that
scope's return cannot be proven, the resource behind it may still be live, so
it stays held and is never reused. The run that owns it ends at once with a
reason naming the unproven cleanup, and the user can return its tool slot.
"""

import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

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


class _LateUnprovenSource(_UnprovenSource):
    """A source return that fails only after the caller stopped waiting."""

    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    @contextmanager
    def activate(self):
        yield
        assert self.gate.wait(10)
        raise OSError("controlled: original source return unavailable")


@pytest.mark.parametrize("stopped_by", ["timeout", "per_call_abandon"])
def test_cleanup_proven_unavailable_after_the_call_returned_still_ends_the_run(
    stopped_by,
):
    """A retention after a timeout or abandon must not wedge later calls."""
    import threading

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    tree = _OrdinaryTree(owner)
    gate = threading.Event()
    tree.source = _LateUnprovenSource(gate)
    service = SimpleNamespace(_ordinary_tree=tree)
    retention = None
    try:
        first = _call_with_timeout(
            lambda: ToolResult(ok=True, content="read"),
            0.2 if stopped_by == "timeout" else 30.0,
            "fs_read",
            should_cancel=lambda: stopped_by == "per_call_abandon",
            execution_owner=service,
            owner=owner,
        )
        assert first.termination_reason is None and first.dispatch_state == "uncertain"
        gate.set()
        deadline = time.monotonic() + 10
        while not capacity.retained() and time.monotonic() < deadline:
            time.sleep(0.01)
        (retention,) = capacity.retained()

        second = _call_with_timeout(
            lambda: ToolResult(ok=True, content="listed"),
            30.0,
            "fs_list",
            execution_owner=service,
            owner=owner,
        )
        # Not "previous_tool_still_running": the run ends, naming the tool
        # whose cleanup is unproven.
        assert second.termination_reason is RunTerminationReason.CLEANUP_UNPROVEN
        assert "cleanup of fs_read could not be proven" in second.error
        assert capacity.snapshot().tool_workers == 1
    finally:
        gate.set()
        if retention is not None:
            retention.prove()
            for worker in _custody_threads():
                worker.join(5)


def test_hold_on_a_joined_thread_keeps_custody_instead_of_unwinding():
    """Scopes are already entered: unwinding them could free a live resource."""
    import threading

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    retention = owner.reserve_tool().retain_unproven("fs_read")
    unwound = []

    def custody():
        try:
            retention.hold()
        finally:
            unwound.append(retention.proven)

    worker = threading.Thread(target=custody)  # a joined, non-daemon worker
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not holds_unproven_custody(worker) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert holds_unproven_custody(worker) and unwound == []
    finally:
        retention.prove()
        worker.join(5)
    # Only the late proof let the scopes unwind.
    assert unwound == [True]


def test_host_return_is_visible_only_with_its_decided_durability():
    """Disposal must never read a returned host whose durability is undecided."""
    import threading

    from tldw_chatbook.Agents.ordinary_tree import OrdinaryTurn, _Member

    owner = object()
    member = _Member(settlement=None, run_id="run-1", row_started=True)
    member.terminal = ("done", None, None)
    member.steps = {0: "{}"}
    seen = []

    class _Database:
        def get_run(self, run_id):
            seen.append((member.host_returned, member.durable))
            return {"status": "done", "result": None, "steps": [{"index": 0}]}

    @contextmanager
    def activate():
        yield

    turn = object.__new__(OrdinaryTurn)
    turn.lock = threading.RLock()
    turn.members = {owner: member}
    turn.source = SimpleNamespace(activate=activate, databases=[_Database()])
    turn._retire_scratch = lambda: None
    turn.host_returned(owner)
    turn.host_returned(owner)  # a second return is ignored
    assert seen == [(False, False)]
    assert member.host_returned and member.durable is False


@pytest.mark.parametrize("path", ["definitive_after_start", "no_timeout"])
def test_a_retained_cleanup_ends_the_run_on_the_unwrapped_dispatch_paths(
    tmp_path, monkeypatch, path
):
    """Calls that skip the timeout wrapper must end the run too, not continue."""
    from tldw_chatbook.Agents.agent_models import RunBudget, ToolCall
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        ToolCatalogRegistry,
        ToolExecutionPolicy,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    invoked = []
    monkeypatch.setattr(
        registry,
        "invoke_by_name",
        lambda name, _args: invoked.append(name) or ToolResult(ok=True, content="ran"),
    )
    if path == "definitive_after_start":
        monkeypatch.setattr(
            registry,
            "execution_policy_for",
            lambda _name: ToolExecutionPolicy.DEFINITIVE_AFTER_START,
        )
    service = AgentService(
        db=AgentRunsDB(tmp_path / "runs.db", client_id="t"),
        registry=registry,
        chat_call=lambda **_kwargs: pytest.fail("no model call"),
    )
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    owner.reserve_tool().retain_unproven("fs_read")
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_tool_call_seconds=0 if path == "no_timeout" else 30),
    )
    with owner.activate():
        invoke_tool = service._make_invoke_tool(
            config, disclosed_names={"calculator"}, run_id="run-1"
        )
    result = invoke_tool(ToolCall(name="calculator", args={"expression": "2+2"}))
    assert result.termination_reason is RunTerminationReason.CLEANUP_UNPROVEN
    assert "cleanup of fs_read could not be proven" in result.error
    assert invoked == []


def test_change_review_snapshot_on_a_retained_owner_fails_closed():
    """Decision: a retained cleanup is not a capacity wait (TASK-40716).

    Waiting would admit the snapshot once the slot is released, and read a
    workspace that may still be changing. Refusing makes the review
    unadmitted at once instead.
    """
    from tldw_chatbook.Agents.execution_capacity import CleanupUnprovenRefused
    from tldw_chatbook.Workspaces.change_review_ownership import _SnapshotSource

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    issuer = SimpleNamespace(
        capacity=capacity,
        activity=SimpleNamespace(validate_opaque_claim=lambda *a, **k: None),
        coordinator=SimpleNamespace(_subscribe_native=lambda *a: None, _publish=None),
    )
    source = _SnapshotSource(
        issuer, None, None, None, None, "c", WorkOrigin.MANUAL, (), (), (), owner=owner
    )
    stopping = owner.reserve_tool()
    stopping.mark_stopping()
    # An ordinary stopping tool is transient: the snapshot waits for a slot.
    assert source._admit_owned() == (None, "capacity")
    stopping.retain_unproven("fs_read")
    with pytest.raises(CleanupUnprovenRefused):
        source._admit_owned()
