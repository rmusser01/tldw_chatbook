"""One unproven tool cleanup must not wedge the rest of its run (TASK-40716).

The Console's ordinary tree gives every tool call a source scope. When that
scope's return cannot be proven, the tool worker is retained on purpose: a
resource that may still be live is never reused. This file pins what that
retention does to the run that owns it.
"""

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Agents.agent_service import _call_with_timeout
from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin


class _UnprovenSource:
    """A source scope whose return fails after the tool itself returned."""

    @contextmanager
    def activate(self):
        yield
        raise OSError("controlled: original source return unavailable")


class _OrdinaryTree:
    """The members of an ordinary tree that `_call_with_timeout` reads."""

    def __init__(self, owner):
        self.owner = owner
        self.capacity = owner.capacity
        self.members = {owner: None}
        self.source = _UnprovenSource()

    @contextmanager
    def retain_source_scope(self):
        yield

    def check_local_sources(self):
        return None


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TASK-40716: an unproven cleanup parks the tool worker on an event "
        "nothing completes, so every later tool call in the run is refused "
        "previous_tool_still_running and the app-wide slot stays held"
    ),
)
def test_unproven_source_return_does_not_refuse_every_later_call(
    monkeypatch, record_property
):
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    service = SimpleNamespace(_ordinary_tree=_OrdinaryTree(owner))

    # Observe the retention, and give this test (only) a way to end it.
    release = threading.Event()
    parked = []
    wait = threading.Event.wait

    def observe(event, timeout=None):
        current = threading.current_thread()
        if (
            timeout is None
            and owner._cleanup_unproven
            and current.name.startswith("tool-")
        ):
            parked.append(current)
            return wait(release, 30)
        return wait(event, timeout)

    monkeypatch.setattr(threading.Event, "wait", observe)
    try:
        started = time.monotonic()
        first = _call_with_timeout(
            lambda: ToolResult(ok=True, content="read"),
            1.0,
            "fs_read",
            execution_owner=service,
            owner=owner,
        )
        first_seconds = time.monotonic() - started
        second = _call_with_timeout(
            lambda: ToolResult(ok=True, content="listed"),
            1.0,
            "fs_list",
            execution_owner=service,
            owner=owner,
        )
        owner.finish_root()
        snapshot = capacity.snapshot()
        record_property("first_call_seconds", round(first_seconds, 2))
        record_property("first_error", first.error)
        record_property("second_error", second.error)
        record_property("tool_workers_after_root_finish", snapshot.tool_workers)

        # The reproduction: the worker is retained on an event nothing sets.
        assert parked and parked[0].is_alive() and owner._cleanup_unproven

        # AC#1: later calls run, or the unproven call itself ends the run with
        # a reason that names the unproven cleanup.
        ends_run = first.termination_reason is not None and "cleanup" in first.error
        assert ends_run or "previous_tool_still_running" not in second.error, (
            f"first={first.error!r} (after {first_seconds:.1f}s), "
            f"second={second.error!r}, "
            f"tool_workers={snapshot.tool_workers} after the root finished"
        )
    finally:
        release.set()
        for worker in parked:
            worker.join(5)
