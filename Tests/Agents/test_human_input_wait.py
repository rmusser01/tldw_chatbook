"""ADR-067: the run-keyed human-input wait registry.

A blocking human prompt (tool approval card, skill install confirm, skill
script confirm) marks its run while a decision is pending so
``AgentService._call_with_timeout`` can pause the per-call wall clock:
machine budget must not tick while a human is being waited on. Mirrors
``run_context``'s dependency-light, stdlib-only discipline.
"""

from __future__ import annotations

import threading

from tldw_chatbook.Agents.human_input_wait import (
    human_input_wait_active,
    use_human_input_wait,
)


def test_not_active_without_a_wait():
    assert human_input_wait_active("run-1") is False


def test_active_inside_context_and_cleared_after():
    with use_human_input_wait("run-1"):
        assert human_input_wait_active("run-1") is True
    assert human_input_wait_active("run-1") is False


def test_cleared_when_wait_ends_via_exception():
    """A round torn down mid-wait (cancel path) must drop its mark too --
    a leaked mark would freeze the per-call clock for that run's next
    tool call forever."""
    try:
        with use_human_input_wait("run-1"):
            raise RuntimeError("round torn down mid-wait")
    except RuntimeError:
        pass
    assert human_input_wait_active("run-1") is False


def test_wait_is_visible_across_threads():
    """The mark is SET on the round's worker thread and POLLED on the
    wrapper's thread, so the registry must be thread-safe process state --
    a ContextVar set on the worker would be invisible where the deadline
    is checked (the trap ``run_context`` documents for its own bindings)."""
    seen: list[bool] = []
    observed = threading.Event()

    def observer() -> None:
        seen.append(human_input_wait_active("run-1"))
        observed.set()

    with use_human_input_wait("run-1"):
        thread = threading.Thread(target=observer)
        thread.start()
        assert observed.wait(timeout=5.0), "observer never ran"
        thread.join(timeout=5.0)
        # A join(timeout=...) that returns says nothing about termination,
        # and a leaked non-daemon thread hangs interpreter shutdown (the
        # exact failure mode in the TASK-16789 lesson).
        assert not thread.is_alive(), "observer thread did not terminate"

    assert seen == [True]
    assert human_input_wait_active("run-1") is False


def test_waits_for_different_runs_are_independent():
    """Concurrent runs are the norm (fleet children share one session);
    one run's pending decision must not pause a sibling's clock."""
    with use_human_input_wait("run-1"):
        assert human_input_wait_active("run-2") is False
        with use_human_input_wait("run-2"):
            assert human_input_wait_active("run-2") is True
        assert human_input_wait_active("run-2") is False
        assert human_input_wait_active("run-1") is True
    assert human_input_wait_active("run-1") is False


def test_concurrent_same_run_waits_refcount():
    """Two rounds for ONE run may overlap (batch review hook + single-call
    fallback approval); the run stays marked until BOTH end."""
    with use_human_input_wait("run-1"):
        with use_human_input_wait("run-1"):
            assert human_input_wait_active("run-1") is True
        assert human_input_wait_active("run-1") is True
    assert human_input_wait_active("run-1") is False


def test_falsy_run_id_is_normalized():
    """``""`` is the no-run key (a round armed outside any agent run, e.g.
    the MCP workbench Test Tool); it marks and clears like any other,
    never crashes, and ``None`` reads the same slot."""
    with use_human_input_wait(None):
        assert human_input_wait_active("") is True
        assert human_input_wait_active(None) is True
    assert human_input_wait_active("") is False
