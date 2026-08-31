"""Trusted run-actor context is exact, scoped, and fail-closed."""

import threading

from tldw_chatbook.Agents.run_context import (
    CurrentRunActor,
    current_run_actor,
    current_run_id,
    use_run_actor,
)


def test_unbound_actor_is_none_in_direct_and_fresh_thread_access():
    """No caller outside an agent run is silently promoted to primary."""
    seen: list[tuple[CurrentRunActor | None, str]] = []

    worker = threading.Thread(
        target=lambda: seen.append((current_run_actor(), current_run_id())),
        daemon=True,
    )
    worker.start()
    worker.join()

    assert not worker.is_alive()
    assert current_run_actor() is None
    assert current_run_id() == ""
    assert seen == [(None, "")]


def test_nested_actor_bindings_restore_exact_outer_actor_lifo():
    """An inline child cannot leak its identity into its parent or caller."""
    parent = CurrentRunActor("primary", "run-parent", None)
    child = CurrentRunActor("subagent", "run-child", parent.run_id)

    assert current_run_actor() is None
    with use_run_actor(parent):
        assert current_run_actor() == parent
        with use_run_actor(child):
            assert current_run_actor() == child
        assert current_run_actor() == parent
    assert current_run_actor() is None
