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


def test_model_cancellation_restores_nested_and_concurrent_native_owners():
    from tldw_chatbook.Agents.run_context import (
        current_model_run_cancellation,
        use_model_run_cancellation,
    )

    parent = CurrentRunActor("primary", "parent", None)
    child = CurrentRunActor("subagent", "child", parent.run_id)
    parent_cancel = threading.Event().is_set
    child_cancel = threading.Event().is_set
    seen = []

    def other_thread():
        assert current_model_run_cancellation() is None
        with use_run_actor(child), use_model_run_cancellation(child_cancel):
            seen.append(current_model_run_cancellation())
        assert current_model_run_cancellation() is None

    with use_run_actor(parent), use_model_run_cancellation(parent_cancel):
        with use_run_actor(child), use_model_run_cancellation(child_cancel):
            assert current_model_run_cancellation() == (
                "child",
                "subagent",
                child_cancel,
            )
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(2)
        assert not thread.is_alive()
        assert current_model_run_cancellation() == ("parent", "primary", parent_cancel)
    assert seen == [("child", "subagent", child_cancel)]
    assert current_model_run_cancellation() is None


def test_model_cancellation_rejects_partial_malformed_and_wrong_native_identity():
    import pytest

    from tldw_chatbook.Agents import run_context as context

    actor = CurrentRunActor("primary", "owner", None)
    predicate = threading.Event().is_set

    class EqualToAnything:
        def __eq__(self, other):
            return True

    with use_run_actor(actor):
        for bad in (
            None,
            ("other", "primary", predicate),
            ("owner", "subagent", predicate),
            ("owner", "primary", False),
            (EqualToAnything(), "primary", predicate),
            ("owner", EqualToAnything(), predicate),
            ("owner", "primary"),
        ):
            token = context._MODEL_RUN_CANCELLATION.set(bad)
            try:
                with pytest.raises(
                    ValueError, match="model_run_cancellation_unavailable"
                ):
                    context.current_model_run_cancellation()
            finally:
                context._MODEL_RUN_CANCELLATION.reset(token)
        with (
            pytest.raises(ValueError, match="model_run_cancellation_unavailable"),
            context.use_model_run_cancellation(False),
        ):
            raise AssertionError("invalid predicate admitted")
    token = context._MODEL_RUN_CANCELLATION.set(("owner", "primary", predicate))
    try:
        with pytest.raises(ValueError, match="model_run_cancellation_unavailable"):
            context.current_model_run_cancellation()
    finally:
        context._MODEL_RUN_CANCELLATION.reset(token)
    with (
        use_run_actor(CurrentRunActor("invalid", "owner", None)),
        pytest.raises(ValueError, match="model_run_cancellation_unavailable"),
        context.use_model_run_cancellation(predicate),
    ):
        raise AssertionError("invalid actor admitted")
