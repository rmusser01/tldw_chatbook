"""Direct tests for the shared recovery-admission runtime (TASK-32860).

The three family suites pin admission behavior end to end (174 green,
A/B-verified against pristine dev); these pin the runtime's own state and
isolation mechanics directly, without touching storage admission.
"""

import pytest

from tldw_chatbook.Backup_Recovery.admission_runtime import (
    ExecutionState,
    RecoveryAdmissionGuard,
    execution_identity,
)


class _Boom(PermissionError):
    pass


@pytest.fixture()
def guard():
    return RecoveryAdmissionGuard("test", error=_Boom, sources=lambda service: ())


def test_execution_identity_binds_task_and_thread(guard) -> None:
    identity = execution_identity()
    assert identity[0] > 0
    assert identity[1] > 0
    # Outside a running loop there is no task; inside asyncio the current
    # task joins the identity, so copied contexts cannot match a foreign one.
    assert identity[2] is None


def test_state_check_refuses_dead_and_forked_states(guard) -> None:
    live = ExecutionState(execution_identity(), {}, ())
    live.check(_Boom)  # alive, same process: passes

    dead = ExecutionState(execution_identity(), {}, ())
    dead.live = False
    with pytest.raises(_Boom):
        dead.check(_Boom)

    forked = ExecutionState((-1, -1, None), {}, ())
    with pytest.raises(_Boom):
        forked.check(_Boom)


def test_state_check_walks_the_parent_chain(guard) -> None:
    dead_parent = ExecutionState(execution_identity(), {}, ())
    dead_parent.live = False
    child = ExecutionState(execution_identity(), {}, (), dead_parent)
    with pytest.raises(_Boom):
        child.check(_Boom)


def test_captured_sources_reuses_only_same_identity_states(guard) -> None:
    state = ExecutionState(execution_identity(), {}, (("config", "p"),))
    token = guard.context.set(state)
    try:
        assert guard.captured_sources(None) == (("config", "p"),)
    finally:
        guard.context.reset(token)

    foreign = ExecutionState((-1, -1, None), {}, ())
    token = guard.context.set(foreign)
    try:
        assert guard.captured_sources(None) == ()  # fresh observation, not borrowed
    finally:
        guard.context.reset(token)


def test_worker_isolation_clears_state_for_the_duration(guard) -> None:
    state = ExecutionState(execution_identity(), {}, ())
    token = guard.context.set(state)
    try:
        with guard.worker_isolation():
            assert guard.context.get() is None
        assert guard.context.get() is state
    finally:
        guard.context.reset(token)
