"""A raw operation is discovered only by the task that owns it (TASK-33641.6).

ADR-126 exposes an operation "on the same PID, Thread and Task". The discovery
slot was a plain thread-local, so every other asyncio task on that thread found
the owner's operation and had its own guarded IO refused by ``_check``.

Only config is entered by a second task here. A task that enters any other
source another task of its thread is in the middle of is still refused:
``test_mcp_source_lifetimes.py::
test_history_private_helpers_reject_other_paths_and_foreign_task``.
"""

import asyncio

import pytest

from Tests.Backup_Recovery.test_config_participant_lifetimes import (
    source as source,  # noqa: PLC0414
)
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.Backup_Recovery import bootstrap, config_participants
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage


def _record_discoveries(monkeypatch):
    """Record each guarded-IO discovery as (operation, its owner task, the caller)."""
    original = raw._runtime_operation
    seen = []

    def discover(path=None):
        operation = original(path)
        if operation is not None:
            seen.append(
                (operation, raw._states[operation].task, asyncio.current_task())
            )
        return operation

    monkeypatch.setattr(raw, "_runtime_operation", discover)
    return seen


async def test_guarded_config_read_ignores_a_scope_another_task_holds_across_an_await(
    source, monkeypatch
):
    seen = _record_discoveries(monkeypatch)
    # Earlier files in this worker may have left retained operations behind.
    before = set(storage._raw_operations)
    held, release = asyncio.Event(), asyncio.Event()
    owner = {}

    async def holder():
        with config_participants.operation(source) as operation:
            owner.update(operation=operation, task=asyncio.current_task())
            held.set()
            await release.wait()
            # The owner's discovery and its strict check survive the other task.
            assert raw._local.operation is operation
            assert raw._check(operation).task is asyncio.current_task()
        assert raw._local.operation is None

    async def reader():
        await held.wait()
        try:
            operation = owner["operation"]
            state = raw._states[operation]
            assert state.active and operation in storage._raw_operations
            assert state.participant is not None, "the held scope is not registered"
            seen.clear()
            text = source.read_cli_config_serialized()
            discoveries = list(seen)
            assert raw._local.operation is None
            assert raw._runtime_operation() is None
            # Not weakened: presenting the owner's operation is still refused.
            with pytest.raises(
                bootstrap.RecoveryRequired, match="raw_operation_provenance_invalid"
            ):
                raw._check(operation)
            return asyncio.current_task(), text, discoveries
        finally:
            release.set()

    _, (reading_task, text, discoveries) = await asyncio.gather(holder(), reader())

    assert 'users_name = "configured"' in text
    assert discoveries, "the read never reached guarded IO"
    assert reading_task is not owner["task"]
    for operation, owning_task, caller in discoveries:
        assert operation is not owner["operation"]
        assert owning_task is reading_task and caller is reading_task
    assert set(storage._raw_operations) == before


async def test_task_started_eagerly_inside_a_config_scope_reads_under_its_own_operation(
    source, monkeypatch
):
    """The boot shape: ``App.run_async`` installs ``asyncio.eager_task_factory``.

    A worker created inside the Console's synchronous config scope then runs
    its first step inside that scope, as another task, with no await anywhere.
    ``run_test`` installs no eager factory, so mounted tests never saw it.
    """
    seen = _record_discoveries(monkeypatch)
    before = set(storage._raw_operations)

    async def worker():
        return asyncio.current_task(), source.read_cli_config_serialized()

    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        with config_participants.operation(source) as operation:
            started = asyncio.create_task(worker())
            assert started.done(), "the worker's first step ran outside the scope"
            worker_task, text = started.result()
            assert raw._local.operation is operation
            assert raw._check(operation).task is asyncio.current_task()
    finally:
        loop.set_task_factory(previous)

    assert 'users_name = "configured"' in text
    assert worker_task is started is not asyncio.current_task()
    assert seen, "the read never reached guarded IO"
    for discovered, owning_task, caller in seen:
        assert discovered is not operation
        assert owning_task is started and caller is started
    assert raw._local.operation is None
    assert set(storage._raw_operations) == before
