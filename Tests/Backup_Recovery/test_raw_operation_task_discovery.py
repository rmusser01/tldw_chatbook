"""A raw operation is discovered only by the task that owns it (TASK-33641.6).

ADR-126 exposes an operation "on the same PID, Thread and Task". The discovery
slot was a plain thread-local, so every other asyncio task on that thread found
the owner's operation and had its own guarded IO refused by ``_check``.

Only config is entered by a second task here. A task that enters any other
source, or any other file, that another task of its thread is in the middle of
is still refused: ``test_mcp_source_lifetimes.py::
test_history_private_helpers_reject_other_paths_and_foreign_task`` for one
source object, and the last two tests here for two objects that share a lock:
one file, or one settings root.

A config write is the one thing a second task cannot wait for: the OS file
lock is not reentrant, and its holder cannot run while the waiter sleeps on
their shared thread. Every call that takes the write lock is refused at once,
before it opens an operation.
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
            # The owner's discovery and its strict check survive the other
            # task's read and its write.
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
            # The holder has a read scope, not the write lock, so a setting
            # saved in this window lands. Before, it was dropped (False).
            saved = source.save_setting_to_cli_config(
                "general", "users_name", "second task"
            )
            discoveries = list(seen)
            assert raw._local.operation is None
            assert raw._runtime_operation() is None
            # Not weakened: presenting the owner's operation is still refused.
            with pytest.raises(
                bootstrap.RecoveryRequired, match="raw_operation_provenance_invalid"
            ):
                raw._check(operation)
            return asyncio.current_task(), text, saved, discoveries
        finally:
            release.set()

    _, (reading_task, text, saved, discoveries) = await asyncio.gather(
        holder(), reader()
    )

    assert 'users_name = "configured"' in text
    assert saved is True
    assert 'users_name = "second task"' in source.get_cli_config_path().read_text()
    assert discoveries, "neither the read nor the save reached guarded IO"
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


_WRITE_LOCK_CALLS = {
    "save_setting": lambda source: source.save_setting_to_cli_config(
        "general", "users_name", "nested"
    ),
    # These two open an operation of their own around the write lock.
    "read_snapshot": lambda source: source.read_cli_config_snapshot(),
    "replace_serialized": lambda source: source.replace_cli_config_serialized(
        '[general]\nusers_name = "nested"\n', create_backup=False
    ),
}


@pytest.mark.parametrize("call", _WRITE_LOCK_CALLS)
async def test_config_write_from_a_task_started_inside_the_write_lock_is_refused_at_once(
    source, monkeypatch, call
):
    """The write lock runs caller code (a mutation builder, a hooks snapshot).

    A task that code starts eagerly could reenter both thread locks, and would
    then sleep on the OS file lock its own thread holds: the loop frozen for
    good. The write is refused before it enters anything, as it was when a
    second task could not enter config at all (before TASK-33641.6).

    A refusal raised inside an operation the call had already opened is
    recorded as a failed config operation, and the config participant then
    cannot drain until the next write that succeeds.
    """
    path = source.get_cli_config_path()
    opened = []
    open_operation = config_participants.operation

    def operation(*args, **kwargs):
        opened.append(asyncio.current_task())
        return open_operation(*args, **kwargs)

    monkeypatch.setattr(config_participants, "operation", operation)
    original = source.portalocker.lock

    def lock(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except source.portalocker.exceptions.AlreadyLocked:
            pytest.fail("the write waited for a lock its own thread holds")

    monkeypatch.setattr(source.portalocker, "lock", lock)

    async def writer():
        try:
            outcome = _WRITE_LOCK_CALLS[call](source)
        except bootstrap.RecoveryRequired as refusal:
            outcome = refusal.args
        return asyncio.current_task(), outcome

    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        with source._config_write_lock(path):
            operation = raw._local.operation
            registered = set(storage._raw_operations)
            error = source._CONFIG_PERSISTENCE_ERROR
            started = asyncio.create_task(writer())
            assert started.done(), "the writer's first step ran outside the lock"
            writing_task, outcome = started.result()
            assert writing_task is started is not asyncio.current_task()
            # The save reports a refusal as False; the other two raise it.
            assert outcome == (
                False
                if call == "save_setting"
                else ("raw_operation_provenance_invalid",)
            )
            # Refused before entry: no operation opened, no failure recorded.
            assert started not in opened
            assert set(storage._raw_operations) == registered
            assert source._CONFIG_PERSISTENCE_ERROR is error
            # The holder still owns its operation and the lock.
            assert raw._local.operation is operation
            assert raw._check(operation).task is asyncio.current_task()
    finally:
        loop.set_task_factory(previous)

    assert "nested" not in path.read_text()
    # Nothing is left held: the next write gets the lock.
    assert source.save_setting_to_cli_config("general", "users_name", "after")
    assert 'users_name = "after"' in path.read_text()


async def test_second_task_is_refused_a_file_another_source_object_holds(
    tmp_path, local_root
):
    """Source locks are keyed by file, not by source object.

    Two objects bound to one file share one reentrant lock, the store and its
    ``.tmp`` member. The second task is refused, as with a single object.
    """
    from tldw_chatbook.Feedback_Interop.local_feedback_service import (
        LocalFeedbackService,
    )

    store = tmp_path / "feedback.json"
    first = LocalFeedbackService(store_path=store)
    second = LocalFeedbackService(store_path=store)

    async def other():
        with (
            pytest.raises(
                bootstrap.RecoveryRequired, match="raw_operation_provenance_invalid"
            ),
            raw._scope(second, "service", writing=True),
        ):
            pytest.fail("a second task entered a file another task is writing")

    with raw._scope(first, "service", writing=True) as operation:
        await asyncio.create_task(other())
        assert raw._check(operation).task is asyncio.current_task()
    # Control: it was the other task's hold that refused it, not the object.
    with raw._scope(second, "service", writing=True) as operation:
        assert raw._check(operation).source is second


async def test_second_task_is_refused_another_file_under_one_settings_root(
    tmp_path, local_root
):
    """A settings source's lock is keyed by its bound root, not by the file.

    Two loaders on one config file share one reentrant lock whichever file each
    scope selects, so comparing the selected files cannot tell them apart.
    """
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    bound = tmp_path / "eval.yaml"
    first, second = EvalConfigLoader(str(bound)), EvalConfigLoader(str(bound))
    export = tmp_path / "export.yaml"

    async def other():
        with (
            pytest.raises(
                bootstrap.RecoveryRequired, match="raw_operation_provenance_invalid"
            ),
            raw._scope(second, "eval_config", writing=True, selected_read=export),
        ):
            pytest.fail("a second task entered a settings root another task is in")

    with raw._scope(first, "eval_config", writing=True) as operation:
        assert raw._selected(operation) != export
        await asyncio.create_task(other())
        assert raw._check(operation).task is asyncio.current_task()
    # Control: it was the other task's hold that refused it, not the export.
    with raw._scope(
        second, "eval_config", writing=True, selected_read=export
    ) as operation:
        assert raw._check(operation).source is second
