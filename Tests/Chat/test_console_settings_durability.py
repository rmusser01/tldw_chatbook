"""Application-lifetime Console settings durability contracts."""

from __future__ import annotations

import asyncio
import threading

import pytest

from tldw_chatbook.Chat.console_settings_durability import (
    ConsoleSettingsDurabilityOwner,
)


@pytest.mark.asyncio
async def test_close_fences_admission_and_waits_for_lease_transfer() -> None:
    owner = ConsoleSettingsDurabilityOwner()
    lease = owner.try_acquire()
    assert lease is not None

    draining = asyncio.create_task(owner.close_and_drain())
    await asyncio.sleep(0)
    assert owner.try_acquire() is None
    assert not draining.done()

    finished = asyncio.Event()

    async def admitted_work() -> None:
        finished.set()

    owner.launch(lease, admitted_work(), name="admitted-settings")
    await draining

    assert finished.is_set()
    assert owner.tasks == set()


@pytest.mark.asyncio
async def test_cancelled_drain_does_not_cancel_admitted_thread_work() -> None:
    owner = ConsoleSettingsDurabilityOwner()
    lease = owner.try_acquire()
    assert lease is not None
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def blocking_write() -> None:
        started.set()
        assert release.wait(timeout=5)
        completed.set()

    async def admitted_work() -> None:
        await asyncio.to_thread(blocking_write)

    task = owner.launch(lease, admitted_work(), name="thread-settings")
    assert await asyncio.to_thread(started.wait, 1)
    draining = asyncio.create_task(owner.close_and_drain())
    draining.cancel()
    with pytest.raises(asyncio.CancelledError):
        await draining

    assert not task.cancelled()
    release.set()
    await owner.close_and_drain()
    assert completed.is_set()


@pytest.mark.asyncio
async def test_admission_registers_before_a_configured_task_factory_can_step_work() -> (
    None
):
    """A custom factory must not run storage before its owner retains the task."""
    owner = ConsoleSettingsDurabilityOwner()
    lease = owner.try_acquire()
    assert lease is not None
    started = []

    async def admitted_work():
        started.append(True)
        assert asyncio.current_task() in owner.tasks
        return "saved"

    def eager_then_raise(_loop, coroutine, **_kwargs):
        coroutine.send(None)
        raise RuntimeError("factory lost the started task")

    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(eager_then_raise)
    try:
        task = owner.launch(lease, admitted_work(), name="registered-settings")
        assert started == []
    finally:
        loop.set_task_factory(previous_factory)
    assert await task == "saved"
    await owner.close_and_drain()
    assert owner.tasks == set()
