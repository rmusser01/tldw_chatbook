"""Live capture waits for app-owned image edits without losing their outcomes."""

import asyncio
import threading
import time

import pytest

from tldw_chatbook.Chat.console_image_edit_operations import (
    ImageEditCompletion,
    ImageEditOperationRegistry,
)


def start(registry, runner, session="session"):
    return registry.start(
        session_id=session,
        attachment_id="attachment",
        captured_draft="unsaved edit prompt",
        cancel_event=threading.Event(),
        runner=runner,
    )


@pytest.mark.asyncio
async def test_maintenance_refuses_new_edits_and_resumes_without_losing_completion():
    registry = ImageEditOperationRegistry()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def runner(generation):
        entered.set()
        await release.wait()
        registry.publish_completion(
            ImageEditCompletion(
                "session", generation, "message", "attachment", "unsaved edit prompt"
            )
        )

    operation = start(registry, runner)
    await entered.wait()
    try:
        registry._maintenance_close_admission()
        assert start(registry, runner, "other") is None
        drain = asyncio.create_task(registry._maintenance_drain(time.monotonic() + 2))
        await asyncio.sleep(0)
        assert not drain.done()
        assert not operation.cancel_event.is_set()
        release.set()
        assert await drain
        registry._maintenance_resume()
        assert registry.completion("session").captured_draft == "unsaved edit prompt"
        next_operation = start(registry, runner, "other")
        assert next_operation is not None
        await next_operation.task
    finally:
        release.set()
        await operation.task


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_session", [False, True])
async def test_timeout_keeps_unsettled_work_owned_even_after_session_is_dropped(
    drop_session,
):
    registry = ImageEditOperationRegistry()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def runner(_generation):
        entered.set()
        await release.wait()

    operation = start(registry, runner)
    await entered.wait()
    try:
        if drop_session:
            registry.drop_session("session")
        registry._maintenance_close_admission()
        assert not await registry._maintenance_drain(time.monotonic())
        assert not operation.task.done()
        assert operation.cancel_event.is_set() is drop_session
        release.set()
        assert await registry._maintenance_drain(time.monotonic() + 2)
    finally:
        release.set()
        await operation.task


@pytest.mark.asyncio
async def test_cancelled_maintenance_wait_does_not_cancel_the_edit():
    registry = ImageEditOperationRegistry()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def runner(_generation):
        entered.set()
        await release.wait()

    operation = start(registry, runner)
    await entered.wait()
    try:
        registry._maintenance_close_admission()
        drain = asyncio.create_task(registry._maintenance_drain(time.monotonic() + 2))
        await asyncio.sleep(0)
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain
        assert not operation.task.done()
        assert not operation.cancel_event.is_set()
        registry._maintenance_resume()
        release.set()
        await operation.task
    finally:
        release.set()
        await operation.task


@pytest.mark.asyncio
async def test_drain_requires_closed_admission():
    with pytest.raises(RuntimeError, match="participant_admission_not_closed"):
        await ImageEditOperationRegistry()._maintenance_drain(time.monotonic() + 2)
