from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.TTS._async_lifecycle import join_retained_task


@pytest.mark.asyncio
async def test_failure_after_cancellation_callback_receives_both_errors() -> None:
    cleanup_started = asyncio.Event()
    allow_cleanup_failure = asyncio.Event()
    cleanup_error = RuntimeError("retained cleanup failed")
    observed_errors: list[BaseException] = []

    async def fail_cleanup() -> None:
        cleanup_started.set()
        await allow_cleanup_failure.wait()
        raise cleanup_error

    retained_task = asyncio.create_task(fail_cleanup())
    waiter = asyncio.create_task(
        join_retained_task(
            retained_task,
            on_failure_after_cancellation=lambda *errors: observed_errors.extend(
                errors
            ),
        )
    )
    await cleanup_started.wait()

    waiter.cancel("caller cancelled")
    await asyncio.sleep(0)
    allow_cleanup_failure.set()

    with pytest.raises(asyncio.CancelledError) as cancellation:
        await waiter

    assert observed_errors == [cancellation.value, cleanup_error]
