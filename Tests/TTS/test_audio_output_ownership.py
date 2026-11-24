"""Admission and physical retirement for one event loop's audio output."""

import asyncio
import copy
from collections.abc import Awaitable, Callable

import pytest

from tldw_chatbook.TTS.audio_output_ownership import (
    AudioOutputLease,
    AudioOutputOwnership,
)


async def retired() -> bool:
    return True


class HeldRetirement:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.confirmation = asyncio.get_running_loop().create_future()
        self.calls = 0

    async def retire(self) -> bool:
        self.calls += 1
        self.started.set()
        try:
            return await self.confirmation
        finally:
            self.finished.set()


async def start_acquire(
    output: AudioOutputOwnership,
    owner: str,
    retire: Callable[[], Awaitable[bool]] = retired,
) -> asyncio.Task[AudioOutputLease]:
    entered = asyncio.Event()

    async def acquire() -> AudioOutputLease:
        entered.set()
        return await output.acquire(owner, retire)

    task = asyncio.create_task(acquire())
    await entered.wait()
    return task


@pytest.mark.asyncio
async def test_only_the_issued_lease_can_release_its_slot() -> None:
    output = AudioOutputOwnership()
    assert output.owner is None
    assert not output.blocked

    first = await output.acquire("console", retired)
    duplicate = copy.copy(first)
    assert not duplicate.is_current()
    duplicate.release()
    assert first.is_current()
    assert output.owner == "console"

    first.release()
    first.release()
    assert not first.is_current()
    assert output.owner is None
    second = await output.acquire("console", retired)
    first.release()
    assert second.generation > first.generation
    assert second.is_current()


@pytest.mark.asyncio
async def test_same_label_replacement_revokes_authority_before_waiting_for_silence() -> (
    None
):
    output = AudioOutputOwnership()
    retirement = HeldRetirement()
    first = await output.acquire("preview", retirement.retire)
    replacement = await start_acquire(output, "preview")
    assert not replacement.done()
    await retirement.started.wait()
    assert not first.is_current()
    assert output.owner == "preview"
    assert not output.blocked

    retirement.confirmation.set_result(True)
    second = await replacement
    assert second.generation > first.generation
    first.release()
    assert second.is_current()
    assert output.owner == "preview"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [False, OSError("cleanup failed"), asyncio.CancelledError()]
)
async def test_failed_retirement_keeps_custody_until_exact_acknowledgement(
    failure: bool | BaseException,
) -> None:
    output = AudioOutputOwnership()
    retirement = HeldRetirement()
    first = await output.acquire("console", retirement.retire)
    attempt = await start_acquire(output, "article")
    assert not attempt.done()
    await retirement.started.wait()
    if isinstance(failure, BaseException):
        retirement.confirmation.set_exception(failure)
    else:
        retirement.confirmation.set_result(failure)
    with pytest.raises(RuntimeError):
        await attempt

    assert not first.is_current()
    assert output.blocked
    assert output.owner == "console"
    copy.copy(first).release()
    with pytest.raises(RuntimeError):
        await output.acquire("library", retired)
    assert retirement.calls == 1

    first.release()
    assert output.owner is None
    assert not output.blocked
    successor = await output.acquire("library", retired)
    first.release()
    assert successor.is_current()


@pytest.mark.asyncio
async def test_only_the_newest_waiter_can_start_after_shared_retirement() -> None:
    output = AudioOutputOwnership()
    retirement = HeldRetirement()
    first = await output.acquire("old", retirement.retire)
    earlier = await start_acquire(output, "earlier")
    await retirement.started.wait()
    latest = await start_acquire(output, "latest")
    assert not latest.done()
    assert output.owner == "old"
    retirement.confirmation.set_result(True)

    with pytest.raises(RuntimeError):
        await earlier
    successor = await latest
    first.release()
    assert retirement.calls == 1
    assert output.owner == "latest"
    assert successor.is_current()


@pytest.mark.asyncio
async def test_cancelled_acquire_leaves_cleanup_running_for_the_next_request() -> None:
    output = AudioOutputOwnership()
    retirement = HeldRetirement()
    first = await output.acquire("old", retirement.retire)
    cancelled = await start_acquire(output, "cancelled")
    await retirement.started.wait()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert not first.is_current()
    assert output.owner == "old"
    assert not retirement.finished.is_set()
    successor = await start_acquire(output, "successor")
    assert not successor.done()

    retirement.confirmation.set_result(True)
    assert (await successor).is_current()
    assert retirement.calls == 1
    assert output.owner == "successor"


@pytest.mark.asyncio
async def test_cancelling_latest_does_not_restore_an_earlier_start_intent() -> None:
    output = AudioOutputOwnership()
    retirement = HeldRetirement()
    await output.acquire("old", retirement.retire)
    earlier = await start_acquire(output, "earlier")
    await retirement.started.wait()
    latest = await start_acquire(output, "latest")
    latest.cancel()
    with pytest.raises(asyncio.CancelledError):
        await latest
    retirement.confirmation.set_result(True)
    with pytest.raises(RuntimeError):
        await earlier
    assert output.owner is None
    assert not output.blocked


@pytest.mark.asyncio
@pytest.mark.parametrize("simultaneous", [False, True])
async def test_cleanup_failure_survives_waiter_cancellation(simultaneous: bool) -> None:
    output = AudioOutputOwnership()
    retirement = HeldRetirement()
    first = await output.acquire("old", retirement.retire)
    attempt = await start_acquire(output, "cancelled")
    await retirement.started.wait()
    if simultaneous:
        retirement.confirmation.set_exception(OSError("failed before cancellation"))
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    if not simultaneous:
        retirement.confirmation.set_exception(OSError("failed after cancellation"))
    await retirement.finished.wait()

    with pytest.raises(RuntimeError):
        await output.acquire("successor", retired)
    assert output.blocked
    assert output.owner == "old"
    assert not first.is_current()
    assert retirement.calls == 1
    first.release()
    assert (await output.acquire("recovered", retired)).is_current()
