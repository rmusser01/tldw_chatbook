"""Admission and physical retirement for one event loop's audio output."""

import asyncio
import copy
from collections.abc import Awaitable, Callable

import pytest

from tldw_chatbook.TTS.audio_output_ownership import (
    AudioOutputLease,
    AudioOutputOwnership,
    OutputBlocked,
    OutputClosed,
    OutputHandoffRequired,
    OutputSuperseded,
)


async def retired() -> bool:
    return True


@pytest.mark.asyncio
async def test_exact_revocation_visible_to_native_worker_before_retirement():
    host = AudioOutputOwnership()
    observed = []

    async def retire_first():
        observed.append(await asyncio.to_thread(lambda: first.revoked))
        return True

    first = await host.acquire("one", retire_first)
    copy.copy(first).revoke()
    assert first.is_current() and not first.revoked
    second = await host.acquire("two", retired)
    assert observed == [True]
    first.revoke()
    assert second.is_current() and not second.revoked
    second.revoke()
    assert second.revoked and not second.is_current()
    assert host.owner == "two"


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


@pytest.mark.asyncio
async def test_eager_reentrant_start_retires_the_actual_successor() -> None:
    output = AudioOutputOwnership()
    callbacks = []
    playlist_start: asyncio.Task[AudioOutputLease] | None = None

    async def retire_playlist() -> bool:
        callbacks.append("playlist")
        return False

    async def retire_old() -> bool:
        nonlocal playlist_start
        callbacks.append("old")
        old.release()
        playlist_start = asyncio.create_task(
            output.acquire("playlist", retire_playlist)
        )
        return True

    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        old = await output.acquire("old", retire_old)
        with pytest.raises(RuntimeError):
            await output.acquire("preview", retired)
        assert playlist_start is not None
        playlist = await playlist_start
        assert playlist.is_current()
        assert callbacks == ["old"]

        with pytest.raises(RuntimeError):
            await output.acquire("console", retired)
        assert callbacks == ["old", "playlist"]
        assert output.owner == "playlist"
        assert output.blocked
        assert not playlist.is_current()
    finally:
        loop.set_task_factory(previous_factory)


@pytest.mark.asyncio
async def test_pending_intents_order_preparations_without_displacing_active_audio() -> (
    None
):
    host = AudioOutputOwnership()
    active = await host.acquire("speech", retired)
    slower = host.begin_intent()
    no_output = host.begin_intent()
    assert not slower.is_current()
    with pytest.raises(OutputSuperseded):
        await host.acquire("stale preparation", retired, intent=slower)
    no_output.cancel()  # Validation refusal or already at logical EOF.
    assert active.is_current() and not active.revoked
    next_play = host.begin_intent()
    with pytest.raises(OutputSuperseded):
        await host.acquire(
            "foreign", retired, intent=AudioOutputOwnership().begin_intent()
        )
    copy.copy(next_play).cancel()
    with pytest.raises(OutputSuperseded):
        await host.acquire("copy", retired, intent=copy.copy(next_play))
    assert next_play.is_current() and active.is_current()
    assert (await host.acquire("next", retired, intent=next_play)).is_current()


@pytest.mark.asyncio
async def test_silent_probes_reserve_action_order_until_they_can_request_sound() -> (
    None
):
    host = AudioOutputOwnership()
    active = await host.acquire("speech", retired)
    older = host.begin_intent(tentative=True)
    sound = host.begin_intent()
    silent = host.begin_intent(tentative=True)
    pending = host.begin_intent(tentative=True)
    assert sound.is_current() and not silent.is_current()
    assert not older.promote() and not copy.copy(pending).promote()
    silent.cancel()
    assert not silent.promote()
    with pytest.raises(OutputSuperseded):
        await host.acquire("tentative", retired, intent=pending)
    assert sound.is_current() and active.is_current()
    assert pending.promote() and pending.is_current()
    assert not sound.is_current() and active.is_current()
    pending.cancel()
    assert not older.promote() and not sound.promote()  # No resurrection.


@pytest.mark.asyncio
async def test_exact_stop_synchronously_fences_and_joins_the_pending_transfer() -> None:
    host = AudioOutputOwnership()
    held = HeldRetirement()
    active = await host.acquire("old", held.retire)
    assert not await host.stop(copy.copy(active))
    assert active.is_current()
    cancelled = host.stop(active)
    assert active.revoked and host.owner == "old"
    cancelled.cancel()
    await held.started.wait()
    next_owner = await start_acquire(host, "new")
    joined = host.stop(active)
    assert not next_owner.done() and held.calls == 1
    held.confirmation.set_result(True)
    assert await joined
    successor = await next_owner
    assert successor.is_current() and not await host.stop(active)
    assert host.owner == "new"


@pytest.mark.asyncio
async def test_blocked_owner_can_only_retry_through_an_explicit_exact_stop() -> None:
    host = AudioOutputOwnership()
    calls = []
    retry = HeldRetirement()

    async def cleanup():
        calls.append(True)
        return False if len(calls) == 1 else await retry.retire()

    old = await host.acquire("resistant", cleanup)
    with pytest.raises(OutputBlocked) as failure:
        await host.acquire("new", retired)
    assert failure.value.owner == "resistant" and host.blocked
    with pytest.raises(OutputBlocked):
        await host.acquire("other", retired)
    assert len(calls) == 1
    a, b = host.stop(old), host.stop(old)
    await retry.started.wait()
    retry.confirmation.set_result(True)
    assert await a and await b
    assert len(calls) == 2 and not host.blocked and host.owner is None


@pytest.mark.asyncio
async def test_late_external_acceptance_survives_retirement_and_fences_in_app_start() -> (
    None
):
    host = AudioOutputOwnership()
    held = HeldRetirement()
    launcher = await host.acquire("external video", held.retire, external=True)
    ticket = host.begin_intent()
    pending = asyncio.create_task(host.acquire("pcm", retired, intent=ticket))
    await held.started.wait()
    assert launcher.revoked and host.handoff_epoch is None
    copy.copy(launcher).record_handoff()
    assert host.handoff_epoch is None
    launcher.record_handoff()  # OS accepted while committed launch was retiring.
    epoch = host.handoff_epoch
    assert epoch is not None
    held.confirmation.set_result(True)
    with pytest.raises(OutputHandoffRequired) as refused:
        await pending
    assert refused.value.epoch is epoch and host.owner is None
    launcher.record_handoff()  # A released old callback cannot add a new epoch.
    assert host.handoff_epoch is epoch
    with pytest.raises(OutputHandoffRequired):
        await host.acquire("pcm", retired, intent=ticket)
    assert not host.acknowledge_handoff(copy.copy(ticket), epoch)
    assert not host.acknowledge_handoff(ticket, object())
    assert host.acknowledge_handoff(ticket, epoch)
    assert host.handoff_epoch is None
    assert (await host.acquire("pcm", retired, intent=ticket)).is_current()


@pytest.mark.asyncio
async def test_explicit_external_opens_advance_epoch_and_reject_old_confirmation() -> (
    None
):
    host = AudioOutputOwnership()
    first = await host.acquire("external file", retired, external=True)
    first.record_handoff()
    old_epoch = host.handoff_epoch
    refused_ticket = host.begin_intent()
    with pytest.raises(OutputHandoffRequired):
        await host.acquire("file", retired, intent=refused_ticket)
    assert first.is_current()  # No displacement merely to discover this barrier.
    second = await host.acquire("external video", retired, external=True)
    second.record_handoff()
    epoch = host.handoff_epoch
    assert epoch is not None and epoch is not old_epoch
    ticket = host.begin_intent()
    assert not host.acknowledge_handoff(refused_ticket, epoch)
    assert not host.acknowledge_handoff(ticket, old_epoch)
    assert await host.stop(second) and host.handoff_epoch is epoch
    ticket.cancel()
    assert not host.acknowledge_handoff(ticket, epoch)
    assert await host.begin_shutdown()
    assert host.handoff_epoch is epoch


@pytest.mark.asyncio
async def test_shutdown_fences_pending_first_grant_and_joins_a_cancelled_observer() -> (
    None
):
    host = AudioOutputOwnership()
    held = HeldRetirement()
    old = await host.acquire("active", held.retire)
    ticket = host.begin_intent()
    awaiting_grant = asyncio.create_task(
        host.acquire("pending", retired, intent=ticket)
    )
    await held.started.wait()
    observation = host.begin_shutdown()
    assert host.closing and old.revoked and not ticket.is_current()
    observation.cancel()
    with pytest.raises(OutputClosed):
        host.begin_intent()
    with pytest.raises(OutputClosed):
        await host.acquire("too late", retired)
    again = host.begin_shutdown()
    assert not again.done() and held.calls == 1
    held.confirmation.set_result(True)
    with pytest.raises(OutputClosed):
        await awaiting_grant
    assert await again and await host.begin_shutdown()
    assert held.calls == 1 and host.owner is None


@pytest.mark.asyncio
async def test_shutdown_before_any_grant_fences_prepared_and_tentative_actions() -> (
    None
):
    host = AudioOutputOwnership()
    ready = host.begin_intent()
    probe = host.begin_intent(tentative=True)
    pending = host.acquire("prepared", retired, intent=ready)
    shutdown = host.begin_shutdown()
    assert not ready.is_current() and not probe.promote()
    with pytest.raises(OutputClosed):
        await pending
    assert await shutdown and host.owner is None


@pytest.mark.asyncio
async def test_failed_shutdown_retains_the_exact_blocked_owner() -> None:
    host = AudioOutputOwnership()
    held = HeldRetirement()

    async def cleanup():
        if held.finished.is_set():
            return True
        return await held.retire()

    old = await host.acquire("resistant", cleanup)
    shutdown = host.begin_shutdown()
    assert old.revoked
    await held.started.wait()
    held.confirmation.set_result(False)
    assert not await shutdown and not await host.begin_shutdown()
    assert host.blocked and host.owner == "resistant" and held.calls == 1
    assert await host.stop(old)
    assert host.closing and not host.blocked and host.owner is None


@pytest.mark.asyncio
async def test_new_preparation_supersedes_a_transfer_already_waiting_for_retirement() -> (
    None
):
    host = AudioOutputOwnership()
    held = HeldRetirement()
    await host.acquire("old", held.retire)
    transfer = await start_acquire(host, "prepared")
    await held.started.wait()
    newest = host.begin_intent()
    held.confirmation.set_result(True)
    with pytest.raises(OutputSuperseded):
        await transfer
    assert newest.is_current() and host.owner is None


@pytest.mark.asyncio
async def test_handoff_acknowledgement_requires_an_issued_ticket() -> None:
    host = AudioOutputOwnership()
    ticket = host.begin_intent()
    launcher = await host.acquire("external", retired, intent=ticket, external=True)
    launcher.record_handoff()
    epoch = host.handoff_epoch
    ticket.cancel()
    assert not host.acknowledge_handoff(None, epoch)
    assert host.handoff_epoch is epoch


@pytest.mark.asyncio
async def test_reusing_one_ticket_for_fallback_only_grants_the_latest_attempt() -> None:
    host = AudioOutputOwnership()
    held = HeldRetirement()
    await host.acquire("old", held.retire)
    ticket = host.begin_intent()
    original = asyncio.create_task(host.acquire("first", retired, intent=ticket))
    await held.started.wait()
    asyncio.get_running_loop().call_soon(held.confirmation.set_result, True)
    fallback = await host.acquire("fallback", retired, intent=ticket)
    with pytest.raises(OutputSuperseded):
        await original
    assert held.calls == 1 and fallback.is_current()


@pytest.mark.asyncio
async def test_only_the_latest_attempt_on_one_ticket_can_request_handoff_ack() -> None:
    host = AudioOutputOwnership()
    held = HeldRetirement()
    old = await host.acquire("external", held.retire, external=True)
    ticket = host.begin_intent()
    earlier = asyncio.create_task(host.acquire("first", retired, intent=ticket))
    await held.started.wait()
    loop = asyncio.get_running_loop()
    loop.call_soon(old.record_handoff)
    loop.call_soon(held.confirmation.set_result, True)
    with pytest.raises(OutputHandoffRequired) as latest:
        await host.acquire("fallback", retired, intent=ticket)
    with pytest.raises(OutputSuperseded):
        await earlier
    assert latest.value.epoch is host.handoff_epoch


@pytest.mark.asyncio
async def test_shutdown_joins_eager_retirement_even_after_callback_releases_lease() -> (
    None
):
    host = AudioOutputOwnership()
    held = HeldRetirement()

    async def cleanup():
        old.release()
        return await held.retire()

    old = await host.acquire("old", cleanup)
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        stopped = host.stop(old)
        assert old.revoked
        await held.started.wait()
        assert host.owner is None
        shutdown = host.begin_shutdown()
        assert not shutdown.done()
        stopped.cancel()
        held.confirmation.set_result(True)
        assert await shutdown and held.calls == 1
    finally:
        loop.set_task_factory(previous_factory)
