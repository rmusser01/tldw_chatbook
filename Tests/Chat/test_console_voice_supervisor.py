from __future__ import annotations

import asyncio
from tldw_chatbook.Chat.console_voice_worker import ConsoleVoiceWorker
import gc
import threading
from types import SimpleNamespace
import weakref

import pytest

from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_voice_supervisor import (
    VoiceDispatchKind,
    VoiceDispatchQuarantined,
    VoiceDispatchSupervisor,
)


@pytest.mark.asyncio
async def test_pending_disposition_transfers_one_slot_without_releasing_survivor():
    supervisor = VoiceDispatchSupervisor()
    first = asyncio.get_running_loop().create_future()
    second = asyncio.get_running_loop().create_future()
    survivor = asyncio.get_running_loop().create_future()
    supervisor.retain_pending_cleanup(first)
    supervisor.retain_pending_cleanup(second)
    with pytest.raises(VoiceDispatchQuarantined):
        supervisor.retain_orphan(survivor)
    assert supervisor.orphan_count == 0
    supervisor.retain_orphan(survivor, pending_cleanup=first)
    assert supervisor.orphan_count == 1
    supervisor.release_pending_cleanup(
        first
    )  # Old disposition cannot release its survivor.
    supervisor.release_pending_cleanup(second)
    with pytest.raises(VoiceDispatchQuarantined):
        supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)
    supervisor.ensure_dispatch_allowed(VoiceDispatchKind.TYPED)
    first.set_result(None)
    second.set_result(None)
    survivor.set_result(None)
    await asyncio.sleep(0)
    supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)


@pytest.mark.asyncio
async def test_quarantine_precedes_resolution_but_typed_dispatch_remains_allowed() -> (
    None
):
    runtime = ConsoleRuntime(SimpleNamespace(persona_buddy_controller=None))
    supervisor = runtime.voice_dispatch_supervisor
    release = asyncio.Event()
    orphan = asyncio.create_task(release.wait(), name="secret response body")
    supervisor.retain_orphan(orphan)
    resolutions: list[str] = []

    async def resolve_provider_and_session() -> str:
        resolutions.append("resolved")
        return "sent"

    with pytest.raises(VoiceDispatchQuarantined) as captured:
        await supervisor.guarded_dispatch(
            VoiceDispatchKind.HANDS_FREE,
            resolve_provider_and_session,
        )
    assert resolutions == []
    assert captured.value.code == "voice_cleanup_stuck"
    assert (
        await supervisor.guarded_dispatch(
            VoiceDispatchKind.TYPED,
            resolve_provider_and_session,
        )
        == "sent"
    )
    assert resolutions == ["resolved"]

    release.set()
    await orphan
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_two_orphans_survive_reentry_provider_and_session_reconstruction() -> (
    None
):
    runtime = ConsoleRuntime(SimpleNamespace(persona_buddy_controller=None))
    supervisor = runtime.voice_dispatch_supervisor
    first_release = asyncio.Event()
    second_release = asyncio.Event()
    first = asyncio.create_task(first_release.wait(), name="private transcript one")
    second = asyncio.create_task(second_release.wait(), name="private response two")
    supervisor.retain_orphan(first)
    supervisor.retain_orphan(second)

    original = supervisor
    runtime.attach_view(object())
    assert runtime.detach_view() is True
    runtime.attach_view(object())
    runtime.set_provider_gateway(object())
    runtime.set_provider_gateway(object())
    runtime.set_chat_controller(SimpleNamespace())
    runtime.set_chat_controller(SimpleNamespace())

    assert runtime.voice_dispatch_supervisor is original
    assert supervisor.orphan_count == 2
    assert supervisor.is_quarantined is True
    assert "private" not in repr(supervisor.orphan_summaries())
    with pytest.raises(VoiceDispatchQuarantined):
        supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)

    first_release.set()
    await first
    await asyncio.sleep(0)
    assert supervisor.orphan_count == 1
    with pytest.raises(VoiceDispatchQuarantined):
        supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)

    second_release.set()
    await second
    await asyncio.sleep(0)
    assert supervisor.orphan_count == 0
    assert supervisor.is_quarantined is False
    supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)


@pytest.mark.asyncio
async def test_orphan_registry_is_opaque_content_free_and_bounded() -> None:
    runtime = ConsoleRuntime(SimpleNamespace(persona_buddy_controller=None))
    supervisor = runtime.voice_dispatch_supervisor
    releases = [asyncio.Event() for _ in range(3)]
    tasks = [
        asyncio.create_task(event.wait(), name=f"secret body {index}")
        for index, event in enumerate(releases)
    ]

    supervisor.retain_orphan(tasks[0])
    supervisor.retain_orphan(tasks[1])
    summaries = supervisor.orphan_summaries()

    assert len(summaries) == 2
    assert all(summary.__slots__ == ("orphan_id",) for summary in summaries)
    assert "secret" not in repr(summaries)
    with pytest.raises(VoiceDispatchQuarantined):
        supervisor.retain_orphan(tasks[2])

    for event in releases:
        event.set()
    await asyncio.gather(*tasks)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_supervisor_strongly_owns_sanitized_cleanup_until_completion() -> None:
    runtime = ConsoleRuntime(SimpleNamespace(persona_buddy_controller=None))
    supervisor = runtime.voice_dispatch_supervisor
    cleanup = asyncio.get_running_loop().create_future()
    cleanup_ref = weakref.ref(cleanup)

    supervisor.retain_orphan(cleanup)
    del cleanup
    gc.collect()

    owned = cleanup_ref()
    assert owned is not None
    assert supervisor.orphan_count == 1
    owned.set_result(None)
    del owned
    await asyncio.sleep(0)
    gc.collect()

    assert supervisor.orphan_count == 0
    assert cleanup_ref() is None


@pytest.mark.asyncio
async def test_worker_owned_orphans_register_release_and_cap_across_loops() -> None:
    from tldw_chatbook.Chat.console_voice_worker import ConsoleVoiceWorker

    worker = ConsoleVoiceWorker()
    supervisor = VoiceDispatchSupervisor()
    registration_threads: list[int] = []
    owner_thread: list[int] = []
    releases = (threading.Event(), threading.Event())

    def create_worker_futures() -> tuple[asyncio.Future[None], ...]:
        owner_thread.append(threading.get_ident())

        class RecordingFuture(asyncio.Future[None]):
            def add_done_callback(self, callback, *, context=None) -> None:
                registration_threads.append(threading.get_ident())
                super().add_done_callback(callback, context=context)

        futures = (RecordingFuture(), RecordingFuture())

        async def settle(index: int) -> None:
            while not releases[index].is_set():
                await asyncio.sleep(0.001)
            futures[index].set_result(None)

        for index in range(2):
            asyncio.create_task(settle(index))
        return futures

    futures = await worker.run(create_worker_futures)
    try:
        summaries = await asyncio.gather(
            *(asyncio.to_thread(supervisor.retain_orphan, item) for item in futures)
        )
        await _wait_until(lambda: len(registration_threads) == 2)

        assert registration_threads == [owner_thread[0], owner_thread[0]]
        assert len({summary.orphan_id for summary in summaries}) == 2
        assert supervisor.orphan_count == 2
        assert len(supervisor.orphan_summaries()) == 2
        with pytest.raises(VoiceDispatchQuarantined):
            supervisor.retain_orphan(asyncio.get_running_loop().create_future())

        releases[0].set()
        await _wait_until(lambda: supervisor.orphan_count == 1)
        assert supervisor.is_quarantined is True
        releases[1].set()
        await _wait_until(lambda: supervisor.orphan_count == 0)
        assert supervisor.is_quarantined is False
    finally:
        for release in releases:
            release.set()
        await worker.aclose()


async def _wait_until(predicate) -> None:
    for _ in range(1_000):
        if predicate():
            return
        await asyncio.sleep(0.001)
    pytest.fail("voice supervisor did not settle")


def test_closed_owner_loop_registration_failure_stays_quarantined() -> None:
    supervisor = VoiceDispatchSupervisor()
    owner_loop = asyncio.new_event_loop()
    cleanup = owner_loop.create_future()
    cleanup_ref = weakref.ref(cleanup)
    owner_loop.close()

    with pytest.raises(VoiceDispatchQuarantined) as captured:
        supervisor.retain_orphan(cleanup)

    del cleanup
    gc.collect()
    assert captured.value.code == "voice_cleanup_stuck"
    assert supervisor.orphan_count == 1
    assert supervisor.is_quarantined is True
    assert len(supervisor.orphan_summaries()) == 1
    assert cleanup_ref() is not None
    with pytest.raises(VoiceDispatchQuarantined):
        supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)


@pytest.mark.asyncio
async def test_cleanup_waiter_survives_atomic_transfer_and_cancellation_across_loops():
    supervisor = VoiceDispatchSupervisor()
    worker = ConsoleVoiceWorker()

    async def originals():
        pending = asyncio.get_running_loop().create_future()
        survivor = asyncio.get_running_loop().create_future()
        supervisor.retain_pending_cleanup(pending)
        return pending, survivor

    pending, survivor = await worker.run(originals)
    observer = asyncio.create_task(supervisor.wait_for_cleanup())
    await asyncio.sleep(0)
    observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observer
    assert not pending.done() and not survivor.done()
    retained = asyncio.create_task(supervisor.wait_for_cleanup())
    try:

        def transfer():
            supervisor.retain_orphan(survivor, pending_cleanup=pending)
            pending.set_result(None)

        await worker.run(transfer)
        await asyncio.sleep(0)
        assert not retained.done() and supervisor.is_quarantined
        await worker.run(lambda: survivor.set_result(None))
        await asyncio.wait_for(retained, 1)
        assert not supervisor.is_quarantined
    finally:
        if not survivor.done():
            await worker.run(lambda: survivor.set_result(None))
        await worker.aclose()
