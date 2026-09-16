"""Audio installation and artifact owners retain work while capture waits."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
    AudioCppArtifactDependencyError,
    AudioCppArtifactLeaseCoordinator,
)
from tldw_chatbook.UI.Navigation.audio_cpp_model_handoff import (
    AudioCppModelInstallOwner,
)


@pytest.mark.asyncio
async def test_install_pause_waits_runner_and_preserves_callback():
    owner = AudioCppModelInstallOwner()
    entered, release = asyncio.Event(), asyncio.Event()
    outcomes = []

    async def run(cancel):
        entered.set()
        await release.wait()
        assert not cancel.is_set()
        return "installed"

    operation = owner.start(run, lambda *args: outcomes.append(args))
    await entered.wait()
    owner.maintenance_close_admission()
    with pytest.raises(RuntimeError, match="maintenance"):
        owner.start(run, lambda *args: None)
    assert not await owner.maintenance_drain(time.monotonic())
    waiter = asyncio.create_task(owner.maintenance_drain(time.monotonic() + 2))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not operation.task.done()
    release.set()
    await owner.wait(operation)
    assert outcomes == [("installed", None, False)]
    assert await owner.maintenance_drain(time.monotonic() + 1)
    owner.maintenance_resume()
    await owner.wait(owner.start(run, lambda *args: None))


@pytest.mark.asyncio
async def test_install_live_lease_blocks_until_consumer_releases():
    owner = AudioCppModelInstallOwner()
    closed = []
    service = SimpleNamespace(
        acquire_installed_root=lambda ref: SimpleNamespace(
            close=lambda: closed.append(ref)
        )
    )
    hold = await owner.acquire_lease_hold(("exact-root",), lambda: service)
    owner.maintenance_close_admission()
    with pytest.raises(RuntimeError, match="maintenance"):
        await owner.acquire_lease_hold((), lambda: service)
    assert not await owner.maintenance_drain(time.monotonic())
    assert not closed
    owner.request_lease_release(hold)
    assert await owner.maintenance_drain(time.monotonic() + 2)
    assert closed == ["exact-root"]


@pytest.mark.asyncio
async def test_install_uncertain_cleanup_stays_blocked_until_successful_retry():
    owner = AudioCppModelInstallOwner()
    fail = True

    def close():
        if fail:
            raise OSError("cleanup failed")

    hold = await owner.acquire_lease_hold(
        ("exact-root",),
        lambda: SimpleNamespace(
            acquire_installed_root=lambda ref: SimpleNamespace(close=close)
        ),
    )
    owner.maintenance_close_admission()
    owner.request_lease_release(hold)
    await owner.wait_lease_hold(hold)
    assert not await owner.maintenance_drain(time.monotonic())
    fail = False
    owner.retry_cleanup()
    assert await owner.maintenance_drain(time.monotonic() + 2)


@pytest.mark.asyncio
async def test_artifact_pause_waits_lease_context_without_cancel():
    owner = AudioCppArtifactLeaseCoordinator(
        SimpleNamespace(),
        saved_settings_snapshot=lambda: (),
        catalog_entries=lambda: (),
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def publish():
        async with owner.lease_consumers(()):
            entered.set()
            await release.wait()

    active = asyncio.create_task(publish())
    await entered.wait()
    owner.maintenance_close_admission()
    with pytest.raises(AudioCppArtifactDependencyError, match="maintenance"):
        async with owner.lease_consumers(()):
            raise AssertionError("closed intake admitted work")
    with pytest.raises(AudioCppArtifactDependencyError, match="maintenance"):
        await owner.remove_if_unchanged(None, "ignored", None)
    assert not await owner.maintenance_drain(time.monotonic())
    waiter = asyncio.create_task(owner.maintenance_drain(time.monotonic() + 2))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not active.done()
    release.set()
    await active
    assert await owner.maintenance_drain(time.monotonic() + 1)
    owner.maintenance_resume()
    async with owner.lease_consumers(()):
        pass


@pytest.mark.asyncio
async def test_maintenance_resume_cannot_reopen_terminal_owners():
    install = AudioCppModelInstallOwner()
    artifacts = AudioCppArtifactLeaseCoordinator(
        SimpleNamespace(), saved_settings_snapshot=lambda: ()
    )
    install.maintenance_close_admission()
    artifacts.maintenance_close_admission()
    await install.shutdown()
    await artifacts.shutdown()
    install.maintenance_resume()
    artifacts.maintenance_resume()
    with pytest.raises(RuntimeError, match="shut down"):
        install.start(None, None)
    with pytest.raises(AudioCppArtifactDependencyError, match="closed"):
        async with artifacts.lease_consumers(()):
            raise AssertionError("shutdown owner reopened")
