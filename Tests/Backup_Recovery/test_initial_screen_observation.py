"""The Windows initial barrier ceiling preserves Textual's real wait contract."""

import asyncio
from types import SimpleNamespace

import pytest
from textual.pilot import Pilot

from Tests.Backup_Recovery import initial_screen_observation as observation


@pytest.fixture
def original_wait(monkeypatch):
    calls = []

    async def wait(pilot, timeout=30):
        calls.append((pilot.app, timeout))
        return calls

    monkeypatch.setattr(Pilot, "_wait_for_screen", wait)
    monkeypatch.setattr(observation, "sys", SimpleNamespace(platform="win32"))
    return wait, calls


@pytest.mark.asyncio
async def test_only_intended_first_default_call_changes_ceiling(original_wait):
    original, calls = original_wait
    app, other = object(), object()
    async with observation.initial_screen_observation(app):
        assert await Pilot(other)._wait_for_screen() is calls
        assert await Pilot(app)._wait_for_screen(7) is calls
        assert await Pilot(app)._wait_for_screen(timeout=8) is calls
        assert Pilot._wait_for_screen is not original
        assert await Pilot(app)._wait_for_screen() is calls
        assert Pilot._wait_for_screen is original
        assert await Pilot(app)._wait_for_screen() is calls
    assert calls == [(other, 30), (app, 7), (app, 8), (app, 135), (app, 30)]
    assert Pilot._wait_for_screen is original


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_original_error_and_cancellation_restore_immediately(monkeypatch, cancel):
    error = asyncio.CancelledError() if cancel else RuntimeError("original barrier")

    async def wait(pilot, timeout=30):
        assert timeout == 135
        raise error

    monkeypatch.setattr(Pilot, "_wait_for_screen", wait)
    monkeypatch.setattr(observation, "sys", SimpleNamespace(platform="win32"))
    app = object()
    async with observation.initial_screen_observation(app):
        with pytest.raises(type(error)) as caught:
            await Pilot(app)._wait_for_screen()
        assert caught.value is error
        assert Pilot._wait_for_screen is wait
    assert Pilot._wait_for_screen is wait


@pytest.mark.asyncio
async def test_task_cancellation_preserves_original_wait_and_restores(monkeypatch):
    entered = asyncio.Event()
    finished = []

    async def wait(pilot, timeout=30):
        assert timeout == 135
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.append(True)

    monkeypatch.setattr(Pilot, "_wait_for_screen", wait)
    monkeypatch.setattr(observation, "sys", SimpleNamespace(platform="win32"))
    app = object()
    async with observation.initial_screen_observation(app):
        task = asyncio.create_task(Pilot(app)._wait_for_screen())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished == [True]
        assert Pilot._wait_for_screen is wait


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_pre_entry_failure_or_cancellation_restores(original_wait, cancel):
    original, calls = original_wait
    error = asyncio.CancelledError() if cancel else RuntimeError("before initial call")
    with pytest.raises(type(error)) as caught:
        async with observation.initial_screen_observation(object()):
            assert Pilot._wait_for_screen is not original
            raise error
    assert caught.value is error and not calls
    assert Pilot._wait_for_screen is original


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["darwin", "linux"])
async def test_non_windows_never_changes_pilot(original_wait, monkeypatch, platform):
    original, calls = original_wait
    monkeypatch.setattr(observation, "sys", SimpleNamespace(platform=platform))
    app = object()
    async with observation.initial_screen_observation(app):
        assert Pilot._wait_for_screen is original
        assert await Pilot(app)._wait_for_screen() is calls
    assert calls == [(app, 30)]
    assert Pilot._wait_for_screen is original
