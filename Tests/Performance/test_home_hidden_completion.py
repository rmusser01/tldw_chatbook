"""Late Home completion must not run dashboard reads behind another screen."""

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.home_screen import HomeContentSnapshot, HomeScreen


class _CompletionReceiver:
    """Exercise original completion bodies with controlled worker boundaries.

    Snapshot providers and the UI sink are the only substitutes: this test owns
    no database and measures whether production dispatch enters that sink.
    """

    def __init__(self):
        self.is_mounted = True
        self.app = SimpleNamespace(screen=object())
        self.adapter = SimpleNamespace(
            refresh_active_work_cache_async=self.refresh_cache,
        )
        self.app_instance = SimpleNamespace(home_active_work_adapter=self.adapter)
        self._home_content_snapshot = HomeContentSnapshot()
        self.next_snapshot = HomeContentSnapshot(conversation_count=7)
        self._home_pending_live_active_work_sync = False
        self._home_pending_refresh_sync = False
        self.cache_ready = True
        self.cache_changed = True
        self.cache_error = False
        self.cache_entered = asyncio.Event()
        self.release_cache = asyncio.Event()
        self.release_cache.set()
        self.paints = []
        self.dispatches = []

    def __getattr__(self, name):
        # Newly introduced completion helpers still execute their real body.
        descriptor = getattr(HomeScreen, name)
        return descriptor.__get__(self, type(self))

    async def _build_home_content_snapshot(self):
        return self.next_snapshot

    async def refresh_cache(self):
        self.cache_entered.set()
        await self.release_cache.wait()
        if self.cache_error:
            raise RuntimeError("cache read failed")
        self.cache_ready = True
        return self.cache_changed

    def _sync_home_triage(self):
        assert self.cache_ready, "dashboard entered a cold synchronous cache read"
        self.paints.append(self._home_content_snapshot)

    def _refresh_home_chatbook_artifact_snapshot(self):
        self.dispatches.append("artifacts")

    def _refresh_home_content_snapshot(self):
        self.dispatches.append("content")

    def _refresh_home_active_work_cache(self):
        self.dispatches.append("cache")

    def _start_home_auto_refresh(self):
        self.dispatches.append("timer")


async def _complete(receiver, kind):
    if kind == "artifacts":
        HomeScreen._refresh_after_chatbook_artifact_snapshot(receiver)
    elif kind == "content":
        await HomeScreen._refresh_home_content_snapshot.__wrapped__(receiver)
    else:
        await HomeScreen._refresh_home_active_work_cache.__wrapped__(receiver)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["artifacts", "content", "cache"])
async def test_hidden_completion_replays_after_return_and_cache_warm(kind):
    """A late callback cannot read hidden Home, or lose unchanged return data."""
    receiver = _CompletionReceiver()
    await _complete(receiver, kind)
    assert receiver.paints == [], "hidden completion rebuilt the dashboard"

    # Return with the completed snapshot unchanged, but an expired cache.
    receiver.app.screen = receiver
    receiver.cache_ready = False
    receiver.cache_changed = False
    receiver.next_snapshot = receiver._home_content_snapshot
    receiver.cache_entered.clear()
    receiver.release_cache.clear()
    HomeScreen.on_screen_resume(receiver)
    assert receiver.dispatches == ["artifacts", "content", "cache", "timer"]
    assert receiver.paints == [], "resume entered synchronous dashboard reads"

    warmer = asyncio.create_task(
        HomeScreen._refresh_home_active_work_cache.__wrapped__(receiver)
    )
    try:
        await asyncio.wait_for(receiver.cache_entered.wait(), timeout=2)
        await _complete(receiver, "content")
        await _complete(receiver, "artifacts")
        assert receiver.paints == [], "completion outran the cache warmer"
    finally:
        receiver.release_cache.set()
        await asyncio.wait_for(warmer, timeout=2)
    assert receiver.paints == [receiver.next_snapshot]

    # No pending work survives the successful paint.
    await _complete(receiver, "cache")
    assert receiver.paints == [receiver.next_snapshot]


@pytest.mark.asyncio
async def test_failed_return_cache_refresh_keeps_hidden_completion_pending():
    """A failed warm must not replay by entering a cold synchronous read."""
    receiver = _CompletionReceiver()
    await _complete(receiver, "content")
    assert receiver.paints == []
    receiver.app.screen = receiver
    receiver.cache_ready = False
    receiver.cache_changed = False
    receiver.cache_error = True
    await _complete(receiver, "cache")
    assert receiver.paints == []
    receiver.cache_error = False
    await _complete(receiver, "cache")
    assert receiver.paints == [receiver.next_snapshot]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["artifacts", "content", "cache"])
async def test_visible_completion_still_presents_current_data(kind):
    receiver = _CompletionReceiver()
    receiver.app.screen = receiver
    await _complete(receiver, kind)
    assert receiver.paints == [receiver._home_content_snapshot]
