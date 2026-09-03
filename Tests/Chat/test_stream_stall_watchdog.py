"""TASK-26003: content-stall watchdog for streamed provider responses."""

from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.Chat.stream_stall_watchdog import (
    StallTracker,
    StreamStallError,
    watch_content_stalls,
)


def _run(coro):
    return asyncio.run(coro)


async def _collect(source, timeout, **kw):
    out = []
    async for item in watch_content_stalls(source, timeout, **kw):
        out.append(item)
    return out


# --- AC#1/#5: productive stream passes through; stalled stream trips ---

def test_productive_stream_passes_through_unchanged():
    async def source():
        for i in range(4):
            await asyncio.sleep(0.01)
            yield i

    assert _run(_collect(source(), 0.5)) == [0, 1, 2, 3]


def test_slow_but_productive_stream_does_not_trip(monkeypatch):
    """AC#5: items arriving within the window each reset the clock."""
    async def source():
        for i in range(5):
            await asyncio.sleep(0.05)  # each < timeout
            yield i

    assert _run(_collect(source(), 0.2)) == [0, 1, 2, 3, 4]


def test_stall_after_some_content_raises(monkeypatch):
    """AC#1: no new content for the window -> StreamStallError, even though a
    real stream would still be receiving keep-alive bytes (not modeled here
    because keep-alives never reach the consumer -- that is the point)."""
    async def source():
        yield "a"
        yield "b"
        await asyncio.sleep(10)  # goes silent
        yield "never"

    with pytest.raises(StreamStallError) as exc:
        _run(_collect(source(), 0.15, provider="acme"))
    assert exc.value.provider == "acme"
    assert exc.value.timeout_seconds == 0.15


def test_immediate_stall_before_any_content_raises():
    async def source():
        await asyncio.sleep(10)
        yield "never"

    with pytest.raises(StreamStallError):
        _run(_collect(source(), 0.1))


# --- AC#1: disabling the watchdog ---

def test_non_positive_timeout_disables_watchdog():
    async def source():
        for i in range(3):
            yield i

    assert _run(_collect(source(), 0)) == [0, 1, 2]
    assert _run(_collect(source(), None)) == [0, 1, 2]


# --- AC#3: distinct from cancel and from other errors ---

def test_stall_error_is_not_cancelled_error():
    assert not issubclass(StreamStallError, asyncio.CancelledError)
    assert isinstance(StreamStallError(1.0), RuntimeError)


def test_user_cancel_propagates_as_cancelled_not_stall():
    """AC#3: a user cancel is never reported as a stall."""
    started = asyncio.Event()

    async def source():
        started.set()
        await asyncio.sleep(10)
        yield "never"

    async def drive():
        async def consume():
            async for _ in watch_content_stalls(source(), 5.0):
                pass
        task = asyncio.create_task(consume())
        await started.wait()
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _run(drive())


# --- cleanup: the source is closed on stall so the stream/worker is cancelled ---

def test_source_is_closed_on_stall():
    closed = {"n": 0}

    class Source:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(10)
            raise StopAsyncIteration

        async def aclose(self):
            closed["n"] += 1

    with pytest.raises(StreamStallError):
        _run(_collect(Source(), 0.1))
    assert closed["n"] >= 1


# --- AC#4: repeated-stall tracking ---

def test_stall_tracker_warns_at_threshold():
    t = StallTracker(warn_threshold=2)
    assert t.record_stall("acme") is False  # first stall: no warning yet
    assert t.record_stall("acme") is True   # second: warn
    assert t.record_stall("acme") is True   # stays warning
    assert t.count("acme") == 3


def test_stall_tracker_is_per_provider_and_resets():
    t = StallTracker(warn_threshold=2)
    t.record_stall("a")
    assert t.record_stall("b") is False  # different provider, own count
    assert t.count("a") == 1 and t.count("b") == 1
    t.reset("a")
    assert t.count("a") == 0
    assert t.count("b") == 1


def test_stall_tracker_threshold_floor():
    t = StallTracker(warn_threshold=0)  # coerced to >=1
    assert t.record_stall("x") is True


# --- AC#4: session-scoped registry ---

def test_session_stall_registry_warns_and_resets():
    import tldw_chatbook.Chat.stream_stall_watchdog as w
    w._SESSION_TRACKERS.clear()
    assert w.record_session_stall("sess1", "acme", warn_threshold=2) is False
    assert w.record_session_stall("sess1", "acme", warn_threshold=2) is True
    # a different session is independent
    assert w.record_session_stall("sess2", "acme", warn_threshold=2) is False
    # a productive turn on the provider clears the session entry
    w.reset_session_stalls("sess1", "acme")
    assert w.record_session_stall("sess1", "acme", warn_threshold=2) is False
    w._SESSION_TRACKERS.clear()


def test_reset_session_stalls_none_drops_whole_session():
    import tldw_chatbook.Chat.stream_stall_watchdog as w
    w._SESSION_TRACKERS.clear()
    w.record_session_stall("s", "a")
    w.record_session_stall("s", "b")
    w.reset_session_stalls("s")  # productive turn, drop all
    assert "s" not in w._SESSION_TRACKERS
    w._SESSION_TRACKERS.clear()


def test_session_registry_is_capped():
    """M3: a long-lived process cannot grow the registry without bound."""
    import tldw_chatbook.Chat.stream_stall_watchdog as w
    w._SESSION_TRACKERS.clear()
    monkeypatch_cap = 8
    orig = w._MAX_TRACKED_SESSIONS
    w._MAX_TRACKED_SESSIONS = monkeypatch_cap
    try:
        for i in range(monkeypatch_cap + 5):
            w.record_session_stall(f"sess-{i}", "acme")
        assert len(w._SESSION_TRACKERS) <= monkeypatch_cap
        # the newest session survived; the oldest was evicted
        assert f"sess-{monkeypatch_cap + 4}" in w._SESSION_TRACKERS
        assert "sess-0" not in w._SESSION_TRACKERS
    finally:
        w._MAX_TRACKED_SESSIONS = orig
        w._SESSION_TRACKERS.clear()


# --- Qodo round: env override, non-finite guard, concurrency ---

def test_env_override_takes_precedence(monkeypatch):
    """Qodo #1: env -> config -> default precedence."""
    from tldw_chatbook.Chat import console_agent_bridge as bridge_mod
    monkeypatch.setenv("TLDW_STREAM_STALL_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setattr(
        bridge_mod, "get_cli_setting", lambda *a, **k: 999, raising=False
    )
    assert bridge_mod._stall_timeout_seconds() == 12.5


def test_non_finite_config_falls_back_to_default(monkeypatch):
    """Qodo #2: inf/nan never reach asyncio.wait_for."""
    from tldw_chatbook.Chat import console_agent_bridge as bridge_mod
    from tldw_chatbook.Chat.stream_stall_watchdog import DEFAULT_STALL_TIMEOUT_SECONDS
    monkeypatch.delenv("TLDW_STREAM_STALL_TIMEOUT_SECONDS", raising=False)
    for bad in ("inf", "nan", "-inf", "notanumber"):
        monkeypatch.setattr(bridge_mod, "get_cli_setting", lambda *a, **k: bad)
        got = bridge_mod._stall_timeout_seconds()
        if bad == "notanumber":
            assert got == DEFAULT_STALL_TIMEOUT_SECONDS
        else:
            # inf/nan -> default (never a non-finite timeout)
            import math
            assert math.isfinite(got)


def test_concurrent_record_session_stall_is_thread_safe():
    """Qodo #5: concurrent fleet + primary completions must not lose counts."""
    import threading
    import tldw_chatbook.Chat.stream_stall_watchdog as w
    w._SESSION_TRACKERS.clear()

    def hammer():
        for _ in range(200):
            w.record_session_stall("sess", "acme", warn_threshold=10_000)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # no lost increments under the lock: exactly 8 * 200
    assert w._SESSION_TRACKERS["sess"].count("acme") == 1600
    w._SESSION_TRACKERS.clear()
