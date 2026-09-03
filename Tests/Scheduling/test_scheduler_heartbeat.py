"""TASK-26025: durable scheduler liveness heartbeat.

A dead loop was indistinguishable from an idle one. The heartbeat persists
the last tick/success/error so a fresh reader (a status surface, or the next
process) can tell live from stalled from never-started -- judged against the
poll interval so a long interval is not mistaken for a stall.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from tldw_chatbook.Scheduling.scheduler_heartbeat import (
    SchedulerHeartbeat,
    classify_scheduler_liveness,
    read_heartbeat,
    write_heartbeat,
)


def _t(offset_seconds=0):
    return datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=offset_seconds
    )


def test_never_started_when_no_heartbeat_exists():
    assert classify_scheduler_liveness(None, now=_t(0), poll_interval=30.0) == (
        "never_started"
    )


def test_live_within_the_staleness_window():
    hb = SchedulerHeartbeat(last_tick_at=_t(0), last_success_at=_t(0))
    # a tick 20s ago with a 30s poll is live (window = poll * factor)
    assert classify_scheduler_liveness(hb, now=_t(20), poll_interval=30.0) == "live"


def test_stale_beyond_the_window_scaled_by_poll_interval():
    hb = SchedulerHeartbeat(last_tick_at=_t(0), last_success_at=_t(0))
    # ~3 hours later, obviously stale
    assert classify_scheduler_liveness(
        hb, now=_t(3 * 3600), poll_interval=30.0
    ) == "stale"
    # AC#4: a LONG poll interval must not read as a stall -- the same age
    # is live when the interval is itself hours
    assert classify_scheduler_liveness(
        hb, now=_t(3 * 3600), poll_interval=4 * 3600
    ) == "live"


def test_round_trip_persists_all_fields(tmp_path: Path):
    path = tmp_path / "hb.json"
    hb = SchedulerHeartbeat(
        last_tick_at=_t(0),
        last_success_at=_t(0),
        last_error="handler blew up",
        poll_interval=30.0,
        tick_count=42,
    )
    write_heartbeat(path, hb)
    loaded = read_heartbeat(path)
    assert loaded is not None
    assert loaded.last_tick_at == _t(0)
    assert loaded.last_error == "handler blew up"
    assert loaded.tick_count == 42


def test_read_missing_or_corrupt_is_none(tmp_path: Path):
    assert read_heartbeat(tmp_path / "absent.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_heartbeat(bad) is None


def test_write_never_raises_on_bad_path(tmp_path: Path):
    # a directory where a file is expected -> write swallows the error
    d = tmp_path / "dir"
    d.mkdir()
    write_heartbeat(d, SchedulerHeartbeat(last_tick_at=_t(0)))  # no raise


import asyncio

import pytest


def _make_loop(tmp_path, dispatch_raises=False):
    from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop

    class _Queue:
        def pop_due(self, now):
            if dispatch_raises:
                raise RuntimeError("handler exploded")
            return []

    loop = SchedulerLoop.__new__(SchedulerLoop)
    # minimal hand-wired instance (avoid the full ctor's DB deps)
    loop.clock = lambda: _t(0)
    loop.poll_interval = 30.0
    loop.queue = _Queue()
    loop._heartbeat_path = tmp_path / "hb.json"
    loop._last_success_at = None
    loop._last_error = None
    loop._last_tick_dispatch_seconds = 0.0
    loop._tick_count = 7
    return loop


@pytest.mark.asyncio
async def test_a_healthy_tick_writes_a_live_heartbeat(tmp_path):
    loop = _make_loop(tmp_path)
    await loop.tick()
    hb = read_heartbeat(tmp_path / "hb.json")
    assert hb is not None
    assert hb.last_tick_at == _t(0)
    assert hb.last_success_at == _t(0)
    assert hb.last_error is None
    assert hb.tick_count == 7
    assert classify_scheduler_liveness(hb, now=_t(10), poll_interval=30.0) == "live"


@pytest.mark.asyncio
async def test_an_erroring_tick_records_the_last_error(tmp_path):
    loop = _make_loop(tmp_path, dispatch_raises=True)
    with pytest.raises(RuntimeError):
        await loop.tick()
    hb = read_heartbeat(tmp_path / "hb.json")
    assert hb is not None
    assert hb.last_tick_at == _t(0), "even a failed tick records liveness (AC#3)"
    assert hb.last_error is not None
    assert "handler exploded" in hb.last_error
    assert hb.last_success_at is None


def test_liveness_summary_distinguishes_all_three_states():
    from tldw_chatbook.Scheduling.scheduler_heartbeat import scheduler_liveness_line

    never = scheduler_liveness_line(None, now=_t(0), poll_interval=30.0)
    assert "not started" in never.lower()

    live_hb = SchedulerHeartbeat(last_tick_at=_t(0), last_success_at=_t(0))
    live = scheduler_liveness_line(live_hb, now=_t(12), poll_interval=30.0)
    assert "live" in live.lower()

    stale = scheduler_liveness_line(
        SchedulerHeartbeat(last_tick_at=_t(0), last_error="boom"),
        now=_t(3 * 3600),
        poll_interval=30.0,
    )
    assert "stall" in stale.lower()
    assert "boom" in stale, "the last error is surfaced, not only logged (AC#3)"
    # empty-queue live state must read differently from a stall
    assert stale != live
