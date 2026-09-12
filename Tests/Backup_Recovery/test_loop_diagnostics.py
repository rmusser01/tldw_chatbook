"""Bounded main-loop profiling observes calls without retaining runtime values."""

import asyncio
import json
import sys
import threading

import pytest

from Tests.Backup_Recovery.loop_diagnostics import observe_loop_profile


@pytest.mark.asyncio
async def test_profile_window_runs_on_loop_thread_and_exports_bounded_data(tmp_path):
    path = tmp_path / "profile.json"
    private_value = "synthetic-runtime-value-must-not-appear"
    thread = threading.get_ident()

    def measured_work(value):
        assert threading.get_ident() == thread
        sum(range(10000))
        return value

    stop = observe_loop_profile(path, delay=0, duration=0.05)
    try:
        await asyncio.sleep(0.001)
        assert measured_work(private_value) == private_value
        async with asyncio.timeout(5):
            while not path.exists():
                await asyncio.sleep(0.01)
        assert sys.getprofile() is None
    finally:
        stop()
    payload = path.read_text()
    assert private_value not in payload and str(tmp_path) not in payload
    data = json.loads(payload)
    assert 0 < len(data["calls"]) <= 32
    assert data["elapsed_s"] > 0 and data["thread_cpu_s"] >= 0
    assert any(row["function"] == "measured_work" for row in data["calls"])
    assert all(
        set(row)
        == {
            "file",
            "function",
            "line",
            "calls",
            "recursive_calls",
            "total_s",
            "inline_s",
        }
        for row in data["calls"]
    )
    assert all(
        "/" not in row["file"] and "\\" not in row["file"] for row in data["calls"]
    )


@pytest.mark.asyncio
async def test_stop_cancels_pending_window_without_creating_profile(tmp_path):
    path = tmp_path / "cancelled.json"
    stop = observe_loop_profile(path, delay=0.02, duration=0.02)
    stop()
    stop()
    await asyncio.sleep(0.06)
    assert not path.exists() and sys.getprofile() is None


@pytest.mark.asyncio
async def test_stop_finalizes_active_profile_once_and_preserves_exception(tmp_path):
    path = tmp_path / "stopped.json"
    stop = observe_loop_profile(path, delay=0, duration=60)
    failure = ValueError("synthetic-exception-value")
    await asyncio.sleep(0.001)
    try:
        with pytest.raises(ValueError) as caught:
            raise failure
        assert caught.value is failure
    finally:
        stop()
    original = path.read_bytes()
    stop()
    await asyncio.sleep(0)
    assert path.read_bytes() == original
    assert sys.getprofile() is None
    assert b"synthetic-exception-value" not in original
