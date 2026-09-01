"""TASK-26004: the global emergency-stop sentinel.

One durable switch several readers consult (agent sends, scheduled
dispatch). It stops NEW work only, survives restart, and -- deliberately --
fails SAFE: if the read itself errors, the system treats it as STOPPED
rather than proceeding.
"""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.emergency_stop import (
    EmergencyStopState,
    clear_emergency_stop,
    emergency_stop_state,
    is_emergency_stopped,
    read_emergency_stop,
    set_emergency_stop,
)


def test_absent_file_means_not_stopped(tmp_path: Path):
    path = tmp_path / "estop.json"
    assert is_emergency_stopped(path) is False
    assert read_emergency_stop(path).active is False


def test_set_and_read_round_trip_survives_a_reread(tmp_path: Path):
    """AC#3: durable across a fresh read (stands in for a restart)."""
    path = tmp_path / "estop.json"
    set_emergency_stop(path, reason="runaway costs")
    state = read_emergency_stop(path)
    assert state.active is True
    assert state.reason == "runaway costs"
    assert is_emergency_stopped(path) is True


def test_clear_resumes_without_restart(tmp_path: Path):
    """AC#6."""
    path = tmp_path / "estop.json"
    set_emergency_stop(path, reason="x")
    clear_emergency_stop(path)
    assert is_emergency_stopped(path) is False


def test_corrupt_file_fails_SAFE_to_stopped(tmp_path: Path):
    """AC#4: a read error is treated as STOPPED, never as proceed."""
    path = tmp_path / "estop.json"
    path.write_text("{ this is not json")
    assert is_emergency_stopped(path) is True
    assert emergency_stop_state(path).active is True


def test_unreadable_path_fails_SAFE_to_stopped(tmp_path: Path):
    """A path that raises on read (a directory) is also treated as stopped."""
    d = tmp_path / "dir"
    d.mkdir()
    assert is_emergency_stopped(d) is True


def test_state_carries_a_clear_hint(tmp_path: Path):
    path = tmp_path / "estop.json"
    set_emergency_stop(path, reason="r")
    msg = emergency_stop_state(path).visible_copy()
    assert "stop" in msg.lower()
    assert "clear" in msg.lower()
