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


def test_a_non_utf8_file_fails_SAFE_to_stopped(tmp_path: Path):
    """TASK-33620.1.12: undecodable bytes read as unreadable, never raise.

    ``read_text`` raises ``UnicodeDecodeError`` (a ``ValueError``, not an
    ``OSError``), which used to escape the read and the send gate with it.
    """
    path = tmp_path / "estop.json"
    path.write_bytes(b'{"active": \xff\xfe}')
    state = read_emergency_stop(path)
    assert state.active is True
    assert state.reason == "stop state unreadable"
    assert is_emergency_stopped(path) is True


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
    # TASK-33620.1 AC#4: "clear" alone passed while the copy said only "Clear
    # it to resume." and never named the command that does it.
    assert "/emergency-stop clear" in msg
    assert "Clear it to resume." not in msg
    assert "scheduled dispatches are held" in msg


import pytest


def test_an_undecodable_file_logs_its_error_type_not_its_bytes(tmp_path: Path):
    """The stop file holds the reason the user typed; a log line never does.

    ``UnicodeDecodeError``'s repr carries the undecodable bytes themselves.
    """
    from loguru import logger

    path = tmp_path / "estop.json"
    path.write_bytes(b'{"active": true, "reason": "PRIVATE-REASON \xff"}')
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="WARNING")
    try:
        assert read_emergency_stop(path).active is True
    finally:
        logger.remove(sink)
    assert len(lines) == 1 and "UnicodeDecodeError" in lines[0]
    assert "PRIVATE-REASON" not in lines[0] and "\\xff" not in lines[0]


def test_an_unavailable_stop_does_not_offer_clear():
    """Clearing writes to the same unreachable folder, so it is not offered."""
    from tldw_chatbook import emergency_stop

    unavailable = EmergencyStopState(
        active=True, reason=emergency_stop.STATE_UNAVAILABLE_REASON
    )
    copy = emergency_stop.held_copy(unavailable)
    assert copy == emergency_stop.STATE_UNAVAILABLE_COPY
    assert "/emergency-stop clear" not in copy
    assert "couldn't be read" in copy
    # Every other hold, an unreadable or corrupt file included, keeps the
    # visible copy: clearing rewrites that file.
    for state in (
        EmergencyStopState(active=True, reason="runaway costs"),
        EmergencyStopState(active=True, reason="stop state unreadable"),
    ):
        assert emergency_stop.held_copy(state) == state.visible_copy()


def _counting_resolver(monkeypatch, path: Path) -> list[Path]:
    from tldw_chatbook import emergency_stop

    resolved: list[Path] = []

    def resolve() -> Path:
        resolved.append(path)
        return path

    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", resolve)
    return resolved


def test_the_stop_path_is_resolved_once_and_reused(tmp_path, monkeypatch):
    """Every tool call, hook action and research step reads the stop; the
    path's resolution (a guarded config read, ~1 ms) runs once."""
    from tldw_chatbook import emergency_stop

    path = tmp_path / "estop.json"
    resolved = _counting_resolver(monkeypatch, path)

    assert emergency_stop.current_emergency_stop().active is False
    set_emergency_stop(path, reason="now")
    assert emergency_stop.current_emergency_stop().reason == "now"
    assert emergency_stop.current_emergency_stop().reason == "now"

    assert len(resolved) == 1


@pytest.mark.parametrize("change", ["config", "home"])
def test_a_moved_data_folder_is_resolved_again(tmp_path, monkeypatch, change):
    """The stop is set at the NEW location after a config change (or a HOME
    change) moves the data folder: a reader that kept the old path would miss
    it, and the stop would not hold."""
    from tldw_chatbook import config, emergency_stop

    old, new = tmp_path / "old.json", tmp_path / "new.json"
    current = {"path": old}
    resolved: list[Path] = []

    def resolve() -> Path:
        resolved.append(current["path"])
        return current["path"]

    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", resolve)
    assert emergency_stop.current_emergency_stop().active is False

    current["path"] = new
    if change == "config":
        monkeypatch.setattr(config, "_CONFIG_GENERATION", config._CONFIG_GENERATION + 1)
    else:
        monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))
    set_emergency_stop(new, reason="set after the move")

    state = emergency_stop.current_emergency_stop()
    assert state.reason == "set after the move", "read the stale path"
    assert resolved == [old, new]


@pytest.mark.emergency_stop_unavailable
def test_a_failed_resolution_holds_and_is_not_cached(tmp_path, monkeypatch):
    from tldw_chatbook import emergency_stop

    path = tmp_path / "estop.json"
    outcomes: list[object] = [RuntimeError("data folder busy"), path]

    def resolve() -> Path:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", resolve)

    held = emergency_stop.current_emergency_stop()
    assert held.active is True
    assert held.reason == emergency_stop.STATE_UNAVAILABLE_REASON
    # The next read resolves again rather than reusing a failure.
    assert emergency_stop.current_emergency_stop().active is False
    assert outcomes == []


@pytest.mark.asyncio
async def test_scheduler_holds_dispatch_when_stopped(tmp_path):
    from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
    from datetime import datetime, timezone

    popped = {"count": 0}

    class _Queue:
        def pop_due(self, now):
            popped["count"] += 1
            return []

    import asyncio

    loop = SchedulerLoop.__new__(SchedulerLoop)
    loop.queue = _Queue()
    loop._emergency_stop_path = tmp_path / "estop.json"
    # TASK-33620.15.24 moved the stop read onto ``_offload``'s worker; a
    # ``__new__``-built loop lacked its task set, so every read failed SAFE to
    # stopped and the "not stopped -> pops" step went red on dev.
    loop._maintenance_db_tasks = set()
    loop._maintenance_changed = asyncio.Event()

    # not stopped -> pops
    await loop._dispatch_due(datetime.now(timezone.utc))
    assert popped["count"] == 1

    # stopped -> does NOT pop (holds without consuming, AC#1/#2)
    set_emergency_stop(loop._emergency_stop_path, reason="halt")
    await loop._dispatch_due(datetime.now(timezone.utc))
    assert popped["count"] == 1, "a stop must not drain the due queue"

    # cleared -> resumes without restart (AC#6)
    clear_emergency_stop(loop._emergency_stop_path)
    await loop._dispatch_due(datetime.now(timezone.utc))
    assert popped["count"] == 2


def test_durable_write_routes_through_the_fsyncing_atomic_helper(
    tmp_path: Path, monkeypatch
):
    """The switch's own docstring calls it durable; that requires an fsync,
    so the write must go through the shared atomic-write helper (which flushes
    + fsyncs before the rename) rather than a bare write+os.replace
    (task-32808.5)."""
    import tldw_chatbook.emergency_stop as es

    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        es,
        "atomic_write_text",
        lambda path, content, **kwargs: calls.append((Path(path), content)),
    )

    target = tmp_path / "estop.json"
    es.set_emergency_stop(target, reason="drain")

    assert len(calls) == 1
    written_path, written_content = calls[0]
    assert written_path == target
    assert '"active": true' in written_content
    assert "drain" in written_content
