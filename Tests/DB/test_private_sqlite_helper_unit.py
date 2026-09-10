"""Focused helper-unit coverage supplementary to real-child lifecycle tests."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from tldw_chatbook.DB import private_sqlite_files as files
from tldw_chatbook.DB import private_sqlite_helper as helper
from tldw_chatbook.DB import private_sqlite_protocol as protocol
from tldw_chatbook.Utils.private_paths import PrivatePathError


def _request(path: Path, operation: str) -> dict[str, object]:
    payload: dict[str, object] = {"version": 1, "operation": operation}
    if operation in {"prepare", "pin_source"}:
        payload.update(
            path=str(path),
            writable=operation == "prepare",
            create_if_missing=operation == "prepare",
            preserve_source_mode=False,
        )
    return payload


def _install_memory_pipe(monkeypatch, *payloads: dict[str, object]) -> io.BytesIO:
    incoming = io.BytesIO(b"".join(protocol.encode_frame(row) for row in payloads))
    outgoing = io.BytesIO()

    class MemoryPipe:
        deadline = None

        def __init__(self, parent_pid: int) -> None:
            self.parent_pid = parent_pid

        def read(self, count: int) -> bytes:
            return incoming.read(count)

        def write(self, frame: bytes) -> None:
            outgoing.write(frame)

    monkeypatch.setattr(helper, "_PrivatePipe", MemoryPipe)
    return outgoing


def _responses(stream: io.BytesIO) -> list[dict[str, object]]:
    stream.seek(0)
    decoded = []
    while response := protocol.read_frame(stream):
        decoded.append(response)
    return decoded


def test_source_pin_rechecks_exact_inode_and_closes_owned_descriptors(tmp_path):
    """Replacing the named file must not remint the retained source authority."""
    selected = tmp_path / "source.db"
    selected.write_bytes(b"original")
    selected.chmod(0o600)
    request = protocol.PrepareRequest(str(selected), False, False, False)
    prepared = files.prepare_batch(request)
    pin = helper.SourcePin(request, prepared)
    original = pin.recheck()

    selected.rename(tmp_path / "original.db")
    selected.write_bytes(b"replacement")
    selected.chmod(0o600)
    try:
        with pytest.raises(PrivatePathError) as caught:
            pin.recheck()
        assert caught.value.result.reason == "private_sqlite_source_identity_changed"
        assert original.main_identity.same_inode(prepared.main_identity)
    finally:
        pin.close()

    assert pin.file_fd == -1
    assert pin.parent_fd == -1
    pin.close()


def test_private_pipe_anchors_one_deadline_and_completes_partial_writes(monkeypatch):
    """Resetting the deadline per read/write chunk would permit an unbounded frame."""
    set_blocking_calls = []
    monkeypatch.setattr(
        helper.os,
        "set_blocking",
        lambda fd, blocking: set_blocking_calls.append((fd, blocking)),
    )
    monkeypatch.setattr(helper.os, "getppid", lambda: 42)
    monkeypatch.setattr(
        helper.select,
        "select",
        lambda reads, writes, errors, timeout: (reads, writes, errors),
    )
    clock = [100.0]
    monkeypatch.setattr(
        helper.time, "monotonic", lambda: clock.pop(0) if clock else 101.0
    )
    reads = iter((b"a", b"b"))
    monkeypatch.setattr(helper.os, "read", lambda fd, count: next(reads))
    written = bytearray()

    def partial_write(fd: int, value: bytes) -> int:
        chunk = value[:2]
        written.extend(chunk)
        return len(chunk)

    monkeypatch.setattr(helper.os, "write", partial_write)
    pipe = helper._PrivatePipe(42)

    assert pipe.read(1) == b"a"
    anchored_deadline = pipe.deadline
    assert anchored_deadline == 100.0 + helper.ROUND_TRIP_SECONDS
    assert pipe.read(1) == b"b"
    assert pipe.deadline == anchored_deadline
    pipe.write(b"abcde")

    assert written == b"abcde"
    assert set_blocking_calls == [(0, False), (1, False)]


def test_private_pipe_wait_refuses_expired_deadline_and_parent_loss(monkeypatch):
    """Deadline expiry and parent replacement must both stop transport polling."""
    monkeypatch.setattr(helper.os, "set_blocking", lambda fd, blocking: None)
    parent = [42]
    monkeypatch.setattr(helper.os, "getppid", lambda: parent[0])
    monkeypatch.setattr(helper.time, "monotonic", lambda: 10.0)
    pipe = helper._PrivatePipe(42)
    pipe.deadline = 10.0

    with pytest.raises(TimeoutError):
        pipe._wait(0)

    parent[0] = 43
    with pytest.raises(helper._ParentGone):
        pipe._wait(0)


def test_run_dispatches_prepare_then_close_with_closed_response_shapes(
    tmp_path, monkeypatch
):
    """A wrong dispatcher branch must not satisfy prepare and close requests."""
    selected = tmp_path / "prepared.db"
    outgoing = _install_memory_pipe(
        monkeypatch,
        _request(selected, "prepare"),
        _request(selected, "close"),
    )
    parent_pid = os.getppid()

    assert helper.run(parent_pid) == 0

    responses = _responses(outgoing)
    assert [row["operation"] for row in responses] == ["prepare", "close"]
    assert [row["status"] for row in responses] == ["ok", "ok"]
    assert set(responses[0]) == {"version", "operation", "status", "result"}
    assert selected.is_file()
    assert selected.stat().st_mode & 0o777 == 0o600


def test_run_eof_releases_real_retained_source_pin(tmp_path, monkeypatch):
    """Losing the request stream must release the helper-owned raw descriptors."""
    selected = tmp_path / "source.db"
    selected.write_bytes(b"source")
    selected.chmod(0o600)
    outgoing = _install_memory_pipe(
        monkeypatch,
        _request(selected, "pin_source"),
    )
    real_source_pin = helper.SourcePin
    pins = []

    def capture_pin(request, result):
        pin = real_source_pin(request, result)
        pins.append(pin)
        return pin

    monkeypatch.setattr(helper, "SourcePin", capture_pin)

    assert helper.run(os.getppid()) == 0

    assert [row["status"] for row in _responses(outgoing)] == ["ok"]
    assert len(pins) == 1
    assert pins[0].file_fd == -1
    assert pins[0].parent_fd == -1
