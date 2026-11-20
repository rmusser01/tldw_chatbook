"""TASK-33620.15.55: how the private SQLite helper is launched.

Every ordinary ChaChaNotes connection opened on a worker thread prepares its
database in a fresh helper process (ADR-125), and a Console send opens 25-55 of
them between Enter and the provider call. On macOS, ``subprocess`` launches a
``close_fds=True`` child with ``fork()``, and CPython holds the GIL for the whole
fork of a large parent (it releases it only around ``vfork()``, which is
Linux-only). Every other Python thread, the UI loop included, then waits for the
fork. In TASK-33620.15.55's live rounds at load1 109-122, every UI-loop
transaction entry of 10 ms or more overlapped a helper launch on another thread
(8 of 8, against 11% of random 10 ms intervals).

These tests pin:

* the launch mechanism: on macOS a helper starts through ``posix_spawn``, never
  through ``fork_exec`` (fails on origin/dev, which forks). Linux keeps
  ``fork_exec``, whose vfork launch already releases the GIL;
* the descriptor posture: the helper closes every descriptor above stderr that it
  inherited before it reads its request, so it works with exactly the three pipes
  a ``close_fds=True`` fork gave it (fails on origin/dev, whose helper keeps an
  inherited descriptor open for as long as it lives);
* the parent's own descriptors stay out of the helper, because Python creates them
  non-inheritable (PEP 446): a preparation still reaps its child and leaves the
  parent's descriptor count unchanged.
"""

from __future__ import annotations

import importlib
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="Windows prepares in-process without a helper"
)


def _process():
    return importlib.import_module("tldw_chatbook.DB.private_sqlite_process")


def _deadline(seconds: float = 5.0):
    return _process().OperationDeadline(time.monotonic() + seconds)


def _request(path: Path):
    codec = importlib.import_module("tldw_chatbook.DB.private_sqlite_protocol")
    return codec.PrepareRequest(str(path), True, True, False)


def _fd_directory() -> str:
    return "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"


@pytest.mark.skipif(
    sys.platform != "darwin" or not getattr(subprocess, "_USE_POSIX_SPAWN", False),
    reason="only macOS forks a close_fds=True child with the GIL held",
)
def test_helper_launch_uses_posix_spawn_not_a_fork(tmp_path, monkeypatch):
    process = _process()
    calls: list[str] = []
    real_spawn = os.posix_spawn
    real_fork_exec = subprocess._fork_exec

    def spawn(*args, **kwargs):
        calls.append("posix_spawn")
        return real_spawn(*args, **kwargs)

    def fork_exec(*args, **kwargs):
        calls.append("fork_exec")
        return real_fork_exec(*args, **kwargs)

    monkeypatch.setattr(os, "posix_spawn", spawn)
    monkeypatch.setattr(subprocess, "_fork_exec", fork_exec)
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=_deadline()
    ) as owner:
        lease = process.HelperLease.start(
            _request(tmp_path / "db"),
            operation="prepare",
            reservation=owner,
            deadline=_deadline(),
        )
        assert lease.initial_result.artifacts[0] == "created_private"
        lease.close()
    assert lease.cleanup_state == "reaped"
    # A fork of this process runs with the GIL held for its whole duration.
    assert calls == ["posix_spawn"], f"the helper launch used {calls}"


def test_helper_closes_descriptors_it_inherits_before_its_request():
    """Launch the fixed entry the way the launcher does, plus one inheritable pipe end.

    The helper waits for its request frame on stdin. While it waits, it must
    already have closed the inherited write end: closing the parent's copy then
    gives the reader EOF although the helper is still alive.
    """
    entry = (
        Path(_process().__file__).with_name("private_sqlite_helper_entry.py").resolve()
    )
    read_end, write_end = os.pipe()
    os.set_inheritable(write_end, True)
    environment = os.environ.copy()
    environment["_TLDW_PRIVATE_SQLITE_PARENT_PID"] = str(os.getpid())
    child = None
    try:
        child = subprocess.Popen(
            [sys.executable, "-I", "-S", str(entry)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=False,
            bufsize=0,
            env=environment,
        )
        os.close(write_end)
        write_end = -1
        readable, _, _ = select.select([read_end], [], [], 10.0)
        alive = child.poll() is None
        assert readable and os.read(read_end, 1) == b"", (
            "the helper kept an inherited descriptor open while it waited"
        )
        assert alive, "the helper exited instead of waiting for its request"
    finally:
        os.close(read_end)
        if write_end >= 0:
            os.close(write_end)
        if child is not None:
            child.stdin.close()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
            child.stdout.close()


def test_preparation_reaps_its_helper_and_leaks_no_parent_descriptor(tmp_path):
    process = _process()
    before = sorted(os.listdir(_fd_directory()))
    children = []
    for index in range(6):
        with process.HelperAdmission().reserve(
            transient=1, retained=0, deadline=_deadline()
        ) as owner:
            lease = process.HelperLease.start(
                _request(tmp_path / f"db{index}"),
                operation="prepare",
                reservation=owner,
                deadline=_deadline(),
            )
            children.append(lease._child)
            lease.close()
        assert lease.cleanup_state == "reaped"
    assert all(child.poll() is not None for child in children)
    assert sorted(os.listdir(_fd_directory())) == before
