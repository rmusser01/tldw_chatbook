"""Stopping a local LLM server must stop the workers it forked.

TASK-32806.5. Every local server was spawned without a process group and
stopped with a bare `Popen.terminate()`. These servers fork workers, so the
worker survived -- still holding the port -- while the app cleared the
handle and reported the server stopped. Measured before the fix with a
parent that forks one grandchild:

    terminate_process_bounded(...) -> True
    grandchild still alive a second later -> True

Nothing in unmount stopped them at all, so quitting the app orphaned every
running server.

These tests use real processes on purpose. A mocked Popen cannot show a
grandchild surviving, which is the entire defect.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import time

import pytest

from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    SERVER_PROCESS_ATTRS,
    process_is_running,
    terminate_process_bounded,
)

# `stop_all_server_processes` is imported inside the two shutdown tests, so
# the grandchild test above still COLLECTS against a tree where it does not
# exist yet -- a test file that cannot be collected proves nothing.


pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="process groups are the posix mechanism here"
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _spawn_server_with_a_worker(marker: pathlib.Path) -> subprocess.Popen:
    """A parent that forks one long-lived child, like llama.cpp does."""
    script = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c',\n"
        "    \"import os, pathlib, time\\n\"\n"
        f"    \"p = pathlib.Path({str(marker)!r})\\n\"\n"
        "    \"while True:\\n\"\n"
        "    \"    p.write_text(str(os.getpid()))\\n\"\n"
        "    \"    time.sleep(0.05)\"])\n"
        "time.sleep(60)\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # The spawn flag under test: run_server_subprocess sets this.
        start_new_session=True,
    )


def _wait_for_worker(marker: pathlib.Path, timeout: float = 10.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists():
            text = marker.read_text().strip()
            if text.isdigit():
                return int(text)
        time.sleep(0.05)
    pytest.fail("the forked worker never reported its pid")


@pytest.fixture()
def server_with_worker():
    marker = pathlib.Path(tempfile.mkdtemp()) / "worker.pid"
    process = _spawn_server_with_a_worker(marker)
    worker_pid = _wait_for_worker(marker)
    yield process, worker_pid
    for pid in (worker_pid,):
        if _pid_alive(pid):
            try:
                os.kill(pid, 9)
            except OSError:
                pass
    if process_is_running(process):
        process.kill()


def test_stopping_a_server_also_stops_its_forked_worker(server_with_worker):
    process, worker_pid = server_with_worker
    assert _pid_alive(worker_pid)

    assert terminate_process_bounded(process, timeout=5.0) is True

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _pid_alive(worker_pid):
        time.sleep(0.05)
    assert not _pid_alive(worker_pid), (
        "the forked worker outlived the server that spawned it, and is "
        "still holding its port"
    )


def test_stopping_does_not_signal_our_own_process_group(server_with_worker):
    """The guard that keeps a group kill from taking the app down with it."""
    process, _worker_pid = server_with_worker
    own_pid = os.getpid()

    terminate_process_bounded(process, timeout=5.0)

    assert _pid_alive(own_pid), "the stop signalled the caller's own group"


def test_shutdown_stops_every_provider_and_clears_its_handle(server_with_worker):
    from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
        stop_all_server_processes,
    )

    process, worker_pid = server_with_worker

    class _App:
        pass

    app = _App()
    for attribute in SERVER_PROCESS_ATTRS.values():
        setattr(app, attribute, None)
    app.llamacpp_server_process = process

    stopped = stop_all_server_processes(app, timeout=5.0)

    assert stopped == ["llamacpp"]
    assert app.llamacpp_server_process is None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _pid_alive(worker_pid):
        time.sleep(0.05)
    assert not _pid_alive(worker_pid)


def test_shutdown_tolerates_an_app_with_no_servers():
    """Teardown must not raise on a handle that is absent or already dead."""
    from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
        stop_all_server_processes,
    )

    class _App:
        pass

    assert stop_all_server_processes(_App(), timeout=1.0) == []
