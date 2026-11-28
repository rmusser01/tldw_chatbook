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
  through ``fork_exec`` (fails on origin/dev, which forks). Every other platform
  keeps ``close_fds=True`` and lets ``subprocess`` pick its launch (on Linux a
  vfork-based ``fork_exec``, which already releases the GIL). The platform choice
  and the launcher's use of it are pinned on every platform;
* the descriptor posture: the helper closes every descriptor it inherited, low or
  high numbered, before it reads its request, so it works with exactly the three
  pipes a ``close_fds=True`` fork gave it (fails on origin/dev, whose helper keeps
  an inherited descriptor open for as long as it lives). It closes only
  inheritable descriptors, so it can never close one its own startup opened, and
  it refuses to start when it can neither list its descriptors nor bound them;
* the parent's own descriptors stay out of the helper, because Python creates them
  non-inheritable (PEP 446): a preparation reaps its child and no descriptor of
  its pipes survives in the parent.
"""

from __future__ import annotations

import importlib
import os
import resource
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="Windows prepares in-process without a helper"
)

#: Above any bounded close of the low descriptor range (e.g. ``closerange(3, 64)``).
HIGH_DESCRIPTOR = 4000


def _process():
    return importlib.import_module("tldw_chatbook.DB.private_sqlite_process")


def _deadline(seconds: float = 5.0):
    return _process().OperationDeadline(time.monotonic() + seconds)


def _request(path: Path):
    codec = importlib.import_module("tldw_chatbook.DB.private_sqlite_protocol")
    return codec.PrepareRequest(str(path), True, True, False)


def _fd_directory() -> str:
    return "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"


def _entry() -> Path:
    return (
        Path(_process().__file__).with_name("private_sqlite_helper_entry.py").resolve()
    )


def _package_bootstrap() -> str:
    """Child prelude: the fixed package namespaces, without running any __init__."""
    package = Path(_process().__file__).resolve().parents[1]
    return f"""
import sys
from pathlib import Path
from types import ModuleType
package = Path({str(package)!r})
for name, directory in (("tldw_chatbook", package), ("tldw_chatbook.DB", package / "DB")):
    namespace = ModuleType(name)
    namespace.__path__ = [str(directory)]
    sys.modules[name] = namespace
"""


def _run_child(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


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


@pytest.mark.parametrize(
    ("platform", "spawns"), [("darwin", True), ("linux", False), ("freebsd14", False)]
)
def test_only_macos_launches_without_a_fork(platform, spawns):
    """Import the launcher in a child that reports ``platform`` (runs on every OS)."""
    code = _package_bootstrap() + (
        "import subprocess, selectors, threading\n"
        f"sys.platform = {platform!r}\n"
        "import tldw_chatbook.DB.private_sqlite_process as process\n"
        "print(process._SPAWN_WITHOUT_FORK)\n"
    )
    completed = _run_child(code)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(spawns)


@pytest.mark.parametrize("spawn_without_fork", [True, False])
def test_launcher_passes_close_fds_from_the_platform_choice(
    tmp_path, monkeypatch, spawn_without_fork
):
    """The real launch honours the module's choice: close_fds is True off macOS."""
    process = _process()
    real_popen = subprocess.Popen
    captured: list[dict] = []

    def launch(*args, **kwargs):
        captured.append(dict(kwargs))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(process, "_SPAWN_WITHOUT_FORK", spawn_without_fork)
    monkeypatch.setattr(process.subprocess, "Popen", launch)
    with process.HelperAdmission().reserve(
        transient=1, retained=0, deadline=_deadline()
    ) as owner:
        lease = process.HelperLease.start(
            _request(tmp_path / "db"),
            operation="prepare",
            reservation=owner,
            deadline=_deadline(),
        )
        lease.close()
    assert lease.cleanup_state == "reaped"
    assert [kwargs["close_fds"] for kwargs in captured] == [not spawn_without_fork]


@pytest.fixture
def high_descriptor_room():
    """Let this process allocate HIGH_DESCRIPTOR; restore the soft limit after."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = HIGH_DESCRIPTOR + 1
    if soft != resource.RLIM_INFINITY and soft < wanted:
        if hard != resource.RLIM_INFINITY and hard < wanted:
            pytest.skip(f"RLIMIT_NOFILE hard limit {hard} is below {wanted}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_helper_closes_descriptors_it_inherits_before_its_request(high_descriptor_room):
    """Launch the fixed entry as the launcher does, with a low and a high inherited end.

    The helper waits for its request frame on stdin. While it waits, it must
    already have closed both inherited write ends: closing the parent's copies
    then gives the reader EOF although the helper is still alive. The high end
    fails any close that stops at a fixed low bound.
    """
    read_end, write_end = os.pipe()
    os.set_inheritable(write_end, True)
    high = os.dup2(write_end, HIGH_DESCRIPTOR, inheritable=True)
    assert high == HIGH_DESCRIPTOR
    environment = os.environ.copy()
    environment["_TLDW_PRIVATE_SQLITE_PARENT_PID"] = str(os.getpid())
    child = None
    try:
        child = subprocess.Popen(
            [sys.executable, "-I", "-S", str(_entry())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=False,
            bufsize=0,
            env=environment,
        )
        os.close(write_end)
        write_end = -1
        os.close(high)
        high = -1
        readable, _, _ = select.select([read_end], [], [], 10.0)
        alive = child.poll() is None
        assert readable and os.read(read_end, 1) == b"", (
            "the helper kept an inherited descriptor open while it waited"
        )
        assert alive, "the helper exited instead of waiting for its request"
    finally:
        os.close(read_end)
        for leftover in (write_end, high):
            if leftover >= 0:
                os.close(leftover)
        if child is not None:
            child.stdin.close()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
            child.stdout.close()


def test_helper_close_spares_descriptors_that_are_not_inheritable():
    """Only an inheritable descriptor can have crossed exec; never close another."""
    code = (
        _package_bootstrap()
        + f"""
import importlib.util, os
spec = importlib.util.spec_from_file_location("entry_under_test", {str(_entry())!r})
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
guarded = os.open(os.devnull, os.O_RDONLY)  # non-inheritable, as Python opens it
inherited = os.open(os.devnull, os.O_RDONLY)
os.set_inheritable(inherited, True)
entry._close_inherited_descriptors()
def is_open(fd):
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True
print(is_open(guarded), is_open(inherited))
"""
    )
    completed = _run_child(code)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["True", "False"]


@pytest.mark.parametrize("limit", ["overflow", "unlimited"])
def test_helper_refuses_to_start_when_it_cannot_list_or_bound_descriptors(limit):
    code = (
        _package_bootstrap()
        + f"""
import importlib.util, os
spec = importlib.util.spec_from_file_location("entry_under_test", {str(_entry())!r})
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
def no_listing(path):
    raise OSError("no descriptor listing")
def sysconf(name):
    if {limit!r} == "overflow":
        raise OverflowError("unlimited")
    return 1 << 20
os.listdir = no_listing
os.sysconf = sysconf
try:
    entry._close_inherited_descriptors()
except SystemExit as refused:
    print("refused", refused.code)
else:
    print("started")
"""
    )
    completed = _run_child(code)
    assert completed.stdout.strip() == "refused 1", completed.stderr


def test_preparation_reaps_its_helper_and_leaks_no_parent_descriptor(tmp_path):
    process = _process()
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
            child = lease._child
            pipes = {
                (info.st_dev, info.st_ino)
                for info in (
                    os.fstat(child.stdin.fileno()),
                    os.fstat(child.stdout.fileno()),
                )
            }
            lease.close()
        assert lease.cleanup_state == "reaped"
        assert child.poll() is not None
        assert child.stdin.closed and child.stdout.closed
        # Identify the pipes themselves, so other threads' descriptors cannot
        # move the result: no descriptor left in this process refers to them.
        surviving = set()
        for name in os.listdir(_fd_directory()):
            try:
                info = os.fstat(int(name))
            except (OSError, ValueError):
                continue
            surviving.add((info.st_dev, info.st_ino))
        assert not pipes & surviving
