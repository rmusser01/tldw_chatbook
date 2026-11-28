"""An owned ``tldw_test_config_*`` root goes on exits pytest_sessionfinish misses.

TASK-42145: ``pytest --version``/``--help`` and a ``SIGTERM`` left one root in
``$TMPDIR`` each, and about 440 had piled up. The helper's children are plain
Python with only ``Tests/real_profile_guard.py`` imported (it installs nothing
on import), no app; the last test runs the root conftest itself.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
# Every child script starts with a fork budget per lineage: a fork that runs
# away (a hook that forked before setting its once-flag filled the process
# table, TASK-42651) then fails at once with EAGAIN instead, at most about
# 2**_FORK_BUDGET processes in, on any platform or PID namespace. A forked child
# inherits its parent's count. No param forks more than twice in one lineage
# (fork-inside-a-fork-window). Only os.fork() is counted, which multiprocessing's
# fork context calls at call time; subprocess, posix_spawn, os.forkpty and
# C-level forks are not.
_FORK_BUDGET = 3
_CHILD = """
import errno, os, pathlib, signal, sys
if hasattr(os, "fork"):
    _real_fork, _forks = os.fork, 0
    def _budgeted_fork():
        global _forks
        if _forks >= {budget}:
            raise BlockingIOError(errno.EAGAIN, "test fork budget spent")
        _forks += 1
        return _real_fork()
    os.fork = _budgeted_fork
sys.path.insert(0, {repo!r})
from Tests.real_profile_guard import remove_sandbox_at_exit
root = pathlib.Path({root!r})
{before}
remove_sandbox_at_exit(root)
{then}
"""
# A fork child inherits the atexit entry and the handlers but owns no root:
# terminating it (Chunking's ebook_chapters strategy does, with a fork
# context) must not remove the parent's live root (TASK-42145 review). It gets
# the default SIGTERM back (TASK-42651): with the inherited Python handler, a
# SIGTERM that landed just before the sleep only ran it when the sleep ended,
# after the join (17 of 1000 runs at -n 8).
_TERMINATE_A_FORK_CHILD = """
import multiprocessing, time
ctx = multiprocessing.get_context("fork")
ready = ctx.Event()
def wait_forever():
    print(f"default={signal.getsignal(signal.SIGTERM) is signal.SIG_DFL}", flush=True)
    ready.set()
    time.sleep(60)
child = ctx.Process(target=wait_forever, daemon=True)
child.start()
ready.wait(30)
child.terminate()
child.join(30)
print(f"child={child.exitcode} kept={root.exists()}")
"""
# The guard's signals stay blocked from just before a fork until the child's
# reset (TASK-42651: one landing earlier tripped the inherited handler and was
# dropped). These hooks are registered before the guard's, so they run inside
# that window: their signal is held ("held"), then the default kills the
# child, or the parent's own handler removes the root and dies.
_FORK = """
pid = os.fork()
if pid == 0:
    os._exit(0)
"""
_SIGNAL_THE_CHILD_BEFORE_ITS_RESET = (
    "os.register_at_fork(after_in_child=lambda: "
    "(os.kill(os.getpid(), signal.{sig}), print('held', flush=True)))"
)
_SIGNAL_THE_PARENT_AS_IT_FORKS = (
    "os.register_at_fork(before=lambda: "
    "(os.kill(os.getpid(), signal.SIGTERM), print('held', flush=True)))"
)
_REAP_THE_CHILD = (
    'print(f"child={os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])} '
    'kept={root.exists()}")'
)
# A fork inside another fork's window: the inner child leaves every window
# unblocked, the parent only the inner one, then the outer one.
_FORK_INSIDE_A_FORK = """
def blocked():
    return signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, [])
nested = []
def fork_inside():
    # Flag first: the os.fork() below runs every before hook again, this one
    # too (setting it after the fork made each level fork again: a fork bomb).
    if nested:
        return
    nested.append(True)
    inner = os.fork()
    if inner == 0:
        print("inner", blocked(), flush=True)
        os._exit(0)
    os.waitpid(inner, 0)
os.register_at_fork(before=fork_inside)
"""
# An exception inside the child's reset still unblocks (TASK-42651 review).
_BREAK_THE_CHILDS_RESET = """
real_signal = signal.signal
def broken(*_args):
    raise KeyboardInterrupt("raised inside the guard's reset")
os.register_at_fork(after_in_child=lambda: setattr(signal, "signal", broken))
"""
_FORK_AND_REPORT_THE_MASK = """
os.register_at_fork(after_in_child=lambda: setattr(signal, "signal", real_signal))
pid = os.fork()
if pid == 0:
    print(signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, []), flush=True)
    os._exit(0)
os.waitpid(pid, 0)
"""
# A handler someone else installed is neither reset nor blocked across a fork:
# the first read runs inside the window (a hook registered before the guard's).
_IGNORE_SIGTERM = """
signal.signal(signal.SIGTERM, signal.SIG_IGN)
os.register_at_fork(after_in_child=lambda: print(
    signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, []), flush=True))
"""
_FORK_AND_REPORT_SIGTERM = """
pid = os.fork()
if pid == 0:
    blocked = signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, [])
    print(signal.getsignal(signal.SIGTERM) is signal.SIG_IGN, blocked, flush=True)
    os._exit(0)
os.waitpid(pid, 0)
"""
_POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")


@pytest.mark.parametrize(
    ("before", "then", "returncode", "output"),
    [
        pytest.param("", "pass", 0, "", id="normal-exit"),
        pytest.param(
            "",
            "os.kill(os.getpid(), signal.SIGTERM)\nsignal.pause()",
            -signal.SIGTERM,
            "",
            id="sigterm",
            marks=_POSIX,
        ),
        # A disposition someone else set stays; atexit still removes the root.
        pytest.param(
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "os.kill(os.getpid(), signal.SIGTERM)",
            0,
            "",
            id="sigterm-ignored",
            marks=_POSIX,
        ),
        pytest.param(
            "",
            _TERMINATE_A_FORK_CHILD,
            0,
            f"default=True\nchild={-signal.SIGTERM} kept=True",
            id="fork-child-terminated",
            marks=_POSIX,
        ),
        *(
            pytest.param(
                _SIGNAL_THE_CHILD_BEFORE_ITS_RESET.format(sig=sig.name),
                _FORK + _REAP_THE_CHILD,
                0,
                f"held\nchild={-sig} kept=True",
                id=f"fork-child-{sig.name.lower()}-before-its-reset",
                marks=_POSIX,
            )
            for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM))
        ),
        pytest.param(
            _SIGNAL_THE_PARENT_AS_IT_FORKS,
            _FORK + 'print("parent lived")',
            -signal.SIGTERM,
            "held",
            id="parent-sigterm-as-it-forks",
            marks=_POSIX,
        ),
        pytest.param(
            _IGNORE_SIGTERM,
            _FORK_AND_REPORT_SIGTERM,
            0,
            "False\nTrue False",
            id="fork-child-keeps-a-foreign-sigterm-handler",
            marks=_POSIX,
        ),
        pytest.param(
            _FORK_INSIDE_A_FORK,
            _FORK + 'os.waitpid(pid, 0)\nprint("parent", blocked())',
            0,
            "inner False\nparent False",
            id="fork-inside-a-fork-window",
            marks=_POSIX,
        ),
        pytest.param(
            _BREAK_THE_CHILDS_RESET,
            _FORK_AND_REPORT_THE_MASK,
            0,
            "False",
            id="fork-child-unblocks-when-its-reset-raises",
            marks=_POSIX,
        ),
    ],
)
def test_owned_root_is_removed(
    tmp_path: Path, before: str, then: str, returncode: int, output: str
) -> None:
    root = tmp_path / "tldw_test_config_probe"
    (root / "home").mkdir(parents=True)
    child = _CHILD.format(
        repo=str(_REPO), root=str(root), budget=_FORK_BUDGET, before=before, then=then
    )
    finished = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=120
    )
    # A SIGTERM still ends the process by that signal, as the default did.
    assert finished.returncode == returncode, finished.stderr
    assert finished.stdout.strip() == output, finished.stderr
    assert not root.exists()


def test_pytest_version_leaves_no_bootstrap_root(tmp_path: Path) -> None:
    """The root conftest's own wiring: ``--version`` loads it (``testpaths``)
    and never reaches ``pytest_sessionfinish``, the run mode that left a
    ``tldw_test_config_*`` root behind every time."""
    env = dict(os.environ, TMPDIR=str(tmp_path))
    # A fresh session, as from a shell: it creates and owns its own root.
    for name in (
        "TLDW_TEST_CONFIG_ROOT",
        "TLDW_TEST_CONFIG_ROOT_OWNER",
        "PYTEST_XDIST_WORKER",
    ):
        env.pop(name, None)
    finished = subprocess.run(
        [sys.executable, "-m", "pytest", "--version", "-p", "no:cacheprovider"],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert finished.returncode == 0, finished.stderr
    assert list(tmp_path.glob("tldw_test_config_*")) == []
