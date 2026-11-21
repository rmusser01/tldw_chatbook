"""An owned ``tldw_test_config_*`` root goes on exits pytest_sessionfinish misses.

TASK-42145: ``pytest --version``/``--help`` and a ``SIGTERM`` left one root in
``$TMPDIR`` each, and about 440 had piled up. The child is plain Python with
only ``Tests/real_profile_guard.py`` imported (it installs nothing on import),
no app.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CHILD = """
import os, pathlib, signal, sys
sys.path.insert(0, {repo!r})
from Tests.real_profile_guard import remove_sandbox_at_exit
root = pathlib.Path({root!r})
{before}
remove_sandbox_at_exit(root)
{then}
"""
# A fork child inherits the atexit entry and the handlers but owns no root:
# terminating it (Chunking's ebook_chapters strategy does, with a fork
# context) must not remove the parent's live root (TASK-42145 review).
_TERMINATE_A_FORK_CHILD = """
import multiprocessing, time
ctx = multiprocessing.get_context("fork")
ready = ctx.Event()
def wait_forever():
    ready.set()
    time.sleep(60)
child = ctx.Process(target=wait_forever, daemon=True)
child.start()
ready.wait(30)
child.terminate()
child.join(30)
print(f"child={child.exitcode} kept={root.exists()}")
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
            f"child={-signal.SIGTERM} kept=True",
            id="fork-child-terminated",
            marks=_POSIX,
        ),
    ],
)
def test_owned_root_is_removed(
    tmp_path: Path, before: str, then: str, returncode: int, output: str
) -> None:
    root = tmp_path / "tldw_test_config_probe"
    (root / "home").mkdir(parents=True)
    child = _CHILD.format(repo=str(_REPO), root=str(root), before=before, then=then)
    finished = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=120
    )
    # A SIGTERM still ends the process by that signal, as the default did.
    assert finished.returncode == returncode, finished.stderr
    assert finished.stdout.strip() == output
    assert not root.exists()
