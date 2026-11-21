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
remove_sandbox_at_exit(pathlib.Path({root!r}))
{then}
"""


@pytest.mark.parametrize(
    ("then", "returncode"),
    [
        pytest.param("pass", 0, id="normal-exit"),
        pytest.param(
            "os.kill(os.getpid(), signal.SIGTERM)\nsignal.pause()",
            -signal.SIGTERM,
            id="sigterm",
            marks=pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals"),
        ),
    ],
)
def test_owned_root_is_removed(tmp_path: Path, then: str, returncode: int) -> None:
    root = tmp_path / "tldw_test_config_probe"
    (root / "home").mkdir(parents=True)
    child = _CHILD.format(repo=str(_REPO), root=str(root), then=then)
    finished = subprocess.run([sys.executable, "-c", child], timeout=60)
    # A SIGTERM still ends the process by that signal, as the default did.
    assert finished.returncode == returncode
    assert not root.exists()
