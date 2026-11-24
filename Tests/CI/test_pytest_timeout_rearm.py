"""TASK-41135: a teardown that hangs after a failed test still times out.

pytest-timeout cancels its timer on any test failure, so before the re-arm
hook (`pytest_exception_interact` in Tests/private_profile.py) each fixture
below slept its full 60 s and the run reported only the call failure. Two
hanging finalizers pin that the re-armed timer repeats: a one-shot timer
frees only the first.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HANGING_TEARDOWN = """
import time

import pytest


@pytest.fixture
def hangs_on_teardown():
    yield
    time.sleep(60)


@pytest.fixture
def also_hangs_on_teardown():
    yield
    time.sleep(60)


def test_fails(hangs_on_teardown, also_hangs_on_teardown):
    assert False
"""


def test_teardown_after_a_failed_call_is_still_timed_out(tmp_path):
    (tmp_path / "test_hang.py").write_text(_HANGING_TEARDOWN, encoding="utf-8")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name != "PYTEST_ADDOPTS" and not name.startswith("PYTEST_XDIST")
    }
    environment.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONPATH=str(PROJECT_ROOT),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    started = time.monotonic()
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pytest_timeout",
            "-p",
            "Tests.private_profile",
            "-p",
            "no:cacheprovider",
            "--noconftest",
            "--rootdir",
            str(tmp_path),
            "-c",
            os.devnull,
            "--timeout=2",
            str(tmp_path / "test_hang.py"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    elapsed = time.monotonic() - started
    output = run.stdout + run.stderr
    assert "Timeout" in output, output
    assert elapsed < 40, (elapsed, output)
