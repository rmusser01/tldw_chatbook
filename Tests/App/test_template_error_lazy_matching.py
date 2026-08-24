"""The except-time template-error matcher must never mask or import (TASK-21102).

``app._template_resolution_errors()`` is evaluated INSIDE the ingest dispatch
loop's ``except _template_resolution_errors() as exc:`` clause, which means it
runs for EVERY exception that reaches that site -- not only template errors.
The review of the first TASK-21102 round proved two failure modes of a naive
lazy import there:

* with Chunking unimportable (broken install), an unrelated ``ValueError``
  propagated as ``ModuleNotFoundError`` raised from the matcher itself -- the
  user-visible error class changed and the original error was buried;
* on a healthy install, an unrelated exception imported all ~39 Chunking
  modules as a side effect of exception HANDLING.

The fix: the matcher returns ``()`` (matches nothing, so the original
exception propagates untouched) unless ``tldw_chatbook.Chunking`` is already
resident -- no template error can be in flight if the package that defines
those exception classes was never imported -- and any failure of the imports
themselves also yields ``()``.

Subprocess-isolated (pattern of ``Tests/Packaging/test_chunking_import_
closure.py`` / TASK-21104): the healthy-env test asserts Chunking is NOT
resident, which an earlier in-process test could invalidate, and the
poisoned-env test must block Chunking before ``tldw_chatbook.app`` is
imported.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter with isolated config/data dirs.

    Args:
        tmp_path: Per-test scratch directory for the subprocess's HOME/XDG so
            the app import can never read or write the live user config.
        code: The Python source to execute with ``python -c``.

    Returns:
        The completed process (never raises on nonzero exit).
    """
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


# Mirrors the production handler shape exactly: the matcher call sits in the
# except clause, after a more specific clause, with no generic handler below
# it -- so whatever the matcher does happens during the handling of ANY
# exception from the try block.
_POISONED_CHUNKING_SNIPPET = """
import importlib.abc
import sys


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "tldw_chatbook" and ".Chunking" in fullname:
            raise ImportError(f"blocked by test: {fullname}")
        if fullname == "tldw_chatbook.Chunking":
            raise ImportError(f"blocked by test: {fullname}")
        return None


sys.meta_path.insert(0, _Blocker())

from tldw_chatbook.app import _template_resolution_errors

try:
    try:
        raise ValueError("the original error")
    except _template_resolution_errors():
        print("OUTCOME:swallowed-as-template-error")
except ValueError as exc:
    assert "the original error" in str(exc)
    print("OUTCOME:original-ValueError-propagated")
except BaseException as exc:  # the naive matcher raised from the except clause
    print(f"OUTCOME:replaced-by:{type(exc).__name__}")
"""


def test_unrelated_error_survives_broken_chunking_install(tmp_path: Path) -> None:
    """With Chunking unimportable, an unrelated error keeps its class.

    Review probe (a): before the guard, the matcher's own import raised
    ``ModuleNotFoundError`` from inside the except clause, replacing the
    in-flight ``ValueError`` as the user-visible error and burying the
    original. The matcher must return ``()`` instead, letting the original
    propagate untouched.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _POISONED_CHUNKING_SNIPPET)
    assert result.returncode == 0, (
        f"probe subprocess crashed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "OUTCOME:original-ValueError-propagated" in result.stdout, (
        "an unrelated ValueError must propagate with its own class when "
        f"Chunking is unimportable:\n{result.stdout}"
    )


_HEALTHY_NO_SIDE_IMPORT_SNIPPET = """
import sys

from tldw_chatbook.app import _template_resolution_errors

assert "tldw_chatbook.Chunking" not in sys.modules, (
    "precondition broken: app import made Chunking resident"
)

try:
    try:
        raise ValueError("unrelated")
    except _template_resolution_errors():
        raise AssertionError("unrelated error swallowed as template error")
except ValueError:
    pass

resident = sorted(
    m for m in sys.modules
    if m == "tldw_chatbook.Chunking" or m.startswith("tldw_chatbook.Chunking.")
)
assert not resident, (
    f"handling an unrelated exception imported Chunking as a side effect: {resident}"
)
print("NO_SIDE_IMPORT_OK")
"""


def test_unrelated_error_does_not_import_chunking(tmp_path: Path) -> None:
    """On a healthy install, matching an unrelated error must not import Chunking.

    Review probe (b): before the guard, evaluating the except clause for ANY
    exception imported all ~39 Chunking modules mid-exception-handling. With
    Chunking absent from ``sys.modules``, no template error can be in flight
    (its classes cannot have been instantiated), so the matcher must return
    ``()`` without importing anything.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _HEALTHY_NO_SIDE_IMPORT_SNIPPET)
    assert result.returncode == 0, (
        f"probe subprocess failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "NO_SIDE_IMPORT_OK" in result.stdout


_REAL_TEMPLATE_ERROR_STILL_CAUGHT_SNIPPET = """
import sys

from tldw_chatbook.app import _template_resolution_errors
from tldw_chatbook.Chunking.template_runtime import TemplateResolutionError
from tldw_chatbook.Chunking.chunking_interop_library import InvalidTemplateError

assert "tldw_chatbook.Chunking" in sys.modules

for error_class in (TemplateResolutionError, InvalidTemplateError):
    try:
        raise error_class("boom")
    except _template_resolution_errors() as exc:
        assert type(exc) is error_class
print("TEMPLATE_ERRORS_STILL_CAUGHT_OK")
"""


def test_real_template_errors_still_caught_when_chunking_resident(
    tmp_path: Path,
) -> None:
    """Once Chunking IS resident, both named error types still match.

    The anti-overcorrection direction: a matcher degraded to ``return ()``
    unconditionally would pass the two guards above while silently turning
    the dispatch loop's named template failure (task 10, AC 37/AC-24b) into
    an unhandled crash. When a template error can actually be in flight --
    i.e. its defining modules are resident -- the matcher must return the
    identical class objects the raising code uses.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _REAL_TEMPLATE_ERROR_STILL_CAUGHT_SNIPPET)
    assert result.returncode == 0, (
        f"probe subprocess failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "TEMPLATE_ERRORS_STILL_CAUGHT_OK" in result.stdout
