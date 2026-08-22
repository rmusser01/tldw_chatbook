"""Every shipped module must parse on the project's MINIMUM Python (TASK-19560 review).

`pyproject.toml` declares `requires-python = ">=3.11"`, but development here
happens on a much newer interpreter. That gap hid a real defect: a nested
same-quote f-string (`f"...{"literal"}..."`) is legal only from Python 3.12
(PEP 701), so `TTS/backends/kokoro.py` failed to import **entirely** on 3.11
while every local test passed on 3.14.

Two things made it invisible:

* `ast.parse(..., feature_version=(3, 11))` does NOT reproduce it -- that
  argument does not downgrade the tokenizer, so it happily accepted the
  3.12-only form. A green `feature_version` check is not evidence.
* The test suite ran on the developer's interpreter, where the syntax is
  valid, so the module imported fine and its tests passed.

This guard compiles every module under a real minimum-version interpreter
when one is available, and SKIPS with a clear reason when it is not -- rather
than silently passing and re-creating exactly the blind spot it exists to
close.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT_ROOT / "tldw_chatbook"


def _declared_floor() -> tuple[int, int]:
    """Read `requires-python` from pyproject rather than hardcoding it.

    Returns:
        The (major, minor) minimum version the project claims to support.
    """
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*"[^"]*?(\d+)\.(\d+)', text)
    assert match, "could not read requires-python from pyproject.toml"
    return int(match.group(1)), int(match.group(2))


def _floor_interpreter(floor: tuple[int, int]) -> str | None:
    """Locate an interpreter matching the declared floor, if one exists."""
    exact = shutil.which(f"python{floor[0]}.{floor[1]}")
    if exact:
        return exact
    uv = shutil.which("uv")
    if not uv:
        return None
    found = subprocess.run(
        [uv, "python", "find", f"{floor[0]}.{floor[1]}"],
        capture_output=True, text=True,
    )
    candidate = found.stdout.strip()
    return candidate if found.returncode == 0 and candidate else None


def test_every_module_compiles_on_the_declared_python_floor() -> None:
    floor = _declared_floor()
    if sys.version_info[:2] == floor:
        pytest.skip("already running on the declared floor")

    interpreter = _floor_interpreter(floor)
    if interpreter is None:
        pytest.skip(
            f"no Python {floor[0]}.{floor[1]} available to check the declared "
            "floor against; install one (`uv python install "
            f"{floor[0]}.{floor[1]}`) to enable this guard"
        )

    probe = (
        "import sys, pathlib\n"
        "bad = []\n"
        f"for f in pathlib.Path({str(PACKAGE)!r}).rglob('*.py'):\n"
        "    try:\n"
        "        compile(f.read_text(encoding='utf-8'), str(f), 'exec')\n"
        "    except SyntaxError as e:\n"
        "        bad.append(f'{f}:{e.lineno}: {e.msg}')\n"
        "    except Exception:\n"
        "        pass\n"
        "print('\\n'.join(bad))\n"
    )
    result = subprocess.run(
        [interpreter, "-c", probe], capture_output=True, text=True, timeout=300
    )
    failures = result.stdout.strip()
    assert not failures, (
        f"modules fail to compile on Python {floor[0]}.{floor[1]}, the declared "
        f"minimum, even though they compile on {sys.version.split()[0]}:\n"
        f"{failures}"
    )
