"""Every shipped module must compile on the project's declared Python floor.

The historical Python 3.11 floor hid a real defect: a nested
same-quote f-string (`f"...{"literal"}..."`) is legal only from Python 3.12
(PEP 701), so `TTS/backends/kokoro.py` failed to import **entirely** on 3.11
while every local test passed on 3.14.

Two things made it invisible:

* `ast.parse(..., feature_version=(3, 11))` does NOT reproduce it -- that
  argument does not downgrade the tokenizer, so it happily accepted the
  3.12-only form. A green `feature_version` check is not evidence.
* The test suite ran on the developer's interpreter, where the syntax is
  valid, so the module imported fine and its tests passed.

The supported floor is now Python 3.12. This guard compiles every module under
that actual interpreter, including when pytest itself is already running on
3.12. The old detector remains only as pinned historical Python 3.11 evidence;
it is not a rejection rule for syntax that 3.12 deliberately supports.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from Tests.floor_syntax import find_floor_breaks, iter_source_files

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


def _version_interpreter(version: tuple[int, int]) -> str | None:
    """Locate an interpreter matching the declared floor, if one exists."""
    if sys.version_info[:2] == version:
        return sys.executable
    exact = shutil.which(f"python{version[0]}.{version[1]}")
    if exact:
        return exact
    uv = shutil.which("uv")
    if not uv:
        return None
    found = subprocess.run(
        [uv, "python", "find", f"{version[0]}.{version[1]}"],
        capture_output=True,
        check=False,
        text=True,
        env={**os.environ, "UV_PYTHON_DOWNLOADS": "never"},
    )
    candidate = found.stdout.strip()
    if found.returncode != 0 or not candidate:
        return None
    return candidate


def _reports_version(interpreter: str) -> tuple[int, int] | None:
    """Ask an interpreter what version it actually is.

    `python3.11` on PATH and `uv python find 3.11` are both *claims*. A guard
    whose whole purpose is "the floor really parses this" must not accept a
    claim it never checked -- a wrong interpreter here turns the check into a
    second run of the developer's own version, which always passes.
    """
    try:
        probe = subprocess.run(
            [
                interpreter,
                "-c",
                "import sys;print('FLOORPROBE',*sys.version_info[:2])",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        # A hung or unrunnable interpreter must not stall the suite. Treating
        # it as "not available" is right: an interpreter we cannot question is
        # one we cannot verify.
        return None
    if probe.returncode != 0:
        return None
    # Scan for the marker line rather than parsing all of stdout. A site
    # customisation, a warning routed to stdout, or a venv activation notice
    # would otherwise make a perfectly good floor interpreter look unusable --
    # and that failure is silent, because it degrades into the skip this whole
    # module exists to avoid.
    for line in probe.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == "FLOORPROBE" and all(
            field.isdigit() for field in fields[1:]
        ):
            return int(fields[1]), int(fields[2])
    return None


def _declared_floor_interpreter() -> tuple[tuple[int, int], str]:
    """Return the verified floor interpreter or report the qualification gap."""
    floor = _declared_floor()
    interpreter = _version_interpreter(floor)
    if interpreter is None or _reports_version(interpreter) != floor:
        pytest.skip(
            f"no Python {floor[0]}.{floor[1]} available to check the declared "
            "floor against; install that interpreter to enable this guard"
        )
    return floor, interpreter


def _compile_sources(interpreter: str, sources: dict[str, str]) -> list[str]:
    """Compile exact source texts through one real-interpreter subprocess."""
    probe = (
        "import sys, json\n"
        "bad = []\n"
        "for name, source in json.loads(sys.stdin.read()).items():\n"
        "    try:\n"
        "        compile(source, name, 'exec')\n"
        "    except SyntaxError as e:\n"
        "        bad.append(f'{name}:{e.lineno}: {e.msg}')\n"
        "print(json.dumps(bad))\n"
    )
    result = subprocess.run(
        [interpreter, "-c", probe],
        input=json.dumps(sources),
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"the floor probe itself failed on {interpreter}:\n{result.stderr}"
    )
    return list(json.loads(result.stdout.strip().splitlines()[-1]))


def test_every_module_compiles_on_the_declared_python_floor() -> None:
    floor, interpreter = _declared_floor_interpreter()

    source_paths = list(iter_source_files(PACKAGE))
    assert source_paths, "found no source files to check -- the sweep is misconfigured"
    sources: dict[str, str] = {}
    unreadable: list[str] = []
    for path in source_paths:
        try:
            sources[str(path)] = path.read_text(encoding="utf-8")
        except OSError as error:
            unreadable.append(f"{path}: unreadable: {error}")
    assert not unreadable, "\n".join(unreadable)

    failures = _compile_sources(interpreter, sources)
    assert not failures, (
        f"modules fail to compile on Python {floor[0]}.{floor[1]}, the declared "
        f"minimum, even though pytest runs on {sys.version.split()[0]}:\n"
        + "\n".join(failures)
    )


def test_declared_floor_compile_harness_accepts_pep701_and_rejects_invalid_syntax() -> (
    None
):
    _floor, interpreter = _declared_floor_interpreter()
    failures = _compile_sources(
        interpreter,
        {
            "pep701.py": 'x = f"{ {"k": 1}["k"] }"',
            "broken.py": "def broken(: pass",
        },
    )

    assert not any(failure.startswith("pep701.py:") for failure in failures)
    assert len(failures) == 1
    assert failures[0].startswith("broken.py:1:")


# --- Historical Python 3.11 evidence -------------------------------------
#
# This detector stays pinned to the prior floor because it records the exact
# PEP 701 defect that motivated the real-interpreter guard. Python 3.12 accepts
# these constructs, so the detector must not scan shipped modules as a current
# rejection rule.

#: (source, is_floor_break) pairs. Every verdict here was taken from a REAL
#: 3.11 and a REAL 3.14 -- see
#: `test_detector_agrees_with_the_real_python311_interpreter`, which
#: re-derives them from a Python 3.11 interpreter rather than
#: trusting this table. The table exists so the detector is still pinned when
#: no floor interpreter is installed.
PEP701_CASES = [
    ('x = f"{value}"', False),
    ('x = f"{ {"k": 1}["k"] }"', True),
    ("""x = f"{ {'k': 1}['k'] }" """, False),
    ("x = f'{ d['k'] }'", True),
    ('x = f"""{ d["k"] }"""', False),
    ('x = f"""{ d["""k"""] }"""', True),
    ('x = f"{ f"{inner}" }"', True),
    ("""x = f"{ f'{inner}' }" """, False),
    ('x = f"{ chr(10).join(p) }"', False),
    ('x = f"{ "a\\nb".strip() }"', True),
    ("""x = f"{ 'a\\nb'.strip() }" """, True),
    ('x = f"{v:\\>10}"', False),
    ("""x = f'{ d["k"] }'""", False),
    ('x = f"a" "b"', False),
    ('x = f"{v:{width}}"', False),
]


@pytest.mark.parametrize("source, expected_break", PEP701_CASES)
def test_detector_matches_the_pinned_python311_verdicts(
    source: str, expected_break: bool
) -> None:
    """The detector's verdict on each pinned case, with no interpreter needed.

    Args:
        source: A one-line module whose floor-compatibility is pinned.
        expected_break: Whether a real Python 3.11 interpreter rejects ``source``.
    """
    found = find_floor_breaks(source, path=Path("synthetic.py"))
    assert bool(found) == expected_break, (
        f"detector said {bool(found)} for {source!r}; expected {expected_break}"
        + (f"\nfindings: {[str(f) for f in found]}" if found else "")
    )


def test_detector_agrees_with_the_real_python311_interpreter() -> None:
    """Re-derive every pinned historical verdict from Python 3.11 itself.

    This is what stops `PEP701_CASES` from drifting into folklore: the table is
    only trustworthy while it still matches what the interpreter does, and
    hand-maintained expectations about another Python version are exactly the
    kind of thing that rots. Skips when no floor interpreter is installed --
    legitimate here, because this test validates the detector rather than the
    package, and the detector stays pinned by the table either way.
    """
    legacy_version = (3, 11)
    interpreter = _version_interpreter(legacy_version)
    if interpreter is None or _reports_version(interpreter) != legacy_version:
        pytest.skip(
            "no Python 3.11 available to re-derive the pinned historical verdicts from"
        )

    probe = (
        "import sys, json\n"
        "out = []\n"
        "for src in json.loads(sys.stdin.read()):\n"
        "    try:\n"
        "        compile(src, '<case>', 'exec')\n"
        "        out.append(False)\n"
        "    except SyntaxError:\n"
        "        out.append(True)\n"
        "print(json.dumps(out))\n"
    )
    sources = [source for source, _ in PEP701_CASES]
    result = subprocess.run(
        [interpreter, "-c", probe],
        input=json.dumps(sources),
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"floor probe failed:\n{result.stderr}"
    real = json.loads(result.stdout.strip().splitlines()[-1])

    disagreements = [
        f"{source!r}: real Python 3.11 says "
        f"{'SyntaxError' if is_real else 'OK'}, table says "
        f"{'SyntaxError' if expected else 'OK'}"
        for (source, expected), is_real in zip(PEP701_CASES, real)
        if expected != is_real
    ]
    assert not disagreements, "PEP701_CASES has drifted from reality:\n" + "\n".join(
        disagreements
    )
