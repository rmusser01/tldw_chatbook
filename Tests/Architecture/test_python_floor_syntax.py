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

import json
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
    probe = subprocess.run(
        [interpreter, "-c", "import sys;print(sys.version_info[0],sys.version_info[1])"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return None
    parts = probe.stdout.split()
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])


def test_every_module_compiles_on_the_declared_python_floor() -> None:
    floor = _declared_floor()
    if sys.version_info[:2] == floor:
        pytest.skip("already running on the declared floor")

    interpreter = _floor_interpreter(floor)
    if interpreter is not None and _reports_version(interpreter) != floor:
        interpreter = None
    if interpreter is None:
        pytest.skip(
            f"no Python {floor[0]}.{floor[1]} available to check the declared "
            "floor against; install one (`uv python install "
            f"{floor[0]}.{floor[1]}`) to enable this guard"
        )

    # The file list is built HERE, on the running interpreter, and passed in --
    # not re-globbed inside the probe. `iter_source_files` prunes `.venv` /
    # `site-packages` / caches, and `.gitignore` does not affect `Path.rglob`:
    # a nested virtualenv under the package root would otherwise feed
    # third-party modules written for a newer Python into this check and report
    # them as this project's floor breaks (TASK-19906 recorded exactly that for
    # a sibling AST sweep).
    sources = [str(path) for path in iter_source_files(PACKAGE)]
    assert sources, "found no source files to check -- the sweep is misconfigured"

    probe = (
        "import sys, json\n"
        "bad = []\n"
        "for name in json.loads(sys.stdin.read()):\n"
        "    try:\n"
        "        with open(name, encoding='utf-8') as handle:\n"
        "            source = handle.read()\n"
        "    except OSError as exc:\n"
        # Not swallowed: a file this sweep cannot read is a file it did not
        # check, and reporting that as success is the failure mode this whole
        # module exists to avoid.
        "        bad.append(f'{name}: unreadable: {exc}')\n"
        "        continue\n"
        "    try:\n"
        "        compile(source, name, 'exec')\n"
        "    except SyntaxError as e:\n"
        "        bad.append(f'{name}:{e.lineno}: {e.msg}')\n"
        "print('\\n'.join(bad))\n"
    )
    result = subprocess.run(
        [interpreter, "-c", probe],
        input=json.dumps(sources),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"the floor probe itself failed on {interpreter}:\n{result.stderr}"
    )
    failures = result.stdout.strip()
    assert not failures, (
        f"modules fail to compile on Python {floor[0]}.{floor[1]}, the declared "
        f"minimum, even though they compile on {sys.version.split()[0]}:\n"
        f"{failures}"
    )


# --- The always-runs half -------------------------------------------------
#
# The compile check above is authoritative but SKIPPABLE, and a skip reports
# exactly as much as a pass. Demonstrated on this branch: with a genuine PEP
# 701 construct injected into `Utils/egress.py`, that test FAILS when 3.11 is
# reachable and SKIPS GREEN with `PATH` stripped of `uv` and `python3.11`. The
# detector below needs no floor interpreter, so it always has teeth -- at the
# cost of covering only the class that actually shipped a broken module rather
# than every possible incompatibility.

#: (source, is_floor_break) pairs. Every verdict here was taken from a REAL
#: 3.11 and a REAL 3.14 -- see `test_detector_agrees_with_the_real_floor`,
#: which re-derives them from the floor interpreter itself rather than
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
def test_detector_matches_the_pinned_floor_verdicts(
    source: str, expected_break: bool
) -> None:
    """The detector's verdict on each pinned case, with no interpreter needed."""
    found = find_floor_breaks(source, path=Path("synthetic.py"))
    assert bool(found) == expected_break, (
        f"detector said {bool(found)} for {source!r}; expected {expected_break}"
        + (f"\nfindings: {[str(f) for f in found]}" if found else "")
    )


def test_detector_agrees_with_the_real_floor_interpreter() -> None:
    """Re-derive every pinned verdict from the floor itself.

    This is what stops `PEP701_CASES` from drifting into folklore: the table is
    only trustworthy while it still matches what the interpreter does, and
    hand-maintained expectations about another Python version are exactly the
    kind of thing that rots. Skips when no floor interpreter is installed --
    legitimate here, because this test validates the detector rather than the
    package, and the detector stays pinned by the table either way.
    """
    floor = _declared_floor()
    interpreter = _floor_interpreter(floor)
    if interpreter is None or _reports_version(interpreter) != floor:
        pytest.skip(
            f"no Python {floor[0]}.{floor[1]} available to re-derive the "
            "pinned verdicts from"
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
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"floor probe failed:\n{result.stderr}"
    real = json.loads(result.stdout.strip().splitlines()[-1])

    disagreements = [
        f"{source!r}: real {floor[0]}.{floor[1]} says "
        f"{'SyntaxError' if is_real else 'OK'}, table says "
        f"{'SyntaxError' if expected else 'OK'}"
        for (source, expected), is_real in zip(PEP701_CASES, real)
        if expected != is_real
    ]
    assert not disagreements, "PEP701_CASES has drifted from reality:\n" + "\n".join(
        disagreements
    )


def test_no_shipped_module_uses_syntax_the_floor_cannot_parse() -> None:
    """The guard that never skips.

    Runs the detector over every shipped module on whatever interpreter is
    present. Partial by construction (see `Tests/floor_syntax`), but it cannot
    be silently disabled by an environment that lacks a floor interpreter --
    which is precisely how `TTS/backends/kokoro.py` shipped unimportable on the
    declared floor.
    """
    findings = []
    for path in iter_source_files(PACKAGE):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            findings.append(f"{path}: unreadable: {exc}")
            continue
        findings.extend(str(item) for item in find_floor_breaks(source, path=path))

    floor = _declared_floor()
    assert not findings, (
        f"modules use syntax Python {floor[0]}.{floor[1]} cannot parse, so they "
        f"cannot be imported on the declared floor:\n" + "\n".join(findings)
    )
