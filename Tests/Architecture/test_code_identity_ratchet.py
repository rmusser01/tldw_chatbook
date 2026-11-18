"""tldw_chatbook must not grow caller-frame or function-identity reads (TASK-40719).

ADR-356 rule 4: authority comes from runtime-owned registration and current
state, never from function objects, code/defaults/closures or caller frames.
``test_chat_code_identity_ratchet.py`` pins three of these names in Chat/ only;
TASK-40719 found the growth from PR #3480 outside Chat/ (Agents, Tools,
Workspaces, MCP, app wiring), so this pins every name below over the whole
package.

Rule: over ``tldw_chatbook/**/*.py``, count every AST node that is
* an ``Attribute`` whose ``attr`` is one of ``PINS`` (``fn.__code__``,
  ``sys._getframe``, ``frame.f_locals``, ``coro.cr_code`` ...), or
* a string ``Constant`` exactly equal to one of ``PINS`` (``getattr(fn,
  "__defaults__")``, ``getattr_static(..., "__func__")``,
  ``vars(fn).get("__wrapped__")``, ``vars(fn)["__closure__"]``).
Counts are per name, so a removal under one name cannot pay for an addition
under another.

Known limits:
* A name built at runtime (``getattr(f, "__co" + "de__")``, or a name held in
  a variable) is not seen; only literal attributes and exact constants are.
* Pins are package totals per name, so a removal in one file can pay for an
  addition in another under the same name.

ponytail: package totals per name, not per file. If the Agents/ or Chat/ hot
spots need it, pin those directories (or files) separately the way
``test_chat_code_identity_ratchet.py`` pins Chat/.
"""

from __future__ import annotations

import ast
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"

# Measured after TASK-40719. Its base (dev 1455ceeb96) was one higher for
# _getframe, __defaults__, __kwdefaults__, __closure__, cell_contents,
# __globals__ and __wrapped__, two higher for f_code and __func__, and three
# higher for f_locals and __code__. Rebased onto dev bf0006bb47, which had
# added one __func__ (de8880504c) and three __code__ (11be39b8c3) sites.
# TASK-41319 (Console Send skill trust decided from the app's registrations)
# removed 68: __func__ 2, __code__ 18, __defaults__ 8, __kwdefaults__ 9,
# __closure__ 12, cell_contents 3, __globals__ 9, __wrapped__ 7.
# Lower a row when you remove sites.
PINS = {
    # caller frames, coroutine/generator implementation
    "_getframe": 14,
    "currentframe": 0,
    "_current_frames": 1,
    "getouterframes": 0,
    "extract_stack": 1,
    "f_back": 2,
    "f_code": 26,
    "f_locals": 5,
    "f_globals": 13,
    "cr_code": 1,
    "cr_frame": 1,
    "gi_code": 0,
    "gi_frame": 0,
    "ag_code": 0,
    "ag_frame": 0,
    # function, code, default and closure identity
    "__func__": 142,
    "__code__": 383,
    "__defaults__": 202,
    "__kwdefaults__": 305,
    "__closure__": 238,
    "cell_contents": 126,
    "__globals__": 209,
    "__wrapped__": 205,
}


def identity_reads(source: str) -> Counter[str]:
    """Count the PINS names ``source`` reads, by the module docstring's rule."""
    counts: Counter[str] = Counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # vendored escapes
        tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in PINS:
            counts[node.attr] += 1
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in PINS
        ):
            counts[node.value] += 1
    return counts


@lru_cache(maxsize=1)
def _package_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in PACKAGE.rglob("*.py"):
        counts += identity_reads(path.read_text(encoding="utf-8"))
    return counts


def over_pin(counts: Counter[str]) -> dict[str, tuple[int, int]]:
    return {name: (counts[name], pin) for name, pin in PINS.items() if counts[name] > pin}


def test_package_identity_reads_do_not_rise():
    over = over_pin(_package_counts())
    assert not over, (
        f"tldw_chatbook reads more caller frames or function identity than pinned "
        f"(name: (count, pin)): {over}. ADR-356 rule 4: decide from a runtime-owned "
        "registration and current state instead."
    )


# --- negative controls: the rule sees every shape, and only these names ---


@pytest.mark.parametrize(
    ("line", "name"),
    [
        ("x = fn.__code__", "__code__"),
        ("x = getattr(fn, '__defaults__', None)", "__defaults__"),
        ("x = inspect.getattr_static(fn, '__kwdefaults__')", "__kwdefaults__"),
        ("x = vars(fn)['__closure__']", "__closure__"),
        ("x = vars(fn).get('__wrapped__')", "__wrapped__"),
        ("x = cell.cell_contents", "cell_contents"),
        ("x = fn.__globals__", "__globals__"),
        ("x = method.__func__ is Owner.method", "__func__"),
        ("x = sys._getframe(1)", "_getframe"),
        ("x = inspect.currentframe()", "currentframe"),
        ("x = frame.f_back.f_code", "f_back"),
        ("x = frame.f_locals.get('self')", "f_locals"),
        ("x = frame.f_globals", "f_globals"),
        ("x = coro.cr_code", "cr_code"),
        ("x = gen.gi_frame", "gi_frame"),
    ],
)
def test_detector_counts_each_shape(line, name):
    assert identity_reads(line)[name] == 1


def test_detector_ignores_lookalikes():
    source = (
        '"""Mentions fn.__code__ and sys._getframe in prose."""\n'
        "code = self.code\n"
        "frames = self.frame_count\n"
        "x = getattr(obj, 'func')\n"
        "y = {'name': 'f_code_value'}\n"
    )
    assert identity_reads(source) == Counter()


def test_one_new_site_trips_the_pin():
    assert over_pin(_package_counts()) == {}
    counts = _package_counts() + identity_reads("x = fn.__code__\ny = sys._getframe(1).f_locals")
    assert set(over_pin(counts)) == {"__code__", "_getframe", "f_locals"}


def test_the_scan_reads_the_whole_package():
    files = list(PACKAGE.rglob("*.py"))
    assert len(files) > 1000
    assert any(path.parent.name == "Agents" for path in files)
    assert _package_counts()["__code__"] > 0
