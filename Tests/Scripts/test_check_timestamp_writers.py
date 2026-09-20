"""The timestamp-writer guard detects the two forbidden idioms (TASK-32803.1)."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SPEC = importlib.util.spec_from_file_location(
    "_cts",
    Path(__file__).resolve().parents[2] / "scripts" / "check_timestamp_writers.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)  # type: ignore[union-attr]


def _scan(src: str):
    v = _mod._Visitor("m")
    v.visit(ast.parse(src))
    return v.hits


def test_flags_datetime_utcnow():
    hits = _scan("import datetime\ndef f():\n    return datetime.utcnow()\n")
    assert hits[("m", "f", _mod.KIND_UTCNOW)] == 1


def test_flags_naive_now_isoformat():
    hits = _scan("import datetime\ndef f():\n    return datetime.now().isoformat()\n")
    assert hits[("m", "f", _mod.KIND_NAIVE_NOW_ISO)] == 1


def test_does_not_flag_aware_now_isoformat():
    src = (
        "import datetime\nfrom datetime import timezone\n"
        "def f():\n    return datetime.now(timezone.utc).isoformat()\n"
    )
    hits = _scan(src)
    assert all(k[2] != _mod.KIND_NAIVE_NOW_ISO for k in hits)


def test_does_not_flag_naive_now_strftime_display():
    # a display clock (strftime, not isoformat) is not the storage bug
    hits = _scan("import datetime\ndef f():\n    return datetime.now().strftime('%H:%M')\n")
    assert all(k[2] == _mod.KIND_UTCNOW for k in hits) or not hits


def test_symbol_is_the_enclosing_qualname():
    src = (
        "import datetime\n"
        "class C:\n    def m(self):\n        return datetime.now().isoformat()\n"
    )
    hits = _scan(src)
    assert hits[("m", "C.m", _mod.KIND_NAIVE_NOW_ISO)] == 1


def test_repo_census_is_in_sync_with_the_tree():
    # the committed census must cover the current tree (no un-pinned violations)
    assert _mod.main.__module__  # sanity: module loaded
    hits = _mod.scan_tree()
    census = _mod.read_census()
    utcnow = [k for k in hits if k[2] == _mod.KIND_UTCNOW]
    grown = [
        k for k, n in hits.items()
        if k[2] == _mod.KIND_NAIVE_NOW_ISO and n > census.get(k, 0)
    ]
    assert not utcnow, f"un-pinned datetime.utcnow(): {utcnow}"
    assert not grown, f"un-pinned naive now().isoformat(): {grown}"
