"""The timestamp-writer guard detects non-canonical writes (ADR-173).

TASK-32897: this guard shipped matching only *naive* writers, and its docstring
declared ``datetime.now(timezone.utc).isoformat()`` "fine" -- the exact
expression ADR-173 names as a drifting shape to eliminate. The census was
therefore empty and the check printed OK while 117 live writers emitted
``+00:00`` where the ADR mandates ``Z``. ``'+'`` is 0x2B and ``'Z'`` is 0x5A,
so for the same instant the ``+00:00`` row always sorts first and a
``WHERE ts >= '<...>Z'`` cutoff silently drops it.

The test below named ``test_does_not_flag_aware_now_isoformat`` used to ASSERT
that whitelisting; it is inverted here. Every "must fail" case runs the whole
script end to end and asserts a non-zero exit -- a guard never observed to fail
is not known to work.
"""

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


# --------------------------------------------------------------------------
# predicate
# --------------------------------------------------------------------------


def test_flags_datetime_utcnow():
    hits = _scan("import datetime\ndef f():\n    return datetime.utcnow()\n")
    assert hits[("m", "f", _mod.KIND_UTCNOW)] == 1


def test_flags_naive_now_isoformat():
    hits = _scan("import datetime\ndef f():\n    return datetime.now().isoformat()\n")
    assert hits[("m", "f", _mod.KIND_NAIVE_NOW_ISO)] == 1


@pytest.mark.parametrize("tz", ["timezone.utc", "UTC", "tz=timezone.utc"])
def test_flags_aware_now_isoformat_because_it_emits_an_offset(tz):
    """The inversion. ADR-173:14 names this shape; the guard called it fine."""
    src = f"import datetime\ndef f():\n    return datetime.now({tz}).isoformat()\n"
    hits = _scan(src)
    assert hits[("m", "f", _mod.KIND_OFFSET_NOW_ISO)] == 1


def test_does_not_flag_the_canonicalising_replace_idiom():
    """`.replace("+00:00", "Z")` is what the shared helper does; flagging it
    would make the census unshrinkable and the guard unmutable."""
    src = (
        "import datetime\nfrom datetime import timezone\n"
        "def f():\n"
        "    return datetime.now(timezone.utc).isoformat("
        "timespec='milliseconds').replace('+00:00', 'Z')\n"
    )
    assert not _scan(src)


@pytest.mark.parametrize("count", ["0", "1", "count=0"])
def test_a_replace_that_cannot_replace_everything_is_still_flagged(count):
    """`replace("+00:00", "Z", 0)` performs NO replacement and still emits
    `+00:00`; the guard checked only `args[:2]` and handed it the exemption.
    Any explicit count is rejected -- the canonical idiom passes exactly two
    arguments."""
    src = (
        "import datetime\nfrom datetime import timezone\n"
        "def f():\n"
        "    return datetime.now(timezone.utc).isoformat()"
        f".replace('+00:00', 'Z', {count})\n"
    )
    hits = _scan(src)
    assert hits[("m", "f", _mod.KIND_OFFSET_NOW_ISO)] == 1


def test_a_replace_on_a_non_utc_now_is_still_flagged():
    """`.replace("+00:00", "Z")` only fires on a UTC offset. Against
    `now(ZoneInfo("Asia/Kolkata"))` it is a no-op and the value still ships
    `+05:30` -- the exact `offset_now_iso` shape, exempted by an idiom that
    never ran."""
    src = (
        "import datetime\nfrom zoneinfo import ZoneInfo\n"
        "def f():\n"
        "    return datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()"
        ".replace('+00:00', 'Z')\n"
    )
    hits = _scan(src)
    assert hits[("m", "f", _mod.KIND_OFFSET_NOW_ISO)] == 1


@pytest.mark.parametrize("tz", ["timezone.utc", "UTC", "datetime.timezone.utc"])
def test_the_canonicalising_replace_is_exempt_for_every_utc_spelling(tz):
    """...and the narrowing must not break the idiom it exists to permit."""
    src = (
        "import datetime\nfrom datetime import timezone, UTC\n"
        "def f():\n"
        f"    return datetime.now({tz}).isoformat(timespec='milliseconds')"
        ".replace('+00:00', 'Z')\n"
    )
    assert not _scan(src)


def test_a_replace_that_is_not_the_canonicalising_one_is_still_flagged():
    """Only the exact ("+00:00", "Z") pair exempts; `.replace("T", " ")`
    produces a different shape entirely and must not buy an exemption."""
    src = (
        "import datetime\nfrom datetime import timezone\n"
        "def f():\n"
        "    return datetime.now(timezone.utc).isoformat().replace('T', ' ')\n"
    )
    hits = _scan(src)
    assert hits[("m", "f", _mod.KIND_OFFSET_NOW_ISO)] == 1


@pytest.mark.parametrize(
    "fmt",
    ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%SZ", "%Y%m%dT%H%M%SZ"],
)
def test_flags_a_hand_rolled_iso_strftime(fmt):
    src = f"def f(dt):\n    return dt.strftime({fmt!r})[:-3] + 'Z'\n"
    hits = _scan(src)
    assert hits[("m", "f", _mod.KIND_STRFTIME_ISO)] == 1


@pytest.mark.parametrize(
    "fmt",
    ["%H:%M", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%B %d, %Y", "observed %Y-%m-%d UTC"],
)
def test_does_not_flag_human_display_strftime(fmt):
    """Display formats are out of scope; flagging them is how a guard gets
    muted. Note "UTC" contains a literal T -- the predicate is anchored on the
    date fields, not on any bare T."""
    src = f"def f(dt):\n    return dt.strftime({fmt!r})\n"
    assert not _scan(src)


def test_symbol_is_the_enclosing_qualname():
    src = (
        "import datetime\n"
        "class C:\n    def m(self):\n        return datetime.now().isoformat()\n"
    )
    hits = _scan(src)
    assert hits[("m", "C.m", _mod.KIND_NAIVE_NOW_ISO)] == 1


# --------------------------------------------------------------------------
# negative controls: the whole script, bad input, non-zero exit
# --------------------------------------------------------------------------


def _run_main(monkeypatch, tmp_path, source: str, census: str) -> int:
    package = tmp_path / "tldw_chatbook"
    package.mkdir()
    (package / "sample.py").write_text(source, encoding="utf-8")
    census_path = tmp_path / "census.tsv"
    census_path.write_text(census, encoding="utf-8")
    monkeypatch.setattr(_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(_mod, "PRODUCTION_ROOT", package)
    monkeypatch.setattr(_mod, "CENSUS", census_path)
    monkeypatch.setattr(_mod, "_EXCLUDE", set())
    monkeypatch.setattr("sys.argv", ["check_timestamp_writers.py"])
    return _mod.main()


_BAD = {
    "utcnow": "import datetime\ndef f():\n    return datetime.utcnow()\n",
    "naive": "import datetime\ndef f():\n    return datetime.now().isoformat()\n",
    "offset": (
        "import datetime\nfrom datetime import timezone\n"
        "def f():\n    return datetime.now(timezone.utc).isoformat()\n"
    ),
    "strftime": (
        "def f(dt):\n    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'\n"
    ),
}


@pytest.mark.parametrize("kind", sorted(_BAD))
def test_main_exits_nonzero_on_each_forbidden_shape(
    monkeypatch, tmp_path, capsys, kind
):
    assert _run_main(monkeypatch, tmp_path, _BAD[kind], "# empty\n") == 1
    assert "FAIL" in capsys.readouterr().out


def test_utcnow_cannot_be_silenced_by_pinning_it(monkeypatch, tmp_path):
    """`datetime.utcnow()` is forbidden outright, not ratcheted: a census row
    for it must not buy a pass."""
    census = "# header\ntldw_chatbook/sample\tf\tutcnow\t1\n"
    assert _run_main(monkeypatch, tmp_path, _BAD["utcnow"], census) == 1


@pytest.mark.parametrize(
    "kind, row",
    [
        ("naive", "tldw_chatbook/sample\tf\tnaive_now_iso\t1"),
        ("offset", "tldw_chatbook/sample\tf\toffset_now_iso\t1"),
        ("strftime", "tldw_chatbook/sample\tf\tstrftime_iso\t1"),
    ],
)
def test_a_pinned_shape_is_a_baseline_not_a_failure(monkeypatch, tmp_path, kind, row):
    assert _run_main(monkeypatch, tmp_path, _BAD[kind], f"# header\n{row}\n") == 0


def test_a_pinned_shape_that_grows_still_fails(monkeypatch, tmp_path):
    """The ratchet only shrinks: a second occurrence in a pinned function
    fails even though the key is already censused."""
    source = _BAD["offset"] + "    " + _BAD["offset"].splitlines()[-1].strip() + "\n"
    census = "# header\ntldw_chatbook/sample\tf\toffset_now_iso\t1\n"
    assert _run_main(monkeypatch, tmp_path, source, census) == 1


def test_repo_census_is_in_sync_with_the_tree():
    """The committed census must cover the current tree under every kind."""
    hits = _mod.scan_tree()
    census = _mod.read_census()
    utcnow = [k for k in hits if k[2] == _mod.KIND_UTCNOW]
    grown = [
        k
        for k, n in hits.items()
        if k[2] in _mod.RATCHETED_KINDS and n > census.get(k, 0)
    ]
    assert not utcnow, f"un-pinned datetime.utcnow(): {utcnow}"
    assert not grown, f"un-pinned non-canonical timestamp write: {grown}"


def test_flags_a_bare_datetime_utcnow_reference_used_as_a_factory():
    """Tier-2 review S06, P2 [D1]: the guard matched CALLS only.

    `Field(default_factory=datetime.utcnow)` passes the function itself,
    so the node is an `ast.Attribute`, never the `func` of an `ast.Call` --
    and the guard reported "0 datetime.utcnow() site(s) ... OK" while
    `tldw_api/chat_loop_schemas.py:37` carried exactly that. The guard is
    the artifact the repo trusts to know this is clean; it reported clean
    while it was not.
    """
    src = (
        "from datetime import datetime\n"
        "from pydantic import BaseModel, Field\n"
        "class E(BaseModel):\n"
        "    ts: datetime = Field(default_factory=datetime.utcnow)\n"
    )
    hits = _scan(src)
    assert hits[("m", "E", _mod.KIND_UTCNOW)] == 1


def test_a_utcnow_call_is_still_counted_exactly_once():
    """The bare-reference rule must not double-count `datetime.utcnow()`.

    A call's `func` IS an `ast.Attribute`, so the naive widening would see
    every call twice.
    """
    hits = _scan("import datetime\ndef f():\n    return datetime.utcnow()\n")
    assert hits[("m", "f", _mod.KIND_UTCNOW)] == 1
