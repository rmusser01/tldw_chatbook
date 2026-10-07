"""The markup-interpolation guard detects unescaped runtime text (TASK-1513).

``App.notify()``, widget ``tooltip`` strings and ``Button(label=...)`` all
parse Rich markup by default; an interpolated user-derived value carrying a
stray ``[/]`` raises ``MarkupError`` at render time. The guard
(``scripts/check_markup_interpolation.py``) censuses every such site and
fails when a NEW one appears without ``escape_markup(...)`` /
``markup=False``. Every "must count" / "must not count" case below runs the
guard's own scanner over a source snippet; the end-to-end cases run the
whole script against the real tree and its committed census.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "check_markup_interpolation.py"

_SPEC = importlib.util.spec_from_file_location("_cmi", _SCRIPT)
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)  # type: ignore[union-attr]


def _scan(src: str):
    visitor = _mod._Visitor("m")
    visitor.visit(ast.parse(src))
    return visitor.hits


# --------------------------------------------------------------------------
# notify
# --------------------------------------------------------------------------

NOTIFY_HAZARD = "self.app.notify(f\"Imported '{name}'.\", severity=\"information\")\n"
NOTIFY_SAFE_ESCAPE = (
    "self.notify(f\"Imported '{escape_markup(name)}'.\", severity=\"information\")\n"
)


def test_notify_interpolated_name_is_counted():
    hits = _scan(f"def f(name):\n    {NOTIFY_HAZARD}")
    assert hits[("m", "f", _mod.KIND_NOTIFY)] == 1


def test_notify_markup_false_is_exempt():
    src = (
        "def f(name):\n"
        "    self.app.notify(\n"
        "        f\"Creating chatbook '{name}'...\", markup=False\n"
        "    )\n"
    )
    assert not _scan(src)


def test_notify_escape_wrapped_atom_is_exempt():
    assert not _scan(f"def f(name):\n    {NOTIFY_SAFE_ESCAPE}")


def test_notify_literal_only_is_exempt():
    src = "def f():\n    self.notify('[b]Saved.[/b]')\n"
    assert not _scan(src)


def test_notify_percent_and_format_and_concat_shapes_are_counted():
    src = (
        "def f(name, count):\n"
        "    self.notify('Loaded %s' % name)\n"
        "    self.notify('Loaded {}'.format(name))\n"
        "    self.notify('Loaded ' + name)\n"
        "    self.notify(f'{count} items')\n"
    )
    hits = _scan(src)
    assert hits[("m", "f", _mod.KIND_NOTIFY)] == 4


# --------------------------------------------------------------------------
# Button label
# --------------------------------------------------------------------------


def test_button_label_interpolated_name_is_counted():
    src = 'def f(self):\n    yield Button(f\'Review "{self.name}"…\')\n'
    hits = _scan(src)
    assert hits[("m", "f", _mod.KIND_BUTTON_LABEL)] == 1


def test_button_label_escape_markup_wrap_is_exempt():
    src = 'def f(self):\n    yield Button(f\'Review "{escape_markup(self.name)}"…\')\n'
    assert not _scan(src)


def test_button_label_whole_expression_escape_is_exempt():
    src = "def f(name):\n    yield Button(escape_markup(f'Review \"{name}\"'))\n"
    assert not _scan(src)


def test_button_label_text_wrap_is_exempt():
    """``Text(str)`` does not parse markup -- safe by construction."""
    src = 'def f(record):\n    yield Button(Text(f"{record.name} - 3 folders"))\n'
    assert not _scan(src)


def test_button_label_literal_is_exempt():
    src = 'def f():\n    yield Button("Delete", id="confirm")\n'
    assert not _scan(src)


# --------------------------------------------------------------------------
# tooltip
# --------------------------------------------------------------------------


def test_tooltip_keyword_interpolated_name_is_counted():
    src = 'def f(name):\n    yield Button("Use", tooltip=f"Use {name} here.")\n'
    hits = _scan(src)
    assert hits[("m", "f", _mod.KIND_TOOLTIP)] == 1


def test_tooltip_assignment_interpolated_name_is_counted():
    src = 'def f(self, status):\n    self.row.tooltip = f"Remove the {status} job."\n'
    hits = _scan(src)
    assert hits[("m", "f", _mod.KIND_TOOLTIP)] == 1


def test_tooltip_escaped_name_is_exempt():
    src = (
        'def f(name):\n'
        '    yield Button("Use", tooltip=f"Use {escape_markup(name)} here.")\n'
    )
    assert not _scan(src)


def test_tooltip_conditional_of_literals_is_exempt():
    src = (
        "def f(is_open):\n"
        '    tooltip = "Collapse." if is_open else "Expand."\n'
        "    self.node.tooltip = tooltip\n"
    )
    assert not _scan(src)


# --------------------------------------------------------------------------
# end to end against the committed census
# --------------------------------------------------------------------------


def test_guard_is_green_against_the_committed_census():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "check_markup_interpolation: OK" in result.stdout


def test_guard_fails_on_one_new_site(tmp_path, monkeypatch):
    """A guard never observed to fail is not known to work: plant one new
    hazard site in a copy of the tree (census unchanged) and the verify mode
    must exit 1 naming it."""
    tree = tmp_path / "tree"
    tree.mkdir()
    package = tree / "tldw_chatbook"
    package.mkdir()
    (package / "planted.py").write_text(
        "def planted(name):\n"
        "    self.notify(f'Loaded {name}')\n"
        "    self.notify(f'Loaded {name}')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_mod, "PRODUCTION_ROOT", package)
    monkeypatch.setattr(_mod, "REPO_ROOT", tree)
    monkeypatch.setattr(_mod, "CENSUS", tree / "census.tsv")

    hits = _mod.scan_tree()
    assert hits[("tldw_chatbook/planted", "planted", _mod.KIND_NOTIFY)] == 2
