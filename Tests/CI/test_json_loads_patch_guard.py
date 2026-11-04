"""TASK-38685: no test patches ``loads`` on the stdlib ``json`` module.

``monkeypatch.setattr(json, "loads", stub)`` -- or ``<module>.json``, which is
the same stdlib module object -- replaces ``json.loads`` for every thread in
the process while it is installed. A counting stub then counts unrelated
parses: two execution-log tests counted 115 storage-admission parses that
way and were red on dev until their stub was scoped. Patch the module under
test's own name instead::

    monkeypatch.setattr(module, "json", SimpleNamespace(**{**vars(json), "loads": stub}))
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
# Cheap text filter: a "loads" literal (setattr/patch.object), a "...json.loads"
# string target, or a `.loads =` assignment. Only matching files are parsed.
_CANDIDATE = re.compile(r"""["']loads["']|json\.loads["']|\.loads\s*=(?!=)""")
_PATCHERS = {"setattr", "delattr", "patch", "object"}


def _process_wide_loads_patches(source: str) -> list[int]:
    """Return the line of every patch of ``loads`` on the stdlib json module."""
    tree = ast.parse(source)
    names = {"json"} | {
        alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "json" and alias.asname
    }

    def is_json(node: ast.AST) -> bool:
        # `json` (or an alias) and `<anything>.json` are both the stdlib module.
        return (isinstance(node, ast.Name) and node.id in names) or (
            isinstance(node, ast.Attribute) and node.attr == "json"
        )

    lines = []
    for node in ast.walk(tree):
        func = getattr(node, "func", None)
        called = getattr(func, "attr", None) or getattr(func, "id", None)
        if isinstance(node, ast.Call) and called in _PATCHERS and node.args:
            first, rest = node.args[0], node.args[1:]
            target = getattr(first, "value", None)
            # setattr / monkeypatch.setattr / patch.object(json, "loads", ...)
            on_json = (
                is_json(first) and rest and getattr(rest[0], "value", None) == "loads"
            )
            # monkeypatch.setattr("json.loads", ...) / patch("pkg.mod.json.loads")
            by_path = isinstance(target, str) and (
                target == "json.loads" or target.endswith(".json.loads")
            )
            if on_json or by_path:
                lines.append(node.lineno)
        elif isinstance(node, ast.Assign):
            # json.loads = stub
            lines.extend(
                node.lineno
                for target in node.targets
                if isinstance(target, ast.Attribute)
                and target.attr == "loads"
                and is_json(target.value)
            )
    return lines


@pytest.mark.parametrize(
    "planted",
    [
        'monkeypatch.setattr(json, "loads", stub)',
        'monkeypatch.setattr(module.json, "loads", stub)',
        'import json as jsonlib\nmonkeypatch.setattr(jsonlib, "loads", stub)',
        'setattr(json, "loads", stub)',
        'mock.patch.object(json, "loads", stub)',
        'monkeypatch.setattr("json.loads", stub)',
        'mock.patch("tldw_chatbook.MCP.execution_log.json.loads")',
        "json.loads = stub",
    ],
)
def test_the_scan_recognises_each_process_wide_form(planted):
    assert _process_wide_loads_patches(planted)


def test_the_scan_allows_the_module_scoped_form():
    scoped = 'monkeypatch.setattr(module, "json", SimpleNamespace(loads=stub))'
    assert _process_wide_loads_patches(scoped) == []


def test_no_test_patches_loads_on_the_stdlib_json_module():
    offenders = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        if "loads" not in source or not _CANDIDATE.search(source):
            continue
        try:
            lines = _process_wide_loads_patches(source)
        except SyntaxError:
            continue  # a file Python cannot parse cannot install a patch either
        offenders.extend(
            f"{path.relative_to(TESTS_ROOT.parent).as_posix()}:{line}" for line in lines
        )
    assert not offenders, (
        "These patch json.loads for the whole process; patch the module under "
        "test's own `json` name instead (see this file's docstring):\n"
        + "\n".join(offenders)
    )
