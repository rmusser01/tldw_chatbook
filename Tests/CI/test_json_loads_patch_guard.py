"""TASK-38685: no test patches ``loads`` on the stdlib ``json`` module.

``monkeypatch.setattr(json, "loads", stub)`` -- or ``<module>.json``, which is
the same stdlib module object -- replaces ``json.loads`` for every thread in
the process while it is installed. A counting stub then counts unrelated
parses: two execution-log tests counted 115 storage-admission parses that
way and were red on dev until their stub was scoped. Patch the module under
test's own name instead::

    monkeypatch.setattr(module, "json", SimpleNamespace(**{**vars(json), "loads": stub}))

The scan reads syntax, so it cannot see a patch that reaches the module or the
attribute name only at run time: rebinding (``j = json`` then
``setattr(j, "loads", ...)``), ``sys.modules["json"]``,
``importlib.import_module("json")``, ``setitem`` on ``json.__dict__`` or
``vars(json)``, or an attribute name held in a variable
(``setattr(json, attr, ...)``).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
# Cheap text filter: a "loads" literal (setattr/patch.object, name=/attribute=),
# a "...json.loads" string target, or `loads=` (patch.multiple) / `.loads =`.
# Only matching files are parsed.
_CANDIDATE = re.compile(r"""["']loads["']|json\.loads["']|\bloads\s*=(?!=)""")
_PATCHERS = {"setattr", "delattr", "patch", "object", "multiple"}


def _process_wide_loads_patches(source: str) -> list[int]:
    """Return the line of every patch of ``loads`` on the stdlib json module."""
    tree = ast.parse(source)
    # `import json as j` and `from pkg import json [as j]` bind the stdlib module.
    names = {"json"} | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        if alias.name == "json"
    }

    def is_json(node: ast.AST | None) -> bool:
        # `json` (or an alias) and `<anything>.json` are both the stdlib module.
        return (isinstance(node, ast.Name) and node.id in names) or (
            isinstance(node, ast.Attribute) and node.attr == "json"
        )

    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            called = getattr(func, "attr", None) or getattr(func, "id", None)
            if called not in _PATCHERS:
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            first = node.args[0] if node.args else keywords.get("target")
            second = (
                node.args[1]
                if len(node.args) > 1
                else keywords.get("name", keywords.get("attribute"))
            )
            text = getattr(first, "value", None)
            if called == "multiple":
                # patch.multiple(json | "json" | "pkg.mod.json", loads=stub)
                hit = "loads" in keywords and (
                    is_json(first)
                    or (
                        isinstance(text, str)
                        and (text == "json" or text.endswith(".json"))
                    )
                )
            else:
                # setattr / monkeypatch.setattr / patch.object(json, "loads", ...),
                # positional or target=/name=/attribute=; or a "json.loads" /
                # "pkg.mod.json.loads" string target.
                hit = (
                    is_json(first) and getattr(second, "value", None) == "loads"
                ) or (
                    isinstance(text, str)
                    and (text == "json.loads" or text.endswith(".json.loads"))
                )
            if hit:
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
        # Review of TASK-38685 (2026-11-04): forms the first scan missed.
        (
            "from tldw_chatbook.MCP.execution_log import json as el_json\n"
            'monkeypatch.setattr(el_json, "loads", stub)'
        ),
        'monkeypatch.setattr(target=json, name="loads", value=stub)',
        'monkeypatch.setattr(json, name="loads", value=stub)',
        'mock.patch.object(target=json, attribute="loads", new=stub)',
        'monkeypatch.setattr(target="json.loads", value=stub)',
        'mock.patch(target="pkg.mod.json.loads")',
        "mock.patch.multiple(json, loads=stub)",
        'mock.patch.multiple("json", loads=stub)',
        'mock.patch.multiple("pkg.mod.json", loads=stub)',
    ],
)
def test_the_scan_recognises_each_process_wide_form(planted):
    assert _CANDIDATE.search(planted), "the text pre-filter skips this file"
    assert _process_wide_loads_patches(planted)


@pytest.mark.parametrize(
    "scoped",
    [
        'monkeypatch.setattr(module, "json", SimpleNamespace(loads=stub))',
        "mock.patch.multiple(module, json=fake_json)",
        'mock.patch.multiple("pkg.mod", loads=stub)',
    ],
)
def test_the_scan_allows_the_module_scoped_form(scoped):
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
