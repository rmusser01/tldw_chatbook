"""TASK-16472: no ``.clear()`` on Textual layout containers, repo-wide.

``Grid``/``Container``/``Vertical``/``Horizontal`` and the scroll containers
have no ``clear()`` method (``remove_children()`` is the idiom), but the call
sites compile fine and only explode when exercised. The TASK-15992 review
found this bug class twice in the selection dialogs and twice more in the
embedding-template selector; this guard extends that review's AST sweep into
a standing test so the class cannot return.

Widgets that DO have ``clear()`` -- ``RichLog``, ``Tree``, ``DataTable``,
``ListView`` -- are deliberately not flagged.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"

# Textual layout containers without a ``clear()`` method.
_CONTAINERS_WITHOUT_CLEAR = frozenset(
    {
        "Grid",
        "Container",
        "Vertical",
        "Horizontal",
        "VerticalScroll",
        "HorizontalScroll",
        "Center",
        "Middle",
    }
)


def _is_query_one_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "query_one"
    )


def _query_one_container_type(call: ast.Call) -> str | None:
    """Return the container type name if the query_one call names one."""
    if not call.args:
        return None
    last = call.args[-1]
    if isinstance(last, ast.Name):
        return last.id if last.id in _CONTAINERS_WITHOUT_CLEAR else None
    if isinstance(last, ast.Attribute):
        return last.attr if last.attr in _CONTAINERS_WITHOUT_CLEAR else None
    return None


def _find_violations(tree: ast.Module, path: Path) -> list[tuple[int, str]]:
    """Flag ``.clear()`` on Textual layout containers, in every repo form.

    Covered (PR #2634 review): plain ``grid = query_one(..., Grid)`` assigns,
    annotated ``grid: Grid = query_one(...)`` assigns, and direct chained
    ``self.query_one(..., Grid).clear()`` calls.
    """
    # Nested scopes (module + function) both walk their descendants, so the
    # same site is discovered once per enclosing scope; key by line so each
    # site reports exactly once.
    violations: dict[int, str] = {}
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            continue
        container_names: dict[str, str] = {}
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and _is_query_one_call(node.value):
                container = _query_one_container_type(node.value)
                if container is None:
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        container_names[target.id] = container
            if isinstance(node, ast.AnnAssign) and _is_query_one_call(node.value):
                container = _query_one_container_type(node.value)
                if container is not None and isinstance(node.target, ast.Name):
                    container_names[node.target.id] = container
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "clear"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in container_names
            ):
                violations[node.lineno] = (
                    f"{path.relative_to(path.parents[1])}: "
                    f"{node.func.value.id}.clear() on a "
                    f"{container_names[node.func.value.id]} -- use "
                    f"remove_children()"
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "clear"
                and _is_query_one_call(node.func.value)
            ):
                container = _query_one_container_type(node.func.value)
                if container is not None:
                    violations[node.lineno] = (
                        f"{path.relative_to(path.parents[1])}: direct "
                        f"query_one(...).clear() on a {container} -- use "
                        f"remove_children()"
                    )
        # Scope-local only: names collected in one scope never leak to the
        # next (walk order guarantees we also visit inner scopes, whose own
        # assignments override nothing -- a same-named reassignment inside an
        # inner scope is collected by that scope's own pass).
    return [(line, message) for line, message in sorted(violations.items())]


def test_no_clear_calls_on_textual_layout_containers() -> None:
    violations: dict[tuple[Path, int], str] = {}
    for source_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"), filename=str(source_path)
        )
        for _line, message in _find_violations(tree, source_path):
            key = (source_path, _line)
            violations.setdefault(key, message)
    assert not violations, "\n".join(sorted(violations.values()))


def test_guard_detects_annotated_assignment_clear() -> None:
    """PR #2634 review proof: ``grid: Grid = query_one(...)`` + ``grid.clear()``
    is flagged, not just plain assigns."""
    tree = ast.parse(
        "def f(self):\n"
        "    from textual.containers import Grid\n"
        "    grid: Grid = self.query_one('#g', Grid)\n"
        "    grid.clear()\n"
    )
    violations = _find_violations(tree, Path("/repo/tldw_chatbook/synthetic.py"))
    assert len(violations) == 1
    assert "grid.clear() on a Grid" in violations[0][1]


def test_guard_detects_direct_chained_clear() -> None:
    """PR #2634 review proof: ``self.query_one(..., Grid).clear()`` is
    flagged even with no intermediate name."""
    tree = ast.parse(
        "def f(self):\n"
        "    from textual.containers import Grid\n"
        "    self.query_one('#g', Grid).clear()\n"
    )
    violations = _find_violations(tree, Path("/repo/tldw_chatbook/synthetic.py"))
    assert len(violations) == 1
    assert "direct query_one(...).clear() on a Grid" in violations[0][1]


def test_guard_ignores_clearable_widget_types() -> None:
    """Widgets that DO have clear() (RichLog/Tree/DataTable) stay unflagged,
    in both chained and named forms."""
    tree = ast.parse(
        "def f(self):\n"
        "    from textual.widgets import RichLog, DataTable\n"
        "    self.query_one('#log', RichLog).clear()\n"
        "    table = self.query_one('#t', DataTable)\n"
        "    table.clear()\n"
    )
    assert _find_violations(tree, Path("/repo/tldw_chatbook/synthetic.py")) == []
