# test_reactive_mutable_default_inventory.py
# Description: Guard against non-callable mutable reactive defaults (task-15771)
"""
task-15771: Textual's ``Reactive._initialize_reactive`` installs
``default_or_callable() if callable(...) else default_or_callable`` — a bare
``[]``/``{}``/``set()`` result default is therefore installed as the *same
object* on every instance of the widget class that never explicitly
reassigns it, and reassigning an empty-equal value is a no-op
(``Reactive._set`` only stores when ``current_value != value``). Any in-place
mutation then leaks across instances and screen remounts.

The idiom is a callable default: ``reactive(list)`` / ``reactive(dict)`` /
``reactive(set)`` (or ``reactive(lambda: [seed, ...])`` for a non-empty
default). This inventory sweep asserts the package contains ZERO
``reactive(...)``/``var(...)`` declarations — including the
subscripted-generic spelling ``reactive[list[dict[str, Any]]](...)`` — whose
default is:

* a list/dict/set literal (empty or not) or a comprehension,
* the *result* of a ``list()``/``dict()``/``set()`` call (not callable), or
* a name bound to a module-level mutable literal (one shared object).

NOT covered: a shared mutable *instance* default (``reactive(SomeClass())``)
is the same one-object-per-class bug and this detector does not see it —
5 known occurrences at the time of writing, tracked as a follow-up; do not
read a green run as clearance for that form.

Born red twice: against the pre-fix tree (27 sites), and again after the
task-15771 review taught ``_call_name`` the ``ast.Subscript`` form the first
sweep was structurally blind to (14 more sites, all in
``UI/Watchlists_Modules/`` — 41 total). If it fails, replace the default
with a callable, never with a "reassign before use" workaround.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"

REACTIVE_NAMES = {"reactive", "var", "Reactive"}
MUTABLE_LITERALS = (
    ast.List,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
)
MUTABLE_FACTORY_NAMES = {"list", "dict", "set"}


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Subscript):
        # The subscripted-generic form: reactive[list[dict[str, Any]]]([]) /
        # Reactive[list]([]). Same descriptor, same shared-default bug — the
        # review of the first sweep found 14 sites written this way that a
        # Name/Attribute-only match was structurally blind to.
        func = func.value
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _default_arg(node: ast.Call) -> ast.expr | None:
    if node.args:
        return node.args[0]
    for keyword in node.keywords:
        if keyword.arg == "default":
            return keyword.value
    return None


def _module_level_mutables(tree: ast.Module) -> set[str]:
    """Names assigned a mutable literal at module level (one shared object)."""
    names: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, MUTABLE_LITERALS):
            names.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
        elif (
            isinstance(stmt, ast.AnnAssign)
            and stmt.value is not None
            and isinstance(stmt.value, MUTABLE_LITERALS)
            and isinstance(stmt.target, ast.Name)
        ):
            names.add(stmt.target.id)
    return names


def _violations_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_mutables = _module_level_mutables(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in REACTIVE_NAMES:
            continue
        default = _default_arg(node)
        if default is None:
            continue
        reason: str | None = None
        if isinstance(default, MUTABLE_LITERALS):
            reason = f"mutable literal default {ast.unparse(default)!r}"
        elif isinstance(default, ast.Name) and default.id in module_mutables:
            reason = f"module-level shared mutable {default.id!r}"
        elif (
            isinstance(default, ast.Call)
            and _call_name(default) in MUTABLE_FACTORY_NAMES
        ):
            reason = (
                f"non-callable call result {ast.unparse(default)!r}"
                " (pass the factory itself, without parentheses)"
            )
        if reason is not None:
            relative = path.relative_to(PACKAGE_ROOT.parent)
            violations.append(f"{relative}:{node.lineno}: {reason}")
    return violations


def test_no_non_callable_mutable_reactive_defaults() -> None:
    """Every reactive()/var() default in the package must be callable-or-immutable."""
    assert PACKAGE_ROOT.is_dir(), f"package root not found: {PACKAGE_ROOT}"
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        violations.extend(_violations_in_file(path))
    assert not violations, (
        "reactive()/var() declarations with a shared (non-callable) mutable "
        "default — use reactive(list)/reactive(dict)/reactive(set) or a "
        "lambda instead:\n" + "\n".join(violations)
    )
