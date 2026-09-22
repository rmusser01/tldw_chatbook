"""Every name a module calls must resolve, without importing the module.

TASK-32912. `0ea2906e99` migrated deepseek and mistral out of
`LLM_API_Calls.py` and took two *Google* helpers with it as collateral:
`_google_tools_payload` and `_google_function_response` kept their call sites
and lost their definitions.

The cost was not the crash. `Tests/Chat/test_google_native_tools.py` imports
`_google_tools_payload` at module scope, so it stopped *collecting* -- and
`nightly-deep.yml` had no `--continue-on-collection-errors`, so one unimportable
module aborted the whole run: `collected 101851 items / 1 error` ->
`Interrupted`, zero tests executed, four consecutive nights on all three
platforms. Meanwhile `test.yml` has no `pull_request` trigger, so nothing
caught it at PR time either.

The test written to prevent exactly this deletion was the thing that blanked
the suite. This check is deliberately import-free (pure AST) so it cannot be
defeated the same way.

Scope: module-level `_private` helpers called within their own defining module.
That is the shape the incident had, it is decidable without imports, and it
does not need a name-resolution model of the whole language.
"""
from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"
SKIP_PARTS = {".venv", "Third_Party", "__pycache__", "node_modules"}
_BUILTINS = set(dir(builtins))

#: Known-broken modules, each with the reason and the task that closes it.
#: Shrink-only: a module listed here that has become clean fails the companion
#: test below, so the list cannot outlive what it excuses.
_EXEMPT: dict[str, str] = {
    "tldw_chatbook/Local_Ingestion/API_Endpoint_Sample.py": (
        "Dead (0 production importers); deleted by TASK-32899. Drop this row "
        "with the file."
    ),
}


def _module_files() -> list[Path]:
    return [p for p in sorted(PACKAGE.rglob("*.py")) if not SKIP_PARTS & set(p.parts)]


def _bound_names(tree: ast.Module) -> set[str]:
    """Every name the module binds at any scope, plus anything it imports."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Global):
            names.update(node.names)
    return names


def _called_private_globals(tree: ast.Module) -> set[str]:
    """`_helper(...)` call targets — bare names only, never attributes."""
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("_")
        and not node.func.id.startswith("__")
    }


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: str(p.name))
def test_every_private_helper_called_in_a_module_is_defined_in_it(path: Path) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:  # a syntax error is a different test's problem
        pytest.skip(f"unparseable: {exc}")

    rel = str(path.relative_to(PACKAGE.parent))
    missing = _called_private_globals(tree) - _bound_names(tree) - _BUILTINS
    if rel in _EXEMPT:
        pytest.skip(f"exempt: {_EXEMPT[rel]}")
    assert not missing, (
        f"{path.relative_to(PACKAGE.parent)} calls private helper(s) it does not "
        f"define and does not import: {sorted(missing)}. A deletion took the "
        f"definition and left the call site — the shape of TASK-32912, which cost "
        f"four nights of CI."
    )


def test_no_exemption_outlives_the_thing_it_excuses() -> None:
    """An exemption that is no longer needed is a lie the next reader believes.

    The repo's own censuses shrink only; so does this list. If a listed module
    was deleted or fixed, the row must go in the same change.
    """
    stale = []
    for rel, reason in _EXEMPT.items():
        path = PACKAGE.parent / rel
        if not path.exists():
            stale.append(f"{rel} (deleted — drop the row): {reason}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not (_called_private_globals(tree) - _bound_names(tree) - _BUILTINS):
            stale.append(f"{rel} (now clean — drop the row): {reason}")
    assert not stale, "Exemptions no longer needed:\n  " + "\n  ".join(stale)
