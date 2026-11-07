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

PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"
SKIP_PARTS = {".venv", "Third_Party", "__pycache__", "node_modules"}
_BUILTINS = set(dir(builtins))

#: Known-broken modules, each with the reason and the task that closes it.
#: Shrink-only: a module listed here that has become clean fails the companion
#: test below, so the list cannot outlive what it excuses.
_EXEMPT: dict[str, str] = {
    # TASK-38989: Local_Ingestion/API_Endpoint_Sample.py's row went with the
    # file (TASK-32899 deleted it).
}


def _module_files(package: Path = PACKAGE) -> list[Path]:
    return [p for p in sorted(package.rglob("*.py")) if not SKIP_PARTS & set(p.parts)]


def _module_level_bindings(tree: ast.Module) -> set[str]:
    """Names bound at MODULE scope, plus every import anywhere in the file.

    Deliberately NOT ``ast.walk`` over the whole tree. Collecting every nested
    scope's parameters and locals into one module-wide set makes the check
    unsound in the direction that matters: a deleted module-level helper whose
    name happens to appear as a parameter in an unrelated function is subtracted
    from ``missing``, and the guard passes over a real deletion. Verified: a file
    calling ``_deleted_helper(1)`` in one function while another takes
    ``_deleted_helper`` as a parameter reported nothing.

    Imports are collected from anywhere because a function-body import genuinely
    binds the name for that call site, and treating it as unbound would be a
    false positive.
    """
    names = _scope_level_names(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Global):
            names.update(node.names)
    return names


def _scope_level_names(body: list[ast.stmt]) -> set[str]:
    """Names one module or class body binds in its own namespace."""
    names: set[str] = set()

    def walk_module_scope(body):
        """Recurse through module-level control flow, never into def/class bodies.

        A `class _N: ...` inside `try: import x / except ImportError:` is a
        module-level binding even though it is not in ``tree.body``. That
        optional-dependency fallback is the dominant idiom here, and treating it
        as unbound made the check fire on two vendored modules that resolve fine.
        """
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                continue  # their bodies are a different scope
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                names.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(node, field, None)
                if isinstance(inner, list):
                    walk_module_scope(inner)
            for handler in getattr(node, "handlers", []) or []:
                walk_module_scope(handler.body)

    walk_module_scope(body)
    return names


def _local_bindings(scope: ast.AST) -> set[str]:
    """Names bound inside one function scope: parameters, assignments, nested defs."""
    names: set[str] = set()
    args = getattr(scope, "args", None)
    if args is not None:
        for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            names.add(a.arg)
        for a in (args.vararg, args.kwarg):
            if a is not None:
                names.add(a.arg)
    for node in ast.walk(scope):
        if node is scope:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _unresolved_private_calls(tree: ast.Module) -> set[str]:
    """`_helper(...)` targets that nothing in their own scope chain binds."""
    module_names = _module_level_bindings(tree)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    unresolved: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("_")
            and not node.func.id.startswith("__")
        ):
            continue
        name = node.func.id
        if name in module_names or name in _BUILTINS:
            continue
        # only the scopes this call actually sits inside can bind the name
        cursor, bound, innermost = parents.get(node), False, True
        while cursor is not None:
            if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if name in _local_bindings(cursor):
                    bound = True
                    break
                innermost = False
            elif isinstance(cursor, ast.ClassDef):
                # TASK-38989: a call made directly in a class body resolves in
                # that class's namespace (e.g. hook_permissions' class-body
                # `_store_lock = _bind_store_lock(...)`). Methods, lambdas and
                # comprehensions never see class-body names.
                if innermost and name in _scope_level_names(cursor.body):
                    bound = True
                    break
                innermost = False
            elif isinstance(cursor, (ast.Lambda, ast.comprehension)) or isinstance(
                cursor, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
            ):
                innermost = False
            cursor = parents.get(cursor)
        if not bound:
            unresolved.add(name)
    return unresolved


def _guard_report(package: Path) -> tuple[list[str], list[str]]:
    """Return ``(offenders, unparseable)`` for every module under ``package``.

    The per-module form reported an unparseable module as a visible skip; one
    node must name it instead, or the guard would pass over it in silence
    (TASK-39054 review).
    """
    offenders: list[str] = []
    unparseable: list[str] = []
    for path in _module_files(package):
        rel = str(path.relative_to(package.parent))
        if rel in _EXEMPT:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            unparseable.append(f"{rel}: line {exc.lineno}: {exc.msg}")
            continue
        missing = _unresolved_private_calls(tree)
        if missing:
            offenders.append(f"{rel}: {sorted(missing)}")
    return offenders, unparseable


def test_every_private_helper_called_in_a_module_is_defined_in_it() -> None:
    """Every private helper a module calls must be defined or imported in it.

    One node for the whole package (TASK-39054): one node per module (about
    2,940) made the conftest's periodic ``gc.collect()`` cost about 6 minutes
    of the Architecture job for about 30 s of AST work. A module that does not
    parse fails it too, by name, since it cannot be checked.
    """
    offenders, unparseable = _guard_report(PACKAGE)
    assert not offenders and not unparseable, (
        "These modules call private helper(s) they do not define and do not "
        "import:\n  " + "\n  ".join(offenders) + "\nA deletion took the "
        "definition and left the call site — the shape of TASK-32912, which cost "
        "four nights of CI.\nThese modules do not parse, so the guard cannot "
        "check them:\n  " + "\n  ".join(unparseable)
    )


def test_guard_names_unparseable_and_offending_modules(tmp_path: Path) -> None:
    """TASK-39054 review: a scratch package's broken module is named, not skipped."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "broken.py").write_text("def f(:\n", encoding="utf-8")
    (package / "caller.py").write_text(
        "def f():\n    return _gone(1)\n", encoding="utf-8"
    )
    (package / "clean.py").write_text(
        "def _ok():\n    return 1\n\n\nvalue = _ok()\n", encoding="utf-8"
    )
    offenders, unparseable = _guard_report(package)
    assert offenders == ["pkg/caller.py: ['_gone']"]
    assert [line.split(":", 1)[0] for line in unparseable] == ["pkg/broken.py"]


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
        if not _unresolved_private_calls(tree):
            stale.append(f"{rel} (now clean — drop the row): {reason}")
    assert not stale, "Exemptions no longer needed:\n  " + "\n  ".join(stale)


def test_class_body_helper_resolves_only_in_its_own_class_body() -> None:
    """TASK-38989: class scope binds for class-body calls, not for methods."""
    head = (
        "class Store:\n"
        "    def _bind(definition):\n"
        "        return definition\n"
    )
    resolved = ast.parse(head + "    _lock = _bind(1)\n    del _bind\n")
    assert _unresolved_private_calls(resolved) == set()
    for unbound_use in (
        "    def method(self):\n        return _bind(2)\n",
        "    handlers = [_bind(x) for x in (3,)]\n",
        "    handler = lambda: _bind(4)\n",
    ):
        tree = ast.parse(head + unbound_use)
        assert _unresolved_private_calls(tree) == {"_bind"}, unbound_use
    deleted = ast.parse("def f():\n    return _deleted_helper(1)\n")
    assert _unresolved_private_calls(deleted) == {"_deleted_helper"}
