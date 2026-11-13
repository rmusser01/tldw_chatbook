"""String lookups on the app object only go down (TASK-36942.4).

Architecture review 2026-10-25 F1: code reaches application services with
``getattr(app, "<name>")`` / ``hasattr(app, "<name>")`` instead of an explicit
service, so a rename or removal fails at runtime, not at import. This counts
those literal-name lookups per top-level package (top-level modules count under
``<root>``) and pins the counts in ``app_lookup_census.tsv``. A receiver is
app-like when its last name segment is ``app``, ``app_instance``, ``tldw_app``,
``_app`` or ``_app_instance`` (``app``, ``self.app_instance``, ``screen.app``).

* A count above its pin fails: reach the service explicitly instead.
* A count below its pin fails too: lower the pin in the same change.
* A lookup of an underscore-prefixed app attribute that is not in
  ``app_private_attrs.txt`` fails. That list is frozen; do not add to it.

A literal can also reach the app through a name: a helper's parameter
(``_optional(app, "chachanotes_db")``) or a loop over a literal tuple
(``for name in ("media_db", ...): getattr(app, name)``). Those literals count
as lookups in the pins and the private-name rule, so moving a literal lookup
into one is not a reduction (TASK-40141).

Every lookup whose name is not a literal is also recorded in
``app_nonliteral_lookups.tsv`` as file, name expression, and the attribute it
reaches when the name is an UPPER_CASE constant bound to one string in its
module or the module it is imported from (``-`` otherwise). That record only
shrinks: a new row fails (reach the service explicitly instead), and a recorded
row no longer found fails until it is removed. Private attributes reached only
through a constant are listed there, not in the frozen allowance.
"""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] / "tldw_chatbook"
CENSUS = HERE / "app_lookup_census.tsv"
PRIVATE_ALLOWED = HERE / "app_private_attrs.txt"
NONLITERAL = HERE / "app_nonliteral_lookups.tsv"
_RECEIVERS = {"app", "app_instance", "tldw_app", "_app", "_app_instance"}
_CONSTANT = re.compile(r"_?[A-Z][A-Z0-9_]*")
_BINDINGS = {ast.Assign, ast.AnnAssign, ast.AugAssign}
_SCOPES = {
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.For,
    ast.AsyncFor,
    ast.GeneratorExp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
}


def _receiver(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_str(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _parameter(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> tuple[int | None, str] | None:
    """``(position, name)`` of ``name`` among ``func``'s parameters, else None.

    The position skips a leading ``self``/``cls`` (callers pass it implicitly)
    and is None for a keyword-only parameter.
    """
    positional = [a.arg for a in (*func.args.posonlyargs, *func.args.args)]
    if positional[:1] in (["self"], ["cls"]):
        positional = positional[1:]
    if name in positional:
        return positional.index(name), name
    if name in (a.arg for a in func.args.kwonlyargs):
        return None, name
    return None


def _loop_items(scope: ast.AST, name: str) -> list[ast.expr] | None:
    """The values a ``for`` or comprehension in ``scope`` binds to ``name``.

    None when ``scope`` binds no such loop variable; only the items of a
    literal tuple or list are returned.
    """
    loops = [scope] if isinstance(scope, (ast.For, ast.AsyncFor)) else []
    for loop in loops or getattr(scope, "generators", []):
        target, items = loop.target, getattr(loop.iter, "elts", [])
        if isinstance(target, ast.Name) and target.id == name:
            return items
        names = [_receiver(t) for t in getattr(target, "elts", [])]
        if name in names:
            i = names.index(name)
            return [e.elts[i] for e in items if len(getattr(e, "elts", ())) > i]
    return None


def _scan(
    root: Path = PACKAGE,
) -> tuple[Counter[str], set[str], Counter[tuple[str, str, str]]]:
    counts: Counter[str] = Counter()
    private: set[str] = set()

    def count(pkg: str, node: ast.AST) -> None:
        if _is_str(node):
            counts[pkg] += 1
            if node.value.startswith("_") and not node.value.startswith("__"):
                private.add(node.value)

    hidden: list[tuple[str, str, ast.expr]] = []
    # Per module: every value bound to a name (None when not one string).
    bound: defaultdict[str, defaultdict[str, set[str | None]]] = defaultdict(
        lambda: defaultdict(set)
    )
    imported: defaultdict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    params: defaultdict[str, set[tuple[int | None, str]]] = defaultdict(set)
    calls: defaultdict[str, list[tuple[str, ast.Call]]] = defaultdict(list)
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        pkg = rel.parts[0] if len(rel.parts) > 1 else "<root>"
        package = ("tldw_chatbook", *rel.parts[:-1])
        module = ".".join(package if rel.stem == "__init__" else (*package, rel.stem))
        scopes: list[ast.AST] = []
        named: list[tuple[ast.Call, str]] = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            kind = type(node)
            if kind in _SCOPES:
                scopes.append(node)
            elif kind in _BINDINGS:
                targets = node.targets if kind is ast.Assign else [node.target]
                for target in targets:
                    simple = isinstance(target, (ast.Name, ast.Attribute))
                    one = simple and kind is not ast.AugAssign
                    value = node.value.value if one and _is_str(node.value) else None
                    for name in [target] if simple else ast.walk(target):
                        if key := _receiver(name):
                            bound[module][key].add(value)
            elif kind is ast.ImportFrom:
                source = node.module.split(".") if node.module else []
                if node.level:
                    source = [*package[: len(package) - node.level + 1], *source]
                for alias in node.names:
                    imported[module][alias.asname or alias.name] = (
                        ".".join(source),
                        alias.name,
                    )
            elif kind is ast.Call:
                callee = _receiver(node.func)
                if not (type(node.func) is ast.Name and callee in ("getattr", "hasattr")):
                    if callee and any(
                        map(_is_str, [*node.args, *(k.value for k in node.keywords)])
                    ):
                        calls[callee].append((pkg, node))
                elif len(node.args) > 1 and _receiver(node.args[0]) in _RECEIVERS:
                    arg = node.args[1]
                    if _is_str(arg):
                        count(pkg, arg)
                        continue
                    hidden.append((rel.as_posix(), module, arg))
                    if type(arg) is ast.Name:
                        named.append((node, arg.id))
        # Where a lookup's name variable comes from: the innermost enclosing
        # function's parameter, or a loop over literals.
        for node, name in named:
            at = (node.lineno, node.col_offset)
            enclosing = [
                s
                for s in scopes
                if (s.lineno, s.col_offset) <= at <= (s.end_lineno, s.end_col_offset)
            ]
            enclosing.sort(key=lambda s: (s.lineno, s.col_offset), reverse=True)
            for scope in enclosing:
                if type(scope) is ast.Lambda:
                    break
                if type(scope) in (ast.FunctionDef, ast.AsyncFunctionDef):
                    if spec := _parameter(scope, name):
                        params[scope.name].add(spec)
                    break
                if (items := _loop_items(scope, name)) is not None:
                    for item in items:
                        count(pkg, item)
                    break

    # A literal passed for a helper's name parameter, at any call site.
    for helper, specs in params.items():
        for pkg, call in calls.get(helper, ()):
            for position, name in specs:
                passed = [k.value for k in call.keywords if k.arg == name]
                if position is not None and not any(
                    isinstance(a, ast.Starred) for a in call.args[: position + 1]
                ):
                    passed += call.args[position : position + 1]
                for value in passed:
                    count(pkg, value)

    # ponytail: a name taken from a dict, a module-level tuple, a nested loop or
    # a call's return value is not followed (its row reaches ``-``), so a
    # literal added there is not counted; and helper call sites match by name,
    # so a same-named unrelated call over-counts. Follow those sources, or
    # resolve callees, if either bites.
    def reaches(module: str, arg: ast.expr) -> str:
        name = _receiver(arg)
        if not (name and _CONSTANT.fullmatch(name)):
            return "-"
        values = bound.get(module, {}).get(name)
        if values is None and isinstance(arg, ast.Name):
            source, original = imported.get(module, {}).get(name, ("", ""))
            values = bound.get(source, {}).get(original)
        if not values or len(values) != 1 or None in values:
            return "-"
        return next(iter(values))

    return counts, private, Counter(
        (file, ast.unparse(arg), reaches(module, arg)) for file, module, arg in hidden
    )


def _pins() -> dict[str, int]:
    rows = (l.split("\t") for l in CENSUS.read_text().splitlines() if l.strip())
    return {pkg: int(n) for pkg, n in rows}


def _recorded() -> Counter[tuple[str, str, str]]:
    lines = NONLITERAL.read_text(encoding="utf-8").splitlines()
    return Counter(tuple(l.split("\t")) for l in lines if l.strip())


def _check_nonliteral(found: Counter[tuple[str, str, str]]) -> None:
    recorded = _recorded()
    new = sorted((found - recorded).elements())
    assert not new, (
        "New non-literal getattr/hasattr on the app object: "
        + "; ".join(f"{f}: {expr} (reaches {to})" for f, expr, to in new)
        + ". A name held in a variable or constant hides the lookup from the "
        "literal census and the private-name rule. Reach the service "
        "explicitly instead (a typed attribute such as `app.notes_service`, or "
        f"pass the service in); do NOT add a row to {NONLITERAL.name}. If you "
        "only renamed or moved a recorded lookup, update its row in the same "
        "change. `reaches -` means the name is not an UPPER_CASE constant bound "
        "to one string in its module or the module it is imported from."
    )
    gone = sorted((recorded - found).elements())
    assert not gone, (
        f"{gone} no longer found; remove those rows from {NONLITERAL.name} "
        "(it only shrinks)."
    )


_COUNTS, _PRIVATE, _NONLITERAL = _scan()
_PINS = _pins()


@pytest.mark.unit
@pytest.mark.parametrize("pkg", sorted(set(_COUNTS) | set(_PINS)))
def test_app_string_lookups_match_their_pin(pkg: str) -> None:
    count, pin = _COUNTS.get(pkg, 0), _PINS.get(pkg, 0)
    assert count <= pin, (
        f"{pkg}: {count} literal getattr/hasattr lookups on the app object "
        f"(pinned {pin}, +{count - pin}). Reach the service explicitly; do NOT "
        f"raise the pin ({CENSUS.name})."
    )
    assert count >= pin, (
        f"{pkg}: {count} lookups, pinned {pin}. Lower the pin to {count} in "
        f"{CENSUS.name} so the real measurement is what's pinned."
    )


@pytest.mark.unit
def test_no_new_private_app_attribute_lookups() -> None:
    allowed = set(PRIVATE_ALLOWED.read_text().split())
    new = sorted(_PRIVATE - allowed)
    assert not new, (
        f"New lookup of private app attribute(s) {new}. Private names are "
        f"frozen in {PRIVATE_ALLOWED.name}; expose a public service instead."
    )
    stale = sorted(allowed - _PRIVATE)
    assert not stale, (
        f"{stale} no longer looked up; remove from {PRIVATE_ALLOWED.name}."
    )


@pytest.mark.unit
def test_no_new_nonliteral_app_lookups() -> None:
    _check_nonliteral(_NONLITERAL)


_LITERAL = 'def f(app):\n    return getattr(app, "_planted_private", None)\n'


def _plant(root: Path, files: dict[str, str]):
    for rel, source in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(source, encoding="utf-8")
    return _scan(root)


@pytest.mark.unit
@pytest.mark.parametrize(
    "rewrite",
    [
        '_PLANTED_ATTR = "_planted_private"\n\n'
        "def f(app):\n    return getattr(app, _PLANTED_ATTR, None)\n",
        "def f(app, name):\n    return getattr(app, name, None)\n",
    ],
    ids=["constant", "parameter"],
)
def test_a_literal_rewritten_as_a_name_fails_rather_than_reading_as_a_reduction(
    tmp_path: Path, rewrite: str
) -> None:
    """Negative control (TASK-40141): plant a lookup beside the real package."""
    before = _plant(tmp_path, {"UI/planted.py": _LITERAL})
    counts, private, hidden = _plant(tmp_path, {"UI/planted.py": rewrite})
    # What the literal census alone sees: one fewer lookup, no private name.
    assert (before[0]["UI"], before[1]) == (1, {"_planted_private"})
    assert (counts["UI"], private) == (0, set())
    with pytest.raises(AssertionError, match=r"UI/planted\.py: .*explicitly") as err:
        _check_nonliteral(_NONLITERAL + hidden)
    if "_PLANTED_ATTR" in rewrite:
        assert "(reaches _planted_private)" in str(err.value)


_HELPERS = (
    "def _optional(app, name):\n    return getattr(app, name, None)\n\n"
    "class Screen:\n"
    "    def _owner(self, channel, name):\n"
    "        return getattr(self.app_instance, name, None)\n\n"
    "def services(app):\n"
    '    for name in ("media_db",):\n        getattr(app, name, None)\n'
    '    for key, attribute in (("k", "notes_db"),):\n'
    "        getattr(app, attribute, None)\n"
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "moved",
    [
        'def f(app):\n    return _optional(app, "_planted_private")\n',
        'def f(app):\n    return _optional(app, name="_planted_private")\n',
        'def f(screen):\n    return screen._owner(None, "_planted_private")\n',
        _HELPERS.replace('("media_db",)', '("media_db", "_planted_private")'),
        _HELPERS.replace('(("k", "notes_db"),)', '(("k", "notes_db"), ("p", "_planted_private"))'),
    ],
    ids=["helper", "helper-keyword", "method", "loop", "unpacked-loop"],
)
def test_a_literal_moved_into_a_helper_or_loop_is_still_counted(
    tmp_path: Path, moved: str
) -> None:
    """A literal passed to an existing helper or loop is not a reduction (I1)."""
    before = _plant(tmp_path, {"UI/helpers.py": _HELPERS, "UI/planted.py": _LITERAL})
    planted = {"UI/planted.py": moved}
    if moved.startswith("def _optional"):
        planted = {"UI/helpers.py": moved, "UI/planted.py": ""}
    after = _plant(tmp_path, planted)
    assert "_planted_private" in after[1]
    assert after == before  # same count, same private names, same record rows


@pytest.mark.unit
def test_a_constant_resolves_through_its_module_and_imports_only(tmp_path: Path) -> None:
    """A rebinding makes it ambiguous; an unrelated same-named one does not (I2, M1)."""
    use = "from .names import ATTR\n\ndef f(app):\n    return getattr(app, ATTR)\n"
    files = {
        "UI/names.py": 'ATTR = "_planted_private"\n',
        "UI/user.py": use,
        "Chat/other.py": 'ATTR = "unrelated"\n',
    }
    hidden = _plant(tmp_path, files)[2]
    assert hidden == Counter({("UI/user.py", "ATTR", "_planted_private"): 1})
    rebound = 'ATTR = "_planted_private"\nATTR: str = "_other_private"\n'
    hidden = _plant(tmp_path, {"UI/names.py": rebound})[2]
    assert hidden == Counter({("UI/user.py", "ATTR", "-"): 1})
