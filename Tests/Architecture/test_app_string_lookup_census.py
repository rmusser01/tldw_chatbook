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
(``_optional(app, "chachanotes_db")``), a loop over a literal tuple
(``for name in ("media_db", ...): getattr(app, name)``), and since TASK-41318 a
plain ``name = ...`` in the scope that binds the name (the function, an
enclosing function or lambda, or the module), an enclosing loop's item (nested
loops and comprehensions included), a dict lookup (``D[key]``,
``D.get(key, default)``) or a same-module function's return value, composed in
any order. Every ``=`` to the name in that scope counts, whatever its order or
branch, unless an enclosing loop binds the same name (the loop wins). Those
literals count as lookups in the pins and the private-name rule, so moving a
literal lookup into one is not a reduction (TASK-40141). A literal counts once
per lookup that can read it, so a dict read at eight sites counts its values
eight times.

Not followed, because TASK-41318 did not name them (follow one if a literal
moves there): conditional and boolean expressions (``"a" if x else "b"``,
``x or "y"``), parameter defaults, a parameter reassigned in its function, a
function's returned parameter (the caller's argument), iterating a dict or its
``.items()``, tuple subscripts, ``*``/``**`` unpacking inside a literal, a
comprehension used as a loop's iterable, a loop variable read after its loop,
names imported from another module, class attributes, functions defined in
another module, a ``global``/``nonlocal`` rebinding from another function, a
function's ``=`` to a name an enclosing loop also binds, and bindings other
than a plain ``=`` (tuple unpacking, ``with``/``except``/``import`` targets,
``match`` captures, walrus, augmented). A name bound only those ways still
shadows a module-level one. A lookup whose own argument is an UPPER_CASE
constant is not followed either: its row in the record names the attribute it
reaches. A constant reached any other way (``x = CONST``, a dict holding one)
is followed.

Every lookup whose name is not a literal, and every ``setattr``/``delattr`` on
the app whose name is not a literal (TASK-41318), is also recorded in
``app_nonliteral_lookups.tsv`` as file, name expression, and the attribute it
reaches when the name is an UPPER_CASE constant bound to one string in its
module or the module it is imported from (``-`` otherwise). That record only
shrinks: a new row fails (reach the service explicitly instead), and a recorded
row no longer found fails until it is removed. Private attributes reached only
through a constant are listed there, not in the frozen allowance. A write is
recorded but not counted: a literal ``setattr(app, "x", v)`` is the assignment
``app.x = v``, not a string lookup.
"""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
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
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_LOOKUPS = ("getattr", "hasattr")
_WRITES = ("setattr", "delattr")
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


def _own(scope: ast.AST) -> Iterator[ast.AST]:
    """The nodes in ``scope``, not descending into a nested function, lambda or class."""
    todo = list(ast.iter_child_nodes(scope))
    while todo:
        node = todo.pop()
        yield node
        if not isinstance(node, (*_FUNCTIONS, ast.Lambda, ast.ClassDef)):
            todo.extend(ast.iter_child_nodes(node))


def _bound_names(scope: ast.AST) -> set[str]:
    """Every name ``scope`` itself binds, by any statement.

    A comprehension's loop variable belongs to the comprehension, not ``scope``.
    """
    nodes = list(_own(scope))
    loop_variables = {
        id(n) for c in nodes if type(c) is ast.comprehension for n in ast.walk(c.target)
    }
    names = set()
    for n in nodes:
        if type(n) is ast.Name and type(n.ctx) is ast.Store and id(n) not in loop_variables:
            names.add(n.id)
        elif isinstance(n, (*_FUNCTIONS, ast.ClassDef, ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            names.add(n.name)
        elif type(n) is ast.MatchMapping:
            names.add(n.rest)
        elif type(n) is ast.alias:
            names.add(n.asname or n.name.partition(".")[0])
    return names


def _assigned(scope: ast.AST, name: str) -> list[ast.expr]:
    """The values ``scope`` itself assigns to ``name`` with ``=``."""
    return [
        node.value
        for node in _own(scope)
        if type(node) in (ast.Assign, ast.AnnAssign)
        and node.value is not None
        and any(
            type(t) is ast.Name and t.id == name
            for t in (node.targets if type(node) is ast.Assign else [node.target])
        )
    ]


def _unpacked(target: ast.expr, name: str) -> int | None:
    """Where a loop ``target`` binds ``name``.

    -1 for the whole item, the index in a flat tuple target, -2 when it is
    bound some other way (nested or starred), and None when it is not bound.
    """
    if type(target) is ast.Name:
        return -1 if target.id == name else None
    stored = (n.id for n in ast.walk(target) if type(n) is ast.Name and type(n.ctx) is ast.Store)
    if name not in stored:
        return None
    elts = getattr(target, "elts", [])
    flat = [e.id if type(e) is ast.Name else None for e in elts]
    if name in flat and not any(type(e) is ast.Starred for e in elts):
        return flat.index(name)
    return -2


class _Follower:
    """Follows a name in one module to the expressions it can hold (TASK-41318)."""

    def __init__(
        self,
        tree: ast.Module,
        scopes: list[ast.AST],
        params: defaultdict[str, set[tuple[int | None, str]]],
    ) -> None:
        self.tree, self.scopes, self.params = tree, scopes, params
        self.defs: defaultdict[str, list[ast.AST]] = defaultdict(list)
        for scope in scopes:
            if isinstance(scope, _FUNCTIONS):
                self.defs[scope.name].append(scope)

    def _chain(self, node: ast.expr) -> list[ast.AST]:
        """The scopes that can bind a name at ``node``, innermost first.

        A loop binds in its body, not its iterable; a comprehension's loop
        binds in its element and in the loops after it.
        """
        at = (node.lineno, node.col_offset)

        def holds(n: ast.AST) -> bool:
            return (n.lineno, n.col_offset) <= at <= (n.end_lineno, n.end_col_offset)

        chain: list[ast.AST] = []
        for scope in sorted(
            filter(holds, self.scopes), key=lambda s: (s.lineno, s.col_offset), reverse=True
        ):
            loops = getattr(scope, "generators", None)
            if loops:
                first = loops[0].target
                element = at < (first.lineno, first.col_offset)
                chain += [
                    loop
                    for loop in reversed(loops)
                    if element or at >= (loop.iter.end_lineno, loop.iter.end_col_offset)
                ]
            elif not (isinstance(scope, (ast.For, ast.AsyncFor)) and holds(scope.iter)):
                chain.append(scope)
        return chain

    def _bindings(self, node: ast.Name, seen: frozenset[int], helper: bool) -> list[ast.expr]:
        """The expressions bound to ``node``'s name where it is read.

        A parameter found while ``helper`` makes its function a lookup helper
        (its call sites' literals count); a returned parameter does not.
        """
        name = node.id
        for scope in self._chain(node):
            if isinstance(scope, (*_FUNCTIONS, ast.Lambda)):
                args = scope.args
                if spec := _parameter(scope, name):
                    if helper and not isinstance(scope, ast.Lambda):
                        self.params[scope.name].add(spec)
                    return []
                if name in (a.arg for a in (args.vararg, args.kwarg) if a):
                    return []
                if name in _bound_names(scope):
                    return _assigned(scope, name)
            elif (index := _unpacked(scope.target, name)) is not None:
                if index == -2:
                    return []
                iterated = self.values(scope.iter, seen, helper)
                items = [e for i in iterated for e in getattr(i, "elts", ())]
                if index == -1:
                    return items
                return [
                    v.elts[index]
                    for item in items
                    for v in self.values(item, seen, helper)
                    if len(getattr(v, "elts", ())) > index
                ]
        return _assigned(self.tree, name)

    def values(
        self, expr: ast.expr, seen: frozenset[int] = frozenset(), helper: bool = True
    ) -> list[ast.expr]:
        """What ``expr`` can evaluate to, through names, dicts and returns."""
        if id(expr) in seen:
            return []
        seen |= {id(expr)}

        def follow(nodes: list[ast.expr], helper: bool = helper) -> list[ast.expr]:
            return [v for n in nodes for v in self.values(n, seen, helper)]

        def dict_values(mapping: ast.expr) -> list[ast.expr]:
            found = self.values(mapping, seen, helper)
            return [v for d in found if type(d) is ast.Dict for v in d.values]

        if type(expr) is ast.Name:
            return follow(self._bindings(expr, seen, helper))
        if type(expr) is ast.Subscript:
            return follow(dict_values(expr.value))
        if type(expr) is ast.Call:
            func = expr.func
            if type(func) is ast.Attribute and func.attr == "get":
                return follow([*dict_values(func.value), *expr.args[1:2]])
            return follow(
                [
                    node.value
                    for f in self.defs.get(_receiver(func), ())
                    for node in _own(f)
                    if type(node) is ast.Return and node.value is not None
                ],
                helper=False,
            )
        return [expr]


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
        followed: list[ast.expr] = []
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
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
                if not (type(node.func) is ast.Name and callee in (*_LOOKUPS, *_WRITES)):
                    if callee and any(
                        map(_is_str, [*node.args, *(k.value for k in node.keywords)])
                    ):
                        calls[callee].append((pkg, node))
                elif len(node.args) > 1 and _receiver(node.args[0]) in _RECEIVERS:
                    arg = node.args[1]
                    if _is_str(arg):
                        if callee in _LOOKUPS:
                            count(pkg, arg)
                        continue
                    hidden.append((rel.as_posix(), module, arg))
                    # A constant's row names what it reaches; a write is
                    # recorded, not counted (a literal one is an assignment).
                    constant = type(arg) is ast.Name and _CONSTANT.fullmatch(arg.id)
                    if callee in _LOOKUPS and not constant:
                        followed.append(arg)
        # The literals a lookup's name can hold, each once: through a helper's
        # parameter (counted at its call sites below), a loop, an assignment, a
        # dict lookup or a same-module function's return value.
        follower = _Follower(tree, scopes, params)
        for arg in followed:
            for value in {id(v): v for v in follower.values(arg)}.values():
                count(pkg, value)

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

    # ponytail: helper call sites (package-wide) and function returns (same
    # module) match callees by name, so a same-named unrelated call over-counts;
    # the sources the docstring lists are not followed. Resolve callees, or
    # follow a source, if either bites.
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
        "New non-literal getattr/hasattr/setattr/delattr on the app object: "
        + "; ".join(f"{f}: {expr} (reaches {to})" for f, expr, to in new)
        + ". A name held in a variable or constant hides the attribute from the "
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
        f"(pinned {pin}, +{count - pin}). A literal counts once per lookup site "
        "that can read it, so one value added to a dict, tuple or return that N "
        "lookups read by name adds N. Reach the service explicitly (for a new "
        "value, move the lookups that read it behind a typed attribute or "
        "explicit accessor, not a by-name helper); do "
        f"NOT raise the pin ({CENSUS.name}). Callees are matched by name (helper "
        "call sites package-wide, function returns within the module): if the "
        "increase is an unrelated call sharing such a name, rename that call "
        "instead."
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


@pytest.mark.unit
@pytest.mark.parametrize("write", ["setattr(app, name, None)", "delattr(app, name)"])
def test_a_nonliteral_app_write_is_recorded(tmp_path: Path, write: str) -> None:
    """Negative control (TASK-41318): a name the app is written through is recorded."""
    hidden = _plant(tmp_path, {"UI/planted.py": f"def f(app, name):\n    {write}\n"})[2]
    with pytest.raises(AssertionError, match=r"UI/planted\.py: name \(reaches -\)"):
        _check_nonliteral(_NONLITERAL + hidden)


_HELPERS = (
    "def _optional(app, name):\n    return getattr(app, name, None)\n\n"
    "class Screen:\n"
    "    def _owner(self, channel, name):\n"
    "        return getattr(self.app_instance, name, None)\n\n"
    "def services(app):\n"
    '    for name in ("media_db",):\n        getattr(app, name, None)\n'
    '    for key, attribute in (("k", "notes_db"),):\n'
    "        getattr(app, attribute, None)\n\n"
    # The name sources TASK-41318 follows, each feeding one lookup.
    '_BY_KIND = {"media": "media_db"}\n'
    '_PAIRS: tuple[tuple[str, str], ...] = (("notes_db", "Notes"),)\n\n'
    '_FALLBACK = {"k": "media_db"}\n\n'
    "def _default(kind):\n"
    '    return "media_db"\n\n'
    "def _first(kind):\n"
    "    if kind:\n        return _second(kind)\n"
    '    return "notes_db"\n\n'
    "def _second(kind):\n"
    "    return _first(kind)\n\n"
    "def sources(app, kind):\n"
    "    attr = _BY_KIND.get(kind)\n"
    "    getattr(app, attr, None)\n"
    '    owners = {"notes": "notes_db"}\n'
    "    getattr(app, owners[kind], None)\n"
    "    rows = tuple(getattr(app, attribute, None) for attribute, label in _PAIRS)\n"
    '    for declarations in ((("media_db", "m"),),):\n'
    "        [getattr(app, attribute, None) for attribute, module in declarations]\n"
    "    returned = _default(kind)\n"
    "    getattr(app, returned, None)\n"
    "    fallback = _FALLBACK.get(kind)\n"
    "    getattr(app, fallback, None)\n"
    "    getattr(app, _first(kind), None)\n"
    '    closed = "notes_db"\n'
    "    pick = lambda: getattr(app, closed, None)\n"
    '    enclosed = "media_db"\n\n'
    "    def inner():\n"
    "        return getattr(app, enclosed, None)\n"
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
        _HELPERS.replace('"media": "media_db"}', '"media": "media_db", "p": "_planted_private"}'),
        _HELPERS.replace('"notes": "notes_db"}', '"notes": "notes_db", "p": "_planted_private"}'),
        _HELPERS.replace(
            '("notes_db", "Notes"),)', '("notes_db", "Notes"), ("_planted_private", "P"))'
        ),
        _HELPERS.replace(
            '(("media_db", "m"),),)', '(("media_db", "m"),), (("_planted_private", "p"),))'
        ),
        _HELPERS.replace(
            '    return "media_db"\n',
            '    if kind:\n        return "_planted_private"\n    return "media_db"\n',
        ),
        _HELPERS.replace("_FALLBACK.get(kind)", '_FALLBACK.get(kind, "_planted_private")'),
        _HELPERS.replace(
            "    return _first(kind)\n",
            '    if not kind:\n        return "_planted_private"\n    return _first(kind)\n',
        ),
        _HELPERS.replace(
            '    closed = "notes_db"\n', '    closed = "notes_db"\n    closed = "_planted_private"\n'
        ),
        _HELPERS.replace(
            '    enclosed = "media_db"\n', '    enclosed = "media_db"\n    enclosed = "_planted_private"\n'
        ),
    ],
    ids=[
        "helper",
        "helper-keyword",
        "method",
        "loop",
        "unpacked-loop",
        "dict-get",
        "dict-subscript",
        "module-tuple",
        "nested-loop",
        "call-result",
        "dict-get-default",
        "call-cycle",
        "lambda-closure",
        "nested-def-closure",
    ],
)
def test_a_literal_moved_into_a_helper_or_loop_is_still_counted(
    tmp_path: Path, moved: str
) -> None:
    """A literal moved into a helper, loop, dict, tuple or return is not a reduction (I1)."""
    before = _plant(tmp_path, {"UI/helpers.py": _HELPERS, "UI/planted.py": _LITERAL})
    planted = {"UI/planted.py": moved}
    if moved.startswith("def _optional"):
        planted = {"UI/helpers.py": moved, "UI/planted.py": ""}
    after = _plant(tmp_path, planted)
    assert "_planted_private" in after[1]
    assert after == before  # same count, same private names, same record rows


_SHADOWED = 'attr = "_planted_private"\n\n'


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    [
        'def other():\n    attr = "_planted_private"\n\ndef f(app):\n    getattr(app, attr, None)\n',
        'def f(app):\n    [x for x in ("_planted_private",)]\n    getattr(app, x, None)\n',
        'def f(app):\n    x = "media_db"\n'
        '    [y for y in (getattr(app, x, None),) for x in ("_planted_private",)]\n',
        _SHADOWED + "def f(app, items):\n    for attr in items:\n        pass\n"
        "    getattr(app, attr, None)\n",
        _SHADOWED + "def f(app, cm):\n    with cm as attr:\n        getattr(app, attr, None)\n",
        _SHADOWED + "def f(app, pair):\n    attr, other = pair\n    getattr(app, attr, None)\n",
        _SHADOWED + "def f(app):\n    try:\n        pass\n    except Exception as attr:\n"
        "        getattr(app, attr, None)\n",
        _SHADOWED + "def f(app, v):\n    match v:\n        case str() as attr:\n"
        "            getattr(app, attr, None)\n",
        _SHADOWED + "def f(app, v):\n    match v:\n        case {**attr}:\n"
        "            getattr(app, attr, None)\n",
        _SHADOWED + "def f(app, svc):\n    attr = svc.name\n\n"
        "    def inner():\n        getattr(app, attr, None)\n",
        "def ident(name):\n    return name\n\ndef f(app, n):\n    getattr(app, ident(n), None)\n\n"
        'def g():\n    ident("_planted_private")\n',
    ],
    ids=[
        "other-function",
        "comprehension-variable",
        "comprehension-own-iterable",
        "loop-variable-after-loop",
        "with-target",
        "tuple-unpacking",
        "except-target",
        "match-capture",
        "match-mapping-rest",
        "enclosing-function",
        "returned-parameter",
    ],
)
def test_a_literal_the_name_cannot_hold_is_not_counted(tmp_path: Path, source: str) -> None:
    """Following stays inside the name's scope (TASK-41318 review).

    A function that binds the name in any way shadows the module's binding, a
    comprehension's loop variable is its own, and a returned parameter does
    not make its function a lookup helper.
    """
    assert "_planted_private" not in _plant(tmp_path, {"UI/scoped.py": source})[1]


@pytest.mark.unit
def test_a_comprehension_variable_does_not_shadow_the_module(tmp_path: Path) -> None:
    """A comprehension's loop variable is its own, so the module's binding still counts."""
    source = _SHADOWED + "def f(app, xs):\n    [attr for attr in xs]\n    getattr(app, attr, None)\n"
    assert "_planted_private" in _plant(tmp_path, {"UI/scoped.py": source})[1]


@pytest.mark.unit
def test_a_literal_reached_twice_by_one_lookup_counts_once(tmp_path: Path) -> None:
    """Two paths to one literal are one lookup of it (TASK-41318 review)."""
    source = (
        'def _named():\n    return "media_db"\n\n'
        "def f(app, k):\n    if k:\n        attr = _named()\n    else:\n        attr = _named()\n"
        "    getattr(app, attr, None)\n"
    )
    assert _plant(tmp_path, {"UI/twice.py": source})[0]["UI"] == 1


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
