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

A lookup whose name is not a literal (``getattr(app, name)``,
``getattr(app, SOME_ATTR)``) escapes both rules, so a literal rewritten that way
reads as a reduction (TASK-40141). Every such lookup is recorded in
``app_nonliteral_lookups.tsv`` as file, name expression, and the attribute it
reaches when the name is an UPPER_CASE constant (``-`` otherwise). That record
only shrinks: a new row fails (reach the service explicitly instead), and a
recorded row no longer found fails until it is removed. Private attributes
reached only through a constant are listed there, not in the frozen allowance.
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


def _receiver(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _scan(
    root: Path = PACKAGE,
) -> tuple[Counter[str], set[str], Counter[tuple[str, str, str]]]:
    counts: Counter[str] = Counter()
    private: set[str] = set()
    hidden: list[tuple[str, ast.expr]] = []
    constants: defaultdict[str, set[str]] = defaultdict(set)
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        pkg = rel.parts[0] if len(rel.parts) > 1 else "<root>"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Module, ast.ClassDef)):
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.Assign)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)
                    ):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                constants[target.id].add(stmt.value.value)
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr")
                and len(node.args) > 1
                and _receiver(node.args[0]) in _RECEIVERS
            ):
                continue
            arg = node.args[1]
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                hidden.append((rel.as_posix(), arg))
                continue
            counts[pkg] += 1
            name = arg.value
            if name.startswith("_") and not name.startswith("__"):
                private.add(name)

    # ponytail: only an UPPER_CASE name (or ``self.NAME``) with one string value
    # package-wide resolves; a literal passed into an existing helper's
    # parameter stays ``-``. Follow call sites if that hides a private name.
    def reaches(arg: ast.expr) -> str:
        name = _receiver(arg)
        values = constants.get(name, ()) if name and _CONSTANT.fullmatch(name) else ()
        return next(iter(values)) if len(values) == 1 else "-"

    return counts, private, Counter(
        (file, ast.unparse(arg), reaches(arg)) for file, arg in hidden
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
        f"pass the service in); do NOT add a row to {NONLITERAL.name}."
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
    planted = tmp_path / "UI" / "planted.py"
    planted.parent.mkdir()
    planted.write_text(_LITERAL, encoding="utf-8")
    before = _scan(tmp_path)
    planted.write_text(rewrite, encoding="utf-8")
    counts, private, hidden = _scan(tmp_path)
    # What the literal census alone sees: one fewer lookup, no private name.
    assert (before[0]["UI"], before[1]) == (1, {"_planted_private"})
    assert (counts["UI"], private) == (0, set())
    with pytest.raises(AssertionError, match=r"UI/planted\.py: .*explicitly") as err:
        _check_nonliteral(_NONLITERAL + hidden)
    if "_PLANTED_ATTR" in rewrite:
        assert "(reaches _planted_private)" in str(err.value)
