# test_reactive_mutable_default_inventory.py
# Description: Guard against non-callable mutable reactive defaults (task-15771, task-16843)
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
* the *result* of a ``list()``/``dict()``/``set()`` call (not callable),
* a name bound to a module-level mutable literal (one shared object), or
* (task-16843) a shared mutable *instance* default — ``reactive(SomeClass())`` —
  unless ``SomeClass`` is in ``IMMUTABLE_INSTANCE_ALLOWLIST`` (see below).

task-16843 closed the gap task-15771's review flagged as F2:
``reactive(SomeClass())`` is the exact same one-object-per-class bug (Textual
installs that one instance on every widget instance) but the original
detector only matched ``list()``/``dict()``/``set()`` call results by name,
so an arbitrary constructor call sailed through unseen. The detector now
also flags a call whose callee looks like a class instantiation — this
repo's PascalCase-class / snake_case-function convention (see CLAUDE.md) is
the only signal available, so a lowercase factory function that returns a
mutable instance is *still* not detected; that gap is deliberate and tested
(``test_unrecognized_lowercase_factory_is_a_documented_gap``), not silently
claimed as covered, per the 15771 review's "state the contract honestly"
finding.

Not every ``reactive(SomeClass())`` is a live bug, though: if ``SomeClass``
is a frozen dataclass whose *every* field is itself an immutable type
(``frozenset``, ``Literal`` str, ``int | None``, an ``Enum``, ...), the
shared instance cannot be mutated in place — frozen blocks reassigning a
field, and there is no mutable container underneath a field to mutate
around that block either. Forcing those into a factory would be a
functionally-identical rewrite, not a fix, so they are instead named in
``IMMUTABLE_INSTANCE_ALLOWLIST`` with the reasoning that earns each entry.
Contrast ``ConsoleContextSnapshot`` (task-16843's one live/latent site): it
is *also* ``frozen=True``, but its ``current_messages: list`` and
``next_send_payload: dict`` fields are mutable containers frozen only
blocks *reassigning* — so it was converted to a lambda factory instead of
allowlisted. See ``Tests/Widgets/test_reactive_default_aliasing.py::
test_console_context_modal_snapshots_do_not_leak_across_instances`` for the
cross-instance leak this shape allows when a class is wrongly allowlisted
(or never fixed).

Born red three times: against the pre-fix tree (27 sites), again after the
task-15771 review taught ``_call_name`` the ``ast.Subscript`` form the first
sweep was structurally blind to (14 more sites, all in
``UI/Watchlists_Modules/`` — 41 total), and again for task-16843's
instance-call shape (the 5 sites named in the module docstring above, before
``ConsoleContextSnapshot``'s was converted to a factory and ``RegionLayout``/
``TreeScope`` were allowlisted). If it fails, replace the default with a
callable, never with a "reassign before use" workaround.
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

#: task-16843: classes whose ``reactive(ClassName(...))`` instance default is
#: provably safe -- a ``frozen=True`` dataclass (blocks reassigning fields)
#: where every field's static type is itself immutable, so there is no
#: mutable container left to mutate in place. Adding a name here requires
#: re-reading the class's field *types*, not just checking for
#: ``frozen=True`` -- a frozen dataclass can still carry a ``list``/``dict``
#: field (``ConsoleContextSnapshot`` is exactly that trap: frozen=True, NOT
#: allowlisted, fixed with a factory instead). Do not add speculatively.
IMMUTABLE_INSTANCE_ALLOWLIST: dict[str, str] = {
    "RegionLayout": (
        "tldw_chatbook/UI/Watchlists_Modules/region_layout.py — "
        "@dataclass(frozen=True); fields: collapsed: frozenset[Region] = "
        "frozenset(), solo_region: Region | None = None, "
        "_pre_solo: frozenset[Region] | None = None"
    ),
    "TreeScope": (
        "tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py — "
        "@dataclass(frozen=True); fields: kind: Literal[...], "
        "watchlist_id: int | None = None, source_id: int | None = None"
    ),
}


def _looks_like_class_instantiation(call_name: str | None) -> bool:
    """Heuristic for "this call constructs an instance": a PascalCase callee
    name, this repo's class-naming convention (CLAUDE.md: "PascalCase
    classes, snake_case functions"). Deliberately does not (and structurally
    cannot) catch a lowercase factory function that returns a mutable
    instance -- see ``test_unrecognized_lowercase_factory_is_a_documented_gap``.
    """
    return bool(call_name) and call_name[0].isupper()


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


def _default_violation_reason(
    default: ast.expr, module_mutables: set[str]
) -> str | None:
    """Return why ``default`` is a disallowed shared-mutable default, or
    ``None`` when it is callable-or-immutable. Path-free so it can be driven
    directly by synthetic source in tests, not just real package files.
    """
    if isinstance(default, MUTABLE_LITERALS):
        return f"mutable literal default {ast.unparse(default)!r}"
    if isinstance(default, ast.Name) and default.id in module_mutables:
        return f"module-level shared mutable {default.id!r}"
    if isinstance(default, ast.Call):
        call_name = _call_name(default)
        if call_name in MUTABLE_FACTORY_NAMES:
            return (
                f"non-callable call result {ast.unparse(default)!r}"
                " (pass the factory itself, without parentheses)"
            )
        if (
            _looks_like_class_instantiation(call_name)
            and call_name not in IMMUTABLE_INSTANCE_ALLOWLIST
        ):
            return (
                f"shared mutable instance default {ast.unparse(default)!r}"
                " (reactive(SomeClass()) installs the SAME instance on"
                " every widget instance until reassigned -- pass"
                " reactive(SomeClass) for a no-arg constructor, or"
                " reactive(lambda: SomeClass(...)) when it needs args; if"
                " SomeClass is a frozen dataclass with only immutable"
                " field types, add it to IMMUTABLE_INSTANCE_ALLOWLIST"
                " instead, with the field-type reasoning that proves it)"
            )
    return None


def _violations_in_tree(tree: ast.Module) -> list[tuple[int, str]]:
    """Return ``(lineno, reason)`` for every disallowed reactive()/var()
    default in an already-parsed module."""
    module_mutables = _module_level_mutables(tree)
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in REACTIVE_NAMES:
            continue
        default = _default_arg(node)
        if default is None:
            continue
        reason = _default_violation_reason(default, module_mutables)
        if reason is not None:
            violations.append((node.lineno, reason))
    return violations


def _violations_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    relative = path.relative_to(PACKAGE_ROOT.parent)
    return [
        f"{relative}:{lineno}: {reason}"
        for lineno, reason in _violations_in_tree(tree)
    ]


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


def test_instance_call_default_is_flagged() -> None:
    """``reactive(SomeClass())`` -- the shape task-15771's review flagged as
    F2, structurally invisible to the pre-16843 detector -- must be flagged.
    Regression-proves the gap this task closes: born red against the
    pre-16843 detector, which only recognised ``list()``/``dict()``/``set()``
    call results, nothing else shaped like a constructor call.
    """
    tree = ast.parse(
        "from textual.reactive import reactive\n"
        "from textual.widget import Widget\n\n"
        "class SomeMutableThing:\n"
        "    pass\n\n"
        "class ScratchWidget(Widget):\n"
        "    thing = reactive(SomeMutableThing())\n"
    )
    violations = _violations_in_tree(tree)
    assert len(violations) == 1
    lineno, reason = violations[0]
    assert lineno == 8
    assert "SomeMutableThing" in reason


def test_allowlisted_immutable_instance_default_is_not_flagged() -> None:
    """An allowlisted class's instance default is documented-safe, not
    silently ignored: ``RegionLayout``/``TreeScope`` are frozen dataclasses
    whose only field types are themselves immutable (``frozenset``,
    ``Literal`` str, ``int | None``, an ``Enum``) -- there is no mutable
    container to leak, so forcing a factory rewrite would be no-op churn,
    not a fix.
    """
    tree = ast.parse(
        "from textual.reactive import reactive\n"
        "from textual.widget import Widget\n\n"
        "class RegionLayout:\n"
        "    pass\n\n"
        "class ScratchWidget(Widget):\n"
        "    layout = reactive(RegionLayout())\n"
    )
    assert _violations_in_tree(tree) == []


def test_unallowlisted_frozen_class_instance_default_is_still_flagged() -> None:
    """The allowlist is opt-in per class name, not a blanket exemption for
    anything that merely looks frozen-dataclass-shaped -- a class NOT in
    ``IMMUTABLE_INSTANCE_ALLOWLIST`` is flagged regardless of whether it
    happens to also be immutable, so a new mutable-fielded class never
    slips through by resembling an allowlisted one.
    """
    tree = ast.parse(
        "from textual.reactive import reactive\n"
        "from textual.widget import Widget\n\n"
        "class NotAllowlisted:\n"
        "    pass\n\n"
        "class ScratchWidget(Widget):\n"
        "    thing = reactive(NotAllowlisted())\n"
    )
    assert len(_violations_in_tree(tree)) == 1


def test_unrecognized_lowercase_factory_is_a_documented_gap() -> None:
    """A lowercase factory function returning a mutable instance is NOT
    detected -- this repo's PascalCase-class convention is the detector's
    only signal that a call constructs an instance, so a snake_case wrapper
    is structurally invisible to it. Pinned so a future change to the
    heuristic is a deliberate decision, not a silent regression in either
    direction (the 15771 review's "state the contract honestly" finding).
    """
    tree = ast.parse(
        "from textual.reactive import reactive\n"
        "from textual.widget import Widget\n\n"
        "def make_mutable_thing():\n"
        "    return {}\n\n"
        "class ScratchWidget(Widget):\n"
        "    thing = reactive(make_mutable_thing())\n"
    )
    assert _violations_in_tree(tree) == []
