"""An extracted Console controller must define every ``self.<name>`` it reads.

TASK-33621.15. The wave-4 moves out of ``ChatScreen`` turned screen methods
into controllers in ``UI/Console_Modules`` that reach the screen through
injected accessors. One moved call kept the screen's spelling:
``ConsoleSessionController._close_console_session_tab`` called
``self._console_runtime()``, but the controller only has
``self._console_runtime_accessor``. The ``AttributeError`` was swallowed by
the close worker's ``exit_on_error=False``, so from b7dd8e53f2 (2026-09-08)
no Console tab could be closed, with no toast and no log line, for three
weeks. The routing tests that encoded the right outcome ran in no PR gate.

This check finds that shape without importing anything: in every class of
``UI/Console_Modules`` that has no base class and no ``__getattr__``, each
``self.<name>`` read must be defined by the class itself -- a method, a class
attribute, a ``self.<name> = ...`` assignment, or ``setattr(self, "<name>",
...)``. ``self.<name> += ...``, ``self.<name>[k] = ...`` and
``self.<name>.x = ...`` read ``<name>`` before storing, so they are reads,
not definitions. A class with a base inherits names this AST cannot see, so
it is out of scope rather than guessed at.

It is the ``self`` analogue of ``test_no_undefined_module_globals.py``.
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONSOLE_MODULES = _REPO_ROOT / "tldw_chatbook" / "UI" / "Console_Modules"

#: ``module path::Class.attribute`` reads known to be unresolved, each with
#: its reason. Keep it empty: an entry here is a controller path that raises
#: ``AttributeError`` the moment it runs.
_EXEMPT: dict[str, str] = {}


def _self_attribute(node: ast.AST) -> str | None:
    """The ``<name>`` of a ``self.<name>`` expression, else ``None``."""

    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def _bound_self_names(target: ast.AST) -> Iterator[str]:
    """The ``<name>`` of each ``self.<name>`` an assignment target binds.

    Only a bare ``self.<name>``, alone or unpacked from a tuple or list,
    binds ``<name>``. ``self.<name>[key] = ...`` and ``self.<name>.x = ...``
    read ``self.<name>`` to store into it, so they define nothing.
    """

    if (name := _self_attribute(target)) is not None:
        yield name
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _bound_self_names(element)
    elif isinstance(target, ast.Starred):
        yield from _bound_self_names(target.value)


def _self_read(node: ast.AST) -> str | None:
    """The ``<name>`` a node reads from ``self``, else ``None``.

    ``self.<name> += ...`` is a read although its target is a ``Store``.
    """

    if isinstance(node, ast.AugAssign):
        return _self_attribute(node.target)
    name = _self_attribute(node)
    if name is not None and isinstance(node.ctx, ast.Load):
        return name
    return None


def _own_nodes(cls: ast.ClassDef):
    """Every node inside ``cls`` except those of classes nested in it.

    A nested class's ``self`` is a different object.
    """

    stack = list(cls.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.ClassDef):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _defined_names(cls: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(
                    sub.id for sub in ast.walk(target) if isinstance(sub, ast.Name)
                )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    for node in _own_nodes(cls):
        # Not ``AugAssign``: ``self.<name> += 1`` reads ``<name>`` first. Not
        # a bare ``self.<name>: T`` either: without a value nothing is bound.
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            targets = []
        for target in targets:
            names.update(_bound_self_names(target))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "self"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            names.add(node.args[1].value)
    return names


def _in_scope(cls: ast.ClassDef) -> bool:
    """A base-less class that resolves attributes only through itself."""

    if any(
        not (isinstance(base, ast.Name) and base.id == "object") for base in cls.bases
    ):
        return False
    return not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"__getattr__", "__getattribute__"}
        for node in cls.body
    )


def _unresolved_in(tree: ast.AST, relative: str) -> tuple[int, dict[str, int]]:
    """Count the in-scope classes of ``tree`` and map each unresolved read to its line."""

    scanned = 0
    unresolved: dict[str, int] = {}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if not _in_scope(cls):
            continue
        scanned += 1
        defined = _defined_names(cls)
        for node in _own_nodes(cls):
            name = _self_read(node)
            if (
                name is None
                or name in defined
                or (name.startswith("__") and name.endswith("__"))
            ):
                continue
            unresolved.setdefault(f"{relative}::{cls.name}.{name}", node.lineno)
    return scanned, unresolved


def _unresolved_reads() -> tuple[int, dict[str, int]]:
    """Count the in-scope classes and map each unresolved read to its line."""

    scanned = 0
    unresolved: dict[str, int] = {}
    for path in sorted(_CONSOLE_MODULES.glob("*.py")):
        relative = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        count, reads = _unresolved_in(tree, relative)
        scanned += count
        unresolved.update(reads)
    return scanned, unresolved


def test_console_controllers_read_only_self_attributes_they_define() -> None:
    """Every ``self.<name>`` a base-less Console_Modules class reads is defined."""

    scanned, unresolved = _unresolved_reads()
    # A glob or scope bug that silently scans nothing must not pass as clean.
    assert scanned >= 50, f"only {scanned} Console_Modules classes were scanned"
    offenders = {key: line for key, line in unresolved.items() if key not in _EXEMPT}
    assert not offenders, (
        "Console_Modules classes read self attributes they never define "
        "(an AttributeError when that path runs -- see this module's docstring):\n"
        + "\n".join(f"  {key} (line {line})" for key, line in sorted(offenders.items()))
    )


@pytest.mark.parametrize("key", sorted(_EXEMPT))
def test_exempt_reads_are_still_unresolved(key: str) -> None:
    """An exemption whose read got fixed must be deleted with it."""

    assert key in _unresolved_reads()[1], f"{key} is resolved now; drop its exemption"


def _probe_reads(body: str) -> set[str]:
    """Unresolved reads of a base-less ``Probe`` class whose body is ``body``."""

    source = "class Probe:\n" + textwrap.indent(body, "    ")
    return set(_unresolved_in(ast.parse(source), "probe.py")[1])


@pytest.mark.parametrize(
    ("body", "missing"),
    [
        ("def bump(self):\n    self.count += 1\n", "count"),
        ("def bump(self, key):\n    self.counts[key] += 1\n", "counts"),
        ("def put(self, key):\n    self.cache[key] = 1\n", "cache"),
        ("def ready(self):\n    self.state.ready = True\n", "state"),
        (
            "def __init__(self):\n    self.ready: bool\n"
            "def read(self):\n    return self.ready\n",
            "ready",
        ),
    ],
    ids=[
        "augassign",
        "augassign-subscript",
        "subscript-store",
        "attr-store",
        "bare-annotation",
    ],
)
def test_a_store_that_reads_first_does_not_define_the_name(
    body: str, missing: str
) -> None:
    """``self.x += 1``, ``self.x[k] = v`` and ``self.x.y = v`` read ``self.x``.

    Each raises ``AttributeError`` when ``x`` was never bound, so none of
    them may count as the definition that clears ``x`` (Qodo, PR #2933).
    """

    assert _probe_reads(body) == {f"probe.py::Probe.{missing}"}


@pytest.mark.parametrize(
    "body",
    [
        "def __init__(self):\n    self.count = 0\n"
        "def bump(self):\n    self.count += 1\n",
        "def __init__(self):\n    self.counts = {}\n"
        "def bump(self, key):\n    self.counts[key] += 1\n",
        "count = 0\ndef bump(self):\n    self.count += 1\n",
        "def __init__(self):\n    self.a, (self.b, *self.c) = 1, (2, 3)\n"
        "def read(self):\n    return self.a, self.b, self.c\n",
        "def __init__(self):\n    self.ready: bool = False\n"
        "def read(self):\n    return self.ready\n",
        "def __init__(self):\n    setattr(self, 'late', 1)\n"
        "def read(self):\n    return self.late\n",
    ],
    ids=[
        "init-then-bump",
        "init-then-subscript-bump",
        "class-attr",
        "unpacking",
        "annotated",
        "setattr",
    ],
)
def test_a_real_binding_defines_the_name(body: str) -> None:
    """Negative control: every way a class really binds ``self.x`` still clears it."""

    assert _probe_reads(body) == set()
