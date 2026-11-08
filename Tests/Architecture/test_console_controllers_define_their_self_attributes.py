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
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test

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


# --- TASK-36947.4: the Console chat controller holds no reference to the app ---
#
# ADR-250 section 7: ``ConsoleChatController`` receives a ``ConsoleControllerHost``
# (members from ``app.services`` plus narrow callables), never the app. A test
# or helper that still sets or reads ``controller.app`` would pass against an
# ignored attribute, so these pin both sides.

_PACKAGE = _REPO_ROOT / "tldw_chatbook"
_APP_NAMES = {"app", "_app", "app_instance", "_app_instance", "tldw_app"}

#: Other names Chat/ code gives the Console chat controller.
_CONTROLLER_ALIASES = {"creator", "preparation_owner"}

#: Modules whose ``controller`` is not the Console chat controller.
_NOT_THE_CONSOLE_CONTROLLER = {
    "tldw_chatbook/UI/Library_Modules/library_file_export.py": (
        "its ``controller`` is the Library screen, which owns an app handle"
    ),
}


def _console_chat_controller_class() -> ast.ClassDef:
    path = _PACKAGE / "Chat" / "console_chat_controller.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ConsoleChatController"
    )


def test_console_chat_controller_never_binds_or_reads_an_app_attribute() -> None:
    """No ``self.app``-like binding, read or class attribute (AST, AC#1)."""

    cls = _console_chat_controller_class()
    bound = _defined_names(cls) & _APP_NAMES
    read = {
        name
        for node in _own_nodes(cls)
        if (name := _self_read(node)) in _APP_NAMES
    }
    assert not bound and not read, (
        f"ConsoleChatController binds {sorted(bound)} / reads {sorted(read)}; "
        "it receives a ConsoleControllerHost, never the app (ADR-250 s7)"
    )


def _controller_receiver(node: ast.AST) -> ast.AST | None:
    """The receiver of a ``<x>.app`` read or ``getattr(<x>, "app")`` call."""

    if isinstance(node, ast.Attribute) and node.attr == "app":
        return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("getattr", "hasattr", "setattr")
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "app"
    ):
        return node.args[0]
    return None


def _controller_app_reads(source: str, *, controller_helpers: bool = False) -> list[int]:
    """Lines reading a controller's ``app``.

    With ``controller_helpers``, a module-level function whose first parameter
    is ``self`` is a controller method moved out of the class (the Chat/
    pattern), so its ``self.app`` counts too.
    """
    lines = []
    with warnings.catch_warnings():  # a module's invalid escapes are not ours
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    helpers = {
        id(node)
        for fn in tree.body
        if controller_helpers
        and isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        and fn.args.args
        and fn.args.args[0].arg == "self"
        for node in ast.walk(fn)
    }
    for node in ast.walk(tree):
        receiver = _controller_receiver(node)
        name = (
            receiver.id
            if isinstance(receiver, ast.Name)
            else receiver.attr
            if isinstance(receiver, ast.Attribute)
            else ""
        )
        if (
            "controller" in name.lower()
            or name in _CONTROLLER_ALIASES
            or (name == "self" and id(node) in helpers)
        ):
            lines.append(node.lineno)
    return lines


def test_no_module_reads_the_console_controllers_app() -> None:
    """No ``controller.app`` read or ``getattr(controller, "app")`` (AST, AC#3)."""

    offenders = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in _NOT_THE_CONSOLE_CONTROLLER:
            continue
        helpers = rel.startswith("tldw_chatbook/Chat/")
        offenders += [
            f"{rel}:{line}"
            for line in _controller_app_reads(
                path.read_text(encoding="utf-8"), controller_helpers=helpers
            )
        ]
    assert not offenders, (
        f"{offenders} read a controller's ``app``; the Console controller has "
        "none -- read the member or callable from ``controller._host``"
    )


@pytest.mark.parametrize(
    "source",
    [
        "x = controller.app.notify",
        "x = getattr(self._chat_controller, 'app', None)",
        "runtime.chat_controller.app = app",
        "x = getattr(self.preparation_owner, 'app', None)",
        "x = getattr(creator, 'app', app)",
    ],
)
def test_controller_app_scan_flags_each_shape(source: str) -> None:
    """Negative control: each shape the scan exists for is caught."""

    assert _controller_app_reads(source) == [1]


def test_controller_app_scan_flags_a_moved_controller_method() -> None:
    """Negative control: a Chat/ helper taking the controller as ``self``."""

    source = "def helper(self):\n    return getattr(self.app, 'x', None)\n"
    assert _controller_app_reads(source, controller_helpers=True) == [2]
    assert _controller_app_reads(source) == []


def _receiver_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _controller_host_writes(source: str, *, controller_helpers: bool = False) -> list[int]:
    """Lines rebinding a Console controller's ``_host`` after construction.

    Counts ``<controller>._host = ...`` and ``setattr(<controller>, "_host",
    ...)``; inside ``ConsoleChatController`` (or a Chat/ helper taking the
    controller as ``self``), ``self._host = ...`` anywhere but ``__init__``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    controller_self: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ConsoleChatController":
            controller_self |= {
                id(inner)
                for fn in node.body
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                and fn.name != "__init__"
                for inner in ast.walk(fn)
            }
    if controller_helpers:
        controller_self |= {
            id(inner)
            for fn in tree.body
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and fn.args.args
            and fn.args.args[0].arg == "self"
            for inner in ast.walk(fn)
        }

    def is_controller(receiver: ast.AST, node: ast.AST) -> bool:
        name = _receiver_name(receiver)
        return (
            "controller" in name.lower()
            or name in _CONTROLLER_ALIASES
            or (name == "self" and id(node) in controller_self)
        )

    lines = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for leaf in ast.walk(target):
                    if (
                        isinstance(leaf, ast.Attribute)
                        and leaf.attr == "_host"
                        and is_controller(leaf.value, node)
                    ):
                        lines.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "_host"
            and is_controller(node.args[0], node)
        ):
            lines.append(node.lineno)
    return sorted(set(lines))


def test_nothing_rebinds_the_console_controllers_host() -> None:
    """The host is bound once, in ``ConsoleChatController.__init__`` (AST, M3)."""

    offenders = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in _NOT_THE_CONSOLE_CONTROLLER:
            continue
        offenders += [
            f"{rel}:{line}"
            for line in _controller_host_writes(
                path.read_text(encoding="utf-8"),
                controller_helpers=rel.startswith("tldw_chatbook/Chat/"),
            )
        ]
    assert not offenders, (
        f"{offenders} rebind a Console controller's ``_host``; it is built once "
        "where the app is and passed to the constructor"
    )


@pytest.mark.parametrize(
    "source",
    [
        "controller._host = host",
        "self._chat_controller._host = host",
        "setattr(controller, '_host', host)",
        "class ConsoleChatController:\n    def rebind(self):\n        self._host = h",
    ],
)
def test_controller_host_write_scan_flags_each_shape(source: str) -> None:
    """Negative control: each rebinding shape is caught; ``__init__`` is not."""

    assert len(_controller_host_writes(source)) == 1
    init = "class ConsoleChatController:\n    def __init__(self):\n        self._host = h"
    assert _controller_host_writes(init) == []


def test_controller_host_for_app_takes_each_member_from_its_slice() -> None:
    """Each member is the ``ApplicationServices`` field it names (M4)."""

    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_controller_host import (
        MEMBER_SOURCES,
        ConsoleControllerHost,
        controller_host_for_app,
    )

    slices: dict[str, SimpleNamespace] = {}
    expected = {}
    for member, (slice_name, field) in MEMBER_SOURCES.items():
        sentinel = object()
        setattr(slices.setdefault(slice_name, SimpleNamespace()), field, sentinel)
        expected[member] = sentinel
    app = SimpleNamespace(services=SimpleNamespace(**slices))
    host = controller_host_for_app(app)
    assert {m: getattr(host, m) for m in MEMBER_SOURCES} == expected
    fields = set(ConsoleControllerHost.__dataclass_fields__)
    assert set(MEMBER_SOURCES) | {"raw_cli_runtime"} <= fields
    assert host.raw_cli_runtime is None, "raw_cli_runtime comes from the runtime"

    from tldw_chatbook import app_services

    bundle = app_services.ApplicationServices.__dataclass_fields__
    for member, (slice_name, field) in MEMBER_SOURCES.items():
        annotation = str(bundle[slice_name].type)
        slice_class = getattr(app_services, annotation.split("[", 1)[0].strip())
        assert field in slice_class.__dataclass_fields__, (member, slice_name, field)


@pytest.mark.asyncio
async def test_console_chat_controller_holds_no_app_after_construction(
    monkeypatch,
) -> None:
    """Runtime check: neither construction site leaves the app on the controller."""

    from dataclasses import fields

    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_controller_host import controller_host_for_app
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    class App:
        def notify(self, *_args, **_kwargs):
            return None

        def call_from_thread(self, fn, *args, **kwargs):
            return fn(*args, **kwargs)

    # The prompt history file belongs to the profile; this check needs none.
    monkeypatch.setattr(ConsoleRuntime, "ensure_prompt_history", lambda self: None)
    app = App()
    direct = ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=object(),
        host=controller_host_for_app(app),
    )
    runtime = ConsoleRuntime(app)
    built = runtime.ensure_chat_controller(
        store=ConsoleChatStore(), provider_gateway=object()
    )
    try:
        for controller in (direct, built):
            assert not hasattr(controller, "app")
            assert not [k for k, v in vars(controller).items() if v is app]
            host = controller._host
            assert host.call_from_thread == app.call_from_thread
            assert not [f.name for f in fields(host) if getattr(host, f.name) is app]
    finally:
        await built.shutdown()
        await direct.shutdown()


# --- TASK-36947.4 fix round 1: nothing the controller reaches holds the app ---

#: Paths the walk does not enter, each with its reason. The owning runtime is
#: built around the app (ADR-250 rule 4) and outlives the controller.
_REACHABILITY_EXEMPT_TYPES = {"ConsoleRuntime": "the owning runtime (ADR-250 rule 4)"}
_NOT_WALKED = (
    "asyncio", "threading", "logging", "loguru", "sqlite3", "concurrent",
    "weakref", "_weakrefset", "textual", "rich", "pathlib", "re",
)


def _app_holders(root, app, *, depth: int = 6) -> list[str]:
    """Paths from ``root`` to ``app`` that end on an object attribute or item.

    The walk enters instance ``__dict__``s, slots and dataclass fields,
    containers, ``functools.partial`` args and keywords, bound-method
    ``__self__`` and function closure cells. It skips modules, classes,
    builtins and the exempt owner types. A path whose last step is a
    callable binding (partial argument, ``__self__``, closure cell) is a
    narrow callable over the app (ADR-250 s7) and is allowed; a path whose
    last step is an attribute or container item is an object holding the
    app, which is what this guard forbids.
    """
    import functools
    import types

    holders: list[str] = []
    seen: set[int] = {id(root)}
    frontier = [(root, "controller")]

    def children(obj):
        if isinstance(obj, functools.partial):
            yield obj.func, ".func", "binding"
            for i, value in enumerate(obj.args):
                yield value, f".args[{i}]", "binding"
            for key, value in (obj.keywords or {}).items():
                yield value, f".keywords[{key}]", "binding"
            return
        if isinstance(obj, types.MethodType):
            yield obj.__self__, ".__self__", "binding"
            yield obj.__func__, ".__func__", "binding"
            return
        if isinstance(obj, types.FunctionType):
            for i, cell in enumerate(obj.__closure__ or ()):
                try:
                    yield cell.cell_contents, f".<cell {i}>", "binding"
                except ValueError:
                    continue
            return
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                yield value, f"[{key!r}]", "attr"
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for i, value in enumerate(list(obj)):
                yield value, f"[{i}]", "attr"
            return
        names = list(getattr(obj, "__dict__", {}) or {})
        for klass in type(obj).__mro__:
            names += [n for n in getattr(klass, "__slots__", ()) if isinstance(n, str)]
        for name in dict.fromkeys(names):
            try:
                value = object.__getattribute__(obj, name)
            except AttributeError:
                continue
            yield value, f".{name}", "attr"

    def walkable(obj) -> bool:
        if obj is None or isinstance(obj, (str, bytes, int, float, bool, complex, type)):
            return False
        if isinstance(obj, types.ModuleType):
            return False
        if type(obj).__name__ in _REACHABILITY_EXEMPT_TYPES:
            return False
        if isinstance(
            obj,
            (dict, list, tuple, set, frozenset, functools.partial,
             types.MethodType, types.FunctionType),
        ):
            return True
        module = (type(obj).__module__ or "").split(".")[0]
        return module not in _NOT_WALKED and module != "builtins"

    for _level in range(depth):
        next_frontier = []
        for obj, path in frontier:
            for value, step, kind in children(obj):
                if value is app:
                    if kind == "attr":
                        holders.append(path + step)
                    continue
                if id(value) in seen or not walkable(value):
                    continue
                seen.add(id(value))
                next_frontier.append((value, path + step))
        frontier = next_frontier
    return sorted(holders)


@private_profile_test
@pytest.mark.asyncio
async def test_no_object_the_console_controller_reaches_holds_the_app(
    request, tmp_path
) -> None:
    """Bounded reachability: from a runtime-built controller over a real app
    (real ``app.services``), the app is reachable only through narrow callables
    and the exempt owning runtime -- never as an object's attribute."""

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    app = _build_test_app()
    app.chachanotes_db = CharactersRAGDB(str(tmp_path / "chacha.db"), "reach")
    app._wire_chat_conversation_services()
    app._publish_application_services()
    runtime = app.console_runtime
    assert isinstance(runtime, ConsoleRuntime)
    store = runtime.ensure_chat_store()
    gateway = runtime.ensure_provider_gateway()
    bridge = runtime.ensure_agent_bridge(
        store_factory=lambda: store, provider_gateway_factory=lambda: gateway
    )
    controller = runtime.ensure_chat_controller(
        store=store, provider_gateway=gateway, agent_bridge=bridge
    )
    runtime.ensure_archive_recovery()
    try:
        assert bridge is not None, "the walk must cover the agent bridge's consumers"
        holders = _app_holders(controller, app)
        assert not holders, "objects holding the app:\n" + "\n".join(holders)
    finally:
        await runtime.dispose()
        app.chachanotes_db.close_connection()


def test_reachability_walk_flags_an_object_holding_the_app() -> None:
    """Negative control: an attribute holder is flagged; a narrow callable is not."""
    import functools

    app = object()

    class Holder:
        def __init__(self):
            self._app = app

    class Root:
        def __init__(self):
            self.child = {"wake": Holder()}
            self.notify = functools.partial(print, app)

    assert _app_holders(Root(), app) == ["controller.child['wake']._app"]
