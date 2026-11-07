"""Each ``server_*_service.py`` types its client as a narrow Protocol (TASK-36953.1).

A service lists, in a ``typing.Protocol`` class in its own module, exactly the
``TLDWAPIClient`` methods it uses. This AST scan fails when

* a service touches a client method its Protocol does not list,
* a Protocol lists a method the service never touches,
* a Protocol lists a method ``TLDWAPIClient`` does not define, or
* the scan cannot follow what a service does with its client.

"The client" is ``self.client``, ``self._require_client()``, a name ``client``,
and whatever is bound from one of these: a plain, annotated, walrus or tuple
assignment to a name or to ``self.<attr>``; an argument to a same-module
function or method, followed by position or keyword to its parameter whatever
that parameter is called; or the result of a same-module function that returns
it. A use is an attribute read on the client or ``getattr(<client>, name)``.
Handing the client to anything else (another module's function, a container)
fails as unresolved.

A non-literal ``getattr`` name fails closed: it must resolve to string
literals through module-level string constants, through a parameter's
arguments at every same-module call site (by keyword or position,
recursively), or, for ``x.<field>``, through every value a same-module
dataclass constructor or ``<field>=`` keyword binds to that field. Any value
that does not resolve is reported with its file and line.

Protocol members are ``(*args: Any, **kwargs: Any) -> Any``: AC#1 asks for the
method list, so argument and return types now come from ``TLDWAPIClient`` (via
mypy) only at call sites where the client is not narrowed to the Protocol;
full signatures are a possible follow-up.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "tldw_chatbook"

# Services with no TLDWAPIClient in their call path; each reason is re-checked
# by test_exempt_service_reaches_no_tldw_api_client.
EXEMPT = {
    "server_notifications_scope_service.py": "wraps ServerNotificationsService, no client",
    "server_runtime_scope_service.py": "wraps ServerRuntimeService, no client",
    "server_unified_service.py": "drives MCPUnifiedClient (tldw_api/mcp_unified_client.py), not TLDWAPIClient",
}
# The exempt services that hold no client of any kind.
HOLDS_NO_CLIENT = {
    "server_notifications_scope_service.py",
    "server_runtime_scope_service.py",
}

# Dead ``getattr(client, name, None)`` probes for a method TLDWAPIClient never
# had, by service path. Only that probe shape in that file is excused: a direct
# call (an AttributeError at runtime) still fails.
ABSENT_PROBES = {
    "Research_Interop/server_research_search_service.py": {"run_research_websearch"},
}

# Builtins that may take the client without calling a method on it.
INTROSPECTION = {"getattr", "hasattr", "callable", "isinstance"}

SERVICE_FILES = sorted(PACKAGE_DIR.rglob("server_*_service.py"))
CLIENT_SERVICES = [p for p in SERVICE_FILES if p.name not in EXEMPT]
EXEMPT_SERVICES = [p for p in SERVICE_FILES if p.name in EXEMPT]

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _is_str(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _is_self(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "self"


def _positional(fn: ast.FunctionDef | ast.AsyncFunctionDef, skip: bool) -> list[str]:
    return [a.arg for a in fn.args.posonlyargs + fn.args.args][int(skip) :]


def _params(scope: ast.AST | None) -> set[str]:
    if scope is None:
        return set()
    a = scope.args
    extra = [x for x in (a.vararg, a.kwarg) if x is not None]
    return {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs + extra}


def _rebinds(scope: ast.AST | None, name: str) -> bool:
    return scope is not None and any(
        isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load)
        for n in ast.walk(scope)
    )


def _default(fn: ast.FunctionDef | ast.AsyncFunctionDef, param: str) -> ast.expr | None:
    a = fn.args
    positional = a.posonlyargs + a.args
    pairs = list(zip(positional[len(positional) - len(a.defaults) :], a.defaults))
    pairs += list(zip(a.kwonlyargs, a.kw_defaults))
    return next((d for p, d in pairs if p.arg == param and d is not None), None)


# ponytail: names are tracked module-wide, not per scope (an over-approximation,
# so it fails closed), and ``for``/``with`` targets are not followed; scope it
# per function if a service ever trips over a reused name.
class ClientScan:
    """What one service module does with its client: ``used`` and ``unresolved``."""

    def __init__(self, tree: ast.Module, rel_path: str) -> None:
        self.tree = tree
        self.rel_path = rel_path
        self.parent = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        self.functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        for node in ast.walk(tree):
            if isinstance(node, _FUNCTIONS):
                self.functions.setdefault(node.name, []).append(node)
        self.classes = {
            n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
        }
        bindings: dict[str, list[ast.expr | None]] = {}
        for stmt in tree.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = (
                    stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                )
                for t in targets:
                    if isinstance(t, ast.Name):
                        bindings.setdefault(t.id, []).append(stmt.value)
        # Module-level names bound exactly once, to a string literal.
        self.constants = {
            name: values[0].value
            for name, values in bindings.items()
            if len(values) == 1 and _is_str(values[0])
        }
        self.names = {"client"}  # names bound to the client
        self.attrs = {"client"}  # self.<attr> bound to the client
        self.returners = {"_require_client"}  # functions that return the client
        self.unresolved: set[str] = set()
        while self._bind_once():
            pass
        self.used = self._uses()

    def _where(self, node: ast.AST) -> str:
        return f"{self.rel_path}:{node.lineno}"

    def is_client(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.names
        if isinstance(node, ast.Attribute):
            return node.attr in self.attrs and _is_self(node.value)
        if isinstance(node, ast.Call):
            func = node.func
            return getattr(func, "attr", getattr(func, "id", None)) in self.returners
        return False

    def _enclosing(self, node: ast.AST, kinds: tuple[type, ...]) -> ast.AST | None:
        node = self.parent.get(node)
        while node is not None and not isinstance(node, kinds):
            node = self.parent.get(node)
        return node

    def _callees(
        self, call: ast.Call, *, strict: bool
    ) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, bool]]:
        """Same-module functions ``call`` may run, each with "skip the self slot".

        ``strict`` only follows calls whose target is certain to be in this module:
        a bare name, ``self.``/``cls.`` methods, and constructors.
        """
        func = call.func
        if isinstance(func, ast.Name):
            cls = (
                self._enclosing(call, (ast.ClassDef,))
                if func.id == "cls"
                else self.classes.get(func.id)
            )
            if cls is not None:
                return [
                    (f, True)
                    for f in cls.body
                    if isinstance(f, _FUNCTIONS) and f.name == "__init__"
                ]
            return [
                (f, False)
                for f in self.functions.get(func.id, [])
                if not isinstance(self.parent.get(f), ast.ClassDef)
            ]
        if isinstance(func, ast.Attribute) and (
            not strict
            or (isinstance(func.value, ast.Name) and func.value.id in {"self", "cls"})
        ):
            return [
                (
                    f,
                    not any(
                        getattr(d, "id", None) == "staticmethod"
                        for d in f.decorator_list
                    ),
                )
                for f in self.functions.get(func.attr, [])
                if isinstance(self.parent.get(f), ast.ClassDef)
            ]
        return []

    def _bind(self, target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
            value, (ast.Tuple, ast.List)
        ):
            for t, v in zip(target.elts, value.elts):
                self._bind(t, v)
            return
        if not self.is_client(value):
            return
        if isinstance(target, ast.Starred):
            target = target.value
        if isinstance(target, ast.Name):
            self.names.add(target.id)
        elif isinstance(target, ast.Attribute) and _is_self(target.value):
            self.attrs.add(target.attr)
        else:
            self.unresolved.add(
                f"client stored in {ast.unparse(target)} at {self._where(target)}"
            )

    def _hand_off(self, call: ast.Call) -> None:
        """Follow a client passed to a same-module function into its parameter."""
        if isinstance(call.func, ast.Name) and call.func.id in INTROSPECTION:
            return
        passed = [(i, None, a) for i, a in enumerate(call.args)]
        passed += [(None, kw.arg or "**", kw.value) for kw in call.keywords]
        for index, keyword, arg in passed:
            if not self.is_client(arg):
                continue
            params = []
            for fn, skip in self._callees(call, strict=True):
                names = _positional(fn, skip) + [a.arg for a in fn.args.kwonlyargs]
                if keyword is not None:
                    params.append(keyword if keyword in names else None)
                elif any(isinstance(a, ast.Starred) for a in call.args[:index]):
                    params.append(None)
                else:
                    positional = _positional(fn, skip)
                    params.append(
                        positional[index] if index < len(positional) else None
                    )
            if not params or None in params:
                self.unresolved.add(
                    f"client handed to {ast.unparse(call.func)} at {self._where(call)}"
                )
            self.names |= {p for p in params if p}

    def _bind_once(self) -> bool:
        before = (len(self.names), len(self.attrs), len(self.returners))
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self._bind(target, node.value)
            elif (
                isinstance(node, (ast.AnnAssign, ast.NamedExpr))
                and node.value is not None
            ):
                self._bind(node.target, node.value)
            elif isinstance(node, ast.Call):
                self._hand_off(node)
            elif (
                isinstance(node, ast.Return)
                and node.value
                and self.is_client(node.value)
            ):
                self.returners.add(self._enclosing(node, _FUNCTIONS).name)
        return (len(self.names), len(self.attrs), len(self.returners)) != before

    def _uses(self) -> set[str]:
        absent = ABSENT_PROBES.get(self.rel_path, set())
        used: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute) and self.is_client(node.value):
                used.add(node.attr)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and self.is_client(node.args[0])
            ):
                names = self._method_names(node.args[1], self._enclosing(node, _SCOPES))
                used |= names - absent if len(node.args) == 3 else names
        return used

    def _method_names(
        self, expr: ast.AST, scope: ast.AST | None, seen: frozenset = frozenset()
    ) -> set[str]:
        """The string values ``expr`` can take; anything else goes to ``unresolved``."""
        if _is_str(expr):
            return {expr.value}
        if isinstance(expr, ast.Name):
            is_param = expr.id in _params(scope)
            rebound = _rebinds(scope, expr.id)
            if is_param and not rebound and isinstance(scope, _FUNCTIONS):
                return self._param_values(scope, expr.id, seen)
            if not is_param and not rebound and expr.id in self.constants:
                return {self.constants[expr.id]}
        elif isinstance(expr, ast.Attribute):
            key = ("field", expr.attr)
            return set() if key in seen else self._field_values(expr.attr, seen | {key})
        self.unresolved.add(f"getattr name {ast.unparse(expr)} at {self._where(expr)}")
        return set()

    def _param_values(
        self, fn: ast.FunctionDef | ast.AsyncFunctionDef, param: str, seen: frozenset
    ) -> set[str]:
        """The values every same-module call site passes as ``fn``'s ``param``."""
        if (fn, param) in seen:
            return set()
        seen = seen | {(fn, param)}
        sites = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                sites += [
                    (node, skip)
                    for f, skip in self._callees(node, strict=False)
                    if f is fn
                ]
            elif getattr(node, "id", getattr(node, "attr", None)) == fn.name:
                parent = self.parent.get(node)
                if not (isinstance(parent, ast.Call) and parent.func is node):
                    self.unresolved.add(
                        f"{fn.name} referenced without a call at {self._where(node)}"
                    )
        if not sites:
            self.unresolved.add(
                f"no same-module call site binds {fn.name}({param}) at {self._where(fn)}"
            )
        found: set[str] = set()
        for call, skip in sites:
            arg = next((kw.value for kw in call.keywords if kw.arg == param), None)
            positional = _positional(fn, skip)
            index = positional.index(param) if param in positional else len(call.args)
            if arg is None and any(
                isinstance(a, ast.Starred) for a in call.args[: index + 1]
            ):
                arg = call
            elif arg is None and index < len(call.args):
                arg = call.args[index]
            elif arg is None and any(kw.arg is None for kw in call.keywords):
                arg = call
            elif arg is None:
                arg = _default(fn, param)
            if arg is not None:
                found |= self._method_names(arg, self._enclosing(call, _SCOPES), seen)
        return found

    def _field_values(self, field: str, seen: frozenset) -> set[str]:
        """The values same-module code binds to ``<dataclass>.<field>``."""
        slots = {}
        for name, cls in self.classes.items():
            fields = [
                s.target.id
                for s in cls.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            ]
            if field in fields:
                slots[name] = fields.index(field)
        if not slots:
            self.unresolved.add(
                f"getattr name .{field}: no same-module dataclass has that field"
            )
            return set()
        found: set[str] = set()
        for node in ast.walk(self.tree):
            values = []
            if isinstance(node, ast.Call):
                values = [kw.value for kw in node.keywords if kw.arg == field]
                slot = slots.get(getattr(node.func, "id", None))
                if slot is not None and any(
                    isinstance(a, ast.Starred) for a in node.args[: slot + 1]
                ):
                    values.append(node)
                elif slot is not None and len(node.args) > slot:
                    values.append(node.args[slot])
                elif slot is not None and any(kw.arg is None for kw in node.keywords):
                    values.append(node)
            elif (
                isinstance(node, ast.AnnAssign)
                and getattr(node.target, "id", None) == field
                and getattr(self.parent.get(node), "name", None) in slots
                and node.value is not None
            ):
                values = [node.value]  # the field's default
            elif isinstance(node, ast.Assign):
                values = [
                    node.value
                    for t in node.targets
                    if getattr(t, "attr", None) == field
                ]
            for value in values:
                found |= self._method_names(value, self._enclosing(node, _SCOPES), seen)
        return found


def _scan(path: Path) -> ClientScan:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return ClientScan(tree, path.relative_to(PACKAGE_DIR).as_posix())


def protocol_classes(tree: ast.Module) -> list[ast.ClassDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(isinstance(b, ast.Name) and b.id == "Protocol" for b in node.bases)
    ]


def protocol_methods(cls: ast.ClassDef) -> set[str]:
    return {
        n.name
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_exemption_tables_name_real_services() -> None:
    assert SERVICE_FILES, "no server_*_service.py modules found"
    assert set(EXEMPT) <= {p.name for p in SERVICE_FILES}
    assert HOLDS_NO_CLIENT <= set(EXEMPT)
    paths = {p.relative_to(PACKAGE_DIR).as_posix() for p in SERVICE_FILES}
    assert set(ABSENT_PROBES) <= paths


@pytest.mark.parametrize("path", CLIENT_SERVICES, ids=lambda p: p.name)
def test_service_protocol_lists_exactly_its_client_calls(path: Path) -> None:
    scan = _scan(path)
    assert not scan.unresolved, (
        f"{path.name}: the scan cannot follow the client here; make each site "
        f"statically resolvable: {sorted(scan.unresolved)}"
    )
    protocols = protocol_classes(scan.tree)
    assert len(protocols) == 1, f"{path.name} needs exactly one client Protocol"
    listed = protocol_methods(protocols[0])
    assert not scan.used - listed, (
        f"{path.name} calls client methods its Protocol does not list: "
        f"{sorted(scan.used - listed)}"
    )
    assert not listed - scan.used, (
        f"{path.name} Protocol lists client methods the service never calls: "
        f"{sorted(listed - scan.used)}"
    )


@pytest.mark.parametrize("path", EXEMPT_SERVICES, ids=lambda p: p.name)
def test_exempt_service_reaches_no_tldw_api_client(path: Path) -> None:
    scan = _scan(path)
    assert not protocol_classes(scan.tree), (
        f"{path.name} is exempt but declares a client Protocol"
    )
    identifiers = {
        getattr(node, field, None)
        for node in ast.walk(scan.tree)
        for field in ("id", "attr", "name")
    }
    reached = identifiers & {"TLDWAPIClient", "_require_client"}
    assert not reached, (
        f"{path.name} is exempt ({EXEMPT[path.name]}) but names {sorted(reached)}"
    )
    if path.name in HOLDS_NO_CLIENT:
        assert not scan.used and not scan.unresolved, (
            f"{path.name} is exempt ({EXEMPT[path.name]}) but uses a client: "
            f"{sorted(scan.used | scan.unresolved)}"
        )


@pytest.mark.parametrize("path", CLIENT_SERVICES, ids=lambda p: p.name)
def test_protocol_lists_only_methods_the_client_defines(path: Path) -> None:
    from tldw_chatbook.tldw_api import TLDWAPIClient

    (protocol,) = protocol_classes(ast.parse(path.read_text(encoding="utf-8")))
    unknown = {m for m in protocol_methods(protocol) if not hasattr(TLDWAPIClient, m)}
    assert not unknown, (
        f"{path.name} Protocol lists methods TLDWAPIClient lacks: {sorted(unknown)}"
    )
