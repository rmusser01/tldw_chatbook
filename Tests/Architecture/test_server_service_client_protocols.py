"""Each ``server_*_service.py`` types its client as a narrow Protocol (TASK-36953.1).

A service lists, in a ``typing.Protocol`` class in its own module, exactly the
``TLDWAPIClient`` methods it uses. This AST scan fails when

* a service touches a client method its Protocol does not list,
* a Protocol lists a method the service never touches, or
* a Protocol lists a method ``TLDWAPIClient`` does not define.

"The client" in a service is ``self.client``, ``self._require_client()``, a
parameter named ``client``, or a name assigned from any of these. A use is an
attribute read on the client or ``getattr(<client>, name)``. When ``name`` is
not a string literal, the method names are the string literals the module binds
to it: ``<name>="..."`` keyword arguments, and the matching positional slot of a
same-module dataclass with a field of that name (Kanban's
``KanbanOperationSpec`` rows). A dynamic name that resolves to nothing fails.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "tldw_chatbook"

# Services with no TLDWAPIClient in their call path.
EXEMPT = {
    "server_notifications_scope_service.py": "wraps ServerNotificationsService, no client",
    "server_runtime_scope_service.py": "wraps ServerRuntimeService, no client",
    "server_unified_service.py": "drives MCPUnifiedClient (tldw_api/mcp_unified_client.py), not TLDWAPIClient",
}

# Dead getattr fallbacks that probe a method TLDWAPIClient never had.
ABSENT_PROBES = {"run_research_websearch"}

SERVICE_FILES = sorted(PACKAGE_DIR.rglob("server_*_service.py"))


def _is_client_expr(node: ast.AST, aliases: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "client" or node.id in aliases
    if isinstance(node, ast.Attribute):
        return (
            node.attr == "client"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_require_client"
    )


def _strings_bound_to(tree: ast.AST, key: str) -> set[str]:
    """String literals the module passes as ``key``: by keyword, or by dataclass slot."""
    found: set[str] = set()
    slots: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            fields = [
                s.target.id
                for s in node.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            ]
            if key in fields:
                slots[node.name] = fields.index(key)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        values = [kw.value for kw in node.keywords if kw.arg == key]
        slot = slots.get(node.func.id) if isinstance(node.func, ast.Name) else None
        if slot is not None and len(node.args) > slot:
            values.append(node.args[slot])
        found |= {
            v.value
            for v in values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        }
    return found


def used_client_methods(tree: ast.AST) -> set[str]:
    """Client method names a service module touches."""
    aliases = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and _is_client_expr(node.value, set())
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_client_expr(node.value, aliases):
            used.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and _is_client_expr(node.args[0], aliases)
        ):
            name = node.args[1]
            if isinstance(name, ast.Constant):
                used.add(name.value)
                continue
            key = getattr(name, "id", None) or getattr(name, "attr", None)
            resolved = _strings_bound_to(tree, key) if key else set()
            used |= resolved or {f"<unresolved getattr {ast.unparse(name)}>"}
    return used - ABSENT_PROBES


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


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_service_inventory_is_complete() -> None:
    assert SERVICE_FILES, "no server_*_service.py modules found"
    assert set(EXEMPT) <= {p.name for p in SERVICE_FILES}


@pytest.mark.parametrize("path", SERVICE_FILES, ids=lambda p: p.name)
def test_service_protocol_lists_exactly_its_client_calls(path: Path) -> None:
    tree = _parse(path)
    protocols = protocol_classes(tree)
    used = used_client_methods(tree)
    if path.name in EXEMPT:
        assert not protocols, f"{path.name} is exempt but declares a client Protocol"
        return
    assert len(protocols) == 1, f"{path.name} needs exactly one client Protocol"
    listed = protocol_methods(protocols[0])
    assert not used - listed, (
        f"{path.name} calls client methods its Protocol does not list: "
        f"{sorted(used - listed)}"
    )
    assert not listed - used, (
        f"{path.name} Protocol lists client methods the service never calls: "
        f"{sorted(listed - used)}"
    )


@pytest.mark.parametrize("path", SERVICE_FILES, ids=lambda p: p.name)
def test_protocol_lists_only_methods_the_client_defines(path: Path) -> None:
    if path.name in EXEMPT:
        pytest.skip(EXEMPT[path.name])
    from tldw_chatbook.tldw_api import TLDWAPIClient

    (protocol,) = protocol_classes(_parse(path))
    unknown = {m for m in protocol_methods(protocol) if not hasattr(TLDWAPIClient, m)}
    assert not unknown, (
        f"{path.name} Protocol lists methods TLDWAPIClient lacks: {sorted(unknown)}"
    )
