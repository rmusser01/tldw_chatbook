from __future__ import annotations

import ast
from pathlib import Path
import warnings

import pytest

from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.state import (
    AppState,
    ChatState,
    NavigationState,
    NotesState,
    UIState,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "tldw_chatbook" / "app.py"
BOOTSTRAP_PATH = PROJECT_ROOT / "tldw_chatbook" / "runtime_policy" / "bootstrap.py"
MEDIA_INGEST_PATH = (
    PROJECT_ROOT / "tldw_chatbook" / "UI" / "Screens" / "media_ingest_screen.py"
)
STUDY_PATH = PROJECT_ROOT / "tldw_chatbook" / "UI" / "Screens" / "study_screen.py"
PROJECTION_NAMES = {
    "current_runtime_backend",
    "runtime_backend",
    "active_server_id",
}
CONTEXT_RECEIVER_NAMES = {
    "ctx",
    "context",
    "runtime_context",
    "runtime_policy",
    "runtime_policy_context",
}
CONTEXT_SENSITIVE_ATTRIBUTES = {
    "state",
    "persist",
    "store",
    "_store",
}


def _parse(path: Path) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _chain(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _chain(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and node.args
    ):
        prefix = _chain(node.args[0])
        attribute = _constant_attribute_name(node)
        if prefix and attribute is not None:
            return f"{prefix}.{attribute}"
    return ""


def _constant_attribute_name(call: ast.Call) -> str | None:
    if len(call.args) < 2:
        return None
    name = call.args[1]
    return (
        name.value
        if isinstance(name, ast.Constant) and isinstance(name.value, str)
        else None
    )


def _has_final_component(chain: str, names: set[str]) -> bool:
    return bool(chain) and chain.rsplit(".", 1)[-1] in names


def _is_context_receiver(chain: str) -> bool:
    return _has_final_component(chain, CONTEXT_RECEIVER_NAMES)


def _parse_snippet(source: str) -> ast.Module:
    return ast.parse(source, filename="tldw_chatbook/ownership_guard_probe.py")


class ScopedVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.classes: list[str] = []
        self.functions: list[str] = []

    @property
    def class_name(self) -> str | None:
        return self.classes[-1] if self.classes else None

    @property
    def function_name(self) -> str | None:
        return self.functions[-1] if self.functions else None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()


class ProjectionWriteVisitor(ScopedVisitor):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.writes: list[str] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr in PROJECTION_NAMES
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and self._is_app_projection_base(_chain(node.value))
        ):
            self.writes.append(self._describe(node, node.attr))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"setattr", "delattr"}
            and node.args
        ):
            name = _constant_attribute_name(node)
            if name in PROJECTION_NAMES and self._is_app_projection_base(
                _chain(node.args[0])
            ):
                self.writes.append(self._describe(node, name))
        self.generic_visit(node)

    def _is_app_projection_base(self, base: str) -> bool:
        if self.path == APP_PATH and base == "self":
            return True
        return base in {"app", "self.app"} or _has_final_component(
            base, {"app_instance"}
        )

    def _describe(self, node: ast.AST, name: str) -> str:
        return (
            f"{self.path.relative_to(PROJECT_ROOT)}:{node.lineno}:"
            f"{self.function_name}:{name}"
        )


class ContextOwnershipVisitor(ScopedVisitor):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.violations: list[str] = []
        self.allowed_owner_store_loads: set[int] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        receiver = _chain(node.value)
        if node.attr in {"store", "_store"} and _is_context_receiver(receiver):
            self._record(node, "context-store access")
        if (
            node.attr == "_store"
            and self.class_name == "RuntimePolicyContext"
            and receiver == "self"
        ):
            allowed = (
                self.function_name == "__init__" and isinstance(node.ctx, ast.Store)
            ) or id(node) in self.allowed_owner_store_loads
            if not allowed:
                self._record(node, "owner private-store access outside commit")
        if node.attr == "persist" and _is_context_receiver(receiver):
            self._record(node, "removed persist access")
        if (
            node.attr == "store"
            and self.class_name == "RuntimePolicyContext"
            and receiver == "self"
        ):
            self._record(node, "public-store escape")
        if (
            node.attr == "state"
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and (
                _is_context_receiver(receiver)
                or (self.class_name == "RuntimePolicyContext" and receiver == "self")
            )
        ):
            self._record(node, "context-state mutation")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            receiver = _chain(node.func.value)
            direct_owner_store_save = (
                node.func.attr == "save"
                and isinstance(node.func.value, ast.Attribute)
                and receiver == "self._store"
                and self.class_name == "RuntimePolicyContext"
                and self.function_name == "commit_state"
            )
            if direct_owner_store_save:
                self.allowed_owner_store_loads.add(id(node.func.value))
            if node.func.attr == "save" and (
                receiver == "self._store"
                or _has_final_component(
                    receiver,
                    {"runtime_store", "runtime_policy_store"},
                )
            ):
                if not direct_owner_store_save:
                    self._record(node, "runtime store save outside commit")
        if isinstance(node.func, ast.Name) and node.args:
            attribute_name = _constant_attribute_name(node)
            target = _chain(node.args[0])
            context_target = _is_context_receiver(target) or (
                self.class_name == "RuntimePolicyContext" and target == "self"
            )
            if (
                node.func.id in {"getattr", "setattr", "delattr"}
                and attribute_name in CONTEXT_SENSITIVE_ATTRIBUTES
                and context_target
            ):
                self._record(node, "dynamic context-sensitive access")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.class_name == "RuntimePolicyContext" and node.name in {
            "persist",
            "store",
        }:
            self._record(node, "public persistence escape")
        super().visit_FunctionDef(node)

    def _record(self, node: ast.AST, kind: str) -> None:
        self.violations.append(
            f"{self.path.relative_to(PROJECT_ROOT)}:{node.lineno}:{kind}"
        )


@pytest.mark.parametrize(
    "source",
    [
        "app.current_runtime_backend = 'server'",
        "self.app.runtime_backend = 'server'",
        "setattr(app, 'active_server_id', 'server-1')",
        "delattr(self.app, 'runtime_backend')",
    ],
)
def test_projection_guard_detects_common_app_receiver_aliases(source: str) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ProjectionWriteVisitor(path)

    visitor.visit(_parse_snippet(source))

    assert visitor.writes


@pytest.mark.parametrize(
    "source",
    [
        "ctx.state = candidate",
        "ctx.persist()",
        "getattr(ctx, 'persist')()",
        "callback = ctx.persist",
    ],
)
def test_context_guard_detects_common_context_aliases(source: str) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(_parse_snippet(source))

    assert visitor.violations


@pytest.mark.parametrize(
    "source",
    [
        """
class RuntimePolicyContext:
    def leak(self, ctx):
        ctx._store.save(candidate)
""",
        """
class RuntimePolicyContext:
    def leak(self):
        getattr(self, "_store").save(candidate)
""",
    ],
)
def test_context_guard_does_not_exempt_sensitive_access_inside_owner_class(
    source: str,
) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(_parse_snippet(source))

    assert visitor.violations


@pytest.mark.parametrize(
    "source",
    [
        "setattr(getattr(app, 'runtime_policy'), 'state', candidate)",
        "getattr(getattr(owner, 'runtime_context'), '_store')",
        "callback = getattr(getattr(app, 'runtime_policy'), 'persist')",
        """
class RuntimePolicyContext:
    def leak(self):
        return self._store
""",
    ],
)
def test_context_guard_detects_nested_getattr_and_owner_store_leaks(
    source: str,
) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(_parse_snippet(source))

    assert visitor.violations


def test_context_guard_permits_required_owner_private_store_accesses() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
class RuntimePolicyContext:
    def __init__(self, store):
        self._store = store

    def commit_state(self, candidate):
        self._store.save(candidate)
"""
        )
    )

    assert visitor.violations == []


@pytest.mark.parametrize(
    "body",
    [
        "return self._store",
        "return self._store.save",
        "callback = self._store.save",
    ],
)
def test_context_guard_rejects_commit_state_store_and_bound_save_escapes(
    body: str,
) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            f"""
class RuntimePolicyContext:
    def commit_state(self):
        {body}
"""
        )
    )

    assert visitor.violations


def test_context_guard_does_not_infer_dynamic_getattr_names() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
attribute_name = choose_attribute()
getattr(ctx, attribute_name)
getattr(getattr(owner, attribute_name), "_store")
"""
        )
    )

    assert visitor.violations == []


@pytest.mark.parametrize("operation", ["getattr", "setattr", "delattr"])
@pytest.mark.parametrize("attribute", ["state", "persist", "store", "_store"])
def test_context_guard_detects_dynamic_sensitive_access(
    operation: str,
    attribute: str,
) -> None:
    args = (
        f"ctx, {attribute!r}, replacement"
        if operation == "setattr"
        else f"ctx, {attribute!r}"
    )
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(_parse_snippet(f"{operation}({args})"))

    assert visitor.violations


def test_ownership_guards_ignore_unrelated_widget_and_domain_state() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    tree = _parse_snippet(
        """
widget.state = candidate
widget.runtime_backend = backend
widget.persist()
getattr(widget, "state")
storage._store.save(candidate)
"""
    )
    projection_visitor = ProjectionWriteVisitor(path)
    context_visitor = ContextOwnershipVisitor(path)

    projection_visitor.visit(tree)
    context_visitor.visit(tree)

    assert projection_visitor.writes == []
    assert context_visitor.violations == []


def test_tldw_cli_neither_imports_nor_instantiates_app_state() -> None:
    tree = _parse(APP_PATH)
    imported = []
    constructed = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.extend(
                alias.name for alias in node.names if alias.name == "AppState"
            )
        elif isinstance(node, ast.Import):
            imported.extend(
                alias.name for alias in node.names if alias.name.endswith(".AppState")
            )
        elif isinstance(node, ast.Call) and _chain(node.func).endswith("AppState"):
            constructed.append(node.lineno)

    assert imported == []
    assert constructed == []


def test_projection_boundary_never_accesses_app_state() -> None:
    tree = _parse(BOOTSTRAP_PATH)
    boundary = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_runtime_policy_to_app"
    )
    accesses = []
    for node in ast.walk(boundary):
        if isinstance(node, ast.Attribute) and node.attr == "app_state":
            accesses.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "setattr", "delattr"}
            and _constant_attribute_name(node) == "app_state"
        ):
            accesses.append(node.lineno)

    assert accesses == []


def test_app_projection_attributes_have_one_production_writer() -> None:
    writes: list[str] = []
    for path in (PROJECT_ROOT / "tldw_chatbook").rglob("*.py"):
        visitor = ProjectionWriteVisitor(path)
        visitor.visit(_parse(path))
        writes.extend(visitor.writes)

    assert len(writes) == 3
    assert all(
        write.startswith("tldw_chatbook/runtime_policy/bootstrap.py:")
        and ":_apply_runtime_policy_to_app:" in write
        for write in writes
    )
    assert {write.rsplit(":", 1)[-1] for write in writes} == PROJECTION_NAMES


def test_runtime_policy_context_has_no_mutation_or_persistence_escape_hatch() -> None:
    violations: list[str] = []
    for path in (PROJECT_ROOT / "tldw_chatbook").rglob("*.py"):
        visitor = ContextOwnershipVisitor(path)
        visitor.visit(_parse(path))
        violations.extend(visitor.violations)

    assert violations == []


def test_legacy_state_exports_remain_serialization_compatible() -> None:
    assert all(
        state_type is not None
        for state_type in (
            AppState,
            ChatState,
            NavigationState,
            NotesState,
            UIState,
        )
    )
    original = AppState(
        runtime_source=RuntimeSourceState(
            active_source="server",
            active_server_id="server-compatible",
            server_configured=True,
        )
    )
    payload = original.to_dict()

    assert AppState.from_dict(payload).to_dict() == payload


def test_legacy_state_docs_describe_compatibility_not_live_authority() -> None:
    app_state_source = (
        PROJECT_ROOT / "tldw_chatbook" / "state" / "app_state.py"
    ).read_text(encoding="utf-8")
    package_source = (
        PROJECT_ROOT / "tldw_chatbook" / "state" / "__init__.py"
    ).read_text(encoding="utf-8")
    rendered = f"{app_state_source}\n{package_source}".lower()

    assert "single source of truth" not in rendered
    assert "centralized state" not in rendered
    assert "compatibility" in rendered
