from __future__ import annotations

import ast
from pathlib import Path
import warnings

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
RUNTIME_POLICY_ROOT = PROJECT_ROOT / "tldw_chatbook" / "runtime_policy"
PROJECTION_NAMES = {
    "current_runtime_backend",
    "runtime_backend",
    "active_server_id",
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
        if self.path == BOOTSTRAP_PATH:
            return base == "app"
        if self.path == APP_PATH:
            return base == "self"
        return base.endswith("app_instance")

    def _describe(self, node: ast.AST, name: str) -> str:
        return (
            f"{self.path.relative_to(PROJECT_ROOT)}:{node.lineno}:"
            f"{self.function_name}:{name}"
        )


class ContextOwnershipVisitor(ScopedVisitor):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.violations: list[str] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = _chain(node)
        if node.attr == "_store" and self.class_name != "RuntimePolicyContext":
            self._record(node, "private-store access")
        if (
            node.attr == "store"
            and self.class_name == "RuntimePolicyContext"
            and _chain(node.value) == "self"
        ):
            self._record(node, "public-store escape")
        if (
            node.attr == "state"
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and any(
                token in chain
                for token in ("runtime_context", "runtime_policy", "context.state")
            )
        ):
            self._record(node, "context-state mutation")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            receiver = _chain(node.func.value)
            if node.func.attr == "persist" and any(
                token in receiver
                for token in ("runtime_context", "runtime_policy", "context")
            ):
                self._record(node, "removed persist call")
            if node.func.attr == "save" and (
                receiver == "self._store" or receiver in {"runtime_store", "store"}
            ):
                if not (
                    self.class_name == "RuntimePolicyContext"
                    and self.function_name == "commit_state"
                    and receiver == "self._store"
                ):
                    self._record(node, "runtime store save outside commit")
        if isinstance(node.func, ast.Name) and node.args:
            attribute_name = _constant_attribute_name(node)
            target = _chain(node.args[0])
            context_target = any(
                token in target
                for token in ("runtime_context", "runtime_policy", "context")
            )
            if (
                node.func.id in {"setattr", "delattr"}
                and attribute_name == "state"
                and context_target
            ):
                self._record(node, "dynamic context-state mutation")
            if (
                node.func.id in {"getattr", "setattr", "delattr"}
                and attribute_name == "_store"
                and self.class_name != "RuntimePolicyContext"
            ):
                self._record(node, "dynamic private-store access")
            if (
                node.func.id == "getattr"
                and attribute_name == "persist"
                and context_target
            ):
                self._record(node, "dynamic removed persist access")
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
    for path in RUNTIME_POLICY_ROOT.glob("*.py"):
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
