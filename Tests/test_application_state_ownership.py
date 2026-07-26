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
SOURCE_STATE_PATH = (
    PROJECT_ROOT / "tldw_chatbook" / "runtime_policy" / "source_state.py"
)
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
RUNTIME_STORE_TYPE = "RuntimeSourceStateStore"
RUNTIME_STORE_OWNER_PATHS = {
    BOOTSTRAP_PATH,
    SOURCE_STATE_PATH,
}
APP_ALIAS = "app"
CONTEXT_ALIAS = "context"
STORE_ALIAS = "store"
STORE_TYPE_ALIAS = "store_type"


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


def _parse_snippet(source: str) -> ast.Module:
    return ast.parse(source, filename="tldw_chatbook/ownership_guard_probe.py")


def _direct_owner_store_save_receiver(statement: ast.stmt) -> ast.Attribute | None:
    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "save"
        and isinstance(statement.value.func.value, ast.Attribute)
        and _chain(statement.value.func.value) == "self._store"
    ):
        return None
    return statement.value.func.value


class OwnScopeYieldVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.found = False

    def visit_Yield(self, node: ast.Yield) -> None:
        self.found = True

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


class ScopedVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.classes: list[str] = []
        self.functions: list[str] = []
        self.alias_scopes: list[dict[str, frozenset[str]]] = [{}]
        self.alias_scope_kinds = ["module"]

    @property
    def class_name(self) -> str | None:
        return self.classes[-1] if self.classes else None

    @property
    def function_name(self) -> str | None:
        return self.functions[-1] if self.functions else None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self.alias_scopes.append({})
        self.alias_scope_kinds.append("class")
        try:
            self.generic_visit(node)
        finally:
            self.alias_scope_kinds.pop()
            self.alias_scopes.pop()
            self.classes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_definition(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_argument_expressions(node.args)
        self.alias_scopes.append({})
        self.alias_scope_kinds.append("function")
        self._bind_arguments(node.args)
        try:
            self.visit(node.body)
        finally:
            self.alias_scope_kinds.pop()
            self.alias_scopes.pop()

    def _visit_function_definition(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_argument_expressions(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        for type_parameter in getattr(node, "type_params", ()):
            self.visit(type_parameter)

        self.functions.append(node.name)
        self.alias_scopes.append({})
        self.alias_scope_kinds.append("function")
        self._bind_arguments(node.args)
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self.alias_scope_kinds.pop()
            self.alias_scopes.pop()
            self.functions.pop()

    def _visit_argument_expressions(self, arguments: ast.arguments) -> None:
        for default in arguments.defaults:
            self.visit(default)
        for default in arguments.kw_defaults:
            if default is not None:
                self.visit(default)
        for parameter in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            if parameter.annotation is not None:
                self.visit(parameter.annotation)
        if arguments.vararg is not None and arguments.vararg.annotation is not None:
            self.visit(arguments.vararg.annotation)
        if arguments.kwarg is not None and arguments.kwarg.annotation is not None:
            self.visit(arguments.kwarg.annotation)

    def visit_Assign(self, node: ast.Assign) -> None:
        alias_kinds = self._expression_alias_kinds(node.value)
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

        if all(isinstance(target, ast.Name) for target in node.targets):
            for target in node.targets:
                self._bind_target(target, alias_kinds)
        else:
            for target in node.targets:
                self._bind_target(target, frozenset())

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        alias_kinds = (
            self._expression_alias_kinds(node.value)
            if node.value is not None and isinstance(node.target, ast.Name)
            else frozenset()
        )
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)
        self._bind_target(node.target, alias_kinds)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        alias_kinds = self._expression_alias_kinds(node.value)
        self.visit(node.value)
        self.visit(node.target)
        self._bind_target(node.target, alias_kinds)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        entry = dict(self.alias_scopes[-1])
        body_environment = self._visit_statements_from_environment(node.body, entry)
        else_environment = (
            self._visit_statements_from_environment(node.orelse, entry)
            if node.orelse
            else entry
        )
        self.alias_scopes[-1] = self._join_environments(
            entry,
            body_environment,
            else_environment,
        )

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        entry = dict(self.alias_scopes[-1])
        body_environment = self._visit_statements_from_environment(node.body, entry)
        completed_body_environment = self._visit_statements_from_environment(
            node.orelse,
            body_environment,
        )
        zero_iteration_environment = self._visit_statements_from_environment(
            node.orelse,
            entry,
        )
        self.alias_scopes[-1] = self._join_environments(
            entry,
            body_environment,
            completed_body_environment,
            zero_iteration_environment,
        )

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self.alias_scopes[-1][node.name] = frozenset()
        for statement in node.body:
            self.visit(statement)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_try(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._visit_try(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, [node.key, node.value])

    def _bind_arguments(self, arguments: ast.arguments) -> None:
        parameters = [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ]
        if arguments.vararg is not None:
            parameters.append(arguments.vararg)
        if arguments.kwarg is not None:
            parameters.append(arguments.kwarg)
        for parameter in parameters:
            self.alias_scopes[-1][parameter.arg] = self._seed_alias_kinds(parameter.arg)

    def _bind_target(
        self,
        target: ast.AST,
        alias_kinds: frozenset[str],
    ) -> None:
        if isinstance(target, ast.Name):
            self.alias_scopes[-1][target.id] = alias_kinds
            return
        if isinstance(target, ast.Starred):
            self._bind_target(target.value, frozenset())
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_target(element, frozenset())

    def _visit_loop(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        entry = dict(self.alias_scopes[-1])
        original = self.alias_scopes[-1]
        self.alias_scopes[-1] = dict(entry)
        try:
            self.visit(node.target)
            self._bind_target(node.target, frozenset())
            for statement in node.body:
                self.visit(statement)
            body_environment = dict(self.alias_scopes[-1])
        finally:
            self.alias_scopes[-1] = original
        completed_body_environment = self._visit_statements_from_environment(
            node.orelse,
            body_environment,
        )
        zero_iteration_environment = self._visit_statements_from_environment(
            node.orelse,
            entry,
        )
        self.alias_scopes[-1] = self._join_environments(
            entry,
            body_environment,
            completed_body_environment,
            zero_iteration_environment,
        )

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
                self._bind_target(item.optional_vars, frozenset())
        for statement in node.body:
            self.visit(statement)

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        result_expressions: list[ast.expr],
    ) -> None:
        self.alias_scopes.append({})
        self.alias_scope_kinds.append("function")
        try:
            for generator in generators:
                self.visit(generator.iter)
                self.visit(generator.target)
                self._bind_target(generator.target, frozenset())
                for condition in generator.ifs:
                    self.visit(condition)
            for expression in result_expressions:
                self.visit(expression)
        finally:
            self.alias_scope_kinds.pop()
            self.alias_scopes.pop()

    def _visit_try(self, node: ast.Try | ast.TryStar) -> None:
        entry = dict(self.alias_scopes[-1])
        original = self.alias_scopes[-1]
        self.alias_scopes[-1] = dict(entry)
        body_snapshots = [entry]
        try:
            for statement in node.body:
                self.visit(statement)
                body_snapshots.append(dict(self.alias_scopes[-1]))
            body_environment = dict(self.alias_scopes[-1])
        finally:
            self.alias_scopes[-1] = original

        normal_environment = self._visit_statements_from_environment(
            node.orelse,
            body_environment,
        )
        handler_entry = self._join_environments(entry, *body_snapshots)
        handler_environments = [
            self._visit_statements_from_environment([handler], handler_entry)
            for handler in node.handlers
        ]
        continuing_environment = self._join_environments(
            entry,
            normal_environment,
            *handler_environments,
        )
        self.alias_scopes[-1] = self._visit_statements_from_environment(
            node.finalbody,
            continuing_environment,
        )

    def _visit_statements_from_environment(
        self,
        statements: list[ast.AST],
        environment: dict[str, frozenset[str]],
    ) -> dict[str, frozenset[str]]:
        original = self.alias_scopes[-1]
        self.alias_scopes[-1] = dict(environment)
        try:
            for statement in statements:
                self.visit(statement)
            return dict(self.alias_scopes[-1])
        finally:
            self.alias_scopes[-1] = original

    def _join_environments(
        self,
        entry: dict[str, frozenset[str]],
        *alternatives: dict[str, frozenset[str]],
    ) -> dict[str, frozenset[str]]:
        names = set(entry)
        for alternative in alternatives:
            names.update(alternative)
        return {
            name: frozenset().union(
                *(
                    self._alias_kinds_in_environment(name, alternative)
                    for alternative in alternatives
                )
            )
            for name in names
        }

    def _alias_kinds_in_environment(
        self,
        name: str,
        environment: dict[str, frozenset[str]],
    ) -> frozenset[str]:
        if name in environment:
            return environment[name]
        return self._outer_or_seed_alias_kinds(name)

    def _expression_alias_kinds(self, node: ast.AST) -> frozenset[str]:
        if isinstance(node, ast.NamedExpr):
            return self._expression_alias_kinds(node.value)

        if isinstance(node, ast.Name):
            if node.id in self.alias_scopes[-1]:
                return self.alias_scopes[-1][node.id]
            return self._outer_or_seed_alias_kinds(node.id)

        if isinstance(node, ast.Attribute):
            return self._attribute_alias_kinds(
                self._expression_alias_kinds(node.value),
                node.attr,
                chain=_chain(node),
            )

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and node.args
        ):
            attribute = _constant_attribute_name(node)
            if attribute is None:
                return frozenset()
            return self._attribute_alias_kinds(
                self._expression_alias_kinds(node.args[0]),
                attribute,
                chain=_chain(node),
            )

        if isinstance(
            node, ast.Call
        ) and STORE_TYPE_ALIAS in self._expression_alias_kinds(node.func):
            return frozenset({STORE_ALIAS})
        return frozenset()

    def _outer_or_seed_alias_kinds(self, name: str) -> frozenset[str]:
        for index in range(len(self.alias_scopes) - 2, -1, -1):
            scope = self.alias_scopes[index]
            if (
                self.alias_scope_kinds[index] == "class"
                and "function" in (self.alias_scope_kinds[index + 1 :])
            ):
                continue
            if name in scope:
                return scope[name]
        return self._seed_alias_kinds(name)

    def _seed_alias_kinds(self, name: str) -> frozenset[str]:
        if name in {"app", "app_instance"}:
            return frozenset({APP_ALIAS})
        if name in CONTEXT_RECEIVER_NAMES:
            return frozenset({CONTEXT_ALIAS})
        if name in {"runtime_store", "runtime_policy_store"}:
            return frozenset({STORE_ALIAS})
        if name == RUNTIME_STORE_TYPE:
            return frozenset({STORE_TYPE_ALIAS})
        if self.path == APP_PATH and name == "self":
            return frozenset({APP_ALIAS})
        return frozenset()

    @staticmethod
    def _attribute_alias_kinds(
        base_kinds: frozenset[str],
        attribute: str,
        *,
        chain: str,
    ) -> frozenset[str]:
        if attribute == RUNTIME_STORE_TYPE:
            return frozenset({STORE_TYPE_ALIAS})
        if attribute == "app" and chain == "self.app":
            return frozenset({APP_ALIAS})
        if attribute == "app_instance":
            return frozenset({APP_ALIAS})
        if attribute in CONTEXT_RECEIVER_NAMES:
            return frozenset({CONTEXT_ALIAS})
        if chain == "self._store":
            return frozenset({STORE_ALIAS})
        if attribute in {"store", "_store"} and CONTEXT_ALIAS in base_kinds:
            return frozenset({STORE_ALIAS})
        if attribute in {"runtime_store", "runtime_policy_store"}:
            return frozenset({STORE_ALIAS})
        return frozenset()


class ProjectionWriteVisitor(ScopedVisitor):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.writes: list[str] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr in PROJECTION_NAMES
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and APP_ALIAS in self._expression_alias_kinds(node.value)
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
            if name in PROJECTION_NAMES and APP_ALIAS in self._expression_alias_kinds(
                node.args[0]
            ):
                self.writes.append(self._describe(node, name))
        self.generic_visit(node)

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
        receiver_kinds = self._expression_alias_kinds(node.value)
        if (
            node.attr == RUNTIME_STORE_TYPE
            and isinstance(node.ctx, ast.Load)
            and self.path not in RUNTIME_STORE_OWNER_PATHS
        ):
            self._record(node, "runtime source-state store reference")
        if node.attr in {"store", "_store"} and CONTEXT_ALIAS in receiver_kinds:
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
        if node.attr == "persist" and CONTEXT_ALIAS in receiver_kinds:
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
                CONTEXT_ALIAS in receiver_kinds
                or (self.class_name == "RuntimePolicyContext" and receiver == "self")
            )
        ):
            self._record(node, "context-state mutation")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        function_kinds = self._expression_alias_kinds(node.func)
        if self.path not in RUNTIME_STORE_OWNER_PATHS:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and _constant_attribute_name(node) == RUNTIME_STORE_TYPE
            ):
                self._record(node, "runtime source-state store reference")
            if STORE_TYPE_ALIAS in function_kinds:
                self._record(node, "runtime source-state store construction")
        if isinstance(node.func, ast.Attribute):
            receiver = _chain(node.func.value)
            receiver_kinds = self._expression_alias_kinds(node.func.value)
            direct_owner_store_save = (
                node.func.attr == "save"
                and isinstance(node.func.value, ast.Attribute)
                and receiver == "self._store"
                and id(node.func.value) in self.allowed_owner_store_loads
            )
            if node.func.attr == "save" and STORE_ALIAS in receiver_kinds:
                if not direct_owner_store_save:
                    self._record(node, "runtime store save outside commit")
        if isinstance(node.func, ast.Name) and node.args:
            attribute_name = _constant_attribute_name(node)
            target = node.args[0]
            context_target = CONTEXT_ALIAS in self._expression_alias_kinds(target) or (
                self.class_name == "RuntimePolicyContext" and _chain(target) == "self"
            )
            if (
                node.func.id in {"getattr", "setattr", "delattr"}
                and attribute_name in CONTEXT_SENSITIVE_ATTRIBUTES
                and context_target
            ):
                self._record(node, "dynamic context-sensitive access")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            node.id == RUNTIME_STORE_TYPE
            and isinstance(node.ctx, ast.Load)
            and self.path not in RUNTIME_STORE_OWNER_PATHS
        ):
            self._record(node, "runtime source-state store reference")

    def visit_Import(self, node: ast.Import) -> None:
        if self.path not in RUNTIME_STORE_OWNER_PATHS and any(
            alias.name.rsplit(".", 1)[-1] == RUNTIME_STORE_TYPE for alias in node.names
        ):
            self._record(node, "runtime source-state store reference")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.path not in RUNTIME_STORE_OWNER_PATHS and any(
            alias.name == RUNTIME_STORE_TYPE for alias in node.names
        ):
            self._record(node, "runtime source-state store reference")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.class_name == "RuntimePolicyContext" and node.name in {
            "persist",
            "store",
        }:
            self._record(node, "public persistence escape")
        is_owner_commit = (
            self.class_name == "RuntimePolicyContext"
            and self.function_name is None
            and node.name == "commit_state"
        )
        if not is_owner_commit:
            super().visit_FunctionDef(node)
            return

        receivers = [
            receiver
            for statement in node.body
            if (receiver := _direct_owner_store_save_receiver(statement)) is not None
        ]
        yield_visitor = OwnScopeYieldVisitor()
        for statement in node.body:
            yield_visitor.visit(statement)

        authorized_receiver = None
        if len(receivers) == 1 and not yield_visitor.found:
            authorized_receiver = receivers[0]
            self.allowed_owner_store_loads.add(id(authorized_receiver))
        else:
            self._record(node, "invalid commit persistence shape")

        try:
            super().visit_FunctionDef(node)
        finally:
            if authorized_receiver is not None:
                self.allowed_owner_store_loads.discard(id(authorized_receiver))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if (
            self.class_name == "RuntimePolicyContext"
            and self.function_name is None
            and node.name == "commit_state"
        ):
            self._record(node, "async commit persistence shape")
        super().visit_AsyncFunctionDef(node)

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


def test_projection_guard_follows_simple_app_alias() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ProjectionWriteVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
application = app
application.runtime_backend = "server"
"""
        )
    )

    assert len(visitor.writes) == 1


def test_projection_guard_follows_alias_chain_and_forgets_reassignment() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ProjectionWriteVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
primary = application = app
secondary = application
secondary.runtime_backend = "server"
secondary = widget
secondary.active_server_id = "unrelated"
"""
        )
    )

    assert len(visitor.writes) == 1
    assert visitor.writes[0].endswith(":runtime_backend")


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
authority = runtime_context
authority.state = candidate
""",
        """
authority = runtime_context
authority._store.save(candidate)
""",
        """
authority = app.runtime_policy
authority.state = candidate
""",
        """
authority = getattr(app, "runtime_policy")
store = getattr(authority, "_store")
store.save(candidate)
""",
    ],
)
def test_context_guard_follows_simple_context_and_store_aliases(source: str) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(_parse_snippet(source))

    assert visitor.violations


def test_context_guard_keeps_aliases_scope_local_and_forgets_reassignment() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
def leaking_scope():
    first = runtime_context
    authority = first
    authority.state = candidate

def unrelated_scope():
    authority = widget
    authority.state = candidate

authority = runtime_context
authority = widget
authority.state = candidate
"""
        )
    )

    assert len(visitor.violations) == 1


def test_context_aliases_from_class_body_do_not_leak_into_method_scope() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
class UnrelatedDomain:
    authority = runtime_context

    def update(self):
        authority.state = candidate
"""
        )
    )

    assert visitor.violations == []


@pytest.mark.parametrize(
    "definition",
    [
        "def f(authority):\n    authority.state = candidate",
        "def f(authority, /):\n    authority.state = candidate",
        "def f(*authority):\n    authority.state = candidate",
        "def f(*, authority):\n    authority.state = candidate",
        "def f(**authority):\n    authority.state = candidate",
        "async def f(authority):\n    authority.state = candidate",
    ],
)
def test_context_aliases_are_shadowed_by_all_function_parameter_kinds(
    definition: str,
) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            f"""
authority = runtime_context
{definition}
"""
        )
    )

    assert visitor.violations == []


@pytest.mark.parametrize("parameter", ["app", "app_instance"])
def test_projection_guard_seeds_recognized_app_parameter_names(
    parameter: str,
) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ProjectionWriteVisitor(path)

    visitor.visit(
        _parse_snippet(
            f"""
def update({parameter}):
    {parameter}.runtime_backend = "server"
"""
        )
    )

    assert visitor.writes


@pytest.mark.parametrize(
    ("parameter", "operation"),
    [
        ("ctx", "ctx.state = candidate"),
        ("runtime_context", "runtime_context.state = candidate"),
        ("runtime_store", "runtime_store.save(candidate)"),
        ("runtime_policy_store", "runtime_policy_store.save(candidate)"),
        ("RuntimeSourceStateStore", "RuntimeSourceStateStore(path)"),
    ],
)
def test_context_guard_seeds_recognized_context_and_store_parameter_names(
    parameter: str,
    operation: str,
) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            f"""
def update({parameter}):
    {operation}
"""
        )
    )

    assert visitor.violations


def test_context_alias_is_shadowed_by_lambda_parameter() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
authority = runtime_context
callback = lambda authority: authority._store.save(candidate)
"""
        )
    )

    assert visitor.violations == []


@pytest.mark.parametrize(
    "definition",
    [
        """
def update(authority=getattr(authority, "persist")):
    authority.state = candidate
""",
        """
callback = lambda authority=getattr(authority, "persist"): authority.state
""",
    ],
)
def test_parameter_defaults_use_outer_alias_before_shadowing(
    definition: str,
) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            f"""
authority = runtime_context
{definition}
"""
        )
    )

    assert len(visitor.violations) == 1
    assert visitor.violations[0].endswith(":dynamic context-sensitive access")


def test_context_alias_survives_if_without_else_at_branch_join() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
authority = runtime_context
if condition:
    authority = widget
authority.state = candidate
"""
        )
    )

    assert visitor.violations


def test_context_alias_is_forgotten_when_both_if_branches_are_unrelated() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
authority = runtime_context
if condition:
    authority = widget
else:
    authority = other_widget
authority.state = candidate
"""
        )
    )

    assert visitor.violations == []


@pytest.mark.parametrize(
    "source",
    [
        """
authority = runtime_context
authority: object = widget
authority.state = candidate
""",
        """
authority = runtime_context
(authority := widget)
authority.state = candidate
""",
        """
authority = runtime_context
with manager as authority:
    authority.state = candidate
""",
        """
authority = runtime_context
try:
    operation()
except Exception as authority:
    authority.state = candidate
""",
        """
authority = runtime_context
authority, other = widget, item
authority.state = candidate
""",
    ],
)
def test_context_alias_bindings_tombstone_unrelated_values(source: str) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(_parse_snippet(source))

    assert visitor.violations == []


def test_exception_target_tombstone_joins_with_normal_path() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
authority = runtime_context
try:
    operation()
except Exception as authority:
    pass
authority.state = candidate
"""
        )
    )

    assert visitor.violations


@pytest.mark.parametrize(
    "source",
    [
        """
authority: object = runtime_context
authority.state = candidate
""",
        """
(authority := runtime_context)
authority.state = candidate
""",
    ],
)
def test_context_alias_bindings_propagate_recognized_values(source: str) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(_parse_snippet(source))

    assert visitor.violations


@pytest.mark.parametrize(
    "source",
    [
        """
authority = runtime_context
for item in items:
    authority = widget
authority.state = candidate
""",
        """
authority = runtime_context
while condition:
    authority = widget
authority.state = candidate
""",
        """
async def check():
    authority = runtime_context
    async for item in items:
        authority = widget
    authority.state = candidate
""",
    ],
)
def test_context_alias_survives_possible_zero_iteration_loop(source: str) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(_parse_snippet(source))

    assert visitor.violations


def test_loop_target_is_unknown_inside_loop_but_outer_alias_survives() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
authority = runtime_context
for authority in items:
    getattr(authority, "persist")
authority.state = candidate
"""
        )
    )

    assert len(visitor.violations) == 1
    assert visitor.violations[0].endswith(":context-state mutation")


def test_comprehension_target_shadows_without_leaking_context_alias() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
authority = runtime_context
callbacks = [getattr(authority, "persist") for authority in widgets]
authority.state = candidate
"""
        )
    )

    assert len(visitor.violations) == 1
    assert visitor.violations[0].endswith(":context-state mutation")


@pytest.mark.parametrize(
    "source",
    [
        """
store = RuntimeSourceStateStore(path)
store.save(candidate)
""",
        """
store = runtime_policy.source_state.RuntimeSourceStateStore(path)
store.save(candidate)
""",
        "factory = RuntimeSourceStateStore",
    ],
)
def test_context_guard_rejects_runtime_store_type_references_and_aliases(
    source: str,
) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(_parse_snippet(source))

    assert visitor.violations


@pytest.mark.parametrize(
    "source",
    [
        'factory = getattr(source_state, "RuntimeSourceStateStore")',
        'getattr(source_state, "RuntimeSourceStateStore")(path)',
        (
            'getattr(getattr(runtime_policy, "source_state"), '
            '"RuntimeSourceStateStore")(path)'
        ),
        ('register_store(getattr(source_state, "RuntimeSourceStateStore")(path))'),
    ],
)
def test_context_guard_rejects_constant_getattr_runtime_store_access(
    source: str,
) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(_parse_snippet(source))

    assert visitor.violations


@pytest.mark.parametrize("path", [SOURCE_STATE_PATH, BOOTSTRAP_PATH])
def test_runtime_store_owner_modules_may_use_constant_getattr(path: Path) -> None:
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet('store = getattr(source_state, "RuntimeSourceStateStore")(path)')
    )

    assert visitor.violations == []


def test_context_guard_ignores_unrelated_constant_getattr_construction() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(_parse_snippet('getattr(source_state, "WidgetStore")(path)'))

    assert visitor.violations == []


def test_context_guard_rejects_non_owner_self_store_save() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
class PersistenceEscape:
    def save(self):
        self._store.save(candidate)
"""
        )
    )

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


@pytest.mark.parametrize(
    "body",
    [
        "return lambda: self._store.save(candidate)",
        "callback = lambda: self._store.save(candidate)",
        "return (self._store.save(item) for item in items)",
        "yield self._store.save(candidate)",
        "self._store.save(candidate)\n        yield candidate",
        "self._store.save(candidate)\n        yield from candidates",
    ],
)
def test_context_guard_rejects_deferred_and_generator_commit_saves(
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


@pytest.mark.parametrize("keyword", ["def", "async def"])
def test_context_guard_does_not_extend_commit_save_permission_to_nested_functions(
    keyword: str,
) -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            f"""
class RuntimePolicyContext:
    def commit_state(self):
        self._store.save(candidate)

        {keyword} deferred():
            self._store.save(candidate)
"""
        )
    )

    assert visitor.violations


@pytest.mark.parametrize(
    "body",
    [
        "return True",
        "self._store.save(first)\n        self._store.save(second)",
        "return self._store.save(candidate)",
        "result = self._store.save(candidate)",
        "if ready:\n            self._store.save(candidate)",
        "return [self._store.save(item) for item in items]",
    ],
)
def test_context_guard_requires_one_top_level_immediate_commit_save(
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


def test_context_guard_rejects_async_commit_state() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
class RuntimePolicyContext:
    async def commit_state(self):
        self._store.save(candidate)
"""
        )
    )

    assert visitor.violations


def test_context_guard_ignores_yield_in_nested_definition() -> None:
    path = PROJECT_ROOT / "tldw_chatbook" / "ownership_guard_probe.py"
    visitor = ContextOwnershipVisitor(path)

    visitor.visit(
        _parse_snippet(
            """
class RuntimePolicyContext:
    def commit_state(self):
        def nested_generator():
            yield candidate

        self._store.save(candidate)
"""
        )
    )

    assert visitor.violations == []


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
authority = widget
authority.state = candidate
store = WidgetStore()
alias = store
alias.save(candidate)
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


def test_runtime_source_state_store_references_are_confined_to_owner_modules() -> None:
    violations: list[str] = []
    for path in (PROJECT_ROOT / "tldw_chatbook").rglob("*.py"):
        visitor = ContextOwnershipVisitor(path)
        visitor.visit(_parse(path))
        violations.extend(
            violation
            for violation in visitor.violations
            if "runtime source-state store reference" in violation
        )

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
