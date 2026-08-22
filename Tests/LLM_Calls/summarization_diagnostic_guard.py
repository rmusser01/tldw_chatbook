"""Test-only extraction and reconciliation for summarization diagnostics."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass

from Tests.ast_shape import stable_dump
from scripts.check_persistent_diagnostic_inventory import (
    LOG_METHODS,
    _is_diagnostic_call,
    _logger_symbols,
    _scope_names,
)


_DIAGNOSTIC_SEVERITIES = {
    "critical": "critical",
    "error": "error",
    "exception": "error",
    "warning": "warning",
    "info": "info",
    "debug": "debug",
    "success": "success",
    "trace": "trace",
    "log": "log",
}


@dataclass(frozen=True)
class DiagnosticCall:
    """Stable description of one diagnostic call."""

    module: str
    qualname: str
    method: str
    event: str
    occurrence: int
    message_shape: str
    expressions: tuple[str, ...]
    captures_exception: bool
    level_expression: str | None = None

    @property
    def identity(self) -> tuple[str, str, str, int]:
        """Return the source-movement-independent identity."""
        return (self.module, self.qualname, self.event, self.occurrence)


def _literal_projection(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else ""
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value
            for value in node.values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _literal_projection(node.left)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_projection(node.left) + _literal_projection(node.right)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return _literal_projection(node.func.value)
    return ""


def _unparse_many(nodes: list[ast.AST]) -> list[str]:
    return [ast.unparse(node) for node in nodes]


def _unparse_keyword(keyword: ast.keyword) -> str:
    value = ast.unparse(keyword.value)
    return f"{keyword.arg}={value}" if keyword.arg is not None else f"**{value}"


def _first_argument_expressions(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant):
        return []
    if isinstance(node, ast.JoinedStr):
        expressions = []
        for value in node.values:
            if not isinstance(value, ast.FormattedValue):
                continue
            expressions.append(ast.unparse(value.value))
            if value.format_spec is not None:
                expressions.extend(_first_argument_expressions(value.format_spec))
        return expressions
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        values = (
            list(node.right.elts)
            if isinstance(node.right, (ast.Tuple, ast.List))
            else [node.right]
        )
        return _unparse_many(values)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return [
            *_first_argument_expressions(node.func.value),
            *_unparse_many([*node.args, *(keyword.value for keyword in node.keywords)]),
        ]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [
            *_first_argument_expressions(node.left),
            *_first_argument_expressions(node.right),
        ]
    return [ast.unparse(node)]


def _receiver_field_expressions(node: ast.AST) -> list[str]:
    expressions: list[str] = []

    def visit(current: ast.AST) -> None:
        if isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
            visit(current.func.value)
            if current.func.attr in {"bind", "opt"}:
                expressions.extend(_unparse_many(list(current.args)))
                expressions.extend(
                    _unparse_keyword(keyword) for keyword in current.keywords
                )
        elif isinstance(current, ast.Attribute):
            visit(current.value)

    visit(node)
    return expressions


def _is_explicitly_disabled(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in {False, None}


def _captures_exception(node: ast.Call, *, method: str) -> bool:
    if "exception" in method.split("|"):
        return True
    if any(
        keyword.arg in {"exc_info", "stack_info"}
        and not _is_explicitly_disabled(keyword.value)
        for keyword in node.keywords
    ):
        return True

    receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
    while isinstance(receiver, (ast.Attribute, ast.Call)):
        if isinstance(receiver, ast.Call):
            if (
                isinstance(receiver.func, ast.Attribute)
                and receiver.func.attr == "opt"
                and any(
                    keyword.arg == "exception"
                    and not _is_explicitly_disabled(keyword.value)
                    for keyword in receiver.keywords
                )
            ):
                return True
            receiver = receiver.func
        else:
            receiver = receiver.value
    return False


def _receiver_root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Call)):
        node = node.value if isinstance(node, ast.Attribute) else node.func
    return node.id if isinstance(node, ast.Name) else None


@dataclass(frozen=True)
class _AliasValue:
    is_logger: bool = False
    fields: tuple[str, ...] = ()
    captures_exception: bool = False
    is_factory: bool = False
    methods: tuple[str, ...] = ()


_AliasState = dict[str, _AliasValue]


def _stable_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _merge_alias_values(values: list[_AliasValue]) -> _AliasValue:
    return _AliasValue(
        is_logger=any(value.is_logger for value in values),
        fields=_stable_unique(
            tuple(field for value in values for field in value.fields)
        ),
        captures_exception=any(value.captures_exception for value in values),
        is_factory=any(value.is_factory for value in values),
        methods=tuple(sorted({method for value in values for method in value.methods})),
    )


def _merge_alias_states(*states: _AliasState) -> _AliasState:
    names = _stable_unique(tuple(name for state in states for name in state))
    return {
        name: _merge_alias_values([state[name] for state in states if name in state])
        for name in names
    }


def _bound_call_fields(node: ast.Call) -> tuple[str, ...]:
    return (
        *_unparse_many(list(node.args)),
        *(_unparse_keyword(keyword) for keyword in node.keywords),
    )


def _alias_from_expression(
    node: ast.AST | None, state: _AliasState
) -> _AliasValue | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return state.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _alias_from_expression(node.value, state)
        if base is None or not base.is_logger:
            return None
        if node.attr == "getLogger":
            return _AliasValue(is_factory=True)
        if node.attr in LOG_METHODS:
            return _AliasValue(methods=(node.attr,))
        return None
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        callee = state.get(node.func.id)
        return _AliasValue(is_logger=True) if callee and callee.is_factory else None
    if not isinstance(node.func, ast.Attribute):
        return None

    base = _alias_from_expression(node.func.value, state)
    if node.func.attr == "getLogger" and base and base.is_logger:
        return _AliasValue(is_logger=True)
    if node.func.attr not in {"bind", "opt"} or base is None or not base.is_logger:
        return None
    captures_exception = base.captures_exception or (
        node.func.attr == "opt"
        and any(
            keyword.arg == "exception" and not _is_explicitly_disabled(keyword.value)
            for keyword in node.keywords
        )
    )
    return _AliasValue(
        is_logger=True,
        fields=_stable_unique((*base.fields, *_bound_call_fields(node))),
        captures_exception=captures_exception,
    )


def _assign_alias_target(
    target: ast.AST, value: _AliasValue | None, state: _AliasState
) -> None:
    if isinstance(target, ast.Name):
        if value is None:
            state.pop(target.id, None)
        else:
            state[target.id] = value
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            _assign_alias_target(element, None, state)


def _record_call_snapshots(
    node: ast.AST | None,
    state: _AliasState,
    snapshots: dict[int, _AliasState],
) -> None:
    if node is None:
        return
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.Call):
            snapshots[id(candidate)] = dict(state)


def _import_aliases(node: ast.Import | ast.ImportFrom, state: _AliasState) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name.split(".", 1)[0]
            state.pop(name, None)
            if alias.name in {"logging", "loguru"}:
                state[name] = _AliasValue(is_logger=True)
        return

    for alias in node.names:
        if alias.name == "*":
            continue
        name = alias.asname or alias.name
        state.pop(name, None)
        if node.module == "logging" and alias.name == "getLogger":
            state[name] = _AliasValue(is_factory=True)
        elif node.module in {"logging", "loguru"} and alias.name in LOG_METHODS:
            state[name] = _AliasValue(methods=(alias.name,))
        elif node.module == "loguru" and alias.name == "logger":
            state[name] = _AliasValue(is_logger=True)


def _function_argument_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    arguments = node.args
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


class _LocalNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()
        self.nonlocal_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(
            alias.asname or alias.name.split(".", 1)[0] for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(
            alias.asname or alias.name for alias in node.names if alias.name != "*"
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_ListComp(self, node: ast.ListComp) -> None:
        pass

    def visit_SetComp(self, node: ast.SetComp) -> None:
        pass

    def visit_DictComp(self, node: ast.DictComp) -> None:
        pass

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        pass

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

    def visit_Global(self, node: ast.Global) -> None:
        self.nonlocal_names.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)


def _function_local_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    collector = _LocalNameCollector()
    for statement in node.body:
        collector.visit(statement)
    return (collector.names | _function_argument_names(node)) - collector.nonlocal_names


_DeferredFunction = tuple[ast.FunctionDef | ast.AsyncFunctionDef, _AliasState]
_DeferredClass = tuple[ast.ClassDef, _AliasState]
_AliasHistory = list[tuple[int, _AliasState]]


def _process_alias_block(
    statements: list[ast.stmt],
    initial: _AliasState,
    snapshots: dict[int, _AliasState],
    history: _AliasHistory,
    deferred: list[_DeferredFunction],
    deferred_classes: list[_DeferredClass],
) -> _AliasState:
    state = dict(initial)
    for statement in statements:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            _import_aliases(statement, state)
        elif isinstance(statement, ast.Assign):
            _record_call_snapshots(statement.value, state, snapshots)
            value = _alias_from_expression(statement.value, state)
            for target in statement.targets:
                _assign_alias_target(target, value, state)
        elif isinstance(statement, ast.AnnAssign):
            _record_call_snapshots(statement.value, state, snapshots)
            if statement.value is not None:
                _assign_alias_target(
                    statement.target,
                    _alias_from_expression(statement.value, state),
                    state,
                )
        elif isinstance(statement, ast.If):
            _record_call_snapshots(statement.test, state, snapshots)
            body = _process_alias_block(
                statement.body,
                state,
                snapshots,
                history,
                deferred,
                deferred_classes,
            )
            if statement.orelse:
                otherwise = _process_alias_block(
                    statement.orelse,
                    state,
                    snapshots,
                    history,
                    deferred,
                    deferred_classes,
                )
                state = _merge_alias_states(body, otherwise)
            else:
                state = _merge_alias_states(state, body)
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for expression in (
                *statement.decorator_list,
                *statement.args.defaults,
                *(default for default in statement.args.kw_defaults if default),
            ):
                _record_call_snapshots(expression, state, snapshots)
            deferred.append((statement, dict(state)))
            state.pop(statement.name, None)
        elif isinstance(statement, ast.ClassDef):
            for expression in (*statement.decorator_list, *statement.bases):
                _record_call_snapshots(expression, state, snapshots)
            deferred_classes.append((statement, dict(state)))
            state.pop(statement.name, None)
        else:
            for child in ast.iter_child_nodes(statement):
                if not isinstance(child, ast.stmt):
                    _record_call_snapshots(child, state, snapshots)
        history.append((statement.end_lineno or statement.lineno, dict(state)))
    return state


def _process_alias_statements(
    statements: list[ast.stmt],
    initial: _AliasState,
    snapshots: dict[int, _AliasState],
    *,
    function_parent: _AliasState | None = None,
) -> _AliasState:
    history: _AliasHistory = []
    deferred: list[_DeferredFunction] = []
    deferred_classes: list[_DeferredClass] = []
    state = _process_alias_block(
        statements,
        initial,
        snapshots,
        history,
        deferred,
        deferred_classes,
    )
    for function, definition_state in deferred:
        if function_parent is None:
            later_states = [
                later
                for lineno, later in history
                if lineno > (function.end_lineno or function.lineno)
            ]
            inherited = _merge_alias_states(definition_state, *later_states)
        else:
            inherited = dict(function_parent)
        local_names = _function_local_names(function)
        inherited = {
            name: value for name, value in inherited.items() if name not in local_names
        }
        _process_alias_statements(function.body, inherited, snapshots)
    for class_node, definition_state in deferred_classes:
        if function_parent is None:
            later_states = [
                later
                for lineno, later in history
                if lineno > (class_node.end_lineno or class_node.lineno)
            ]
            method_parent = _merge_alias_states(definition_state, *later_states)
        else:
            method_parent = function_parent
        class_body_initial = (
            definition_state if function_parent is None else function_parent
        )
        _process_alias_statements(
            class_node.body,
            class_body_initial,
            snapshots,
            function_parent=method_parent,
        )
    return state


def _alias_snapshots(tree: ast.Module) -> dict[int, _AliasState]:
    scanner_symbols = _logger_symbols(tree)
    initial = {
        name: _AliasValue(is_logger=True)
        for name in {"logger", "logging", "loguru_logger"}
        if name in scanner_symbols
    }
    snapshots: dict[int, _AliasState] = {}
    _process_alias_statements(tree.body, initial, snapshots)
    return snapshots


def _diagnostic_method(node: ast.Call, state: _AliasState) -> str | None:
    if isinstance(node.func, ast.Name):
        value = state.get(node.func.id)
        if value is not None and value.methods:
            return "|".join(sorted(value.methods))
        return None
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in LOG_METHODS:
        return None
    logger_symbols = {name for name, value in state.items() if value.is_logger}
    return node.func.attr if _is_diagnostic_call(node, logger_symbols) else None


def _message_parts(
    node: ast.Call, method: str
) -> tuple[ast.AST | None, list[ast.AST], list[ast.AST], str | None]:
    message_index = 1 if method == "log" else 0
    level_node = node.args[0] if method == "log" and node.args else None
    consumed_keywords: set[int] = set()

    if len(node.args) > message_index:
        message = node.args[message_index]
    else:
        message = None
        for index, keyword in enumerate(node.keywords):
            if keyword.arg in {"msg", "message"}:
                message = keyword.value
                consumed_keywords.add(index)
                break
    if method == "log" and level_node is None:
        for index, keyword in enumerate(node.keywords):
            if keyword.arg == "level":
                level_node = keyword.value
                consumed_keywords.add(index)
                break

    positional_fields = list(node.args[message_index + 1 :])
    keyword_fields = [
        keyword.value
        for index, keyword in enumerate(node.keywords)
        if index not in consumed_keywords
    ]
    level_expression = ast.unparse(level_node) if level_node is not None else None
    return message, positional_fields, keyword_fields, level_expression


def discover_diagnostic_calls(source: str, *, module: str) -> list[DiagnosticCall]:
    """Extract diagnostic calls with stable identities from Python source."""
    tree = ast.parse(source, filename=module)
    snapshots = _alias_snapshots(tree)
    fallback_state = {
        name: _AliasValue(is_logger=True)
        for name in ("logger", "logging", "loguru_logger")
    }
    scopes = _scope_names(tree)
    recognized: list[tuple[ast.Call, str, _AliasState]] = []
    for candidate in ast.walk(tree):
        if not isinstance(candidate, ast.Call):
            continue
        state = snapshots.get(id(candidate), fallback_state)
        method = _diagnostic_method(candidate, state)
        if method is not None:
            recognized.append((candidate, method, state))
    recognized.sort(key=lambda item: (item[0].lineno, item[0].col_offset))

    occurrences: defaultdict[tuple[str, str], int] = defaultdict(int)
    calls: list[DiagnosticCall] = []
    for node, method, state in recognized:
        first, positional_fields, keyword_fields, level_expression = _message_parts(
            node, method
        )
        qualname = scopes.get(id(node), "") or "<module>"
        event = _literal_projection(first) if first is not None else ""
        occurrence_key = (qualname, event)
        occurrences[occurrence_key] += 1
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else node
        receiver_root = _receiver_root_name(receiver)
        alias_value = state.get(receiver_root or "")
        receiver_fields = _stable_unique(
            (
                *(alias_value.fields if alias_value and alias_value.is_logger else ()),
                *_receiver_field_expressions(receiver),
            )
        )
        expressions = [
            *(_first_argument_expressions(first) if first is not None else []),
            *_unparse_many(positional_fields),
            *_unparse_many(keyword_fields),
            *receiver_fields,
        ]
        calls.append(
            DiagnosticCall(
                module=module,
                qualname=qualname,
                method=method,
                event=event,
                occurrence=occurrences[occurrence_key],
                message_shape=(
                    stable_dump(first, include_attributes=False)
                    if first is not None
                    else "<missing>"
                ),
                expressions=tuple(expressions),
                captures_exception=(
                    _captures_exception(node, method=method)
                    or bool(
                        alias_value
                        and alias_value.is_logger
                        and alias_value.captures_exception
                    )
                ),
                level_expression=level_expression,
            )
        )
    return calls


def _has_constant_string_message(call: DiagnosticCall) -> bool:
    prefix = "Constant(value="
    if not call.message_shape.startswith(prefix) or not call.message_shape.endswith(
        ")"
    ):
        return False
    try:
        return isinstance(ast.literal_eval(call.message_shape[len(prefix) : -1]), str)
    except (SyntaxError, ValueError):
        return False


def _is_approved_numeric_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, ast.Name):
        return (
            node.id in {"i", "index", "idx", "attempt"}
            or node.id.startswith("retry_")
            or node.id.endswith(("_count", "_length", "_retries"))
        )
    if isinstance(node, ast.Attribute):
        return (
            node.attr == "status_code"
            and isinstance(node.value, ast.Name)
            and node.value.id == "response"
        )
    if isinstance(node, ast.Call):
        return (
            isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
        )
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.Add, ast.Sub))
        and _is_approved_numeric_expression(node.left)
        and _is_approved_numeric_expression(node.right)
    )


def _is_approved_metadata_expression(expression: str) -> bool:
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return False

    if isinstance(node, ast.Constant):
        return type(node.value) in {bool, int}
    if isinstance(node, ast.Name):
        return (
            node.id == "streaming"
            or node.id == "attempt"
            or node.id.startswith(("is_", "has_", "retry_"))
            or node.id.endswith(
                ("_count", "_length", "_enabled", "_disabled", "_retries")
            )
        )
    if isinstance(node, ast.Attribute):
        return (
            node.attr == "status_code"
            and isinstance(node.value, ast.Name)
            and node.value.id == "response"
        )
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
        ):
            return True
        return (
            isinstance(node.func, ast.Name)
            and node.func.id == "safe_metadata_token"
            and len(node.args) == 1
            and not isinstance(node.args[0], ast.Starred)
            and not node.keywords
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _is_approved_numeric_expression(node)
    return False


def _unambiguous_diagnostic_severity(method: str) -> str | None:
    severities = {_DIAGNOSTIC_SEVERITIES.get(member) for member in method.split("|")}
    return severities.pop() if len(severities) == 1 and None not in severities else None


def assert_review_outcome(
    starting: DiagnosticCall, current: DiagnosticCall, *, outcome: str
) -> None:
    """Assert one reviewed call obeys its immutable outcome contract."""
    if outcome in {"pending", "frozen"}:
        assert starting == current, f"{outcome} diagnostic changed"
    elif outcome == "metadata":
        starting_severity = _unambiguous_diagnostic_severity(starting.method)
        current_severity = _unambiguous_diagnostic_severity(current.method)
        assert starting_severity is not None and current_severity is not None, (
            "metadata repair requires unambiguous diagnostic severity"
        )
        assert starting_severity == current_severity, (
            "metadata repair must preserve diagnostic severity"
        )
        if starting.method == "log":
            assert starting.level_expression == current.level_expression, (
                "metadata repair must preserve log level"
            )
        assert _has_constant_string_message(current), (
            "metadata requires a constant string first argument"
        )
        rejected = tuple(
            expression
            for expression in current.expressions
            if not _is_approved_metadata_expression(expression)
        )
        assert not rejected, (
            f"metadata contains unapproved metadata expression(s): {rejected!r}"
        )
    else:
        raise AssertionError(f"unknown diagnostic outcome: {outcome}")

    if outcome in {"frozen", "metadata"}:
        assert not current.captures_exception, (
            f"{outcome} diagnostic must not capture exception or traceback"
        )
