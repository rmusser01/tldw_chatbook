"""Test-only extraction and reconciliation for summarization diagnostics."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass

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
    if method == "exception":
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


def _imported_log_methods(tree: ast.AST) -> dict[str, str]:
    methods: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module not in {
            "logging",
            "loguru",
        }:
            continue
        for alias in node.names:
            if alias.name in LOG_METHODS:
                methods[alias.asname or alias.name] = alias.name
    return methods


def _opt_captures_exception(node: ast.AST) -> bool:
    return any(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Attribute)
        and candidate.func.attr == "opt"
        and any(
            keyword.arg == "exception" and not _is_explicitly_disabled(keyword.value)
            for keyword in candidate.keywords
        )
        for candidate in ast.walk(node)
    )


def _derived_logger_aliases(
    tree: ast.AST, symbols: set[str]
) -> tuple[dict[str, tuple[str, ...]], dict[str, bool]]:
    fields: dict[str, tuple[str, ...]] = {}
    captures: dict[str, bool] = {}
    assignments = [
        node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    assignments.sort(key=lambda node: (node.lineno, node.col_offset))
    for assignment in assignments:
        value = assignment.value
        if (
            not isinstance(value, ast.Call)
            or not isinstance(value.func, ast.Attribute)
            or value.func.attr not in {"bind", "opt"}
        ):
            continue
        root = _receiver_root_name(value.func.value)
        if root is None or (
            root not in symbols
            and root not in fields
            and root.casefold() not in {"log", "logger", "loguru_logger"}
            and not root.casefold().endswith("_logger")
        ):
            continue
        targets = (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else [assignment.target]
        )
        alias_fields = (
            *fields.get(root, ()),
            *_receiver_field_expressions(value),
        )
        alias_captures = captures.get(root, False) or _opt_captures_exception(value)
        for target in targets:
            if isinstance(target, ast.Name):
                fields[target.id] = alias_fields
                captures[target.id] = alias_captures
    return fields, captures


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
    symbols = _logger_symbols(tree)
    imported_methods = _imported_log_methods(tree)
    alias_fields, alias_captures = _derived_logger_aliases(tree, symbols)
    symbols.update(alias_fields)
    scopes = _scope_names(tree)
    nodes = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                _is_diagnostic_call(node, symbols)
                or (
                    isinstance(node.func, ast.Name) and node.func.id in imported_methods
                )
            )
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    occurrences: defaultdict[tuple[str, str], int] = defaultdict(int)
    calls: list[DiagnosticCall] = []
    for node in nodes:
        method = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else imported_methods[node.func.id]
        )
        first, positional_fields, keyword_fields, level_expression = _message_parts(
            node, method
        )
        qualname = scopes.get(id(node), "") or "<module>"
        event = _literal_projection(first) if first is not None else ""
        occurrence_key = (qualname, event)
        occurrences[occurrence_key] += 1
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else node
        receiver_root = _receiver_root_name(receiver)
        expressions = [
            *(_first_argument_expressions(first) if first is not None else []),
            *_unparse_many(positional_fields),
            *_unparse_many(keyword_fields),
            *alias_fields.get(receiver_root or "", ()),
            *_receiver_field_expressions(receiver),
        ]
        calls.append(
            DiagnosticCall(
                module=module,
                qualname=qualname,
                method=method,
                event=event,
                occurrence=occurrences[occurrence_key],
                message_shape=(
                    ast.dump(first, include_attributes=False)
                    if first is not None
                    else "<missing>"
                ),
                expressions=tuple(expressions),
                captures_exception=(
                    _captures_exception(node, method=method)
                    or alias_captures.get(receiver_root or "", False)
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
            and not node.keywords
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _is_approved_numeric_expression(node)
    return False


def assert_review_outcome(
    starting: DiagnosticCall, current: DiagnosticCall, *, outcome: str
) -> None:
    """Assert one reviewed call obeys its immutable outcome contract."""
    if outcome in {"pending", "frozen"}:
        assert starting == current, f"{outcome} diagnostic changed"
    elif outcome == "metadata":
        assert (
            _DIAGNOSTIC_SEVERITIES[starting.method]
            == _DIAGNOSTIC_SEVERITIES[current.method]
        ), "metadata repair must preserve diagnostic severity"
        if starting.method == "log":
            assert starting.level_expression == current.level_expression, (
                "metadata repair must preserve log level"
            )
        assert _has_constant_string_message(current), (
            "metadata requires a constant string first argument"
        )
        assert all(
            _is_approved_metadata_expression(expression)
            for expression in current.expressions
        ), "metadata contains an unapproved metadata expression"
    else:
        raise AssertionError(f"unknown diagnostic outcome: {outcome}")

    if outcome in {"frozen", "metadata"}:
        assert not current.captures_exception, (
            f"{outcome} diagnostic must not capture exception or traceback"
        )
