"""Test-only extraction and reconciliation for summarization diagnostics."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass

from scripts.check_persistent_diagnostic_inventory import (
    _is_diagnostic_call,
    _logger_symbols,
    _scope_names,
)


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
                expressions.extend(
                    _unparse_many(
                        [
                            *current.args,
                            *(keyword.value for keyword in current.keywords),
                        ]
                    )
                )
        elif isinstance(current, ast.Attribute):
            visit(current.value)

    visit(node)
    return expressions


def _is_explicitly_disabled(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in {False, None}


def _captures_exception(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Attribute) and node.func.attr == "exception":
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


def _message_parts(
    node: ast.Call,
) -> tuple[ast.AST | None, list[ast.AST], list[ast.AST], str | None]:
    method = node.func.attr if isinstance(node.func, ast.Attribute) else ""
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
    scopes = _scope_names(tree)
    nodes = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _is_diagnostic_call(node, symbols)
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    occurrences: defaultdict[tuple[str, str], int] = defaultdict(int)
    calls: list[DiagnosticCall] = []
    for node in nodes:
        first, positional_fields, keyword_fields, level_expression = _message_parts(
            node
        )
        qualname = scopes.get(id(node), "") or "<module>"
        event = _literal_projection(first) if first is not None else ""
        occurrence_key = (qualname, event)
        occurrences[occurrence_key] += 1
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else node
        expressions = [
            *(_first_argument_expressions(first) if first is not None else []),
            *_unparse_many(positional_fields),
            *_unparse_many(keyword_fields),
            *_receiver_field_expressions(receiver),
        ]
        calls.append(
            DiagnosticCall(
                module=module,
                qualname=qualname,
                method=(
                    node.func.attr if isinstance(node.func, ast.Attribute) else "call"
                ),
                event=event,
                occurrence=occurrences[occurrence_key],
                message_shape=(
                    ast.dump(first, include_attributes=False)
                    if first is not None
                    else "<missing>"
                ),
                expressions=tuple(expressions),
                captures_exception=_captures_exception(node),
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


def assert_review_outcome(
    starting: DiagnosticCall, current: DiagnosticCall, *, outcome: str
) -> None:
    """Assert one reviewed call obeys its immutable outcome contract."""
    if outcome in {"pending", "frozen"}:
        assert starting == current, f"{outcome} diagnostic changed"
    elif outcome == "metadata":
        assert _has_constant_string_message(current), (
            "metadata requires a constant string first argument"
        )
    else:
        raise AssertionError(f"unknown diagnostic outcome: {outcome}")

    if outcome in {"frozen", "metadata"}:
        assert not current.captures_exception, (
            f"{outcome} diagnostic must not capture exception or traceback"
        )
