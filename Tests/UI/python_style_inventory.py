"""Pure AST inventory for ADR-161's Python visual-style hard floor.

The scope deliberately matches the original property regex (not min/max sizes).
A ``# ds-runtime: <specific reason>`` comment on the write/call line or the
immediately preceding standalone comment line documents measured geometry or
user-input previews. Only nonliteral expressions may use this exception;
``None`` resets are independently legal. The inventory always retains every
write, including exempt writes, so reviewers can inspect the claimed exceptions.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass

_PROPERTY = re.compile(
    r"(?:background|color|border\w*|width|height|padding\w*|margin\w*|opacity\w*)\Z"
)
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


@dataclass(frozen=True)
class StyleWrite:
    """One covered property write, or ``*`` for opaque set_styles input."""

    property: str
    line: int
    form: str
    value_kind: str
    runtime_reason: str | None
    violation: bool


def inventory_styles(source: str) -> list[StyleWrite]:
    """Return covered writes, preserving unmarked and annotated exceptions.

    This is a conservative syntax inventory, not whole-program dataflow. It
    resolves simple lexical aliases and finite literal branches, and treats
    uppercase constant references as static even when imported. An annotation
    on an unresolved expression remains a reviewable claim about runtime data.
    """
    tree = ast.parse(source)
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    scopes = (
        ast.Module,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
    )

    def scope(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, scopes):
                return node
        return tree

    bindings: dict[ast.AST, dict[str, list[ast.AST]]] = {}

    def pairs(target, value):
        if isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(
                value.elts
            ):
                for left, right in zip(target.elts, value.elts):
                    yield from pairs(left, right)
            else:
                for child in target.elts:
                    yield from pairs(child, value)
        else:
            yield target, value

    for node in ast.walk(tree):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if targets and node.value is not None:
            for target in targets:
                for left, value in pairs(target, node.value):
                    if isinstance(left, ast.Name):
                        bindings.setdefault(scope(node), {}).setdefault(
                            left.id, []
                        ).append(value)

    def static(node, seen=frozenset()):
        if node in seen:
            return False
        seen = seen | {node}
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return all(static(item, seen) for item in node.elts)
        if isinstance(node, ast.Dict):
            return all(
                key is not None and static(key, seen) and static(value, seen)
                for key, value in zip(node.keys, node.values)
            )
        if isinstance(node, ast.IfExp):
            return static(node.body, seen) and static(node.orelse, seen)
        if isinstance(node, ast.Name):
            owner = scope(node)
            while True:
                if isinstance(
                    owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
                ):
                    arguments = owner.args
                    names = [
                        arg.arg
                        for arg in (
                            *arguments.posonlyargs,
                            *arguments.args,
                            *arguments.kwonlyargs,
                        )
                    ]
                    names += [
                        arg.arg
                        for arg in (arguments.vararg, arguments.kwarg)
                        if arg is not None
                    ]
                    if node.id in names:
                        return False
                values = bindings.get(owner, {}).get(node.id)
                if values:
                    return all(static(value, seen) for value in values)
                if owner is tree:
                    break
                owner = scope(owner)
            return node.id.isupper()
        if isinstance(node, ast.Attribute):
            return node.attr.isupper()
        if isinstance(node, ast.UnaryOp):
            return static(node.operand, seen)
        if isinstance(node, ast.BinOp):
            return static(node.left, seen) and static(node.right, seen)
        if isinstance(node, ast.JoinedStr):
            return all(
                static(
                    item.value if isinstance(item, ast.FormattedValue) else item, seen
                )
                for item in node.values
            )
        if isinstance(node, ast.Subscript):
            return static(node.value, seen)
        if isinstance(node, ast.Call) and ast.unparse(node.func) in {
            "int",
            "float",
            "str",
            "tuple",
            "list",
            "round",
            "min",
            "max",
            "Color",
            "Color.parse",
            "Scalar",
            "Scalar.parse",
            "Spacing",
        }:
            return all(static(value, seen) for value in node.args) and all(
                keyword.arg is not None and static(keyword.value, seen)
                for keyword in node.keywords
            )
        return False

    lines = source.splitlines()
    comments = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            match = re.fullmatch(r"# ds-runtime:\s*(.+?)\s*", token.string)
            if match:
                reason = match.group(1)
                if len(
                    re.findall(r"[A-Za-z]+", reason)
                ) >= 2 and not reason.lower().startswith(("todo", "fixme")):
                    comments[token.start[0]] = reason

    def reason_at(*locations):
        for line in locations:
            if line in comments:
                return comments[line]
            if (
                line > 1
                and line - 1 in comments
                and lines[line - 2].lstrip().startswith("#")
            ):
                return comments[line - 1]
        return None

    writes = []

    def record(prop, value, node, form, call=None):
        prop = prop.replace("-", "_")
        if prop != "*" and not _PROPERTY.fullmatch(prop):
            return
        kind = (
            "reset"
            if isinstance(value, ast.Constant) and value.value is None
            else "static"
            if static(value)
            else "runtime"
        )
        reason = reason_at(node.lineno, (call or node).lineno)
        writes.append(
            StyleWrite(
                prop,
                node.lineno,
                form,
                kind,
                reason,
                kind != "reset" and (kind == "static" or reason is None),
            )
        )

    def styles(node):
        return isinstance(node, ast.Attribute) and node.attr == "styles"

    for node in ast.walk(tree):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, (ast.AnnAssign, ast.AugAssign))
            else []
        )
        if targets and node.value is not None:
            for target in targets:
                for left, value in pairs(target, node.value):
                    if isinstance(left, ast.Attribute) and styles(left.value):
                        record(left.attr, value, left, "assignment")
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 3
            and styles(node.args[0])
        ):
            prop = node.args[1]
            record(
                prop.value
                if isinstance(prop, ast.Constant) and isinstance(prop.value, str)
                else "*",
                node.args[2],
                node,
                "setattr",
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "set_styles":
            for keyword in node.keywords:
                if keyword.arg != "css":
                    record(
                        keyword.arg or "*", keyword.value, keyword, "set_styles", node
                    )
            css_values = list(node.args) + [
                keyword.value for keyword in node.keywords if keyword.arg == "css"
            ]
            for value in css_values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    css = _CSS_COMMENT.sub("", value.value)
                    for declaration in css.split(";"):
                        match = re.match(r"\s*([\w-]+)\s*:", declaration)
                        if match:
                            record(match.group(1), value, node, "set_styles CSS")
                else:
                    record("*", value, node, "set_styles CSS")
    return sorted(writes, key=lambda write: write.line)
