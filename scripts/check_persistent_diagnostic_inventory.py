#!/usr/bin/env python3
"""Check the reviewed inventory of production diagnostics and disk sinks.

The inventory is keyed on the CONTENT of each diagnostic and sink -- the
statement's own source text plus its log method -- and never on where in the
file it sits.  Moving a logger call is therefore not a review event, while
adding, deleting, rewording, or re-levelling one still is.  See task-3750: a
digest that fires on pure line movement trains reviewers to regenerate this
file without reading it, which is the one failure mode it exists to prevent.

TASK-19572: any non-zero exit now prints the full committed-vs-rebuild report --
rows only-in-committed / only-in-rebuild / changed with
``old_count/old_digest -> new_count/new_digest``, per-entry sink-topology
deltas, metadata deltas, and the exact next command. No flag is needed; ``--diff``
only adds an explicit "no drift" line when the tree is already in sync (there is
no report to print in that case). Reading that report IS the review the artifact
demands, so it deliberately reports what changed rather than regenerating
anything.

The pin stores an aggregate per-file digest and no statement text, so the report
is at the maximum resolution the artifact allows: it names which files drifted
and by how much, and ``--statements <path> --since <rev>`` recovers the
statements themselves. That mode exists because the obvious alternative is
wrong: the per-call digest is taken over the statement's raw source segment,
indentation included, so a call that merely shifted nesting level moves the
file's digest, and a line diff shows it as removed+added inside whatever else
changed. Measured during the TASK-19572 pre-merge review:
``Chat/console_fleet_wake.py`` drifted inside a 328-line diff in which not one
diagnostic statement had actually changed. ``--statements`` pairs those off and
prints only the text that really needs reading.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import warnings
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=SyntaxWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "tldw_chatbook"
INVENTORY_PATH = REPO_ROOT / "Docs/security/production-diagnostic-inventory.json"

LOG_METHODS = {
    "critical",
    "debug",
    "error",
    "exception",
    "info",
    "log",
    "success",
    "trace",
    "warning",
}
TASK_492_PREFIXES = (
    "tldw_chatbook/Chat/",
    "tldw_chatbook/LLM_Calls/",
    "tldw_chatbook/MCP/",
    "tldw_chatbook/Tools/",
)
TASK_492_FILES = {
    "tldw_chatbook/Agents/mcp_tool_provider.py",
}
TASK_31551_FILES = {
    "tldw_chatbook/Audio/meeting_capture.py",
    "tldw_chatbook/Audio/meeting_owner.py",
    "tldw_chatbook/Audio/meeting_session.py",
    "tldw_chatbook/Audio/system_audio_tap.py",
    "tldw_chatbook/UI/Screens/meetings_screen.py",
}
SINK_CALL_NAMES = {
    "FileHandler",
    "PrivateRotatingFileHandler",
    "RotatingFileHandler",
    "TimedRotatingFileHandler",
    "addHandler",
    "atomic_private_write_bytes",
    "open_private_text_append",
    "open_private_text_append_stream",
}
PATH_TERMINAL_TOKENS = frozenset(
    {"path", "paths", "root", "roots", "dir", "directory", "folder"}
)
SAFE_PATH_TRANSFORMS = frozenset({"content_fingerprint", "redact_user_paths"})
TRANSPARENT_SAFE_WRAPPERS = frozenset({"str"})
LOG_SANITIZER_MODULE = ("tldw_chatbook", "Utils", "log_sanitizer")
_COMPREHENSION_SCOPES = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
_ANONYMOUS_SCOPES = (ast.Lambda, *_COMPREHENSION_SCOPES)
_CLOSURE_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    *_ANONYMOUS_SCOPES,
)
PATH_PRIVACY_RULES = {
    "candidate_status": "legacy_unreviewed",
    "status_meaning": (
        "unresolved baseline candidate; inventory presence is not approval or "
        "a reviewed-safe classification"
    ),
    "identifier_rule": (
        "bounded snake-case terminal path/root/dir/directory/folder tokens and "
        "explicit *_path_str forms"
    ),
    "safe_transforms": [
        "content_fingerprint(path)",
        "redact_user_paths(path)",
        "path.suffix",
        "len(paths)",
        "type(exc).__name__",
    ],
}


class PathState(Enum):
    """Classification of whether an expression can expose a raw path."""

    UNKNOWN = 0
    PROVEN_SAFE = 1
    TAINTED = 2


def _attribute_parts(node: ast.AST) -> list[str]:
    parts: list[str] = []
    while isinstance(node, (ast.Attribute, ast.Call)):
        if isinstance(node, ast.Call):
            node = node.func
            continue
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return list(reversed(parts))


def _logger_symbols(tree: ast.AST) -> set[str]:
    symbols = {"logger", "logging", "loguru_logger"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"logging", "loguru"}:
                    symbols.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module in {"logging", "loguru"}:
                for alias in node.names:
                    symbols.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not isinstance(value, ast.Call):
                continue
            parts = _attribute_parts(value.func)
            if not parts or parts[-1] != "getLogger":
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
    return symbols


def _target_bound_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.List, ast.Tuple)):
        return {
            name for element in target.elts for name in _target_bound_names(element)
        }
    if isinstance(target, ast.Starred):
        return _target_bound_names(target.value)
    return set()


def _parameter_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> set[str]:
    arguments = node.args
    names = {
        argument.arg
        for argument in [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ]
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


def _approved_sanitizer_qualifier(
    node: ast.Import | ast.ImportFrom, alias: ast.alias
) -> tuple[str, ...] | None:
    if isinstance(node, ast.Import):
        if tuple(alias.name.split(".")) != LOG_SANITIZER_MODULE:
            return None
        return (alias.asname,) if alias.asname else LOG_SANITIZER_MODULE
    if (
        node.module == ".".join(LOG_SANITIZER_MODULE[:-1])
        and alias.name == LOG_SANITIZER_MODULE[-1]
    ):
        return (alias.asname or alias.name,)
    if (
        node.module == ".".join(LOG_SANITIZER_MODULE)
        and alias.name in SAFE_PATH_TRANSFORMS
    ):
        return (alias.asname or alias.name,)
    return None


def _import_bound_name(node: ast.Import | ast.ImportFrom, alias: ast.alias) -> str:
    if alias.asname:
        return alias.asname
    if isinstance(node, ast.Import):
        return alias.name.split(".", 1)[0]
    return alias.name


def _enclosing_lexical_shadowed_names(
    scope: ast.AST,
    definition_parent_scopes: dict[int, ast.AST],
    shadowed: dict[int, set[str]],
) -> set[str]:
    """Collect shadows inherited from enclosing closure scopes."""
    inherited: set[str] = set()
    parent = definition_parent_scopes.get(id(scope))
    while parent is not None:
        if isinstance(parent, _CLOSURE_SCOPES):
            inherited.update(shadowed[id(parent)])
        parent = definition_parent_scopes.get(id(parent))
    return inherited


def _safe_transform_contexts(
    tree: ast.Module,
    lexical_scopes: dict[int, ast.AST],
    definition_parent_scopes: dict[int, ast.AST],
) -> dict[int, tuple[frozenset[tuple[str, ...]], frozenset[str]]]:
    """Resolve approved sanitizer names without leaking aliases across scopes."""
    scope_ids = {id(scope) for scope in lexical_scopes.values()}
    local_qualifiers: dict[int, set[tuple[str, ...]]] = {
        scope_id: set() for scope_id in scope_ids
    }
    shadowed: dict[int, set[str]] = {scope_id: set() for scope_id in scope_ids}
    module_scope_id = id(tree)
    local_qualifiers[module_scope_id].add(LOG_SANITIZER_MODULE)

    for node in ast.walk(tree):
        scope_id = id(lexical_scopes[id(node)])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            shadowed[id(definition_parent_scopes[id(node)])].add(node.name)
            shadowed[scope_id].update(_parameter_names(node))
        elif isinstance(node, ast.Lambda):
            shadowed[scope_id].update(_parameter_names(node))
        elif isinstance(node, _COMPREHENSION_SCOPES):
            for generator in node.generators:
                shadowed[scope_id].update(_target_bound_names(generator.target))
        elif isinstance(node, ast.ClassDef):
            shadowed[id(definition_parent_scopes[id(node)])].add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                shadowed[scope_id].update(_target_bound_names(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            shadowed[scope_id].update(_target_bound_names(node.target))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            shadowed[scope_id].update(_target_bound_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    shadowed[scope_id].update(_target_bound_names(item.optional_vars))
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            shadowed[scope_id].add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                qualifier = _approved_sanitizer_qualifier(node, alias)
                if qualifier is not None:
                    local_qualifiers[scope_id].add(qualifier)
                    continue
                is_approved_direct_import = (
                    isinstance(node, ast.ImportFrom)
                    and node.module == ".".join(LOG_SANITIZER_MODULE)
                    and alias.name in SAFE_PATH_TRANSFORMS
                    and (alias.asname or alias.name) in SAFE_PATH_TRANSFORMS
                )
                if not is_approved_direct_import:
                    shadowed[scope_id].add(_import_bound_name(node, alias))

    module_shadowed = shadowed[module_scope_id]
    module_qualifiers = {
        qualifier
        for qualifier in local_qualifiers[module_scope_id]
        if qualifier[0] not in module_shadowed
    }
    contexts: dict[int, tuple[frozenset[tuple[str, ...]], frozenset[str]]] = {}
    for scope_id in scope_ids:
        local_shadowed = shadowed[scope_id]
        inherited_shadowed = _enclosing_lexical_shadowed_names(
            lexical_scopes[scope_id], definition_parent_scopes, shadowed
        )
        visible_shadowed = module_shadowed | inherited_shadowed | local_shadowed
        qualifiers = {
            qualifier
            for qualifier in local_qualifiers[scope_id]
            if qualifier[0] not in visible_shadowed
        }
        if scope_id != module_scope_id:
            qualifiers.update(
                qualifier
                for qualifier in module_qualifiers
                if qualifier[0] not in visible_shadowed
            )
        contexts[scope_id] = (
            frozenset(qualifiers),
            frozenset(visible_shadowed),
        )
    return contexts


def _is_diagnostic_call(node: ast.Call, logger_symbols: set[str]) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in LOG_METHODS:
        return False
    receiver = _attribute_parts(node.func.value)
    return any(
        part in logger_symbols
        or part.casefold() in {"log", "logger", "logging", "loguru_logger"}
        or part.casefold().endswith("_logger")
        for part in receiver
    )


def _scope_contexts(
    tree: ast.Module,
) -> tuple[
    dict[int, str],
    dict[int, ast.AST],
    dict[int, list[tuple[ast.AST, ast.AST]]],
    dict[int, ast.AST],
]:
    """Collect scope names, lexical owners, assignments, and definition parents."""
    names: dict[int, str] = {}
    lexical_scopes: dict[int, ast.AST] = {id(tree): tree}
    assignments: dict[int, list[tuple[ast.AST, ast.AST]]] = {}
    definition_parent_scopes: dict[int, ast.AST] = {}
    stack: list[tuple[ast.AST, str, ast.AST]] = [(tree, "", tree)]
    while stack:
        node, prefix, scope = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child_prefix = f"{prefix}.{child.name}" if prefix else child.name
                child_scope = child
                definition_parent_scopes[id(child)] = scope
            elif isinstance(child, _ANONYMOUS_SCOPES):
                child_prefix = prefix
                child_scope = child
                definition_parent_scopes[id(child)] = scope
            else:
                child_prefix = prefix
                child_scope = scope
            names[id(child)] = child_prefix
            lexical_scopes[id(child)] = child_scope
            if isinstance(child, ast.Assign):
                targets = child.targets
            elif isinstance(child, ast.AnnAssign) and child.value is not None:
                targets = [child.target]
            elif isinstance(child, ast.NamedExpr):
                targets = [child.target]
            else:
                targets = []
            for target in targets:
                assignments.setdefault(id(child_scope), []).append(
                    (target, child.value)
                )
            stack.append((child, child_prefix, child_scope))
    return names, lexical_scopes, assignments, definition_parent_scopes


def _scope_names(tree: ast.Module) -> dict[int, str]:
    """Map every node to the dotted name of its enclosing def/class scope.

    Used to give persistent-sink entries a human navigation handle that, unlike
    a line number, survives unrelated edits elsewhere in the file. Iterative
    (explicit worklist) rather than recursive, so a pathologically nested
    expression cannot raise ``RecursionError`` and crash the gate itself --
    the gate failing for a reason unrelated to diagnostics would train people
    to bypass it.

    Args:
        tree: Parsed module to walk.

    Returns:
        dict[int, str]: ``id(node)`` -> dotted scope name (`""` at module
            scope). Keyed by identity because AST nodes are unhashable by
            value and identity is stable for the lifetime of ``tree``.
    """
    names, _lexical_scopes, _assignments, _definition_parents = _scope_contexts(tree)
    return names


def _call_entry(
    source: str,
    node: ast.Call,
    *,
    kind: str | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    """Describe one call by its CONTENT.

    The entry deliberately carries no line number or byte offset: a call that
    moves within a file is not a review event, while any change to the
    statement's own text (message, level, arguments) changes ``digest``.
    """
    segment = ast.get_source_segment(source, node) or ""
    return {
        "method": (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else "call"
        ),
        "digest": hashlib.sha256(segment.encode("utf-8")).hexdigest()[:16],
        **({"kind": kind} if kind else {}),
        **({"scope": scope or "<module>"} if scope is not None else {}),
    }


def diagnostic_digest(diagnostics: list[dict[str, Any]]) -> str:
    """Digest a file's diagnostics by content, independently of their order.

    The projection is a sorted LIST, never a set, so multiplicity is part of
    the digest: deleting one of two identical logger calls still changes it.

    Args:
        diagnostics: Entries from `scan_source` for one file; only `method`
            and the per-call source `digest` participate -- deliberately not
            `line`, which is the whole point of task-3750.

    Returns:
        str: 20-hex-char content digest for the file's diagnostics.
    """
    content = sorted((entry["method"], entry["digest"]) for entry in diagnostics)
    return hashlib.sha256(
        json.dumps(content, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def _owner(path_text: str) -> tuple[str, str]:
    if path_text in TASK_492_FILES or path_text.startswith(TASK_492_PREFIXES):
        return (
            "TASK-492",
            "high-risk Chat/provider/summarization/tool/MCP diagnostic owner",
        )
    if path_text in TASK_31551_FILES:
        return (
            "TASK-31551",
            "meeting transcription (Audio capture/session/owner + Meetings screen)",
        )
    return (
        "TASK-494",
        "remaining Chatbook production diagnostic owner",
    )


def _identifier_is_path_shaped(identifier: str) -> bool:
    tokens = [token for token in identifier.casefold().split("_") if token]
    if not tokens:
        return False
    return tokens[-1] in PATH_TERMINAL_TOKENS or tokens[-2:] == ["path", "str"]


def _assignment_target_label(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return ast.unparse(target)
    return None


def _is_safe_path_transform(
    node: ast.AST,
    log_sanitizer_qualifiers: frozenset[tuple[str, ...]],
    shadowed_names: frozenset[str],
) -> bool:
    if isinstance(node, ast.Attribute):
        if node.attr == "suffix":
            return True
        if (
            node.attr == "__name__"
            and isinstance(node.value, ast.Call)
            and _attribute_parts(node.value.func) == ["type"]
            and "type" not in shadowed_names
        ):
            return True
    if not isinstance(node, ast.Call):
        return False
    parts = _attribute_parts(node.func)
    if not parts:
        return False
    if parts == ["len"]:
        return "len" not in shadowed_names
    if len(parts) == 1:
        return parts[0] not in shadowed_names and (
            parts[0] in SAFE_PATH_TRANSFORMS or (parts[0],) in log_sanitizer_qualifiers
        )
    return (
        parts[-1] in SAFE_PATH_TRANSFORMS
        and tuple(parts[:-1]) in log_sanitizer_qualifiers
    )


def _is_known_path_producer(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    parts = _attribute_parts(node.func)
    if not parts:
        return False
    terminal = parts[-1].casefold()
    return (
        parts == ["Path"]
        or parts == ["os", "getcwd"]
        or parts == ["Path", "home"]
        or terminal == "resolve"
        or terminal.startswith("validate_path")
    )


def _get_literal_path_key(node: ast.AST) -> str | None:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return None
    key = node.args[0].value
    return key if _identifier_is_path_shaped(key) else None


def _anonymous_scope_bound_names(scope: ast.AST) -> set[str]:
    if isinstance(scope, ast.Lambda):
        return _parameter_names(scope)
    if isinstance(scope, _COMPREHENSION_SCOPES):
        return {
            name
            for generator in scope.generators
            for name in _target_bound_names(generator.target)
        }
    return set()


def _diagnostic_alias_scope(
    scope: ast.AST,
    definition_parent_scopes: dict[int, ast.AST],
) -> ast.AST:
    """Return the nearest named/module frame that owns assignment aliases."""
    while isinstance(scope, _ANONYMOUS_SCOPES):
        scope = definition_parent_scopes[id(scope)]
    return scope


def _visible_path_aliases(
    aliases: set[str],
    scope: ast.AST,
    definition_parent_scopes: dict[int, ast.AST],
) -> set[str]:
    """Remove aliases shadowed by intervening lambda/comprehension bindings."""
    visible = aliases.copy()
    while isinstance(scope, _ANONYMOUS_SCOPES):
        visible.difference_update(_anonymous_scope_bound_names(scope))
        scope = definition_parent_scopes[id(scope)]
    return visible


def _scope_local_bound_names(
    scope: ast.AST,
    lexical_scopes: dict[int, ast.AST],
    definition_parent_scopes: dict[int, ast.AST],
) -> set[str]:
    """Return names that shadow captured aliases in one lexical scope."""
    names: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        names.update(_parameter_names(scope))
    if isinstance(scope, _COMPREHENSION_SCOPES):
        for generator in scope.generators:
            names.update(_target_bound_names(generator.target))

    for node in ast.walk(scope):
        if node is scope:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if definition_parent_scopes.get(id(node)) is scope:
                names.add(node.name)
            continue
        if lexical_scopes.get(id(node)) is not scope:
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_bound_names(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            names.update(_target_bound_names(node.target))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            names.update(_target_bound_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    names.update(_target_bound_names(item.optional_vars))
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(_import_bound_name(node, alias) for alias in node.names)
    return names


def _alias_parent_scope(
    scope: ast.AST,
    definition_parent_scopes: dict[int, ast.AST],
) -> ast.AST | None:
    """Return the enclosing module/function scope visible to bare names."""
    parent = definition_parent_scopes.get(id(scope))
    while parent is not None:
        if isinstance(parent, ast.Module):
            return parent
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return parent
        parent = definition_parent_scopes.get(id(parent))
    return None


def _expression_path_state(
    node: ast.AST,
    aliases: set[str],
    log_sanitizer_qualifiers: frozenset[tuple[str, ...]],
    shadowed_names: frozenset[str],
    *,
    lexical_scopes: dict[int, ast.AST] | None = None,
    safe_transform_contexts: dict[
        int, tuple[frozenset[tuple[str, ...]], frozenset[str]]
    ]
    | None = None,
    definition_parent_scopes: dict[int, ast.AST] | None = None,
) -> PathState:
    if lexical_scopes is not None and safe_transform_contexts is not None:
        scope = lexical_scopes.get(id(node))
        if scope is not None:
            log_sanitizer_qualifiers, shadowed_names = safe_transform_contexts[
                id(scope)
            ]
            if definition_parent_scopes is not None:
                aliases = _visible_path_aliases(
                    aliases, scope, definition_parent_scopes
                )

    def child_state(child: ast.AST) -> PathState:
        return _expression_path_state(
            child,
            aliases,
            log_sanitizer_qualifiers,
            shadowed_names,
            lexical_scopes=lexical_scopes,
            safe_transform_contexts=safe_transform_contexts,
            definition_parent_scopes=definition_parent_scopes,
        )

    if _is_safe_path_transform(node, log_sanitizer_qualifiers, shadowed_names):
        return PathState.PROVEN_SAFE
    if _is_known_path_producer(node) or _get_literal_path_key(node) is not None:
        return PathState.TAINTED
    if isinstance(node, ast.Name):
        if node.id in aliases or _identifier_is_path_shaped(node.id):
            return PathState.TAINTED
        return PathState.UNKNOWN
    if isinstance(node, ast.Attribute):
        label = ast.unparse(node)
        if label in aliases or _identifier_is_path_shaped(node.attr):
            return PathState.TAINTED
    if isinstance(node, ast.Constant):
        return PathState.PROVEN_SAFE

    if isinstance(node, ast.Call):
        function_state = child_state(node.func)
        value_states = [
            child_state(value)
            for value in [
                *node.args,
                *(keyword.value for keyword in node.keywords),
            ]
        ]
        if function_state is PathState.TAINTED or any(
            state is PathState.TAINTED for state in value_states
        ):
            return PathState.TAINTED
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in TRANSPARENT_SAFE_WRAPPERS
            and node.func.id not in shadowed_names
            and value_states
            and all(state is PathState.PROVEN_SAFE for state in value_states)
        ):
            return PathState.PROVEN_SAFE
        if (
            isinstance(node.func, ast.Attribute)
            and child_state(node.func.value) is PathState.PROVEN_SAFE
            and all(state is PathState.PROVEN_SAFE for state in value_states)
        ):
            return PathState.PROVEN_SAFE
        return PathState.UNKNOWN

    child_states = [
        child_state(child)
        for child in ast.iter_child_nodes(node)
        if not isinstance(
            child,
            (
                ast.boolop,
                ast.cmpop,
                ast.expr_context,
                ast.operator,
                ast.unaryop,
            ),
        )
    ]
    if any(state is PathState.TAINTED for state in child_states):
        return PathState.TAINTED
    if child_states and all(state is PathState.PROVEN_SAFE for state in child_states):
        return PathState.PROVEN_SAFE
    return PathState.UNKNOWN


def _scope_path_aliases(
    assignments: dict[int, list[tuple[ast.AST, ast.AST]]],
    active_scope_ids: set[int],
    safe_transform_contexts: dict[
        int, tuple[frozenset[tuple[str, ...]], frozenset[str]]
    ],
    *,
    lexical_scopes: dict[int, ast.AST],
    definition_parent_scopes: dict[int, ast.AST],
) -> dict[int, set[str]]:
    scope_by_id = {id(scope): scope for scope in lexical_scopes.values()}
    resolved: dict[int, set[str]] = {}

    def resolve(scope: ast.AST) -> set[str]:
        scope_id = id(scope)
        if scope_id in resolved:
            return resolved[scope_id]

        parent = _alias_parent_scope(scope, definition_parent_scopes)
        visible = resolve(parent).copy() if parent is not None else set()
        visible.difference_update(
            _scope_local_bound_names(
                scope,
                lexical_scopes,
                definition_parent_scopes,
            )
        )
        log_sanitizer_qualifiers, shadowed_names = safe_transform_contexts[scope_id]
        changed = True
        while changed:
            changed = False
            for target, value in assignments.get(scope_id, []):
                label = _assignment_target_label(target)
                if label is None:
                    continue
                if label in visible:
                    continue
                if (
                    _expression_path_state(
                        value,
                        visible,
                        log_sanitizer_qualifiers,
                        shadowed_names,
                        lexical_scopes=lexical_scopes,
                        safe_transform_contexts=safe_transform_contexts,
                        definition_parent_scopes=definition_parent_scopes,
                    )
                    is PathState.TAINTED
                ):
                    visible.add(label)
                    changed = True
        resolved[scope_id] = visible
        return visible

    return {
        scope_id: resolve(scope_by_id[scope_id])
        for scope_id in active_scope_ids
        if scope_id in scope_by_id
    }


def _formatted_expressions(node: ast.AST) -> list[tuple[ast.AST, str | None]]:
    if isinstance(node, ast.JoinedStr):
        return [
            (child.value, None)
            for child in ast.walk(node)
            if isinstance(child, ast.FormattedValue)
        ]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        if isinstance(node.right, (ast.Tuple, ast.List)):
            return [(element, None) for element in node.right.elts]
        if isinstance(node.right, ast.Dict):
            return [
                (
                    value,
                    key.value
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    else None,
                )
                for key, value in zip(node.right.keys, node.right.values)
            ]
        return [(node.right, None)]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return [
            *((argument, None) for argument in node.args),
            *((keyword.value, keyword.arg) for keyword in node.keywords),
        ]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return []
    return [(node, None)]


def _diagnostic_dynamic_expressions(
    node: ast.Call,
) -> list[tuple[ast.AST, str | None]]:
    expressions: list[tuple[ast.AST, str | None]] = []
    for argument in node.args:
        expressions.extend(_formatted_expressions(argument))
    for keyword in node.keywords:
        expressions.extend(
            (expression, hint or keyword.arg)
            for expression, hint in _formatted_expressions(keyword.value)
        )
    return expressions


def _path_candidate_entry(
    source: str,
    node: ast.Call,
    *,
    scope: str,
    aliases: set[str],
    log_sanitizer_qualifiers: frozenset[tuple[str, ...]],
    shadowed_names: frozenset[str],
    lexical_scopes: dict[int, ast.AST],
    safe_transform_contexts: dict[
        int, tuple[frozenset[tuple[str, ...]], frozenset[str]]
    ],
    definition_parent_scopes: dict[int, ast.AST],
) -> dict[str, Any] | None:
    labels: set[str] = set()
    for expression, hint in _diagnostic_dynamic_expressions(node):
        expression_label = ast.unparse(expression)
        state = _expression_path_state(
            expression,
            aliases,
            log_sanitizer_qualifiers,
            shadowed_names,
            lexical_scopes=lexical_scopes,
            safe_transform_contexts=safe_transform_contexts,
            definition_parent_scopes=definition_parent_scopes,
        )
        if state is PathState.TAINTED:
            labels.add(expression_label)
        elif (
            state is PathState.UNKNOWN
            and hint is not None
            and _identifier_is_path_shaped(hint)
        ):
            labels.add(f"{hint}={expression_label}")
    sorted_labels = sorted(labels)
    if not sorted_labels:
        return None
    call = _call_entry(source, node)
    return {
        "method": call["method"],
        "call_digest": call["digest"],
        "scope": scope or "<module>",
        "path_expressions": sorted_labels,
        "status": "legacy_unreviewed",
    }


def _scan_parsed_source(
    source: str, tree: ast.Module
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    symbols = _logger_symbols(tree)
    (
        scope_names,
        lexical_scopes,
        assignments,
        definition_parent_scopes,
    ) = _scope_contexts(tree)
    safe_transform_contexts = _safe_transform_contexts(
        tree, lexical_scopes, definition_parent_scopes
    )
    diagnostics: list[dict[str, Any]] = []
    sinks: list[dict[str, Any]] = []
    diagnostic_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_diagnostic_call(node, symbols):
            diagnostics.append(_call_entry(source, node))
            diagnostic_calls.append(node)
        parts = _attribute_parts(node.func)
        call_name = parts[-1] if parts else ""
        is_loguru_add = call_name == "add" and any(
            part in symbols for part in parts[:-1]
        )
        is_console_loguru_add = (
            is_loguru_add
            and bool(node.args)
            and _attribute_parts(node.args[0]) in (["sys", "stdout"], ["sys", "stderr"])
        )
        if call_name in SINK_CALL_NAMES or (
            is_loguru_add and not is_console_loguru_add
        ):
            sinks.append(
                _call_entry(
                    source,
                    node,
                    kind="loguru_sink" if is_loguru_add else call_name,
                    scope=scope_names.get(id(node), ""),
                )
            )

    alias_scopes = {
        id(node): _diagnostic_alias_scope(
            lexical_scopes[id(node)], definition_parent_scopes
        )
        for node in diagnostic_calls
    }
    aliases = _scope_path_aliases(
        assignments,
        {id(alias_scopes[id(node)]) for node in diagnostic_calls},
        safe_transform_contexts,
        lexical_scopes=lexical_scopes,
        definition_parent_scopes=definition_parent_scopes,
    )
    candidates: list[dict[str, Any]] = []
    for node in diagnostic_calls:
        scope_id = id(lexical_scopes[id(node)])
        log_sanitizer_qualifiers, shadowed_names = safe_transform_contexts[scope_id]
        candidate = _path_candidate_entry(
            source,
            node,
            scope=scope_names.get(id(node), ""),
            aliases=aliases.get(id(alias_scopes[id(node)]), set()),
            log_sanitizer_qualifiers=log_sanitizer_qualifiers,
            shadowed_names=shadowed_names,
            lexical_scopes=lexical_scopes,
            safe_transform_contexts=safe_transform_contexts,
            definition_parent_scopes=definition_parent_scopes,
        )
        if candidate is not None:
            candidates.append(candidate)

    diagnostics.sort(key=lambda entry: (entry["method"], entry["digest"]))
    sinks.sort(
        key=lambda entry: (
            entry["scope"],
            entry["kind"],
            entry["method"],
            entry["digest"],
        )
    )
    candidates.sort(
        key=lambda entry: (
            entry["scope"],
            entry["method"],
            entry["call_digest"],
            entry["path_expressions"],
        )
    )
    return diagnostics, sinks, candidates


def scan_source(
    source: str, *, filename: str = "<source>"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return diagnostic and sink entries for one module's source.

    Args:
        source: Python source text to scan.
        filename: Source name used in syntax errors.

    Returns:
        A tuple containing diagnostic entries followed by persistent-sink entries.
    """
    tree = ast.parse(source, filename=filename)
    diagnostics, sinks, _candidates = _scan_parsed_source(source, tree)
    return diagnostics, sinks


def scan_path_diagnostic_candidates(
    source: str, *, filename: str = "<source>"
) -> list[dict[str, Any]]:
    """Return unresolved path-shaped diagnostics in one module.

    Args:
        source: Python source text to scan.
        filename: Source name used in syntax errors.

    Returns:
        Candidate diagnostics whose dynamic values can contain raw paths.
    """
    tree = ast.parse(source, filename=filename)
    _diagnostics, _sinks, candidates = _scan_parsed_source(source, tree)
    return candidates


def _scan_file(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    return _scan_parsed_source(source, tree)


def _inventory_path_sort_key(path: Path) -> str:
    """Return the case-sensitive POSIX spelling used by the inventory."""
    return path.relative_to(REPO_ROOT).as_posix()


def build_inventory() -> dict[str, Any]:
    owners: list[dict[str, Any]] = []
    topology: list[dict[str, Any]] = []
    path_privacy_candidates: list[dict[str, Any]] = []
    # Path ordering is case-folded on Windows. Sort by the serialized POSIX
    # spelling so every host produces the same ordered inventory lists.
    for path in sorted(PACKAGE_ROOT.rglob("*.py"), key=_inventory_path_sort_key):
        relative = path.relative_to(REPO_ROOT).as_posix()
        diagnostics, sinks, candidates = _scan_file(path)
        if diagnostics:
            owner, reason = _owner(relative)
            owners.append(
                {
                    "path": relative,
                    "owner": owner,
                    "reason": reason,
                    "call_count": len(diagnostics),
                    "diagnostic_digest": diagnostic_digest(diagnostics),
                }
            )
        if sinks:
            topology.append({"path": relative, "sinks": sinks})
        if candidates:
            path_privacy_candidates.append({"path": relative, "candidates": candidates})

    task_492_calls = sum(
        entry["call_count"] for entry in owners if entry["owner"] == "TASK-492"
    )
    task_31551_calls = sum(
        entry["call_count"] for entry in owners if entry["owner"] == "TASK-31551"
    )
    task_494_calls = sum(
        entry["call_count"] for entry in owners if entry["owner"] == "TASK-494"
    )
    return {
        # 3: adds the unresolved path-privacy candidate projection. Existing
        # owner digests and sink identities retain their schema-v2 meaning.
        "schema_version": 3,
        "scope": "tldw_chatbook/**/*.py",
        "classification_rules": {
            "TASK-492": {
                "prefixes": list(TASK_492_PREFIXES),
                "files": sorted(TASK_492_FILES),
                "reason": "Chat, provider, summarization, tool, and MCP paths",
            },
            "TASK-31551": {
                "files": sorted(TASK_31551_FILES),
                "reason": "meeting transcription (Audio capture/session/owner + Meetings screen)",
            },
            "TASK-494": {
                "rule": "all other production diagnostic owners",
                "reason": "remaining production domains",
            },
        },
        "path_privacy_rules": PATH_PRIVACY_RULES,
        "reviewed_exclusions": [
            {
                "paths": ["Tests/**", "Docs/**", "backlog/**", "examples/**"],
                "reason": "non-production code does not feed an application sink",
            },
            {
                "paths": ["third-party packages"],
                "reason": "not Chatbook-owned; persistent filtering is tested separately",
            },
        ],
        "summary": {
            "owner_files": len(owners),
            "task_492_calls": task_492_calls,
            "task_31551_calls": task_31551_calls,
            "task_494_calls": task_494_calls,
            "persistent_sink_files": len(topology),
            "path_privacy_candidate_calls": sum(
                len(row["candidates"]) for row in path_privacy_candidates
            ),
        },
        "owners": owners,
        "persistent_sink_topology": topology,
        "path_privacy_candidates": path_privacy_candidates,
    }


def _encoded(inventory: dict[str, Any]) -> str:
    return json.dumps(inventory, indent=2, sort_keys=True) + "\n"


NEXT_STEPS = (
    "Next: read every row above and confirm each change is one you intended.\n"
    "  - a call_count delta means a diagnostic was added or deleted;\n"
    "  - an unchanged count with a changed digest means one was reworded,\n"
    "    re-levelled, given different arguments, or merely RE-INDENTED -- check\n"
    "    it does not now interpolate user content, secrets, or paths into a\n"
    "    persistent sink;\n"
    "  - a sink-topology row means a new file/handler destination appeared.\n"
    "The pin stores only an aggregate per-file digest, so the rows above can name\n"
    "WHICH files changed and by how much, never the statement text -- and the\n"
    "interpolation check just above needs that text. Recover it with:\n"
    "  base=$(git log -1 --format=%H -- "
    "Docs/security/production-diagnostic-inventory.json)\n"
    "  python scripts/check_persistent_diagnostic_inventory.py \\\n"
    "      --statements <each path listed above> --since $base\n"
    "That prints the added and removed STATEMENTS themselves, and separates the\n"
    "ones that only moved or re-indented from the ones whose text really changed.\n"
    "Do NOT reach for `git diff` here: the digest covers a statement's own source\n"
    "text, indentation included, so a call that merely shifted nesting level\n"
    "reports as changed, and a line diff buries it in unrelated edits -- measured\n"
    "on tldw_chatbook/Chat/console_fleet_wake.py, whose row changed inside a\n"
    "328-line diff in which not one statement had actually changed.\n"
    "Treat that base revision as a LOWER BOUND, not the truth: the pin has been\n"
    "committed stale before (TASK-19572 review found two rows whose drift predated\n"
    "the pin's own commit), so if a listed file shows no logger change in that\n"
    "range, widen it rather than assuming the row is noise.\n"
    "Only then run:  python scripts/check_persistent_diagnostic_inventory.py --write\n"
    "and commit Docs/security/production-diagnostic-inventory.json with the "
    "review recorded in the task/PR notes."
)

_METADATA_KEYS = (
    "schema_version",
    "scope",
    "classification_rules",
    "path_privacy_rules",
    "reviewed_exclusions",
)


def _sink_key(entry: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(entry.get("scope", "")),
        str(entry.get("kind", "")),
        str(entry.get("method", "")),
        str(entry.get("digest", "")),
    )


def _describe_sink_key(key: tuple[str, str, str, str]) -> str:
    scope, kind, method, digest = key
    return f"{scope or '<module>'}: {kind}.{method} ({digest})"


def _describe_sink(entry: dict[str, Any]) -> str:
    return _describe_sink_key(_sink_key(entry))


def _owner_rows(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["path"]): row for row in inventory.get("owners", [])}


def _sink_rows(inventory: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(row["path"]): list(row.get("sinks", []))
        for row in inventory.get("persistent_sink_topology", [])
    }


def _path_candidate_key(
    entry: dict[str, Any],
) -> tuple[str, str, str, tuple[str, ...], str]:
    return (
        str(entry.get("scope", "")),
        str(entry.get("method", "")),
        str(entry.get("call_digest", "")),
        tuple(str(label) for label in entry.get("path_expressions", [])),
        str(entry.get("status", "")),
    )


def _describe_path_candidate_key(
    key: tuple[str, str, str, tuple[str, ...], str],
) -> str:
    scope, method, digest, expressions, status = key
    labels = ", ".join(expressions) or "<none>"
    return (
        f"{scope or '<module>'}: {method} ({digest}) paths=[{labels}] status={status}"
    )


def _path_candidate_rows(
    inventory: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(row["path"]): list(row.get("candidates", []))
        for row in inventory.get("path_privacy_candidates", [])
    }


def _summary_lines(committed: dict[str, Any], rebuilt: dict[str, Any]) -> list[str]:
    old, new = committed.get("summary", {}), rebuilt.get("summary", {})
    lines: list[str] = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            lines.append(f"    {key}: {old.get(key)} -> {new.get(key)}")
    return lines


def _owner_lines(committed: dict[str, Any], rebuilt: dict[str, Any]) -> list[str]:
    old, new = _owner_rows(committed), _owner_rows(rebuilt)
    lines: list[str] = []
    for path in sorted(set(old) - set(new)):
        row = old[path]
        lines.append(
            f"  - only in committed (diagnostics gone from this file): {path} "
            f"[{row.get('owner')}] count={row.get('call_count')} "
            f"digest={row.get('diagnostic_digest')}"
        )
    for path in sorted(set(new) - set(old)):
        row = new[path]
        lines.append(
            f"  + only in rebuild (file now has diagnostics): {path} "
            f"[{row.get('owner')}] count={row.get('call_count')} "
            f"digest={row.get('diagnostic_digest')}"
        )
    for path in sorted(set(old) & set(new)):
        before, after = old[path], new[path]
        if before == after:
            continue
        old_count = before.get("call_count")
        new_count = after.get("call_count")
        old_digest = before.get("diagnostic_digest")
        new_digest = after.get("diagnostic_digest")
        if old_count == new_count:
            note = (
                "same count, content changed "
                "(reworded / re-levelled / new args / re-indented) "
                "-- use --statements to see which"
            )
        else:
            delta = (new_count or 0) - (old_count or 0)
            note = f"{delta:+d} diagnostic call(s)"
        lines.append(
            f"  ~ changed: {path} "
            f"{old_count}/{old_digest} -> {new_count}/{new_digest}  ({note})"
        )
        if before.get("owner") != after.get("owner"):
            lines.append(f"      owner: {before.get('owner')} -> {after.get('owner')}")
    return lines


def _sink_lines(committed: dict[str, Any], rebuilt: dict[str, Any]) -> list[str]:
    old, new = _sink_rows(committed), _sink_rows(rebuilt)
    lines: list[str] = []
    for path in sorted(set(old) - set(new)):
        lines.append(
            f"  - only in committed (no persistent sink left here): {path} "
            f"({len(old[path])} sink entr{'y' if len(old[path]) == 1 else 'ies'})"
        )
        for entry in old[path]:
            lines.append(f"      - {_describe_sink(entry)}")
    for path in sorted(set(new) - set(old)):
        lines.append(
            f"  + only in rebuild (NEW persistent sink file): {path} "
            f"({len(new[path])} sink entr{'y' if len(new[path]) == 1 else 'ies'})"
        )
        for entry in new[path]:
            lines.append(f"      + {_describe_sink(entry)}")
    for path in sorted(set(old) & set(new)):
        # Counted (multiset), not a dict keyed by _sink_key: two identical
        # sink calls (e.g. the same FileHandler installed twice) share a key,
        # and a plain `{key: entry}` collapse silently drops the duplicate --
        # a drift that is purely a MULTIPLICITY change then reports as no
        # change at all, or worse, as "only serialization differs" (Qodo PR
        # #1947 finding 1). Counter equality compares counts, so it is the
        # correct notion of "unchanged" here.
        before_counts = Counter(_sink_key(entry) for entry in old[path])
        after_counts = Counter(_sink_key(entry) for entry in new[path])
        if before_counts == after_counts:
            continue
        lines.append(
            f"  ~ changed sinks: {path} ({len(old[path])} -> {len(new[path])} entries)"
        )
        for key in sorted(set(before_counts) | set(after_counts)):
            before_n, after_n = before_counts[key], after_counts[key]
            if before_n == after_n:
                continue
            description = _describe_sink_key(key)
            if before_n == 0:
                suffix = f"  (new, x{after_n})" if after_n > 1 else ""
                lines.append(f"      + {description}{suffix}")
            elif after_n == 0:
                suffix = f"  (removed, was x{before_n})" if before_n > 1 else ""
                lines.append(f"      - {description}{suffix}")
            else:
                lines.append(
                    f"      ~ {description}: "
                    f"{before_n} -> {after_n}  ({after_n - before_n:+d})"
                )
    return lines


def _path_candidate_lines(
    committed: dict[str, Any], rebuilt: dict[str, Any]
) -> list[str]:
    old = _path_candidate_rows(committed)
    new = _path_candidate_rows(rebuilt)
    lines: list[str] = []
    for path in sorted(set(old) | set(new)):
        before_counts = Counter(
            _path_candidate_key(entry) for entry in old.get(path, [])
        )
        after_counts = Counter(
            _path_candidate_key(entry) for entry in new.get(path, [])
        )
        if before_counts == after_counts:
            continue

        if path not in new:
            lines.append(
                f"  - only in committed (candidate file removed): {path} "
                f"({sum(before_counts.values())} candidate call(s))"
            )
        elif path not in old:
            lines.append(
                f"  + only in rebuild (NEW candidate file): {path} "
                f"({sum(after_counts.values())} candidate call(s))"
            )
        else:
            lines.append(
                f"  ~ changed candidates: {path} "
                f"({sum(before_counts.values())} -> "
                f"{sum(after_counts.values())} calls)"
            )

        for key in sorted(set(before_counts) | set(after_counts)):
            before_n, after_n = before_counts[key], after_counts[key]
            if before_n == after_n:
                continue
            description = _describe_path_candidate_key(key)
            if before_n == 0:
                lines.append(f"      + {description} x{after_n}")
            elif after_n == 0:
                lines.append(f"      - {description} x{before_n}")
            else:
                lines.append(
                    f"      ~ {description}: x{before_n} -> x{after_n} "
                    f"({after_n - before_n:+d})"
                )
    if not lines:
        return []
    return [
        "  ! Legacy path-privacy candidates are unresolved; inventory presence "
        "is not approved.",
        *lines,
    ]


def _metadata_lines(committed: dict[str, Any], rebuilt: dict[str, Any]) -> list[str]:
    """Name drift in the inventory's non-row metadata.

    The check compares the whole encoded file, so a changed classification rule
    or scope fails it just as a new logger call does. Without this section that
    failure would report zero rows and read as a false alarm.
    """
    lines: list[str] = []
    for key in _METADATA_KEYS:
        before, after = committed.get(key), rebuilt.get(key)
        if before == after:
            continue
        lines.append(f"  ~ {key}:")
        lines.append(f"      committed: {json.dumps(before, sort_keys=True)}")
        lines.append(f"      rebuild:   {json.dumps(after, sort_keys=True)}")
    return lines


def render_diff(committed_text: str, rebuilt: dict[str, Any]) -> str:
    """Render a reviewable report of how the committed inventory differs.

    Args:
        committed_text: Raw text of the committed inventory file.
        rebuilt: Freshly scanned inventory from ``build_inventory``.

    Returns:
        str: A multi-section report naming rows only-in-committed,
            only-in-rebuild and changed (with ``old_count/old_digest ->
            new_count/new_digest``), sink-topology deltas, unresolved
            path-candidate deltas, metadata deltas, and the exact next command.
            Never empty: a formatting-only drift still yields an explanation
            rather than silence.
    """
    try:
        committed = json.loads(committed_text)
    except json.JSONDecodeError as exc:
        return (
            f"the committed inventory is not valid JSON ({exc}); it cannot be "
            "diffed. Restore it from git, or -- if the rebuild is what you "
            "want -- review the working tree and run --write.\n" + NEXT_STEPS
        )

    sections = (
        ("summary", _summary_lines(committed, rebuilt)),
        ("owners", _owner_lines(committed, rebuilt)),
        ("persistent sink topology", _sink_lines(committed, rebuilt)),
        (
            "path privacy candidates",
            _path_candidate_lines(committed, rebuilt),
        ),
        ("inventory metadata", _metadata_lines(committed, rebuilt)),
    )
    body = [
        line for title, lines in sections if lines for line in (f"{title}:", *lines)
    ]
    if not body:
        # Parsed content is identical, so only the serialization differs --
        # whitespace, key order, or a hand-edit that JSON normalizes away.
        return (
            "the committed inventory's CONTENT matches the rebuild; only its "
            "serialization differs (whitespace, key order, or a hand edit). "
            "Run --write to re-normalize it.\n" + NEXT_STEPS
        )
    return "\n".join(body) + "\n" + NEXT_STEPS


def _statement_entries(source: str, path: str) -> list[dict[str, Any]]:
    """Every diagnostic statement in one module, with its text and line.

    Uses the same scanner and the same per-call digest as the pin, so a key
    printed here is the key that moved the file's aggregate digest.
    """
    tree = ast.parse(source, filename=path)
    symbols = _logger_symbols(tree)
    entries: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_diagnostic_call(node, symbols):
            continue
        entry = _call_entry(source, node)
        entry["text"] = ast.get_source_segment(source, node) or ""
        entry["line"] = node.lineno
        entry["col"] = node.col_offset
        entries.append(entry)
    entries.sort(key=lambda item: item["line"])
    return entries


def _normalized(entry: dict[str, Any]) -> tuple[str, str]:
    """Key a statement by level + whitespace-collapsed text.

    Two statements sharing this key differ only in layout, which the module
    docstring says is explicitly NOT a review event -- but the per-call digest
    is taken over the raw source segment, continuation-line indentation
    included, so re-indenting a call still moves the file's digest. Separating
    those out is the difference between a report that teaches and one that
    trains people to regenerate without reading (task-3750).
    """
    return (str(entry["method"]), " ".join(str(entry["text"]).split()))


def _indent_block(entry: dict[str, Any], prefix: str = "      | ") -> str:
    """Render a statement's source under a gutter, at its original shape.

    ``ast.get_source_segment`` returns the first line already stripped of its
    leading indentation while continuation lines keep their absolute column,
    so printing it verbatim renders a multi-line call as a staircase. Restoring
    the first line's column and then dedenting the whole block puts the call
    back the shape it has in the file, which is how a reviewer reads it.
    """
    import textwrap

    text = str(entry.get("text", ""))
    restored = " " * int(entry.get("col", 0)) + text
    body = textwrap.dedent(restored)
    return "\n".join(prefix + line for line in body.splitlines())


def render_statement_diff(old_source: str, new_source: str, path: str) -> str:
    """Report which diagnostic STATEMENTS changed between two revisions of a file.

    Args:
        old_source: The module's source at the base revision.
        new_source: The module's source now.
        path: Repo-relative path, used only for the heading.

    Returns:
        str: A report separating statements that merely moved or were
            re-indented -- which need no privacy review -- from those actually
            added, removed, or reworded, printing the full text of each of the
            latter so the interpolation check can be made on real text.
    """
    old = _statement_entries(old_source, path)
    new = _statement_entries(new_source, path)
    old_keys: dict[tuple[str, str], list[dict[str, Any]]] = {}
    new_keys: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in old:
        old_keys.setdefault((entry["method"], entry["digest"]), []).append(entry)
    for entry in new:
        new_keys.setdefault((entry["method"], entry["digest"]), []).append(entry)

    removed: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    for key, entries in old_keys.items():
        removed.extend(entries[len(new_keys.get(key, [])) :])
    for key, entries in new_keys.items():
        added.extend(entries[len(old_keys.get(key, [])) :])

    # Pair off statements whose only difference is layout, so they stop
    # competing for the reviewer's attention with real content changes.
    layout_only: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pending = list(added)
    still_removed: list[dict[str, Any]] = []
    for gone in removed:
        match = next((e for e in pending if _normalized(e) == _normalized(gone)), None)
        if match is None:
            still_removed.append(gone)
            continue
        pending.remove(match)
        layout_only.append((gone, match))

    lines = [
        f"{path}: {len(old)} -> {len(new)} diagnostic call(s)",
        f"  moved/re-indented only: {len(layout_only)}   "
        f"removed: {len(still_removed)}   added: {len(pending)}",
    ]
    if layout_only:
        lines.append(
            "\n= moved or re-indented -- statement text is unchanged, NO review needed:"
        )
        for gone, match in layout_only:
            lines.append(
                f"  = {gone['method']} {gone['digest']} -> {match['digest']}  "
                f"(line {gone['line']} -> {match['line']})"
            )
    if still_removed:
        lines.append("\n- REMOVED -- these statements no longer exist:")
        for entry in still_removed:
            lines.append(
                f"  - {entry['method']} {entry['digest']} (was line {entry['line']})"
            )
            lines.append(_indent_block(entry))
    if pending:
        lines.append(
            "\n+ ADDED -- read each one: does it interpolate user content, a "
            "secret, a path, or a URL?"
        )
        for entry in pending:
            lines.append(
                f"  + {entry['method']} {entry['digest']} (now line {entry['line']})"
            )
            lines.append(_indent_block(entry))
    if not layout_only and not still_removed and not pending:
        lines.append(
            "\nno diagnostic statement changed in this file between the two "
            "revisions. If the pin still lists it, the pin was already stale "
            "when it was committed -- widen the base revision."
        )
    return "\n".join(lines) + "\n"


def _source_at(revision: str, path: str) -> str | None:
    """Read one path's source at a git revision.

    Uses ``git show`` via stdlib ``subprocess`` so the checker stays
    install-free; this is a review aid, never part of the gate's own verdict.

    Returns:
        str | None: The source, or ``None`` when the path did not exist at that
            revision -- the ordinary case for an "only in rebuild" row, which
            must not be mistaken for a broken revision argument.
    """
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{revision}:{path}"],
        capture_output=True,
    )
    if result.returncode == 0:
        return result.stdout.decode("utf-8", errors="replace")
    resolved = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "rev-parse",
            "--quiet",
            "--verify",
            f"{revision}^{{commit}}",
        ],
        capture_output=True,
    )
    if resolved.returncode == 0:
        return None
    raise SystemExit(
        f"cannot resolve revision {revision!r}: "
        f"{result.stderr.decode('utf-8', 'replace').strip()}"
    )


def _repo_relative(raw: str) -> Path | None:
    """Resolve a ``--statements PATH`` argument to a path inside ``REPO_ROOT``.

    Accepts a path given relative to ``REPO_ROOT`` or an absolute path that
    already resolves inside it; returns ``None`` -- never raises -- for
    anything else, so the caller can print a clean error instead of two
    failure modes Qodo flagged on PR #1947: an absolute path outside the repo
    used to blow up with an unhandled ``ValueError`` from
    ``Path.relative_to`` (finding 4), and a relative path containing ``..``
    could walk out of ``REPO_ROOT`` and read an arbitrary file on disk with
    no indication it had done so (finding 3).

    This deliberately does not call ``Utils/path_validation.py``: that module
    imports ``Metrics.metrics_logger``, which imports ``psutil`` -- a
    third-party package. Every derived-artifact checker is stdlib-only and
    install-free by design (see the module docstring and
    ``.github/workflows/derived-artifacts.yml``'s ~90s, no-dependency-install
    budget), so pulling in the app's path-validation helper here would
    silently break that contract for a script that only ever reads files
    inside this repo for review purposes -- and it is invoked with no
    external/CI-controlled input in the first place (`--statements` is never
    populated from a workflow; both call sites run the checker bare).
    """
    candidate = Path(raw)
    full = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    try:
        resolved = full.resolve()
        return resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return None


def _run_statements(paths: list[str], since: str | None) -> int:
    reports: list[str] = []
    for raw in paths:
        path = _repo_relative(raw)
        if path is None:
            print(
                f"cannot use {raw!r}: it does not resolve inside the repository "
                f"({REPO_ROOT}); pass a path relative to the repo root or an "
                "absolute path under it",
                file=sys.stderr,
            )
            return 1
        text = path.as_posix()
        try:
            current = (REPO_ROOT / path).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"cannot read {text}: {exc}", file=sys.stderr)
            return 1
        if since is None:
            entries = _statement_entries(current, text)
            body = [f"{text}: {len(entries)} diagnostic call(s)"]
            for entry in entries:
                body.append(
                    f"  {entry['method']} {entry['digest']} (line {entry['line']})"
                )
                body.append(_indent_block(entry))
            reports.append("\n".join(body) + "\n")
            continue
        before = _source_at(since, text)
        if before is None:
            reports.append(
                f"{text}: did not exist at {since}; every statement below is new.\n"
            )
            before = ""
        reports.append(render_statement_diff(before, current, text))
    print("\n".join(reports), end="")
    return 0


def _emit_failure(message: str, detail: str) -> None:
    """Print the failure headline and its full diff report.

    The report goes to stderr on every non-zero exit -- no flag required. The
    one-line ``::error::`` annotation goes to stdout only under GitHub Actions,
    which reads workflow commands from there; locally it would be noise.
    """
    print(message, file=sys.stderr)
    print(detail, file=sys.stderr)
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::error::{message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the checked inventory after explicit review",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help=(
            "confirm explicitly that the committed inventory matches the "
            "rebuild; on drift the full report is printed anyway, with or "
            "without this flag"
        ),
    )
    parser.add_argument(
        "--statements",
        nargs="+",
        metavar="PATH",
        help=(
            "print the diagnostic statements in these files; with --since, "
            "print only what changed, separating pure movement/re-indentation "
            "from real content changes. This is the review the report asks for."
        ),
    )
    parser.add_argument(
        "--since",
        metavar="REV",
        help="git revision to compare --statements against (e.g. the pin's commit)",
    )
    args = parser.parse_args()
    if args.statements:
        return _run_statements(args.statements, args.since)
    if args.since:
        parser.error("--since is only meaningful with --statements")
    inventory = build_inventory()
    actual = _encoded(inventory)
    if args.write:
        INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        INVENTORY_PATH.write_text(actual, encoding="utf-8")
        print(f"wrote {INVENTORY_PATH.relative_to(REPO_ROOT)}")
        return 0
    try:
        expected = INVENTORY_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        _emit_failure(
            "diagnostic inventory is missing; review and run with --write",
            f"{INVENTORY_PATH.relative_to(REPO_ROOT)} does not exist, so there "
            "is nothing to diff against. The rebuild found "
            f"{inventory['summary']['owner_files']} owner files and "
            f"{inventory['summary']['persistent_sink_files']} sink files.\n"
            + NEXT_STEPS,
        )
        return 1
    if actual != expected:
        _emit_failure(
            "production diagnostic owners or persistent-sink topology changed; "
            "review the diff below before running --write",
            render_diff(expected, inventory),
        )
        return 1
    if args.diff:
        print("no drift: the committed inventory matches the rebuild exactly.")
    inventory = json.loads(actual)
    summary = inventory["summary"]
    print(
        "diagnostic inventory verified: "
        f"{summary['owner_files']} owners, "
        f"{summary['task_492_calls']} TASK-492 calls, "
        f"{summary['task_31551_calls']} TASK-31551 calls, "
        f"{summary['task_494_calls']} TASK-494 calls, "
        f"{summary['persistent_sink_files']} sink files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
