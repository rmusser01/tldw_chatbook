#!/usr/bin/env python3
"""Check the reviewed inventory of production diagnostics and disk sinks.

The inventory is keyed on the CONTENT of each diagnostic and sink -- the
statement's own source text plus its log method -- and never on where in the
file it sits.  Moving a logger call is therefore not a review event, while
adding, deleting, rewording, or re-levelling one still is.  See task-3750: a
digest that fires on pure line movement trains reviewers to regenerate this
file without reading it, which is the one failure mode it exists to prevent.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import warnings
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


def _scope_names(tree: ast.Module) -> dict[int, str]:
    """Map every node to the dotted name of its enclosing def/class scope.

    Used to give persistent-sink entries a human navigation handle that, unlike
    a line number, survives unrelated edits elsewhere in the file.
    """
    names: dict[int, str] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                child_prefix = f"{prefix}.{child.name}" if prefix else child.name
            else:
                child_prefix = prefix
            names[id(child)] = child_prefix
            visit(child, child_prefix)

    visit(tree, "")
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
    """
    content = sorted(
        (entry["method"], entry["digest"]) for entry in diagnostics
    )
    return hashlib.sha256(
        json.dumps(content, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def _owner(path_text: str) -> tuple[str, str]:
    if path_text in TASK_492_FILES or path_text.startswith(TASK_492_PREFIXES):
        return (
            "TASK-492",
            "high-risk Chat/provider/summarization/tool/MCP diagnostic owner",
        )
    return (
        "TASK-494",
        "remaining Chatbook production diagnostic owner",
    )


def scan_source(
    source: str, *, filename: str = "<source>"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the (diagnostics, sinks) content entries for one module's source."""
    tree = ast.parse(source, filename=filename)
    symbols = _logger_symbols(tree)
    scopes = _scope_names(tree)
    diagnostics: list[dict[str, Any]] = []
    sinks: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_diagnostic_call(node, symbols):
            diagnostics.append(_call_entry(source, node))
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
                    scope=scopes.get(id(node), ""),
                )
            )
    diagnostics.sort(key=lambda entry: (entry["method"], entry["digest"]))
    sinks.sort(
        key=lambda entry: (
            entry["scope"],
            entry["kind"],
            entry["method"],
            entry["digest"],
        )
    )
    return diagnostics, sinks


def _scan_file(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return scan_source(path.read_text(encoding="utf-8"), filename=str(path))


def build_inventory() -> dict[str, Any]:
    owners: list[dict[str, Any]] = []
    topology: list[dict[str, Any]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        diagnostics, sinks = _scan_file(path)
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

    task_492_calls = sum(
        entry["call_count"] for entry in owners if entry["owner"] == "TASK-492"
    )
    task_494_calls = sum(
        entry["call_count"] for entry in owners if entry["owner"] == "TASK-494"
    )
    return {
        # 2: digests and sink entries are keyed on diagnostic CONTENT only.
        # Line numbers are no longer an input, so v1 and v2 digests for an
        # unchanged file differ and must never be compared across the bump.
        "schema_version": 2,
        "scope": "tldw_chatbook/**/*.py",
        "classification_rules": {
            "TASK-492": {
                "prefixes": list(TASK_492_PREFIXES),
                "files": sorted(TASK_492_FILES),
                "reason": "Chat, provider, summarization, tool, and MCP paths",
            },
            "TASK-494": {
                "rule": "all other production diagnostic owners",
                "reason": "remaining production domains",
            },
        },
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
            "task_494_calls": task_494_calls,
            "persistent_sink_files": len(topology),
        },
        "owners": owners,
        "persistent_sink_topology": topology,
    }


def _encoded(inventory: dict[str, Any]) -> str:
    return json.dumps(inventory, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the checked inventory after explicit review",
    )
    args = parser.parse_args()
    actual = _encoded(build_inventory())
    if args.write:
        INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        INVENTORY_PATH.write_text(actual, encoding="utf-8")
        print(f"wrote {INVENTORY_PATH.relative_to(REPO_ROOT)}")
        return 0
    try:
        expected = INVENTORY_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(
            "diagnostic inventory is missing; review and run with --write",
            file=sys.stderr,
        )
        return 1
    if actual != expected:
        print(
            "production diagnostic owners or persistent-sink topology changed; "
            "review the diff before running --write",
            file=sys.stderr,
        )
        return 1
    inventory = json.loads(actual)
    summary = inventory["summary"]
    print(
        "diagnostic inventory verified: "
        f"{summary['owner_files']} owners, "
        f"{summary['task_492_calls']} TASK-492 calls, "
        f"{summary['task_494_calls']} TASK-494 calls, "
        f"{summary['persistent_sink_files']} sink files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
