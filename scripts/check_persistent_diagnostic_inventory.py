#!/usr/bin/env python3
"""Check the reviewed inventory of production diagnostics and disk sinks."""

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


def _call_entry(
    path: Path,
    source: str,
    node: ast.Call,
    *,
    kind: str | None = None,
) -> dict[str, Any]:
    segment = ast.get_source_segment(source, node) or ""
    return {
        "line": node.lineno,
        "method": (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else "call"
        ),
        "digest": hashlib.sha256(segment.encode("utf-8")).hexdigest()[:16],
        **({"kind": kind} if kind else {}),
    }


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


def _scan_file(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    symbols = _logger_symbols(tree)
    diagnostics: list[dict[str, Any]] = []
    sinks: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_diagnostic_call(node, symbols):
            diagnostics.append(_call_entry(path, source, node))
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
                    path,
                    source,
                    node,
                    kind="loguru_sink" if is_loguru_add else call_name,
                )
            )
    diagnostics.sort(key=lambda entry: (entry["line"], entry["method"]))
    sinks.sort(key=lambda entry: (entry["line"], entry["method"]))
    return diagnostics, sinks


def build_inventory() -> dict[str, Any]:
    owners: list[dict[str, Any]] = []
    topology: list[dict[str, Any]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        diagnostics, sinks = _scan_file(path)
        if diagnostics:
            owner, reason = _owner(relative)
            diagnostic_digest = hashlib.sha256(
                json.dumps(diagnostics, sort_keys=True).encode("utf-8")
            ).hexdigest()[:20]
            owners.append(
                {
                    "path": relative,
                    "owner": owner,
                    "reason": reason,
                    "call_count": len(diagnostics),
                    "diagnostic_digest": diagnostic_digest,
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
        "schema_version": 1,
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
