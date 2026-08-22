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
and by how much, and ``NEXT_STEPS`` tells the reader how to recover the
statements themselves.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
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
    names: dict[int, str] = {}
    stack: list[tuple[ast.AST, str]] = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                child_prefix = f"{prefix}.{child.name}" if prefix else child.name
            else:
                child_prefix = prefix
            names[id(child)] = child_prefix
            stack.append((child, child_prefix))
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


NEXT_STEPS = (
    "Next: read every row above and confirm each change is one you intended.\n"
    "  - a call_count delta means a diagnostic was added or deleted;\n"
    "  - an unchanged count with a changed digest means one was reworded,\n"
    "    re-levelled, or given different arguments -- check it does not now\n"
    "    interpolate user content, secrets, or paths into a persistent sink;\n"
    "  - a sink-topology row means a new file/handler destination appeared.\n"
    "The pin stores only an aggregate per-file digest, so the rows above can name\n"
    "WHICH files changed and by how much, never the statement text -- and the\n"
    "interpolation check just above needs that text. Read it with:\n"
    "  base=$(git log -1 --format=%H -- "
    "Docs/security/production-diagnostic-inventory.json)\n"
    "  git diff $base -- <each path listed above>\n"
    "Treat that revision as a LOWER BOUND, not the truth: the pin has been\n"
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
    "reviewed_exclusions",
)


def _sink_key(entry: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(entry.get("scope", "")),
        str(entry.get("kind", "")),
        str(entry.get("method", "")),
        str(entry.get("digest", "")),
    )


def _describe_sink(entry: dict[str, Any]) -> str:
    scope, kind, method, digest = _sink_key(entry)
    return f"{scope or '<module>'}: {kind}.{method} ({digest})"


def _owner_rows(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["path"]): row for row in inventory.get("owners", [])}


def _sink_rows(inventory: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(row["path"]): list(row.get("sinks", []))
        for row in inventory.get("persistent_sink_topology", [])
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
            note = "same count, content changed (reworded / re-levelled / new args)"
        else:
            delta = (new_count or 0) - (old_count or 0)
            note = f"{delta:+d} diagnostic call(s)"
        lines.append(
            f"  ~ changed: {path} "
            f"{old_count}/{old_digest} -> {new_count}/{new_digest}  ({note})"
        )
        if before.get("owner") != after.get("owner"):
            lines.append(
                f"      owner: {before.get('owner')} -> {after.get('owner')}"
            )
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
        before = {_sink_key(entry): entry for entry in old[path]}
        after = {_sink_key(entry): entry for entry in new[path]}
        if before == after:
            continue
        lines.append(
            f"  ~ changed sinks: {path} "
            f"({len(before)} -> {len(after)} entries)"
        )
        for key in sorted(set(before) - set(after)):
            lines.append(f"      - {_describe_sink(before[key])}")
        for key in sorted(set(after) - set(before)):
            lines.append(f"      + {_describe_sink(after[key])}")
    return lines


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
            new_count/new_digest``), sink-topology deltas, metadata deltas,
            and the exact next command. Never empty: a formatting-only drift
            still yields an explanation rather than silence.
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
        ("inventory metadata", _metadata_lines(committed, rebuilt)),
    )
    body = [
        line
        for title, lines in sections
        if lines
        for line in (f"{title}:", *lines)
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
    args = parser.parse_args()
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
        f"{summary['task_494_calls']} TASK-494 calls, "
        f"{summary['persistent_sink_files']} sink files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
