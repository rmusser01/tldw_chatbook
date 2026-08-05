"""Sync core implementations for workspace-local agent tools.

Plain functions, no async, no Textual, no event loop — callable from the
agent runtime's worker thread via Agents/local_tool_provider.py. Every
failure raises LocalToolError; the provider converts those (and any other
exception) into ToolResult error strings — nothing raises across the
provider boundary.
"""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Utils.path_validation import validate_path

MAX_LIST_ENTRIES = 200
#: Upper bound on how many directory entries ``list_directory`` will even
#: LOOK AT before giving up — without it, a pathological directory (a
#: million-entry build tree) is materialized and sorted in full before the
#: ``max_entries`` display cap ever applies.
MAX_SCAN_ENTRIES = 10_000
MAX_READ_CHARS = 32 * 1024  # provider byte-fits too; core caps content meaningfully


class LocalToolError(ValueError):
    """Model-actionable failure from a local tool (path, not-found, …)."""


def resolve_workspace_path(path: str, workspace_root: Path) -> Path:
    """Resolve ``path`` against ``workspace_root``, confined to it.

    Hidden components (``.github/``) are allowed under the root; anything
    resolving outside it is refused.

    Args:
        path: The user/model-supplied path, absolute or relative to
            ``workspace_root``.
        workspace_root: The confinement root the resolved path must stay
            within.

    Returns:
        The validated absolute ``Path`` inside ``workspace_root``.

    Raises:
        LocalToolError: If the path resolves outside ``workspace_root``.
    """
    try:
        return validate_path(path, workspace_root, allow_hidden=True)
    except ValueError as exc:
        raise LocalToolError(
            f"Path '{path}' is outside the workspace root ({workspace_root})"
        ) from exc


def list_directory(
    path: str, *, workspace_root: Path, max_entries: int = MAX_LIST_ENTRIES
) -> str:
    """One-level listing of ``path``: ``name/`` for dirs, ``name`` for files.

    Directories sort before files, each group case-insensitively by name.
    Output is capped at ``max_entries`` with a trailing truncation notice.
    The directory SCAN itself is also capped at ``MAX_SCAN_ENTRIES`` — only
    the scanned entries are sorted (dirs-first contract preserved for the
    scanned set), and hitting the scan cap appends a "directory too large"
    notice instead of silently presenting a partial listing as complete.

    Args:
        path: Directory to list, absolute or relative to
            ``workspace_root``.
        workspace_root: The confinement root ``path`` must resolve within.
        max_entries: Maximum number of entries included in the output
            before a truncation notice is appended.

    Returns:
        The newline-joined listing, with a truncation and/or scan-cap
        notice appended when the directory exceeded either cap.

    Raises:
        LocalToolError: If ``path`` is not an existing directory, or
            resolves outside ``workspace_root``.
    """
    root = resolve_workspace_path(path, workspace_root)
    if not root.is_dir():
        raise LocalToolError(f"not a directory: {path}")
    scanned: list[Path] = []
    scan_capped = False
    for index, entry in enumerate(root.iterdir()):
        if index >= MAX_SCAN_ENTRIES:
            scan_capped = True
            break
        scanned.append(entry)
    entries = sorted(scanned, key=lambda p: (p.is_file(), p.name.lower()))
    lines = [
        f"{p.name}/" if p.is_dir() else p.name for p in entries[:max_entries]
    ]
    remaining = len(entries) - max_entries
    if remaining > 0:
        lines.append(f"… ({remaining} more entries, truncated)")
    if scan_capped:
        lines.append(
            f"… (directory too large; showing first {len(entries)} of many entries)"
        )
    return "\n".join(lines)


def read_file(
    path: str,
    *,
    workspace_root: Path,
    offset: int = 1,
    limit: int | None = None,
) -> str:
    """Read ``path`` with 1-based line numbers, ``offset``/``limit`` paging.

    Lines are numbered from 1 (matching claude-code's Read). ``offset`` is
    the 1-based first line to return; ``limit`` caps the line count.
    Binary files (NUL byte in the first 8 KiB) and missing files raise
    LocalToolError with model-actionable messages.
    """
    root = resolve_workspace_path(path, workspace_root)
    if not root.is_file():
        raise LocalToolError(f"file not found: {path}")
    with open(root, "rb") as fh:
        sniff = fh.read(8192)
    if b"\x00" in sniff:
        raise LocalToolError(f"'{path}' appears to be binary; fs_read only reads text files")
    text = root.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = max(offset, 1) - 1
    if start >= len(lines) and lines:
        return f"(offset {offset} is past end of file; {len(lines)} lines total)"
    window = lines[start:] if limit is None else lines[start:start + max(limit, 0)]
    numbered = "\n".join(f"{i}\t{line}" for i, line in enumerate(window, start=start + 1))
    if len(numbered) > MAX_READ_CHARS:
        numbered = numbered[:MAX_READ_CHARS] + "\n… [truncated]"
    return numbered
