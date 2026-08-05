"""Sync core implementations for workspace-local agent tools.

Plain functions, no async, no Textual, no event loop — callable from the
agent runtime's worker thread via Agents/local_tool_provider.py. Every
failure raises LocalToolError; the provider converts those (and any other
exception) into ToolResult error strings — nothing raises across the
provider boundary.
"""

from __future__ import annotations

import os
from pathlib import Path

from tldw_chatbook.Utils.path_validation import validate_path

MAX_LIST_ENTRIES = 200
MAX_READ_CHARS = 32 * 1024  # provider byte-fits too; core caps content meaningfully
MAX_GLOB_RESULTS = 100
MAX_GREP_RESULTS = 100
_MAX_GREP_FILE_BYTES = 2 * 1024 * 1024  # skip huge files


class LocalToolError(ValueError):
    """Model-actionable failure from a local tool (path, not-found, …)."""


def resolve_workspace_path(path: str, workspace_root: Path) -> Path:
    """Resolve ``path`` against ``workspace_root``, confined to it.

    Hidden components (``.github/``) are allowed under the root; anything
    resolving outside it is refused. Raises LocalToolError.
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
    Raises LocalToolError when ``path`` is not an existing directory.
    """
    root = resolve_workspace_path(path, workspace_root)
    if not root.is_dir():
        raise LocalToolError(f"not a directory: {path}")
    entries = sorted(
        root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
    )
    lines = [
        f"{p.name}/" if p.is_dir() else p.name for p in entries[:max_entries]
    ]
    remaining = len(entries) - max_entries
    if remaining > 0:
        lines.append(f"… ({remaining} more entries, truncated)")
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


def write_file(path: str, content: str, *, workspace_root: Path) -> str:
    """Create or overwrite ``path`` with ``content`` (full-file write).

    The parent directory must already exist (deliberate divergence from
    claude-code's Write, to catch model path typos early — spec §2).
    """
    root = resolve_workspace_path(path, workspace_root)
    if not root.parent.is_dir():
        raise LocalToolError(f"parent directory does not exist for: {path}")
    root.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to {path}"


def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    *,
    workspace_root: Path,
    replace_all: bool = False,
) -> str:
    """Replace exact ``old_string`` with ``new_string`` in ``path``.

    Fails unless the match is unique (or ``replace_all=True``); ambiguity
    errors include the match count so the model can self-correct. Exact
    semantics per spec §2 (claude-code Edit parity). Reads and writes with
    ``newline=""`` so CRLF files are not silently converted to LF.
    """
    if not old_string:
        raise LocalToolError("old_string must not be empty")
    root = resolve_workspace_path(path, workspace_root)
    if not root.is_file():
        raise LocalToolError(f"file not found: {path}")
    with open(root, encoding="utf-8", newline="") as fh:
        content = fh.read()
    count = content.count(old_string)
    if count == 0:
        raise LocalToolError(f"old_string not found in {path}")
    if count > 1 and not replace_all:
        raise LocalToolError(
            f"old_string appears {count} times in {path}; "
            "provide more context to make it unique, or set replace_all=true"
        )
    updated = content.replace(old_string, new_string)
    root.write_text(updated, encoding="utf-8", newline="")
    n = count if replace_all else 1
    return f"made {n} replacement{'s' if n != 1 else ''} in {path}"


def glob_files(
    pattern: str, *, workspace_root: Path, max_results: int = MAX_GLOB_RESULTS
) -> str:
    """Match ``pattern`` under the workspace, newest-mtime first, capped.

    Paths in the result are workspace-relative. Hidden files/dirs under the
    root ARE matched (workspace policy, ADR-032). Matches that escape the
    root via ``..`` pattern segments are excluded (lexical check only —
    symlinks are not resolved, per ADR-032 review).
    """
    root = workspace_root.resolve()
    matches = [
        p for p in root.glob(pattern)
        if p.is_file() and Path(os.path.normpath(p)).is_relative_to(root)
    ]
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    lines = [str(p.relative_to(root)) for p in matches[:max_results]]
    if len(matches) > max_results:
        lines.append(f"… ({len(matches) - max_results} more, truncated)")
    return "\n".join(lines) if lines else f"(no files matching {pattern!r})"


def grep_files(
    pattern: str,
    *,
    workspace_root: Path,
    mode: str = "content",  # content | files | count
    max_results: int = MAX_GREP_RESULTS,
) -> str:
    """Regex search under the workspace.

    Modes: ``content`` -> ``relpath:lineno:line``; ``files`` -> one relpath
    per matching file; ``count`` -> ``relpath:N``. Binary and >2 MiB files
    are skipped. Invalid regex raises LocalToolError.
    """
    import re

    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise LocalToolError(f"invalid regex: {exc}") from exc
    if mode not in ("content", "files", "count"):
        raise LocalToolError(f"unknown mode: {mode}")
    root = workspace_root.resolve()
    content_hits: list[str] = []
    file_hits: list[str] = []
    count_hits: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.stat().st_size > _MAX_GREP_FILE_BYTES:
            continue
        if not p.resolve().is_relative_to(root):
            continue  # symlink escaping the root — never read outside content
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary/unreadable — skip
        rel = str(p.relative_to(root))
        lines = [f"{i}:{line}" for i, line in enumerate(text.splitlines(), 1) if rx.search(line)]
        if not lines:
            continue
        file_hits.append(rel)
        count_hits.append(f"{rel}:{len(lines)}")
        content_hits.extend(f"{rel}:{hit}" for hit in lines)
    if mode == "files":
        out, total = file_hits, len(file_hits)
    elif mode == "count":
        out, total = count_hits, len(count_hits)
    else:
        out, total = content_hits, len(content_hits)
    shown = out[:max_results]
    if total > max_results:
        shown.append(f"… ({total - max_results} more, truncated)")
    return "\n".join(shown) if shown else f"(no matches for {pattern!r})"
