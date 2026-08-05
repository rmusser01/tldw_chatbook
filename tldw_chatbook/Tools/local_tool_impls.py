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
#: Upper bound on how many directory entries ``list_directory`` will even
#: LOOK AT before giving up — without it, a pathological directory (a
#: million-entry build tree) is materialized and sorted in full before the
#: ``max_entries`` display cap ever applies.
MAX_SCAN_ENTRIES = 10_000
MAX_READ_CHARS = 32 * 1024  # provider byte-fits too; core caps content meaningfully
MAX_GLOB_RESULTS = 100
MAX_GREP_RESULTS = 100
_MAX_GREP_FILE_BYTES = 2 * 1024 * 1024  # skip huge files


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
    LocalToolError with model-actionable messages. UTF-16 files trip the
    binary sniff; other non-UTF-8 text reads with U+FFFD replacement.
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
    if not lines:
        return "(empty file)"
    start = max(offset, 1) - 1
    if start >= len(lines):
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
    try:
        data = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise LocalToolError(
            f"content is not UTF-8 encodable (lone surrogate?): {exc}"
        ) from exc
    # encode BEFORE opening for write — a failed encode must never
    # truncate an existing file
    root.write_bytes(data)
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
    ``newline=""`` so CRLF files are not silently converted to LF. The
    result is encoded BEFORE the file is opened for writing, so an
    unencodable ``new_string`` (e.g. a lone surrogate from tool-call JSON)
    fails without truncating the file.
    """
    if not old_string:
        raise LocalToolError("old_string must not be empty")
    if old_string == new_string:
        raise LocalToolError("old_string and new_string are identical")
    root = resolve_workspace_path(path, workspace_root)
    if not root.is_file():
        raise LocalToolError(f"file not found: {path}")
    try:
        with open(root, encoding="utf-8", newline="") as fh:
            content = fh.read()
    except UnicodeDecodeError as exc:
        raise LocalToolError(
            f"'{path}' is not valid UTF-8; fs_edit only edits text files"
        ) from exc
    count = content.count(old_string)
    if count == 0:
        raise LocalToolError(f"old_string not found in {path}")
    if count > 1 and not replace_all:
        raise LocalToolError(
            f"old_string appears {count} times in {path}; "
            "provide more context to make it unique, or set replace_all=true"
        )
    updated = content.replace(old_string, new_string)
    try:
        data = updated.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise LocalToolError(
            f"new_string is not UTF-8 encodable (lone surrogate?): {exc}"
        ) from exc
    root.write_bytes(data)
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
    # Render from the normpath'd path so `..` re-entry patterns
    # ("../<wsname>/*.py") stay workspace-relative instead of "../…".
    lines = [
        str(Path(os.path.normpath(p)).relative_to(root))
        for p in matches[:max_results]
    ]
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
