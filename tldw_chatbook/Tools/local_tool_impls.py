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
MAX_READ_CHARS = 32 * 1024  # provider byte-fits too; core caps content meaningfully


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
    semantics per spec §2 (claude-code Edit parity).
    """
    if not old_string:
        raise LocalToolError("old_string must not be empty")
    root = resolve_workspace_path(path, workspace_root)
    if not root.is_file():
        raise LocalToolError(f"file not found: {path}")
    content = root.read_text(encoding="utf-8")
    count = content.count(old_string)
    if count == 0:
        raise LocalToolError(f"old_string not found in {path}")
    if count > 1 and not replace_all:
        raise LocalToolError(
            f"old_string appears {count} times in {path}; "
            "provide more context to make it unique, or set replace_all=true"
        )
    updated = content.replace(old_string, new_string)
    root.write_text(updated, encoding="utf-8")
    n = count if replace_all else 1
    return f"made {n} replacement{'s' if n != 1 else ''} in {path}"
