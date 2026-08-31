"""Sync core implementations for workspace-local agent tools.

Plain functions, no async, no Textual, no event loop — callable from the
agent runtime's worker thread via Agents/local_tool_provider.py. Every
failure raises LocalToolError; the provider converts those (and any other
exception) into ToolResult error strings — nothing raises across the
provider boundary.

Path safety has TWO layers, and both are mandatory (TASK-19551):

1. **Confinement** to the configured ``[console] workspace_root``
   (ADR-032), enforced by ``validate_path`` inside
   ``resolve_workspace_path`` — the single choke point every path-taking
   function here (and in ``patch_tool_impls``/``git_tool_impls``) funnels
   through.
2. **The sensitive-path denylist** (``Utils/sensitive_paths.py``), also
   enforced inside that same choke point. Confinement alone is not enough:
   the shipped ``workspace_root`` default is the app's cwd at startup, so
   launching from ``$HOME`` makes ``$HOME`` the confinement root — and
   ``~/.ssh/id_rsa``, ``~/.aws/credentials``, this app's own ``config.toml``
   and ``mcp_permissions.json`` are all inside it. Reading them exfiltrates
   credentials into a transcript sent to a provider; WRITING
   ``mcp_permissions.json`` turns every ``ask`` into ``allow``, a one-step
   bypass of the permission gate that authorized the call.

The choke point covers a path the model NAMES. It cannot cover entries a
tool presents that the model never named, so the three ENUMERATING tools
(``list_directory``/``glob_files``/``grep_files``) — which resolve only the
workspace ROOT through it — each filter their own candidates against the
same denylist, resolving the sensitive-path context ONCE per invocation
(see ``Utils.sensitive_paths.resolve_sensitive_context``) rather than once
per candidate. ``grep_files`` is the sharpest of the three — it READS every
file it walks and prints matching lines — so its check runs before the
read, not after.

``Tools/git_tool_impls.py`` shares this choke point for its path arguments,
and ``path`` is optional on ``git_status``/``git_log``/``git_diff``: with
it omitted the choke point sees only the repository root, and ``git_diff``
returned the CONTENT of a denylisted file from a CLEAN worktree
(TASK-19632). That family solves the same problem a third way -- it cannot
filter candidates it never sees, because git enumerates the repository for
it, so it instead excludes denylisted paths from git's INPUT by pathspec
(``git_tool_impls._denylist_pathspecs``). Do not extend that family
assuming the choke point alone makes its output safe: whichever of the
three mechanisms fits, a tool that PRESENTS paths the model never named
needs one.
"""

from __future__ import annotations

import difflib
import hashlib
import heapq
import json
import os
import stat as stat_module
import threading
import uuid
from pathlib import Path
from typing import Literal

from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.Utils.sensitive_paths import (
    is_git_metadata_write,
    SensitivePathContext,
    SensitiveExclusion,
    is_sensitive_path,
    refuses_new_directory_chain,
    resolve_sensitive_context,
    sensitive_exclusions_under,
)

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
_MAX_WRITE_DIFF_CHARS = 12 * 1024
_WRITE_LOCKS_GUARD = threading.Lock()
_WRITE_LOCKS: dict[str, threading.Lock] = {}


class LocalToolError(ValueError):
    """Model-actionable failure from a local tool (path, not-found, …)."""


#: Intent of the access a caller is about to perform, threaded into the
#: choke point below. It selects the refusal verb AND, for ``"write"``,
#: the extra ``refuses_new_directory_chain`` guard — never a weaker check.
PathIntent = Literal["read", "write", "list"]

_INTENT_VERBS: dict[str, str] = {"read": "read", "write": "written", "list": "listed"}


def resolve_workspace_path(
    path: str,
    workspace_root: Path,
    *,
    intent: PathIntent = "read",
    context: SensitivePathContext | None = None,
) -> Path:
    """Resolve ``path`` against ``workspace_root``, confined and denylisted.

    The single choke point for this tool family: every ``fs_*`` and
    ``git_*`` core function resolves a model-supplied path here (the git
    ones one hop away, via ``prepare_repository``/``_prepare_for_path``/
    ``_repo_relative_path``), so both path checks are enforced in ONE place
    rather than re-implemented per tool — which is exactly how the denylist
    came to be missing from all seven ``fs_*`` tools (TASK-19551). It
    governs the path a caller PASSES; it does not filter what a tool
    returns (see the module docstring for where that distinction bites).

    Two checks, in order:

    1. **Confinement** (``validate_path``). Hidden components (``.github/``,
       ``.gitignore``) are allowed under the root, per ADR-032: a coding
       agent that cannot read a repository's dotfile configuration is
       useless, and the ADR adopted ``allow_hidden`` for exactly this
       family. That parameter is deliberately KEPT — dotted names are how
       ``~/.ssh``/``~/.aws`` are spelled, but "starts with a dot" is a name
       heuristic, not a security boundary, and check 2 is the one designed
       to answer that question (by resolved ancestry, so a symlink or a
       ``~/.sshfoo`` lookalike cannot game it).
    2. **The sensitive-path denylist** (``is_sensitive_path``), matching
       what ``Tools/file_operation_tools.py``'s ``ReadFileTool``/
       ``WriteFileTool``/``ListDirectoryTool`` already do for the other
       file-tool family, message shape included, so agent-facing refusals
       stay consistent across the two.

    For ``intent="write"`` the denylist check is additionally applied to
    every not-yet-existing ancestor of the target
    (``refuses_new_directory_chain``), the same guard ``WriteFileTool``
    consults before ``mkdir(parents=True)``. No tool in this family creates
    directories today (a write target's parent must already exist), so this
    normally short-circuits on the first existing ancestor; it is here so
    that a future ``create_directories``-style option cannot reintroduce
    TASK-849's shadow-directory denial of service by forgetting it.

    Args:
        path: The user/model-supplied path, absolute or relative to
            ``workspace_root``.
        workspace_root: The confinement root the resolved path must stay
            within.
        intent: What the caller is about to do with the path — selects the
            refusal verb and enables the new-directory-chain guard for
            writes. Never weakens a check.
        context: Optional pre-resolved ``SensitivePathContext``. Callers
            that check many paths in one tool invocation (the enumerating
            tools, ``patch_files``' multi-file loop) resolve one with
            ``Utils.sensitive_paths.resolve_sensitive_context()`` and pass
            it through, so the ~11 config accessors behind the denylist are
            resolved once per CALL rather than once per path. ``None``
            resolves it fresh — that still enforces the denylist.

    Returns:
        The validated absolute ``Path`` inside ``workspace_root``.

    Raises:
        LocalToolError: If the path resolves outside ``workspace_root``, or
            is a protected credential/gate-state/app-state path.
    """
    try:
        # `redact_paths=True` (TASK-19558): `path` here is MODEL-supplied and
        # `validate_path` otherwise echoes it, and its resolved form, into a
        # WARNING/ERROR log line -- so a prompt-injected traversal probe
        # writes attacker-chosen text (and the user's real directory layout)
        # into the diagnostics bundle. Only 9 of this function's ~30 sibling
        # call sites passed it. This does not weaken what the MODEL sees:
        # the refusal below is built from `path` locally and is unchanged;
        # redaction bounds only the log line and the discarded ValueError.
        resolved = validate_path(
            path, workspace_root, redact_paths=True, allow_hidden=True
        )
    except ValueError as exc:
        raise LocalToolError(
            f"Path '{path}' is outside the workspace root ({workspace_root})"
        ) from exc

    verb = _INTENT_VERBS.get(intent, "accessed")
    if is_sensitive_path(resolved, context=context):
        raise LocalToolError(
            f"Refused: '{path}' is a protected path and cannot be {verb}"
        )
    if intent == "write" and is_git_metadata_write(resolved):
        # TASK-19700: the upstream cause of TASK-16801's four
        # repository-supplied argv vectors. Write-only and checked here
        # rather than folded into `is_sensitive_path`, which also governs
        # reads -- reading repository state stays legitimate.
        raise LocalToolError(
            f"Refused: '{path}' is inside a repository's .git metadata and "
            f"cannot be {verb}"
        )
    if intent == "write" and refuses_new_directory_chain(
        resolved.parent, context=context
    ):
        raise LocalToolError(
            f"Refused: creating '{resolved.parent}' would collide with a protected path"
        )
    return resolved


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

    Individual denylisted ENTRIES are omitted from the listing (TASK-19551),
    mirroring ``ListDirectoryTool``'s per-entry check in
    ``Tools/file_operation_tools.py``: refusing the target directory alone
    would still disclose this app's own ``mcp_permissions.json`` or
    ``chachanotes.db`` by name and existence whenever an ordinary,
    listable ancestor happens to contain them.

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
    sensitive_ctx = resolve_sensitive_context()
    root = resolve_workspace_path(
        path, workspace_root, intent="list", context=sensitive_ctx
    )
    return _list_relative_directory(
        root.relative_to(Path(workspace_root).resolve()),
        workspace=Path(workspace_root).resolve(),
        max_entries=max_entries,
        sensitive_exclusions=sensitive_exclusions_under(
            Path(workspace_root).resolve(), sensitive_ctx
        ),
        display_path=path,
    )


def _list_relative_directory(
    relative: Path,
    *,
    workspace: Path,
    max_entries: int,
    sensitive_exclusions: tuple[SensitiveExclusion, ...],
    display_path: str | None = None,
) -> str:
    """List a pinned-root-relative directory without opening an absolute path."""
    target = workspace / relative
    if not _relative_target_is_safe(
        relative, workspace, sensitive_exclusions, is_directory=True
    ) or not target.is_dir():
        raise LocalToolError(f"not a directory: {display_path or relative}")
    scanned: list[Path] = []
    scan_capped = False
    for index, entry in enumerate(target.iterdir()):
        if index >= MAX_SCAN_ENTRIES:
            scan_capped = True
            break
        # Skipped entries still count against the scan cap: the cap bounds
        # the WORK done, and a denied entry was still scanned.
        entry_relative = _workspace_relative_path(entry, workspace)
        if not _relative_target_is_safe(
            entry_relative, workspace, sensitive_exclusions, is_directory=entry.is_dir()
        ):
            continue
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
    root = resolve_workspace_path(path, workspace_root, intent="read")
    return _read_relative_file(
        root.relative_to(Path(workspace_root).resolve()),
        workspace=Path(workspace_root).resolve(),
        offset=offset,
        limit=limit,
        sensitive_exclusions=sensitive_exclusions_under(Path(workspace_root).resolve()),
        display_path=path,
    )


def _read_relative_file(
    relative: Path,
    *,
    workspace: Path,
    offset: int,
    limit: int | None,
    sensitive_exclusions: tuple[SensitiveExclusion, ...],
    display_path: str | None = None,
) -> str:
    """Read a pinned-root-relative text file without reopening its resolved path."""
    target = workspace / relative
    if not _relative_target_is_safe(
        relative, workspace, sensitive_exclusions, is_directory=False
    ) or not target.is_file():
        raise LocalToolError(f"file not found: {display_path or relative}")
    with open(target, "rb") as fh:
        sniff = fh.read(8192)
    if b"\x00" in sniff:
        raise LocalToolError(
            f"'{display_path or relative}' appears to be binary; fs_read only reads text files"
        )
    text = target.read_text(encoding="utf-8", errors="replace")
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


def stat_path(path: str, *, workspace_root: Path) -> str:
    """Return a small allowlisted metadata view for one workspace path.

    Args:
        path: File or directory to inspect.
        workspace_root: Confinement root the path must resolve within.

    Returns:
        Workspace-relative path, kind, size, nanosecond mtime, and mode.

    Raises:
        LocalToolError: If the path is outside the workspace or protected.
        OSError: If the resolved path cannot be inspected.
    """
    root = Path(workspace_root).resolve()
    resolved = resolve_workspace_path(path, root, intent="read")
    relative = resolved.relative_to(root)
    return _format_stat_result(relative, resolved.stat())


def _stat_relative_path(relative: Path) -> str:
    """Inspect one already-validated path relative to the pinned worker root."""
    if relative.is_absolute() or ".." in relative.parts:
        raise LocalToolError("stat path must be workspace-relative")
    return _format_stat_result(relative, relative.stat())


def _format_stat_result(relative: Path, info: os.stat_result) -> str:
    """Format the stable allowlisted stat fields for one relative path."""
    kind = (
        "directory"
        if stat_module.S_ISDIR(info.st_mode)
        else "file"
        if stat_module.S_ISREG(info.st_mode)
        else "other"
    )
    return "\n".join(
        (
            f"path: {relative}",
            f"type: {kind}",
            f"size: {info.st_size}",
            f"modified_ns: {info.st_mtime_ns}",
            f"mode: {info.st_mode & 0o7777:04o}",
        )
    )


def write_file(
    path: str,
    content: str,
    *,
    workspace_root: Path,
    dry_run: bool = False,
    expected_sha256: str | None = None,
    expected_absent: bool = False,
) -> str:
    """Create or overwrite ``path`` with ``content`` (full-file write).

    The parent directory must already exist (deliberate divergence from
    claude-code's Write, to catch model path typos early — spec §2).
    """
    workspace = Path(workspace_root).resolve()
    root = resolve_workspace_path(path, workspace, intent="write")
    return _write_relative_file(
        root.relative_to(workspace),
        content,
        workspace=workspace,
        display_path=path,
        dry_run=dry_run,
        expected_sha256=expected_sha256,
        expected_absent=expected_absent,
    )


def _write_relative_file(
    relative: Path,
    content: str,
    *,
    workspace: Path,
    display_path: str | None = None,
    dry_run: bool = False,
    expected_sha256: str | None = None,
    expected_absent: bool = False,
) -> str:
    """Preview or atomically write one admitted path with optional CAS."""
    target = workspace / relative
    shown = display_path or str(relative)
    if not target.parent.is_dir():
        raise LocalToolError(f"parent directory does not exist for: {shown}")
    if expected_sha256 is not None and expected_absent:
        raise LocalToolError(
            "expected_sha256 and expected_absent are mutually exclusive"
        )
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise LocalToolError("expected_sha256 must be a lowercase SHA-256 digest")
    if type(expected_absent) is not bool or type(dry_run) is not bool:
        raise LocalToolError("write flags must be boolean")
    try:
        data = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise LocalToolError(
            f"content is not UTF-8 encodable (lone surrogate?): {exc}"
        ) from exc
    canonical_key = os.path.normcase(str(target.absolute()))
    lock = _write_lock_for(canonical_key)
    with lock:
        current = _read_write_target(target, shown)
        current_digest = (
            None if current is None else hashlib.sha256(current).hexdigest()
        )
        if expected_absent and current is not None:
            raise LocalToolError("write precondition failed: target is present")
        if expected_sha256 is not None and current_digest != expected_sha256:
            raise LocalToolError("write precondition failed: target digest changed")
        if dry_run:
            return _write_preview_json(
                shown=shown,
                current=current,
                current_digest=current_digest,
                replacement=data,
            )
        _atomic_write_target(
            target,
            data,
            shown=shown,
            expected_sha256=expected_sha256,
            expected_absent=expected_absent,
        )
    return f"wrote {len(content)} characters to {shown}"


def _write_lock_for(key: str) -> threading.Lock:
    with _WRITE_LOCKS_GUARD:
        return _WRITE_LOCKS.setdefault(key, threading.Lock())


def _read_write_target(target: Path, shown: str) -> bytes | None:
    try:
        before = os.lstat(target)
    except FileNotFoundError:
        return None
    if (
        not stat_module.S_ISREG(before.st_mode)
        or stat_module.S_ISLNK(before.st_mode)
    ):
        raise LocalToolError(f"write target is not a regular file: {shown}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags)
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            finished = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = os.lstat(target)
    except OSError:
        raise LocalToolError(f"write target changed while reading: {shown}") from None
    identities = tuple(
        (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
        )
        for value in (before, opened, finished, after)
    )
    if len(set(identities)) != 1:
        raise LocalToolError(f"write target changed while reading: {shown}")
    return b"".join(chunks)


def _write_preview_json(
    *,
    shown: str,
    current: bytes | None,
    current_digest: str | None,
    replacement: bytes,
) -> str:
    before_text = "" if current is None else current.decode("utf-8", errors="replace")
    after_text = replacement.decode("utf-8")
    diff = "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=shown if current is not None else "/dev/null",
            tofile=shown,
        )
    )
    if len(diff) > _MAX_WRITE_DIFF_CHARS:
        diff = diff[:_MAX_WRITE_DIFF_CHARS] + "\n… [truncated]"
    return json.dumps(
        {
            "target_state": "absent" if current is None else "present",
            "current_sha256": current_digest or "absent",
            "replacement_sha256": hashlib.sha256(replacement).hexdigest(),
            "replacement_bytes": len(replacement),
            "diff": diff,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write_target(
    target: Path,
    data: bytes,
    *,
    shown: str,
    expected_sha256: str | None,
    expected_absent: bool,
) -> None:
    """Write through one pinned parent descriptor and same-directory temp."""
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        parent_fd = os.open(target.parent, directory_flags)
    except OSError:
        raise LocalToolError(f"write parent changed: {shown}") from None
    temp_name = f".chatbook-write-{uuid.uuid4().hex}.tmp"
    temp_created = False
    try:
        # Recheck after acquiring the retained directory descriptor.  This is
        # the mutation boundary; no later path traversal is used.
        live, live_mode = _read_relative_descriptor(parent_fd, target.name, shown)
        live_digest = None if live is None else hashlib.sha256(live).hexdigest()
        if expected_absent and live is not None:
            raise LocalToolError("write precondition failed: target is present")
        if expected_sha256 is not None and live_digest != expected_sha256:
            raise LocalToolError("write precondition failed: target digest changed")

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        temp_fd = os.open(temp_name, flags, 0o666, dir_fd=parent_fd)
        temp_created = True
        try:
            if live_mode is not None:
                os.fchmod(temp_fd, live_mode)
            view = memoryview(data)
            while view:
                written = os.write(temp_fd, view)
                view = view[written:]
            os.fsync(temp_fd)
        finally:
            os.close(temp_fd)
        if expected_absent:
            os.link(
                temp_name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            os.unlink(temp_name, dir_fd=parent_fd)
            temp_created = False
        else:
            os.replace(
                temp_name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            temp_created = False
        os.fsync(parent_fd)
    except FileExistsError:
        raise LocalToolError("write precondition failed: target is present") from None
    except LocalToolError:
        raise
    except OSError:
        raise LocalToolError(f"atomic write failed for: {shown}") from None
    finally:
        if temp_created:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except OSError:
                pass
        os.close(parent_fd)


def _read_relative_descriptor(
    parent_fd: int, filename: str, shown: str
) -> tuple[bytes | None, int | None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(filename, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return None, None
    except OSError:
        raise LocalToolError(f"write target changed: {shown}") from None
    try:
        info = os.fstat(descriptor)
        if not stat_module.S_ISREG(info.st_mode):
            raise LocalToolError(f"write target is not a regular file: {shown}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), info.st_mode & 0o7777
    finally:
        os.close(descriptor)


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
    workspace = Path(workspace_root).resolve()
    root = resolve_workspace_path(path, workspace, intent="write")
    return _edit_relative_file(
        root.relative_to(workspace),
        old_string,
        new_string,
        workspace=workspace,
        replace_all=replace_all,
        display_path=path,
    )


def _edit_relative_file(
    relative: Path,
    old_string: str,
    new_string: str,
    *,
    workspace: Path,
    replace_all: bool = False,
    display_path: str | None = None,
) -> str:
    """Edit one already-admitted path relative to an I/O root."""
    shown = display_path or str(relative)
    if not old_string:
        raise LocalToolError("old_string must not be empty")
    if old_string == new_string:
        raise LocalToolError("old_string and new_string are identical")
    target = workspace / relative
    if not target.is_file():
        raise LocalToolError(f"file not found: {shown}")
    try:
        with open(target, encoding="utf-8", newline="") as fh:
            content = fh.read()
    except UnicodeDecodeError as exc:
        raise LocalToolError(
            f"'{shown}' is not valid UTF-8; fs_edit only edits text files"
        ) from exc
    count = content.count(old_string)
    if count == 0:
        raise LocalToolError(f"old_string not found in {shown}")
    if count > 1 and not replace_all:
        raise LocalToolError(
            f"old_string appears {count} times in {shown}; "
            "provide more context to make it unique, or set replace_all=true"
        )
    updated = content.replace(old_string, new_string)
    try:
        data = updated.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise LocalToolError(
            f"new_string is not UTF-8 encodable (lone surrogate?): {exc}"
        ) from exc
    target.write_bytes(data)
    n = count if replace_all else 1
    return f"made {n} replacement{'s' if n != 1 else ''} in {shown}"


def glob_files(
    pattern: str, *, workspace_root: Path, max_results: int = MAX_GLOB_RESULTS
) -> str:
    """Match ``pattern`` under the workspace, newest-mtime first, capped.

    Paths in the result are workspace-relative. Hidden files/dirs under the
    root ARE matched (workspace policy, ADR-032). Matches that escape the
    root via ``..`` pattern segments are excluded (lexical check only —
    symlinks are not resolved, per ADR-032 review). Denylisted matches are
    excluded too (TASK-19551): under a home-rooted workspace ``**/*`` would
    otherwise report ``.ssh/id_rsa`` back to the model by name.

    Memory is bounded at ``max_results``: a min-heap keeps only the newest N
    while the total is counted in one pass (no full-list materialization or
    sort of a huge workspace).

    Raises:
        LocalToolError: If ``max_results`` is below 1.
    """
    if max_results < 1:
        raise LocalToolError("max_results must be >= 1")
    sensitive_ctx = resolve_sensitive_context()
    root = resolve_workspace_path(
        ".", workspace_root, intent="list", context=sensitive_ctx
    )
    return _glob_relative_files(
        pattern,
        workspace=Path(workspace_root).resolve(),
        max_results=max_results,
        sensitive_exclusions=sensitive_exclusions_under(root, sensitive_ctx),
    )


def _glob_relative_files(
    pattern: str,
    *,
    workspace: Path,
    max_results: int,
    sensitive_exclusions: tuple[SensitiveExclusion, ...],
    validate_targets: bool = False,
) -> str:
    """Glob from the pinned working directory using only relative I/O paths."""
    heap: list[tuple[float, Path]] = []  # min-heap of (mtime, normpath)
    total = 0
    for p in workspace.glob(pattern):
        try:
            if not p.is_file():
                continue
            norm = Path(os.path.normpath(p))
            if not norm.is_relative_to(workspace):
                continue
            rendered = norm.relative_to(workspace)
            if validate_targets:
                # The pinned worker validates a live link target immediately
                # before disclosure, but keeps ``rendered`` for all I/O/output.
                if not _relative_target_is_safe(
                    rendered, workspace, sensitive_exclusions, is_directory=False
                ):
                    continue
            elif _is_relative_sensitive_path(
                rendered, sensitive_exclusions, is_directory=False
            ):
                continue
            mtime = p.stat().st_mtime
        except OSError:
            continue  # racy/unreadable entry — skip it, not the whole search
        total += 1
        if len(heap) < max_results:
            heapq.heappush(heap, (mtime, rendered))
        elif mtime > heap[0][0]:
            heapq.heapreplace(heap, (mtime, rendered))
    best = sorted(heap, key=lambda t: t[0], reverse=True)
    lines = [str(relative) for _, relative in best]
    if total > max_results:
        lines.append(f"… ({total - max_results} more, truncated)")
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
    are skipped. Invalid regex raises LocalToolError. File order is
    filesystem order (no global sort — that would materialize the whole
    tree); within a file, lines are in order.

    Denylisted files are skipped BEFORE they are read (TASK-19551): this is
    the sharpest of the three enumerating tools, since ``content`` mode
    prints matching LINES — a home-rooted workspace and a pattern as bland
    as ``KEY`` would otherwise emit ``~/.ssh/id_rsa`` into the transcript.

    Raises:
        LocalToolError: If ``max_results`` is below 1.
    """
    import re

    try:
        re.compile(pattern)
    except re.error as exc:
        raise LocalToolError(f"invalid regex: {exc}") from exc
    if mode not in {"content", "files", "count"}:
        raise LocalToolError(f"unknown mode: {mode}")
    if max_results < 1:
        raise LocalToolError("max_results must be >= 1")
    sensitive_ctx = resolve_sensitive_context()
    root = resolve_workspace_path(
        ".", workspace_root, intent="list", context=sensitive_ctx
    )
    return _grep_relative_files(
        pattern,
        workspace=Path(workspace_root).resolve(),
        mode=mode,
        max_results=max_results,
        sensitive_exclusions=sensitive_exclusions_under(root, sensitive_ctx),
    )


def _grep_relative_files(
    pattern: str,
    *,
    workspace: Path,
    mode: str,
    max_results: int,
    sensitive_exclusions: tuple[SensitiveExclusion, ...],
) -> str:
    """Grep pinned-root-relative files while refusing escaping symlink content."""
    import re

    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise LocalToolError(f"invalid regex: {exc}") from exc
    if mode not in ("content", "files", "count"):
        raise LocalToolError(f"unknown mode: {mode}")
    if max_results < 1:
        raise LocalToolError("max_results must be >= 1")
    # Memory-bounded: only the first max_results output lines are kept; the
    # rest are counted, not stored. Per-entry fs errors (races, permissions)
    # skip the entry rather than failing the whole search.
    shown: list[str] = []
    total = 0
    for p in workspace.rglob("*"):
        try:
            if not p.is_file() or p.stat().st_size > _MAX_GREP_FILE_BYTES:
                continue
            relative = _workspace_relative_path(p, workspace)
            if not _relative_target_is_safe(
                relative, workspace, sensitive_exclusions, is_directory=False
            ):
                continue  # protected path — skipped BEFORE it is read
        except OSError:
            continue  # racy/unreadable entry — skip it, not the whole search
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary/unreadable — skip
        rel = str(relative)
        hits = [f"{i}:{line}" for i, line in enumerate(text.splitlines(), 1) if rx.search(line)]
        if not hits:
            continue
        if mode == "content":
            for hit in hits:
                total += 1
                if len(shown) < max_results:
                    shown.append(f"{rel}:{hit}")
        else:
            total += 1
            if len(shown) < max_results:
                shown.append(rel if mode == "files" else f"{rel}:{len(hits)}")
    if total > max_results:
        shown.append(f"… ({total - max_results} more, truncated)")
    return "\n".join(shown) if shown else f"(no matches for {pattern!r})"


def _relative_target_is_safe(
    relative: Path,
    workspace: Path,
    exclusions: tuple[SensitiveExclusion, ...],
    *,
    is_directory: bool,
) -> bool:
    """Require both lexical and resolved targets to be admissible for I/O."""
    try:
        if _is_relative_sensitive_path(
            relative, exclusions, is_directory=is_directory
        ):
            return False
        resolved_workspace = workspace.resolve()
        resolved = (workspace / relative).resolve()
        if not resolved.is_relative_to(resolved_workspace):
            return False
        return not _is_relative_sensitive_path(
            resolved.relative_to(resolved_workspace), exclusions, is_directory=is_directory
        )
    except OSError:
        return False


def _workspace_relative_path(path: Path, workspace: Path) -> Path:
    """Return a lexical candidate name relative to a workspace I/O base."""
    return path.relative_to(workspace)


def _is_relative_sensitive_path(
    relative: Path,
    exclusions: tuple[SensitiveExclusion, ...],
    *,
    is_directory: bool,
) -> bool:
    """Apply parent-derived sensitive exclusions to one relative candidate."""
    parts = tuple(part.casefold() for part in relative.parts)
    for kind, value in exclusions:
        value_parts = tuple(part.casefold() for part in Path(value).parts)
        if kind == "subtree" and parts[: len(value_parts)] == value_parts:
            return True
        if kind == "file" and parts == value_parts:
            return True
        if kind == "direct_children" and (
            not is_directory
            and parts[:-1] == value_parts
        ):
            return True
        if kind == "name" and not is_directory and relative.name.casefold() == value:
            return True
    return False
