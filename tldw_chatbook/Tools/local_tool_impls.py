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
from typing import BinaryIO, Literal

from tldw_chatbook.Tools.worker_watchdog import register_temp, unregister_temp
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

#: TASK-32806.6: `read_file` output is capped at MAX_READ_CHARS, but it
#: still `read_text()`s the WHOLE file first -- a 150 MB single-line file
#: cost ~390 MB RSS and ~2.8 s to return 32 KB. Bound the source too, on
#: `st_size` before materialising it. Well above any real source file the
#: 32 KB window would show.
MAX_READ_FILE_BYTES = 10 * 1024 * 1024
MAX_GLOB_RESULTS = 100
MAX_GREP_RESULTS = 100
_MAX_GREP_FILE_BYTES = 2 * 1024 * 1024  # skip huge files
_MAX_WRITE_DIFF_CHARS = 12 * 1024
_WRITE_LOCKS_GUARD = threading.Lock()
_WRITE_LOCKS: dict[str, threading.Lock] = {}

# --- TASK-18927: self-recovery hints -------------------------------------
#
# Recovery messages are bounded, clearly machine-generated hints -- never
# instructions the model must obey. Every hint line is prefixed with
# ``HINT_LABEL`` so a transcript reader can always tell a tool-authored
# suggestion apart from a tool's factual result.

HINT_LABEL = "hint (tool-generated)"
#: Hard cap on any single hint block. Diagnoses are advisory scaffolding
#: around a failure, never a channel for replaying file content back.
_MAX_HINT_CHARS = 600
#: Cap on near-miss candidates offered in one hint.
_MAX_HINT_CANDIDATES = 3
#: difflib similarity cutoff for a line/name to qualify as a near miss.
_HINT_SIMILARITY_CUTOFF = 0.6
#: Cap on characters of one candidate line shown in a hint.
_MAX_HINT_LINE_CHARS = 100
#: Cap on workspace names collected during a glob walk for near-miss
#: probing (memory bound independent of workspace size).
_MAX_HINT_NAME_POOL = 2_000


def _visualize_whitespace(text: str) -> str:
    """Render a snippet with visible whitespace (TASK-18927 AC#2)."""
    return (
        text.replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace(" ", "·")
    )


def _bound_hint(text: str) -> str:
    """Clamp a hint block to ``_MAX_HINT_CHARS``."""
    if len(text) <= _MAX_HINT_CHARS:
        return text
    return text[: _MAX_HINT_CHARS - 1] + "…"


def _closest_lines(
    needle: str, haystack: list[str], limit: int = _MAX_HINT_CANDIDATES
) -> list[tuple[int, str]]:
    """Return up to ``limit`` ``(index, line)`` near misses for ``needle``.

    Exact bodies first (the text exists, just not where expected), then
    difflib-similar bodies above ``_HINT_SIMILARITY_CUTOFF``.
    """
    import difflib

    stripped = needle.strip()
    if not stripped:
        return []
    found: list[tuple[int, str]] = []
    for index, line in enumerate(haystack):
        if line.strip() == stripped:
            found.append((index, line))
            if len(found) >= limit:
                return found
    if found:
        return found
    # Needle pinned as seq2 so its b-side match cache survives the loop;
    # each candidate line swaps in as seq1 only.
    matcher = difflib.SequenceMatcher(b=stripped)
    scored: list[tuple[float, int, str]] = []
    for index, line in enumerate(haystack):
        body = line.strip()
        if not body:
            continue
        matcher.set_seq1(body)
        ratio = matcher.ratio()
        if ratio >= _HINT_SIMILARITY_CUTOFF:
            scored.append((ratio, index, line))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(index, line) for _, index, line in scored[:limit]]


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
    # review-B B27: one workspace resolution per listing (the target check
    # and every entry check share it; None falls back to per-entry
    # resolution when even this fails).
    try:
        resolved_workspace: Path | None = workspace.resolve()
    except (OSError, RuntimeError, ValueError):
        resolved_workspace = None
    # Loop-invariant check input: denylist entries case-folded once for the
    # whole scan instead of once per candidate.
    folded = _folded_exclusions(sensitive_exclusions)
    if (
        not _relative_target_is_safe(
            relative,
            workspace,
            sensitive_exclusions,
            is_directory=True,
            resolved_workspace=resolved_workspace,
            folded_exclusions=folded,
        )
        or not target.is_dir()
    ):
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
            entry_relative,
            workspace,
            sensitive_exclusions,
            is_directory=entry.is_dir(),
            resolved_workspace=resolved_workspace,
            folded_exclusions=folded,
        ):
            continue
        scanned.append(entry)
    entries = sorted(scanned, key=lambda p: (p.is_file(), p.name.lower()))
    lines = [f"{p.name}/" if p.is_dir() else p.name for p in entries[:max_entries]]
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
    content_stamps: bool = False,
) -> str:
    """Read a pinned-root-relative text file without reopening its resolved path.

    The file is opened exactly ONCE and fully read as bytes: the binary
    sniff, the decoded text, and (when ``content_stamps`` is set) the
    CAS stamp digest all come from that single read, so the rendered
    content and its sha256/size stamps can never be a torn pair under
    concurrent modification. (``str.splitlines`` splits untranslated
    ``\\r``/``\\r\\n`` exactly where the old universal-newline
    ``read_text`` translation produced splits, so the output is
    byte-identical to the previous two-open implementation.)

    Args:
        relative: Root-relative target, already safety-checked.
        workspace: The confinement root ``relative`` resolves against.
        offset: 1-based first line to render.
        limit: Optional cap on the rendered line count.
        sensitive_exclusions: Deny rules applied to the target.
        display_path: Caller-facing path for error messages.
        content_stamps: Append the worker-reported CAS tail
            (``\\nsha256: <hex>\\nsize: <n>``) computed from the same
            read — the pinned dispatch path sets this; the plain local
            ``read_file`` surface does not.

    Returns:
        The numbered (or notice) body, plus the stamp tail when
        requested.

    Raises:
        LocalToolError: If the target is missing/protected, oversized,
            or binary.
    """
    target = workspace / relative
    if (
        not _relative_target_is_safe(
            relative, workspace, sensitive_exclusions, is_directory=False
        )
        or not target.is_file()
    ):
        raise LocalToolError(f"file not found: {display_path or relative}")
    file_size = target.stat().st_size
    if file_size > MAX_READ_FILE_BYTES:
        raise LocalToolError(
            f"'{display_path or relative}' is too large to read "
            f"({file_size} bytes; maximum {MAX_READ_FILE_BYTES})"
        )
    data = target.read_bytes()
    if b"\x00" in data[:8192]:
        raise LocalToolError(
            f"'{display_path or relative}' appears to be binary; fs_read only reads text files"
        )
    lines = data.decode("utf-8", errors="replace").splitlines()
    if not lines:
        body = "(empty file)"
    else:
        start = max(offset, 1) - 1
        if start >= len(lines):
            body = f"(offset {offset} is past end of file; {len(lines)} lines total)"
        else:
            window = (
                lines[start:] if limit is None else lines[start : start + max(limit, 0)]
            )
            body = "\n".join(
                f"{i}\t{line}" for i, line in enumerate(window, start=start + 1)
            )
            if len(body) > MAX_READ_CHARS:
                body = body[:MAX_READ_CHARS] + "\n… [truncated]"
    if content_stamps:
        body += f"\nsha256: {hashlib.sha256(data).hexdigest()}\nsize: {len(data)}"
    return body


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


def create_directory(path: str, *, workspace_root: Path) -> str:
    """Create directory ``path`` (and missing parents) inside the root.

    TASK-35081: the fs_* family had no directory-creation capability --
    every writer requires an existing parent (spec 2026-08-04 §2's
    deliberate typo-catching divergence), so a model asked to "create a
    folder" had no honest path to success and confabulated instead.
    ``fs_mkdir`` keeps every write-intent guard: confinement, the
    sensitive-path denylist (the target itself included), the
    ``.git``-metadata write refusal, and the
    ``refuses_new_directory_chain`` shadow-directory guard applied to the
    target and each not-yet-existing ancestor — the exact chain
    ``mkdir(parents=True)`` is about to create.

    Creating a directory that already exists is a success no-op (a retried
    mkdir must not burn an approval cycle); a non-directory at the target
    is refused.

    Args:
        path: Directory to create, absolute or relative to
            ``workspace_root``.
        workspace_root: The confinement root ``path`` must resolve within.

    Returns:
        ``created directory <path>`` or ``directory already exists:
        <path>``.

    Raises:
        LocalToolError: If ``path`` resolves outside ``workspace_root``,
            is a protected path, would collide with one, or creation
            fails.
    """
    sensitive_ctx = resolve_sensitive_context()
    workspace = Path(workspace_root).resolve()
    root = resolve_workspace_path(
        path, workspace, intent="write", context=sensitive_ctx
    )
    if refuses_new_directory_chain(root, context=sensitive_ctx):
        raise LocalToolError(
            f"Refused: creating '{path}' would collide with a protected path"
        )
    return _mkdir_relative_directory(
        root.relative_to(workspace),
        workspace=workspace,
        sensitive_exclusions=sensitive_exclusions_under(workspace, sensitive_ctx),
        display_path=path,
    )


def _mkdir_relative_directory(
    relative: Path,
    *,
    workspace: Path,
    sensitive_exclusions: tuple[SensitiveExclusion, ...],
    display_path: str | None = None,
) -> str:
    """Create one already-admitted directory (with parents) relative to an I/O root.

    The parent-side admission (executor ``_build_request``/remote builder)
    has already run the absolute denylist and the new-directory-chain
    guard; this core re-checks the request-carried exclusions with a
    DIRECTORY classification so a binding-excluded or remote-home-denied
    subtree cannot be planted, and refuses a symlinked chain whose
    resolution escapes the root (``_relative_target_is_safe`` resolves
    existing ancestors before anything is created).

    Args:
        relative: Root-relative target, already confined.
        workspace: The confinement root ``relative`` resolves against.
        sensitive_exclusions: Deny rules applied to the target chain.
        display_path: Caller-facing path for messages.

    Returns:
        The creation (or already-exists) summary.

    Raises:
        LocalToolError: If the target is protected, resolves outside the
            root, a non-directory exists at the target, or creation fails.
    """
    shown = display_path or str(relative)
    target = workspace / relative
    if not _relative_target_is_safe(
        relative, workspace, sensitive_exclusions, is_directory=True
    ):
        raise LocalToolError(
            f"Refused: '{shown}' is a protected path and cannot be created"
        )
    if target.exists() and not target.is_dir():
        raise LocalToolError(f"a non-directory already exists at: {shown}")
    if target.is_dir():
        return f"directory already exists: {shown}"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LocalToolError(
            f"could not create directory {shown}: {exc.strerror or exc}"
        ) from exc
    return f"created directory {shown}"


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
    content_stamps: bool = False,
) -> str:
    """Preview or atomically write one admitted path with optional CAS.

    Args:
        relative: Root-relative target, already safety-checked.
        content: Full replacement content.
        workspace: The confinement root ``relative`` resolves against.
        display_path: Caller-facing path for messages.
        dry_run: Preview only; nothing is written and no stamp tail is
            appended (the preview is a JSON object).
        expected_sha256: Optional CAS precondition on the current bytes.
        expected_absent: Optional CAS precondition that no file exists.
        content_stamps: Append the worker-reported CAS tail
            (``\\nsha256: <hex>\\nsize: <n>``) of the bytes just written
            (Task 16 write-path parity with ``_read_relative_file``).
            Computed from ``data`` -- the exact in-memory bytes handed to
            the atomic writer -- so content and stamps can never be a
            torn pair; the plain local ``write_file`` surface does not
            set this and stays byte-identical.
    """
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
        # TASK-18927 AC#3: verify on-disk bytes against the intended
        # content and report any mismatch honestly. Still under the write
        # lock, so a concurrent writer through this process cannot slip in
        # between write and verification.
        verified = _read_write_target(target, shown)
        if verified != data:
            raise LocalToolError(
                f"write verification failed for {shown}: on-disk bytes "
                f"differ from the intended content (read back "
                f"{len(verified) if verified is not None else 0} of "
                f"{len(data)} bytes) — the file may be torn or modified "
                "concurrently; re-read it before retrying"
            )
    summary = f"wrote {len(content)} characters to {shown}"
    if content_stamps and not dry_run:
        summary += f"\nsha256: {hashlib.sha256(data).hexdigest()}\nsize: {len(data)}"
    return summary


def _write_lock_for(key: str) -> threading.Lock:
    with _WRITE_LOCKS_GUARD:
        return _WRITE_LOCKS.setdefault(key, threading.Lock())


def _read_write_target(target: Path, shown: str) -> bytes | None:
    try:
        before = os.lstat(target)
    except FileNotFoundError:
        return None
    if not stat_module.S_ISREG(before.st_mode) or stat_module.S_ISLNK(before.st_mode):
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
    """Write through one pinned parent descriptor and same-directory temp.

    task-32808.5 AC#3: deliberately NOT routed through
    ``Utils/atomic_file_ops.atomic_write_text`` -- this path needs a pinned
    parent ``dir_fd`` (TOCTOU-safe rename), ``O_NOFOLLOW``, an sha256/absent
    precondition and a target lock, none of which the shared helper offers.
    """
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
    # Watchdog data plane (Task 12): while the temp exists (creation
    # until rename/link or cleanup), its ABSOLUTE path sits in the
    # registry the worker's tier-1 timer sweeps — a mid-write watchdog
    # death must not leave the partial temp behind. Absolute because
    # the sweep unlinks by path from another thread while this one
    # holds only the parent descriptor.
    temp_path = os.path.abspath(os.path.join(str(target.parent), temp_name))
    temp_created = False
    target_lock = None
    try:
        if expected_sha256 is not None:
            target_lock = _acquire_expected_target_lock(
                parent_fd,
                target.name,
                shown,
                expected_sha256,
            )
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
        register_temp(temp_path)
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
            unregister_temp(temp_path)
        else:
            if target_lock is not None:
                _assert_expected_target_is_current(
                    parent_fd,
                    target.name,
                    shown,
                    expected_sha256,
                    target_lock,
                )
            os.replace(
                temp_name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            temp_created = False
            unregister_temp(temp_path)
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
            unregister_temp(temp_path)
        if target_lock is not None:
            _unlock_target_handle(target_lock)
            target_lock.close()
        os.close(parent_fd)


def _portalocker_module():
    """Return the ``portalocker`` module, or ``None`` where absent.

    The remote worker bundle (Phase 1d, Task 8) concatenates this module
    onto a bare remote interpreter that has no third-party packages, so
    the CAS write lock needs a stdlib fallback there. In the parent
    application ``portalocker`` is a pinned dependency and the fallback
    never runs.
    """
    try:
        import portalocker
    except ImportError:
        return None
    return portalocker


def _lock_expected_target(handle: BinaryIO) -> None:
    """Take one non-blocking exclusive advisory lock on an open target.

    Uses portalocker where installed; otherwise falls back to the same
    POSIX advisory lock portalocker itself delegates to (``fcntl.flock``,
    exclusive + non-blocking). Contention or a lock failure raises
    ``LocalToolError`` with the fixed precondition message callers map to.
    """
    portalocker = _portalocker_module()
    if portalocker is not None:
        try:
            portalocker.lock(
                handle,
                portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING,
            )
        except portalocker.exceptions.LockException:
            handle.close()
            raise LocalToolError(
                "write precondition failed: target is being modified"
            ) from None
        return
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise LocalToolError(
            "write precondition failed: target is being modified"
        ) from None


def _unlock_target_handle(handle: BinaryIO) -> None:
    """Best-effort release of the advisory lock taken by the helper above."""
    portalocker = _portalocker_module()
    try:
        if portalocker is not None:
            portalocker.unlock(handle)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:  # noqa: BLE001 - closing the handle releases the lock anyway
        pass


def _acquire_expected_target_lock(
    parent_fd: int,
    filename: str,
    shown: str,
    expected_sha256: str,
) -> BinaryIO:
    """Acquire a non-blocking process lock on the expected target inode."""
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(filename, flags, dir_fd=parent_fd)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
    except OSError:
        raise LocalToolError(
            "write precondition failed: target digest changed"
        ) from None
    _lock_expected_target(handle)
    try:
        _assert_expected_target_is_current(
            parent_fd,
            filename,
            shown,
            expected_sha256,
            handle,
        )
    except Exception:
        _unlock_target_handle(handle)
        handle.close()
        raise
    return handle


def _assert_expected_target_is_current(
    parent_fd: int,
    filename: str,
    shown: str,
    expected_sha256: str | None,
    locked_handle: BinaryIO,
) -> None:
    """Verify the locked inode is still the named path with the expected bytes."""
    if expected_sha256 is None:
        raise LocalToolError("write precondition failed: target digest changed")
    locked_info = os.fstat(locked_handle.fileno())
    if not stat_module.S_ISREG(locked_info.st_mode):
        raise LocalToolError(f"write target is not a regular file: {shown}")
    locked_handle.seek(0)
    locked_digest = hashlib.sha256(locked_handle.read()).hexdigest()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        current_fd = os.open(filename, flags, dir_fd=parent_fd)
    except OSError:
        raise LocalToolError(
            "write precondition failed: target digest changed"
        ) from None
    try:
        current_info = os.fstat(current_fd)
    finally:
        os.close(current_fd)
    if (locked_info.st_dev, locked_info.st_ino) != (
        current_info.st_dev,
        current_info.st_ino,
    ) or locked_digest != expected_sha256:
        raise LocalToolError("write precondition failed: target digest changed")


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
    content_stamps: bool = False,
) -> str:
    """Edit one already-admitted path relative to an I/O root.

    Args:
        relative: Root-relative target, already safety-checked.
        old_string: Exact text to replace.
        new_string: Replacement text.
        workspace: The confinement root ``relative`` resolves against.
        replace_all: Replace every occurrence instead of requiring a
            unique match.
        display_path: Caller-facing path for messages.
        content_stamps: Append the worker-reported CAS tail of the bytes
            just written (``\\nsha256: <hex>\\nsize: <n>``), computed
            from the same in-memory ``data`` handed to the atomic writer
            (Task 16 write-path parity with ``_read_relative_file``);
            the plain local ``edit_file`` surface does not set this and
            stays byte-identical.
    """
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
        # TASK-18927 AC#1: an edit whose change is already present is a
        # success no-op, not a match error — the model retrying an already
        # applied edit (a common re-run after a lost race or a re-sent
        # turn) no longer burns an approval cycle on it. Evidence must be
        # strong, though: a new_string that appears exactly once (the
        # signature of a completed single replace), or any presence under
        # replace_all (whose retry leaves N occurrences). A short or
        # coincidentally-present new_string stays an error, with the
        # ambiguity said out loud.
        if new_string:
            occurrences = content.count(new_string)
            # Review fix (PR #3086): presence alone is NOT proof the edit
            # landed — a coincidental short new_string ("x" inside "item_x")
            # produced false successes. Require distinctive evidence: a
            # stripped length of at least 8, or any presence under
            # replace_all (whose completed retry leaves N occurrences of a
            # distinctive-enough string). Weaker evidence falls through to
            # the ambiguity note below.
            distinctive = len(new_string.strip()) >= 8
            if distinctive and (
                occurrences == 1 or (replace_all and occurrences)
            ):
                return (
                    f"edit already applied to {shown}: the replacement text "
                    f"is present ({occurrences} occurrence"
                    f"{'s' if occurrences != 1 else ''}) and the old text is "
                    "absent (0 replacements, no change made)"
                )
        # TASK-18927 AC#2: diagnose the near miss instead of a bare error.
        parts: list[str] = []
        if new_string:
            occurrences = content.count(new_string)
            if occurrences:
                parts.append(
                    f"new_string is already present {occurrences} time(s) — "
                    "ambiguous evidence the edit was already applied; "
                    "refine old_string or verify the current file state"
                )
        hint = _edit_no_match_hint(content, old_string)
        if hint:
            parts.append(hint)
        message = f"old_string not found in {shown}"
        if parts:
            message += f"\n{HINT_LABEL}: " + _bound_hint(" | ".join(parts))
        raise LocalToolError(message)
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
    # TASK-32806.6: route the write through the same O_EXCL-temp + fsync +
    # os.replace path fs_write uses, instead of a plain in-place
    # `write_bytes` (truncate-then-write, no fsync, follows a symlink at the
    # target). The file exists here -- we just read it -- so it is a replace,
    # not a create.
    _atomic_write_target(
        target,
        data,
        shown=shown,
        expected_sha256=None,
        expected_absent=False,
    )
    n = count if replace_all else 1
    summary = f"made {n} replacement{'s' if n != 1 else ''} in {shown}"
    if content_stamps:
        summary += f"\nsha256: {hashlib.sha256(data).hexdigest()}\nsize: {len(data)}"
    return summary


def _edit_no_match_hint(content: str, old_string: str) -> str | None:
    """Diagnose an unmatched ``old_string`` (TASK-18927 AC#2), or ``None``.

    Two probes, cheapest first:

    1. Whitespace-insensitive containment — if collapsing all whitespace
       runs makes ``old_string`` appear in the file, the text IS there and
       only indentation, spacing, or line endings differ; say so.
    2. Nearest line — the closest body lines to ``old_string``'s first
       line (exact bodies first, then difflib similarity), shown with
       visible whitespace so the difference is readable at a glance.

    Bounded to ``_MAX_HINT_CHARS``; adversarial inputs (huge files, odd
    content) can only shrink the hint, never grow it, and any internal
    failure yields ``None`` (the plain error survives).
    """
    try:
        collapsed_content = " ".join(content.split())
        collapsed_old = " ".join(old_string.split())
        if collapsed_old and collapsed_old in collapsed_content:
            return _bound_hint(
                "the same text IS present with different whitespace "
                "(indentation, spacing, or line endings) — re-read the "
                "exact region and copy its literal whitespace"
            )
        content_lines = content.splitlines()
        old_lines = [line for line in old_string.splitlines() if line.strip()]
        if not old_lines:
            return None
        near = _closest_lines(old_lines[0], content_lines)
        if not near:
            return None
        rendered = "; ".join(
            f"line {index + 1}: {_visualize_whitespace(line.strip())[:_MAX_HINT_LINE_CHARS]}"
            for index, line in near
        )
        return _bound_hint(
            "closest candidate lines (whitespace shown as · for space, "
            f"\\t for tab): {rendered}"
        )
    except Exception:
        return None


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
    # TASK-18927 AC#4: bounded pool of admissible names collected during
    # the SAME filtered walk, for near-miss probing on a zero-match
    # result. Only names this walk was already willing to disclose can
    # enter a hint — the probe can never surface a sensitive path.
    name_pool: list[str] = []
    # review-B B27: one workspace resolution per walk (None falls back to the
    # old per-entry resolution when even this fails).
    try:
        resolved_workspace: Path | None = workspace.resolve()
    except (OSError, RuntimeError, ValueError):
        resolved_workspace = None
    # Loop-invariant check input: denylist entries case-folded once for the
    # whole glob instead of once per candidate.
    folded = _folded_exclusions(sensitive_exclusions)
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
                    rendered,
                    workspace,
                    sensitive_exclusions,
                    is_directory=False,
                    resolved_workspace=resolved_workspace,
                    folded_exclusions=folded,
                ):
                    continue
            elif _is_relative_sensitive_path(
                rendered,
                sensitive_exclusions,
                is_directory=False,
                folded_exclusions=folded,
            ):
                continue
            mtime = p.stat().st_mtime
        except OSError:
            continue  # racy/unreadable entry — skip it, not the whole search
        total += 1
        if len(name_pool) < _MAX_HINT_NAME_POOL:
            name_pool.append(rendered.as_posix())
        if len(heap) < max_results:
            heapq.heappush(heap, (mtime, rendered))
        elif mtime > heap[0][0]:
            heapq.heapreplace(heap, (mtime, rendered))
    best = sorted(heap, key=lambda t: t[0], reverse=True)
    lines = [str(relative) for _, relative in best]
    if total > max_results:
        lines.append(f"… ({total - max_results} more, truncated)")
    if lines:
        return "\n".join(lines)
    # Zero matches: ``Path.glob`` yields only matches, so the pool collected
    # during the walk is empty here — take ONE bounded supplementary walk
    # (entry-capped, same admissibility checks) for near-miss probing.
    if not name_pool:
        name_pool = _bounded_admissible_name_pool(
            workspace, sensitive_exclusions, resolved_workspace, validate_targets
        )
    empty = f"(no files matching {pattern!r})"
    hint = _glob_near_miss_hint(pattern, name_pool)
    if hint:
        return f"{empty}\n{HINT_LABEL}: {hint}"
    return empty


def _bounded_admissible_name_pool(
    workspace: Path,
    sensitive_exclusions: tuple[SensitiveExclusion, ...],
    resolved_workspace: Path | None,
    validate_targets: bool,
) -> list[str]:
    """Collect a bounded pool of admissible names for near-miss hints.

    Considers at most ``_MAX_HINT_NAME_POOL`` entries and stops; the probe
    is best-effort, never a second full walk. Applies the exact
    admissibility checks the glob walk uses, so a name a match would have
    been refused on (sensitive, escaping) can never enter a hint.
    """
    pool: list[str] = []
    examined = 0
    try:
        for p in workspace.rglob("*"):
            examined += 1
            if examined > MAX_SCAN_ENTRIES or len(pool) >= _MAX_HINT_NAME_POOL:
                break
            try:
                if not p.is_file():
                    continue
                rendered = p.relative_to(workspace)
                if validate_targets:
                    if not _relative_target_is_safe(
                        rendered,
                        workspace,
                        sensitive_exclusions,
                        is_directory=False,
                        resolved_workspace=resolved_workspace,
                    ):
                        continue
                elif _is_relative_sensitive_path(
                    rendered, sensitive_exclusions, is_directory=False
                ):
                    continue
                pool.append(rendered.as_posix())
            except (OSError, ValueError):
                continue
    except (OSError, RuntimeError):
        return pool
    return pool


def _glob_near_miss_hint(pattern: str, name_pool: list[str]) -> str | None:
    """Suggest close existing names for a zero-match pattern (TASK-18927).

    Extracts the longest literal (wildcard-free) segment of the pattern and
    difflib-matches it against the pool's segments and full paths. Purely
    advisory and bounded; ``None`` when there is nothing close or nothing
    literal to probe (e.g. ``**/*``).
    """
    import difflib

    try:
        literals: list[str] = []
        for segment in pattern.split("/"):
            if not segment or any(ch in segment for ch in "*?["):
                continue
            literals.append(segment)
        if not literals:
            return None
        needle = max(literals, key=len)
        if len(needle) < 3:
            return None
        segment_example: dict[str, str] = {}
        for name in name_pool:
            for segment in name.split("/"):
                if segment not in segment_example:
                    segment_example[segment] = name
        close_segments = difflib.get_close_matches(
            needle,
            list(segment_example),
            n=_MAX_HINT_CANDIDATES,
            cutoff=_HINT_SIMILARITY_CUTOFF,
        )
        if close_segments:
            rendered = "; ".join(
                f'"{segment}" (e.g. {segment_example[segment]})'
                for segment in close_segments
            )
            return _bound_hint(f"closest existing names: {rendered}")
        close_paths = difflib.get_close_matches(
            needle, name_pool, n=_MAX_HINT_CANDIDATES, cutoff=_HINT_SIMILARITY_CUTOFF
        )
        if close_paths:
            return _bound_hint(
                "closest existing paths: " + "; ".join(f'"{p}"' for p in close_paths)
            )
        return None
    except Exception:
        return None


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
        compiled = re.compile(pattern)
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
        compiled,  # validated once here; the relative core must not recompile
        workspace=Path(workspace_root).resolve(),
        mode=mode,
        max_results=max_results,
        sensitive_exclusions=sensitive_exclusions_under(root, sensitive_ctx),
    )


def _grep_relative_files(
    pattern: str | re.Pattern[str],
    *,
    workspace: Path,
    mode: str,
    max_results: int,
    sensitive_exclusions: tuple[SensitiveExclusion, ...],
) -> str:
    """Grep pinned-root-relative files while refusing escaping symlink content."""
    import re

    if isinstance(pattern, re.Pattern):
        rx = pattern
        display_pattern = pattern.pattern
    else:
        display_pattern = pattern
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
    # TASK-18927 AC#4: single-pass near-miss probe. Any line an exact
    # case-sensitive search matches is also a case-insensitive match (for
    # ordinary patterns), so counting the ci-only lines during the same
    # walk costs one extra `search` per line and never needs a second
    # walk. Only used when the exact search found nothing.
    ci_near_total = 0
    ci_near_example: str | None = None
    try:
        rx_ci = re.compile(display_pattern, re.IGNORECASE)
    except re.error:
        rx_ci = rx  # unreachable (validated above); keep the type narrow
    # review-B B27: resolve the workspace once for the whole walk instead of
    # once per entry. If even the workspace resolution fails (pathological
    # ancestry), None falls back to the old per-entry behavior exactly.
    try:
        resolved_workspace: Path | None = workspace.resolve()
    except (OSError, RuntimeError, ValueError):
        resolved_workspace = None
    # Loop-invariant check input: denylist entries case-folded once for the
    # whole search instead of once per candidate.
    folded = _folded_exclusions(sensitive_exclusions)
    for p in workspace.rglob("*"):
        try:
            if not p.is_file() or p.stat().st_size > _MAX_GREP_FILE_BYTES:
                continue
            relative = _workspace_relative_path(p, workspace)
            if not _relative_target_is_safe(
                relative,
                workspace,
                sensitive_exclusions,
                is_directory=False,
                resolved_workspace=resolved_workspace,
                folded_exclusions=folded,
                # rglob never descends INTO symlinked directories, so a
                # candidate that is not itself a symlink has only real
                # directories between the (already resolved) workspace root
                # and its lexical path — its realpath equals that lexical
                # path and the two checks below already decided. Symlinks
                # still get the full resolved-target check, so escaping or
                # sensitive link targets are refused exactly as before.
                resolve_regular_targets=False,
            ):
                continue  # protected path — skipped BEFORE it is read
        except OSError:
            continue  # racy/unreadable entry — skip it, not the whole search
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary/unreadable — skip
        rel = str(relative)
        hits = []
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{i}:{line}")
            elif rx_ci.search(line):
                ci_near_total += 1
                if ci_near_example is None:
                    ci_near_example = f"{rel}:{i}"
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
    if shown:
        return "\n".join(shown)
    empty = f"(no matches for {display_pattern!r})"
    if ci_near_total:
        # Review fix (PR #3086 Minor-2): route through _bound_hint so the
        # path-bearing example respects the same cap as every other hint
        # (AC#5's blanket bound claim).
        return f"{empty}\n{HINT_LABEL}: " + _bound_hint(
            f"{ci_near_total} line(s) match case-insensitively "
            f"(e.g. {ci_near_example}) \u2014 prefix the pattern with "
            "(?i) if that is intended"
        )
    return empty



#: One denylist entry with its path parts case-folded once, ready for the
#: per-candidate comparisons in :func:`_is_relative_sensitive_path`:
#: ``(kind, folded value parts, raw value)``. The raw value is kept because
#: the ``"name"`` rule compares against it verbatim.
_FoldedExclusion = tuple[str, tuple[str, ...], str]


def _folded_exclusions(
    exclusions: tuple[SensitiveExclusion, ...],
) -> tuple[_FoldedExclusion, ...]:
    """Case-fold every exclusion's path parts ONCE per call.

    ``_is_relative_sensitive_path`` used to re-fold ``Path(value).parts`` for
    every exclusion entry for every candidate — O(candidates x exclusions)
    tuple builds that are loop-invariant for one tool invocation.
    """
    return tuple(
        (kind, tuple(part.casefold() for part in Path(value).parts), value)
        for kind, value in exclusions
    )


def _relative_target_is_safe(
    relative: Path,
    workspace: Path,
    exclusions: tuple[SensitiveExclusion, ...],
    *,
    is_directory: bool,
    resolved_workspace: Path | None = None,
    folded_exclusions: tuple[_FoldedExclusion, ...] | None = None,
    resolve_regular_targets: bool = True,
) -> bool:
    """Require both lexical and resolved targets to be admissible for I/O.

    ``resolved_workspace`` accepts a precomputed ``workspace.resolve()``
    (review-B B27) so per-entry callers -- grep/glob/list walks -- resolve
    the workspace once per operation instead of once per entry. Omitted, the
    workspace is resolved here as before; the decision is identical either
    way.

    ``folded_exclusions`` is :func:`_folded_exclusions`' once-per-call
    output; ``None`` folds here, which decides identically.

    ``resolve_regular_targets=False`` (grep only) skips the realpath walk for
    candidates that pass the lexical check and are not themselves symlinks:
    grep enumerates with ``rglob``, which never descends INTO symlinked
    directories, so a non-symlink candidate's ancestors below the (resolved)
    workspace are all real directories and its resolved form equals its
    lexical form — the same two checks then decide identically. Candidates
    that ARE symlinks still get the full resolved-target check, so the
    escaping-symlink and symlinked-sensitive-path refusals are unchanged;
    see ``test_fs_grep_skips_symlinks_escaping_workspace``.
    """
    try:
        if _is_relative_sensitive_path(
            relative,
            exclusions,
            is_directory=is_directory,
            folded_exclusions=folded_exclusions,
        ):
            return False
        target = workspace / relative
        if resolved_workspace is None:
            resolved_workspace = workspace.resolve()
        if resolve_regular_targets or target.is_symlink():
            resolved = target.resolve()
            if not resolved.is_relative_to(resolved_workspace):
                return False
            return not _is_relative_sensitive_path(
                resolved.relative_to(resolved_workspace),
                exclusions,
                is_directory=is_directory,
                folded_exclusions=folded_exclusions,
            )
        return True
    except (OSError, RuntimeError, ValueError):
        # Python <= 3.12 raises RuntimeError for a symlink loop (3.13+:
        # OSError); ValueError covers an embedded NUL. Any unresolvable
        # entry is inadmissible -- skip it rather than fail the listing.
        return False


def _workspace_relative_path(path: Path, workspace: Path) -> Path:
    """Return a lexical candidate name relative to a workspace I/O base."""
    return path.relative_to(workspace)


def _is_relative_sensitive_path(
    relative: Path,
    exclusions: tuple[SensitiveExclusion, ...],
    *,
    is_directory: bool,
    folded_exclusions: tuple[_FoldedExclusion, ...] | None = None,
) -> bool:
    """Apply parent-derived sensitive exclusions to one relative candidate.

    ``folded_exclusions`` is :func:`_folded_exclusions`' once-per-call output;
    ``None`` folds here, which decides identically.
    """
    if folded_exclusions is None:
        folded_exclusions = _folded_exclusions(exclusions)
    parts = tuple(part.casefold() for part in relative.parts)
    for kind, value_parts, value in folded_exclusions:
        if kind == "subtree" and parts[: len(value_parts)] == value_parts:
            return True
        if kind == "file" and parts == value_parts:
            return True
        if kind == "direct_children" and (
            not is_directory and parts[:-1] == value_parts
        ):
            return True
        if kind == "name" and not is_directory and relative.name.casefold() == value:
            return True
    return False
