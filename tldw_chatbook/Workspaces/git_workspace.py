"""Change-review git modes: user-repo runner + workspace detection.

TASK-16801 arc B, spec
`Docs/superpowers/specs/2026-08-20-console-review-git-modes-design.md`.

This module is the THIRD git-invocation posture in the codebase. Three
runners exist for three different jobs, and each one's environment
handling would be WRONG for the other two:

| Runner | Posture | Why |
|---|---|---|
| Shadow tracker (`Workspaces/change_tracking.py`) | scrubs ALL `GIT_*` env, explicit `--git-dir`/`--work-tree` | a per-turn snapshot repo lives entirely under the app data dir and must NEVER touch the user's actual git state (repo, index, HEAD, stashes) -- ambient `GIT_DIR`/`GIT_WORK_TREE` pointing anywhere near the user's repo would be a correctness disaster there. |
| Read-only agent tools (`Tools/git_tool_impls.py`) | minimal env, `HOME` stripped | model-driven tool calls read repo state on the model's behalf; no user identity (`~/.gitconfig`, credential helpers, SSH agent) should be reachable from a model-issued `git status`/`git log`/`git diff`. |
| **Git modes (this module)** | ambient env preserved, only repo-TARGETING vars scrubbed | this module acts AS THE USER in their own repository -- committing with their identity, pushing with their credentials. `HOME`, `SSH_AUTH_SOCK`, credential helpers, `GIT_SSH_COMMAND`, `GIT_ASKPASS`/`SSH_ASKPASS` are all preserved deliberately (an https push popping the user's own GUI askpass is their configuration working as intended). Only vars that could REDIRECT a command into the wrong repository (`GIT_DIR`, `GIT_WORK_TREE`, etc.) are scrubbed -- the app itself sets none of these, but the app may be launched from a shell/hook that does, and a stray targeting var must not silently commit into the wrong repo. |

Detection here is read-only plumbing only: `rev-parse --show-toplevel`,
branch/detached/unborn probes, remotes, upstream, and ahead/behind counts.
It never raises to callers -- any :class:`GitWorkspaceError` from a probe
degrades detection to ``None`` (spec §3: "detection must never raise to
the UI"). Mutating operations (commit/push/PR-url) are later tasks in this
arc; this module lays only the runner and the detection groundwork.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from tldw_chatbook.Workspaces.change_tracking import ChangedFile

#: Read-only probes (status, branch, remote listings).
READ_TIMEOUT_SECONDS = 30.0
#: Commit -- user hooks (pre-commit, commit-msg) and gpg signing may run.
COMMIT_TIMEOUT_SECONDS = 120.0
#: Push -- network round-trip to a remote.
PUSH_TIMEOUT_SECONDS = 300.0


class GitWorkspaceError(Exception):
    """A git-modes operation failed. Detection degrades this to ``None``
    (or a typed refusal) rather than letting it reach the UI as a crash;
    mutating operations (a later task) surface it as an honest per-step
    failure instead."""


@dataclass(frozen=True)
class GitCmdResult:
    """The raw result of one ``git`` invocation via :func:`_run_user_git`.

    Attributes:
        returncode: The process exit code.
        stdout: Captured stdout, text-decoded.
        stderr: Captured stderr, text-decoded.
    """

    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class GitWorkspaceInfo:
    """A detected real git repository at a workspace root.

    Attributes:
        root: The workspace root that was probed.
        repo_root: The repository's toplevel, per ``rev-parse
            --show-toplevel`` (equal to ``root`` -- detection refuses
            otherwise; see :class:`GitWorkspaceRefusal`).
        branch: The current branch's short name, or ``None`` when
            ``detached`` is True.
        detached: True when HEAD is not on a branch.
        unborn: True when the current branch exists but has no commits
            yet (a fresh ``git init`` with no history).
        upstream: The configured upstream ref (e.g. ``"origin/feat/x"``),
            or ``None`` when unset.
        upstream_remote: The upstream's remote NAME (e.g. ``"origin"``),
            read from ``%(upstream:remotename)`` -- never derived by
            splitting ``upstream`` on ``"/"``, because remote names can
            themselves contain ``"/"`` (spec §2 probe 6). ``None`` when
            there is no upstream.
        remotes: Ordered, unique ``(name, push_url)`` pairs from
            ``remote -v``.
        ahead: Commits on HEAD not yet on the upstream (0 with no
            upstream).
        behind: Commits on the upstream not yet on HEAD (0 with no
            upstream).
    """

    root: Path
    repo_root: Path
    branch: str | None
    detached: bool
    unborn: bool
    upstream: str | None
    upstream_remote: str | None
    remotes: tuple[tuple[str, str], ...]
    ahead: int
    behind: int


@dataclass(frozen=True)
class GitWorkspaceRefusal:
    """Detection found a repository but declines to offer git modes on it.

    Attributes:
        reason: Human-readable copy suitable for the mode's "why
            unavailable" surface.
    """

    reason: str


#: Env vars that could redirect a git invocation at a DIFFERENT
#: repository than ``root`` -- these must never leak in from the
#: ambient environment the app happened to be launched under.
#: Case-insensitive: Windows env vars are case-insensitive, so
#: `Git_Dir` reaches git exactly as `GIT_DIR` does; harmless over-scrub
#: on POSIX.
_SCRUBBED_VARS = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_COMMON_DIR",
        "GIT_CEILING_DIRECTORIES",
    }
)


def _user_git_env() -> dict[str, str]:
    """Ambient environment, minus repo-targeting vars, plus safety pins.

    See the module docstring's three-runner table for why this posture
    (preserve identity/credentials, scrub only targeting vars) is correct
    HERE and would be wrong for the other two git runners in this repo.
    """
    env = {k: v for k, v in os.environ.items() if k.upper() not in _SCRUBBED_VARS}
    # Fail honestly rather than hang a TUI on a hidden credential prompt.
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_PAGER"] = "cat"
    return env


def _run_user_git(
    root: Path,
    *args: str,
    timeout: float = READ_TIMEOUT_SECONDS,
    check: bool = True,
) -> GitCmdResult:
    """Run one ``git`` invocation AS THE USER, rooted at ``root``.

    Args:
        root: Working directory for the invocation (``cwd``, not
            ``-C``/``--git-dir`` -- this runner acts on the user's own
            checkout, not a shadow repo elsewhere).
        *args: Git subcommand and its arguments (argv, never shell).
        timeout: Seconds before the process is killed and
            :class:`GitWorkspaceError` is raised. Use
            :data:`READ_TIMEOUT_SECONDS`, :data:`COMMIT_TIMEOUT_SECONDS`,
            or :data:`PUSH_TIMEOUT_SECONDS` depending on the operation.
        check: When True (default), a nonzero exit raises
            :class:`GitWorkspaceError` with a capped stderr excerpt.

    Returns:
        The command's :class:`GitCmdResult`.

    Raises:
        GitWorkspaceError: git is not installed, the process timed out,
            an OS-level error occurred launching it, or (when ``check``)
            it exited nonzero.
    """
    git = shutil.which("git")
    if git is None:
        raise GitWorkspaceError("git is not installed")
    try:
        proc = subprocess.run(
            [git, *args],
            cwd=str(root),
            env=_user_git_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitWorkspaceError(
            f"git {args[0]} timed out after {timeout:.0f}s"
        ) from exc
    except OSError as exc:
        raise GitWorkspaceError(str(exc)) from exc
    if check and proc.returncode != 0:
        raise GitWorkspaceError(
            f"git {args[0]} failed ({proc.returncode}): "
            f"{(proc.stderr or '').strip()[:400]}"
        )
    return GitCmdResult(proc.returncode, proc.stdout, proc.stderr)


def _parse_remotes(remote_v_output: str) -> tuple[tuple[str, str], ...]:
    """Parse ``remote -v`` output into ordered, unique ``(name, url)`` push pairs.

    Each line is ``<name>\\t<url> (fetch|push)``. Only ``(push)`` lines are
    kept (fetch and push URLs are usually identical; the push URL is the
    one that matters for this arc's actions).
    """
    seen: dict[str, str] = {}
    for line in remote_v_output.splitlines():
        line = line.rstrip()
        if not line or "(push)" not in line:
            continue
        name_and_rest = line.split("\t", 1)
        if len(name_and_rest) != 2:
            continue
        name, rest = name_and_rest
        url = rest.split(" ", 1)[0]
        if name not in seen:
            seen[name] = url
    return tuple(seen.items())


def detect_git_workspace(root: Path) -> GitWorkspaceInfo | GitWorkspaceRefusal | None:
    """Read-only probe: is ``root`` a real git repository, and what state is it in?

    Never raises -- any :class:`GitWorkspaceError` from an underlying git
    call degrades the result to ``None`` (detection must never crash the
    UI; a `None` result simply hides the git-modes surface).

    Args:
        root: The workspace root to probe. Git modes require the root to
            BE the repository toplevel (see the refusal case below).

    Returns:
        ``None`` when ``root`` is not inside a git repository (or git is
        unavailable). A :class:`GitWorkspaceRefusal` when ``root`` is
        inside a repository whose toplevel is a different (ancestor)
        directory. Otherwise a populated :class:`GitWorkspaceInfo`.
    """
    try:
        return _detect_git_workspace(root)
    except GitWorkspaceError as exc:
        logger.debug(f"git_workspace: detection failed for {root}: {exc}")
        return None


def _detect_git_workspace(root: Path) -> GitWorkspaceInfo | GitWorkspaceRefusal | None:
    toplevel_result = _run_user_git(root, "rev-parse", "--show-toplevel", check=False)
    if toplevel_result.returncode != 0:
        return None
    repo_root = Path(toplevel_result.stdout.strip())
    resolved_root = Path(root).resolve()
    resolved_repo_root = repo_root.resolve()
    if resolved_repo_root != resolved_root:
        return GitWorkspaceRefusal(
            "workspace is inside a repository — git actions need the "
            "workspace root to be the repository root"
        )

    branch_result = _run_user_git(
        root, "symbolic-ref", "--short", "-q", "HEAD", check=False
    )
    if branch_result.returncode == 0:
        branch: str | None = branch_result.stdout.strip()
        detached = False
    else:
        branch = None
        detached = True

    verify_result = _run_user_git(root, "rev-parse", "--verify", "-q", "HEAD", check=False)
    unborn = verify_result.returncode != 0

    remote_result = _run_user_git(root, "remote", "-v", check=False)
    remotes = _parse_remotes(remote_result.stdout) if remote_result.returncode == 0 else ()

    upstream: str | None = None
    upstream_remote: str | None = None
    ahead = 0
    behind = 0
    if not detached and not unborn:
        upstream_result = _run_user_git(
            root, "rev-parse", "--abbrev-ref", "@{upstream}", check=False
        )
        if upstream_result.returncode == 0:
            upstream = upstream_result.stdout.strip()
            remotename_result = _run_user_git(
                root,
                "for-each-ref",
                "--format=%(upstream:remotename)",
                f"refs/heads/{branch}",
                check=False,
            )
            remotename = (
                remotename_result.stdout.strip()
                if remotename_result.returncode == 0
                else ""
            )
            upstream_remote = remotename or None

            ahead_behind_result = _run_user_git(
                root,
                "rev-list",
                "--left-right",
                "--count",
                "@{upstream}...HEAD",
                check=False,
            )
            if ahead_behind_result.returncode == 0:
                parts = ahead_behind_result.stdout.strip().split("\t")
                if len(parts) == 2:
                    try:
                        behind = int(parts[0])
                        ahead = int(parts[1])
                    except ValueError:
                        pass

    return GitWorkspaceInfo(
        root=resolved_root,
        repo_root=resolved_repo_root,
        branch=branch,
        detached=detached,
        unborn=unborn,
        upstream=upstream,
        upstream_remote=upstream_remote,
        remotes=remotes,
        ahead=ahead,
        behind=behind,
    )


# ---------------------------------------------------------------------------
# Working-tree status, per-file diff, untracked preview (TASK-16801 T2).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurrentRootStatus:
    """One root's real working-tree status snapshot, for the `current` mode.

    Attributes:
        root: The workspace root this status was read from (as passed to
            :func:`working_tree_status`, not re-resolved here -- callers
            already hold a resolved root via :class:`GitWorkspaceInfo`).
        info: The :class:`GitWorkspaceInfo` this status was read against;
            its ``unborn`` flag decided whether the ``diff HEAD --numstat``
            call ran at all.
        files: One :class:`ChangedFile` per changed path (tracked changes
            plus untracked adds), in porcelain order. Renames carry
            ``old_path``; untracked adds carry zeroed ``adds``/``dels``
            (counting every new file's lines would mean reading them all
            at load).
        untracked: Root-relative paths that are untracked in this working
            tree. Carried separately from `files` (never as a new
            `ChangedFile` field) -- the screen needs untracked-ness
            independently of the collapsed status letter, e.g. to route a
            leaf's diff through :func:`untracked_preview` instead of
            :func:`working_tree_diff`.
    """

    root: Path
    info: GitWorkspaceInfo
    files: tuple[ChangedFile, ...]
    untracked: frozenset[str]


def _collapse_status_xy(xy: str) -> str:
    """Collapse one porcelain v1 XY code to the single letter callers group on.

    Precedence (spec §4, rationale per branch):

    - ``"??"`` -> ``"A"``. Untracked-ness is reported separately via
      :attr:`CurrentRootStatus.untracked`; the caller synthesizes an
      untracked-preview diff for these rather than running ``git diff``.
    - a rename in either column (``"R"`` in ``xy``) wins next -- a rename
      always carries an ``old_path`` the caller must not drop by
      collapsing to something else first.
    - then a copy (``"C"`` in ``xy``), same reasoning as rename.
    - then ``"D"`` -- this is what makes ``"AD"`` (staged-add,
      then-deleted-in-worktree) collapse to ``"D"`` rather than ``"A"``:
      the file is GONE from disk, so reporting ``"A"`` would advertise a
      file that does not exist.
    - then ``"A"`` for the remaining add cases (e.g. ``"A "``, ``" A"``
      is not a real git code but ``"AM"`` etc. fall through to here).
    - anything else passes through as its first non-space character
      VERBATIM (``"M"``, ``"T"`` typechange, ``"U"`` unmerged, ...),
      matching :class:`ChangedFile`'s "unknown letters pass through"
      contract rather than coercing them into a lie.
    """
    if xy == "??":
        return "A"
    if "R" in xy:
        return "R"
    if "C" in xy:
        return "C"
    if "D" in xy:
        return "D"
    if "A" in xy:
        return "A"
    for ch in xy:
        if ch != " ":
            return ch
    return xy


def _parse_porcelain_v1(
    raw: str,
) -> tuple[list[tuple[str, str, str | None]], frozenset[str]]:
    """NUL-token walk over ``git status --porcelain=v1 -z -uall`` output.

    Each record is one NUL-terminated ``"XY PATH"`` token; when the index
    status (``X``, the first char) is ``R`` or ``C``, ONE additional
    NUL-terminated token follows carrying the OLD path -- the NEW path
    comes first (empirically verified, spec §2 probe 2: a rename record
    is ``R<XY>\\0new\\0old\\0``).

    Args:
        raw: The command's raw (already text-decoded) stdout.

    Returns:
        A tuple of ``(collapsed_status, path, old_path)`` rows in
        porcelain order, plus the frozenset of untracked (``"??"``)
        paths.
    """
    tokens = [t for t in raw.split("\0") if t]
    entries: list[tuple[str, str, str | None]] = []
    untracked: set[str] = set()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        xy, path = token[:2], token[3:]
        i += 1
        old_path: str | None = None
        if xy[:1] in ("R", "C"):
            old_path = tokens[i]
            i += 1
        if xy == "??":
            untracked.add(path)
        entries.append((_collapse_status_xy(xy), path, old_path))
    return entries, frozenset(untracked)


def _parse_numstat(raw: str) -> dict[str, tuple[int, int, bool]]:
    """NUL-token walk over ``git diff HEAD --numstat -z`` output.

    Rename-tolerant token walk, copied from
    :meth:`tldw_chatbook.Workspaces.change_tracking.ShadowRepo.changed_files`:
    a normal record is one token ``"adds\\tdels\\tpath"``; a rename record
    splits into three tokens -- ``"adds\\tdels\\t"`` (empty path field),
    then the OLD path, then the NEW path -- and this merge is keyed on
    the NEW path (the same path :func:`_parse_porcelain_v1` uses for
    renames), so the two streams merge cleanly on path.

    Args:
        raw: The command's raw (already text-decoded) stdout.

    Returns:
        ``path -> (adds, dels, binary)``. A binary change reports
        ``adds=0, dels=0, binary=True`` (git prints ``"-"`` for a binary
        file's counts).
    """
    tokens = [t for t in raw.split("\0") if t]
    counts: dict[str, tuple[int, int, bool]] = {}
    i = 0
    while i < len(tokens):
        head = tokens[i]
        adds_s, dels_s, rest = head.split("\t", 2)
        if rest == "":
            path = tokens[i + 2]
            i += 3
        else:
            path = rest
            i += 1
        binary = adds_s == "-"
        counts[path] = (
            0 if binary else int(adds_s),
            0 if binary else int(dels_s),
            binary,
        )
    return counts


def working_tree_status(root: Path, info: GitWorkspaceInfo) -> CurrentRootStatus:
    """Read the real working tree's status at ``root``.

    Runs ``status --porcelain=v1 -z -uall`` (untracked directories
    expanded to one row per file, never collapsed to the directory) and,
    unless ``info.unborn``, ONE ``diff HEAD --numstat -z`` call to merge
    in tracked line counts -- ``git diff HEAD`` is a fatal error on an
    unborn branch (spec §2 probe 4), so it is never invoked there; every
    file on an unborn branch is untracked and gets zeroed counts.

    Args:
        root: Workspace root (must be the repo toplevel; use
            :func:`detect_git_workspace` to establish that first).
        info: The root's detected :class:`GitWorkspaceInfo`.

    Returns:
        The root's :class:`CurrentRootStatus`.

    Raises:
        GitWorkspaceError: A git invocation failed (git missing, timed
            out, or exited nonzero).
    """
    status_result = _run_user_git(root, "status", "--porcelain=v1", "-z", "-uall")
    entries, untracked = _parse_porcelain_v1(status_result.stdout)

    counts: dict[str, tuple[int, int, bool]] = {}
    if not info.unborn:
        numstat_result = _run_user_git(root, "diff", "HEAD", "--numstat", "-z")
        counts = _parse_numstat(numstat_result.stdout)

    files = tuple(
        ChangedFile(
            path=path,
            status=status,
            adds=counts.get(path, (0, 0, False))[0],
            dels=counts.get(path, (0, 0, False))[1],
            old_path=old_path,
            binary=counts.get(path, (0, 0, False))[2],
        )
        for status, path, old_path in entries
    )
    return CurrentRootStatus(root=root, info=info, files=files, untracked=untracked)


def working_tree_diff(root: Path, path: str) -> str:
    """Unified diff for one TRACKED file, working tree vs HEAD.

    Callers must not invoke this for an unborn HEAD (there is no HEAD to
    diff against -- every file there is untracked) or for an untracked
    path; use :func:`untracked_preview` for those instead.

    Args:
        root: Workspace root.
        path: Root-relative path to diff.

    Returns:
        Unified diff text (empty when there is no textual difference).

    Raises:
        GitWorkspaceError: The git invocation failed.
    """
    result = _run_user_git(root, "diff", "HEAD", "--", path)
    return result.stdout


def untracked_preview(root: Path, path: str, max_lines: int) -> str:
    """Render a bounded preview of an untracked file for the diff pane.

    Synthesized entirely in Python from a capped read -- never
    ``git diff --no-index`` (exit-code-1 semantics, platform quirks) and
    never an index trick like ``--intent-to-add`` (mutating the user's
    index from a VIEW is forbidden).

    Args:
        root: Workspace root.
        path: Root-relative path of the untracked file.
        max_lines: Maximum number of content lines to render.

    Returns:
        A ``"new file: <path>"`` header, a blank line, then up to
        ``max_lines`` ``"+"``-prefixed lines with a trailing truncation
        note when the file has more; or, for a file whose first 8KB
        contain a NUL byte, a one-line binary label; or, on any
        :class:`OSError` (vanished file, permission denied, ...), an
        honest one-line error message. Never raises.
    """
    target = root / path
    try:
        size = target.stat().st_size
        cap = max_lines * 400
        with target.open("rb") as fh:
            head = fh.read(cap)
    except OSError as exc:
        return f"could not read {path}: {exc}"

    if b"\x00" in head[:8192]:
        return f"new file: {path}\n(binary file, {size} bytes)"

    text = head.decode("utf-8", "replace")
    lines = text.splitlines()
    capped_read = size > len(head)
    shown = lines[:max_lines]
    out = [f"new file: {path}", ""]
    out.extend(f"+{line}" for line in shown)
    if capped_read or len(lines) > max_lines:
        out.append(f"… truncated at {max_lines} lines")
    return "\n".join(out) + "\n"
