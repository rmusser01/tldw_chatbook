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
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import quote, urlparse

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
    #: Every push destination per remote (TASK-19701). `remotes` keeps ONE
    #: entry per remote because the sole-remote derivations key off its
    #: length; a remote with several `pushurl`s reaches all of these.
    #: Defaulted so existing constructions stay valid.
    remote_push_urls: tuple[tuple[str, tuple[str, ...]], ...] = ()


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


#: The sibling global pathspec-interpretation vars. They must be scrubbed
#: rather than merely overridden: git REFUSES outright ("fatal: global
#: 'literal' pathspec setting is incompatible with all other global
#: pathspec settings") when ``GIT_LITERAL_PATHSPECS`` is set alongside
#: ``GIT_GLOB_PATHSPECS`` or ``GIT_ICASE_PATHSPECS``, so a user who
#: exports one of those in their shell would otherwise see EVERY git
#: invocation in this module fail. Scrubbing them is also correct on its
#: own terms -- an ambient `GIT_ICASE_PATHSPECS` would silently widen a
#: file selection.
_SCRUBBED_PATHSPEC_VARS = frozenset(
    {
        "GIT_GLOB_PATHSPECS",
        "GIT_NOGLOB_PATHSPECS",
        "GIT_ICASE_PATHSPECS",
    }
)


def _user_git_env() -> dict[str, str]:
    """Ambient environment, minus repo-targeting vars, plus safety pins.

    See the module docstring's three-runner table for why this posture
    (preserve identity/credentials, scrub only targeting vars) is correct
    HERE and would be wrong for the other two git runners in this repo.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in _SCRUBBED_VARS and k.upper() not in _SCRUBBED_PATHSPEC_VARS
    }
    # Fail honestly rather than hang a TUI on a hidden credential prompt.
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_PAGER"] = "cat"
    # `--` stops OPTION parsing; it does NOT stop PATHSPEC MAGIC. Every
    # path this module passes to git comes from `git status` -- i.e. from
    # the REPOSITORY -- and a file may legally be NAMED `:!nothing`,
    # `:(glob)*`, or `:/`. Reproduced against real git: a one-file
    # selection of a file named `:!nothing` was read as the exclude
    # pathspec "everything except paths matching nothing", and
    # `add -A -- ':!nothing'` + `commit -- ':!nothing'` committed FOUR
    # files (the canonical index-hijack bug, reached through a filename);
    # `diff HEAD -- ':!nothing'` likewise rendered other files' diffs into
    # the pane. `GIT_LITERAL_PATHSPECS=1` makes every pathspec literal at
    # ONE choke point, covering every call site in this module; verified
    # to leave normal, spaced, UTF-8 and directory pathspecs working.
    env["GIT_LITERAL_PATHSPECS"] = "1"
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


def _parse_remote_push_urls(
    remote_v_output: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Parse ``remote -v`` into ``(name, (every push URL))`` pairs.

    Each line is ``<name>\\t<url> (fetch|push)``; only ``(push)`` lines are
    kept (the push URL is the one this arc's actions use, and git has
    already resolved ``remote.<name>.pushurl`` and
    ``url.<other>.pushInsteadOf`` into it).

    Two parsing details, both real defects found by Qodo on PR #1959 and
    both reproduced against live git first:

    * the URL is recovered by stripping the trailing ``" (push)"``, NOT by
      splitting on the first space -- a local-path remote at
      ``/tmp/with space.git`` was otherwise reported as ``/tmp/with``, a
      path that does not exist;
    * a remote may configure SEVERAL ``pushurl`` entries, and git emits one
      ``(push)`` line per destination -- a push reaches all of them, so
      keeping only the first understates where the user's code is about to
      go.
    """
    ordered: dict[str, list[str]] = {}
    for line in remote_v_output.splitlines():
        line = line.rstrip()
        if not line.endswith("(push)"):
            continue
        name_and_rest = line.split("\t", 1)
        if len(name_and_rest) != 2:
            continue
        name, rest = name_and_rest
        url = rest[: -len(" (push)")].rstrip()
        if not url:
            continue
        urls = ordered.setdefault(name, [])
        if url not in urls:
            urls.append(url)
    return tuple((name, tuple(urls)) for name, urls in ordered.items())


def _parse_remotes(remote_v_output: str) -> tuple[tuple[str, str], ...]:
    """One ``(name, url)`` pair per remote -- the de-duplicated view.

    Kept one-entry-per-remote deliberately: ``_resolve_push_remote`` and
    the push dialog's "do I need to ask which remote?" check both key off
    this tuple's LENGTH, so letting a multi-``pushurl`` remote appear twice
    would make a single remote look like a choice between two. The full
    destination set lives in :func:`_parse_remote_push_urls`.
    """
    return tuple(
        (name, urls[0])
        for name, urls in _parse_remote_push_urls(remote_v_output)
        if urls
    )


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
    _push_urls = (
        _parse_remote_push_urls(remote_result.stdout)
        if remote_result.returncode == 0
        else ()
    )
    remotes = tuple((name, urls[0]) for name, urls in _push_urls if urls)

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
        remote_push_urls=_push_urls,
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
        root: The workspace root this status was read from, ALWAYS
            ``info.root`` (i.e. resolved) rather than whatever spelling
            the caller passed to :func:`working_tree_status` -- so
            ``status.root == status.info.root`` always holds and a caller
            that keyed a dict by ``status.root`` string can never
            silently miss a lookup keyed by ``info.root`` instead.
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
        # Deliberately checks X (the index column) only, not "R"/"C" in
        # xy as a whole: git only ever records a rename/copy in the
        # index column, never the worktree column (Y), so a real "R"/"C"
        # can't appear at xy[1] -- do not "generalize" this to `in xy`.
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


#: Flags that stop repository CONFIGURATION from rewriting what a diff
#: says. Ported whole from the read-only agent git tools
#: (``Tools/git_tool_impls.py``'s ``git_diff``/``_run_diff_command``, which
#: call them the machine-safe flags) rather than two-thirds of the set --
#: all three shapes are reachable by anything that can write
#: ``.git/config`` in the root this feature operates on, and all three
#: were reproduced against real git:
#:
#: * ``diff.external = <script>`` -- the pane printed ``TOTALLY FABRICATED
#:   DIFF OUTPUT`` for a real edit (``--no-ext-diff``);
#: * a ``.gitattributes`` ``diff=<driver>`` textconv whose output is
#:   constant -- the pane rendered NOTHING (0 bytes) for a genuinely
#:   changed file, while ``--numstat`` still reported ``1 1 a.txt`` and
#:   status still listed the row, so the review surface contradicted its
#:   own file list (``--no-textconv``);
#: * ``color.ui = always`` -- ANSI escapes in a CAPTURED, non-tty diff
#:   (``--no-color``).
#:
#: This is a REVIEW surface whose whole purpose is telling the user the
#: truth about their changes before they commit, so a faked or blanked
#: diff is deception rather than mere breakage. ``--numstat`` is unaffected
#: by all three (verified), but the flags are applied there too so no call
#: site depends on which of them git happens to honour today.
_MACHINE_SAFE_DIFF_FLAGS: tuple[str, ...] = (
    "--no-ext-diff",
    "--no-textconv",
    "--no-color",
)


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
        numstat_result = _run_user_git(
            root, "diff", *_MACHINE_SAFE_DIFF_FLAGS, "HEAD", "--numstat", "-z"
        )
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
    # Always store info.root (resolved), never the raw `root` argument --
    # a caller passing a relative or symlinked spelling must not get a
    # `status.root` that silently disagrees with `status.info.root`.
    return CurrentRootStatus(root=info.root, info=info, files=files, untracked=untracked)


def working_tree_diff(root: Path, path: str) -> str:
    """Unified diff for one TRACKED file, working tree vs HEAD.

    Callers must not invoke this for an unborn HEAD (there is no HEAD to
    diff against -- every file there is untracked) or for an untracked
    path; use :func:`untracked_preview` for those instead.

    Carries :data:`_MACHINE_SAFE_DIFF_FLAGS`, without which repository
    configuration can make this REVIEW surface lie -- print a fabricated
    diff, or print nothing at all for a file the same read lists as
    changed.

    Args:
        root: Workspace root.
        path: Root-relative path to diff.

    Returns:
        Unified diff text (empty when there is no textual difference).

    Raises:
        GitWorkspaceError: The git invocation failed.
    """
    result = _run_user_git(
        root, "diff", *_MACHINE_SAFE_DIFF_FLAGS, "HEAD", "--", path
    )
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
    # Qodo #1 (PR #1914): git lists a SYMLINK as an ordinary untracked
    # entry, and `read`/`stat` follow it -- so a link planted inside the
    # root (agent write tools can create one) would render an
    # out-of-workspace file's content into the review pane, which the
    # V1.5 annotate/delivery loop then feeds back to the model. Validate
    # through the repo's shared boundary helper (CLAUDE.md: "Use
    # path_validation.py for file paths") rather than an ad-hoc check;
    # `validate_path` resolves symlinks before comparing, so this covers
    # both traversal segments and link escapes.
    from tldw_chatbook.Utils.path_validation import is_safe_path

    if not is_safe_path(target, root):
        return f"could not read {path}: resolves outside the workspace root"
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


# ---------------------------------------------------------------------------
# Commit engine -- pathspec commit with per-step outcomes (TASK-16801 T3).
# ---------------------------------------------------------------------------

#: Refs whose existence marks an in-progress merge/rebase/cherry-pick --
#: git refuses a partial (pathspec) commit during any of these, and the
#: raw git error is worse copy than naming the operation ourselves.
_IN_PROGRESS_HEAD_REFS: tuple[str, ...] = (
    "MERGE_HEAD",
    "REBASE_HEAD",
    "CHERRY_PICK_HEAD",
)


class CommitRefusedError(GitWorkspaceError):
    """The commit was refused before touching anything (active run).

    Mirrors :class:`tldw_chatbook.Workspaces.change_revert.RevertRefusedError`
    -- the per-root lock serializes *git* operations, but an agent's own
    file tools do not take it, so committing under a writing agent could
    stage/commit a half-written file. The caller injects the
    ``run_active`` probe, and this raises BEFORE any git command runs.
    """


@dataclass(frozen=True)
class GitStepOutcome:
    """One step's outcome from :func:`commit_selected`.

    Attributes:
        step: The step's name (``"in-progress-check"``, ``"validate-branch"``,
            ``"create-branch"``, ``"stage"``, ``"commit"``, or
            ``"resolve-sha"``).
        ok: Whether the step succeeded.
        detail: Human-readable failure copy (a capped stderr excerpt, or a
            fixed reason string for a guard) when ``ok`` is False; empty
            on success.
    """

    step: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class CommitResult:
    """The full outcome of one :func:`commit_selected` call.

    Attributes:
        outcomes: The steps worth REPORTING, in execution order -- read it
            as failure detail, not as a log of everything that ran. It is
            deliberately lossy in both directions: a guard step
            (``"in-progress-check"``, ``"validate-branch"``) that PASSED
            appends nothing and is only present when it BLOCKED the
            commit, and ``"stage"`` is absent entirely when the add was
            skipped (every selected path already absent from the
            worktree -- a staged deletion or a completed ``git mv``).
            What IS guaranteed, and what
            :meth:`~tldw_chatbook.UI.Screens.change_review_screen.ChangeReviewScreen._land_commit_result`
            relies on: the mutating steps that actually run
            (``"create-branch"``, ``"stage"``, ``"commit"``,
            ``"resolve-sha"``) each append win or lose, so the LAST row is
            always the step that stopped the run, and a run that reached
            the end carries an all-``ok`` tail. Do not infer from a
            MISSING row that its step did not happen.
        short_sha: ``rev-parse --short HEAD`` after a landed commit, or
            ``None`` on any failure (including a failure to resolve the
            sha of an otherwise-landed commit).
    """

    outcomes: tuple[GitStepOutcome, ...]
    short_sha: str | None


def _first_in_progress_ref(root: Path) -> str | None:
    """Return the first in-progress-operation ref that exists, if any.

    Args:
        root: Workspace root.

    Returns:
        The ref name (e.g. ``"MERGE_HEAD"``), or ``None`` when none of
        :data:`_IN_PROGRESS_HEAD_REFS` resolve.
    """
    for ref in _IN_PROGRESS_HEAD_REFS:
        result = _run_user_git(root, "rev-parse", "--verify", "-q", ref, check=False)
        if result.returncode == 0:
            return ref
    return None


def commit_selected(
    root: Path,
    files: Sequence[str],
    message: str,
    new_branch: str | None,
    *,
    run_active: Callable[[], bool],
) -> CommitResult:
    """Commit exactly the selected files at ``root`` as a pathspec commit.

    Runs ``git add -A -- <files>`` then ``git commit -m <message> --
    <files>`` -- a bare ``git commit -m`` would sweep in whatever the user
    had already staged in a terminal (spec §2 probe 1); the pathspec form
    commits EXACTLY the selected paths and leaves any unrelated pre-staged
    index entry staged and uncommitted.

    Step order, stopping at the first failure:

    1. ``run_active()`` -- refuse before touching anything.
    2. ``in-progress-check`` -- refuse during a merge/rebase/cherry-pick.
    3. When ``new_branch``: ``validate-branch`` (``check-ref-format``,
       also an option-injection guard against a leading ``-``), then
       ``create-branch`` (``checkout -b``).
    4. ``stage`` -- ``git add -A -- <files present in the worktree>``,
       SKIPPED entirely when that filtered list is empty.
    5. ``commit`` -- ``git commit -m <message> -- <files>`` (the FULL
       pathspec; user hooks and gpg signing run -- this is the user's
       repo, their rules).
    6. ``resolve-sha`` -- ``git rev-parse --short HEAD``.

    Args:
        root: Workspace root (must be the repo toplevel).
        files: Root-relative paths to stage and commit. Always placed
            after ``--`` in both the ``add`` and ``commit`` argv, so a
            path that happens to start with ``-`` is never parsed as an
            option -- and, since ``--`` does NOT stop pathspec MAGIC,
            :func:`_user_git_env` additionally pins
            ``GIT_LITERAL_PATHSPECS=1`` so a path NAMED ``:!nothing``
            cannot hijack the selection.
        message: The commit message, passed as ``-m``'s argv element
            (never shell-interpolated) -- a leading ``-`` is safe (spec
            §2 probe 5: ``-m``'s sticky-arg consumption makes a
            dash-leading message safe as argv, e.g. message ``"--amend"``
            commits literally rather than amending).
        new_branch: When set, create and check out this branch before
            staging. ``None`` or empty commits to the current branch.
        run_active: Probe for an active run on this root's workspace;
            True refuses the whole commit before any git command runs
            (mirrors :func:`tldw_chatbook.Workspaces.change_revert.revert_paths`).

    Returns:
        The :class:`CommitResult`.

    Raises:
        CommitRefusedError: A run is active -- finish or stop it first.
        GitWorkspaceError: ``files`` is empty or ``message`` is blank
            (the UI validates first; the engine still refuses rather
            than run a no-op/empty-message commit).
    """
    if run_active():
        raise CommitRefusedError(
            "a run is active on this workspace — finish or stop the run first"
        )
    if not files:
        raise GitWorkspaceError("no files selected to commit")
    if not message.strip():
        raise GitWorkspaceError("commit message must not be blank")

    outcomes: list[GitStepOutcome] = []

    in_progress_ref = _first_in_progress_ref(root)
    if in_progress_ref is not None:
        outcomes.append(
            GitStepOutcome(
                "in-progress-check",
                False,
                "finish or abort the merge/rebase/cherry-pick first",
            )
        )
        return CommitResult(tuple(outcomes), None)

    if new_branch:
        validate_result = _run_user_git(
            root, "check-ref-format", "--branch", new_branch, check=False
        )
        if validate_result.returncode != 0:
            outcomes.append(
                GitStepOutcome(
                    "validate-branch",
                    False,
                    (validate_result.stderr or "").strip()[:400],
                )
            )
            return CommitResult(tuple(outcomes), None)

        create_result = _run_user_git(root, "checkout", "-b", new_branch, check=False)
        create_ok = create_result.returncode == 0
        outcomes.append(
            GitStepOutcome(
                "create-branch",
                create_ok,
                "" if create_ok else (create_result.stderr or "").strip()[:400],
            )
        )
        if not create_ok:
            return CommitResult(tuple(outcomes), None)

    # `git add` refuses a pathspec matching nothing in the WORKTREE, so a
    # path whose change is already recorded in the index and absent from
    # disk -- a queued `git mv` rename's old name, a `git rm` staged
    # deletion, a plain unstaged deletion -- used to dead-end the whole
    # commit at this step (`fatal: pathspec 'a.txt' did not match any
    # files`). `git commit -- <path>` records those on its own, so only
    # the ADD list is filtered; the COMMIT keeps the full pathspec.
    #
    # `os.path.lexists`, never `Path.exists()`: a BROKEN SYMLINK is a real
    # worktree entry git can stage, and `Path.exists()` follows the link
    # and calls it absent.
    add_files = [path for path in files if os.path.lexists(os.path.join(root, path))]
    if add_files:
        stage_result = _run_user_git(root, "add", "-A", "--", *add_files, check=False)
        stage_ok = stage_result.returncode == 0
        outcomes.append(
            GitStepOutcome(
                "stage",
                stage_ok,
                "" if stage_ok else (stage_result.stderr or "").strip()[:400],
            )
        )
        if not stage_ok:
            return CommitResult(tuple(outcomes), None)
    # ...and when NOTHING in the selection is on disk, the add is skipped
    # outright rather than run with an empty pathspec: `git add -A --`
    # with no paths stages the WHOLE TREE (verified), which would commit
    # files the user never checked.

    commit_result = _run_user_git(
        root,
        "commit",
        "-m",
        message,
        "--",
        *files,
        timeout=COMMIT_TIMEOUT_SECONDS,
        check=False,
    )
    commit_ok = commit_result.returncode == 0
    outcomes.append(
        GitStepOutcome(
            "commit",
            commit_ok,
            "" if commit_ok else (commit_result.stderr or "").strip()[:400],
        )
    )
    if not commit_ok:
        return CommitResult(tuple(outcomes), None)

    sha_result = _run_user_git(root, "rev-parse", "--short", "HEAD", check=False)
    short_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else None
    outcomes.append(
        GitStepOutcome(
            "resolve-sha",
            short_sha is not None,
            "" if short_sha is not None else (sha_result.stderr or "").strip()[:400],
        )
    )
    return CommitResult(tuple(outcomes), short_sha)


# ---------------------------------------------------------------------------
# Push engine + PR compare URL (TASK-16801 T4).
# ---------------------------------------------------------------------------

#: Marker substrings in a push's stderr excerpt that indicate git could not
#: obtain credentials non-interactively under ``GIT_TERMINAL_PROMPT=0``
#: (spec §6) -- each maps to the same appended hint, :data:`_CREDENTIAL_HINT`.
_CREDENTIAL_FAILURE_MARKERS: tuple[str, ...] = (
    "could not read Username",
    "terminal prompts disabled",
    "Permission denied",
    "Authentication failed",
)

_CREDENTIAL_HINT = (
    " — credentials were not available non-interactively; push once from a "
    "terminal or configure a credential helper/ssh agent"
)

#: Hosts this arc's PR compare-URL builder supports, in the order named to
#: the user in refusal copy (spec §6, AC #2).
_SUPPORTED_PR_HOSTS: tuple[str, ...] = (
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "codeberg.org",
)

_UNSUPPORTED_HOST_REASON = "PR links support " + ", ".join(_SUPPORTED_PR_HOSTS)

#: Matches the scp-like remote URL shape ``[user@]host:path`` (e.g.
#: ``git@github.com:o/r.git``) -- anything containing ``"://"`` is a real
#: URL (``https://...``, ``ssh://...``) and is parsed by
#: :func:`urllib.parse.urlparse` instead.
_SCP_LIKE_REMOTE_RE = re.compile(r"^(?:[^@/]+@)?(?P<host>[^:/]+):(?P<path>.+)$")


@dataclass(frozen=True)
class PushResult:
    """The outcome of one :func:`push_current` call.

    Attributes:
        state: One of ``"pushed"``, ``"up_to_date"``, or ``"failed"``.
        detail: Empty on success. On failure, a capped stderr excerpt --
            possibly with :data:`_CREDENTIAL_HINT` appended by
            :func:`_push_failure_detail` -- never a rolled-up "git
            failed" (spec §8 per-step honesty).
    """

    state: str
    detail: str = ""


def _push_failure_detail(stderr_excerpt: str) -> str:
    """Append the credential-helper hint when the excerpt looks like one.

    Pure string classification, no I/O, so it is unit-testable without a
    repository (spec §6).

    Args:
        stderr_excerpt: The push's (already capped) stderr excerpt.

    Returns:
        ``stderr_excerpt`` unchanged, or with :data:`_CREDENTIAL_HINT`
        appended when it contains any of
        :data:`_CREDENTIAL_FAILURE_MARKERS`.
    """
    if any(marker in stderr_excerpt for marker in _CREDENTIAL_FAILURE_MARKERS):
        return stderr_excerpt + _CREDENTIAL_HINT
    return stderr_excerpt


def _resolve_push_remote(info: GitWorkspaceInfo, remote: str | None) -> str | None:
    """Resolve the remote name :func:`push_current` should push to.

    Args:
        info: The root's detected workspace info.
        remote: An explicit remote name, or ``None`` to derive one.

    Returns:
        ``remote`` when given. Otherwise ``info.upstream_remote`` when an
        upstream is configured, else the sole entry of ``info.remotes``
        when there is exactly one. ``None`` when none of these resolve --
        callers must supply an explicit ``remote`` in the ambiguous case
        (more than one remote, no upstream); this function does not guess
        between them.
    """
    if remote is not None:
        return remote
    if info.upstream_remote is not None:
        return info.upstream_remote
    if len(info.remotes) == 1:
        return info.remotes[0][0]
    return None


def _upstream_remote_ref(root: Path, branch: str) -> str | None:
    """The ref ``branch``'s upstream names ON THE REMOTE, fully qualified.

    Read from ``%(upstream:remoteref)`` -- the remote-side counterpart of
    the ``%(upstream:remotename)`` field detection already uses, and the
    only source that is guaranteed to agree with ``info.upstream``.
    ``git config --get branch.<b>.merge`` is NOT equivalent: a
    multi-valued ``merge`` (legal, and writable by anything that can write
    ``.git/config``) makes ``--get`` return the LAST value while
    ``@{upstream}`` resolves the FIRST -- i.e. the push would target a
    different ref than the UI names.

    A branch name can never glob-poison the ``refs/heads/<branch>``
    pattern: git refuses ``*``, ``?`` and ``[`` in ref names outright.

    Args:
        root: Workspace root.
        branch: The current branch's short name.

    Returns:
        The remote-side ref (e.g. ``"refs/heads/main"``), or ``None`` when
        it cannot be read or is not fully qualified -- callers must REFUSE
        on ``None`` rather than fall back to a refspec-less push.
    """
    result = _run_user_git(
        root,
        "for-each-ref",
        "--format=%(upstream:remoteref)",
        f"refs/heads/{branch}",
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.splitlines()[0].strip() if result.stdout.strip() else ""
    # Must be fully qualified: a bare or empty value would build a refspec
    # whose destination git resolves by its own rules (or, when empty,
    # `refs/heads/x:` -- an invalid refspec at best).
    if not value.startswith("refs/"):
        return None
    return value


def push_current(root: Path, info: GitWorkspaceInfo, remote: str | None) -> PushResult:
    """Push the current branch, honestly reporting a non-fast-forward rejection.

    Never passes ``--force``/``--force-with-lease`` (spec §6,
    no-silent-destructive precedent) -- a rejected push surfaces git's own
    stderr excerpt rather than retrying with force. Equally important, and
    NOT visible in an argv audit: the push always carries an EXPLICIT,
    fully-qualified refspec, so no repository-supplied configuration can
    turn our own argv into a destructive push (see the comment at the
    refspec construction).

    Args:
        root: Workspace root (must be the repo toplevel).
        info: The root's detected :class:`GitWorkspaceInfo`.
        remote: Explicit target remote name, or ``None`` to derive one via
            :func:`_resolve_push_remote`.

    Returns:
        The push's :class:`PushResult`.

    Raises:
        GitWorkspaceError: HEAD is detached (``"no branch checked out"``),
            no remote could be resolved
            (``"no git remote configured"``), the BRANCH
            (``"unsupported branch name"``) or resolved REMOTE
            (``"unsupported remote name"``) has a name beginning with
            ``"-"`` -- git would read either as an option rather than as a
            ref/remote, which is a ref-destruction vector; see the comments
            at the two checks -- or the upstream's remote-side ref could
            not be resolved (``"could not resolve where the upstream
            points"``).
    """
    if info.detached or info.branch is None:
        raise GitWorkspaceError("no branch checked out")

    # THE SAME ARGUMENT INJECTION AS THE REMOTE NAME BELOW, through the
    # BRANCH (T8 re-review). `info.branch` lands in argv position 4 of
    # `push -u <remote> <branch>`, and git reads a leading-dash argument
    # there as an OPTION. It arrives straight from `symbolic-ref` with no
    # validator, and `check-ref-format` does NOT cover it in either sense:
    # that call guards only `commit_selected`'s NEW-branch path, and
    # `git check-ref-format refs/heads/--mirror` exits **0** regardless.
    #
    # A repository can simply ship `.git/HEAD` = `ref: refs/heads/--mirror`.
    # Reproduced against real git: `push -u origin --mirror` DELETED
    # `refs/heads/release` from the shared remote, force-rewound
    # `refs/heads/main` off another clone's commit, and pushed junk
    # `refs/remotes/origin/*` refs into it; `--all` published every local
    # branch, leaking private WIP.
    #
    # Refused, never sanitized -- same reasoning as the remote name: there
    # is no `--` escape for this positional (`push -u <remote> -- <branch>`
    # is not a refspec), and rewriting the name would push a DIFFERENT
    # branch than the one checked out. `git branch -m` renames it.
    if info.branch.startswith("-"):
        raise GitWorkspaceError("unsupported branch name")

    target_remote = _resolve_push_remote(info, remote)
    if target_remote is None:
        raise GitWorkspaceError("no git remote configured")

    # ARGUMENT INJECTION (T8 review, verified against real git): the remote
    # name lands in argv position 1 of `git push <remote>`, and git reads a
    # leading-dash argument there as an OPTION, not as a remote. A remote
    # NAMED `--force` is perfectly legal to create -- `git remote add --
    # --force <url>` exits 0 -- and a repository can simply SHIP one in
    # `.git/config`, upstream and all. `git push --force` then rewrites the
    # remote branch: the reproduction produced `+ 4fd1108...c1e7731 main ->
    # main (forced update)`, destroying a second clone's commit, and this
    # module's own "never force-push" rule was never violated by any
    # literal we wrote. `--mirror` is the same shape and DELETES remote refs.
    #
    # Refused, never sanitized: stripping the dashes would push to a
    # different remote than the one named, and a `--`/`=` escape does not
    # exist for this positional (`git push -- <remote>` is not accepted).
    # A repository whose remote is option-shaped simply cannot be pushed
    # from here; it can be renamed with one `git remote rename`.
    if target_remote.startswith("-"):
        raise GitWorkspaceError("unsupported remote name")

    # THE THIRD ARGUMENT-INJECTION SHAPE (whole-branch review), and the one
    # neither guard above can see: the destructive option never appears in
    # OUR argv at all. A push that carries NO refspec lets `.git/config`
    # decide what the push does -- and an agent CAN write `.git/config`,
    # because no `.git` exclusion exists in `workspace_file_roots.py` /
    # `file_operation_tools.py`, and this feature operates on exactly that
    # root. Reproduced against real git, with a non-dash remote and a
    # non-dash branch (both existing guards passing cleanly):
    #
    #   remote.origin.push = +refs/heads/*:refs/heads/*
    #       -> `+ aa1df20...80ecf04 main -> main (forced update)`; another
    #          clone's commit destroyed. Leaks into the `-u` form too, which
    #          looks its bare branch name up in the configured push refspecs
    #          and inherits that refspec's `+`.
    #   remote.origin.push = :refs/heads/release
    #       -> `- [deleted] release`, a branch this push never named.
    #   remote.origin.mirror = true
    #       -> forced update PLUS `- [deleted] release` / `- [deleted] v1`.
    #   push.default = matching
    #       -> published an unrelated local branch's private commit while
    #          the modal named a different branch.
    #
    # An EXPLICIT, fully-qualified refspec takes every one of those
    # decisions back: a command-line refspec supersedes `remote.<n>.push`
    # (so the `+` is gone and the force config is rejected non-fast-forward
    # instead), makes `mirror` fail honestly ("--mirror can't be combined
    # with refspecs"), and makes `push.default` irrelevant. Verified: the
    # precious commit and the `release`/`v1` refs survive all four.
    if info.upstream is not None:
        # The remote name comes from detection's `%(upstream:remotename)`
        # field (never derived by splitting `info.upstream` on "/" --
        # remote names can themselves contain "/", spec §2 probe 6), and
        # the DESTINATION from the matching `%(upstream:remoteref)` -- the
        # local branch's name would be wrong whenever it differs from the
        # upstream's.
        remote_ref = _upstream_remote_ref(root, info.branch)
        if remote_ref is None:
            # Never degrade to a refspec-less push: that IS the vector.
            raise GitWorkspaceError("could not resolve where the upstream points")
        args: tuple[str, ...] = (
            "push",
            target_remote,
            f"refs/heads/{info.branch}:{remote_ref}",
        )
    else:
        # No upstream yet, so no configured merge ref to honour: the
        # destination is the branch's own fully-qualified name, which is
        # also what `-u` then records as `branch.<b>.merge`.
        args = (
            "push",
            "-u",
            target_remote,
            f"refs/heads/{info.branch}:refs/heads/{info.branch}",
        )

    result = _run_user_git(root, *args, timeout=PUSH_TIMEOUT_SECONDS, check=False)
    if result.returncode == 0:
        combined = f"{result.stdout}\n{result.stderr}"
        if "Everything up-to-date" in combined:
            return PushResult("up_to_date")
        return PushResult("pushed")

    excerpt = (result.stderr or "").strip()[:400]
    return PushResult("failed", _push_failure_detail(excerpt))


def _parse_remote_url(url: str) -> tuple[str, str, str] | None:
    """Parse a git remote push URL into ``(host, owner_path, repo)``.

    Handles the three shapes a git remote URL takes in practice:
    ``https://host/owner/repo(.git)``, ``ssh://git@host/owner/repo(.git)``,
    and the scp-like ``git@host:owner/repo(.git)``. ``owner_path`` may
    itself contain ``"/"`` (a GitLab subgroup, e.g. ``"g/sub"``). This
    function does not judge whether ``host`` is one this arc supports --
    that check belongs to :func:`pr_compare_url`.

    Args:
        url: A remote's push URL, exactly as read from ``git remote -v``.

    Returns:
        ``(host, owner_path, repo)``, or ``None`` when ``url`` does not
        match any of the three known shapes (empty host or path, no
        owner/repo split, or an unrecognized scheme-less form).
    """
    url = url.strip()
    if not url:
        return None

    if "://" in url:
        parsed = urlparse(url)
        host = parsed.hostname
        path = (parsed.path or "").lstrip("/")
    else:
        match = _SCP_LIKE_REMOTE_RE.match(url)
        if match is None:
            return None
        host = match.group("host")
        path = match.group("path")

    if not host or not path:
        return None

    if path.endswith(".git"):
        path = path[: -len(".git")]
    path = path.strip("/")
    if "/" not in path:
        return None

    owner_path, _, repo = path.rpartition("/")
    if not owner_path or not repo:
        return None
    return host, owner_path, repo


def _codeberg_default_branch(root: Path, remote_name: str) -> str | None:
    """Resolve a Gitea-family remote's default branch via its local HEAD symref.

    Args:
        root: Workspace root.
        remote_name: The KNOWN remote name. The prefix is stripped from
            ``refs/remotes/<remote_name>/HEAD``'s resolved value by THIS
            name's length, never by splitting the value on ``"/"`` --
            remote names can themselves contain ``"/"`` (spec §2 probe 6
            applied to this lookup).

    Returns:
        The default branch name, or ``None`` when
        ``refs/remotes/<remote_name>/HEAD`` does not resolve locally (the
        remote was never fetched, or carries no such symref).
    """
    result = _run_user_git(
        root,
        "symbolic-ref",
        "--short",
        "-q",
        f"refs/remotes/{remote_name}/HEAD",
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    prefix = f"{remote_name}/"
    if not value.startswith(prefix):
        return None
    return value[len(prefix):]


def pr_compare_url(root: Path, info: GitWorkspaceInfo) -> str | GitWorkspaceRefusal:
    """Build a browser compare/merge-request URL for the current branch.

    Every interpolated component -- the owner path, the repository name,
    the branch, and codeberg's base branch -- is percent-encoded by the
    same rule before it reaches the URL; nothing repository-supplied is
    spliced in raw (spec §6).

    Args:
        root: Workspace root -- used only for the Gitea-family (codeberg)
            local default-branch lookup.
        info: The root's detected :class:`GitWorkspaceInfo`.

    Returns:
        The compare URL. A :class:`GitWorkspaceRefusal` when there is no
        upstream yet, the upstream remote's push URL is missing or
        unparseable, the host is unsupported, or (codeberg only) the
        default branch can't be determined locally.
    """
    if info.upstream is None:
        return GitWorkspaceRefusal("push the branch first")

    push_url: str | None = None
    for name, url in info.remotes:
        if name == info.upstream_remote:
            push_url = url
            break
    if push_url is None:
        return GitWorkspaceRefusal(_UNSUPPORTED_HOST_REASON)

    parsed = _parse_remote_url(push_url)
    if parsed is None:
        return GitWorkspaceRefusal(_UNSUPPORTED_HOST_REASON)
    host, owner_path, repo = parsed

    branch = info.branch
    if branch is None:
        return GitWorkspaceRefusal("no branch checked out")

    # Every interpolated component is encoded by the SAME rule, not just
    # the branch: `owner_path`, `repo` and codeberg's `base` all come from
    # the repository (a remote URL, a local symref) and were previously
    # spliced in raw while `branch` beside them was encoded. `safe="/"` for
    # path segments -- `owner_path` legitimately contains "/" (a GitLab
    # subgroup) -- and `safe=""` for anything landing in a query string.
    owner_seg = quote(owner_path, safe="/")
    repo_seg = quote(repo, safe="/")

    if host == "github.com":
        encoded = quote(branch, safe="/")
        return f"https://github.com/{owner_seg}/{repo_seg}/compare/{encoded}?expand=1"
    if host == "gitlab.com":
        encoded = quote(branch, safe="")
        return (
            f"https://gitlab.com/{owner_seg}/{repo_seg}/-/merge_requests/new"
            f"?merge_request%5Bsource_branch%5D={encoded}"
        )
    if host == "bitbucket.org":
        encoded = quote(branch, safe="")
        return (
            f"https://bitbucket.org/{owner_seg}/{repo_seg}/pull-requests/new"
            f"?source={encoded}"
        )
    if host == "codeberg.org":
        base = _codeberg_default_branch(root, info.upstream_remote)
        if base is None:
            return GitWorkspaceRefusal(
                "can't determine the default branch — open the PR on codeberg.org"
            )
        encoded = quote(branch, safe="/")
        base_seg = quote(base, safe="/")
        return (
            f"https://codeberg.org/{owner_seg}/{repo_seg}/compare/"
            f"{base_seg}...{encoded}"
        )

    return GitWorkspaceRefusal(_UNSUPPORTED_HOST_REASON)
