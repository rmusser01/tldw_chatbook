"""Shadow-git change tracking for workspace roots (Agent Change Review).

TASK-1970, spec `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.

One shadow repo per CANONICAL root path (symlinks resolved), `GIT_DIR` under
the app data dir, `core.worktree` pointing at the root — nothing named
``.git`` is ever created inside the user's tree, and the user's own git
state (repo, index, HEAD, stashes) is never touched.

Hardening that is not optional (each item is a real first-turn failure on a
real dev machine):

* local ``user.name``/``user.email`` — ``git commit`` fails outright without
  an identity;
* ``commit.gpgsign=false`` — a global ``gpgsign=true`` would try to sign (or
  prompt on) every snapshot;
* ``core.hooksPath`` pinned to an empty directory — global husky-style hooks
  must not fire on snapshots (``--no-verify`` is belt-and-braces on top);
* ``gc.auto=0`` — GC is scheduled by retention (TASK-1975), never mid-turn;
* every invocation passes explicit ``--git-dir``/``--work-tree``, scrubs
  ``GIT_*`` from the environment, and sets ``GIT_TERMINAL_PROMPT=0``.

All porcelain/diff parsing is ``-z`` NUL-delimited: paths containing spaces,
newlines, or arbitrary UTF-8 are data, and ``restore_paths`` executes file
operations from parsed paths.

Locking is a per-repo in-process lock plus a portable atomic-``mkdir``
lockdir (``flock`` does not exist on Windows and CI runs Windows lanes),
with stale-lock takeover so a crashed process cannot starve snapshots.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from loguru import logger

#: Patterns for the shadow repo's ``info/exclude`` — noise no review should
#: carry. The user's own ``.gitignore`` files are additionally honored by git
#: itself (with the tool-touched force-add carve-out applied in TASK-1971).
FORCED_EXCLUDES: tuple[str, ...] = (
    # Both spellings: `.git/` (directory) and `.git` (the FILE a linked git
    # worktree carries). Git's own path special-casing already refuses to
    # track a top-level `.git` of either kind -- verified empirically -- so
    # these are belt-and-braces pinning the guarantee against future
    # exclude edits, not a live bug fix.
    ".git/",
    ".git",
    "node_modules/",
    ".venv/",
    "venv/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".tox/",
    "dist/",
    "build/",
)

def _exclude_pattern(rel_path: str) -> str:
    """Turn a root-relative path into an anchored, literal exclude pattern.

    Git exclude patterns treat ``* ? [ ] \\`` as globs and a trailing
    space as trimmable — paths are data, so each is escaped and the
    pattern anchored with a leading ``/``.

    Args:
        rel_path: POSIX-style path relative to the root.

    Returns:
        One ``info/exclude`` line matching exactly that path.
    """
    escaped = (
        rel_path.replace("\\", "\\\\")
        .replace("*", "\\*")
        .replace("?", "\\?")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )
    if escaped.endswith(" "):
        escaped = escaped[:-1] + "\\ "
    return "/" + escaped


#: A lockdir older than this is treated as abandoned by a crashed process and
#: taken over. Snapshot operations are seconds, not minutes.
_STALE_LOCK_SECONDS = 300.0

_LOCK_TIMEOUT_SECONDS = 60.0
_LOCK_RETRY_SECONDS = 0.05

_GIT_TIMEOUT_SECONDS = 120.0


class ChangeTrackingError(Exception):
    """A shadow-repo operation failed. Callers treat this as degradation,
    never as a reason to block an agent run (spec §2 failure posture)."""


class ChangeTrackingUnavailableError(ChangeTrackingError):
    """No usable ``git`` binary — the feature is absent, honestly."""


@dataclass(frozen=True)
class ChangedFile:
    """One file's change between two snapshots.

    Attributes:
        path: Path relative to the root (the NEW path for renames).
        status: ``"A"``/``"M"``/``"D"``/``"R"`` (added/modified/deleted/
            renamed) for the overwhelmingly common cases; rarer git letters
            (``"T"`` typechange, ``"C"`` copy) pass through VERBATIM rather
            than being coerced into a lie -- consumers group unknown letters
            into an "other" bucket.
        adds: Added-line count (0 for binary).
        dels: Deleted-line count (0 for binary).
        old_path: The pre-rename path, or ``None``.
        binary: True when git's numstat reports the change as binary.
    """

    path: str
    status: str
    adds: int = 0
    dels: int = 0
    old_path: str | None = None
    binary: bool = False


#: In-process per-repo locks, keyed by git-dir string. Module-level so every
#: `ShadowRepo` instance for one root shares the same lock.
_REPO_LOCKS: dict[str, threading.Lock] = {}
_REPO_LOCKS_GUARD = threading.Lock()


def _in_process_lock(key: str) -> threading.Lock:
    with _REPO_LOCKS_GUARD:
        return _REPO_LOCKS.setdefault(key, threading.Lock())


class ShadowRepoService:
    """Factory/owner of per-root shadow repos.

    Constructing the service and probing :attr:`available` never raise —
    using an unavailable service raises :class:`ChangeTrackingUnavailableError`
    (a silent no-op snapshot would be this programme's canonical false-pass
    bug, so unavailability is loud at the call site and soft at the probe).
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        git_executable: str | None = None,
    ) -> None:
        """Create the service.

        Args:
            data_dir: Base directory for shadow repos; defaults to the app
                data dir's ``change_review/`` subtree.
            git_executable: Override for tests; defaults to ``git`` on PATH.
        """
        if data_dir is None:
            from tldw_chatbook.Utils.paths import get_user_data_dir

            data_dir = get_user_data_dir() / "change_review"
        self._data_dir = Path(data_dir)
        self._git = (
            git_executable
            if git_executable is not None
            else shutil.which("git")
        )

    @property
    def available(self) -> bool:
        """Whether a git binary is usable. Never raises."""
        if not self._git:
            return False
        return Path(self._git).exists() or shutil.which(self._git) is not None

    def repo_for_root(self, root: Path | str) -> "ShadowRepo":
        """Return the shadow repo for ``root``'s canonical path.

        Args:
            root: A workspace folder root; symlinks are resolved so every
                spelling of one directory shares one shadow repo.

        Raises:
            ChangeTrackingUnavailableError: No usable git binary.
        """
        if not self.available:
            raise ChangeTrackingUnavailableError(
                "change tracking needs git — install git to enable"
            )
        canonical = Path(root).expanduser().resolve()
        if not canonical.is_dir():
            # Roots come from the workspace registry, not from agent input,
            # so this is a sanity guard rather than a security boundary --
            # but a vanished/mistyped root must fail HERE with a clear
            # message, not five calls later inside a git subprocess.
            raise ChangeTrackingError(
                f"change tracking root is not a directory: {canonical}"
            )
        key = hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()[:16]
        repo_home = self._data_dir / key
        assert self._git is not None  # narrowed by `available`
        return ShadowRepo(
            root=canonical,
            git_dir=repo_home / "git",
            hooks_dir=repo_home / "hooks_empty",
            lock_dir=repo_home / "lock.d",
            git_executable=self._git,
        )


class ShadowRepo:
    """One root's shadow repo. All operations run under the per-repo lock."""

    def __init__(
        self,
        root: Path,
        git_dir: Path,
        hooks_dir: Path,
        lock_dir: Path,
        git_executable: str,
    ) -> None:
        self.root = root
        self.git_dir = git_dir
        self.hooks_dir = hooks_dir
        self.lock_dir = lock_dir
        self._git = git_executable
        #: Root-relative paths excluded as oversized by the LAST snapshot
        #: on this instance (TASK-1975 disclosure).
        self.last_oversize_excluded: tuple[str, ...] = ()
        #: Nested repos seen by the LAST snapshot's scan (TASK-1976).
        self.last_nested_repos: tuple[str, ...] = ()

    # -- plumbing ----------------------------------------------------------

    def _env(self) -> dict[str, str]:
        """Environment for git calls: ``GIT_*`` scrubbed, prompts disabled."""
        # Case-INSENSITIVE match: Windows environment variables are
        # case-insensitive, so `Git_Index_File` reaches git just as
        # GIT_INDEX_FILE does. Harmless over-scrub on POSIX.
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.upper().startswith("GIT_")
        }
        env["GIT_TERMINAL_PROMPT"] = "0"
        return env

    def _run(
        self,
        *args: str,
        check: bool = True,
        binary: bool = False,
    ) -> subprocess.CompletedProcess:
        cmd = [
            self._git,
            "--git-dir",
            str(self.git_dir),
            "--work-tree",
            str(self.root),
            *args,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                env=self._env(),
                timeout=_GIT_TIMEOUT_SECONDS,
                text=not binary,
            )
        except subprocess.TimeoutExpired as exc:
            raise ChangeTrackingError(
                f"git {args[0]} timed out after {_GIT_TIMEOUT_SECONDS:.0f}s"
            ) from exc
        except OSError as exc:
            raise ChangeTrackingUnavailableError(str(exc)) from exc
        if check and proc.returncode != 0:
            stderr = proc.stderr if isinstance(proc.stderr, str) else (
                proc.stderr.decode("utf-8", "replace") if proc.stderr else ""
            )
            raise ChangeTrackingError(
                f"git {args[0]} failed ({proc.returncode}): {stderr.strip()[:400]}"
            )
        return proc

    def _locked(self):
        """Context manager: in-process lock + portable cross-process lockdir."""
        repo = self

        class _Lock:
            def __enter__(self) -> None:
                self._thread_lock = _in_process_lock(str(repo.git_dir))
                if not self._thread_lock.acquire(timeout=_LOCK_TIMEOUT_SECONDS):
                    raise ChangeTrackingError(
                        "change-tracking lock timed out (in-process)"
                    )
                deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
                repo.lock_dir.parent.mkdir(parents=True, exist_ok=True)
                while True:
                    try:
                        repo.lock_dir.mkdir()
                        return
                    except FileExistsError:
                        try:
                            age = time.time() - repo.lock_dir.stat().st_mtime
                        except OSError:
                            continue  # holder released between mkdir and stat
                        if age > _STALE_LOCK_SECONDS:
                            # A crashed process must not starve snapshots
                            # forever. Takeover is logged, not silent.
                            logger.warning(
                                "change_tracking: taking over stale lock "
                                f"({age:.0f}s old) at {repo.lock_dir}"
                            )
                            try:
                                repo.lock_dir.rmdir()
                            except OSError:
                                pass
                            continue
                        if time.monotonic() > deadline:
                            self._thread_lock.release()
                            raise ChangeTrackingError(
                                "change-tracking lock timed out (cross-process)"
                            )
                        time.sleep(_LOCK_RETRY_SECONDS)

            def __exit__(self, *exc_info) -> None:
                try:
                    repo.lock_dir.rmdir()
                except OSError:
                    pass
                self._thread_lock.release()

        return _Lock()

    # -- lifecycle ---------------------------------------------------------

    def ensure_initialized(self) -> None:
        """Create + pin the shadow repo. Idempotent, self-healing (config and
        excludes rewritten every call — they are cheap, and drift heals)."""
        if not (self.git_dir / "HEAD").exists():
            self.git_dir.parent.mkdir(parents=True, exist_ok=True)
            self._run("init", "--quiet")
        self.hooks_dir.mkdir(parents=True, exist_ok=True)
        pins = (
            ("user.name", "tldw-chatbook change review"),
            ("user.email", "change-review@tldw-chatbook.invalid"),
            ("commit.gpgsign", "false"),
            ("core.hooksPath", str(self.hooks_dir)),
            ("gc.auto", "0"),
            ("core.untrackedCache", "true"),
            ("core.worktree", str(self.root)),
        )
        for key, value in pins:
            self._run("config", key, value)
        exclude = self.git_dir / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text(
            "# managed by tldw-chatbook change review (TASK-1970)\n"
            + "\n".join(FORCED_EXCLUDES)
            + "\n"
        )

    # -- snapshots ---------------------------------------------------------

    def tip(self) -> str | None:
        """Return the current snapshot tip.

        Returns:
            The tip commit sha, or ``None`` before the first snapshot.
        """
        proc = self._run("rev-parse", "--verify", "HEAD", check=False)
        if proc.returncode != 0:
            return None
        return str(proc.stdout).strip()

    def has_snapshot(self, sha: str) -> bool:
        """Whether a snapshot commit still exists in this shadow repo.

        Retention (TASK-1975) can reset a repo whose rows were pruned; a
        surviving history row must then render "pruned by retention"
        rather than erroring, and this probe is how callers tell.

        Args:
            sha: A snapshot commit sha.

        Returns:
            True when the commit object is present.
        """
        if not sha:
            return False
        proc = self._run("cat-file", "-e", f"{sha}^{{commit}}", check=False)
        return proc.returncode == 0

    def _exact_force_paths(self, paths: Sequence[str]) -> list[str]:
        """Return safe existing file paths relative to the root."""
        exact_paths: list[str] = []
        for raw in paths:
            if not raw or raw == ".":
                continue
            path = Path(raw)
            if path.is_absolute():
                continue
            try:
                resolved = (self.root / path).resolve()
                relative = resolved.relative_to(self.root)
            except (OSError, RuntimeError, ValueError):
                continue
            if relative == Path(".") or not resolved.exists():
                continue
            if resolved.is_dir():
                continue
            ancestor = resolved.parent
            while ancestor != self.root:
                marker = ancestor / ".git"
                if marker.is_file() or marker.is_dir():
                    break
                ancestor = ancestor.parent
            if ancestor != self.root:
                continue
            exact_paths.append(relative.as_posix())
        return exact_paths

    def _drop_new_force_paths_over_cap(
        self, paths: Sequence[str]
    ) -> tuple[str, ...]:
        """Remove and return newly indexed force paths over the size cap."""
        from tldw_chatbook.Workspaces.change_bounds import (
            DEFAULT_MAX_FILE_BYTES,
            change_review_setting,
        )

        cap = change_review_setting("max_file_bytes", DEFAULT_MAX_FILE_BYTES)
        tip = self.tip()
        removed: list[str] = []
        for rel in paths:
            literal = f":(literal){rel}"
            if tip and self._z_tokens(
                "ls-tree", "-z", "--name-only", tip, "--", literal
            ):
                continue
            staged = self._z_tokens(
                "ls-files", "--stage", "-z", "--", literal
            )
            if not staged:
                continue
            fields = staged[0].split(" ", 2)
            if len(fields) < 2:
                raise ChangeTrackingError("git ls-files returned malformed output")
            size = int(str(self._run("cat-file", "-s", fields[1]).stdout))
            if size > cap:
                self._run("update-index", "--force-remove", "--", rel)
                removed.append(rel)
        return tuple(removed)

    def _drop_new_force_paths_no_longer_safe(
        self, staged_paths: Sequence[str], safe_paths: Sequence[str]
    ) -> tuple[str, ...]:
        """Remove newly indexed paths absent from the final safe set."""
        safe = set(safe_paths)
        tip = self.tip()
        removed: list[str] = []
        for rel in staged_paths:
            if rel in safe:
                continue
            literal = f":(literal){rel}"
            if tip and self._z_tokens(
                "ls-tree", "-z", "--name-only", tip, "--", literal
            ):
                continue
            if not self._z_tokens("ls-files", "--stage", "-z", "--", literal):
                continue
            self._run("update-index", "--force-remove", "--", rel)
            removed.append(rel)
        return tuple(removed)

    def snapshot(self, message: str, *, force_paths: Sequence[str] = ()) -> str:
        """Stage everything and commit if anything changed; return the tip.

        A clean tree returns the existing tip without a new commit. The very
        first snapshot commits even an empty tree (``--allow-empty``) so a
        baseline tip always exists.

        TASK-1975: git cannot exclude by size, so every snapshot re-scans
        the root and appends files over ``max_file_bytes`` to
        ``info/exclude`` (``ensure_initialized`` rewrites the static block
        first, so entries never accumulate stale). The excluded set lands on
        :attr:`last_oversize_excluded` for disclosure. Limit: excludes only
        stop UNTRACKED files — a file committed while small that later grew
        stays tracked, which shows in diffs rather than lying by omission.

        Args:
            message: Commit message recorded on the snapshot (turn labels).
            force_paths: Existing root-relative paths to stage despite ignore
                rules before the ordinary snapshot add.

        Returns:
            The tip sha after the snapshot.

        Raises:
            ChangeTrackingError: A git step failed or produced no tip.
        """
        import sys as _sys

        from tldw_chatbook.Workspaces.change_bounds import scan_root

        with self._locked():
            self.ensure_initialized()
            exact_paths = self._exact_force_paths(force_paths)
            if exact_paths:
                self._run("update-index", "--add", "--", *exact_paths)
                self._drop_new_force_paths_over_cap(exact_paths)
            scan = scan_root(
                self.root,
                max_files=_sys.maxsize,
                max_total_bytes=_sys.maxsize,
            )
            self.last_oversize_excluded = scan.oversized
            self.last_nested_repos = scan.nested_repos
            # info/exclude is line-oriented and has NO newline escaping --
            # a filename carrying \n would INJECT extra patterns (Qodo
            # #1251 finding 5). Such paths are unexcludable; they are
            # unstaged after add instead (argv is newline-safe).
            excludable = [
                rel
                for rel in scan.oversized
                if "\n" not in rel and "\r" not in rel
            ]
            unexcludable = [
                rel for rel in scan.oversized if rel not in excludable
            ]
            # TASK-1976: nested repos are excluded from tracking entirely.
            # This is not merely hygiene -- `git add -A` HARD-FAILS (128,
            # "does not have a commit checked out") on a commitless child
            # repo, which would kill tracking for the whole root; and a
            # committed child would land as a gitlink whose inner changes
            # are invisible anyway. Excluded + disclosed is the honest,
            # uniform behavior (test: nested-edit-invisible).
            nested_excludable = [
                rel
                for rel in scan.nested_repos
                if "\n" not in rel and "\r" not in rel
            ]
            # Qodo #1254 finding 5: a newline-named nested repo cannot go
            # into info/exclude, and a commitless child makes `add -A`
            # FATAL -- exclude it at add time via pathspec magic instead
            # (argv is newline-safe; `literal` disables glob semantics).
            nested_unexcludable = [
                rel for rel in scan.nested_repos if rel not in nested_excludable
            ]
            if excludable or nested_excludable:
                exclude = self.git_dir / "info" / "exclude"
                with exclude.open("a", encoding="utf-8") as fh:
                    fh.write(
                        "# oversize + nested (TASK-1975/1976), "
                        "rewritten per snapshot\n"
                    )
                    for rel in excludable:
                        fh.write(_exclude_pattern(rel) + "\n")
                    for rel in nested_excludable:
                        fh.write(_exclude_pattern(rel) + "/\n")
            add_args = ["add", "-A", "--", "."]
            add_args.extend(
                f":(literal,exclude){rel}" for rel in nested_unexcludable
            )
            self._run(*add_args)
            for rel in unexcludable:
                self._run(
                    "rm", "--cached", "--ignore-unmatch", "--quiet", "--", rel,
                    check=False,
                )
            if force_paths:
                final_paths = self._exact_force_paths(force_paths)
                self._drop_new_force_paths_no_longer_safe(
                    exact_paths, final_paths
                )
                if final_paths:
                    tip = self.tip()
                    new_final_paths = {
                        rel
                        for rel in final_paths
                        if not tip
                        or not self._z_tokens(
                            "ls-tree",
                            "-z",
                            "--name-only",
                            tip,
                            "--",
                            f":(literal){rel}",
                        )
                    }
                    self._run("update-index", "--add", "--", *final_paths)
                    late_oversize = self._drop_new_force_paths_over_cap(final_paths)
                    included = new_final_paths.difference(late_oversize)
                    self.last_oversize_excluded = tuple(
                        rel
                        for rel in dict.fromkeys(
                            (*self.last_oversize_excluded, *late_oversize)
                        )
                        if rel not in included
                    )
            had_tip = self.tip() is not None
            if had_tip:
                staged = self._run("diff", "--cached", "--quiet", check=False)
                if staged.returncode == 0:
                    return self.tip()  # type: ignore[return-value]
                if staged.returncode not in (0, 1):
                    raise ChangeTrackingError(
                        "git diff --cached failed while checking cleanliness"
                    )
            commit_args = ["commit", "--quiet", "--no-verify", "-m", message]
            if not had_tip:
                commit_args.append("--allow-empty")
            self._run(*commit_args)
            new_tip = self.tip()
            if not new_tip:
                raise ChangeTrackingError("snapshot commit produced no tip")
            return new_tip

    # -- reading changes ---------------------------------------------------

    def changed_files(self, base: str, end: str) -> list[ChangedFile]:
        """The files changed between two snapshots, rename-aware.

        Merges ``--name-status -z`` (status/rename pairs) with
        ``--numstat -z`` (line counts, binary detection). Both streams are
        NUL-delimited; paths are data.

        Args:
            base: Older snapshot sha (exclusive).
            end: Newer snapshot sha (inclusive).

        Returns:
            One :class:`ChangedFile` per changed path, in git's diff order.
        """
        status_by_path: dict[str, tuple[str, str | None]] = {}
        ordered: list[str] = []
        tokens = self._z_tokens("diff", "-M", "--name-status", "-z", base, end)
        i = 0
        while i < len(tokens):
            raw_status = tokens[i]
            code = raw_status[:1]
            if code == "R":
                old, new = tokens[i + 1], tokens[i + 2]
                status_by_path[new] = ("R", old)
                ordered.append(new)
                i += 3
            else:
                path = tokens[i + 1]
                status_by_path[path] = (code, None)
                ordered.append(path)
                i += 2

        counts: dict[str, tuple[int, int, bool]] = {}
        tokens = self._z_tokens("diff", "-M", "--numstat", "-z", base, end)
        i = 0
        while i < len(tokens):
            head = tokens[i]
            adds_s, dels_s, rest = head.split("\t", 2)
            if rest == "":
                # Rename record: counts \t\0 old \0 new \0
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

        out: list[ChangedFile] = []
        for path in ordered:
            code, old = status_by_path[path]
            adds, dels, binary = counts.get(path, (0, 0, False))
            out.append(
                ChangedFile(
                    path=path,
                    status=code,
                    adds=adds,
                    dels=dels,
                    old_path=old,
                    binary=binary,
                )
            )
        return out

    def _z_tokens(self, *args: str) -> list[str]:
        proc = self._run(*args, binary=True)
        raw: bytes = proc.stdout or b""
        return [t.decode("utf-8", "surrogateescape") for t in raw.split(b"\0") if t]

    def diff_text(self, base: str, end: str, path: str) -> str:
        """Return the unified diff for one file between two snapshots.

        Args:
            base: Older snapshot sha.
            end: Newer snapshot sha.
            path: Root-relative path to diff.

        Returns:
            Unified diff text (rename-aware), empty for no change.
        """
        proc = self._run("diff", "-M", base, end, "--", path)
        return str(proc.stdout)

    def file_bytes(self, commit: str, path: str) -> bytes | None:
        """Return a file's content at a snapshot.

        Args:
            commit: Snapshot sha to read from.
            path: Root-relative path.

        Returns:
            Raw bytes, or ``None`` when the path does not exist there.
        """
        proc = self._run("show", f"{commit}:{path}", check=False, binary=True)
        if proc.returncode != 0:
            return None
        return bytes(proc.stdout or b"")

    def force_add(self, paths: Sequence[str]) -> None:
        """Stage ``paths`` even when ignore rules would exclude them.

        TASK-1971's ``.gitignore`` carve-out: a WRITE tool's edit to an
        ignored file (``.env``) must surface in the turn's diff. Missing
        paths are skipped (the tool may have deleted its own file) rather
        than failing the snapshot.

        Args:
            paths: Root-relative paths to stage exactly despite ignore rules.
        """
        if not paths:
            return
        with self._locked():
            self.ensure_initialized()
            exact_paths = self._exact_force_paths(paths)
            if exact_paths:
                self._run("update-index", "--add", "--", *exact_paths)
                final_paths = self._exact_force_paths(exact_paths)
                unsafe = self._drop_new_force_paths_no_longer_safe(
                    exact_paths, final_paths
                )
                oversized = self._drop_new_force_paths_over_cap(final_paths)
                if unsafe:
                    paths_text = ", ".join(unsafe)
                    raise ChangeTrackingError(
                        (
                            "forced path is no longer safe at staging boundary: "
                            f"{paths_text}"
                        )[:400]
                    )
                if oversized:
                    paths_text = ", ".join(oversized)
                    raise ChangeTrackingError(
                        (
                            "forced path exceeds change-tracking size cap: "
                            f"{paths_text}"
                        )[:400]
                    )

    # -- low-level restore (full revert semantics live in TASK-1974) -------

    def restore_paths(self, commit: str, paths: Sequence[str]) -> None:
        """Restore ``paths`` in the work tree to their state at ``commit``.

        Low-level primitive: every path must EXIST at ``commit`` (un-create —
        deleting a path absent from the snapshot — is TASK-1974's guarded
        delete, deliberately not hidden inside this call).

        Args:
            commit: Snapshot sha to restore from.
            paths: Root-relative paths, all present at ``commit``.

        Raises:
            ChangeTrackingError: The checkout failed (including any path
                absent from ``commit``).
        """
        if not paths:
            return
        with self._locked():
            self._run("checkout", commit, "--", *paths)
