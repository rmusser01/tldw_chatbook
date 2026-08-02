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
from typing import Iterable, Sequence

from loguru import logger

#: Patterns for the shadow repo's ``info/exclude`` — noise no review should
#: carry. The user's own ``.gitignore`` files are additionally honored by git
#: itself (with the tool-touched force-add carve-out applied in TASK-1971).
FORCED_EXCLUDES: tuple[str, ...] = (
    ".git/",
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
        status: ``"A"``/``"M"``/``"D"``/``"R"`` (added/modified/deleted/renamed).
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
        """Args:
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

    # -- plumbing ----------------------------------------------------------

    def _env(self) -> dict[str, str]:
        """Environment for git calls: ``GIT_*`` scrubbed, prompts disabled."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
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
        """Current snapshot tip sha, or ``None`` before the first snapshot."""
        proc = self._run("rev-parse", "--verify", "HEAD", check=False)
        if proc.returncode != 0:
            return None
        return str(proc.stdout).strip()

    def snapshot(self, message: str) -> str:
        """Stage everything and commit if anything changed; return the tip.

        A clean tree returns the existing tip without a new commit. The very
        first snapshot commits even an empty tree (``--allow-empty``) so a
        baseline tip always exists.
        """
        with self._locked():
            self.ensure_initialized()
            self._run("add", "-A", "--", ".")
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
        """Unified diff for one file between two snapshots."""
        proc = self._run("diff", "-M", base, end, "--", path)
        return str(proc.stdout)

    def file_bytes(self, commit: str, path: str) -> bytes | None:
        """A file's content at a snapshot, or ``None`` if absent there."""
        proc = self._run("show", f"{commit}:{path}", check=False, binary=True)
        if proc.returncode != 0:
            return None
        return bytes(proc.stdout or b"")

    # -- low-level restore (full revert semantics live in TASK-1974) -------

    def restore_paths(self, commit: str, paths: Sequence[str]) -> None:
        """Restore ``paths`` in the work tree to their state at ``commit``.

        Low-level primitive: every path must EXIST at ``commit`` (un-create —
        deleting a path absent from the snapshot — is TASK-1974's guarded
        delete, deliberately not hidden inside this call).
        """
        if not paths:
            return
        with self._locked():
            self._run("checkout", commit, "--", *paths)
