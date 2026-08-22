"""Sync ``run_git`` wrapper + repository preparation for the read-only git tools.

Ported from tldw_server's
tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py
@ 5605b9d9906322c2e6b5342b48c391ae674d315e
(https://github.com/rmusser01/tldw_server, GPL-3.0-only).

``run_git`` (argv validation, sanitized environment, bounded output reads)
is a near-verbatim sync port of ``AsyncGitCommandRunner`` (reference
:120-297). The async ``_communicate_bounded``/``_read_bounded_stream`` pair
becomes a Popen + reader-thread bounded read loop so output is killed AT the
cap rather than fully buffered — ``subprocess.run(capture_output=True)``
would let a runaway ``git log`` buffer unbounded memory before the 1 MB cap
applies, so capture-then-truncate is deliberately NOT used.

``prepare_repository`` adapts the reference's ``_prepare_repository``
(:1134+) to the chatbook shape: confinement via ``resolve_workspace_path``,
sync ``run_git``, and ``LocalToolError`` (shared model-actionable error)
instead of the reference's ``_GitToolError``. Deviations from the reference
(deliberate, per the phase-3b-ii plan):

1. A timeout raises ``LocalToolError("git command timed out ...")`` instead
   of returning a result with ``timed_out=True`` (the field is kept on
   ``GitCommandResult`` for shape fidelity with the reference).
2. ``GitCommandResult.duration_ms`` is dropped; truncation is surfaced with
   a human-readable marker appended to the affected stream.
3. The repo root must be the workspace root or INSIDE it; a repo root above
   the workspace root is refused (the reference's ``_path_inside`` rule,
   :2062-2063), so the model cannot read repo state outside confinement.

The tool cores (``git_status``/``git_branches``/``git_log``/``git_diff``/
``git_blame``) are sync adaptations of the reference's ``_execute_status``
(:570), ``_execute_branches`` (:621), ``_execute_log`` (:831),
``_execute_diff`` (:709) + ``_run_diff_command`` (:1049-1080 — the
``--no-ext-diff``/``--no-textconv``/``--no-color`` machine-safe flags are
ported), and ``_execute_blame`` (:892), returning plain text for the agent
provider instead of the reference's structured dicts. Disclosed deviations
from the reference (deliberate, per the phase-3b-ii plan):

(a) ``git_diff`` adds ``commit_range`` and ``stat`` modes the reference
    does not have; the reference's third ``working_tree`` scope is omitted
    (``staged=False`` maps to ``unstaged``, ``staged=True`` to ``staged``).
    ``commit_range`` is regex-validated (``^[A-Za-z0-9._/~^-]+$``) before
    entering argv to keep the fixed-argv guarantee meaningful.
(b) ``git_log`` defaults ``count=20`` (the reference has no default — an
    absent limit falls back to its max-100); both clamp to 1..100.
(c) ``_parse_blame_header`` accepts 3-field headers (``sha orig final``),
    not just 4-field group headers: git emits 3-field headers for the
    remaining lines of a commit group, and the reference's 4-field minimum
    silently drops those lines.
(d) ``git_blame``'s ``-L`` range is optional (the reference always passes
    one); the range is capped at ``GIT_BLAME_MAX_LINES`` lines.

TASK-19632 adds one thing the reference has no equivalent of: these tools
enumerate a repository on the model's behalf, so ``path`` being optional
meant no candidate ever reached the sensitive-path denylist and
``git_diff`` returned ``~/.ssh/id_rsa``'s CONTENT from a CLEAN worktree.
The fix constrains git's INPUT rather than filtering its OUTPUT -- see
``_denylist_pathspecs`` for the mechanism and ``Utils/sensitive_paths.py``
for why that direction was chosen. Two properties of pathspecs are
load-bearing there and are easy to lose in a later edit: an exclude-only
pathspec list applies to the whole tree (no positive pathspec is needed),
and pathspec MAGIC is honoured after ``--``, so every pathspec built here
carries explicit magic -- ``:(literal)`` for the one that SCOPES output,
``:(exclude,literal,icase)``/``:(exclude,glob,icase)`` for the ones that
DENY from it. The ``icase`` asymmetry is deliberate; see
``_literal_pathspec`` and ``_denylist_pathspecs``.
"""

from __future__ import annotations

import contextlib
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from tldw_chatbook.Tools.local_tool_impls import LocalToolError, resolve_workspace_path
from tldw_chatbook.Utils.sensitive_paths import (
    SensitivePathContext,
    is_sensitive_path,
    resolve_sensitive_context,
    sensitive_exclusions_under,
)

GIT_TIMEOUT_SECONDS = 30.0
GIT_MAX_OUTPUT_BYTES = 1_000_000
REPOSITORY_DISCOVERY_TIMEOUT_SECONDS = 5.0
GIT_TRUNCATED_MARKER = "\n...[output truncated]"

_ALLOWED_GIT_SUBCOMMANDS = frozenset(
    {
        "--version",
        "blame",
        "branch",
        "diff",
        "log",
        "ls-files",
        "rev-parse",
        "status",
    }
)


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """Result returned by :func:`run_git` (sync port of the reference's)."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False


def _git_environment() -> dict[str, str]:
    env: dict[str, str] = {}
    path_value = os.environ.get("PATH")
    if path_value:
        env["PATH"] = path_value
    for key in ("SYSTEMROOT", "WINDIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    # Built from scratch, so ambient GIT_*_PATHSPECS never reaches git --
    # and nothing here may ever ADD one. TASK-16801's lesson recommends
    # `GIT_LITERAL_PATHSPECS=1` as blanket hardening for git argv; it is
    # incompatible with this module's denylist exclusions and fails
    # SILENTLY, not loudly: under it, `:(exclude,literal)<path>` is taken
    # as a literal FILENAME, matches nothing, and every `git_diff` /
    # `git_status` returns "(no changes)" / "(working tree clean)" with
    # exit 0 (verified on git 2.39). The per-pathspec `:(literal)` magic
    # in `_literal_pathspec` is this module's equivalent, and it composes
    # with exclusions instead of disabling them.
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_EXTERNAL_DIFF": "",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.fsmonitor",
            "GIT_CONFIG_VALUE_0": "false",
        }
    )
    return env


def _extract_subcommand_and_validate_globals(argv: list[str]) -> str | None:
    index = 1
    while index < len(argv):
        value = argv[index]
        if value == "--version":
            if len(argv) != 2:
                raise LocalToolError("git global option --version must be used alone")
            return value
        if value == "-C":
            if index + 1 >= len(argv):
                raise LocalToolError("git global option -C requires a workspace path")
            index += 2
            continue
        if value == "--no-pager":
            index += 1
            continue
        if value.startswith("-"):
            raise LocalToolError(f"git global option is not allowlisted: {value}")
        return value
    return None


def _validate_argv(argv: list[str]) -> None:
    if not argv or argv[0] != "git":
        raise LocalToolError("git runner only executes git commands")
    subcommand = _extract_subcommand_and_validate_globals(argv)
    if subcommand is None:
        raise LocalToolError("git requires a subcommand (e.g. status, diff, log)")
    if subcommand not in _ALLOWED_GIT_SUBCOMMANDS:
        raise LocalToolError(f"git subcommand is not allowlisted: {subcommand}")


def _read_bounded_stream(stream, max_output_bytes: int) -> tuple[bytes, bool]:
    """Read a stream until EOF or the byte cap; report cap hits."""
    output = bytearray()
    while len(output) <= max_output_bytes:
        read_size = min(8192, max_output_bytes + 1 - len(output))
        chunk = stream.read(read_size)
        if not chunk:
            return bytes(output), False
        output.extend(chunk)
        if len(output) > max_output_bytes:
            return bytes(output[:max_output_bytes]), True
    return bytes(output[:max_output_bytes]), True


def run_git(
    argv: list[str],
    *,
    timeout: float = GIT_TIMEOUT_SECONDS,
    max_output_bytes: int = GIT_MAX_OUTPUT_BYTES,
) -> GitCommandResult:
    """Run an allowlisted git command with bounded output and a timeout.

    Fixed argv only: ``argv[0]`` must be ``git``, the only permitted global
    options are ``-C <path>`` and ``--no-pager`` (``--version`` must stand
    alone), and the subcommand must be in ``_ALLOWED_GIT_SUBCOMMANDS``. The
    environment is sanitized (PATH + git safety vars only) and stdin is
    DEVNULL. Output per stream is capped at ``max_output_bytes`` — the
    process is killed when the cap is exceeded (never fully buffered) and a
    truncation marker is appended. Timeout kills the process and raises.

    NOTE: validation stops at the subcommand. Everything AFTER the
    subcommand is caller-constructed and NOT validated at this layer — e.g.
    ``["git", "diff", "--output=/tmp/pwn"]`` passes ``_validate_argv``.
    Callers (the tool cores below) must therefore build argv only from
    fixed literals plus values they have validated themselves (see the
    ``commit_range`` leading-dash/regex checks in :func:`git_diff`) and
    must never splice raw model-controlled strings into flag positions.

    Raises:
        LocalToolError: argv validation failure, git unavailable, or timeout.
    """
    _validate_argv(argv)
    if shutil.which("git") is None:
        raise LocalToolError("git is not available on this system")
    if not isinstance(max_output_bytes, int) or isinstance(max_output_bytes, bool) or max_output_bytes <= 0:
        max_output_bytes = GIT_MAX_OUTPUT_BYTES

    process = subprocess.Popen(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
        # Own process group so the timeout/cap kills can reap grandchildren
        # too (a bare process.kill() leaves e.g. a textconv child alive —
        # and a live grandchild holding the pipe write end would stall the
        # truncation fast-path until the full timeout).
        start_new_session=os.name == "posix",
    )

    results: dict[str, tuple[bytes, bool]] = {}
    done: queue.Queue[tuple[str, tuple[bytes, bool]]] = queue.Queue()

    def _reader(name: str, stream) -> None:  # noqa: ANN001 - pipe reader
        try:
            done.put((name, _read_bounded_stream(stream, max_output_bytes)))
        except Exception:  # defensive: a broken pipe must not hang the join loop
            done.put((name, (b"", False)))

    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        threading.Thread(target=_reader, args=(name, stream), daemon=True).start()

    deadline = time.monotonic() + float(timeout)
    while len(results) < 2:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_process(process)
            with contextlib.suppress(Exception):
                process.wait()
            raise LocalToolError(f"git command timed out after {timeout} seconds: {list(argv)}")
        try:
            name, data = done.get(timeout=min(remaining, 0.05))
        except queue.Empty:
            continue
        results[name] = data
        if data[1]:
            # Cap exceeded: kill now (the reference behaviour) instead of
            # letting the process keep producing output we will discard.
            _kill_process(process)

    with contextlib.suppress(Exception):
        process.wait(timeout=REPOSITORY_DISCOVERY_TIMEOUT_SECONDS)

    stdout_bytes, stdout_truncated = results.get("stdout", (b"", False))
    stderr_bytes, stderr_truncated = results.get("stderr", (b"", False))
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    if stdout_truncated:
        stdout += GIT_TRUNCATED_MARKER
    if stderr_truncated:
        stderr += GIT_TRUNCATED_MARKER
    return GitCommandResult(
        argv=list(argv),
        returncode=int(process.returncode if process.returncode is not None else -1),
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        truncated=stdout_truncated or stderr_truncated,
    )


def _kill_process(process: subprocess.Popen) -> None:
    """Kill the whole process group on POSIX; fall back to the direct child."""
    if os.name == "posix":
        # process was started with start_new_session=True, so its pid is
        # the group id: SIGKILL the group to reap grandchildren as well.
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(process.pid, signal.SIGKILL)
        return
    with contextlib.suppress(ProcessLookupError):
        process.kill()


#: Characters wildmatch (git's glob engine) treats as metacharacters. A
#: backslash escapes the next character, so escaping these renders a real
#: filename literally inside a ``:(exclude,glob)`` pathspec. Verified
#: against git 2.39: a container directory literally named ``co*ntainer``
#: excluded ``coXntainer/`` too until its ``*`` was escaped.
_GLOB_METACHARACTERS = "\\*?["


def _glob_escape(value: str) -> str:
    """Escape ``value`` so a ``:(...,glob)`` pathspec matches it literally."""
    return "".join(
        f"\\{char}" if char in _GLOB_METACHARACTERS else char for char in value
    )


def _literal_pathspec(relative_posix: str) -> str:
    """Render a repo-relative path as a pathspec that CANNOT carry magic.

    A pathspec is not a path. ``--`` ends git's OPTION parsing, not its
    pathspec-magic parsing, so a repository file legitimately named
    ``:(exclude)notes.txt`` -- resolved, confined and denylist-checked by
    the choke point exactly like any other filename -- inverted the scope
    of the diff it was spliced into and returned the rest of the
    repository, ``~/.ssh/id_rsa`` included (measured; TASK-19632).
    ``:(literal)`` disables magic AND wildcard interpretation for the
    remainder of the element, so the value is matched as the byte string
    it is.

    ``:(literal).`` behaves identically to a bare ``.`` (verified), so the
    repo-root case needs no special handling.

    Deliberately NOT given the ``icase`` magic the EXCLUSIONS carry. This
    pathspec SCOPES the output; theirs DENY from it, and the two fail in
    opposite directions -- exactly the asymmetry TASK-19800 records for
    ``_compare_key`` versus confinement. Folding case here would ADD files
    to what the model gets back (a ``README`` alongside the ``readme`` it
    asked for); folding it there only ever removes more.
    """
    return f":(literal){relative_posix}"


def _denylist_pathspecs(
    repo_root: Path, context: SensitivePathContext | None = None
) -> tuple[str, ...]:
    """Render the sensitive-path denylist as git exclude pathspecs.

    The TASK-19632 fix, in one place. ``git_status``/``git_diff`` hand a
    whole repository to git and return what comes back, so a path the
    model never named -- ``~/.ssh/id_rsa`` under a ``$HOME``-rooted
    workspace -- never reached ``Utils/sensitive_paths.py`` at all.

    Excluding by PATHSPEC rather than filtering git's output is the
    deliberate choice: git stays the authority on what matches, the
    exclusions are recomputed from the live denylist on every call (so a
    ``TLDW_CONFIG_PATH`` switch or a relocated database is observed
    immediately), and no unified-diff or porcelain text is ever parsed to
    decide what to withhold -- a half-parsed diff is worse than none.

    Each :class:`~tldw_chatbook.Utils.sensitive_paths.SensitiveExclusion`
    kind maps to the pathspec form that expresses exactly that rule and
    no more, which is what keeps a legitimate diff intact:

    * ``subtree``/``file`` -> ``:(exclude,literal,icase)<rel>``. Literal
      magic, so a real filename containing ``*`` excludes only itself.
    * ``direct_children`` -> ``:(exclude,glob,icase)<rel>/*``. Under
      ``glob`` magic ``*`` does not cross ``/``, so this refuses the
      container's direct child FILES and leaves its subdirectories fully
      visible -- the same distinction ``is_sensitive_path``'s own
      container rule draws (``tool_sandbox/`` stays diffable; a loose file
      beside it does not).
    * ``name`` -> ``:(exclude,glob,icase)**/<name>``, which matches at
      every depth INCLUDING the repository root (verified).

    Every one of them carries ``icase``, for the reason TASK-19800 gives
    for folding the denylist itself: macOS and Windows filesystems are
    case-insensitive by default, git records whatever spelling a path was
    added under, and a denial that misses ``.SSH/id_rsa`` because the
    denylist says ``.ssh`` is a leak. Folding an EXCLUSION only ever
    removes more from the output, so it fails in the cheap direction --
    unlike folding a scoping pathspec (see ``_literal_pathspec``) or a
    confinement check.

    An unrecognized kind raises rather than being skipped: a denial added
    to the denylist that this renderer does not understand must fail the
    call, not silently pass through.

    Args:
        repo_root: The already-resolved repository root; every pathspec
            is rendered relative to it, which is what ``-C <repo_root>``
            makes correct (git resolves pathspecs against the process's
            working directory).
        context: Optional pre-resolved ``SensitivePathContext``, so one
            tool call resolves the denylist once.

    Returns:
        Exclude pathspecs, possibly empty of location-based entries but
        never empty overall (the name rule always applies). They may be
        passed as the ONLY pathspecs after ``--``: git applies an
        exclude-only list to the whole tree (verified).

    Raises:
        LocalToolError: The repository root is itself a protected path,
            or the denylist produced an exclusion kind this renderer does
            not know how to express.
    """
    specs: list[str] = []
    for kind, value in sensitive_exclusions_under(repo_root, context=context):
        if kind in {"subtree", "file"}:
            if not value:
                raise LocalToolError(
                    f"repository root ({repo_root}) is a protected path; refusing"
                )
            specs.append(f":(exclude,literal,icase){value}")
        elif kind == "direct_children":
            prefix = f"{_glob_escape(value)}/" if value else ""
            specs.append(f":(exclude,glob,icase){prefix}*")
        elif kind == "name":
            specs.append(f":(exclude,glob,icase)**/{_glob_escape(value)}")
        else:  # pragma: no cover - defensive; see the docstring
            raise LocalToolError(
                f"unsupported sensitive-path exclusion kind: {kind!r}"
            )
    return tuple(specs)


def prepare_repository(
    workspace_root: Path,
    path: str = ".",
    *,
    context: SensitivePathContext | None = None,
) -> Path:
    """Resolve the git repo root for ``path``, confined to ``workspace_root``.

    Refuses (LocalToolError) when git is unavailable, ``path`` escapes the
    workspace, no repository is found, or the discovered repo root is ABOVE
    the workspace root (the repo root must be the workspace root or inside
    it — a workspace nested inside a repo is refused so the model cannot
    read repo state outside the confinement).

    Args:
        workspace_root: The confinement root ``path`` must resolve inside.
        path: The workspace-relative (or absolute-but-confined) location to
            discover a repository from; ``"."`` (the default) discovers
            from ``workspace_root`` itself.
        context: Optional pre-resolved ``SensitivePathContext``. Passed
            through to both denylist checks this function makes (the
            ``path`` argument, via ``resolve_workspace_path``, and the
            DISCOVERED repo root below) so a caller that also needs the
            same context for e.g. ``_denylist_pathspecs`` resolves the
            ~11 config accessors behind the denylist once per tool call
            instead of once per check. ``None`` resolves fresh each time —
            still enforces the denylist, just not shared.

    Returns:
        The resolved repository root path.
    """
    if shutil.which("git") is None:
        raise LocalToolError("git is not available on this system")
    workspace_root = Path(workspace_root).resolve()
    target = resolve_workspace_path(path, workspace_root, context=context)
    result = run_git(
        ["git", "-C", str(target), "rev-parse", "--show-toplevel"],
        timeout=REPOSITORY_DISCOVERY_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        gist = (result.stderr or result.stdout).strip().splitlines()
        gist_text = gist[0][:200] if gist else "unknown error"
        combined = (result.stderr + result.stdout).lower()
        if "not a git repository" in combined:
            raise LocalToolError(f"'{path}' is not a git repository: {gist_text}")
        raise LocalToolError(f"git repository discovery failed: {gist_text}")
    if result.truncated:
        raise LocalToolError("git returned truncated repository information")
    first_line = result.stdout.strip().splitlines()
    if not first_line:
        raise LocalToolError("git returned invalid repository information")
    repo_root = Path(first_line[0]).expanduser()
    if not repo_root.is_absolute():
        raise LocalToolError("git returned invalid repository information")
    repo_root = repo_root.resolve()
    if not (repo_root == workspace_root or workspace_root in repo_root.parents):
        raise LocalToolError(
            f"repository root ({repo_root}) is outside the workspace root ({workspace_root}); refusing"
        )
    # The repo root is DISCOVERED by git, not supplied by the model, so it
    # is the one path in this family the choke point never sees. Today it
    # can only be the workspace root or below it, and both are already
    # denylist-checked -- this is here so that stays true if either
    # relationship is ever relaxed: excluding denied paths from a
    # repository that is ITSELF denied would leave nothing honest to show.
    if is_sensitive_path(repo_root, context=context):
        raise LocalToolError(
            f"repository root ({repo_root}) is a protected path; refusing"
        )
    return repo_root


# ---------------------------------------------------------------------------
# Tool cores (sync adaptations of the reference's _execute_* functions)
# ---------------------------------------------------------------------------

GIT_LOG_DEFAULT_COUNT = 20
GIT_LOG_MAX_COUNT = 100
GIT_STATUS_MAX_ENTRIES = 200
GIT_BLAME_MAX_LINES = 500

_COMMIT_RANGE_PATTERN = re.compile(r"^[A-Za-z0-9._/~^-]+$")
_EMAIL_PATTERN = re.compile(r"<[^<>\s@]+@[^<>\s@]+>|\b\S+@\S+\b")


def _stderr_gist(result: GitCommandResult) -> str:
    for line in (result.stderr or result.stdout).strip().splitlines():
        if line.strip():
            return line.strip()[:200]
    return f"exit code {result.returncode}"


def _run_git_checked(argv: list[str], *, subcommand: str) -> GitCommandResult:
    result = run_git(argv)
    # A truncated result means the output cap fired and WE killed git
    # (SIGKILL -> returncode -9): that is not a git failure. Deliver the
    # bounded partial output (run_git already appended the truncation
    # marker) — that is the whole point of the bounded-read design.
    if result.returncode != 0 and not result.truncated:
        raise LocalToolError(f"git {subcommand} failed: {_stderr_gist(result)}")
    return result


def _repo_relative_path(
    workspace_root: Path,
    repo_root: Path,
    path: str,
    *,
    context: SensitivePathContext | None = None,
) -> str:
    """Resolve ``path`` confined to the workspace, rendered repo-relative.

    Args:
        workspace_root: The confinement root ``path`` must resolve inside.
        repo_root: The already-discovered repository root.
        path: The workspace-relative (or absolute-but-confined) path.
        context: Optional pre-resolved ``SensitivePathContext``, threaded
            through to ``resolve_workspace_path`` so a caller that has
            already resolved one for the same tool call does not pay for
            it again here.
    """
    resolved = resolve_workspace_path(path, Path(workspace_root).resolve(), context=context)
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        raise LocalToolError(
            f"path '{path}' is outside the repository root ({repo_root})"
        ) from None
    return relative.as_posix() or "."


def _prepare_for_path(
    workspace_root: Path,
    path: str | None,
    *,
    context: SensitivePathContext | None = None,
) -> Path:
    """Repo discovery that tolerates ``path`` being a file (uses its parent).

    Args:
        workspace_root: The confinement root ``path`` must resolve inside.
        path: The workspace-relative (or absolute-but-confined) path to
            discover a repository from, or ``None`` to discover from
            ``workspace_root`` itself.
        context: Optional pre-resolved ``SensitivePathContext``, threaded
            through to every denylist check this function (and
            ``prepare_repository`` beneath it) makes, so a caller that
            resolves one per tool call — rather than letting each check
            resolve its own — pays the ~11 config-accessor cost once.
    """
    if path is None:
        return prepare_repository(workspace_root, ".", context=context)
    resolved = resolve_workspace_path(path, Path(workspace_root).resolve(), context=context)
    discovery = resolved if resolved.is_dir() else resolved.parent
    relative = os.path.relpath(discovery, Path(workspace_root).resolve())
    return prepare_repository(workspace_root, relative, context=context)


def _sanitize_author_name(value: str) -> str:
    return " ".join(_EMAIL_PATTERN.sub("", value).split())


def _nul_records(stdout: str) -> list[str]:
    return [record for record in stdout.split("\0") if record]


def git_status(workspace_root: Path, path: str = ".") -> str:
    """Branch header + staged/unstaged/untracked/conflicted entries as text.

    Sync adaptation of the reference's ``_execute_status`` (:570): porcelain
    v2 ``-z`` output with the ``--branch`` header, parsed and rendered as
    ``category: XY path`` lines capped at ``GIT_STATUS_MAX_ENTRIES``.

    Denylisted paths are excluded by pathspec (``_denylist_pathspecs``):
    this tool named ``~/.ssh/id_rsa`` on a dirty ``$HOME``-rooted
    workspace (TASK-19632). Existence and a name are all a status entry
    carries, so excluding them is the whole refusal.

    Args:
        workspace_root: The confinement root ``path`` must resolve inside.
        path: Used ONLY to discover which repository to report on (a file
            or directory anywhere inside the target repo works, since
            discovery walks up to the repo root). It is NOT applied as a
            scoping pathspec: unlike ``git_diff``/``git_log``, this
            function's argv carries no positive pathspec for ``path``, so
            the status returned always covers the WHOLE repository —
            asking for a subdirectory's status still returns every
            changed file in the repo, not just that subdirectory's.

    Returns:
        The branch header line, one ``category: XY path`` (or
        ``untracked: path``) line per entry up to ``GIT_STATUS_MAX_ENTRIES``,
        ``"(working tree clean)"`` when there are none, and a truncation
        note appended if the repository has more entries than the cap.
    """
    context = resolve_sensitive_context()
    repo_root = _prepare_for_path(workspace_root, path, context=context)
    result = _run_git_checked(
        [
            "git",
            "--no-pager",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v2",
            "-z",
            "--branch",
            "--untracked-files=all",
            "--",
            *_denylist_pathspecs(repo_root, context=context),
        ],
        subcommand="status",
    )
    branch, entries, truncated = _parse_status_porcelain_v2(
        result.stdout, limit=GIT_STATUS_MAX_ENTRIES
    )
    lines = [_format_branch_header(branch)]
    lines.extend(_format_status_entry(entry) for entry in entries)
    if not entries:
        lines.append("(working tree clean)")
    if truncated:
        lines.append("… (more entries, truncated)")
    return "\n".join(lines)


def _parse_status_porcelain_v2(
    stdout: str, *, limit: int
) -> tuple[dict[str, object], list[dict[str, object]], bool]:
    branch: dict[str, object] = {
        "branch": None,
        "upstream": None,
        "ahead": None,
        "behind": None,
    }
    entries: list[dict[str, object]] = []
    total = 0
    for record in _nul_records(stdout):
        if record.startswith("# "):
            _parse_status_branch_header(record, branch)
            continue
        if record.startswith("! "):
            continue
        entry = _parse_status_entry(record)
        if entry is None:
            continue
        total += 1
        if len(entries) < limit:
            entries.append(entry)
    return branch, entries, total > limit


def _parse_status_branch_header(record: str, branch: dict[str, object]) -> None:
    if record.startswith("# branch.head "):
        value = record.removeprefix("# branch.head ").strip()
        branch["branch"] = None if value == "(detached)" else value or None
        return
    if record.startswith("# branch.upstream "):
        branch["upstream"] = record.removeprefix("# branch.upstream ").strip() or None
        return
    if record.startswith("# branch.ab "):
        for part in record.removeprefix("# branch.ab ").split():
            if part.startswith("+"):
                with contextlib.suppress(ValueError):
                    branch["ahead"] = int(part[1:])
            elif part.startswith("-"):
                with contextlib.suppress(ValueError):
                    branch["behind"] = int(part[1:])


def _parse_status_entry(record: str) -> dict[str, object] | None:
    if record.startswith("? "):
        path = record[2:].strip()
        if not path:
            return None
        return {"path": path, "xy": "??", "category": "untracked"}
    if record.startswith("1 "):
        parts = record.split(" ", 8)
        if len(parts) < 9:
            return None
        return _status_entry_from_xy(parts[1], parts[8])
    if record.startswith("2 "):
        parts = record.split(" ", 9)
        if len(parts) < 10:
            return None
        return _status_entry_from_xy(parts[1], parts[9])
    if record.startswith("u "):
        parts = record.split(" ", 10)
        if len(parts) < 11 or not parts[10].strip():
            return None
        return {"path": parts[10].strip(), "xy": parts[1], "category": "conflicted"}
    return None


def _status_entry_from_xy(xy: str, path_raw: str) -> dict[str, object] | None:
    path = path_raw.strip()
    if not path or len(xy) < 2:
        return None
    staged = xy[0] not in {".", "?", "!"}
    unstaged = xy[1] not in {".", "?", "!"}
    if staged and unstaged:
        category = "staged+unstaged"
    elif staged:
        category = "staged"
    elif unstaged:
        category = "unstaged"
    else:
        category = "clean"
    return {"path": path, "xy": xy, "category": category}


def _format_branch_header(branch: dict[str, object]) -> str:
    name = branch.get("branch") or "(detached)"
    extras: list[str] = []
    if branch.get("upstream"):
        extras.append(f"upstream: {branch['upstream']}")
    if branch.get("ahead") is not None:
        extras.append(f"ahead: {branch['ahead']}")
    if branch.get("behind") is not None:
        extras.append(f"behind: {branch['behind']}")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"branch: {name}{suffix}"


def _format_status_entry(entry: dict[str, object]) -> str:
    category = entry["category"]
    if category == "untracked":
        return f"untracked: {entry['path']}"
    return f"{category}: {entry['xy']} {entry['path']}"


def git_branches(workspace_root: Path) -> str:
    """Verbose branch list with the current branch marked by ``*``.

    Sync adaptation of the reference's ``_execute_branches`` (:621).
    """
    repo_root = prepare_repository(workspace_root, ".", context=resolve_sensitive_context())
    result = _run_git_checked(
        [
            "git",
            "--no-pager",
            "-C",
            str(repo_root),
            "branch",
            "--format=%(HEAD)%00%(refname:short)%00%(upstream:short)%00%(objectname)",
        ],
        subcommand="branch",
    )
    lines: list[str] = []
    for record in result.stdout.splitlines():
        if not record:
            continue
        parts = record.split("\0")
        if len(parts) < 4:
            continue
        marker, name, upstream, commit = (part.strip() for part in parts[:4])
        if not name:
            continue
        extras: list[str] = []
        if commit:
            extras.append(commit[:12])
        if upstream:
            extras.append(f"upstream: {upstream}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"* {name}{suffix}" if marker == "*" else f"  {name}{suffix}")
    return "\n".join(lines) if lines else "(no branches)"


def git_log(
    workspace_root: Path,
    *,
    count: int = GIT_LOG_DEFAULT_COUNT,
    path: str | None = None,
) -> str:
    """Bounded commit log, newest first; ``count`` is clamped to 1..100.

    Sync adaptation of the reference's ``_execute_log`` (:831). Deviation:
    ``count`` defaults to 20 here (the reference has no default and falls
    back to its max-100 when no limit is given).

    Deliberately NOT given the denylist exclusions ``git_status``/
    ``git_diff`` carry: the ``--format`` below emits commit metadata only
    -- no paths, no content -- so this tool was measured leaking nothing,
    and excluding denied paths here would silently drop commits from a
    legitimate history instead of protecting anything (TASK-19632). Its
    ``path`` pathspec IS rendered literally, for the same reason
    ``git_diff``'s is: the value is a model-supplied filename and a bare
    one would be parsed as pathspec magic.

    Args:
        workspace_root: The confinement root ``path`` must resolve inside.
        count: Maximum number of commits to return, clamped to 1..100.
        path: When given, scopes the log to commits touching this file or
            directory (rendered as a literal pathspec — see the note
            above); ``None`` (the default) returns the log for the whole
            repository ``path`` (or ``workspace_root`` when ``path`` is
            also omitted) discovers.

    Returns:
        One ``short_hash date author: subject`` line per commit, newest
        first, or ``"(no commits)"`` when there are none.
    """
    count = min(max(int(count), 1), GIT_LOG_MAX_COUNT)
    context = resolve_sensitive_context()
    repo_root = _prepare_for_path(workspace_root, path, context=context)
    argv = [
        "git",
        "--no-pager",
        "-C",
        str(repo_root),
        "log",
        "--format=%H%x1f%h%x1f%an%x1f%aI%x1f%s%x1e",
        "-n",
        str(count),
    ]
    if path is not None:
        argv.extend(
            [
                "--",
                _literal_pathspec(
                    _repo_relative_path(workspace_root, repo_root, path, context=context)
                ),
            ]
        )
    result = _run_git_checked(argv, subcommand="log")
    lines: list[str] = []
    for record in result.stdout.split("\x1e"):
        fields = record.strip("\n").split("\x1f", 4)
        if len(fields) < 5:
            continue
        _commit_hash, short_hash, author_name, author_date, subject = fields
        lines.append(
            f"{short_hash} {author_date} {_sanitize_author_name(author_name)}: {subject}"
        )
    return "\n".join(lines) if lines else "(no commits)"


def git_diff(
    workspace_root: Path,
    *,
    staged: bool = False,
    commit_range: str | None = None,
    path: str | None = None,
    stat: bool = False,
) -> str:
    """Unified diff of the worktree (default) or the index (``staged=True``).

    Sync adaptation of the reference's ``_execute_diff`` (:709) +
    ``_run_diff_command`` (:1049-1080 — ``--no-ext-diff``/``--no-textconv``/
    ``--no-color`` ported). Disclosed deviations: adds ``commit_range``
    (regex-validated before entering argv) and ``stat`` modes; the
    reference's third ``working_tree`` scope is omitted.

    Denylisted paths are excluded by pathspec (``_denylist_pathspecs``)
    in every mode — worktree, index and ``commit_range`` alike, since the
    leak this closes was reachable from a CLEAN worktree by reading the
    credential out of history (TASK-19632). Exclusions apply whether or
    not the caller supplied ``path``: a denylisted ``path`` is refused by
    the choke point, but the leak was in the no-``path`` case, where the
    model names nothing and git enumerates the repository.

    Nothing announces that an exclusion took effect, deliberately. The
    only honest note would state that this repository contains a protected
    path — which is the same disclosure ``stat=True``/``git_status`` were
    leaking. A model that names the path still gets a "protected path"
    refusal, which is the case where the information is actionable.

    Args:
        workspace_root: The confinement root ``path`` must resolve inside.
        staged: When True, diff the index against ``HEAD`` (``--cached``)
            instead of the worktree.
        commit_range: When given, diff across this ref/range instead of
            against the worktree or index (validated against
            ``^[A-Za-z0-9._/~^-]+$`` and refused if it starts with ``-``).
        path: When given, scopes the diff to this file or directory
            (rendered as a literal pathspec, so a magic-shaped filename is
            matched literally); ``None`` (the default) diffs the whole
            repository ``path`` (or ``workspace_root`` when ``path`` is
            also omitted) discovers.
        stat: When True, return a ``--stat`` summary instead of a unified
            patch.

    Returns:
        The unified diff (or ``--stat`` summary) text, or ``"(no
        changes)"`` when the diff is empty.
    """
    if commit_range is not None:
        # Leading-dash values are FLAGS, not refnames (git refnames cannot
        # begin with a dash). The regex alone allows "--textconv" et al.,
        # and because the range lands LAST in argv, git's
        # last-occurrence-wins would re-enable textconv/ext-diff over the
        # machine-safe --no-textconv/--no-ext-diff already present —
        # verified as a command-execution escape via a hostile repo's
        # .gitattributes diff driver. Refuse outright.
        if commit_range.startswith("-") or not _COMMIT_RANGE_PATTERN.match(commit_range):
            raise LocalToolError(
                f"invalid commit_range {commit_range!r}: "
                "must be a ref/range matching [A-Za-z0-9._/~^-] and not start with '-'"
            )
    context = resolve_sensitive_context()
    repo_root = _prepare_for_path(workspace_root, path, context=context)
    argv = [
        "git",
        "--no-pager",
        "-C",
        str(repo_root),
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
    ]
    if stat:
        argv.append("--stat")
    else:
        # --unified implies patch output; combining it with --stat would
        # emit BOTH formats, so patch context is only set in patch mode.
        argv.append("--unified=3")
    if staged:
        argv.append("--cached")
    if commit_range is not None:
        argv.append(commit_range)
    pathspecs: list[str] = []
    if path is not None:
        pathspecs.append(
            _literal_pathspec(
                _repo_relative_path(workspace_root, repo_root, path, context=context)
            )
        )
    # Exclusions come LAST and are never empty (the name rule always
    # applies), so `--` is always present: an exclude-only pathspec list
    # is applied to the whole tree, which is exactly the no-`path` case.
    pathspecs.extend(_denylist_pathspecs(repo_root, context=context))
    argv.extend(["--", *pathspecs])
    result = _run_git_checked(argv, subcommand="diff")
    return result.stdout if result.stdout.strip() else "(no changes)"


def git_blame(
    workspace_root: Path,
    path: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Per-line blame for ``path``; optional 1-based inclusive line range.

    Sync adaptation of the reference's ``_execute_blame`` (:892) — line
    porcelain parse — except the ``-L`` range is optional here (omitted when
    neither bound is given) and the range is capped at
    ``GIT_BLAME_MAX_LINES`` lines.
    """
    context = resolve_sensitive_context()
    resolved = resolve_workspace_path(path, Path(workspace_root).resolve(), context=context)
    if not resolved.is_file():
        raise LocalToolError(f"file not found: {path}")
    repo_root = _prepare_for_path(workspace_root, path, context=context)
    try:
        repo_relative = resolved.relative_to(repo_root).as_posix()
    except ValueError:
        raise LocalToolError(
            f"path '{path}' is outside the repository root ({repo_root})"
        ) from None

    argv = [
        "git",
        "--no-pager",
        "-C",
        str(repo_root),
        "blame",
        "--line-porcelain",
        "--no-textconv",
    ]
    if start_line is not None or end_line is not None:
        start = int(start_line) if start_line is not None else 1
        if start < 1:
            raise LocalToolError(f"start_line must be >= 1, got {start}")
        end = int(end_line) if end_line is not None else start + GIT_BLAME_MAX_LINES - 1
        if end < start:
            raise LocalToolError(f"end_line ({end}) is before start_line ({start})")
        end = min(end, start + GIT_BLAME_MAX_LINES - 1)
        argv.extend(["-L", f"{start},{end}"])
    # NOT wrapped in `:(literal)` like the diff/log pathspecs above:
    # `git blame` takes a plain PATH here, not a pathspec, and rejects
    # magic outright (`fatal: no such path ':(literal)a.txt' in HEAD`,
    # verified on git 2.39). It therefore never interprets a magic-shaped
    # filename either -- blaming a repository file literally named
    # `:(exclude)notes.txt` blames that file, checked the same way as any
    # other path: `resolve_workspace_path` above already denylist-checked
    # it, and `resolved.is_file()` means it must genuinely exist.
    argv.extend(["--", repo_relative])

    result = _run_git_checked(argv, subcommand="blame")
    lines = [f"{ln}: {author}: {text}" for ln, author, text in _parse_blame(result.stdout)]
    return "\n".join(lines) if lines else "(no blame output)"


def _parse_blame(stdout: str) -> list[tuple[int, str, str]]:
    """Parse ``blame --line-porcelain`` into (line_number, author, text)."""
    lines: list[tuple[int, str, str]] = []
    current: dict[str, object] | None = None
    commit_metadata: dict[str, dict[str, object]] = {}
    for raw_line in stdout.splitlines():
        if raw_line.startswith("\t"):
            if current is None:
                continue
            author = _sanitize_author_name(str(current.get("author_name") or ""))
            lines.append((int(current["line_number"]), author, raw_line[1:]))
            current = None
            continue
        header = _parse_blame_header(raw_line)
        if header is not None:
            cached = commit_metadata.get(str(header["commit"]))
            if cached:
                header.update(cached)
            current = header
            continue
        if current is None:
            continue
        if raw_line.startswith("author "):
            author_name = raw_line.removeprefix("author ")
            current["author_name"] = author_name
            commit_metadata.setdefault(str(current["commit"]), {})["author_name"] = author_name
    return lines


def _parse_blame_header(raw_line: str) -> dict[str, object] | None:
    # Group headers carry 4 fields (sha orig final count); subsequent headers
    # for lines of the same commit group carry only 3 (sha orig final) — the
    # reference's `len(parts) < 4` check drops those lines, so this port
    # deliberately accepts >= 3 (deviation; see module header).
    parts = raw_line.split()
    if len(parts) < 3:
        return None
    commit_hash = parts[0]
    if len(commit_hash) < 8 or not all(
        character in "0123456789abcdefABCDEF" for character in commit_hash
    ):
        return None
    with contextlib.suppress(ValueError):
        return {
            "commit": commit_hash,
            "line_number": int(parts[2]),
            "author_name": None,
        }
    return None
