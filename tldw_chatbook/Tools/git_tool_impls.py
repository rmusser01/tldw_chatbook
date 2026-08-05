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


def prepare_repository(workspace_root: Path, path: str = ".") -> Path:
    """Resolve the git repo root for ``path``, confined to ``workspace_root``.

    Refuses (LocalToolError) when git is unavailable, ``path`` escapes the
    workspace, no repository is found, or the discovered repo root is ABOVE
    the workspace root (the repo root must be the workspace root or inside
    it — a workspace nested inside a repo is refused so the model cannot
    read repo state outside the confinement).

    Returns:
        The resolved repository root path.
    """
    if shutil.which("git") is None:
        raise LocalToolError("git is not available on this system")
    workspace_root = Path(workspace_root).resolve()
    target = resolve_workspace_path(path, workspace_root)
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


def _repo_relative_path(workspace_root: Path, repo_root: Path, path: str) -> str:
    """Resolve ``path`` confined to the workspace, rendered repo-relative."""
    resolved = resolve_workspace_path(path, Path(workspace_root).resolve())
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        raise LocalToolError(
            f"path '{path}' is outside the repository root ({repo_root})"
        ) from None
    return relative.as_posix() or "."


def _prepare_for_path(workspace_root: Path, path: str | None) -> Path:
    """Repo discovery that tolerates ``path`` being a file (uses its parent)."""
    if path is None:
        return prepare_repository(workspace_root, ".")
    resolved = resolve_workspace_path(path, Path(workspace_root).resolve())
    discovery = resolved if resolved.is_dir() else resolved.parent
    relative = os.path.relpath(discovery, Path(workspace_root).resolve())
    return prepare_repository(workspace_root, relative)


def _sanitize_author_name(value: str) -> str:
    return " ".join(_EMAIL_PATTERN.sub("", value).split())


def _nul_records(stdout: str) -> list[str]:
    return [record for record in stdout.split("\0") if record]


def git_status(workspace_root: Path, path: str = ".") -> str:
    """Branch header + staged/unstaged/untracked/conflicted entries as text.

    Sync adaptation of the reference's ``_execute_status`` (:570): porcelain
    v2 ``-z`` output with the ``--branch`` header, parsed and rendered as
    ``category: XY path`` lines capped at ``GIT_STATUS_MAX_ENTRIES``.
    """
    repo_root = _prepare_for_path(workspace_root, path)
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
    repo_root = prepare_repository(workspace_root, ".")
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
    """
    count = min(max(int(count), 1), GIT_LOG_MAX_COUNT)
    repo_root = _prepare_for_path(workspace_root, path)
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
        argv.extend(["--", _repo_relative_path(workspace_root, repo_root, path)])
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
    repo_root = _prepare_for_path(workspace_root, path)
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
    if path is not None:
        argv.extend(["--", _repo_relative_path(workspace_root, repo_root, path)])
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
    resolved = resolve_workspace_path(path, Path(workspace_root).resolve())
    if not resolved.is_file():
        raise LocalToolError(f"file not found: {path}")
    repo_root = _prepare_for_path(workspace_root, path)
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
