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

Note for Task 2 (tool cores): diff argv must carry the machine-safe flags
from the reference's ``_run_diff_command`` (:1049-1080): ``--no-ext-diff``,
``--no-textconv``, ``--no-color``.
"""

from __future__ import annotations

import contextlib
import os
import queue
import shutil
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
