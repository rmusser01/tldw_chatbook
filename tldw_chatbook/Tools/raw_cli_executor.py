"""Value contracts and pure boundary helpers for raw one-shot CLI execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
from typing import Literal, TypeAlias

MAX_RAW_COMMAND_BYTES = 16 * 1024
MAX_RAW_TIMEOUT_SECONDS = 300.0
MAX_RAW_PREVIEW_BYTES = 32 * 1024

RawCliCaller: TypeAlias = Literal["user", "model"]
RawCliShell: TypeAlias = Literal["auto", "bash", "powershell", "cmd"]
RawCliTerminalState: TypeAlias = Literal[
    "refused",
    "shell_unavailable",
    "spawn_failed",
    "containment_unavailable",
    "exited",
    "timed_out",
    "cancelled",
    "cleanup_unproven",
]
RawCliStream: TypeAlias = Literal["stdout", "stderr"]

_SHELL_ENVIRONMENT_KEYS = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "LC_COLLATE",
    "LC_MONETARY",
    "LC_NUMERIC",
    "LC_TIME",
    "LC_PAPER",
    "LC_NAME",
    "LC_ADDRESS",
    "LC_TELEPHONE",
    "LC_MEASUREMENT",
    "LC_IDENTIFICATION",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
)


@dataclass(frozen=True, slots=True)
class RawCliRequest:
    """One validated, non-interactive raw shell request."""

    invocation_id: str
    caller: RawCliCaller
    command: str
    shell: RawCliShell
    initial_directory: Path
    timeout_seconds: float
    console_session_id: str
    transcript_anchor_id: str | None = None


@dataclass(frozen=True, slots=True)
class RawCliStreamEvent:
    """One bounded update from exactly one child output stream."""

    stream: RawCliStream
    text: str
    total_bytes: int
    truncated: bool

    def __post_init__(self) -> None:
        if self.stream not in ("stdout", "stderr"):
            raise ValueError("stream must be stdout or stderr")


@dataclass(frozen=True, slots=True)
class RawCliResult:
    """Terminal result shared by user and model raw CLI adapters."""

    invocation_id: str
    caller: RawCliCaller
    resolved_shell: str
    initial_directory: Path
    elapsed_seconds: float
    stdout_preview: str
    stderr_preview: str
    record_output: str
    exit_code: int | None
    terminal_state: RawCliTerminalState
    truncated: bool
    cleanup_proven: bool


def validate_raw_cli_request(request: RawCliRequest) -> None:
    """Reject a request that cannot safely cross the executor boundary."""
    if not isinstance(request.command, str) or not request.command.strip():
        raise ValueError("raw CLI command must not be empty or whitespace")
    if "\x00" in request.command:
        raise ValueError("raw CLI command must not contain NUL")
    try:
        command_bytes = len(request.command.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("raw CLI command must be valid UTF-8") from exc
    if command_bytes > MAX_RAW_COMMAND_BYTES:
        raise ValueError("raw CLI command exceeds the 16 KiB UTF-8 limit")

    timeout = request.timeout_seconds
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
        or timeout > MAX_RAW_TIMEOUT_SECONDS
    ):
        raise ValueError(
            "raw CLI timeout must be greater than 0 and at most 300 seconds"
        )

    directory = request.initial_directory
    if (
        not isinstance(directory, Path)
        or not directory.is_absolute()
        or not directory.exists()
        or not directory.is_dir()
    ):
        raise ValueError(
            "raw CLI initial directory must be an absolute existing directory"
        )


def resolve_shell_argv(
    selector: RawCliShell,
    command: str,
    *,
    executable_lookup: Callable[[str], str | None] = shutil.which,
    platform_name: str | None = None,
) -> tuple[str, ...]:
    """Return profile-disabled argv using deterministic injected shell lookup."""
    platform_name = os.name if platform_name is None else platform_name
    if selector == "auto":
        candidates = (
            ("pwsh", "powershell", "cmd.exe")
            if platform_name == "nt"
            else ("bash", "sh")
        )
    elif selector == "bash":
        candidates = ("bash",)
    elif selector == "powershell":
        candidates = ("pwsh", "powershell")
    elif selector == "cmd":
        candidates = ("cmd.exe",)
    else:
        raise ValueError(f"unsupported raw CLI shell selector: {selector!r}")

    for shell_name in candidates:
        executable = executable_lookup(shell_name)
        if executable:
            break
    else:
        raise FileNotFoundError(f"raw CLI shell unavailable for selector {selector!r}")

    if shell_name == "bash":
        return (executable, "--noprofile", "--norc", "-c", command)
    if shell_name == "sh":
        return (executable, "-c", command)
    if shell_name == "cmd.exe":
        return (executable, "/D", "/S", "/C", command)
    return (
        executable,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        command,
    )


def build_scrubbed_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy only shell usability variables into a new empty environment."""
    source = os.environ if source is None else source
    return {key: source[key] for key in _SHELL_ENVIRONMENT_KEYS if key in source}
