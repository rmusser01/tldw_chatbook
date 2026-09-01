"""Value contracts and pure boundary helpers for raw one-shot CLI execution."""

from __future__ import annotations

import codecs
from collections.abc import Callable, Mapping
import contextlib
from dataclasses import dataclass, replace
import math
import multiprocessing
import ntpath
import os
from pathlib import Path
import posixpath
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, BinaryIO, Literal, TypeAlias

from tldw_chatbook.Agents.run_log import configured_max_record_bytes
from tldw_chatbook.Utils.input_validation import (
    RAW_CLI_COMMAND_MAX_BYTES,
    RAW_CLI_TIMEOUT_MAX_SECONDS,
    validate_raw_cli_command,
)
from tldw_chatbook.Utils.path_validation import (
    validate_existing_absolute_directory,
)
from tldw_chatbook.STT.executor_process_tree import (
    ExecutorProcessTree,
    WorkerContainmentIdentity,
    enter_worker_containment,
)

MAX_RAW_COMMAND_BYTES = RAW_CLI_COMMAND_MAX_BYTES
MAX_RAW_TIMEOUT_SECONDS = RAW_CLI_TIMEOUT_MAX_SECONDS
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
RawCliAdmissionCallback: TypeAlias = Callable[
    [ExecutorProcessTree, Callable[[], float | None]], object
]

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

_RAW_OUTPUT_CHUNK_BYTES = 8 * 1024
_RAW_OUTPUT_QUEUE_SIZE = 16
_RAW_OUTPUT_FLUSH_SECONDS = 0.05
_RAW_STARTUP_TIMEOUT_SECONDS = 10.0
_RAW_QUEUE_POLL_SECONDS = 0.05
_RAW_POST_EXIT_DRAIN_SECONDS = 1.0
_RAW_CLI_PROCESS_CONSTRUCTION_LOCK = threading.Lock()
_RAW_CLI_STDERR_FALLBACK: Any | None = None


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


# ---------------------------------------------------------------------------
# TASK-25905: the unbypassable hardline command floor.
#
# A deliberately SMALL set of catastrophic command shapes is refused at this
# validation boundary -- before any permission state, approval stamp, or
# session grant is consulted, for BOTH callers. This is a floor under the
# approval card, not a replacement for it. It is not configurable: the rule
# tuple is a module constant and nothing reads config here (AC#5); any
# user-supplied deny list elsewhere is additive on top.
#
# Detection philosophy: normalize away TRIVIAL obfuscation (quotes,
# backslashes, whitespace padding) and match argument SHAPES anchored to
# command position, so `$X -rf /` trips on the arguments even when the
# command word is hidden behind a variable. A determined adversary can
# still evade a static floor; the approval card remains the real gate.

#: Fork bomb: a function that pipes into itself and backgrounds, then calls
#: itself (the classic ``:(){ :|:& };:`` and named variants). Matched on the
#: RAW text because shlex cannot tokenize it -- and its pattern is specific
#: enough to carry no false-positive risk.
_FORK_BOMB_RE = re.compile(
    r"(\S+)\s*\(\s*\)\s*\{[^}]*\1[^}]*\|[^}]*\1[^}]*&[^}]*\}\s*;\s*\1"
)

#: Block-device targets for a `dd of=` catastrophe. Includes rdisk (macOS
#: raw disk -- this app's own platform), md/dm-/mapper (RAID/LVM).
_DD_BLOCK_DEVICE_RE = re.compile(
    r"^/dev/(?:sd|hd|nvme|disk|rdisk|mmcblk|vd|xvd|loop|md|dm-|mapper/)"
)

#: Catastrophic delete targets: filesystem root and home in their common
#: literal and variable forms (shlex does not expand, so these stay literal).
_HARDLINE_ROOT_TARGETS = frozenset({"/", "//", "/*", "~", "~/", "$HOME", "${HOME}"})

#: Command wrappers whose leading occurrence is transparent to the floor.
_HARDLINE_WRAPPERS = frozenset(
    {"sudo", "env", "command", "nice", "nohup", "time", "doas", "exec"}
)

#: Runtime commands that shut the machine down when in command position.
_SHUTDOWN_COMMANDS = frozenset({"shutdown", "poweroff", "halt", "reboot"})


class RawCliHardlineViolation(ValueError):
    """A catastrophic command shape hit the hardline floor.

    Distinct from every permission refusal on purpose (AC#4): callers
    surface it as "the safety floor refused this", never as a user denial,
    and no approval option can clear it.
    """

    def __init__(self, rule: str) -> None:
        self.rule = rule
        super().__init__(
            f"hardline safety floor: rule '{rule}' refuses this command"
        )


def _basename(token: str) -> str:
    """Command basename without a path or variable-expansion prefix."""
    return token.rsplit("/", 1)[-1]


def _split_simple_commands(command: str) -> list[list[str]]:
    """Tokenize with quote/escape awareness and split on UNQUOTED control
    operators (``; | || & && newline``).

    Uses shlex in POSIX mode with ``punctuation_chars`` so a control
    operator INSIDE a quoted argument (``git commit -m "fix; reboot"``)
    stays part of its token and never starts a new command -- the fix for
    the quote-blind false positives. On a parse error (unbalanced quotes)
    it falls back to a naive whitespace split, which for the floor's
    purpose over-includes rather than under-includes.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = command.split()
    commands: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token and set(token) <= set(";|&\n"):
            if current:
                commands.append(current)
                current = []
        elif token in ("(", ")"):
            # subshell grouping is transparent -- unwrap so `(rm -rf /)` is
            # analyzed as `rm -rf /` rather than command word `(`.
            continue
        else:
            current.append(token)
    if current:
        commands.append(current)
    return commands


def _strip_wrappers(tokens: list[str]) -> list[str]:
    """Drop leading VAR=val assignments and sudo/env-style wrappers so the
    real command word surfaces (`FOO=1 sudo rm ...` -> `rm ...`)."""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            index += 1
        elif _basename(token) in _HARDLINE_WRAPPERS:
            index += 1
            # skip the wrapper's own short options (e.g. `sudo -n`).
            # ponytail: a wrapper OPTION that takes an argument
            # (`sudo -u root ...`) leaves that argument as the apparent
            # command word, so `sudo -u root rm -rf /` slips this floor --
            # a documented, accepted gap (the approval card is the real
            # gate; parsing every wrapper's option arity is not worth it).
            while index < len(tokens) and tokens[index].startswith("-"):
                index += 1
        else:
            break
    return tokens[index:]


def _simple_command_violation(tokens: list[str]) -> str | None:
    """The hardline rule one already-split simple command violates."""
    tokens = _strip_wrappers(tokens)
    if not tokens:
        return None
    command = tokens[0]
    base = _basename(command)
    args = tokens[1:]

    # recursive-root-delete: `rm` (or a variable command word hiding it)
    # with recursive AND force among the flags and a root target among the
    # args. Anchored to rm-family so `cp -rf /` and `ls -laRF /` -- which
    # do NOT delete -- are not floored (the false positives the old
    # `\S+`-anchored regex produced).
    is_rm = base == "rm"
    is_variable_command = command.startswith("$") or command.startswith("${")
    if is_rm or is_variable_command:
        flag_letters = "".join(
            token[1:]
            for token in args
            if token.startswith("-") and not token.startswith("--")
        )
        long_flags = {token for token in args if token.startswith("--")}
        recursive = (
            "r" in flag_letters
            or "R" in flag_letters
            or "--recursive" in long_flags
        )
        force = "f" in flag_letters or "--force" in long_flags
        targets = {
            token for token in args if not token.startswith("-") and token != "--"
        }
        if recursive and force and targets & _HARDLINE_ROOT_TARGETS:
            return "recursive-root-delete"

    if base == "mkfs" or base.startswith("mkfs."):
        return "filesystem-format"

    if base == "dd":
        for token in args:
            if token.startswith("of="):
                if _DD_BLOCK_DEVICE_RE.match(token[3:]):
                    return "dd-to-block-device"

    if base in _SHUTDOWN_COMMANDS:
        return "system-shutdown"
    if base == "init" and any(arg in ("0", "6") for arg in args):
        return "system-shutdown"
    if base == "systemctl" and any(
        arg in ("poweroff", "reboot", "halt") for arg in args
    ):
        return "system-shutdown"
    return None


def hardline_violation(command: str) -> str | None:
    """Name the hardline rule ``command`` violates, or ``None``.

    Tokenizes with quote/escape awareness (so `rm -\'r\'f /` and
    `rm -"rf" /` collapse to `rm -rf /`, and a quoted `;` never splits a
    command) and matches argument SHAPES per simple command. The fork bomb
    is matched separately on the raw text since it is not shell-tokenizable.
    """
    if _FORK_BOMB_RE.search(command):
        return "fork-bomb"
    for tokens in _split_simple_commands(command):
        rule = _simple_command_violation(tokens)
        if rule is not None:
            return rule
    return None


# ---------------------------------------------------------------------------
# TASK-26006: actionable failure hints. Data, not branching (AC#6): each row
# is (compiled pattern, hint line). First match wins; hints apply only to
# non-zero exits; the original output is never altered -- callers APPEND the
# hint after it, labeled so it can never be mistaken for command output.

FAILURE_HINT_TABLE: tuple[tuple["re.Pattern[str]", str], ...] = (
    (
        re.compile(r"command not found|is not recognized as an internal"),
        "the command is not installed or not on PATH; check the spelling "
        "or install it first",
    ),
    (
        re.compile(r"[Pp]ermission denied"),
        "permission was denied; check file ownership/mode -- the raw shell "
        "runs as the app user with no sudo",
    ),
    (
        re.compile(r"No such file or directory"),
        "a path does not exist relative to the initial directory; verify "
        "with ls or use an absolute path",
    ),
    (
        re.compile(r"ModuleNotFoundError|ImportError: No module named"),
        "a Python dependency is missing in this environment; install it or "
        "run inside the right virtualenv",
    ),
    (
        re.compile(
            r"Connection refused|Could not resolve host|Network is unreachable"
        ),
        "the network call failed; check the host, port, and connectivity "
        "before retrying",
    ),
    (
        re.compile(r"not a git repository"),
        "this directory is not inside a git repository; cd into one or "
        "pass an explicit path",
    ),
    (
        re.compile(r"syntax error near unexpected token|unbound variable"),
        "the shell rejected the command's own syntax; check quoting and "
        "variable expansions",
    ),
    (
        re.compile(r"Address already in use"),
        "the port is taken by another process; pick another port or stop "
        "the process holding it",
    ),
)

#: Prefix marking every hint as tool-generated (AC#4).
FAILURE_HINT_PREFIX = "[tool hint]"


def failure_hint(exit_code: int | None, output: str) -> str | None:
    """One bounded, labeled recovery line for a known failure shape.

    Args:
        exit_code: The command's exit code; hints apply only when it is a
            non-zero integer (AC#2).
        output: Combined stdout/stderr text to match against.

    Returns:
        ``"[tool hint] ..."`` for the FIRST matching table row (AC#3), or
        ``None`` -- unrecognized failures return exactly today's output
        with nothing added (AC#5).
    """
    if not isinstance(exit_code, int) or exit_code == 0:
        return None
    for pattern, hint in FAILURE_HINT_TABLE:
        if pattern.search(output):
            return f"{FAILURE_HINT_PREFIX} {hint}"
    return None


def validate_raw_cli_request(request: RawCliRequest) -> RawCliRequest:
    """Validate and normalize a request crossing the executor boundary."""
    if request.caller not in ("user", "model"):
        raise ValueError("raw CLI caller must be user or model")
    if request.shell not in ("auto", "bash", "powershell", "cmd"):
        raise ValueError("raw CLI shell must be auto, bash, powershell, or cmd")
    for field_name in ("invocation_id", "console_session_id"):
        value = getattr(request, field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"raw CLI {field_name} must be a nonblank string")
    command = validate_raw_cli_command(
        request.command,
        max_bytes=MAX_RAW_COMMAND_BYTES,
    )
    violation = hardline_violation(command)
    if violation is not None:
        raise RawCliHardlineViolation(violation)

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

    try:
        directory = validate_existing_absolute_directory(request.initial_directory)
    except ValueError as exc:
        raise ValueError(
            "raw CLI initial directory must be an absolute existing directory"
        ) from exc
    return replace(request, command=command, initial_directory=directory)


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
            if platform_name == "nt":
                drive, tail = ntpath.splitdrive(executable)
                is_absolute = bool(drive) and tail.startswith(("/", "\\"))
                path_module = ntpath
            else:
                is_absolute = posixpath.isabs(executable)
                path_module = posixpath
            if not is_absolute:
                executable = path_module.abspath(executable)
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


def _stream_fileno(stream: Any) -> int:
    """Return a usable descriptor, or ``-1`` for Textual-style streams."""
    try:
        descriptor = stream.fileno()
    except Exception:
        return -1
    return descriptor if isinstance(descriptor, int) and descriptor >= 0 else -1


def _raw_cli_real_stderr() -> Any:
    """Return an fd-backed stderr kept alive for the resource tracker."""
    real = sys.__stderr__
    if real is not None and _stream_fileno(real) >= 0:
        return real
    global _RAW_CLI_STDERR_FALLBACK
    if _RAW_CLI_STDERR_FALLBACK is None:
        _RAW_CLI_STDERR_FALLBACK = open(os.devnull, "w")
    return _RAW_CLI_STDERR_FALLBACK


def _multiprocessing_stderr_context() -> Any:
    """Protect CPython's resource tracker from ``fileno() == -1`` stderr."""
    if _stream_fileno(sys.stderr) >= 0:
        return contextlib.nullcontext()
    return contextlib.redirect_stderr(_raw_cli_real_stderr())


class _StreamSanitizer:
    """Incrementally remove terminal controls while preserving literal text."""

    def __init__(self) -> None:
        self._state = "text"
        self._pending_cr = False

    def feed(self, text: str, *, final: bool = False) -> str:
        """Return sanitized text, retaining incomplete controls for the next call."""
        output: list[str] = []
        for character in text:
            if self._state == "text":
                if self._pending_cr:
                    output.append("\n")
                    self._pending_cr = False
                    if character == "\n":
                        continue
                codepoint = ord(character)
                if character == "\r":
                    self._pending_cr = True
                elif character in ("\n", "\t"):
                    output.append(character)
                elif character == "\x1b":
                    self._state = "escape"
                elif character == "\x9b":
                    self._state = "csi"
                elif character == "\x9d":
                    self._state = "osc"
                elif codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
                    continue
                else:
                    output.append(character)
            elif self._state == "escape":
                if character == "[":
                    self._state = "csi"
                elif character == "]":
                    self._state = "osc"
                elif character == "\x1b":
                    self._state = "escape"
                else:
                    self._state = "text"
                    codepoint = ord(character)
                    if character == "\r":
                        self._pending_cr = True
                    elif character in ("\n", "\t"):
                        output.append(character)
                    elif not (codepoint < 0x20 or 0x7F <= codepoint <= 0x9F):
                        output.append(character)
            elif self._state == "csi":
                if 0x40 <= ord(character) <= 0x7E:
                    self._state = "text"
            elif self._state == "osc":
                if character in ("\x07", "\x9c"):
                    self._state = "text"
                elif character == "\x1b":
                    self._state = "osc_escape"
            elif self._state == "osc_escape":
                if character == "\\" or character in ("\x07", "\x9c"):
                    self._state = "text"
                elif character != "\x1b":
                    self._state = "osc"

        if final:
            if self._pending_cr:
                output.append("\n")
            self._pending_cr = False
            self._state = "text"
        return "".join(output)


def _utf8_prefix(text: str, byte_limit: int) -> tuple[str, bool]:
    """Return the largest UTF-8-safe prefix within ``byte_limit``."""
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text, False
    if byte_limit <= 0:
        return "", bool(encoded)
    prefix = encoded[:byte_limit].decode("utf-8", errors="ignore")
    return prefix, True


class _OutputAccumulator:
    """Own bounded sanitized previews and the private record spool."""

    def __init__(self, spool: BinaryIO, max_record_bytes: int) -> None:
        self._spool = spool
        self._max_record_bytes = max_record_bytes
        self._record_bytes = 0
        self._preview_bytes = 0
        self._previews: dict[RawCliStream, list[str]] = {
            "stdout": [],
            "stderr": [],
        }
        self._raw_bytes: dict[RawCliStream, int] = {"stdout": 0, "stderr": 0}
        self._decoders = {
            stream: codecs.getincrementaldecoder("utf-8")(errors="replace")
            for stream in ("stdout", "stderr")
        }
        self._sanitizers = {
            stream: _StreamSanitizer() for stream in ("stdout", "stderr")
        }
        self._finished: set[RawCliStream] = set()
        self._reported_truncation: set[RawCliStream] = set()
        self.truncated = False

    def consume(
        self,
        stream: RawCliStream,
        payload: bytes,
        on_event: Callable[[RawCliStreamEvent], None],
    ) -> None:
        """Decode and sanitize one worker chunk at the parent choke point."""
        self._raw_bytes[stream] += len(payload)
        text = self._decoders[stream].decode(payload)
        self._accept(stream, self._sanitizers[stream].feed(text), on_event)

    def finish(
        self,
        stream: RawCliStream,
        on_event: Callable[[RawCliStreamEvent], None],
    ) -> None:
        """Flush one stream's incremental decoder and sanitizer once."""
        if stream in self._finished:
            return
        decoded = self._decoders[stream].decode(b"", final=True)
        text = self._sanitizers[stream].feed(decoded, final=True)
        self._accept(stream, text, on_event)
        self._finished.add(stream)

    def finish_all(self, on_event: Callable[[RawCliStreamEvent], None]) -> None:
        """Flush both streams after terminal settlement or forced cleanup."""
        self.finish("stdout", on_event)
        self.finish("stderr", on_event)

    def preview(self, stream: RawCliStream) -> str:
        """Return one stream's bounded transcript preview."""
        return "".join(self._previews[stream])

    def record_output(self) -> str:
        """Read the bounded private spool without exposing its path."""
        try:
            self._spool.flush()
            self._spool.seek(0)
            payload = self._spool.read(self._max_record_bytes)
        except (OSError, ValueError):
            raise OSError("raw CLI output spool I/O failed") from None
        return payload.decode("utf-8", errors="replace")

    def _accept(
        self,
        stream: RawCliStream,
        text: str,
        on_event: Callable[[RawCliStreamEvent], None],
    ) -> None:
        if not text:
            return
        preview_text, preview_truncated = _utf8_prefix(
            text,
            MAX_RAW_PREVIEW_BYTES - self._preview_bytes,
        )
        preview_size = len(preview_text.encode("utf-8"))
        if preview_text:
            self._previews[stream].append(preview_text)
            self._preview_bytes += preview_size

        record = f"[{stream}] {text}"
        record_text, record_truncated = _utf8_prefix(
            record,
            self._max_record_bytes - self._record_bytes,
        )
        if record_text:
            encoded_record = record_text.encode("utf-8")
            try:
                self._spool.write(encoded_record)
            except (OSError, ValueError):
                raise OSError("raw CLI output spool I/O failed") from None
            self._record_bytes += len(encoded_record)

        newly_truncated = preview_truncated or record_truncated
        self.truncated = self.truncated or newly_truncated
        if preview_text or (
            newly_truncated and stream not in self._reported_truncation
        ):
            on_event(
                RawCliStreamEvent(
                    stream=stream,
                    text=preview_text,
                    total_bytes=self._raw_bytes[stream],
                    truncated=self.truncated,
                )
            )
        if newly_truncated:
            self._reported_truncation.add(stream)


def _launch_shell(argv: tuple[str, ...], request: RawCliRequest) -> Any:
    """Launch the fixed outer shell as a non-interactive ordinary process."""
    return subprocess.Popen(
        argv,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(request.initial_directory),
        env=build_scrubbed_environment(),
    )


class _CoalescingOutput:
    """Batch small reads while preserving each stream's byte order."""

    def __init__(self, stream: RawCliStream, output_queue: Any) -> None:
        self._stream = stream
        self._output_queue = output_queue
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._closed = False

    def add(self, payload: bytes) -> None:
        with self._lock:
            self._buffer.extend(payload)
            if len(self._buffer) >= _RAW_OUTPUT_CHUNK_BYTES:
                self._cancel_timer_locked()
                self._flush_locked()
            elif self._timer is None:
                self._timer = threading.Timer(
                    _RAW_OUTPUT_FLUSH_SECONDS,
                    self._flush_on_timer,
                )
                self._timer.daemon = True
                self._timer.start()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._cancel_timer_locked()
            self._flush_locked()

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _flush_on_timer(self) -> None:
        with self._lock:
            self._timer = None
            if not self._closed:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        payload = bytes(self._buffer)
        self._buffer.clear()
        self._output_queue.put(("output", self._stream, payload))


def _drain_pipe(stream: RawCliStream, pipe: BinaryIO, output_queue: Any) -> None:
    """Drain one child pipe completely into bounded IPC chunks."""
    total_bytes = 0
    read = getattr(pipe, "read1", pipe.read)
    coalesced = _CoalescingOutput(stream, output_queue)
    try:
        while True:
            payload = read(_RAW_OUTPUT_CHUNK_BYTES)
            if not payload:
                break
            total_bytes += len(payload)
            coalesced.add(payload)
    finally:
        coalesced.close()
        pipe.close()
        output_queue.put(("stream_end", stream, total_bytes))


def _close_worker_queue(output_queue: Any) -> None:
    """Close the worker's queue handles after its final payload."""
    output_queue.close()
    output_queue.join_thread()


def _raw_cli_worker_entry(
    request: RawCliRequest,
    identity_connection: Any,
    admission_event: Any,
    launch_event: Any,
    abort_event: Any,
    shell_exited_event: Any,
    shell_exit_code: Any,
    output_queue: Any,
) -> None:
    """Enter containment, await admission, then run exactly one shell."""
    terminal: tuple[str, str, int | None, str] | None = None
    shell_process: Any | None = None
    try:
        identity = enter_worker_containment()
        identity_connection.send(identity)
        identity_connection.close()
        while not admission_event.wait(_RAW_QUEUE_POLL_SECONDS):
            if abort_event.is_set():
                return
        while not launch_event.wait(_RAW_QUEUE_POLL_SECONDS):
            if abort_event.is_set():
                return

        try:
            argv = resolve_shell_argv(request.shell, request.command)
        except FileNotFoundError:
            terminal = ("terminal", "shell_unavailable", None, request.shell)
            return

        try:
            shell_process = _launch_shell(argv, request)
        except OSError:
            terminal = ("terminal", "spawn_failed", None, argv[0])
            return

        output_queue.put(("launched", argv[0]))
        assert shell_process.stdout is not None
        assert shell_process.stderr is not None
        readers = [
            threading.Thread(
                target=_drain_pipe,
                args=("stdout", shell_process.stdout, output_queue),
                name="raw-cli-stdout",
            ),
            threading.Thread(
                target=_drain_pipe,
                args=("stderr", shell_process.stderr, output_queue),
                name="raw-cli-stderr",
            ),
        ]
        for reader in readers:
            reader.start()
        exit_code = shell_process.wait()
        shell_exit_code.value = exit_code
        shell_exited_event.set()
        output_queue.put(("terminal", "exited", exit_code, argv[0]))
        for reader in readers:
            reader.join()
    finally:
        try:
            identity_connection.close()
        except OSError:
            pass
        if shell_process is not None:
            for pipe in (shell_process.stdout, shell_process.stderr):
                if pipe is not None and not pipe.closed:
                    pipe.close()
        if terminal is not None:
            output_queue.put(terminal)
        _close_worker_queue(output_queue)


def _process_is_alive(process: Any | None) -> bool:
    if process is None:
        return False
    try:
        return bool(process.is_alive())
    except (AssertionError, ValueError):
        return False


def _stop_process(process: Any | None) -> bool:
    """Boundedly stop a worker that has not produced a containment owner."""
    if not _process_is_alive(process):
        return True
    process.terminate()
    process.join(2.0)
    if _process_is_alive(process):
        process.kill()
        process.join(2.0)
    return not _process_is_alive(process)


def _safe_close(value: Any | None) -> None:
    if value is None:
        return
    try:
        value.close()
    except (OSError, ValueError):
        pass


def _close_parent_queue(output_queue: Any) -> None:
    """Close parent queue handles without waiting on a terminated writer."""
    try:
        output_queue.cancel_join_thread()
    except (AttributeError, OSError, ValueError):
        pass
    try:
        output_queue.close()
    except (OSError, ValueError):
        pass
    _safe_close(getattr(output_queue, "_reader", None))
    _safe_close(getattr(output_queue, "_writer", None))


class _QueueRelay:
    """Confine a possibly corrupt multiprocessing queue read to one daemon."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self._messages: queue.Queue[tuple[Any, ...]] = queue.Queue(
            maxsize=_RAW_OUTPUT_QUEUE_SIZE
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="raw-cli-queue-relay",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def get(self, *, timeout: float) -> tuple[Any, ...]:
        """Read only the safe in-process relay queue."""
        return self._messages.get(timeout=timeout)

    def request_stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float) -> bool:
        """Wait boundedly and report whether the relay actually stopped."""
        self._thread.join(max(0.0, timeout))
        return not self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                message = self._source.get(timeout=_RAW_QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                return
            while not self._stop.is_set():
                try:
                    self._messages.put(message, timeout=_RAW_QUEUE_POLL_SECONDS)
                    break
                except queue.Full:
                    continue


def _request_relay_stop(relay: _QueueRelay | None) -> None:
    """Stop new IPC reads and boundedly settle an ordinary queue read."""
    if relay is None:
        return
    relay.request_stop()
    relay.join(_RAW_QUEUE_POLL_SECONDS * 2)


class _LaunchCommit:
    """Monotonically commit launch and timestamp it inside the runtime lock."""

    def __init__(
        self,
        tree: ExecutorProcessTree,
        launch_event: Any,
        cancel_event: Any,
    ) -> None:
        self._tree = tree
        self._launch_event = launch_event
        self._cancel_event = cancel_event
        self._lock = threading.Lock()
        self._committed = threading.Event()
        self._closed = False
        self._started_at: float | None = None

    def __call__(self) -> float | None:
        with self._lock:
            if self._cancel_event.is_set():
                self._closed = True
                return None
            if self._closed or self._started_at is not None or not self._tree.admitted:
                return None
            self._started_at = time.monotonic()
            self._launch_event.set()
            self._committed.set()
            return self._started_at

    def wait(self, timeout: float) -> bool:
        return self._committed.wait(timeout)

    def settle(self) -> float | None:
        """Atomically honor an existing commit or permanently refuse a later one."""
        with self._lock:
            self._closed = True
            return self._started_at

    def close(self) -> None:
        self.settle()


class _DiskSpoolOwner:
    """Own one anonymous/delete-on-close disk spool through proven close."""

    def __init__(self, file: BinaryIO) -> None:
        self.file = file

    @property
    def closed(self) -> bool:
        return bool(self.file.closed)

    def close(self) -> None:
        for _attempt in range(2):
            if self.closed:
                return
            try:
                self.file.close()
            except OSError:
                pass
        if not self.closed:
            raise OSError("raw CLI output spool cleanup failed") from None


def _open_spool(_max_record_bytes: int) -> _DiskSpoolOwner:
    """Create disk storage that has no durable name before sensitive writes."""
    if os.name != "posix":
        try:
            return _DiskSpoolOwner(tempfile.TemporaryFile(mode="w+b"))
        except OSError:
            raise OSError("raw CLI output spool unavailable") from None

    descriptor: int | None = None
    path: str | None = None
    try:
        descriptor, path = tempfile.mkstemp()
        os.fchmod(descriptor, 0o600)
        try:
            os.unlink(path)
        except OSError:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise OSError("raw CLI output spool unavailable") from None
        file = os.fdopen(descriptor, "w+b", buffering=0)
        descriptor = None
        return _DiskSpoolOwner(file)
    except OSError:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass
        raise OSError("raw CLI output spool unavailable") from None


def _cleanup_tree(tree: ExecutorProcessTree, *, terminate: bool) -> bool:
    """Run bounded tree cleanup and convert missing death proof to ``False``."""
    try:
        return tree.terminate_tree() if terminate else tree.close()
    except Exception:
        return False


class RawShellExecutor:
    """Synchronously execute one admitted, bounded raw shell request."""

    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")

    def execute(
        self,
        request: RawCliRequest,
        *,
        cancel_event: Any,
        on_event: Callable[[RawCliStreamEvent], None],
        admit_worker: RawCliAdmissionCallback,
    ) -> RawCliResult:
        """Execute once; ``admit_worker``'s return value is ignored.

        The callback must call ``tree.admit()`` and then ``commit_launch()`` while
        holding its authority lock. Launch commitment alone signals success.
        """
        request = validate_raw_cli_request(request)
        if not callable(on_event) or not callable(admit_worker):
            raise TypeError("on_event and admit_worker must be callable")

        record_limit = configured_max_record_bytes()
        spool_owner = _open_spool(record_limit)
        accumulator = _OutputAccumulator(spool_owner.file, record_limit)
        process: Any | None = None
        tree: ExecutorProcessTree | None = None
        identity_receive: Any | None = None
        identity_send: Any | None = None
        output_queue: Any | None = None
        output_relay: _QueueRelay | None = None
        abort_event: Any | None = None
        launch_event: Any | None = None
        launch_commit: _LaunchCommit | None = None
        shell_exited_event: Any | None = None
        shell_exit_code: Any | None = None
        started_at: float | None = None
        cleanup_proven = True
        resolved_shell = request.shell
        terminal_state: RawCliTerminalState = "cleanup_unproven"
        exit_code: int | None = None
        try:
            # Textual replaces stderr with a capture whose fileno() is -1.
            # A fresh CPython resource tracker passes that sentinel to
            # fork_exec, so build launch resources under a real fd-backed
            # stream. Serialize the brief redirect because sys.stderr is global.
            with _RAW_CLI_PROCESS_CONSTRUCTION_LOCK:
                with _multiprocessing_stderr_context():
                    identity_receive, identity_send = self._context.Pipe(duplex=False)
                    admission_event = self._context.Event()
                    launch_event = self._context.Event()
                    abort_event = self._context.Event()
                    shell_exited_event = self._context.Event()
                    shell_exit_code = self._context.Value("q", 0)
                    output_queue = self._context.Queue(maxsize=_RAW_OUTPUT_QUEUE_SIZE)
                    process = self._context.Process(
                        target=_raw_cli_worker_entry,
                        args=(
                            request,
                            identity_send,
                            admission_event,
                            launch_event,
                            abort_event,
                            shell_exited_event,
                            shell_exit_code,
                            output_queue,
                        ),
                        name=f"raw-cli-{request.invocation_id}",
                    )
                    try:
                        process.start()
                    except (OSError, RuntimeError):
                        terminal_state = "spawn_failed"
                        cleanup_proven = _stop_process(process)
                        return self._result(
                            request,
                            accumulator,
                            resolved_shell,
                            started_at,
                            exit_code,
                            terminal_state,
                            cleanup_proven,
                        )
            _safe_close(identity_send)
            identity_send = None

            identity = self._receive_identity(
                process,
                identity_receive,
                cancel_event,
            )
            if identity is None:
                terminal_state = (
                    "cancelled" if cancel_event.is_set() else "containment_unavailable"
                )
                cleanup_proven = _stop_process(process)
                return self._result(
                    request,
                    accumulator,
                    resolved_shell,
                    started_at,
                    exit_code,
                    terminal_state,
                    cleanup_proven,
                )

            tree = ExecutorProcessTree(process, admission_event, identity)
            launch_commit = _LaunchCommit(tree, launch_event, cancel_event)
            admission_done = threading.Event()

            def run_admission() -> None:
                try:
                    admit_worker(tree, launch_commit)
                except Exception:
                    pass
                finally:
                    admission_done.set()

            threading.Thread(
                target=run_admission,
                name=f"raw-cli-admission-{request.invocation_id}",
                daemon=True,
            ).start()
            admission_deadline = time.monotonic() + _RAW_STARTUP_TIMEOUT_SECONDS
            while not launch_commit.wait(_RAW_QUEUE_POLL_SECONDS):
                if cancel_event.is_set():
                    terminal_state = "cancelled"
                    break
                if admission_done.is_set() or not _process_is_alive(process):
                    terminal_state = "containment_unavailable"
                    break
                if time.monotonic() >= admission_deadline:
                    terminal_state = "containment_unavailable"
                    break
            started_at = launch_commit.settle()

            if started_at is None:
                launch_commit.close()
                abort_event.set()
                cleanup_proven = (
                    _cleanup_tree(tree, terminate=True)
                    if tree is not None
                    else _stop_process(process)
                )
                return self._result(
                    request,
                    accumulator,
                    resolved_shell,
                    started_at,
                    exit_code,
                    terminal_state,
                    cleanup_proven,
                )

            output_relay = _QueueRelay(output_queue)
            output_relay.start()
            terminal_state, exit_code, resolved_shell, triggered = self._consume(
                request,
                process,
                output_relay,
                accumulator,
                cancel_event,
                on_event,
                started_at,
                resolved_shell,
                shell_exited_event,
                shell_exit_code,
            )
            if triggered:
                _request_relay_stop(output_relay)
                cleanup_proven = _cleanup_tree(tree, terminate=True)
            else:
                process.join(_RAW_QUEUE_POLL_SECONDS * 4)
                _request_relay_stop(output_relay)
                cleanup_proven = _cleanup_tree(tree, terminate=False)
            accumulator.finish_all(on_event)
            return self._result(
                request,
                accumulator,
                resolved_shell,
                started_at,
                exit_code,
                terminal_state,
                cleanup_proven,
            )
        finally:
            if launch_commit is not None:
                launch_commit.close()
            if abort_event is not None:
                abort_event.set()
            _request_relay_stop(output_relay)
            if tree is not None:
                _cleanup_tree(tree, terminate=False)
            elif _process_is_alive(process):
                _stop_process(process)
            _safe_close(identity_receive)
            _safe_close(identity_send)
            if output_queue is not None:
                _close_parent_queue(output_queue)
            if output_relay is not None:
                output_relay.join(_RAW_QUEUE_POLL_SECONDS * 2)
            spool_owner.close()

    @staticmethod
    def _receive_identity(
        process: Any,
        connection: Any,
        cancel_event: Any,
    ) -> WorkerContainmentIdentity | None:
        empty_polls = 0
        while empty_polls * _RAW_QUEUE_POLL_SECONDS < _RAW_STARTUP_TIMEOUT_SECONDS:
            if connection.poll(_RAW_QUEUE_POLL_SECONDS):
                try:
                    identity = connection.recv()
                except EOFError:
                    return None
                if (
                    type(identity) is WorkerContainmentIdentity
                    and identity.pid == process.pid
                ):
                    return identity
                return None
            if cancel_event.is_set() or not _process_is_alive(process):
                return None
            empty_polls += 1
        return None

    @staticmethod
    def _consume(
        request: RawCliRequest,
        process: Any,
        messages: Any,
        accumulator: _OutputAccumulator,
        cancel_event: Any,
        on_event: Callable[[RawCliStreamEvent], None],
        started_at: float,
        resolved_shell: str,
        shell_exited_event: Any,
        shell_exit_code: Any,
    ) -> tuple[RawCliTerminalState, int | None, str, bool]:
        dead_empty_polls = 0
        ended_streams: set[RawCliStream] = set()
        exited_at: float | None = None
        exited_code: int | None = None
        exited_terminal_seen = False
        while True:
            message: tuple[Any, ...] | None = None
            try:
                message = messages.get(timeout=_RAW_QUEUE_POLL_SECONDS)
                dead_empty_polls = 0
            except queue.Empty:
                if not _process_is_alive(process):
                    dead_empty_polls += 1

            if message is not None:
                kind = message[0]
                if kind == "output":
                    accumulator.consume(message[1], message[2], on_event)
                elif kind == "stream_end":
                    accumulator.finish(message[1], on_event)
                    ended_streams.add(message[1])
                elif kind == "launched":
                    resolved_shell = str(message[1])
                elif kind == "terminal":
                    if message[1] != "exited":
                        return message[1], message[2], str(message[3]), False
                    exited_terminal_seen = True
                    exited_code = int(message[2])
                    resolved_shell = str(message[3])

            if exited_at is None and shell_exited_event.is_set():
                exited_code = int(shell_exit_code.value)
                exited_at = time.monotonic()
            if exited_at is not None:
                if exited_terminal_seen and ended_streams == {"stdout", "stderr"}:
                    return "exited", exited_code, resolved_shell, False
                if time.monotonic() - exited_at >= _RAW_POST_EXIT_DRAIN_SECONDS:
                    if ended_streams != {"stdout", "stderr"}:
                        accumulator.truncated = True
                    return "exited", exited_code, resolved_shell, True

            if dead_empty_polls >= 4 and exited_at is None:
                return "cleanup_unproven", None, resolved_shell, False

            if not _process_is_alive(process):
                continue
            if cancel_event.is_set() and not shell_exited_event.is_set():
                return "cancelled", None, resolved_shell, True
            if (
                time.monotonic() - started_at >= request.timeout_seconds
                and not shell_exited_event.is_set()
            ):
                return "timed_out", None, resolved_shell, True

    @staticmethod
    def _result(
        request: RawCliRequest,
        accumulator: _OutputAccumulator,
        resolved_shell: str,
        started_at: float | None,
        exit_code: int | None,
        terminal_state: RawCliTerminalState,
        cleanup_proven: bool,
    ) -> RawCliResult:
        elapsed = 0.0 if started_at is None else max(0.0, time.monotonic() - started_at)
        return RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell=resolved_shell,
            initial_directory=request.initial_directory,
            elapsed_seconds=elapsed,
            stdout_preview=accumulator.preview("stdout"),
            stderr_preview=accumulator.preview("stderr"),
            record_output=accumulator.record_output(),
            exit_code=exit_code,
            terminal_state=terminal_state,
            truncated=accumulator.truncated,
            cleanup_proven=cleanup_proven,
        )
