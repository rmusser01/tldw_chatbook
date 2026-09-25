"""Loopback executor for the remote workspace worker bundle (Phase 1e).

Task 9's harness executes the COMMITTED bundle — not the local worker
module — exactly the way the ssh transport will (Task 11):
``python -I -c '<bootstrap>'`` with the zlib-compressed bundle followed
by the request JSON on stdin. Loopback swaps only the process spawn
(a local interpreter instead of ``ssh host python3 -I -c ...``), so
every caller of this harness exercises the real remote code path:
bootstrap, decompress, exec, magic-prefixed frames, and the two-line
response contract ``_parse_worker_output`` established for the local
worker.

Pinned behaviours:

* The bootstrap string is fixed by the spec and charset-asserted by
  ``Tests/Tools/test_remote_executor_loopback.py`` (letters, digits,
  space and ``_ . , ( ) " ; : < > = + % \\ [ ] -`` — no shell-active
  character; the hyphen is required by the spec's own
  ``tldw-worker`` marker).
* ``RESPONSE_MAGIC`` is IMPORTED from the bundle artifact — one
  definition site — and pinned against its literal by test; it is never
  duplicated silently.
* Noise: stdout is scanned at the BYTE level for the magic (never
  line-based), so noise containing newlines cannot inject fake response
  lines; magicless output is the typed ``stdout_noise`` error.
* ``ping`` is the bootstrap probe (dispatched before the root pin): its
  captured identity chain is the source for every other operation's
  pinned request, and its ``bundle_sha256`` echo is verified against the
  committed artifact's stamp.

SSH transport mode (Phase 2e, Task 14): ``RemoteWorkspaceToolExecutor.
for_ssh`` wires the same exchange onto Task 11's transport — the status
cache supplies identity and the per-call client guard, the master
manager owns connection health, a per-host semaphore (shared across
bindings, ``max_concurrent_calls``) bounds concurrent clients, every
call carries the REMAINING budget, and the debounced recovery probe
re-admits a degraded binding in the background. Transport-stack imports
stay inside the ssh entry points: the transport imports this module (for
``_bundle_payload``), so module-level imports would be circular.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import uuid
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Tools.build_remote_worker_bundle import expected_bundle_stamp
from tldw_chatbook.Tools.remote_worker_bundle import RESPONSE_MAGIC
from tldw_chatbook.Tools.workspace_tool_executor import (
    WORKSPACE_HELPER_TIMEOUT_SECONDS,
    WorkspaceToolExecutionError,
    _parse_worker_output,
)
from tldw_chatbook.Tools.workspace_tool_protocol import (
    MAX_RESPONSE_BYTES,
    WorkspaceProtocolError,
    WorkspaceToolRequest,
    WorkspaceToolResponse,
)
from tldw_chatbook.Tools.workspace_wire_decode import WIRE_VERSION

if TYPE_CHECKING:  # pragma: no cover - import-cycle guard (see for_ssh)
    from tldw_chatbook.Tools.remote_binding_locator import RemoteLocator
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache
    from tldw_chatbook.Tools.remote_workspace_transport import (
        RemoteCallResult,
        RemoteWorkspaceTransport,
        SshMasterManager,
    )

#: Committed bundle artifact executed by the loopback spawn.
_BUNDLE_PATH = Path(__file__).resolve().parent / "remote_worker_bundle.py"

#: Leading stdout garbage tolerated before the first response magic.
_NOISE_GARBAGE_CAP = 4 * 1024

#: Hard capture ceiling for one exchange (response ceiling plus noise
#: headroom, mirroring the local executor's bounded readers).
_STDOUT_CAPTURE_CAP = MAX_RESPONSE_BYTES + (64 * 1024)

_DEFAULT_BUDGET_SECONDS = 30.0

#: The spec's fixed bootstrap (N = compressed bundle byte length). The
#: version gate runs BEFORE the bundle is compiled so an old remote
#: python exits 76 with its found version on stderr instead of dying in
#: a bare ``SyntaxError``; the charset test pins this string verbatim.
_BOOTSTRAP_TEMPLATE = (
    "import sys,zlib;exec(compile(zlib.decompress("
    "sys.stdin.buffer.read({size})),\"b\",\"exec\"))"
    "if sys.version_info>=(3,10)else(sys.stderr.write("
    '"tldw-worker:python3.10+:found:%d.%d\\n"%sys.version_info[:2]),'
    "sys.exit(76))"
)

#: Identity values a first-contact ping request carries. The worker
#: ignores them for ping (it captures fresh); they exist only because
#: the request schema requires identity fields.
_PING_PROBE_IDENTITY: dict[str, Any] = {
    "device": 0,
    "inode": 0,
    "mode": 0,
    "reparse": False,
}

_RESPONSE_STAMP_TAIL = re.compile(r"\nsha256: ([0-9a-f]{64})\nsize: ([0-9]+)\Z")

_PING_RESULT_KEYS = frozenset(
    {"identity_chain", "canonical_path", "python_version", "bundle_sha256"}
)

#: Remaining-budget floor: an executor whose whole budget was consumed
#: before the spawn still sends a (tiny) positive budget — the wire
#: requires a positive int and the deadlines require a positive float.
_MIN_TRANSPORT_BUDGET_SECONDS = 1.0


class RemoteWorkspaceLoopbackError(RuntimeError):
    """The loopback harness could not complete one bundle exchange."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(
            message if message is not None else f"remote loopback failed ({code})"
        )


class RemoteWorkspaceExecutionError(RuntimeError):
    """A worker-reported failure, carrying whether admission preceded it.

    ``admitted`` is the taxonomy bit the status cache reads (Task 13):
    failures with NO admitted marker are setup/transport-class (the
    worker never accepted the root); failures after the marker are
    typed operation errors that must not flip a binding's status.
    """

    def __init__(
        self, code: str, message: str | None = None, *, admitted: bool
    ) -> None:
        self.code = code
        self.admitted = admitted
        super().__init__(
            message if message is not None else f"remote op failed ({code})"
        )


def bootstrap_source(compressed_size: int) -> str:
    """Return the fixed bootstrap with ``compressed_size`` embedded."""
    if type(compressed_size) is not int or compressed_size <= 0:
        raise ValueError("compressed bundle size must be a positive int")
    return _BOOTSTRAP_TEMPLATE.format(size=compressed_size)


def parse_fs_read_stamps(result: str) -> tuple[str, int] | None:
    """Extract the worker-reported CAS stamps from one fs_read result.

    Args:
        result: The fs_read response's result string.

    Returns:
        ``(sha256_hex, size_bytes)`` from the result's final two lines,
        or ``None`` when the stamp tail is absent.
    """
    match = _RESPONSE_STAMP_TAIL.search(result)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


@lru_cache(maxsize=1)
def _bundle_payload() -> tuple[bytes, bytes, str]:
    """The artifact bytes, their zlib payload, and the bootstrap source."""
    bundle = _BUNDLE_PATH.read_bytes()
    compressed = zlib.compress(bundle)
    return bundle, compressed, bootstrap_source(len(compressed))


def _spawn_loopback_worker(
    payload: bytes, *, budget_seconds: float
) -> subprocess.CompletedProcess[bytes]:
    """Spawn ``python -I -c <bootstrap>`` and feed it bundle + request.

    This is the LOOPBACK transport: the only seam Task 11 swaps (the
    ssh transport spawns ``ssh ... python3 -I -c <bootstrap>`` with the
    same stdin payload instead).
    """
    _bundle, compressed, bootstrap = _bundle_payload()
    argv = [sys.executable, "-I", "-c", bootstrap]
    try:
        return subprocess.run(
            argv,
            input=compressed + payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=budget_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RemoteWorkspaceLoopbackError("loopback_timeout") from error


def _magic_frame_segments(raw: bytes) -> list[bytes]:
    """Byte-level magic scan: strip noise, return bare frame lines.

    Never line-based: the scan finds ``RESPONSE_MAGIC`` anywhere in the
    raw bytes, so noise containing newlines cannot inject fake response
    lines. Every emitted frame carries its own magic prefix (the bundle
    IO adapter prefixes each write), so all of them are stripped here.
    """
    if RESPONSE_MAGIC not in raw:
        raise RemoteWorkspaceLoopbackError("stdout_noise")
    noise, *segments = raw.split(RESPONSE_MAGIC)
    if len(noise) > _NOISE_GARBAGE_CAP:
        raise RemoteWorkspaceLoopbackError("stdout_noise")
    frames: list[bytes] = []
    for segment in segments:
        frame = segment[:-1] if segment.endswith(b"\n") else segment
        if not frame or b"\n" in frame:
            raise RemoteWorkspaceLoopbackError("protocol_failure")
        frames.append(frame)
    return frames


def _encode_request(request: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            request,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise RemoteWorkspaceLoopbackError("invalid_request") from error


def _require_exchange_inputs(root: Path, request: Mapping[str, Any]) -> str:
    """Validate the harness preconditions; return the expected operation id."""
    if not root.is_dir():
        raise RemoteWorkspaceLoopbackError("loopback_root_missing")
    if not isinstance(request, Mapping) or not isinstance(
        request.get("operation_id"), str
    ):
        raise RemoteWorkspaceLoopbackError("invalid_request")
    return request["operation_id"]


def _exchange(
    request_bytes: bytes,
    *,
    budget_seconds: float,
    expected_operation_id: str,
    spawn: Callable[..., Any],
) -> tuple[list[WorkspaceToolResponse], int]:
    """Run one exchange and parse frames per the two-line contract."""
    completed = spawn(request_bytes, budget_seconds=budget_seconds)
    if len(completed.stdout) > _STDOUT_CAPTURE_CAP:
        raise RemoteWorkspaceLoopbackError("protocol_failure")
    frames = _magic_frame_segments(completed.stdout)
    joined = b"".join(frame + b"\n" for frame in frames)
    try:
        # The established contract: 1-2 frames, admitted shape, terminal
        # outcome/code — everything the local executor enforces.
        _parse_worker_output(
            joined, expected_operation_id=expected_operation_id
        )
        responses = [
            WorkspaceToolResponse.from_bytes(
                frame, expected_operation_id=expected_operation_id
            )
            for frame in frames
        ]
    except (WorkspaceToolExecutionError, WorkspaceProtocolError):
        raise RemoteWorkspaceLoopbackError("protocol_failure") from None
    return responses, completed.returncode


def _wire_dict(response: WorkspaceToolResponse) -> dict[str, Any]:
    """Render one parsed response as its wire-frame dict form."""
    return {
        "version": WIRE_VERSION,
        "operation_id": response.operation_id,
        "outcome": response.outcome,
        "code": response.code,
        "result": response.result,
        "error": response.error,
        "elapsed_ms": response.elapsed_ms,
        "truncated": response.truncated,
        "cleanup_proven": response.cleanup_proven,
    }


def run_bundle_loopback_frames(
    root: Path,
    request: dict[str, Any],
    *,
    budget_seconds: float = _DEFAULT_BUDGET_SECONDS,
) -> tuple[dict[str, Any], ...]:
    """Execute the bundle once and return EVERY response frame as a dict.

    The frames-level view exists because the admitted marker is the
    protocol's own transport-failure classifier: a pin failure emits
    exactly ONE failure frame (no admitted marker), a dispatched
    operation emits admitted-then-terminal. Tests and the status cache
    (Task 13) read that distinction here.
    """
    operation_id = _require_exchange_inputs(root, request)
    responses, _returncode = _exchange(
        _encode_request(request),
        budget_seconds=budget_seconds,
        expected_operation_id=operation_id,
        spawn=_spawn_loopback_worker,
    )
    return tuple(_wire_dict(response) for response in responses)


def run_bundle_loopback(
    root: Path,
    request: dict[str, Any],
    *,
    budget_seconds: float = _DEFAULT_BUDGET_SECONDS,
) -> dict[str, Any]:
    """Execute the bundle once and return the FINAL response frame dict.

    Raises:
        RemoteWorkspaceLoopbackError: On harness-level failures (missing
            root, timeout, magicless/noisy stdout, contract violations).
            A worker-refused operation is NOT a harness failure: its
            failure frame is returned like any other final frame.
    """
    return run_bundle_loopback_frames(
        root, request, budget_seconds=budget_seconds
    )[-1]


# ---------------------------------------------------------------------------
# SSH transport mode (Phase 2e, Task 14)
# ---------------------------------------------------------------------------

#: Per-host call semaphores, shared across every executor in this
#: process for the same resolved host identity — sshd's ``MaxSessions``
#: is per connection and multiple bindings on one host share one master,
#: so the cap is keyed by host, never per binding (spec: "Concurrency").
#: A request whose cap differs from the registered one replaces the
#: semaphore: holders of the old one simply finish, and in production
#: the cap is read once per run so the replace path is inert.
_HOST_SEMAPHORE_LOCK = threading.Lock()
_HOST_SEMAPHORES: dict[
    tuple[str | None, str, int | None],
    tuple[int, "threading.Semaphore"],
] = {}


def _host_key_tuple(
    loc: "RemoteLocator",
) -> tuple[str | None, str, int | None]:
    """The resolved host identity — the master manager's own key."""
    return (loc.user, loc.host, loc.port)


def _host_key_string(loc: "RemoteLocator") -> str:
    """The same identity rendered as the status cache's probe key."""
    user = loc.user if loc.user is not None else ""
    port = loc.port if loc.port is not None else 22
    return f"{user}@{loc.host}:{port}"


def _host_semaphore(
    key: tuple[str | None, str, int | None], max_concurrent: int
) -> "threading.Semaphore":
    with _HOST_SEMAPHORE_LOCK:
        registered = _HOST_SEMAPHORES.get(key)
        if registered is not None and registered[0] == max_concurrent:
            return registered[1]
        semaphore = threading.Semaphore(max_concurrent)
        _HOST_SEMAPHORES[key] = (max_concurrent, semaphore)
        return semaphore


@dataclass
class _SshModeConfig:
    """Everything the ssh transport mode wires around one binding."""

    loc: "RemoteLocator"
    binding_id: str
    cache: "RemoteBindingStatusCache"
    masters: "SshMasterManager"
    transport: "RemoteWorkspaceTransport"
    python: str
    max_concurrent_calls: int
    recovery_probes: bool


def _ssh_identity_source_stub() -> Any:
    """Never called: ssh mode resolves identity from the status cache."""
    raise RuntimeError(
        "ssh mode resolves identity through the binding status cache"
    )


class RemoteWorkspaceToolExecutor:
    """Remote workspace executor: loopback harness AND the ssh transport.

    ``execute`` slots into the same call surface as
    ``WorkspaceToolExecutor`` so provider dispatch stays
    transport-agnostic; Task 15 types its result into
    ``RunAdmittedWorkspaceRoot``. The loopback constructor (Task 9)
    stays untouched; :meth:`for_ssh` (Task 14) attaches the transport
    pieces — status cache, master manager, per-host call cap — and both
    modes share every request-building/parsing rule above the spawn
    seam.
    """

    def __init__(
        self,
        root: Path,
        *,
        root_locator: str,
        identity_chain_source: Callable[[], Mapping[str, Any]],
        budget_seconds: float = _DEFAULT_BUDGET_SECONDS,
    ) -> None:
        """Configure one executor around one (loopback) workspace root.

        Args:
            root: The workspace root this executor drives (loopback: a
                local directory standing in for the remote one).
            root_locator: The locator string ping requests carry (the
                ssh locator's path component in remote mode).
            identity_chain_source: Zero-arg callable returning the ping
                payload (``identity_chain`` + ``canonical_path``) whose
                captured identities pin every non-ping request — over
                ssh this is exactly "the last ping's answer".
            budget_seconds: Per-exchange deadline for the spawn.
        """
        self._root = Path(root)
        self._root_locator = root_locator
        self._identity_chain_source = identity_chain_source
        self._budget_seconds = budget_seconds
        #: SSH transport mode (Task 14); ``None`` keeps the loopback
        #: behaviour Tasks 9/12 pinned.
        self._ssh: _SshModeConfig | None = None

    # -- construction: ssh mode ---------------------------------------------

    @classmethod
    def for_ssh(
        cls,
        loc: "RemoteLocator",
        binding_id: str,
        *,
        cache: "RemoteBindingStatusCache",
        masters: "SshMasterManager | None" = None,
        ssh_bin: str = "ssh",
        python: str = "python3",
        grace: float = 5.0,
        budget_seconds: float = WORKSPACE_HELPER_TIMEOUT_SECONDS,
        max_concurrent_calls: int | None = None,
        recovery_probes: bool = True,
    ) -> "RemoteWorkspaceToolExecutor":
        """Build the executor that drives one remote binding over ssh.

        Imports of the transport stack are deferred to THIS method on
        purpose: ``remote_workspace_transport`` imports this module (for
        ``_bundle_payload``), and the status cache imports the transport
        — a module-level import here would be circular.

        Args:
            loc: The binding's validated locator; argv is rebuilt from
                its parsed parts inside the transport.
            binding_id: The registry binding's identifier — the status
                cache key for every recorded state transition.
            cache: The shared :class:`RemoteBindingStatusCache`
                (identity source, guard states, probe debounce).
            masters: The ControlMaster lifecycle manager; ``None``
                constructs one around ``ssh_bin`` with defaults (the
                app passes the configured singleton instead).
            ssh_bin: The ssh binary for the default manager above; an
                explicitly supplied manager owns its own binary and
                this value is unused.
            python: The remote interpreter the call spawns
                (``<python> -I -c <bootstrap>``).
            grace: Deadline slack on both transport deadlines (spec
                default 5.0); a knob so tests can tighten it.
            budget_seconds: The per-exchange budget the remaining-budget
                rule subtracts elapsed time from (spec default
                ``WORKSPACE_HELPER_TIMEOUT_SECONDS``).
            max_concurrent_calls: Per-host cap on in-flight ssh calls;
                ``None`` reads ``[console_ssh] max_concurrent_calls``.
            recovery_probes: ``False`` disables the background recovery
                probe dispatch (tests drive :meth:`ping` directly for
                determinism).

        Returns:
            An executor in ssh mode; :meth:`execute` and :meth:`ping`
            route through the transport, the status cache, the master
            manager, and the per-host semaphore.

        Raises:
            ValueError: On non-positive budget/grace/cap values or an
                unquotable interpreter name.
        """
        from tldw_chatbook.Tools.remote_workspace_transport import (
            RemoteWorkspaceTransport,
            SshMasterManager,
        )

        if masters is None:
            masters = SshMasterManager(ssh_bin=ssh_bin)
        if python.startswith("-") or any(char.isspace() for char in python):
            raise ValueError(
                f"python must be a bare interpreter name: {python!r}"
            )
        if budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive")
        if max_concurrent_calls is None:
            from tldw_chatbook.config import get_console_ssh_settings

            max_concurrent_calls = get_console_ssh_settings().max_concurrent_calls
        if max_concurrent_calls < 1:
            raise ValueError("max_concurrent_calls must be at least 1")

        executor = cls(
            Path(str(loc.path)),
            root_locator=str(loc.path),
            identity_chain_source=_ssh_identity_source_stub,
            budget_seconds=budget_seconds,
        )
        executor._ssh = _SshModeConfig(
            loc=loc,
            binding_id=binding_id,
            cache=cache,
            masters=masters,
            transport=RemoteWorkspaceTransport(masters, grace_seconds=grace),
            python=python,
            max_concurrent_calls=int(max_concurrent_calls),
            recovery_probes=recovery_probes,
        )
        return executor

    @property
    def max_concurrent_calls(self) -> int | None:
        """The per-host ssh call cap (ssh mode); ``None`` in loopback."""
        return None if self._ssh is None else self._ssh.max_concurrent_calls

    # -- transport seam ----------------------------------------------------

    def _spawn_worker(
        self, payload: bytes, *, budget_seconds: float
    ) -> subprocess.CompletedProcess[bytes]:
        """TRANSPORT SEAM — loopback spawns a local interpreter.

        Task 11 swaps this single method for the ssh transport spawn
        (``ssh ... python3 -I -c <bootstrap>``); everything above it in
        the exchange is transport-agnostic.
        """
        return _spawn_loopback_worker(payload, budget_seconds=budget_seconds)

    # -- public surface ----------------------------------------------------

    def execute(
        self, tool: str, args: dict[str, Any], *, intent: str
    ) -> dict[str, Any]:
        """Execute one tool through the bundle; return the final frame dict.

        Raises:
            RemoteWorkspaceExecutionError: For request admission
                failures (``admitted=False``) and worker-reported
                operation failures (``admitted=True``) — the taxonomy
                bit the status cache consumes. In ssh mode this also
                covers the per-call client guard (BLOCKED/MISSING fail
                fast without spawning) and typed transport failures.
            RemoteWorkspaceLoopbackError: For harness-level failures
                (loopback mode only).
        """
        if self._ssh is not None:
            return self._execute_ssh(tool, args, intent=intent)
        chain = None if tool == "ping" else self._identity_chain_source()
        request = self._build_request(tool, args, intent=intent, chain=chain)
        responses, returncode = self._exchange_request(request)
        admitted = len(responses) == 2
        terminal = responses[-1]
        if terminal.outcome == "failure":
            raise RemoteWorkspaceExecutionError(
                terminal.code, terminal.error, admitted=admitted
            )
        if returncode != 0 or terminal.result is None:
            raise RemoteWorkspaceExecutionError(
                "protocol_failure", admitted=admitted
            )
        return _wire_dict(terminal)

    def ping(self) -> dict[str, Any]:
        """Probe the root: full identity chain, canonical path, versions.

        In ssh mode this is THE recovery/status probe: it bypasses the
        per-call client guard (its whole purpose is to flip a BLOCKED
        binding back), classifies its own result into the status cache,
        and re-captures the identity chain on success. A probe whose
        root would not capture records STALE_IDENTITY, never BLOCKED —
        the committed worker answers both an absent and a rejected root
        with the single ``root_pin_failed`` refusal (Task 13 finding),
        so MISSING stays reserved until a worker distinguishes them and
        the next operation's re-capture decides what the stale chain
        meant.

        Returns:
            The ping payload: ``{"identity_chain": [[path, dev, ino,
            mode], ...], "canonical_path": str, "python_version":
            "3.x.y", "bundle_sha256": str}``.

        Raises:
            RemoteWorkspaceExecutionError: When the root cannot be
                captured (``root_pin_failed``, no admitted marker), the
                payload is malformed, or the echoed bundle stamp does
                not match the committed artifact.
        """
        if self._ssh is not None:
            payload, _terminal = self._ssh_ping(time.monotonic())
            return payload
        request = self._build_request("ping", {}, intent="read", chain=None)
        responses, _returncode = self._exchange_request(request)
        admitted = len(responses) == 2
        terminal = responses[-1]
        if terminal.outcome == "failure":
            raise RemoteWorkspaceExecutionError(
                terminal.code, terminal.error, admitted=admitted
            )
        try:
            payload = json.loads(terminal.result or "")
        except ValueError:
            raise RemoteWorkspaceExecutionError(
                "protocol_failure", admitted=admitted
            ) from None
        _require_ping_payload(payload)
        artifact, _compressed, _bootstrap = _bundle_payload()
        if payload["bundle_sha256"] != expected_bundle_stamp(artifact):
            raise RemoteWorkspaceExecutionError(
                "bundle_mismatch", admitted=admitted
            )
        return payload

    # -- internals ----------------------------------------------------------

    def _exchange_request(
        self, request: Mapping[str, Any]
    ) -> tuple[list[WorkspaceToolResponse], int]:
        operation_id = _require_exchange_inputs(self._root, request)
        return _exchange(
            _encode_request(request),
            budget_seconds=self._budget_seconds,
            expected_operation_id=operation_id,
            spawn=self._spawn_worker,
        )

    def _build_request(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        intent: str,
        chain: Mapping[str, Any] | None,
        timeout_seconds: int = WORKSPACE_HELPER_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Build one wire-legal request dict, validating it as the parent.

        ``timeout_seconds`` is the budget the worker's watchdog arms
        from; ssh mode passes the remaining budget so the server-side
        clock starts only after connect and transfer.
        """
        try:
            if chain is None:
                # First-contact ping: identity fields are schema filler —
                # the worker captures fresh and never compares them.
                locator = self._root_locator
                identities = [dict(_PING_PROBE_IDENTITY)]
            else:
                identities = [
                    {
                        "device": entry[1],
                        "inode": entry[2],
                        "mode": entry[3],
                        "reparse": False,
                    }
                    for entry in chain["identity_chain"]
                ]
                locator = chain["canonical_path"]
            request = {
                "version": WIRE_VERSION,
                "operation_id": uuid.uuid4().hex,
                "operation": tool,
                "intent": intent,
                "root_locator": locator,
                "root_identity": identities[0],
                "ancestor_identities": identities,
                "arguments": dict(args),
                # The worker's watchdog budget: over ssh this is the
                # REMAINING budget (spec: "Worker-side watchdog"), so
                # the server clock starts only after the transfer.
                "timeout_seconds": timeout_seconds,
                "output_max_bytes": MAX_RESPONSE_BYTES,
            }
            WorkspaceToolRequest.from_bytes(_encode_request(request))
        except (
            WorkspaceProtocolError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            RemoteWorkspaceLoopbackError,
        ):
            raise RemoteWorkspaceExecutionError(
                "invalid_request", admitted=False
            ) from None
        return request

    # -- ssh mode internals (Task 14) ----------------------------------------

    def _execute_ssh(
        self, tool: str, args: dict[str, Any], *, intent: str
    ) -> dict[str, Any]:
        """Run one operation over the ssh transport, cache-guarded.

        Order matters: the per-call client guard runs BEFORE anything
        spawns (BLOCKED/MISSING cost zero subprocesses); a cold root
        pings first to capture identity (two calls, one thereafter);
        every transport call carries the REMAINING budget.
        """
        from tldw_chatbook.Tools.remote_binding_status import BindingState

        cfg = self._ssh
        assert cfg is not None  # routed here only in ssh mode
        status = cfg.cache.status(cfg.binding_id)
        if status.state in (BindingState.BLOCKED, BindingState.MISSING):
            # Fail fast, no spawn; a degraded binding still deserves the
            # debounced recovery probe (the hot path only schedules it).
            self.maybe_schedule_recovery_probe()
            raise RemoteWorkspaceExecutionError(
                "binding_blocked"
                if status.state is BindingState.BLOCKED
                else "binding_missing",
                status.reason,
                admitted=False,
            )
        started = time.monotonic()
        if tool == "ping":
            _payload, terminal = self._ssh_ping(started)
            return _wire_dict(terminal)
        chain = cfg.cache.identity_for(cfg.binding_id)
        if not chain:
            # Cold root: the ping captures the identity chain (and
            # flips the cache READY) before the pinned op runs.
            self._ssh_ping(started)
            chain = cfg.cache.identity_for(cfg.binding_id)
            if not chain:
                raise RemoteWorkspaceExecutionError(
                    "protocol_failure", admitted=False
                )
        remaining = self._remaining_budget(started)
        request = self._build_request(
            tool,
            args,
            intent=intent,
            # The chain's root entry IS the canonical path the ping
            # captured — the locator every later pin must reuse.
            chain={"identity_chain": chain, "canonical_path": chain[0][0]},
            timeout_seconds=max(1, int(remaining)),
        )
        result = self._ssh_call(_encode_request(request), budget=remaining)
        return self._map_ssh_result(result, request["operation_id"])

    def _ssh_ping(
        self, started: float
    ) -> tuple[dict[str, Any], WorkspaceToolResponse]:
        """Run the bootstrap probe over ssh and record its classification.

        The single round trip powering status refresh and identity
        (re-)capture: success flips the cache READY with the fresh
        chain; a framed ``root_pin_failed`` records STALE_IDENTITY (the
        worker does not distinguish a missing root from a rejected one,
        so the next operation's re-capture decides and MISSING stays
        reserved — see :meth:`ping`); transport failures record their
        taxonomy row; state-preserving outcomes touch nothing.
        """
        from tldw_chatbook.Tools.remote_binding_status import (
            PROBE_PIN_FAILED,
            classify_probe_result,
        )

        cfg = self._ssh
        assert cfg is not None
        remaining = self._remaining_budget(started)
        request = self._build_request(
            "ping",
            {},
            intent="read",
            chain=None,
            timeout_seconds=max(1, int(remaining)),
        )
        result = self._ssh_call(_encode_request(request), budget=remaining)
        if classify_probe_result(result) == PROBE_PIN_FAILED:
            cfg.cache.record_pin_failure(cfg.binding_id)
            raise RemoteWorkspaceExecutionError(
                "root_pin_failed",
                "worker refused the pinned root identity",
                admitted=False,
            )
        if result.failure is not None:
            cfg.cache.record_transport_failure(
                cfg.binding_id, result.failure.kind, result.failure.reason
            )
            raise RemoteWorkspaceExecutionError(
                result.failure.kind.value,
                result.failure.reason,
                admitted=result.admitted,
            )
        if result.response is None:  # contract-impossible; stay conservative
            raise RemoteWorkspaceExecutionError(
                "protocol_failure", admitted=result.admitted
            )
        terminal = self._parse_terminal(
            result.response, request["operation_id"], admitted=result.admitted
        )
        if terminal.outcome != "success":
            # An unclassifiable probe frame preserves all cache state.
            raise RemoteWorkspaceExecutionError(
                terminal.code, terminal.error, admitted=result.admitted
            )
        try:
            payload = json.loads(terminal.result or "")
        except ValueError:
            raise RemoteWorkspaceExecutionError(
                "protocol_failure", admitted=result.admitted
            ) from None
        _require_ping_payload(payload)
        artifact, _compressed, _bootstrap = _bundle_payload()
        if payload["bundle_sha256"] != expected_bundle_stamp(artifact):
            raise RemoteWorkspaceExecutionError(
                "bundle_mismatch", admitted=False
            )
        cfg.cache.record_success(
            cfg.binding_id, identity_chain=payload["identity_chain"]
        )
        return payload, terminal

    def _ssh_call(self, request_bytes: bytes, *, budget: float) -> "RemoteCallResult":
        """One transport call under the master and the host cap.

        ``ensure_master`` is a cheap no-op while the tracked socket
        lives; the per-host semaphore bounds concurrent ssh clients
        (shared across bindings on one host); the transport never
        retries — recovery is the status cache and the probe's job.
        """
        cfg = self._ssh
        assert cfg is not None
        cfg.masters.ensure_master(cfg.loc)
        semaphore = _host_semaphore(
            _host_key_tuple(cfg.loc), cfg.max_concurrent_calls
        )
        with semaphore:
            return cfg.transport.call(
                cfg.loc, request_bytes, budget=budget, python=cfg.python
            )

    def _map_ssh_result(
        self, result: "RemoteCallResult", operation_id: str
    ) -> dict[str, Any]:
        """Fold one transport outcome into cache state and a typed result.

        Transport failures record their taxonomy row (only the
        BLOCKING kinds flip the state — Task 13's rule) and raise with
        the admitted bit preserved; a framed ``root_pin_failed`` with no
        admitted marker records STALE_IDENTITY and never BLOCKED; a
        delivered final frame parses through the same two-frame
        contract the loopback enforces; success flips the cache READY
        without touching the identity chain (the ping owns capture).
        """
        from tldw_chatbook.Tools.remote_binding_status import (
            _ROOT_PIN_FAILED_CODE,
            BLOCKING_TRANSPORT_KINDS,
        )

        cfg = self._ssh
        assert cfg is not None
        if result.failure is not None:
            cfg.cache.record_transport_failure(
                cfg.binding_id, result.failure.kind, result.failure.reason
            )
            if result.failure.kind in BLOCKING_TRANSPORT_KINDS:
                self.maybe_schedule_recovery_probe()
            raise RemoteWorkspaceExecutionError(
                result.failure.kind.value,
                result.failure.reason,
                admitted=result.admitted,
            )
        if result.response is None:  # contract-impossible; stay conservative
            raise RemoteWorkspaceExecutionError(
                "protocol_failure", admitted=result.admitted
            )
        terminal = self._parse_terminal(
            result.response, operation_id, admitted=result.admitted
        )
        if terminal.outcome == "failure":
            if not result.admitted and terminal.code == _ROOT_PIN_FAILED_CODE:
                # The host answered; the identity did not pin. The next
                # operation's re-capture decides what the stale chain
                # meant — never a BLOCKED flip (spec: "Identity
                # freshness").
                cfg.cache.record_pin_failure(cfg.binding_id)
                self.maybe_schedule_recovery_probe()
            raise RemoteWorkspaceExecutionError(
                terminal.code, terminal.error, admitted=result.admitted
            )
        if not result.admitted or terminal.result is None:
            # A dispatched operation must have emitted the admitted
            # marker before its success frame, and success carries a
            # result — anything else violates the two-frame contract.
            raise RemoteWorkspaceExecutionError(
                "protocol_failure", admitted=result.admitted
            )
        cfg.cache.record_success(cfg.binding_id)
        return _wire_dict(terminal)

    def _parse_terminal(
        self, frame: bytes, operation_id: str, *, admitted: bool
    ) -> WorkspaceToolResponse:
        """Parse the transport's terminal frame under the shared contract."""
        try:
            return _parse_worker_output(frame, expected_operation_id=operation_id)
        except WorkspaceToolExecutionError:
            raise RemoteWorkspaceExecutionError(
                "protocol_failure", admitted=admitted
            ) from None

    def _remaining_budget(self, started: float) -> float:
        """``budget_seconds - elapsed`` since ``started``, floored at 1s."""
        return max(
            _MIN_TRANSPORT_BUDGET_SECONDS,
            self._budget_seconds - (time.monotonic() - started),
        )

    def maybe_schedule_recovery_probe(self) -> None:
        """Schedule the debounced, non-blocking recovery probe.

        Called when the executor OBSERVES a degraded binding (the guard
        refusing a BLOCKED/MISSING call, a blocking transport failure,
        a framed pin refusal). The claim inside
        ``should_schedule_probe`` is the concurrency brake: exactly one
        probe per resolved host per debounce window wins, and the
        dispatched probe's own failures never schedule further probes —
        recovery stays driven by real calls, never a self-perpetuating
        poll. The caller NEVER waits on the probe (hot-path rule);
        errors inside the thread are swallowed after logging.
        """
        cfg = self._ssh
        if cfg is None or not cfg.recovery_probes:
            return
        host_key = _host_key_string(cfg.loc)
        if not cfg.cache.should_schedule_probe(host_key):
            return

        def _probe() -> None:
            try:
                self.ping()
            except RemoteWorkspaceExecutionError:
                # Expected for a still-degraded binding; the probe's own
                # classification already updated the cache.
                pass
            except Exception as error:  # noqa: BLE001 - fire-and-forget
                from loguru import logger

                logger.debug(
                    f"recovery probe for {cfg.binding_id} failed: {error!r}"
                )

        threading.Thread(
            target=_probe,
            daemon=True,
            name=f"ssh-recovery-probe-{cfg.loc.host}",
        ).start()


def _require_ping_payload(payload: object) -> None:
    """Validate the ping result shape before the executor returns it."""
    if not isinstance(payload, dict) or set(payload) != set(_PING_RESULT_KEYS):
        raise RemoteWorkspaceExecutionError("protocol_failure", admitted=False)
    chain = payload["identity_chain"]
    if not isinstance(chain, list) or not chain:
        raise RemoteWorkspaceExecutionError("protocol_failure", admitted=False)
    for entry in chain:
        if not isinstance(entry, list) or len(entry) != 4:
            raise RemoteWorkspaceExecutionError(
                "protocol_failure", admitted=False
            )
        if not isinstance(entry[0], str) or not entry[0].startswith("/"):
            raise RemoteWorkspaceExecutionError(
                "protocol_failure", admitted=False
            )
        for number in entry[1:]:
            if type(number) is not int or number < 0:
                raise RemoteWorkspaceExecutionError(
                    "protocol_failure", admitted=False
                )
    if not isinstance(payload["canonical_path"], str) or not (
        payload["canonical_path"].startswith("/")
    ):
        raise RemoteWorkspaceExecutionError("protocol_failure", admitted=False)
    version = payload["python_version"]
    if not isinstance(version, str) or not re.fullmatch(r"3\.\d+\.\d+", version):
        raise RemoteWorkspaceExecutionError("protocol_failure", admitted=False)
    stamp = payload["bundle_sha256"]
    if not isinstance(stamp, str) or not re.fullmatch(r"[0-9a-f]{64}", stamp):
        raise RemoteWorkspaceExecutionError("protocol_failure", admitted=False)


__all__ = [
    "RESPONSE_MAGIC",
    "RemoteWorkspaceExecutionError",
    "RemoteWorkspaceLoopbackError",
    "RemoteWorkspaceToolExecutor",
    "bootstrap_source",
    "parse_fs_read_stamps",
    "run_bundle_loopback",
    "run_bundle_loopback_frames",
]
