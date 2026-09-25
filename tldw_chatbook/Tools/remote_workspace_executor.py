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
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
import zlib
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

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


class RemoteWorkspaceToolExecutor:
    """Remote workspace executor, loopback mode (ssh transport is Task 11).

    ``execute`` slots into the same call surface as
    ``WorkspaceToolExecutor`` so provider dispatch stays
    transport-agnostic; Task 15 types its result into
    ``RunAdmittedWorkspaceRoot``.
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
                bit the status cache consumes.
            RemoteWorkspaceLoopbackError: For harness-level failures.
        """
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
    ) -> dict[str, Any]:
        """Build one wire-legal request dict, validating it as the parent."""
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
                "timeout_seconds": WORKSPACE_HELPER_TIMEOUT_SECONDS,
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
