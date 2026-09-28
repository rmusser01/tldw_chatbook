"""Laptop side of the per-run SSH session worker (spec 2026-09-27).

One long-lived ssh channel per binding per Console run: the stage-1 loader
handshake happens once, then concurrent callers multiplex framed requests
over it. Each call returns exactly the :class:`RemoteCallResult` the
one-shot :meth:`RemoteWorkspaceTransport.call` would, classified through
the same taxonomy, so the executor and status cache stay unchanged.

Imported only from the SSH path; module-level imports are limited to what
the transport/executor already load (no UI).
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from tldw_chatbook.Tools.build_remote_worker_bundle import (
    expected_bundle_stamp,
    loader_payload,
)
from tldw_chatbook.Tools.remote_binding_locator import RemoteLocator, build_ssh_argv
from tldw_chatbook.Tools.remote_session_frames import (
    CANCEL,
    HELLO,
    LINE,
    REQUEST,
    STATUS,
    FrameReader,
    decode_status,
    encode_frame,
)
from tldw_chatbook.Tools.remote_worker_bundle import RESPONSE_MAGIC
from tldw_chatbook.Tools.remote_workspace_executor import _bundle_payload, bootstrap_source
from tldw_chatbook.Tools.remote_workspace_transport import (
    RemoteCallResult,
    RemoteWorkspaceTransport,
    TransportFailure,
    TransportFailureKind,
    _BoundedCapture,
    _frame_is_admitted_marker,
    _remote_command,
)
from tldw_chatbook.Tools.workspace_tool_protocol import MAX_RESPONSE_BYTES

#: Leading stdout garbage tolerated before a handshake line's magic (same
#: 4 KiB bound as the one-shot transport).
_NOISE_CAP = 4096
#: Longest handshake line accepted after the magic (``READY <64 hex>``).
_HANDSHAKE_LINE_CAP = 256
#: Wall-clock ceiling on the whole handshake (connect + loader + bundle).
_HANDSHAKE_TIMEOUT_S = 30.0
_STDERR_CAP = 64 * 1024
_CLOSE_WAIT_S = 2.0

#: Start failures the status cache records as transport-class (R3).
_TRANSPORT_START_KINDS = frozenset({
    TransportFailureKind.UNREACHABLE,
    TransportFailureKind.INTERPRETER_MISSING,
    TransportFailureKind.PYTHON_TOO_OLD,
    TransportFailureKind.STDOUT_NOISE,
    TransportFailureKind.DESTINATION_CHANGED,
})


class SessionStartError(Exception):
    """The session could not start.

    Attributes:
        transport: True for transport-class failures (record in the status
            cache, no one-shot retry); False for protocol-class ones (fall
            back to the one-shot path for the rest of the run).
        failure: The typed failure when the taxonomy classified one.
    """

    def __init__(self, transport: bool, failure: TransportFailure | None, reason: str) -> None:
        super().__init__(reason)
        self.transport = transport
        self.failure = failure


class _HandshakeFailed(Exception):
    """Internal: the handshake ended; carries what classification needs."""

    def __init__(self, *, noise: bool = False, stalled: bool = False) -> None:
        super().__init__()
        self.noise = noise
        self.stalled = stalled


@dataclass
class _Pending:
    done: threading.Event = field(default_factory=threading.Event)
    admitted_at: float | None = None
    terminal: bytes | None = None
    status: tuple[int | None, int | None] | None = None
    dead_exit: int | None = None


class RemoteSessionWorker:
    """One fork-server session for one binding in one Console run.

    Args:
        loc: The binding's validated locator.
        transport: Source of the ssh binary/options, the grace, and the
            failure taxonomy (:meth:`RemoteWorkspaceTransport.classify_exchange_failure`).
        python: Remote interpreter (bare name/path).
        max_children: Host-side concurrency cap sent in HELLO.
        idle_s: Host-side idle exit sent in HELLO.
        cache: Whether the loader may use its ``$XDG_RUNTIME_DIR`` cache.
        spawn: Test seam replacing the ssh ``Popen``; receives the ssh argv.
    """

    def __init__(
        self,
        loc: RemoteLocator,
        *,
        transport: RemoteWorkspaceTransport,
        python: str,
        max_children: int,
        idle_s: float,
        cache: bool,
        spawn: Callable[[list[str]], subprocess.Popen[bytes]] | None = None,
    ) -> None:
        self._loc = loc
        self._transport = transport
        self._python = python
        self._max_children = max_children
        self._idle_s = idle_s
        self._cache = cache
        self._spawn = spawn or (
            lambda argv: subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        )
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr = _BoundedCapture(_STDERR_CAP)
        self._stderr_thread: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._lock = threading.Lock()  # guards _pending, _next_id, _idle_since, _alive
        self._next_id = 1
        self._alive = False
        self._idle_since: float | None = None

    @property
    def alive(self) -> bool:
        """Whether the session can take calls."""
        return self._alive

    @property
    def idle_since(self) -> float | None:
        """Monotonic time the last call finished; ``None`` while calls are in flight."""
        return self._idle_since

    # -- start -------------------------------------------------------------

    def start(self) -> None:
        """Spawn the channel and run the loader handshake.

        Raises:
            SessionStartError: With ``transport`` set per the failure class.
        """
        artifact, compressed, _bootstrap = _bundle_payload()
        loader = loader_payload()
        manager = self._transport.master_manager
        manager.ensure_master(self._loc)
        argv = [
            manager.ssh_bin,
            *manager.client_options(self._loc),
            *build_ssh_argv(
                self._loc, [], _remote_command(self._python, bootstrap_source(len(loader)))
            ),
        ]
        try:
            proc = self._spawn(argv)
        except OSError as exc:
            failure = TransportFailure(
                TransportFailureKind.UNREACHABLE, None, f"ssh could not be run: {exc}"
            )
            raise SessionStartError(True, failure, "ssh could not be run") from None
        self._proc = proc
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="ssh-session-stderr", daemon=True
        )
        self._stderr_thread.start()

        bundle_hash = hashlib.sha256(compressed).hexdigest()
        header = json.dumps({"hash": bundle_hash, "cache": self._cache}).encode() + b"\n"
        deadline = time.monotonic() + _HANDSHAKE_TIMEOUT_S
        try:
            # LOCKSTEP: after each write, send nothing until the loader answers.
            self._handshake_write(loader + header)
            line = self._read_handshake_line(deadline)
            if line == b"NEED " + bundle_hash.encode():
                self._handshake_write(len(compressed).to_bytes(4, "big") + compressed)
                line = self._read_handshake_line(deadline)
        except _HandshakeFailed as ended:
            self._fail_start(ended)
        if line != b"READY " + expected_bundle_stamp(artifact).encode():
            self._kill()
            raise SessionStartError(False, None, "handshake: unexpected loader line or stamp")
        try:
            self._write(
                encode_frame(
                    HELLO,
                    0,
                    json.dumps({"max_children": self._max_children, "idle_s": self._idle_s}).encode(),
                )
            )
        except OSError:
            self._kill()
            raise SessionStartError(False, None, "session closed before HELLO") from None
        with self._lock:
            self._alive = True
            self._idle_since = time.monotonic()
        threading.Thread(target=self._reader, name="ssh-session-reader", daemon=True).start()

    def _drain_stderr(self) -> None:
        stream = self._proc.stderr if self._proc else None
        if stream is None:
            return
        try:
            while chunk := stream.read(65536):
                self._stderr.append(chunk)
        except (OSError, ValueError):
            pass

    def _handshake_write(self, data: bytes) -> None:
        try:
            self._write(data)
        except OSError:
            pass  # the process died; the read below sees EOF and classifies it

    def _read_handshake_line(self, deadline: float) -> bytes:
        """Read one magic-prefixed line byte-wise, never past its newline.

        Returns:
            The line after the magic, newline stripped.

        Raises:
            _HandshakeFailed: On EOF, the noise cap, or the deadline.
        """
        fd = self._proc.stdout.fileno()  # type: ignore[union-attr]
        buf = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise _HandshakeFailed(stalled=True)
            byte = os.read(fd, 1)  # byte-wise: frames after READY must stay unread
            if not byte:
                raise _HandshakeFailed()
            buf += byte
            start = buf.find(RESPONSE_MAGIC)
            if start == -1:
                if len(buf) - (len(RESPONSE_MAGIC) - 1) > _NOISE_CAP:
                    raise _HandshakeFailed(noise=True)
                continue
            if start > _NOISE_CAP:
                raise _HandshakeFailed(noise=True)
            if byte == b"\n":
                return bytes(buf[start + len(RESPONSE_MAGIC) : -1])
            if len(buf) - start > _HANDSHAKE_LINE_CAP:
                raise _HandshakeFailed()  # magic then an absurd line: protocol

    def _fail_start(self, ended: _HandshakeFailed) -> None:
        proc = self._proc
        assert proc is not None
        killed = proc.poll() is None
        self._kill()
        exit_code = proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(_CLOSE_WAIT_S)
        failure = self._transport.classify_exchange_failure(
            self._loc,
            exit_code=exit_code,
            admitted=False,
            admitted_at=None,
            budget=_HANDSHAKE_TIMEOUT_S,
            killed=killed and ended.stalled,
            noise_capped=ended.noise,
            stderr=self._stderr.value(),
        )
        raise SessionStartError(
            failure.kind in _TRANSPORT_START_KINDS, failure, f"session start failed: {failure.reason}"
        )

    # -- steady state ------------------------------------------------------

    def _write(self, data: bytes) -> None:
        fd = self._proc.stdin.fileno()  # type: ignore[union-attr]
        with self._write_lock:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]

    def _reader(self) -> None:
        """Route inbound frames to waiters; any error is session death.

        R7: per-frame cap ``MAX_RESPONSE_BYTES + 1`` — a LINE body is one
        worker line plus its newline (same bound the loopback harness and
        the one-shot terminal-frame cap use).
        """
        reader = FrameReader(max_body=MAX_RESPONSE_BYTES + 1)
        fd = self._proc.stdout.fileno()  # type: ignore[union-attr]
        try:
            while data := os.read(fd, 65536):
                for kind, request_id, body in reader.feed(data):
                    status = decode_status(body) if kind == STATUS else None
                    with self._lock:
                        pending = self._pending.get(request_id)
                    if pending is None:
                        continue  # abandoned request (timed out / cancelled): drop
                    if kind == LINE:
                        line = body[:-1] if body.endswith(b"\n") else body
                        if pending.admitted_at is None and _frame_is_admitted_marker(line):
                            pending.admitted_at = time.monotonic()
                        else:
                            pending.terminal = line
                    elif kind == STATUS:
                        pending.status = status
                        pending.done.set()
                    # BUSY is informational: the request stays queued host-side.
        except Exception:  # noqa: BLE001 - FrameError/JSON/KeyError/TypeError/OSError: session death
            pass
        self._die()

    def _die(self) -> None:
        """Mark the session dead, reap it, and release every waiter."""
        with self._lock:
            self._alive = False
        proc = self._proc
        exit_code = 255
        if proc is not None:
            self._kill()
            exit_code = proc.wait()
        with self._lock:
            waiters = list(self._pending.values())
        for pending in waiters:
            if pending.dead_exit is None:
                pending.dead_exit = exit_code
            pending.done.set()

    def call(self, request_bytes: bytes, *, budget: float) -> RemoteCallResult:
        """Run one request over the session; same result shape as the one-shot call.

        Deadline: (admitted_at or send time) + budget + grace, then CANCEL;
        a STATUS still missing after another grace means the parent is
        stuck and the whole session is killed.
        """
        with self._lock:
            if not self._alive:
                return RemoteCallResult(
                    False,
                    None,
                    TransportFailure(
                        TransportFailureKind.WORKER_FAILED_TO_START, None, "session not running"
                    ),
                )
            request_id = self._next_id
            self._next_id += 1
            pending = self._pending[request_id] = _Pending()
            self._idle_since = None
        grace = self._transport.grace_seconds
        sent_at = time.monotonic()
        killed = False
        try:
            self._write(encode_frame(REQUEST, request_id, request_bytes))
            while not pending.done.is_set():
                remaining = (pending.admitted_at or sent_at) + budget + grace - time.monotonic()
                if remaining <= 0:
                    killed = True
                    self._write(encode_frame(CANCEL, request_id, b""))
                    if not pending.done.wait(grace):
                        self._die()  # parent stuck: treated as session death
                    break
                pending.done.wait(min(remaining, 0.5))
        except (OSError, ValueError):  # ValueError: stdin already closed
            self._die()
        finally:
            with self._lock:
                self._pending.pop(request_id, None)
                if not self._pending:
                    self._idle_since = time.monotonic()
        return self._result(pending, budget, killed=killed)

    def _result(self, pending: _Pending, budget: float, *, killed: bool) -> RemoteCallResult:
        admitted = pending.admitted_at is not None
        if pending.terminal is not None:
            # As the one-shot: a terminal frame is the response, whatever the exit.
            return RemoteCallResult(admitted, pending.terminal, None)
        if pending.status is not None:
            exit_code, signal_no = pending.status
            code = exit_code if exit_code is not None else -(signal_no or 9)
        else:
            code = pending.dead_exit if pending.dead_exit is not None else 255
        failure = self._transport.classify_exchange_failure(
            self._loc,
            exit_code=code,
            admitted=admitted,
            admitted_at=pending.admitted_at,
            budget=budget,
            killed=killed,
            noise_capped=False,
            stderr=b"",  # session stderr cannot be attributed to one request
        )
        return RemoteCallResult(admitted, None, failure)

    def _kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def close(self) -> None:
        """Close stdin (host exits on EOF), wait briefly, then kill. Never raises."""
        with self._lock:
            self._alive = False
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=_CLOSE_WAIT_S)
        except subprocess.TimeoutExpired:
            self._kill()
            try:
                proc.wait(timeout=_CLOSE_WAIT_S)
            except subprocess.TimeoutExpired:
                pass
