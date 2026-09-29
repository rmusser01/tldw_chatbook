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
import selectors
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, NoReturn

from loguru import logger

from tldw_chatbook.Tools.build_remote_worker_bundle import (
    expected_bundle_stamp,
    loader_payload,
)
from tldw_chatbook.Tools.remote_binding_locator import RemoteLocator, build_ssh_argv
from tldw_chatbook.Tools.remote_session_frames import (
    CANCEL,
    HELLO,
    HOST_SPAWN_FAILED,
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
    _frame_is_admitted_marker,
    _remote_command,
)
from tldw_chatbook.Tools.workspace_tool_protocol import MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES
from tldw_chatbook.Tools.workspace_wire_decode import WIRE_VERSION, encode_response

#: Leading stdout garbage tolerated before a handshake line's magic (same
#: 4 KiB bound as the one-shot transport).
_NOISE_CAP = 4096
#: Longest handshake line accepted after the magic (``READY <64 hex>``).
_HANDSHAKE_LINE_CAP = 256
#: Wall-clock ceiling on the whole handshake (connect + loader + bundle).
_HANDSHAKE_TIMEOUT_S = 30.0
_STDERR_CAP = 64 * 1024
#: What the one-shot worker answers a request over MAX_REQUEST_BYTES with
#: (unadmitted ``invalid_request`` failure; see ``run_workspace_worker``).
_OVERSIZED_REQUEST_FRAME = encode_response({
    "version": WIRE_VERSION,
    "operation_id": "unknown",
    "outcome": "failure",
    "code": "invalid_request",
    "result": None,
    "error": "workspace operation failed",
    "elapsed_ms": 0,
    "truncated": False,
    "cleanup_proven": True,
})
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


class SessionClosed(Exception):
    """The laptop closed this healthy session (idle reap, run end, app
    exit) before the call's request was registered (TASK-33401) or before
    any byte of it was written (TASK-33421): nothing was sent, so the
    caller may ask the registry again."""


class _HandshakeFailed(Exception):
    """Internal: the handshake ended; carries what classification needs."""

    def __init__(self, *, noise: bool = False, stalled: bool = False) -> None:
        super().__init__()
        self.noise = noise
        self.stalled = stalled


class _WriteStalled(Exception):
    """Internal: a write stalled MID-FRAME; the stream is unusable."""


class _NotSent(Exception):
    """Internal: the write lock stayed busy until the deadline; nothing was
    written and the stream is intact."""


class _StdinClosed(BrokenPipeError):
    """Internal: stdin was already closed (by ``close()``) before any byte
    of this write went out -- distinct from a mid-write EPIPE."""


def _checked_status(body: bytes) -> tuple[int | None, int | None]:
    """``decode_status`` plus type checks: a malformed STATUS is a frame
    error (the reader kills the session), never a TypeError on a caller."""
    status = decode_status(body)
    if not all(v is None or (type(v) is int) for v in status):
        raise ValueError("malformed STATUS frame")
    return status


def _wait_fd(fd: int, events: int, timeout: float) -> bool:
    """Wait for one fd (``selectors``: no FD_SETSIZE limit, unlike ``select.select``)."""
    if timeout <= 0:
        return False
    with selectors.DefaultSelector() as selector:
        selector.register(fd, events)
        return bool(selector.select(timeout))


class _TailCapture:
    """Byte sink keeping the LAST ``cap`` bytes (TASK-33403).

    A long-lived session's death reason (a mux marker, ssh's final error)
    is at the end of its stderr; a head capture drops it once earlier
    output fills the cap, turning a status-preserving MUX_ERROR into
    UNREACHABLE (BLOCKED).
    """

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._buf = bytearray()
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self._buf += chunk
            if len(self._buf) > self._cap:
                del self._buf[: len(self._buf) - self._cap]

    def value(self) -> bytes:
        with self._lock:
            return bytes(self._buf)


@dataclass
class _Pending:
    done: threading.Event = field(default_factory=threading.Event)
    admitted_at: float | None = None
    terminal: bytes | None = None
    status: tuple[int | None, int | None] | None = None


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
        handshake_timeout: Caps the handshake wall-clock ceiling: ``None``
            means the module's ``_HANDSHAKE_TIMEOUT_S`` (30 s), a number is
            capped at it (a call's own budget may never outlive it).
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
        handshake_timeout: float | None = None,
    ) -> None:
        self._loc = loc
        self._transport = transport
        self._python = python
        self._max_children = max_children
        self._idle_s = idle_s
        self._cache = cache
        self._handshake_timeout = handshake_timeout
        self._handshake_limit = _HANDSHAKE_TIMEOUT_S
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
        self._stderr = _TailCapture(_STDERR_CAP)
        self._stderr_thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._lock = threading.Lock()  # guards _pending, _next_id, _idle_since, _alive
        self._next_id = 1
        self._alive = False
        self._idle_since: float | None = None
        #: Death cause (R9): None while alive/unstarted, True for a natural
        #: death (EOF / ssh exited), False when the laptop ended it.
        self._death_natural: bool | None = None
        self._death_code: int | None = None
        #: True once close() closed a session that was alive at that moment
        #: (TASK-33401): the caller's request never reached this session.
        self._retired = False
        self._settled = threading.Event()  # set once _die has reaped

    @property
    def alive(self) -> bool:
        """Whether the session can take calls."""
        return self._alive

    @property
    def idle_since(self) -> float | None:
        """Monotonic time the last call finished; ``None`` while calls are in flight."""
        return self._idle_since

    def ended_cleanly(self) -> bool:
        """Whether the session ended NATURALLY with ssh exit code 0 (R10).

        A host idle-exit or clean EOF is a benign session end, not a death:
        the registry recreates it without spending the run key's single
        restart. Waits (bounded) for the reap so the real exit code is known.
        """
        if self._death_natural is not True:
            return False
        self._settled.wait(self._transport.grace_seconds + _CLOSE_WAIT_S)
        return self._death_code == 0

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
        try:
            self._handshake(proc, loader=loader, compressed=compressed, artifact=artifact)
        except SessionStartError:
            self._abandon_start()
            raise
        except OSError:
            # A local pipe error outside the classified paths: nothing can
            # be said about the host, so fall back to one-shot for the run.
            self._abandon_start()
            raise SessionStartError(False, None, "session start failed: local pipe error") from None
        except BaseException:
            self._abandon_start()
            raise

    def _handshake(
        self, proc: subprocess.Popen[bytes], *, loader: bytes, compressed: bytes, artifact
    ) -> None:
        # Every write is select-bounded (see _write): a stalled remote can
        # never wedge a writer, and so never the write lock.
        os.set_blocking(proc.stdin.fileno(), False)  # type: ignore[union-attr]
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="ssh-session-stderr", daemon=True
        )
        self._stderr_thread.start()

        bundle_hash = hashlib.sha256(compressed).hexdigest()
        header = json.dumps({"hash": bundle_hash, "cache": self._cache}).encode() + b"\n"
        limit = _HANDSHAKE_TIMEOUT_S
        if self._handshake_timeout is not None:
            limit = min(self._handshake_timeout, limit)
        deadline = time.monotonic() + limit
        self._handshake_limit = limit
        try:
            # LOCKSTEP: after each write, send nothing until the loader answers.
            self._handshake_write(loader + header, deadline)
            line = self._read_handshake_line(deadline)
            cache_hit = line != b"NEED " + bundle_hash.encode()
            if not cache_hit:
                # The host answered NEED: from here a deadline is never
                # transport-class (TASK-33420).
                self._handshake_write(
                    len(compressed).to_bytes(4, "big") + compressed, deadline, answered=True
                )
                try:
                    line = self._read_handshake_line(deadline)
                except _HandshakeFailed as ended:
                    if ended.stalled:
                        self._stalled_after_answer("handshake stalled after the host answered")
                    raise
        except _HandshakeFailed as ended:
            self._fail_start(ended)
        if line != b"READY " + expected_bundle_stamp(artifact).encode():
            self._kill_and_reap()
            raise SessionStartError(False, None, "handshake: unexpected loader line or stamp")
        hello = json.dumps({"max_children": self._max_children, "idle_s": self._idle_s})
        try:
            self._write(
                encode_frame(HELLO, 0, hello.encode()),
                time.monotonic() + self._transport.grace_seconds,
            )
        except (OSError, _WriteStalled, _NotSent):
            self._kill_and_reap()
            raise SessionStartError(False, None, "session closed before HELLO") from None
        with self._lock:
            self._alive = True
            self._idle_since = time.monotonic()
        self._reader_thread = threading.Thread(
            target=self._reader, name="ssh-session-reader", daemon=True
        )
        self._reader_thread.start()
        logger.debug(
            "ssh session worker started; bundle cache {}", "hit" if cache_hit else "miss"
        )

    def _abandon_start(self) -> None:
        """Leave nothing behind after a failed start: reap ssh, close its pipes."""
        self._kill_and_reap()
        if self._stderr_thread is not None:
            self._stderr_thread.join(_CLOSE_WAIT_S)
        proc = self._proc
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is None:
                continue
            if stream is proc.stderr and self._stderr_thread is not None and self._stderr_thread.is_alive():
                continue  # still read by the drain thread: left to GC, never closed under it
            try:
                stream.close()
            except OSError:
                pass

    def _drain_stderr(self) -> None:
        stream = self._proc.stderr if self._proc else None
        if stream is None:
            return
        try:
            # os.read returns whatever is available: each chunk lands in the
            # capture at once, so a join timeout still sees what was read.
            while chunk := os.read(stream.fileno(), 65536):
                self._stderr.append(chunk)
        except (OSError, ValueError):
            pass

    def _handshake_write(self, data: bytes, deadline: float, *, answered: bool = False) -> None:
        try:
            self._write(data, deadline)
        except OSError:
            pass  # the process died; the read below sees EOF and classifies it
        except (_WriteStalled, _NotSent):
            if answered:
                self._stalled_after_answer("handshake write stalled")
            self._kill_and_reap()
            raise SessionStartError(False, None, "handshake write stalled") from None

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
            if not _wait_fd(fd, selectors.EVENT_READ, remaining):
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
        try:
            exit_code = proc.wait(timeout=self._transport.grace_seconds)
        except subprocess.TimeoutExpired:
            exit_code = -signal.SIGKILL  # unreapable for now; give up quietly
        if self._stderr_thread is not None:
            self._stderr_thread.join(_CLOSE_WAIT_S)
        failure = self._transport.classify_exchange_failure(
            self._loc,
            exit_code=exit_code,
            admitted=False,
            admitted_at=None,
            budget=self._handshake_limit,
            killed=killed and ended.stalled,
            noise_capped=ended.noise,
            stderr=self._stderr.value(),
        )
        raise SessionStartError(
            failure.kind in _TRANSPORT_START_KINDS, failure, f"session start failed: {failure.reason}"
        )

    def _stalled_after_answer(self, reason: str) -> NoReturn:
        """The host answered the handshake, then the deadline passed.

        A live channel proves reachability, so this is never transport-class
        (ADR-181, R8). When the deadline was the call's own budget (a slow
        cache-miss upload, a short tool timeout), this call fails as
        ``OP_TIMEOUT`` and the next call tries a session again. When it was
        the 30 s cap, the loader is stuck: protocol-class (``reason``),
        one-shot for the run.
        """
        self._kill_and_reap()
        if self._handshake_limit < _HANDSHAKE_TIMEOUT_S:
            failure = TransportFailure(TransportFailureKind.OP_TIMEOUT, None, "operation timed out")
            raise SessionStartError(True, failure, "session start ran out of the call's budget") from None
        raise SessionStartError(False, None, reason) from None

    # -- steady state ------------------------------------------------------

    def _write(self, data: bytes, deadline: float) -> None:
        """Write ``data`` whole under the write lock, by ``deadline``.

        Once the lock is held, the holder always gets at least ``grace`` to
        write: a caller that wins the lock at its deadline (typically right
        after another caller's upload filled the pipe) must not mistake a
        briefly full pipe for a stuck host.

        Raises:
            _NotSent: The lock stayed busy (another caller's slow upload)
                until the deadline: nothing written, the stream is intact.
            _WriteStalled: Holding the lock, the pipe stopped draining for
                the whole write window before the frame was out: the host is
                not reading (and a partial frame corrupts the stream), so
                the session must die.
            _StdinClosed: stdin was closed before anything was written.
            OSError: The pipe broke (the process is gone).
        """
        if not self._write_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise _NotSent()
        try:
            # Read the fd UNDER the lock: close() closes stdin only while
            # holding it, so the number cannot be closed (and reused by
            # another thread's open) while this loop writes to it.
            stdin = self._proc.stdin  # type: ignore[union-attr]
            if stdin is None or stdin.closed:
                raise _StdinClosed("session stdin closed")
            fd = stdin.fileno()
            view = memoryview(data)
            write_deadline = max(deadline, time.monotonic() + self._transport.grace_seconds)
            while view:
                if not _wait_fd(fd, selectors.EVENT_WRITE, write_deadline - time.monotonic()):
                    # Holding the lock, the pipe took nothing (or stopped
                    # mid-frame) for a full write window: the host is not
                    # reading, so this is a stuck parent either way.
                    raise _WriteStalled()
                try:
                    view = view[os.write(fd, view) :]
                except BlockingIOError:
                    continue
        finally:
            self._write_lock.release()

    def _reader(self) -> None:
        """Route inbound frames to waiters; EOF is a natural death, any error a laptop kill.

        R7: per-frame cap ``MAX_RESPONSE_BYTES + 1`` — a LINE body is one
        worker line plus its newline (same bound the loopback harness and
        the one-shot terminal-frame cap use).
        """
        reader = FrameReader(max_body=MAX_RESPONSE_BYTES + 1)
        fd = self._proc.stdout.fileno()  # type: ignore[union-attr]
        try:
            while data := os.read(fd, 65536):
                for kind, request_id, body in reader.feed(data):
                    status = _checked_status(body) if kind == STATUS else None
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
        except Exception:  # noqa: BLE001 - FrameError/JSON/KeyError/TypeError/OSError
            self._die(natural=False)
            return
        self._die(natural=True)

    def _die(self, *, natural: bool) -> None:
        """Mark the session dead, reap it, and release every waiter.

        The FIRST recorded cause wins (R9): a later reader EOF after a
        laptop kill or ``close()`` stays self-inflicted. A natural death
        reaps before killing so ssh's real exit code survives (ssh closes
        stdout on channel EOF and exits once exit-status arrives), exactly
        as the one-shot ``_settle_process``.
        """
        with self._lock:
            if self._death_natural is None:
                self._death_natural = natural
            natural = self._death_natural
            was_alive, self._alive = self._alive, False
        proc = self._proc
        code = 255
        if proc is not None:
            if natural:
                try:
                    code = proc.wait(timeout=self._transport.grace_seconds)
                except subprocess.TimeoutExpired:
                    self._kill()
                    try:
                        code = proc.wait(timeout=self._transport.grace_seconds)
                    except subprocess.TimeoutExpired:
                        code = -signal.SIGKILL
            else:
                self._kill()
                try:
                    code = proc.wait(timeout=self._transport.grace_seconds)
                except subprocess.TimeoutExpired:
                    code = -signal.SIGKILL  # unreapable for now; give up quietly
        with self._lock:
            if self._death_code is None:
                self._death_code = code
            waiters = list(self._pending.values())
        self._settled.set()
        for pending in waiters:
            pending.done.set()
        if was_alive:
            logger.debug(
                "ssh session worker ended ({}); exit code {}",
                "natural" if natural else "ended by the laptop",
                code,
            )

    def call(self, request_bytes: bytes, *, budget: float) -> RemoteCallResult:
        """Run one request over the session; same result shape as the one-shot call.

        Deadline: (admitted_at or send time) + budget + grace, then CANCEL;
        a STATUS still missing after another grace means the parent is
        stuck and the whole session is killed.

        A request over ``MAX_REQUEST_BYTES`` never touches the session (the
        host would reject the frame and end the session): it gets the
        one-shot path's answer at once -- the worker's unadmitted
        ``invalid_request`` failure frame. The executor already refuses
        such requests while building them, so this is a defensive bound.

        Raises:
            SessionClosed: ``close()`` retired this healthy session before
                any byte of the REQUEST was written (TASK-33401/33421); a
                request that was written is never reported this way.
        """
        if len(request_bytes) > MAX_REQUEST_BYTES:
            return RemoteCallResult(False, _OVERSIZED_REQUEST_FRAME, None)
        with self._lock:
            alive = self._alive
            retired = self._retired
            if alive:
                request_id = self._next_id
                self._next_id += 1
                pending = self._pending[request_id] = _Pending()
                self._idle_since = None
        if not alive:
            if retired:
                raise SessionClosed()
            return self._dead_session_result()
        grace = self._transport.grace_seconds
        sent_at = time.monotonic()
        killed = False
        not_sent = False
        try:
            try:
                self._write(encode_frame(REQUEST, request_id, request_bytes), sent_at + budget + grace)
            except _StdinClosed:
                if self._retired:
                    # close() retired this healthy session after the request
                    # registered but before any REQUEST byte was written:
                    # nothing ran, so the caller may ask again (TASK-33421).
                    raise SessionClosed() from None
                raise
            while not pending.done.is_set():
                remaining = (pending.admitted_at or sent_at) + budget + grace - time.monotonic()
                if remaining <= 0:
                    killed = True
                    self._write(encode_frame(CANCEL, request_id, b""), time.monotonic() + grace)
                    if not pending.done.wait(grace):
                        self._die(natural=False)  # parent stuck: laptop kills the session
                    break
                pending.done.wait(min(remaining, 0.5))
        except _NotSent:
            # The lock stayed busy (another caller's slow upload): our
            # REQUEST or CANCEL never reached the wire, the stream is intact,
            # and the session stays up. Popping
            # the pending entry below makes any late frames "abandoned".
            not_sent = True
        except _WriteStalled:
            self._die(natural=False)
        except (OSError, ValueError):  # broken pipe / stdin closed by close()
            self._die(natural=True)
        finally:
            with self._lock:
                self._pending.pop(request_id, None)
                if not self._pending:
                    self._idle_since = time.monotonic()
        if not_sent and self._alive and pending.terminal is None and pending.status is None:
            # R8 (session alive, never admitted or never cancelled): an op timeout.
            return RemoteCallResult(
                pending.admitted_at is not None,
                None,
                TransportFailure(TransportFailureKind.OP_TIMEOUT, None, "operation timed out"),
            )
        return self._result(pending, budget, killed=killed)

    def _dead_session_result(self) -> RemoteCallResult:
        """R9: a natural death classifies by the real ssh exit code; a laptop end is REMOTE_OP_FAILED.

        R10: a natural death with exit code 0 (host idle-exit, clean EOF) is
        a benign session end -- REMOTE_OP_FAILED (status-preserving), never
        a transport failure.
        """
        if self._proc is None:  # never started
            return RemoteCallResult(
                False,
                None,
                TransportFailure(
                    TransportFailureKind.WORKER_FAILED_TO_START, None, "session not running"
                ),
            )
        if self._death_natural:
            self._settled.wait(self._transport.grace_seconds + _CLOSE_WAIT_S)
            code = self._death_code if self._death_code is not None else 255
            if code == 0:
                return RemoteCallResult(
                    False,
                    None,
                    TransportFailure(TransportFailureKind.REMOTE_OP_FAILED, 0, "session ended"),
                )
            failure = self._transport.classify_exchange_failure(
                self._loc,
                exit_code=code,
                admitted=False,
                admitted_at=None,
                budget=1.0,
                killed=False,
                noise_capped=False,
                stderr=self._stderr.value(),
            )
            return RemoteCallResult(False, None, failure)
        return RemoteCallResult(
            False,
            None,
            TransportFailure(
                TransportFailureKind.REMOTE_OP_FAILED, self._death_code, "session ended by the laptop"
            ),
        )

    def _result(self, pending: _Pending, budget: float, *, killed: bool) -> RemoteCallResult:
        admitted = pending.admitted_at is not None
        if pending.terminal is not None:
            # As the one-shot: a terminal frame is the response, whatever the exit.
            return RemoteCallResult(admitted, pending.terminal, None)
        if pending.status is None:
            if not admitted:
                return self._dead_session_result()
            code = self._death_code if self._death_code is not None else -9
        else:
            exit_code, signal_no = pending.status
            if not admitted and exit_code == HOST_SPAWN_FAILED:
                # The host could not fork for this request (process limit):
                # the live session proves reachability, so status-preserving.
                return RemoteCallResult(
                    False,
                    None,
                    TransportFailure(
                        TransportFailureKind.REMOTE_OP_FAILED,
                        exit_code,
                        "host could not start the operation",
                    ),
                )
            code = exit_code if exit_code is not None else -(signal_no or 9)
            refused = exit_code is None and signal_no == signal.SIGKILL
            if (killed or refused) and not admitted:
                # R8: our CANCEL ended a never-admitted (e.g. queued) request
                # on a live session — an op timeout, never UNREACHABLE. The
                # host refuses a request its bounded queue cannot take (or a
                # duplicate id) with the same STATUS, so it maps the same.
                return RemoteCallResult(
                    False,
                    None,
                    TransportFailure(TransportFailureKind.OP_TIMEOUT, code, "operation timed out"),
                )
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
        """SIGKILL the session: its whole group when it leads one (real ssh
        spawns with start_new_session, so ProxyCommand children die too),
        else just the process (a test spawn shares pytest's group)."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.getpgid(proc.pid) == proc.pid:
                os.killpg(proc.pid, signal.SIGKILL)
                return
        except OSError:
            pass  # fall back to the process itself
        try:
            proc.kill()
        except OSError:
            pass

    def _kill_and_reap(self) -> None:
        self._kill()
        if self._proc is not None:
            try:
                self._proc.wait(timeout=_CLOSE_WAIT_S)
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        """Close stdin (host exits on EOF), wait briefly, then kill. Never raises.

        stdin is closed only under the write lock, so a writer mid-frame
        never sees its fd number closed (and reused) under it. A writer that
        holds the lock past the grace is stuck on a stalled host: the kill
        breaks its pipe (EPIPE) and it releases the lock. Once the child is
        reaped and the reader/stderr threads have finished (bounded joins),
        stdout and stderr are closed too.
        """
        with self._lock:
            if self._death_natural is None:
                self._death_natural = False
            was_alive, self._alive = self._alive, False
            self._retired = self._retired or was_alive
        proc = self._proc
        if proc is None:
            return
        if was_alive:
            logger.debug("ssh session worker closing")
        locked = self._write_lock.acquire(timeout=self._transport.grace_seconds)
        if not locked:
            self._kill()
            locked = self._write_lock.acquire(timeout=_CLOSE_WAIT_S)
        try:
            # Still locked out: leave the fd to the (killed) process's reap.
            if locked and proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        finally:
            if locked:
                self._write_lock.release()
        try:
            proc.wait(timeout=_CLOSE_WAIT_S)
        except subprocess.TimeoutExpired:
            self._kill_and_reap()
        # Close the output pipes only once no thread can still be in an
        # os.read on them (a closed fd number can be reused by another
        # open). A thread still running after its bounded join keeps its
        # pipe: that one is left to GC rather than closed under it.
        current = threading.current_thread()
        for thread, stream in (
            (self._reader_thread, proc.stdout),
            (self._stderr_thread, proc.stderr),
        ):
            if thread is not None and thread is not current:
                thread.join(_CLOSE_WAIT_S)
            if stream is None or (thread is not None and thread.is_alive()):
                continue
            try:
                stream.close()
            except OSError:
                pass
