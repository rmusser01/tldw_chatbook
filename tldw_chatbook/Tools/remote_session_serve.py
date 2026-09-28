"""Fork-server parent for the SSH session worker (stdlib-only; bundled).

One parent per session: reads frames from ``in_fd``, forks a child per
REQUEST (the child runs ``run_request`` with stdin=/dev/null and stdout
= a pipe back here), relays the child's lines as LINE frames, reaps it,
then sends STATUS. Single-threaded by design: forking a threaded
process is unsafe, and the per-request watchdog lives in the child.
"""

from __future__ import annotations

import json
import math
import os
import selectors
import signal
import time
from collections import deque
from typing import BinaryIO, Callable

from tldw_chatbook.Tools.remote_session_frames import (
    BUSY, CANCEL, HELLO, HOST_SPAWN_FAILED, LINE, REQUEST, STATUS, FrameError, FrameReader,
    encode_frame, encode_status,
)

_CHILD_CRASH_EXIT = 70
#: Upper clamp on HELLO's ``idle_s``: a larger timeout overflows the
#: selector (kqueue rejects ~1e12 s; epoll takes an int of milliseconds).
_MAX_IDLE_S = 1_000_000.0
#: Queue bounds: at most this many queued REQUESTs per allowed child, and
#: at most this many ``max_request_bytes`` of queued bodies in aggregate.
_QUEUE_PER_CHILD = 4
_QUEUE_BYTES_FACTOR = 2
#: How long the parent holds a still-running child's output before writing
#: it, so a fast operation's admitted marker, result and STATUS leave in one
#: write (one ssh packet) instead of two small ones. A slow operation's
#: marker goes out this much later; nothing else waits. 0 disables it.
_COALESCE_S = 0.010


class _Child:
    __slots__ = ("pid", "fd", "request_id", "partial", "sent", "capped")

    def __init__(self, pid: int, fd: int, request_id: int) -> None:
        self.pid, self.fd, self.request_id = pid, fd, request_id
        self.partial = bytearray()
        self.sent = 0
        self.capped = False


def _close_fds_above_2(keep: int) -> None:
    """Close every open fd above 2 except ``keep``, in a forked child.

    Enumerates the process's actual open descriptors via ``/proc/self/fd``
    (Linux) or ``/dev/fd`` (macOS/BSD) and closes only those, instead of
    calling ``close()`` on every integer up to ``SC_OPEN_MAX`` -- with a
    large (or misreported, e.g. ``-1``) nofile limit that is one syscall
    per fd number and dominates fork latency (measured ~134 ms per fork
    here with ``SC_OPEN_MAX`` = 1048576, vs ~1.5 ms for a bare fork).

    Args:
        keep: The one fd above 2 to leave open (everything else above 2
            is closed unconditionally).
    """
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            names = os.listdir(fd_dir)
        except OSError:
            continue
        for name in names:
            try:
                fd = int(name)
            except ValueError:
                continue
            if fd <= 2 or fd == keep:
                continue
            # `fd_dir` was itself opened (and already closed again) just to
            # read this listing, so its own fd number can appear here even
            # though it is already gone (observed on Linux) -- closing an
            # already-closed fd just raises OSError, which is ignored below
            # like every other fd this loop cannot close for other reasons.
            try:
                os.close(fd)
            except OSError:
                pass
        return
    # Neither directory is readable (e.g. a locked-down container) -- fall
    # back to a bounded range rather than iterating a possibly-huge
    # SC_OPEN_MAX.
    os.closerange(3, keep)
    os.closerange(keep + 1, 4096)


def _spawn(raw: bytes, request_id: int, run_request: Callable[[bytes, BinaryIO], int]) -> _Child:
    read_fd, write_fd = os.pipe()
    try:
        pid = os.fork()
    except OSError:
        os.close(read_fd)
        os.close(write_fd)
        raise
    if pid == 0:  # child
        code = _CHILD_CRASH_EXIT
        try:
            devnull = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull, 0)
            os.dup2(write_fd, 1)
            _close_fds_above_2(keep=2)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            with os.fdopen(1, "wb", closefd=False) as out:
                code = run_request(raw, out)
                out.flush()
        except BaseException:  # noqa: BLE001 - a child must never return into the parent loop
            code = _CHILD_CRASH_EXIT
        os._exit(code if isinstance(code, int) else _CHILD_CRASH_EXIT)
    os.close(write_fd)
    os.set_blocking(read_fd, False)
    return _Child(pid, read_fd, request_id)


def serve(
    in_fd: int,
    out_fd: int,
    *,
    run_request: Callable[[bytes, BinaryIO], int],
    max_request_bytes: int,
    max_response_bytes: int,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run one session's fork-server loop until stdin EOF or an idle timeout.

    Reads frames from ``in_fd``. The first inbound frame must be ``HELLO``
    carrying a JSON body ``{"max_children": int, "idle_s": float}`` that
    sets the concurrency cap and the idle-exit timeout; ``max_children``
    is clamped to at least 1 and ``idle_s`` to ``[1.0, _MAX_IDLE_S]``
    seconds (``max_children`` must be a JSON integer and ``idle_s`` a
    finite JSON number; a boolean is neither). Any
    ``REQUEST`` received before ``HELLO`` is silently dropped. Each
    ``REQUEST`` forks a child that runs ``run_request(body, out)`` with
    stdin bound to ``/dev/null`` and stdout bound to a pipe back to this
    loop; every line the child writes is relayed as a ``LINE`` frame, and
    once the child exits a ``STATUS`` frame reports its exit code and
    signal. A ``REQUEST`` received while ``max_children`` children are
    already running is queued and echoed back as a ``BUSY`` frame; it
    starts once a slot frees up. The queue is bounded to
    ``_QUEUE_PER_CHILD * max_children`` requests and
    ``_QUEUE_BYTES_FACTOR * max_request_bytes`` bytes of queued bodies; a
    ``REQUEST`` past either bound, or one whose id is already running or
    queued, is refused at once with ``STATUS`` (exit ``None``, signal
    ``SIGKILL``) -- the same encoding as a cancelled queued request, which
    the laptop maps to an unadmitted OP_TIMEOUT (R8) -- and never runs; a
    duplicate never replaces the first request's cancellation mapping. A
    ``CANCEL`` kills the matching live
    child with SIGKILL, or -- if the request is still queued -- drops it
    from the queue and immediately sends its ``STATUS`` (exit ``None``,
    signal ``SIGKILL``), since it never ran and so will never trigger the
    normal child-exit path. A child whose combined output exceeds
    ``max_response_bytes`` is killed and its STATUS still reports the
    kill signal. A malformed ``HELLO`` body (bad JSON, a missing key, a
    value of the wrong type, or a non-finite ``idle_s``) ends the session
    with exit code 3 rather than raising into the caller. A ``REQUEST``
    whose fork or pipe the host refuses (process or fd limit) gets
    ``STATUS`` (``HOST_SPAWN_FAILED``, ``None``) at once; only that
    request fails, and the session keeps serving. The loop is single-threaded: forking a
    multi-threaded process is unsafe, and any per-request timeout is the
    child's own responsibility, not the parent's.

    Args:
        in_fd: Readable file descriptor carrying inbound frames.
        out_fd: Writable file descriptor for outbound frames.
        run_request: Called in the child as ``run_request(body, out)``;
            its return value becomes the child's exit code.
        max_request_bytes: Per-frame body cap enforced while reading
            inbound frames (``FrameReader``'s ``max_body``).
        max_response_bytes: Cap on a child's total relayed output before
            it is killed.
        clock: Monotonic time source; overridable for tests.

    Returns:
        ``0`` on a clean stdin EOF or idle exit, ``3`` if an inbound frame
        violates the codec's size cap or ``HELLO``'s body is malformed.

    Raises:
        OSError: If a low-level file descriptor operation (pipe, read,
            write) fails for a reason other than the cases already
            handled above. A failed fork or pipe while starting a request
            never raises here; see the ``HOST_SPAWN_FAILED`` case above.
    """
    reader = FrameReader(max_body=max_request_bytes)
    selector = selectors.DefaultSelector()
    selector.register(in_fd, selectors.EVENT_READ, None)
    children: dict[int, _Child] = {}          # fd -> child
    by_request: dict[int, _Child] = {}
    queue: deque[tuple[int, bytes]] = deque()
    queued_bytes = 0
    max_children, idle_s, hello = 8, 60.0, False
    last_activity = clock()
    outbox = bytearray()
    hold_until: float | None = None
    urgent = False

    def flush() -> None:
        view = memoryview(outbox)
        while view:
            written = os.write(out_fd, view)
            view = view[written:]
        view.release()  # drop the last (empty) slice's export before clear()
        outbox.clear()

    def start(request_id: int, raw: bytes) -> None:
        nonlocal urgent
        try:
            child = _spawn(raw, request_id, run_request)
        except OSError:
            # fork/pipe refused (process or fd limit): only this request
            # fails; the loop, its children and its queue carry on.
            outbox.extend(encode_frame(STATUS, request_id, encode_status(HOST_SPAWN_FAILED, None)))
            urgent = True
            return
        children[child.fd] = child
        by_request[request_id] = child
        selector.register(child.fd, selectors.EVENT_READ, child)

    def finish(child: _Child) -> None:
        nonlocal queued_bytes, urgent
        selector.unregister(child.fd)
        os.close(child.fd)
        if child.partial and not child.capped:
            outbox.extend(encode_frame(LINE, child.request_id, bytes(child.partial)))
        # Blocking is fine here: EOF on child.fd only happens once the child
        # has exited (it never closes fd 1 early), so it is already a
        # zombie or about to be and this reap returns immediately.
        # (Deferred: WNOHANG.)
        _, status = os.waitpid(child.pid, 0)
        exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else None
        signal_no = os.WTERMSIG(status) if os.WIFSIGNALED(status) else None
        outbox.extend(encode_frame(STATUS, child.request_id, encode_status(exit_code, signal_no)))
        urgent = True
        del children[child.fd]
        by_request.pop(child.request_id, None)
        while queue and len(children) < max_children:
            request_id, raw = queue.popleft()
            queued_bytes -= len(raw)
            start(request_id, raw)

    def refuse(request_id: int) -> None:
        nonlocal urgent
        # Never ran: the same STATUS a cancelled queued request gets.
        outbox.extend(encode_frame(STATUS, request_id, encode_status(None, signal.SIGKILL)))
        urgent = True

    def kill_all() -> None:
        for child in list(children.values()):
            try:
                os.kill(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            selector.unregister(child.fd)
            os.close(child.fd)
            os.waitpid(child.pid, 0)
        children.clear()

    try:
        while True:
            busy = bool(children or queue)
            if hold_until is not None:
                timeout = max(0.0, hold_until - clock())
            elif busy:
                timeout = None
            else:
                timeout = max(0.0, idle_s - (clock() - last_activity))
            events = selector.select(timeout)
            if not events and not busy and hold_until is None and clock() - last_activity >= idle_s:
                return 0
            for key, _mask in events:
                if key.data is None:  # stdin
                    data = os.read(in_fd, 65536)
                    if not data:
                        return 0
                    last_activity = clock()
                    for kind, request_id, body in reader.feed(data):
                        if kind == HELLO:
                            try:
                                limits = json.loads(body)
                                raw_children = limits["max_children"]
                                raw_idle = limits["idle_s"]
                                # type() not isinstance(): bool is an int.
                                if (
                                    type(raw_children) is not int
                                    or type(raw_idle) not in (int, float)
                                    or not math.isfinite(raw_idle)
                                ):
                                    raise ValueError("HELLO limits out of range")
                                max_children = max(1, raw_children)
                                idle_s = min(_MAX_IDLE_S, max(1.0, float(raw_idle)))
                            except (ValueError, KeyError, TypeError, OverflowError):
                                # `finally` below still reaps/kills any
                                # children and closes the selector.
                                return 3
                            hello = True
                        elif kind == REQUEST and hello:
                            if request_id in by_request or any(
                                item[0] == request_id for item in queue
                            ):
                                refuse(request_id)  # duplicate id: keep the first's mapping
                            elif len(children) < max_children:
                                start(request_id, body)
                            elif (
                                len(queue) >= _QUEUE_PER_CHILD * max_children
                                or queued_bytes + len(body)
                                > _QUEUE_BYTES_FACTOR * max_request_bytes
                            ):
                                refuse(request_id)  # queue full
                            else:
                                queue.append((request_id, body))
                                queued_bytes += len(body)
                                outbox.extend(encode_frame(BUSY, request_id, b""))
                                urgent = True
                        elif kind == CANCEL:
                            child = by_request.get(request_id)
                            if child is not None:
                                try:
                                    os.kill(child.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                                # STATUS follows the normal child-exit path
                                # (finish()) once its pipe hits EOF.
                            else:
                                remaining = [item for item in queue if item[0] != request_id]
                                if len(remaining) != len(queue):
                                    # Cancelled before it ever ran: it will
                                    # never hit finish(), so send its STATUS
                                    # here or the caller waits forever.
                                    outbox.extend(encode_frame(
                                        STATUS, request_id, encode_status(None, signal.SIGKILL),
                                    ))
                                    urgent = True
                                queue.clear()
                                queue.extend(remaining)
                                queued_bytes = sum(len(item[1]) for item in queue)
                else:
                    child: _Child = key.data
                    try:
                        chunk = os.read(child.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        finish(child)
                        last_activity = clock()
                        continue
                    if child.capped:
                        continue
                    child.sent += len(chunk)
                    if child.sent > max_response_bytes:
                        child.capped = True
                        os.kill(child.pid, signal.SIGKILL)
                        continue
                    child.partial += chunk
                    # One split per chunk instead of a partition-per-line
                    # loop: the old loop re-scanned (and re-copied, via
                    # bytes(child.partial)) the shrinking remainder once per
                    # line, which is O(n^2) on a chunk containing many
                    # lines. split() finds them all in a single O(n) pass;
                    # the last element (no trailing newline) is the new
                    # partial, in the same order as before.
                    lines = bytes(child.partial).split(b"\n")
                    child.partial = bytearray(lines[-1])
                    for line in lines[:-1]:
                        outbox.extend(encode_frame(LINE, child.request_id, line + b"\n"))
                    if lines[:-1] and hold_until is None:
                        hold_until = clock() + _COALESCE_S
            if outbox and (urgent or hold_until is None or clock() >= hold_until):
                flush()
                urgent, hold_until = False, None
    except FrameError:
        return 3
    finally:
        kill_all()
        selector.close()
