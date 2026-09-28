"""Fork-server parent for the SSH session worker (stdlib-only; bundled).

One parent per session: reads frames from ``in_fd``, forks a child per
REQUEST (the child runs ``run_request`` with stdin=/dev/null and stdout
= a pipe back here), relays the child's lines as LINE frames, reaps it,
then sends STATUS. Single-threaded by design: forking a threaded
process is unsafe, and the per-request watchdog lives in the child.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import time
from collections import deque
from typing import BinaryIO, Callable

from tldw_chatbook.Tools.remote_session_frames import (
    BUSY, CANCEL, HELLO, LINE, REQUEST, STATUS, FrameError, FrameReader,
    encode_frame, encode_status,
)

_CHILD_CRASH_EXIT = 70


class _Child:
    __slots__ = ("pid", "fd", "request_id", "partial", "sent", "capped")

    def __init__(self, pid: int, fd: int, request_id: int) -> None:
        self.pid, self.fd, self.request_id = pid, fd, request_id
        self.partial = bytearray()
        self.sent = 0
        self.capped = False


def _close_fds_above_2(keep: int) -> None:
    try:
        max_fd = os.sysconf("SC_OPEN_MAX")
    except (AttributeError, ValueError, OSError):
        max_fd = 1024
    os.closerange(3, keep)
    os.closerange(keep + 1, max(max_fd, keep + 2))


def _spawn(raw: bytes, request_id: int, run_request: Callable[[bytes, BinaryIO], int]) -> _Child:
    read_fd, write_fd = os.pipe()
    pid = os.fork()
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
    sets the concurrency cap and the idle-exit timeout; frames before
    ``HELLO`` are ignored. Each ``REQUEST`` forks a child that runs
    ``run_request(body, out)`` with stdin bound to ``/dev/null`` and
    stdout bound to a pipe back to this loop; every line the child writes
    is relayed as a ``LINE`` frame, and once the child exits a ``STATUS``
    frame reports its exit code and signal. A ``REQUEST`` received while
    ``max_children`` children are already running is queued and echoed
    back as a ``BUSY`` frame; it starts once a slot frees up. A ``CANCEL``
    kills the matching live child (or drops it from the queue) with
    SIGKILL. A child whose combined output exceeds ``max_response_bytes``
    is killed and its STATUS still reports the kill signal. The loop is
    single-threaded: forking a multi-threaded process is unsafe, and any
    per-request timeout is the child's own responsibility, not the
    parent's.

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
        violates the codec's size cap.

    Raises:
        OSError: If a low-level file descriptor operation (fork, pipe,
            read, write) fails for a reason other than the cases already
            handled above.
    """
    reader = FrameReader(max_body=max_request_bytes)
    selector = selectors.DefaultSelector()
    selector.register(in_fd, selectors.EVENT_READ, None)
    children: dict[int, _Child] = {}          # fd -> child
    by_request: dict[int, _Child] = {}
    queue: deque[tuple[int, bytes]] = deque()
    max_children, idle_s, hello = 8, 60.0, False
    last_activity = clock()
    outbox = bytearray()

    def flush() -> None:
        view = memoryview(outbox)
        while view:
            written = os.write(out_fd, view)
            view = view[written:]
        view.release()  # drop the last (empty) slice's export before clear()
        outbox.clear()

    def start(request_id: int, raw: bytes) -> None:
        child = _spawn(raw, request_id, run_request)
        children[child.fd] = child
        by_request[request_id] = child
        selector.register(child.fd, selectors.EVENT_READ, child)

    def finish(child: _Child) -> None:
        selector.unregister(child.fd)
        os.close(child.fd)
        if child.partial and not child.capped:
            outbox.extend(encode_frame(LINE, child.request_id, bytes(child.partial)))
        _, status = os.waitpid(child.pid, 0)
        exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else None
        signal_no = os.WTERMSIG(status) if os.WIFSIGNALED(status) else None
        outbox.extend(encode_frame(STATUS, child.request_id, encode_status(exit_code, signal_no)))
        del children[child.fd]
        by_request.pop(child.request_id, None)
        while queue and len(children) < max_children:
            start(*queue.popleft())

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
            timeout = None if busy else max(0.0, idle_s - (clock() - last_activity))
            events = selector.select(timeout)
            if not events and not busy and clock() - last_activity >= idle_s:
                return 0
            for key, _mask in events:
                if key.data is None:  # stdin
                    data = os.read(in_fd, 65536)
                    if not data:
                        return 0
                    last_activity = clock()
                    for kind, request_id, body in reader.feed(data):
                        if kind == HELLO:
                            limits = json.loads(body)
                            max_children = max(1, int(limits["max_children"]))
                            idle_s = max(1.0, float(limits["idle_s"]))
                            hello = True
                        elif kind == REQUEST and hello:
                            if len(children) < max_children:
                                start(request_id, body)
                            else:
                                queue.append((request_id, body))
                                outbox.extend(encode_frame(BUSY, request_id, b""))
                        elif kind == CANCEL:
                            child = by_request.get(request_id)
                            if child is not None:
                                try:
                                    os.kill(child.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                            remaining = [item for item in queue if item[0] != request_id]
                            queue.clear()
                            queue.extend(remaining)
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
                    while b"\n" in child.partial:
                        line, _, rest = bytes(child.partial).partition(b"\n")
                        outbox.extend(encode_frame(LINE, child.request_id, line + b"\n"))
                        child.partial = bytearray(rest)
            if outbox:
                flush()
    except FrameError:
        return 3
    finally:
        kill_all()
        selector.close()
