"""Bounded, allowlisted diagnostics; arbitrary child output is never retained."""

from __future__ import annotations

import os
import subprocess
import time
from collections import deque
from threading import RLock
from typing import Self

CHUNK_BYTES = 8192
MAX_ENTRIES = 128
_CATEGORIES = (
    (
        (b"out of memory", b"failed to allocate", b"cannot allocate"),
        "Insufficient memory. Reduce context, batch size, or GPU layers.",
    ),
    (
        (
            b"unknown argument",
            b"unrecognized option",
            b"invalid argument",
            b"invalid value for",
        ),
        "Unsupported option or value. Check tuning and the installed runtime version.",
    ),
    (
        (b"address already in use", b"failed to bind", b"cannot bind"),
        "Listener unavailable. Choose an unused host and port.",
    ),
    (
        (
            b"library not loaded",
            b"cannot open shared object",
            b"no cuda",
            b"failed to initialize",
            b"failed to load backend",
        ),
        "Backend or library unavailable. Check the runtime installation and GPU support.",
    ),
    (
        (b"failed to load model", b"error loading model", b"invalid magic"),
        "Model load failed. Verify the GGUF and runtime compatibility.",
    ),
    (
        (b"load_tensors", b"loading model", b"llama_model_loader"),
        "Loading model. API readiness has not yet been verified.",
    ),
)


class DiagnosticSink:
    """Thread-safe fixed messages only, owned by one exact launch claim."""

    def __init__(self) -> None:
        self._entries: deque[str] = deque(maxlen=MAX_ENTRIES)
        self._lock = RLock()

    def feed(self, chunk: bytes) -> None:
        """Classify a bounded transient chunk without retaining its contents."""
        if len(chunk) > CHUNK_BYTES + 128:
            raise ValueError("Diagnostic chunk exceeds the read bound.")
        lower = chunk.lower()
        with self._lock:
            for needles, message in _CATEGORIES:
                if any(needle in lower for needle in needles):
                    if not self._entries or self._entries[-1] != message:
                        self._entries.append(message)
                    break

    def exited(self, code: int) -> None:
        with self._lock:
            self._entries.append(f"Process exited (code={int(code)}).")

    def snapshot(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._entries)


class DiagnosticPump:
    """Drain both pipes fairly on the owning worker with no reader threads.

    Nonblocking pipe support on Windows requires Python 3.12, the app minimum.
    A descendant keeping a pipe open cannot keep this owner alive after exit.
    """

    def __init__(self, process: subprocess.Popen, sink: DiagnosticSink) -> None:
        self.process = process
        self.sink = sink
        self._streams = [
            stream for stream in (process.stdout, process.stderr) if stream is not None
        ]
        self._tails: dict[int, bytes] = {}
        try:
            for stream in self._streams:
                os.set_blocking(stream.fileno(), False)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        for stream in self._streams:
            stream.close()
        self._streams.clear()
        self._tails.clear()

    def _drain_round(self) -> bool:
        progressed = False
        for stream in self._streams:
            fd = stream.fileno()
            try:
                chunk = os.read(fd, CHUNK_BYTES)
            except BlockingIOError:
                continue
            if chunk:
                progressed = True
                self.sink.feed(self._tails.get(fd, b"") + chunk)
                self._tails[fd] = chunk[-128:]
        return progressed

    def wait(self, *, timeout: float | None = None) -> int:
        """Drain until the direct child exits; timeout is for bounded fixtures."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        while self.process.poll() is None:
            progressed = self._drain_round()
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("diagnostic child", timeout)
            if not progressed:
                time.sleep(0.02)
        # Bound the final drain even if a descendant still writes to these pipes.
        for _ in range(32):
            if not self._drain_round():
                break
        code = self.process.wait()
        self.sink.exited(code)
        return code
