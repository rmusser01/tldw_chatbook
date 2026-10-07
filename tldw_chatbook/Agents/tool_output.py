"""Optional, bounded, session-only output observation (ADR-205)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

MAX_TOOL_OUTPUT_CHARS = 16_000
_OUTPUT_INTERVAL_SECONDS = 0.1
_TRUNCATED = "\n… live output truncated"
ToolOutputSink = Callable[[str, str], None]
_sink: ContextVar[ToolOutputSink | None] = ContextVar("tool_output_sink", default=None)


def current_tool_output_sink() -> ToolOutputSink | None:
    """Return this execution's optional text observer."""
    return _sink.get()


@contextmanager
def bind_tool_output_sink(sink: ToolOutputSink | None) -> Iterator[None]:
    """Bind only output observation when crossing an existing worker boundary."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


class _OutputBuffer:
    def __init__(self, observe: Callable[[str], None]) -> None:
        self.observe = observe
        self.parts: dict[str, str] = {}
        self.truncated = False
        self.condition = threading.Condition()
        self.closing = False
        self.retired = False
        self.dirty = False
        self.worker = threading.Thread(
            target=self.deliver, name="tool-output-display", daemon=True
        )
        self.worker.start()

    def add(self, channel: str, text: str) -> None:
        if (
            channel not in {"stdout", "stderr", "progress"}
            or not isinstance(text, str)
            or not text
        ):
            return
        with self.condition:
            if self.closing:
                return
            previous = "" if channel == "progress" else self.parts.get(channel, "")
            room = (
                MAX_TOOL_OUTPUT_CHARS
                - sum(len(v) for k, v in self.parts.items() if k != channel)
                - len(previous)
            )
            self.parts[channel] = previous + text[: max(0, room)]
            self.truncated |= len(text) > max(0, room)
            self.dirty = True
            self.condition.notify()

    def deliver(self) -> None:
        last_flush = 0.0
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.dirty or self.closing)
                if self.retired or (self.closing and not self.dirty):
                    return
                delay = _OUTPUT_INTERVAL_SECONDS - (time.monotonic() - last_flush)
                if not self.closing and delay > 0:
                    self.condition.wait(delay)
                    continue
                self.dirty = False
                text = "\n\n".join(
                    f"{key}\n{value}" for key, value in self.parts.items()
                )
                if self.truncated or len(text) > MAX_TOOL_OUTPUT_CHARS:
                    text = text[: MAX_TOOL_OUTPUT_CHARS - len(_TRUNCATED)] + _TRUNCATED
            # Never hold a producer/retirement lock across an optional observer.
            last_flush = time.monotonic()
            try:
                self.observe(text)
            except Exception:  # noqa: BLE001, S110 — no tool-body diagnostics
                pass

    def close(self) -> None:
        with self.condition:
            self.closing = True
            self.condition.notify()
        # Best-effort final flush has one refresh interval of grace. A wedged
        # display cannot make timeout/cancellation wait for callback completion.
        self.worker.join(_OUTPUT_INTERVAL_SECONDS)
        with self.condition:
            self.retired = True
            self.dirty = False
            self.condition.notify()


@contextmanager
def tool_output_scope(observe: Callable[[str], None] | None) -> Iterator[None]:
    """Flush pending text with at most 100ms grace; reject abandoned-worker output.

    Args:
        observe: Snapshot consumer; None keeps producers on the final-only path.
    """
    try:
        buffer = _OutputBuffer(observe) if observe is not None else None
    except Exception:  # noqa: BLE001 — optional display falls back to final-only
        buffer = None
    with bind_tool_output_sink(buffer.add if buffer is not None else None):
        try:
            yield
        finally:
            if buffer is not None:
                buffer.close()
