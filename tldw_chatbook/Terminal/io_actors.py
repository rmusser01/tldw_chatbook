"""Bounded input/output actors for persistent terminal sessions."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Event, Lock
from time import monotonic

from .contracts import (
    MAX_IO_CHUNK_BYTES,
    MAX_PARSER_TURN_BYTES,
    MAX_PARSER_TURN_SECONDS,
    MAX_PENDING_INPUT_BYTES,
    MAX_PENDING_OUTPUT_BYTES,
)
from tldw_chatbook.Utils.input_validation import (
    validate_terminal_key_input,
    validate_terminal_output_input,
    validate_terminal_paste_input,
    validate_terminal_reply_input,
    validate_terminal_resize_input,
)


BRACKETED_PASTE_START = b"\x1b[200~"
BRACKETED_PASTE_END = b"\x1b[201~"
MAX_REPLY_BYTES = 256
MAX_REPLY_RATE_BYTES = 4 * 1024
REPLY_RATE_WINDOW_SECONDS = 1.0
MAX_RESIZE_DEBOUNCE_SECONDS = 0.05
MAX_PARSER_SLICE_BYTES = 1024

_PASTE_TOO_LARGE_MESSAGE = "Paste refused because it exceeds the size limit."
_PASTE_BACKPRESSURE_MESSAGE = "Paste refused because terminal input is full."
_PASTE_CONTROL_MESSAGE = "Paste refused because it contains a prohibited control."
_INPUT_BACKPRESSURE_MESSAGE = "Input refused because terminal input is full."
_REPLY_TOO_LARGE_MESSAGE = "Terminal reply refused because it is too large."
_REPLY_RATE_LIMIT_MESSAGE = "Terminal reply refused because its rate limit was reached."


class InputEventKind(str, Enum):
    """Ordered terminal-input event kinds."""

    KEY = "key"
    PASTE = "paste"
    REPLY = "reply"


class InputRefusalReason(str, Enum):
    """Content-free reasons ordinary input was refused."""

    BACKPRESSURE = "input_backpressure"
    REPLY_TOO_LARGE = "reply_too_large"
    REPLY_RATE_LIMIT = "reply_rate_limit"


class PasteRefusalReason(str, Enum):
    """Content-free reasons an atomic paste was refused."""

    TOO_LARGE = "paste_too_large"
    BACKPRESSURE = "input_backpressure"
    PROHIBITED_CONTROL = "prohibited_control"


class OutputRefusalReason(str, Enum):
    """Content-free reasons terminal output was not admitted."""

    CHUNK_TOO_LARGE = "chunk_too_large"
    BACKPRESSURE = "output_backpressure"


@dataclass(frozen=True, slots=True)
class TerminalInputEvent:
    """One immutable ordered input envelope.

    Attributes:
        kind: Input source category.
        data: Complete transport bytes for the event.
        encoded_size: Precomputed byte length used for credit accounting.
    """

    kind: InputEventKind
    data: bytes
    encoded_size: int


@dataclass(frozen=True, slots=True)
class TerminalResize:
    """One latest-only terminal resize request.

    Attributes:
        columns: Requested terminal width.
        rows: Requested terminal height.
    """

    columns: int
    rows: int


@dataclass(frozen=True, slots=True)
class InputOfferResult:
    """Content-free input admission result.

    Attributes:
        accepted: Whether the complete event was admitted.
        reason: Structured refusal category when admission failed.
        safe_message: User-facing message that never includes input content.
    """

    accepted: bool = False
    reason: InputRefusalReason | PasteRefusalReason | None = None
    safe_message: str = ""


@dataclass(frozen=True, slots=True)
class OutputOfferResult:
    """Content-free output admission result.

    Attributes:
        accepted: Whether the complete output chunk was admitted.
        reason: Structured refusal category when admission failed.
    """

    accepted: bool = False
    reason: OutputRefusalReason | None = None


@dataclass(frozen=True, slots=True)
class ParserTurnResult:
    """Content-free accounting for one bounded parser turn.

    Attributes:
        processed_bytes: Bytes delivered to the parser this turn.
        processed_chunks: Parser calls made this turn.
        pending_bytes: Bytes still held by the output actor.
        refresh_requested: Whether this turn scheduled a visible refresh.
    """

    processed_bytes: int = 0
    processed_chunks: int = 0
    pending_bytes: int = 0
    refresh_requested: bool = False


class TerminalPriorityControl:
    """Independent, idempotent close signal that cannot wait behind I/O queues."""

    def __init__(self) -> None:
        self._event = Event()
        self._request_lock = Lock()

    @property
    def requested(self) -> bool:
        """Return whether priority close has been requested.

        Returns:
            Whether the independent close signal is set.
        """
        return self._event.is_set()

    def request_priority_close(self) -> bool:
        """Set the close signal and report whether this call set it first.

        Returns:
            ``True`` only for the first request.
        """
        with self._request_lock:
            if self._event.is_set():
                return False
            self._event.set()
            return True

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for priority close without consulting either I/O queue.

        Args:
            timeout: Maximum seconds to wait, or ``None`` to wait indefinitely.

        Returns:
            Whether the close signal was set before the timeout.
        """
        return self._event.wait(timeout)


class TerminalInputActor:
    """Thread-safe bounded queue for ordered terminal input events.

    Args:
        capacity_bytes: Input credit in bytes, from one byte through 512 KiB.
        clock: Injected monotonic clock used by reply-rate accounting.
        resize_debounce_seconds: Resize debounce from zero through 50 ms.

    Raises:
        TypeError: If a configured limit has the wrong type.
        ValueError: If a configured limit is outside its contract.
    """

    def __init__(
        self,
        capacity_bytes: int = MAX_PENDING_INPUT_BYTES,
        *,
        clock: Callable[[], float] | None = None,
        resize_debounce_seconds: float = 0.0,
    ) -> None:
        _validate_positive_limit(
            "capacity_bytes", capacity_bytes, MAX_PENDING_INPUT_BYTES
        )
        if not isinstance(resize_debounce_seconds, (int, float)) or isinstance(
            resize_debounce_seconds, bool
        ):
            raise TypeError("resize_debounce_seconds must be a number")
        if not 0.0 <= resize_debounce_seconds <= MAX_RESIZE_DEBOUNCE_SECONDS:
            raise ValueError("resize_debounce_seconds is outside contract")
        self.capacity_bytes = capacity_bytes
        self._clock = clock or monotonic
        self._resize_debounce_seconds = float(resize_debounce_seconds)
        self._queue: deque[TerminalInputEvent] = deque()
        self._pending_bytes = 0
        self._reply_usage: deque[tuple[float, int]] = deque()
        self._reply_usage_bytes = 0
        self._latest_resize: TerminalResize | None = None
        self._lock = Lock()

    @property
    def pending_bytes(self) -> int:
        """Return bytes currently consuming input credit.

        Returns:
            Number of admitted bytes waiting for transport.
        """
        with self._lock:
            return self._pending_bytes

    @property
    def pending_events(self) -> int:
        """Return complete ordered events waiting for transport.

        Returns:
            Number of queued key, paste, and reply envelopes.
        """
        with self._lock:
            return len(self._queue)

    def offer_key(self, data: bytes) -> InputOfferResult:
        """Atomically offer encoded key bytes.

        Args:
            data: Complete key encoding to enqueue.

        Returns:
            Content-free admission result.

        Raises:
            TypeError: If ``data`` is not immutable bytes.
        """
        data = validate_terminal_key_input(data).data
        with self._lock:
            return self._offer_locked(
                InputEventKind.KEY,
                data,
                InputRefusalReason.BACKPRESSURE,
                _INPUT_BACKPRESSURE_MESSAGE,
            )

    def offer_paste(self, text: str, *, bracketed: bool) -> InputOfferResult:
        """Validate and atomically offer one paste operation.

        Args:
            text: Untrusted paste text.
            bracketed: Whether to wrap accepted bytes in bracketed-paste markers.

        Returns:
            Content-free admission result.

        Raises:
            TypeError: If ``text`` is not text or ``bracketed`` is not boolean.
        """
        validated = validate_terminal_paste_input(text, bracketed)
        violation, payload = validated.classify()
        if violation == "too_large":
            return InputOfferResult(
                reason=PasteRefusalReason.TOO_LARGE,
                safe_message=_PASTE_TOO_LARGE_MESSAGE,
            )
        if violation == "prohibited_control":
            return InputOfferResult(
                reason=PasteRefusalReason.PROHIBITED_CONTROL,
                safe_message=_PASTE_CONTROL_MESSAGE,
            )
        with self._lock:
            return self._offer_paste_locked(payload, bracketed=validated.bracketed)

    def offer_reply(self, data: bytes) -> InputOfferResult:
        """Atomically offer a bounded terminal-protocol reply.

        Args:
            data: Complete fixed reply bytes.

        Returns:
            Content-free admission result.

        Raises:
            TypeError: If ``data`` is not immutable bytes.
        """
        data = validate_terminal_reply_input(data).data
        if not data:
            return InputOfferResult(accepted=True)
        if len(data) > MAX_REPLY_BYTES:
            return InputOfferResult(
                reason=InputRefusalReason.REPLY_TOO_LARGE,
                safe_message=_REPLY_TOO_LARGE_MESSAGE,
            )

        with self._lock:
            if self._pending_bytes + len(data) > self.capacity_bytes:
                return InputOfferResult(
                    reason=InputRefusalReason.BACKPRESSURE,
                    safe_message=_INPUT_BACKPRESSURE_MESSAGE,
                )
            now = self._clock()
            self._discard_expired_reply_usage_locked(now)
            if self._reply_usage_bytes + len(data) > MAX_REPLY_RATE_BYTES:
                return InputOfferResult(
                    reason=InputRefusalReason.REPLY_RATE_LIMIT,
                    safe_message=_REPLY_RATE_LIMIT_MESSAGE,
                )
            self._reply_usage.append((now, len(data)))
            self._reply_usage_bytes += len(data)
            return self._offer_locked(
                InputEventKind.REPLY,
                data,
                InputRefusalReason.BACKPRESSURE,
                _INPUT_BACKPRESSURE_MESSAGE,
            )

    def offer_resize(self, *, columns: int, rows: int) -> None:
        """Replace the pending resize with the newest valid dimensions.

        Args:
            columns: Requested terminal width.
            rows: Requested terminal height.

        Raises:
            ValueError: If either dimension is outside the terminal contract.
            TypeError: If either dimension is not an integer.
        """
        validated = validate_terminal_resize_input(columns, rows)
        with self._lock:
            self._latest_resize = TerminalResize(
                columns=validated.columns,
                rows=validated.rows,
            )

    def take_nowait(self) -> TerminalInputEvent | None:
        """Take the oldest complete input event without waiting.

        Returns:
            Oldest event, or ``None`` when the queue is empty.
        """
        with self._lock:
            if not self._queue:
                return None
            event = self._queue.popleft()
            self._pending_bytes -= event.encoded_size
            return event

    async def take_resize_debounced(self) -> TerminalResize | None:
        """Yield once, debounce briefly if configured, then take the latest resize.

        Returns:
            Latest pending dimensions, or ``None`` when no resize is pending.
        """
        await asyncio.sleep(self._resize_debounce_seconds)
        with self._lock:
            resize = self._latest_resize
            self._latest_resize = None
            return resize

    def _offer_locked(
        self,
        kind: InputEventKind,
        data: bytes,
        refusal_reason: InputRefusalReason | PasteRefusalReason,
        safe_message: str,
    ) -> InputOfferResult:
        """Enqueue one event while the actor lock is held."""
        encoded_size = len(data)
        if encoded_size == 0:
            return InputOfferResult(accepted=True)
        if self._pending_bytes + encoded_size > self.capacity_bytes:
            return InputOfferResult(reason=refusal_reason, safe_message=safe_message)
        self._queue.append(
            TerminalInputEvent(kind=kind, data=data, encoded_size=encoded_size)
        )
        self._pending_bytes += encoded_size
        return InputOfferResult(accepted=True)

    def _offer_paste_locked(
        self, payload: bytes, *, bracketed: bool
    ) -> InputOfferResult:
        """Reserve paste credit before constructing optional transport markers."""
        marker_size = (
            len(BRACKETED_PASTE_START) + len(BRACKETED_PASTE_END) if bracketed else 0
        )
        encoded_size = len(payload) + marker_size
        if encoded_size == 0:
            return InputOfferResult(accepted=True)
        if self._pending_bytes + encoded_size > self.capacity_bytes:
            return InputOfferResult(
                reason=PasteRefusalReason.BACKPRESSURE,
                safe_message=_PASTE_BACKPRESSURE_MESSAGE,
            )
        data = (
            BRACKETED_PASTE_START + payload + BRACKETED_PASTE_END
            if bracketed
            else payload
        )
        self._queue.append(
            TerminalInputEvent(
                kind=InputEventKind.PASTE,
                data=data,
                encoded_size=encoded_size,
            )
        )
        self._pending_bytes += encoded_size
        return InputOfferResult(accepted=True)

    def _discard_expired_reply_usage_locked(self, now: float) -> None:
        """Drop reply-rate entries outside the sliding window."""
        while (
            self._reply_usage
            and now - self._reply_usage[0][0] >= REPLY_RATE_WINDOW_SECONDS
        ):
            _, size = self._reply_usage.popleft()
            self._reply_usage_bytes -= size


@dataclass(frozen=True, slots=True)
class _OutputChunk:
    """One immutable admitted output chunk."""

    data: bytes
    encoded_size: int


class TerminalOutputActor:
    """Thread-safe bounded output queue with parser-turn budgets.

    Args:
        capacity_bytes: Output credit in bytes, from one byte through 512 KiB.
        max_chunk_bytes: Maximum backend chunk, up to 64 KiB.
        max_turn_bytes: Maximum parser work per turn, up to 256 KiB.
        max_turn_seconds: Maximum parser-turn duration, up to 8 ms.
        clock: Injected monotonic clock used by parser-turn accounting.

    Raises:
        TypeError: If a configured limit has the wrong type.
        ValueError: If a configured limit is outside its contract.
    """

    def __init__(
        self,
        capacity_bytes: int = MAX_PENDING_OUTPUT_BYTES,
        *,
        max_chunk_bytes: int | None = None,
        max_turn_bytes: int | None = None,
        max_turn_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        selected_chunk_bytes = (
            MAX_IO_CHUNK_BYTES if max_chunk_bytes is None else max_chunk_bytes
        )
        selected_turn_bytes = (
            MAX_PARSER_TURN_BYTES if max_turn_bytes is None else max_turn_bytes
        )
        selected_turn_seconds = (
            MAX_PARSER_TURN_SECONDS if max_turn_seconds is None else max_turn_seconds
        )
        _validate_positive_limit(
            "capacity_bytes", capacity_bytes, MAX_PENDING_OUTPUT_BYTES
        )
        _validate_positive_limit(
            "max_chunk_bytes", selected_chunk_bytes, MAX_IO_CHUNK_BYTES
        )
        _validate_positive_limit(
            "max_turn_bytes", selected_turn_bytes, MAX_PARSER_TURN_BYTES
        )
        if not isinstance(selected_turn_seconds, (int, float)) or isinstance(
            selected_turn_seconds, bool
        ):
            raise TypeError("max_turn_seconds must be a number")
        if not 0.0 < selected_turn_seconds <= MAX_PARSER_TURN_SECONDS:
            raise ValueError("max_turn_seconds is outside contract")
        self.capacity_bytes = capacity_bytes
        self.max_chunk_bytes = selected_chunk_bytes
        self.max_turn_bytes = selected_turn_bytes
        self.max_turn_seconds = float(selected_turn_seconds)
        self._clock = clock or monotonic
        self._queue: deque[_OutputChunk] = deque()
        self._pending_bytes = 0
        self._refresh_pending = False
        self._output_closed = False
        self._lock = Lock()
        self._parser_lock = Lock()

    @property
    def pending_bytes(self) -> int:
        """Return bytes queued or currently being delivered to the parser.

        Returns:
            Number of admitted bytes still consuming output credit.
        """
        with self._lock:
            return self._pending_bytes

    @property
    def pending_chunks(self) -> int:
        """Return chunks currently waiting for parser delivery.

        Returns:
            Number of queued output envelopes or remainders.
        """
        with self._lock:
            return len(self._queue)

    @property
    def read_credit_bytes(self) -> int:
        """Return output bytes that may currently be admitted.

        Returns:
            Remaining byte credit for backend output.
        """
        with self._lock:
            return self.capacity_bytes - self._pending_bytes

    @property
    def next_read_size(self) -> int:
        """Return the maximum safe size for the next backend read.

        Returns:
            Bounded next-read size, or zero when output credit is full.
        """
        with self._lock:
            return min(
                self.max_chunk_bytes,
                self.capacity_bytes - self._pending_bytes,
            )

    def offer_output(self, data: bytes) -> OutputOfferResult:
        """Atomically admit one complete backend-output chunk.

        Args:
            data: Complete bytes returned by one bounded backend read.

        Returns:
            Content-free admission result.

        Raises:
            TypeError: If ``data`` is not immutable bytes.
        """
        data = validate_terminal_output_input(data).data
        encoded_size = len(data)
        if encoded_size > self.max_chunk_bytes:
            return OutputOfferResult(reason=OutputRefusalReason.CHUNK_TOO_LARGE)
        with self._lock:
            if self._output_closed:
                return OutputOfferResult()
            if encoded_size == 0:
                return OutputOfferResult(accepted=True)
            if self._pending_bytes + encoded_size > self.capacity_bytes:
                return OutputOfferResult(reason=OutputRefusalReason.BACKPRESSURE)
            self._queue.append(_OutputChunk(data=data, encoded_size=encoded_size))
            self._pending_bytes += encoded_size
            return OutputOfferResult(accepted=True)

    def close_output(self) -> int:
        """Atomically close admission and return already-admitted pending bytes.

        Returns:
            Bytes admitted before the output-close boundary that still need parsing.
        """
        with self._lock:
            self._output_closed = True
            return self._pending_bytes

    def process_parser_turn(
        self,
        consumer: Callable[[bytes], None],
        *,
        visible: bool,
    ) -> ParserTurnResult:
        """Deliver output under exact byte and monotonic-time budgets.

        Args:
            consumer: Parser callback receiving each admitted byte slice.
            visible: Whether parsed output belongs to the visible terminal.

        Returns:
            Content-free accounting and refresh-coalescing result.

        Raises:
            TypeError: If ``consumer`` is not callable or ``visible`` is not boolean.
            Exception: Re-raises consumer failures after retiring the ambiguous
                attempted slice; failed parser callbacks must not be retried.
        """
        if not callable(consumer):
            raise TypeError("consumer must be callable")
        if type(visible) is not bool:
            raise TypeError("visible must be bool")

        processed_bytes = 0
        processed_chunks = 0
        with self._parser_lock:
            started_at = self._clock()
            while processed_bytes < self.max_turn_bytes:
                if (
                    processed_chunks
                    and self._clock() - started_at >= self.max_turn_seconds
                ):
                    break
                remaining_budget = self.max_turn_bytes - processed_bytes
                delivery_size = min(remaining_budget, MAX_PARSER_SLICE_BYTES)
                with self._lock:
                    if not self._queue:
                        break
                    chunk = self._queue.popleft()
                    delivered = chunk.data[:delivery_size]
                    remainder = chunk.data[delivery_size:]
                    if remainder:
                        self._queue.appendleft(
                            _OutputChunk(data=remainder, encoded_size=len(remainder))
                        )
                try:
                    consumer(delivered)
                finally:
                    with self._lock:
                        self._pending_bytes -= len(delivered)
                processed_bytes += len(delivered)
                processed_chunks += 1

            refresh_requested = False
            with self._lock:
                if processed_bytes and visible and not self._refresh_pending:
                    self._refresh_pending = True
                    refresh_requested = True
                pending_bytes = self._pending_bytes
        return ParserTurnResult(
            processed_bytes=processed_bytes,
            processed_chunks=processed_chunks,
            pending_bytes=pending_bytes,
            refresh_requested=refresh_requested,
        )

    def acknowledge_visible_refresh(self) -> bool:
        """Clear one coalesced visible-refresh request if present.

        Returns:
            Whether a pending refresh was acknowledged.
        """
        with self._lock:
            if not self._refresh_pending:
                return False
            self._refresh_pending = False
            return True


def _validate_positive_limit(name: str, value: int, maximum: int) -> None:
    """Validate one positive integer limit against its global ceiling."""
    if type(value) is not int:
        raise TypeError(f"{name} must be int")
    if not 0 < value <= maximum:
        raise ValueError(f"{name} is outside contract")
