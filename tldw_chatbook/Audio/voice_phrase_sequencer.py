"""Attempt-fenced phrase speech over the app-owned duplex PCM transport."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Callable
import re
import time
from typing import Protocol

from tldw_chatbook.Audio.duplex_contracts import RenderSubmission
from tldw_chatbook.Audio.voice_process_protocol import (
    Delivery,
    Mailbox,
    ProtocolError,
    ReceiveStream,
    Record,
    StreamKey,
)
from tldw_chatbook.Audio.voice_process_types import (
    NormalizedPcmError,
    NormalizedPcmStream,
)


_ABBREVIATIONS = frozenset(
    {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc"}
)
_WORD_RE = re.compile(r"\S+")
_WS_RE = re.compile(r"\s+")
_MAX_AMBIGUOUS_MARKDOWN_CHARS = 8_192
_MAX_SPEAKABLE_CHARS = 32_768
_MAX_QUEUED_PHRASES = 128
_CANCEL_TIMEOUT_SECONDS = 0.5
_RENDER_RETRY_INTERVAL_SECONDS = 0.005
_RENDER_RETRY_TIMEOUT_SECONDS = 0.5


class _HandsFreeSynthesizer(Protocol):
    async def synthesize_hands_free(self, *, text: str) -> NormalizedPcmStream: ...


class _PcmSink(Protocol):
    def queue_render(self, pcm16: bytes) -> object | None: ...

    def fence_output(self) -> None: ...


class ProcessPcmStream:
    """Child-local frame custody with independent parent cleanup observation.

    The pipe dispatcher passes records from ``inbound`` to ``receive``. A frame
    yield resumes only after the sequencer copied it to the bounded render ring;
    only then may a whole packet's credit return. Local fencing discards current
    and late data immediately but never invents a parent resource-close receipt.
    """

    def __init__(
        self,
        key: StreamKey,
        inbound: Mailbox,
        *,
        cleanup_receiver: ReceiveStream,
        on_credit: Callable[[StreamKey, int, int], None],
        on_cancel: Callable[[StreamKey], None],
    ) -> None:
        if (
            key.lane != "pcm"
            or inbound.outbound
            or inbound.direction != "parent_to_child"
            or key.generation != inbound.generation
            or cleanup_receiver.key
            != StreamKey(key.generation, key.request_id, "tts_closed")
        ):
            raise ProtocolError()
        self.receiver = ReceiveStream(key)
        self._cleanup_receiver = cleanup_receiver
        self._inbound = inbound
        self._on_credit = on_credit
        self._on_cancel = on_cancel
        self._deliveries: deque[Delivery] = deque()
        self._offset = 0
        self._changed = asyncio.Event()
        self._closed = asyncio.Event()
        self._fenced = False
        self._failed = False
        self._claimed = False
        self._active = False
        self._cleanup_last_sequence: int | None = None
        self._end_last_sequence: int | None = None
        self.end_received = False

    @property
    def final_sequence(self) -> int | None:
        """Original cleanup boundary; never an invented consumption receipt."""
        return self._cleanup_last_sequence

    def _end_data(self, last_sequence: int) -> None:
        if self._end_last_sequence is None:
            self.receiver.end(last_sequence)
            self._end_last_sequence = last_sequence
        elif self._end_last_sequence != last_sequence:
            raise ProtocolError()

    def activate(self) -> None:
        """Select the sole PCM window; queued cleanup observers do not select it."""
        if self._active or self._fenced:
            raise ProtocolError()
        self._inbound.open_stream(self.receiver.key)
        self._active = True

    def normalized_stream(self) -> NormalizedPcmStream:
        """Give the phrase sequencer the sole consumer and a cleanup observer."""
        if self._claimed:
            raise ProtocolError()
        self._claimed = True
        return NormalizedPcmStream(self, self.wait_for_cleanup())

    def receive(self, record: Record) -> None:
        """Dispatch validated phrase data/control without waiting on playback."""
        if not self.receiver.key.matches(record):
            raise ProtocolError()
        op = record.header["op"]
        if op == "pcm":
            if not self._active:
                raise ProtocolError()
            delivery = self.receiver.receive(record, mailbox=self._inbound)
            self._deliveries.append(delivery)
            if self._fenced or delivery.discard_only:
                self.fence()
        elif op == "pcm_end":
            last_sequence = record.header["last_sequence"]
            if self.end_received or (
                self._cleanup_last_sequence is not None
                and self._cleanup_last_sequence != last_sequence
            ):
                raise ProtocolError()
            self._end_data(last_sequence)
            self.end_received = True
            self._inbound.release(record)
        elif op == "tts_closed":
            if self.receiver.cleanup_outcome is not None:
                raise ProtocolError()
            last_sequence = record.header["last_sequence"]
            if (
                self._end_last_sequence is not None
                and self._end_last_sequence != last_sequence
            ):
                raise ProtocolError()
            self._cleanup_last_sequence = last_sequence
            if self._fenced:
                self._end_data(last_sequence)
            delivery = self._cleanup_receiver.receive(record, mailbox=self._inbound)
            self.receiver.closed(record.header["outcome"])
            self._closed.set()
            sequence, size = delivery.consume()
            self._on_credit(self._cleanup_receiver.key, sequence, size)
            if self.receiver.cleanup_outcome != "clean":
                self.fail()
        else:
            raise ProtocolError()
        self._changed.set()

    def fence(self) -> None:
        """Discard matching old PCM now; parent cancellation is only a request."""
        first = not self._fenced
        if first and self._active and not self.receiver.data_complete:
            self._inbound.fence_stream(self.receiver.key)
        self._fenced = True
        if self._cleanup_last_sequence is not None:
            self._end_data(self._cleanup_last_sequence)
        self._offset = 0
        while self._deliveries:
            delivery = self._deliveries.popleft()
            sequence, size = delivery.discard(self.receiver.key)
            self._on_credit(self.receiver.key, sequence, size)
        self._changed.set()
        if first:
            self._on_cancel(self.receiver.key)

    def fail(self) -> None:
        """Wake/fence failed transport without pretending its owner has closed."""
        self._failed = True
        self.fence()

    async def wait_for_cleanup(self) -> None:
        """Observe actual categorical TTS closure, independent of data or device."""
        await self._closed.wait()
        if self.receiver.cleanup_outcome != "clean":
            raise NormalizedPcmError("synthesis_failed")

    def __aiter__(self) -> ProcessPcmStream:
        return self

    async def __anext__(self) -> bytes:
        while True:
            if self._failed:
                raise NormalizedPcmError("synthesis_failed")
            if self._fenced:
                raise StopAsyncIteration
            if self._deliveries:
                delivery = self._deliveries[0]
                if delivery.discard_only:
                    self.fence()
                    continue
                payload = delivery.record.payload
                if self._offset < len(payload):
                    frame = payload[self._offset : self._offset + 960]
                    self._offset += 960
                    return frame
                del payload
                self._deliveries.popleft()
                self._offset = 0
                sequence, size = delivery.consume()
                self._on_credit(self.receiver.key, sequence, size)
                continue
            if self.receiver.data_complete:
                raise StopAsyncIteration
            self._changed.clear()
            await self._changed.wait()

    async def aclose(self) -> None:
        """Stop consumption; only the separate receipt can finish parent cleanup."""
        if not self.receiver.data_complete:
            self.fence()


class PhraseSpeechSequencer:
    """Extract and speak safe Markdown prose for one immutable attempt epoch."""

    def __init__(
        self,
        *,
        epoch: int,
        synthesizer: _HandsFreeSynthesizer,
        sink: _PcmSink,
        clock: Callable[[], int] = time.monotonic_ns,
        fallback_word_threshold: int = 12,
        fallback_max_words: int = 24,
        fallback_delay_ns: int = 400_000_000,
        on_playback_started: Callable[[int], None] | None = None,
        on_failed: Callable[[int, str], None] | None = None,
        on_first_eligible_phrase: Callable[[int], None] | None = None,
        on_first_synthesis_complete: Callable[[int], None] | None = None,
        diagnostic_sink: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("attempt epoch must be non-negative")
        if (
            type(fallback_word_threshold) is not int
            or type(fallback_max_words) is not int
            or not 0 < fallback_word_threshold <= fallback_max_words
        ):
            raise ValueError("fallback word bounds are invalid")
        if type(fallback_delay_ns) is not int or fallback_delay_ns <= 0:
            raise ValueError("fallback delay must be positive")
        self._epoch = epoch
        self._synthesizer = synthesizer
        self._sink = sink
        self._clock = clock
        self._fallback_word_threshold = fallback_word_threshold
        self._fallback_max_words = fallback_max_words
        self._fallback_delay_ns = fallback_delay_ns
        self._on_playback_started = on_playback_started
        self._on_failed = on_failed
        self._on_first_eligible_phrase = on_first_eligible_phrase
        self._on_first_synthesis_complete = on_first_synthesis_complete
        self._diagnostic_sink = diagnostic_sink

        self._markdown_pending = ""
        self._fence_character: str | None = None
        self._fence_length = 0
        self._at_line_start = True
        self._speakable = ""
        self._speakable_started_ns: int | None = None
        self._phrases: deque[str] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._active_phrase = False
        self._input_finished = False
        self._cancelled = False
        self._failure_code: str | None = None
        self._playback_started = False
        self._failure_notified = False
        self._eligible_phrase_notified = False
        self._synthesis_complete_notified = False
        self._first_submission: RenderSubmission | None = None
        self._last_submission: RenderSubmission | None = None
        self._supervised_cleanup: set[asyncio.Task[object]] = set()

    @property
    def first_submission(self) -> RenderSubmission | None:
        """Return the first accepted render identity for paired receipt matching."""

        return self._first_submission

    @property
    def final_submission(self) -> RenderSubmission | None:
        """Expose the final identity only after all input and queueing finishes."""
        if (
            self._cancelled
            or self._failure_code is not None
            or not self._input_finished
            or self._phrases
            or self._active_phrase
        ):
            return None
        return self._last_submission

    @property
    def pending_phrase_count(self) -> int:
        """Return the bounded sequential phrase backlog."""

        return len(self._phrases)

    @property
    def synthesis_in_flight(self) -> bool:
        """Return whether the sole worker owns an active phrase."""

        return self._active_phrase

    @property
    def failure_code(self) -> str | None:
        """Return a content-free speech failure code, if speech stopped."""

        return self._failure_code

    @property
    def recoverable_failure(self) -> bool:
        """Return whether text generation may continue after speech failure."""

        return self._failure_code is not None

    @property
    def retained_character_count(self) -> int:
        """Return the content-free number of retained input characters."""

        return len(self._markdown_pending) + len(self._speakable)

    @property
    def supervised_cleanup_count(self) -> int:
        """Return the content-free count of cleanup tasks still quarantined."""

        return len(self._supervised_cleanup)

    async def feed(self, attempt_epoch: int, delta: str) -> bool:
        """Accept one current-attempt response delta and queue safe phrases."""

        if not self._accepts(attempt_epoch) or self._input_finished:
            return False
        if type(delta) is not str:
            raise TypeError("voice phrase delta must be text")
        if len(self._markdown_pending) + len(delta) > _MAX_AMBIGUOUS_MARKDOWN_CHARS:
            self._fail("speech_input_limit")
            return False
        self._markdown_pending += delta
        self._append_visible(self._consume_resolved_markdown())
        self._extract_punctuation_phrases()
        self._ensure_worker()
        return True

    async def poll(self, attempt_epoch: int) -> bool:
        """Apply the bounded time+word fallback without resolving ambiguity."""

        if not self._accepts(attempt_epoch) or self._input_finished:
            return False
        if self._markdown_pending or self._fence_character is not None:
            return True
        self._extract_fallback_phrase()
        self._ensure_worker()
        return True

    async def finish(self, attempt_epoch: int) -> bool:
        """Close phrase input and emit only a fully resolved final prose tail."""

        if not self._accepts(attempt_epoch) or self._input_finished:
            return False
        self._input_finished = True
        if self._markdown_pending or self._fence_character is not None:
            self._markdown_pending = ""
            self._speakable = ""
            self._speakable_started_ns = None
        else:
            phrase = _normalize_phrase(self._speakable)
            self._speakable = ""
            self._speakable_started_ns = None
            if phrase:
                self._enqueue_phrase(phrase)
        self._ensure_worker()
        return True

    def fence(self, attempt_epoch: int) -> bool:
        """Reject old producers and queued PCM before any asynchronous cleanup."""
        if not self._matches_epoch(attempt_epoch):
            return False
        self._sink.fence_output()
        self._cancelled = True
        self._phrases.clear()
        self._markdown_pending = ""
        self._speakable = ""
        self._speakable_started_ns = None
        return True

    async def cancel(self, attempt_epoch: int) -> bool:
        """Fence output and observe bounded cleanup of the active TTS stream."""
        if type(attempt_epoch) is not int or attempt_epoch != self._epoch:
            return False
        self.fence(attempt_epoch)
        worker = self._worker
        if worker is None or worker.done():
            return True
        if not worker.cancelling():
            worker.cancel()
        cleanup: set[asyncio.Task[object]] = {worker}
        try:
            done, pending = await asyncio.wait(
                cleanup,
                timeout=_CANCEL_TIMEOUT_SECONDS,
            )
        except BaseException:
            self._supervise_cleanup(cleanup)
            raise
        self._supervise_cleanup(pending)
        first_error: BaseException | None = None
        for task in done:
            if task.cancelled():
                continue
            error = task.exception()
            if error is not None:
                first_error = first_error or error
        if first_error is not None:
            raise first_error
        return True

    async def wait_for_cleanup(self) -> None:
        """Observe real cleanup after bounded cancellation has returned."""
        while self._supervised_cleanup:
            await asyncio.shield(
                asyncio.gather(*tuple(self._supervised_cleanup), return_exceptions=True)
            )

    def _matches_epoch(self, attempt_epoch: int) -> bool:
        return bool(
            type(attempt_epoch) is int
            and attempt_epoch == self._epoch
            and not self._cancelled
        )

    def _accepts(self, attempt_epoch: int) -> bool:
        return bool(
            type(attempt_epoch) is int
            and attempt_epoch == self._epoch
            and not self._cancelled
            and self._failure_code is None
        )

    def _append_visible(self, visible: str) -> None:
        if not visible:
            return
        if len(self._speakable) + len(visible) > _MAX_SPEAKABLE_CHARS:
            self._fail("speech_input_limit")
            return
        if not self._speakable:
            self._speakable_started_ns = self._clock()
        self._speakable += visible

    def _consume_resolved_markdown(self) -> str:
        source = self._markdown_pending
        visible: list[str] = []
        index = 0
        length = len(source)
        while index < length:
            if self._fence_character is not None:
                newline = source.find("\n", index)
                if newline < 0:
                    break
                line = source[index:newline]
                if _is_matching_fence_close(
                    line,
                    character=self._fence_character,
                    minimum_length=self._fence_length,
                ):
                    self._fence_character = None
                    self._fence_length = 0
                index = newline + 1
                self._at_line_start = True
                continue

            if self._at_line_start:
                remaining = source[index:]
                stripped = remaining.lstrip(" ")
                leading = len(remaining) - len(stripped)
                if (
                    stripped
                    and stripped[0] in "`~"
                    and set(stripped) == {stripped[0]}
                    and len(stripped) < 3
                ):
                    break
                fence = _opening_fence(stripped)
                if fence is not None:
                    newline = source.find("\n", index + leading)
                    if newline < 0:
                        break
                    self._fence_character, self._fence_length = fence
                    index = newline + 1
                    self._at_line_start = True
                    continue
                marker_end = _list_marker_end(stripped)
                if marker_end == -1:
                    break
                if marker_end:
                    index += leading + marker_end
                    self._at_line_start = False
                    continue

            character = source[index]
            if character == "`":
                delimiter_length = _run_length(source, index, "`")
                closing = _matching_inline_code_close(
                    source,
                    start=index + delimiter_length,
                    delimiter_length=delimiter_length,
                )
                if closing < 0:
                    break
                index = closing
                self._at_line_start = False
                continue
            if character == "[":
                label_end = source.find("]", index + 1)
                if label_end < 0 or label_end + 1 >= length:
                    break
                if source[label_end + 1] != "(":
                    visible.append(source[index : label_end + 1])
                    index = label_end + 1
                    self._at_line_start = False
                    continue
                url_end = source.find(")", label_end + 2)
                if url_end < 0:
                    break
                visible.append(source[index + 1 : label_end])
                index = url_end + 1
                self._at_line_start = False
                continue
            if character == "\n":
                visible.append(" ")
                index += 1
                self._at_line_start = True
                continue
            visible.append(character)
            index += 1
            if not character.isspace():
                self._at_line_start = False

        self._markdown_pending = source[index:]
        return "".join(visible)

    def _extract_punctuation_phrases(self) -> None:
        while True:
            boundary = _phrase_boundary(self._speakable)
            if boundary is None:
                return
            phrase = _normalize_phrase(self._speakable[: boundary + 1])
            self._speakable = self._speakable[boundary + 1 :].lstrip()
            self._speakable_started_ns = self._clock() if self._speakable else None
            if phrase:
                if not self._enqueue_phrase(phrase):
                    return

    def _extract_fallback_phrase(self) -> None:
        if self._speakable_started_ns is None:
            return
        words = list(_WORD_RE.finditer(self._speakable))
        if len(words) < self._fallback_word_threshold:
            return
        if self._clock() - self._speakable_started_ns < self._fallback_delay_ns:
            return
        final_word = words[min(len(words), self._fallback_max_words) - 1]
        phrase = _normalize_phrase(self._speakable[: final_word.end()])
        self._speakable = self._speakable[final_word.end() :].lstrip()
        self._speakable_started_ns = self._clock() if self._speakable else None
        if phrase:
            self._enqueue_phrase(phrase)

    def _ensure_worker(self) -> None:
        if self._phrases and (self._worker is None or self._worker.done()):
            self._worker = asyncio.create_task(self._drain_phrases())

    async def _drain_phrases(self) -> None:
        try:
            while self._phrases and self._accepts(self._epoch):
                phrase = self._phrases.popleft()
                self._active_phrase = True
                emitted_pcm = False
                try:
                    stream = await self._synthesizer.synthesize_hands_free(text=phrase)
                    cleanup = asyncio.ensure_future(stream.cleanup)
                    try:
                        async for frame in stream.frames:
                            if not self._accepts(self._epoch):
                                return
                            if type(frame) is not bytes or len(frame) != 960:
                                raise ValueError("invalid_normalized_pcm_frame")
                            if not await self._queue_render(frame):
                                return
                            emitted_pcm = True
                    finally:
                        close_frames = getattr(stream.frames, "aclose", None)
                        receipts = [cleanup]
                        if callable(close_frames):
                            receipts.append(asyncio.create_task(close_frames()))
                        await self._settle_stream_receipts(receipts)
                    if (
                        emitted_pcm
                        and self._accepts(self._epoch)
                        and not self._synthesis_complete_notified
                    ):
                        self._synthesis_complete_notified = True
                        self._notify(self._on_first_synthesis_complete, self._epoch)
                except asyncio.CancelledError:
                    raise
                except _OutputRejected:
                    self._fail("output_rejected")
                except NormalizedPcmError as error:
                    self._fail(error.code)
                except Exception as error:
                    # Exceptions can contain spoken text, paths, or provider
                    # credentials. Retain only the class for diagnosis.
                    if self._diagnostic_sink is not None:
                        with contextlib.suppress(Exception):
                            self._diagnostic_sink(
                                "tts_pipeline_failed",
                                {
                                    "status": "failed",
                                    "exception_type": type(error).__name__,
                                },
                            )
                    self._fail("synthesis_failed")
                finally:
                    self._active_phrase = False
        finally:
            self._worker = None

    @staticmethod
    async def _settle_stream_receipts(
        receipts: list[asyncio.Future[object]],
    ) -> None:
        group = asyncio.gather(*receipts, return_exceptions=True)
        cancelled = False
        while not group.done():
            try:
                await asyncio.shield(group)
            except asyncio.CancelledError:
                cancelled = True
        results = group.result()
        if cancelled:
            raise asyncio.CancelledError
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _fail(self, code: str) -> None:
        self._failure_code = code
        self._phrases.clear()
        self._markdown_pending = ""
        self._speakable = ""
        self._speakable_started_ns = None
        if not self._failure_notified:
            self._failure_notified = True
            self._notify(self._on_failed, self._epoch, code)

    def _enqueue_phrase(self, phrase: str) -> bool:
        if len(self._phrases) >= _MAX_QUEUED_PHRASES:
            self._fail("speech_input_limit")
            return False
        self._phrases.append(phrase)
        if not self._eligible_phrase_notified:
            self._eligible_phrase_notified = True
            self._notify(self._on_first_eligible_phrase, self._epoch)
        return True

    async def _queue_render(self, frame: bytes) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _RENDER_RETRY_TIMEOUT_SECONDS
        while self._accepts(self._epoch):
            receipt = self._sink.queue_render(frame)
            if receipt is not None:
                if isinstance(receipt, RenderSubmission):
                    if self._first_submission is None:
                        self._first_submission = receipt
                    self._last_submission = receipt
                if not self._playback_started:
                    self._playback_started = True
                    self._notify(self._on_playback_started, self._epoch)
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise _OutputRejected
            await asyncio.sleep(min(_RENDER_RETRY_INTERVAL_SECONDS, remaining))
        return False

    @staticmethod
    def _notify(callback: Callable[..., None] | None, *args: object) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            return

    def _supervise_cleanup(self, tasks: set[asyncio.Task[object]]) -> None:
        for task in tasks:
            if task.done():
                self._retire_cleanup(task)
                continue
            self._supervised_cleanup.add(task)
            task.add_done_callback(self._retire_cleanup)

    def _retire_cleanup(self, task: asyncio.Task[object]) -> None:
        self._supervised_cleanup.discard(task)
        if task.cancelled():
            return
        task.exception()


class _OutputRejected(RuntimeError):
    pass


def _list_marker_end(text: str) -> int:
    if not text:
        return 0
    if text[0] in "-*+":
        if len(text) == 1:
            return -1
        if text[1].isspace():
            return 2
        return 0
    if not text[0].isdigit():
        return 0
    end = 0
    while end < len(text) and text[end].isdigit():
        end += 1
    if end == len(text):
        return -1
    if text[end] not in ".)":
        return 0
    if end + 1 == len(text):
        return -1
    return end + 2 if text[end + 1].isspace() else 0


def _phrase_boundary(text: str) -> int | None:
    for index, character in enumerate(text):
        if character not in ".!?":
            continue
        if character == ".":
            if (
                (index > 0 and index + 1 < len(text))
                and text[index - 1].isdigit()
                and text[index + 1].isdigit()
            ):
                continue
            if (index > 0 and text[index - 1] == ".") or (
                index + 1 < len(text) and text[index + 1] == "."
            ):
                continue
            word = re.search(r"([A-Za-z]+)$", text[:index])
            if word and word.group(1).lower() in _ABBREVIATIONS:
                continue
        if index + 1 == len(text) and character in "!?":
            return index
        if index + 1 < len(text) and text[index + 1].isspace():
            return index
    return None


def _run_length(text: str, start: int, character: str) -> int:
    end = start
    while end < len(text) and text[end] == character:
        end += 1
    return end - start


def _opening_fence(text: str) -> tuple[str, int] | None:
    if not text or text[0] not in "`~":
        return None
    length = _run_length(text, 0, text[0])
    if length < 3:
        return None
    return text[0], length


def _is_matching_fence_close(
    line: str,
    *,
    character: str,
    minimum_length: int,
) -> bool:
    stripped = line.lstrip(" ")
    length = _run_length(stripped, 0, character)
    return bool(length >= minimum_length and not stripped[length:].strip())


def _matching_inline_code_close(
    text: str,
    *,
    start: int,
    delimiter_length: int,
) -> int:
    position = start
    while True:
        position = text.find("`", position)
        if position < 0:
            return -1
        length = _run_length(text, position, "`")
        if length == delimiter_length:
            return position + length
        position += length


def _normalize_phrase(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()
