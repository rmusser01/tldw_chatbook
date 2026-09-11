from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from Tests.Audio.fakes.fake_duplex_backend import (
    FakeDuplexBackend,
    ManualClock,
)
from tldw_chatbook.Audio.duplex_transport import DuplexAudioTransport
from tldw_chatbook.TTS.adapter_types import TTSAudioResponse


_FRAME_BYTES = 960


def _sequencer_type():
    try:
        from tldw_chatbook.Chat.voice_phrase_sequencer import PhraseSpeechSequencer
    except ModuleNotFoundError:
        pytest.fail("the speculative voice phrase sequencer is not implemented")
    return PhraseSpeechSequencer


@pytest.mark.asyncio
async def test_synthesis_failure_reports_only_exception_class(monkeypatch):
    from tldw_chatbook.Chat import voice_phrase_sequencer as module

    events = []
    monkeypatch.setattr(
        module, "persist_event", lambda *args, **kwargs: events.append((args, kwargs))
    )

    class FailingSynthesizer:
        async def synthesize_hands_free(self, *, text):
            raise RuntimeError("private speech and provider secret")

    sequencer = module.PhraseSpeechSequencer(
        epoch=1, synthesizer=FailingSynthesizer(), sink=None
    )
    await sequencer.feed(1, "Hello. ")
    await _wait_until(lambda: sequencer.failure_code is not None)
    assert events == [
        (
            ("speculative_voice", "tts_pipeline_failed"),
            {"status": "failed", "exception_type": "RuntimeError"},
        )
    ]
    assert sequencer.failure_code == "synthesis_failed"


class _Clock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, nanoseconds: int) -> None:
        self.now_ns += nanoseconds


class _ChunkStream:
    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        block_at_end: bool = False,
    ) -> None:
        self._chunks = list(chunks)
        self._block_at_end = block_at_end
        self._release = asyncio.Event()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self.closed:
            raise StopAsyncIteration
        if self._chunks:
            return self._chunks.pop(0)
        if not self._block_at_end:
            raise StopAsyncIteration
        await self._release.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True
        self._release.set()


def _response(
    stream: _ChunkStream,
    *,
    audio_format: str = "pcm",
    sample_rate: int | None = 48_000,
) -> TTSAudioResponse:
    return TTSAudioResponse(
        provider_id="openai",
        model_id="tts-test",
        audio_format=audio_format,
        content_type=f"audio/{audio_format}",
        byte_stream=stream,
        sample_rate=sample_rate,
        metadata={"channels": 1},
    )


@pytest.mark.asyncio
async def test_normalized_response_closes_when_cancelled_before_first_frame():
    from tldw_chatbook.Chat.voice_phrase_sequencer import _NormalizedSynthesizer

    byte_stream = _ChunkStream((bytes(_FRAME_BYTES),), block_at_end=True)

    class Synthesizer:
        async def synthesize_hands_free(self, *, text):
            return _response(byte_stream)

    normalized = await _NormalizedSynthesizer(Synthesizer()).synthesize_hands_free(
        text="Hello."
    )
    await normalized.frames.aclose()
    await normalized.cleanup
    assert byte_stream.closed


class _Synthesizer:
    def __init__(
        self,
        response_factory: Callable[[str], TTSAudioResponse] | None = None,
        *,
        block_first: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self.active = 0
        self.maximum_active = 0
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self._block_first = block_first
        self._response_factory = response_factory or (
            lambda _text: _response(_ChunkStream((bytes(_FRAME_BYTES),)))
        )

    async def synthesize_hands_free(self, *, text: str) -> TTSAudioResponse:
        self.calls.append(text)
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self._block_first and len(self.calls) == 1:
                self.first_started.set()
                await self.release_first.wait()
            return self._response_factory(text)
        finally:
            self.active -= 1


class _Sink:
    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.abort_count = 0

    def queue_render(self, pcm16: bytes) -> object:
        self.frames.append(bytes(pcm16))
        return object()

    def fence_output(self) -> None:
        self.abort_count += 1
        self.frames.clear()


class _BlockingCloseStream(_ChunkStream):
    def __init__(self, chunks: tuple[bytes, ...], *, natural_eof: bool = False) -> None:
        super().__init__(chunks, block_at_end=not natural_eof)
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.actual_closed = False

    async def aclose(self) -> None:
        self.closed = True
        self.close_started.set()
        await self.release_close.wait()
        self.actual_closed = True


@pytest.mark.asyncio
async def test_cancel_during_natural_response_close_retains_real_cleanup_owner():
    stream = _BlockingCloseStream((bytes(_FRAME_BYTES),), natural_eof=True)
    synthesizer = _Synthesizer(lambda _text: _response(stream))
    sequencer = _sequencer_type()(epoch=1, synthesizer=synthesizer, sink=_Sink())
    try:
        await sequencer.feed(1, "First phrase. Second phrase. ")
        await sequencer.finish(1)
        await asyncio.wait_for(stream.close_started.wait(), 1)
        await sequencer.cancel(1)
        assert sequencer.supervised_cleanup_count == 1
        assert sequencer.synthesis_in_flight
        assert synthesizer.calls == ["First phrase."]
        waiting = asyncio.create_task(sequencer.wait_for_cleanup())
        await asyncio.sleep(0)
        assert not waiting.done()
        assert not stream.actual_closed
        stream.release_close.set()
        await asyncio.wait_for(waiting, 1)
        assert stream.actual_closed
        assert synthesizer.calls == ["First phrase."]
    finally:
        stream.release_close.set()
        await sequencer.cancel(1)
        await sequencer.wait_for_cleanup()


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    pytest.fail("voice phrase sequencer did not settle")


@pytest.mark.asyncio
async def test_punctuation_phrases_synthesize_one_at_a_time_and_write_fifo_pcm() -> (
    None
):
    first = b"\x01\x00" * 480
    second = b"\x02\x00" * 480
    streams: list[_ChunkStream] = []

    def response_for(text: str) -> TTSAudioResponse:
        frame = first if text == "First phrase." else second
        stream = _ChunkStream((frame[:317], frame[317:]))
        streams.append(stream)
        return _response(stream)

    synthesizer = _Synthesizer(response_for, block_first=True)
    sink = _Sink()
    sequencer = _sequencer_type()(epoch=3, synthesizer=synthesizer, sink=sink)

    assert await sequencer.feed(3, "First phrase. Second phrase. ") is True
    await asyncio.wait_for(synthesizer.first_started.wait(), timeout=1)

    assert synthesizer.calls == ["First phrase."]
    assert sequencer.pending_phrase_count == 1
    synthesizer.release_first.set()
    await _wait_until(
        lambda: (
            sequencer.pending_phrase_count == 0 and not sequencer.synthesis_in_flight
        )
    )

    assert synthesizer.calls == ["First phrase.", "Second phrase."]
    assert synthesizer.maximum_active == 1
    assert sink.frames == [first, second]
    assert all(stream.closed for stream in streams)


@pytest.mark.asyncio
async def test_first_stage_callbacks_wait_for_phrase_eligibility_and_checked_cleanup() -> (
    None
):
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    stages: list[tuple[str, int]] = []
    calls = 0

    async def cleanup() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            cleanup_started.set()
            await release_cleanup.wait()

    def response_for(_text: str) -> TTSAudioResponse:
        response = _response(_ChunkStream((bytes(_FRAME_BYTES),)))
        response.add_cleanup(cleanup)
        return response

    sequencer = _sequencer_type()(
        epoch=18,
        synthesizer=_Synthesizer(response_for),
        sink=_Sink(),
        on_first_eligible_phrase=lambda epoch: stages.append(("phrase", epoch)),
        on_first_synthesis_complete=lambda epoch: stages.append(("synthesis", epoch)),
    )

    await sequencer.feed(18, "First phrase")
    await asyncio.sleep(0)
    assert stages == []
    await sequencer.feed(18, ". Second phrase. ")
    await asyncio.wait_for(cleanup_started.wait(), 1)
    assert stages == [("phrase", 18)]
    release_cleanup.set()
    await _wait_until(lambda: not sequencer.synthesis_in_flight)

    assert stages == [("phrase", 18), ("synthesis", 18)]


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_zero_byte_or_failed_synthesis_has_no_completion_stage(
    cleanup_fails: bool,
) -> None:
    stages: list[int] = []

    async def cleanup() -> None:
        if cleanup_fails:
            raise RuntimeError("private cleanup failure")

    response = _response(_ChunkStream((bytes(_FRAME_BYTES),) if cleanup_fails else ()))
    response.add_cleanup(cleanup)
    sequencer = _sequencer_type()(
        epoch=19,
        synthesizer=_Synthesizer(lambda _text: response),
        sink=_Sink(),
        on_first_synthesis_complete=stages.append,
    )

    await sequencer.feed(19, "Empty phrase. ")
    await _wait_until(lambda: not sequencer.synthesis_in_flight)

    assert stages == []


@pytest.mark.asyncio
async def test_resolved_markdown_uses_bounded_word_and_time_fallback() -> None:
    clock = _Clock()
    synthesizer = _Synthesizer()
    sequencer = _sequencer_type()(
        epoch=4,
        synthesizer=synthesizer,
        sink=_Sink(),
        clock=clock,
        fallback_word_threshold=5,
        fallback_max_words=8,
        fallback_delay_ns=100,
    )

    assert (
        await sequencer.feed(
            4,
            "- This is a resolved [Markdown link](https://example.test) phrase "
            "without punctuation and extra words",
        )
        is True
    )
    await sequencer.poll(4)
    assert synthesizer.calls == []

    clock.advance(100)
    await sequencer.poll(4)
    await _wait_until(lambda: len(synthesizer.calls) == 1)

    assert synthesizer.calls == ["This is a resolved Markdown link phrase without"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fragment",
    (
        "```python\nprint('never speak. this')",
        "~~~python\nprint('never speak. this') ",
        "Use `never speak. this",
        "Use ``never speak. this. ",
        "Read [never speak. this](https://unfinished",
        "1.",
        "1)",
        "-",
    ),
)
async def test_unresolved_markdown_never_emits_punctuation_or_fallback(
    fragment: str,
) -> None:
    clock = _Clock()
    synthesizer = _Synthesizer()
    sequencer = _sequencer_type()(
        epoch=5,
        synthesizer=synthesizer,
        sink=_Sink(),
        clock=clock,
        fallback_word_threshold=1,
        fallback_max_words=4,
        fallback_delay_ns=1,
    )

    await sequencer.feed(5, fragment)
    clock.advance(1)
    await sequencer.poll(5)
    await asyncio.sleep(0)

    assert synthesizer.calls == []
    assert sequencer.pending_phrase_count == 0


@pytest.mark.asyncio
async def test_commonmark_fence_run_stays_silent_until_matching_close() -> None:
    synthesizer = _Synthesizer()
    sequencer = _sequencer_type()(epoch=6, synthesizer=synthesizer, sink=_Sink())

    await sequencer.feed(6, "~~~~python\nHidden sentence. \n~~~\nStill hidden. ")
    await asyncio.sleep(0)
    assert synthesizer.calls == []

    await sequencer.feed(6, "\n~~~~\nVisible now!")
    await _wait_until(lambda: len(synthesizer.calls) == 1)

    assert synthesizer.calls == ["Visible now!"]


@pytest.mark.asyncio
async def test_inline_code_run_split_across_deltas_never_leaks_code() -> None:
    synthesizer = _Synthesizer()
    sequencer = _sequencer_type()(epoch=7, synthesizer=synthesizer, sink=_Sink())

    await sequencer.feed(7, "Use ``never speak. ")
    await asyncio.sleep(0)
    assert synthesizer.calls == []

    await sequencer.feed(7, "this`` safely!")
    await _wait_until(lambda: len(synthesizer.calls) == 1)

    assert synthesizer.calls == ["Use safely!"]


@pytest.mark.asyncio
async def test_ordered_parenthesis_marker_split_across_deltas_is_not_spoken() -> None:
    synthesizer = _Synthesizer()
    sequencer = _sequencer_type()(epoch=8, synthesizer=synthesizer, sink=_Sink())

    await sequencer.feed(8, "1)")
    await asyncio.sleep(0)
    assert synthesizer.calls == []

    await sequencer.feed(8, " Item ready!")
    await _wait_until(lambda: len(synthesizer.calls) == 1)

    assert synthesizer.calls == ["Item ready!"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fragment", ("Are you ready?", "Stop now!"))
async def test_unambiguous_terminal_punctuation_emits_at_buffer_end(
    fragment: str,
) -> None:
    synthesizer = _Synthesizer()
    sequencer = _sequencer_type()(epoch=9, synthesizer=synthesizer, sink=_Sink())

    await sequencer.feed(9, fragment)
    await _wait_until(lambda: len(synthesizer.calls) == 1)

    assert synthesizer.calls == [fragment]


@pytest.mark.asyncio
async def test_stale_epoch_feed_and_cancel_cannot_publish_or_abort_current_audio() -> (
    None
):
    synthesizer = _Synthesizer()
    sink = _Sink()
    sequencer = _sequencer_type()(epoch=7, synthesizer=synthesizer, sink=sink)

    assert await sequencer.feed(6, "Obsolete phrase. ") is False
    assert await sequencer.cancel(6) is False
    assert sink.abort_count == 0

    assert await sequencer.feed(7, "Current phrase. ") is True
    await _wait_until(lambda: len(synthesizer.calls) == 1)
    assert synthesizer.calls == ["Current phrase."]


@pytest.mark.asyncio
async def test_cancel_fences_writes_closes_lazy_stream_aborts_and_discards_queue() -> (
    None
):
    stream = _ChunkStream((bytes(_FRAME_BYTES),), block_at_end=True)
    synthesizer = _Synthesizer(lambda _text: _response(stream))
    sink = _Sink()
    sequencer = _sequencer_type()(epoch=9, synthesizer=synthesizer, sink=sink)

    await sequencer.feed(9, "First phrase. Second phrase. ")
    await _wait_until(lambda: len(sink.frames) == 1)
    assert sequencer.pending_phrase_count == 1

    assert await sequencer.cancel(9) is True

    assert sink.abort_count == 1
    assert sink.frames == []
    assert sequencer.pending_phrase_count == 0
    assert sequencer.synthesis_in_flight is False
    assert stream.closed is True


@pytest.mark.asyncio
async def test_cancel_after_stream_failure_still_aborts_queued_audio() -> None:
    frame = b"\x05\x00" * 480
    stream = _ChunkStream((frame, frame, b"\x00"))
    sink = _Sink()
    sequencer = _sequencer_type()(
        epoch=10,
        synthesizer=_Synthesizer(lambda _text: _response(stream)),
        sink=sink,
    )

    await sequencer.feed(10, "Malformed stream phrase. ")
    await _wait_until(lambda: sequencer.failure_code is not None)
    assert sink.frames

    assert await sequencer.cancel(10) is True
    assert sink.abort_count == 1
    assert sink.frames == []


@pytest.mark.asyncio
async def test_cancel_bounds_and_supervises_uncooperative_cleanup() -> None:
    stream = _BlockingCloseStream((bytes(_FRAME_BYTES), bytes(_FRAME_BYTES)))
    sink = _Sink()
    sequencer = _sequencer_type()(
        epoch=11,
        synthesizer=_Synthesizer(lambda _text: _response(stream)),
        sink=sink,
    )
    await sequencer.feed(11, "Blocked cleanup phrase. ")
    await _wait_until(lambda: bool(sink.frames))
    loop = asyncio.get_running_loop()
    loop.call_later(0.8, stream.release_close.set)

    try:
        started = loop.time()
        assert await sequencer.cancel(11) is True
        elapsed = loop.time() - started

        assert elapsed < 0.7
        assert sink.frames == []  # output fencing is synchronous
        assert sequencer.supervised_cleanup_count == 1
        settled = asyncio.create_task(sequencer.wait_for_cleanup())
        await asyncio.sleep(0)
        assert not settled.done()
        assert await sequencer.feed(11, "Late output must stay fenced. ") is False
        stream.release_close.set()
        await _wait_until(lambda: sequencer.supervised_cleanup_count == 0)
        await settled
        assert sink.frames == []
    finally:
        stream.release_close.set()


@pytest.mark.asyncio
async def test_unsupported_audio_is_recoverable_and_never_uses_external_playback() -> (
    None
):
    stream = _ChunkStream((b"compressed",))
    synthesizer = _Synthesizer(
        lambda _text: _response(
            stream,
            audio_format="mp3",
            sample_rate=None,
        )
    )
    sink = _Sink()
    sequencer = _sequencer_type()(epoch=11, synthesizer=synthesizer, sink=sink)

    await sequencer.feed(11, "First phrase. Second phrase. ")
    await _wait_until(lambda: sequencer.failure_code is not None)

    assert sequencer.failure_code == "unsupported_audio_format"
    assert sequencer.recoverable_failure is True
    assert synthesizer.calls == ["First phrase."]
    assert sink.frames == []
    assert stream.closed is True


@pytest.mark.asyncio
async def test_finish_emits_final_safe_prose_without_splitting_abbreviation() -> None:
    synthesizer = _Synthesizer()
    sequencer = _sequencer_type()(epoch=12, synthesizer=synthesizer, sink=_Sink())

    await sequencer.feed(12, "Dr. Smith finished.")
    await asyncio.sleep(0)
    assert synthesizer.calls == []

    assert await sequencer.finish(12) is True
    await _wait_until(lambda: len(synthesizer.calls) == 1)

    assert synthesizer.calls == ["Dr. Smith finished."]


@pytest.mark.asyncio
async def test_render_lifecycle_exposes_final_submission_only_after_input_closes() -> (
    None
):
    from tldw_chatbook.Audio.duplex_contracts import RenderSubmission

    events: list[tuple[str, int, RenderSubmission | str | None]] = []

    class SubmissionSink(_Sink):
        def queue_render(self, pcm16):
            super().queue_render(pcm16)
            return RenderSubmission(3, 7, len(self.frames) - 1)

    sink = SubmissionSink()
    sequencer = _sequencer_type()(
        epoch=12,
        synthesizer=_Synthesizer(),
        sink=sink,
        on_playback_started=lambda epoch: events.append(("started", epoch, None)),
        on_failed=lambda epoch, code: events.append(("failed", epoch, code)),
    )

    await sequencer.feed(12, "A rendered phrase. ")
    await _wait_until(lambda: len(sink.frames) == 1)
    assert sequencer.first_submission == RenderSubmission(3, 7, 0)
    assert sequencer.final_submission is None
    await sequencer.finish(12)
    await _wait_until(lambda: sequencer.final_submission is not None)

    assert events == [
        ("started", 12, None),
    ]
    assert sequencer.final_submission == RenderSubmission(3, 7, 0)
    await sequencer.cancel(12)
    assert sequencer.final_submission is None


@pytest.mark.asyncio
async def test_private_pcm_sink_seam_matches_the_concrete_duplex_transport() -> None:
    frame = b"\x03\x00" * 480
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(
        backend=backend,
        clock=ManualClock(),
    )
    synthesizer = _Synthesizer(lambda _text: _response(_ChunkStream((frame,))))
    sequencer = _sequencer_type()(
        epoch=13,
        synthesizer=synthesizer,
        sink=transport,
    )
    await transport.start()

    try:
        await sequencer.feed(13, "Concrete transport phrase. ")
        await _wait_until(
            lambda: len(synthesizer.calls) == 1 and not sequencer.synthesis_in_flight
        )

        assert backend.stream.emit_capture(bytes(_FRAME_BYTES)) == frame
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_real_transport_backpressure_plays_every_frame_in_a_long_phrase() -> None:
    frames = tuple(bytes((index + 1, 0)) * 480 for index in range(100))
    backend = FakeDuplexBackend()
    clock = ManualClock()
    transport = DuplexAudioTransport(backend=backend, clock=clock)
    sequencer = _sequencer_type()(
        epoch=14,
        synthesizer=_Synthesizer(lambda _text: _response(_ChunkStream(frames))),
        sink=transport,
    )
    await transport.start()
    played: list[bytes] = []

    async def drain_render_ring() -> None:
        while len(played) < len(frames):
            clock.advance(10_000_000)
            rendered = backend.stream.emit_capture(bytes(_FRAME_BYTES))
            if rendered != bytes(_FRAME_BYTES):
                played.append(rendered)
            await asyncio.sleep(0.001)

    try:
        await sequencer.feed(14, "Long transport phrase. ")
        await _wait_until(lambda: transport.render_overflows > 0)
        await asyncio.wait_for(drain_render_ring(), timeout=1.0)
        await _wait_until(
            lambda: (
                sequencer.pending_phrase_count == 0
                and not sequencer.synthesis_in_flight
            )
        )

        assert played == list(frames)
        assert sequencer.failure_code is None
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_phrase_and_markdown_buffers_fail_closed_at_bounded_sizes() -> None:
    blocked = _Synthesizer(block_first=True)
    phrase_sequencer = _sequencer_type()(
        epoch=15,
        synthesizer=blocked,
        sink=_Sink(),
    )
    for _ in range(10_000):
        if not await phrase_sequencer.feed(15, "A! "):
            break

    assert phrase_sequencer.failure_code == "speech_input_limit"
    assert phrase_sequencer.pending_phrase_count == 0
    assert phrase_sequencer.retained_character_count == 0

    markdown_sequencer = _sequencer_type()(
        epoch=16,
        synthesizer=_Synthesizer(),
        sink=_Sink(),
    )
    await markdown_sequencer.feed(16, "[" + "x" * 200_000)

    assert markdown_sequencer.failure_code == "speech_input_limit"
    assert markdown_sequencer.retained_character_count == 0

    finishing_sequencer = _sequencer_type()(
        epoch=17,
        synthesizer=_Synthesizer(block_first=True),
        sink=_Sink(),
    )
    for _ in range(128):
        assert await finishing_sequencer.feed(17, "A! ") is True
    assert await finishing_sequencer.feed(17, "final tail") is True
    assert await finishing_sequencer.finish(17) is True

    assert finishing_sequencer.failure_code == "speech_input_limit"
    assert finishing_sequencer.pending_phrase_count == 0
