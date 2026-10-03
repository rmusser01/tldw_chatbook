"""Sentence-chunked Console speech: many short requests, one continuous stream.

What this file has to prove, and why each assertion is here rather than a
narrower one:

* The WHOLE utterance is played -- every piece, in order, once. The failure
  mode this guards is the one `backlog/docs/lessons-testing-evidence.md`
  records for TASK-32027 ("Decode the entire utterance when validating
  playback"): chunked synthesis that keeps only the first chunk, or that
  concatenates separately-encoded segments into a file whose header describes
  only one of them. A process exit status and a non-silent first word do not
  establish that the rest survived, so the joined PCM is compared against the
  concatenation of every request's own payload, in request order.
* The pieces are joined with their WAV headers STRIPPED. If a header were fed
  to the sink as audio, the joined bytes would not equal the payloads.
* The ordinary single-request path is untouched for everything the chunked
  path does not claim: short text, a non-WAV format, and a machine with no
  sink.
* A mid-utterance failure is reported, never silently replayed from the top --
  by then the user has already heard audio, so falling back would say part of
  the reply twice.

Only synthesis and the audio device are replaced (`_RecordingSink` drives the
REAL `pump()`); admission, selection, format handling, event posting and the
sink wiring are the shipped ones.
"""

from __future__ import annotations

import asyncio
import io
import struct
import wave

import pytest

from Tests.TTS_Events.test_spoken_feedback_streaming import _RecordingSink
from tldw_chatbook.Event_Handlers.TTS_Events import tts_events
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.legacy_bridge import legacy_provider_specs
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import TTSService

RATE = 24000
#: Long enough to split into several pieces at the shipped
#: `_SPEECH_PIECE_MAX_TOKENS` budget, and deliberately made of many short
#: sentences so the split has real boundaries to find.
LONG_TEXT = " ".join(
    f"Sentence number {index} carries a few short words here." for index in range(1, 9)
)


def _payload_for(text: str) -> bytes:
    """A WAV payload unique to `text`, so playback can be traced back to it."""
    seed = text.encode("utf-8")[:1] or b"?"
    frame = struct.pack("<h", (seed[0] % 100) + 1)
    return frame * 400


def _wav(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(RATE)
        audio.writeframes(payload)
    return buffer.getvalue()


def _service(
    *,
    audio_format: str = "wav",
    fail_before_request: int | None = None,
    request_body: bytes | None = None,
):
    """A real legacy-admitted service whose synthesis yields a WAV per request.

    Args:
        audio_format: The format every request must ask for.
        fail_before_request: 1-based request index that raises instead of
            yielding, so a mid-utterance failure can be driven deterministically.
        request_body: Fixed response body, when a test needs a body that is not
            derived from the request text (e.g. a non-WAV payload).
    """
    requests = []

    class Backend:
        def set_progress_callback(self, callback):
            pass

        async def generate_speech_stream(self, request):
            requests.append(request)
            assert request.response_format == audio_format
            if fail_before_request is not None and len(requests) == fail_before_request:
                raise RuntimeError("synthetic provider failure")
            if request_body is not None:
                yield request_body
                return
            payload = _payload_for(request.input)
            yield _wav(payload)[:11]
            yield _wav(payload)[11:]

    class Manager:
        async def get_backend(self, internal_id):
            return Backend()

        async def close_all_backends(self):
            pass

    registry = TTSAdapterRegistry(
        specs=legacy_provider_specs(
            {"app_tts": {}},
            manager_factory=lambda _provider, _config: Manager(),
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=TTSPreferencesSnapshot(
            provider_id="openai",
            model_mode="exact",
            model_id="tts-1",
            voice_mode="exact",
            voice_id="alloy",
            response_format=audio_format,
            speed=1.0,
        ),
    )
    return service, requests


class _Handler(tts_events.TTSEventHandler):
    """The shipped handler with message posting captured instead of routed."""

    def __init__(self, service):
        super().__init__()
        self._tts_service = service
        self.messages = []

    async def post_message(self, message):
        self.messages.append(message)

    def notify(self, message, severity="information"):
        pass


class _SinkRegistry(_RecordingSink):
    """`_RecordingSink`, plus a list of every instance a test can inspect."""

    instances: list["_SinkRegistry"] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _SinkRegistry.instances.append(self)


def _completion(handler):
    events = [
        event
        for event in handler.messages
        if isinstance(event, tts_events.TTSCompleteEvent)
    ]
    assert len(events) == 1
    return events[0]


async def _run(handler, service, text):
    try:
        await handler._generate_tts(text, "chunked-turn", None)
    finally:
        await service.close()
        await service.wait_closed()
        # This test fakes the play path, which is also the path that schedules
        # artifact cleanup -- so the handler's own teardown has to run here,
        # while the interpreter is alive, or each test leaks its temp WAV into
        # an `__del__`-time secure delete where `open` is already gone
        # (lessons-testing-evidence.md, TASK-32013).
        await handler.cleanup_tts_resources()


@pytest.fixture(autouse=True)
def _sink_instances():
    _SinkRegistry.instances = []
    yield _SinkRegistry.instances
    _SinkRegistry.instances = []


@pytest.mark.asyncio
async def test_long_utterance_plays_every_piece_in_order_through_one_sink(monkeypatch):
    service, requests = _service()
    handler = _Handler(service)
    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", _SinkRegistry)

    await _run(handler, service, LONG_TEXT)

    # More than one request, i.e. the chunked path actually engaged.
    assert len(requests) > 1
    # ...and every request carried a piece of the same utterance.
    assert " ".join(request.input for request in requests).split() == LONG_TEXT.split()

    assert len(_SinkRegistry.instances) == 1  # one sink for the whole utterance
    sink = _SinkRegistry.instances[0]
    assert sink.opened_with == (RATE, 1)
    # THE completeness assertion: every piece's payload, in request order,
    # with each piece's WAV header stripped and nothing else dropped.
    assert b"".join(sink.fed) == b"".join(
        _payload_for(request.input) for request in requests
    )
    played = _completion(handler)
    assert played.error is None
    assert played.audio_file is None  # nothing downstream auto-plays it again


@pytest.mark.asyncio
async def test_short_utterance_keeps_the_single_request_path(monkeypatch):
    service, requests = _service()
    handler = _Handler(service)
    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", _SinkRegistry)

    await _run(handler, service, "A short reply.")

    assert len(requests) == 1
    assert requests[0].input == "A short reply."


@pytest.mark.asyncio
async def test_non_wav_request_keeps_the_single_request_path(monkeypatch):
    service, requests = _service(audio_format="pcm", request_body=b"\x01\x02" * 600)
    handler = _Handler(service)
    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", _SinkRegistry)

    await _run(handler, service, LONG_TEXT)

    assert len(requests) == 1


@pytest.mark.asyncio
async def test_no_sink_keeps_the_single_request_path(monkeypatch):
    service, requests = _service()
    handler = _Handler(service)
    monkeypatch.setattr(tts_events, "sink_available", lambda: False)

    await _run(handler, service, LONG_TEXT)

    assert len(requests) == 1


@pytest.mark.asyncio
async def test_mid_utterance_failure_is_reported_and_not_replayed(monkeypatch):
    service, requests = _service(fail_before_request=2)
    handler = _Handler(service)
    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", _SinkRegistry)

    await _run(handler, service, LONG_TEXT)

    # The first piece was requested, the second raised, and the path did NOT
    # then re-request the whole utterance (which would replay what was heard).
    assert len(requests) == 2
    assert {request.input for request in requests} != {LONG_TEXT}
    sink = _SinkRegistry.instances[0]
    assert b"".join(sink.fed) == _payload_for(requests[0].input)
    failed = _completion(handler)
    assert failed.error is not None


@pytest.mark.asyncio
async def test_sink_open_failure_falls_back_to_the_whole_utterance(monkeypatch):
    service, requests = _service()
    handler = _Handler(service)

    class CannotOpen(_SinkRegistry):
        _open_should_fail = True

    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", CannotOpen)

    await _run(handler, service, LONG_TEXT)

    # One piece was generated before the sink was opened, the sink failed, and
    # the utterance was then delivered through the ordinary whole-text request.
    assert requests[-1].input == LONG_TEXT
    assert not any(sink.fed for sink in _SinkRegistry.instances)
