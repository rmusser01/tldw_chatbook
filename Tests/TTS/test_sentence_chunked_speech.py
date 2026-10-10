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
import hashlib
import wave
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from Tests.TTS.test_console_speak_autoplay import _FakeApp
from Tests.TTS_Events.test_spoken_feedback_streaming import _RecordingSink
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
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
    return hashlib.sha256(text.encode("utf-8")).digest() * 25


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
        self.finished = []

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

    def feed(self, pcm):
        accepted = super().feed(pcm)
        if accepted and pcm and len(self.fed) == 1:
            self.on_event(tts_events.SinkStarted())
        return accepted


def _completion(handler):
    events = [
        event
        for event in handler.messages
        if isinstance(event, tts_events.TTSCompleteEvent)
    ]
    assert len(events) == 1
    return events[0]


async def _run(handler, service, text, *, console=True, public=False, resolution=None):
    try:
        if public:
            await handler.speak_utterance(text, on_finished=handler.finished.append)
        else:
            await handler._generate_tts(
                text,
                "chunked-turn",
                None,
                resolution,
                on_finished=handler.finished.append if console else None,
            )
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

    await _run(handler, service, LONG_TEXT, public=True)

    assert handler.finished == [True]
    # More than one request, i.e. the chunked path actually engaged.
    assert len(requests) > 1
    # ...and every request carried a piece of the same utterance.
    assert " ".join(request.input for request in requests).split() == LONG_TEXT.split()

    assert len(_SinkRegistry.instances) == 1  # one sink for the whole utterance
    sink = _SinkRegistry.instances[0]
    assert sink.opened_with == (RATE, 1)
    assert len({_payload_for(request.input) for request in requests}) == len(requests)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_failure", [False, True])
async def test_cancellation_during_piece_close_propagates(stream_failure):
    from tldw_chatbook.TTS.adapter_types import TTSAudioResponse

    closing = asyncio.Event()

    async def audio():
        yield _wav(_payload_for("cancel"))
        if stream_failure:
            raise RuntimeError("stream failed before close")

    async def close():
        closing.set()
        await asyncio.Event().wait()

    response = TTSAudioResponse(
        provider_id="openai",
        model_id="tts-1",
        audio_format="wav",
        content_type="audio/wav",
        byte_stream=audio(),
        cleanup=close,
    )

    async def request(text, **kwargs):
        return response, None

    task = asyncio.create_task(_Handler(None)._collect_speech_piece(request, "cancel"))
    await asyncio.wait_for(closing.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_manual_long_wav_retains_the_original_file_delivery(monkeypatch):
    body = _wav(b"\x01\x02" * (8 * 1024 * 1024 + 1))
    service, requests = _service(request_body=body)
    handler = _Handler(service)
    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", _SinkRegistry)

    await _run(handler, service, LONG_TEXT, console=False)

    assert len(requests) == 1
    assert requests[0].input == LONG_TEXT
    assert _completion(handler).audio_file is not None
    assert not _SinkRegistry.instances


@pytest.mark.asyncio
async def test_unspaced_long_reply_is_bounded_and_spoken_completely(monkeypatch):
    text = "你好世界。" * 80
    service, requests = _service()
    handler = _Handler(service)
    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", _SinkRegistry)

    await _run(handler, service, text, public=True)

    assert len(requests) > 1
    assert max(len(request.input) for request in requests) <= 200
    assert "".join(request.input for request in requests) == text
    assert b"".join(_SinkRegistry.instances[0].fed) == b"".join(
        _payload_for(request.input) for request in requests
    )
    assert handler.finished == [True]


@pytest.mark.asyncio
async def test_piece_completion_reports_overall_generation_progress(monkeypatch):
    from tldw_chatbook.TTS.adapter_types import TTSProgress

    service, requests = _service()
    handler = _Handler(service)
    synthesize = service.synthesize_default

    async def completed_piece(**kwargs):
        await kwargs["progress_sink"](TTSProgress(status="done", fraction=1.0))
        return await synthesize(**kwargs)

    monkeypatch.setattr(service, "synthesize_default", completed_piece)
    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", _SinkRegistry)

    await _run(handler, service, LONG_TEXT)

    progress = [
        event.progress
        for event in handler.messages
        if isinstance(event, tts_events.TTSProgressEvent)
        and event.status == "Generating audio"
    ]
    assert len(requests) > 1
    assert progress[0] < 0.5
    assert progress == sorted(progress)
    assert progress[-1] == 0.9


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "axis",
    [
        "provider_id",
        "model_id",
        "voice_id",
        "response_format",
        "speed",
        "provider_options",
        "response_model",
    ],
)
@pytest.mark.parametrize("bad_piece", [1, 2])
async def test_every_piece_rejects_mismatched_exact_selection(
    monkeypatch, axis, bad_piece
):
    from tldw_chatbook.TTS.adapter_types import TTSRequest
    from tldw_chatbook.TTS.character_request_resolver import (
        CharacterTTSRequestResolution,
    )

    service, requests = _service()
    handler = _Handler(service)
    synthesize = service.synthesize_default
    closed = []
    count = 0

    async def effective(**kwargs):
        nonlocal count
        count += 1
        response = await synthesize(
            text=kwargs["text"], progress_sink=kwargs["progress_sink"]
        )
        selection = SimpleNamespace(
            provider_id="openai",
            model_id="tts-1",
            voice_id="alloy",
            response_format="wav",
            speed=1.0,
            provider_options={},
            revisions=SimpleNamespace(provider_configuration=1),
        )

        async def close():
            closed.append(response)

        response.add_cleanup(close)
        if count == bad_piece:
            if axis == "response_model":
                response.model_id = "wrong-model"
            else:
                setattr(
                    selection,
                    axis,
                    2.0
                    if axis == "speed"
                    else {"wrong": True}
                    if axis == "provider_options"
                    else "wrong",
                )
        return response, selection

    monkeypatch.setattr(service, "synthesize_effective", effective)
    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", _SinkRegistry)
    resolution = CharacterTTSRequestResolution(
        source="default_profile",
        request=TTSRequest(
            provider_id="openai",
            model_id="tts-1",
            text=LONG_TEXT,
            voice="alloy",
            response_format="wav",
        ),
        repository_generation=1,
        profile_id=uuid4(),
        profile_revision=1,
    )

    await _run(handler, service, LONG_TEXT, resolution=resolution)

    assert count == bad_piece
    assert len(closed) == bad_piece
    assert _completion(handler).error is not None
    if bad_piece == 1:
        assert not _SinkRegistry.instances
    else:
        assert b"".join(_SinkRegistry.instances[0].fed) == _payload_for(
            requests[0].input
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        ("word " * 100) + ", another short clause.",
        ("word " * 30) + ",;,," + ("tail " * 30) + ",",
    ],
    ids=["oversized-comma-clause", "repeated-trailing-delimiters"],
)
async def test_oversized_clauses_preserve_all_punctuation(monkeypatch, text):
    service, requests = _service()
    handler = _Handler(service)
    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", _SinkRegistry)

    await _run(handler, service, text, public=True)

    assert len(requests) > 1
    assert max(len(request.input) for request in requests) <= 200
    assert "".join("".join(request.input for request in requests).split()) == "".join(
        text.split()
    )
    assert handler.finished == [True]


@pytest.mark.asyncio
async def test_long_reply_routes_through_public_console_snapshot_event(monkeypatch):
    service, requests = _service()
    handler = _Handler(service)
    app = _FakeApp()
    app._ensure_tts_handler = AsyncMock(return_value=handler)
    handler.app = app
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=LONG_TEXT
    )
    states, outcomes = [], []
    lifecycle = tts_events.TTSPlaybackLifecycle(
        message_id=message.id,
        request_id=1,
        validator=lambda: True,
        callback=states.append,
    )
    destination = await handler.resolve_console_speech_destination(None, None)
    assert destination is not None
    event = tts_events.TTSMessageSpeechRequestEvent(
        store.issue_tts_message_speech_snapshot(message.id),
        store.validate_tts_message_speech_snapshot,
        expected_destination_fingerprint=destination.fingerprint,
        outcome_callback=outcomes.append,
        playback_lifecycle=lifecycle,
    )
    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", _SinkRegistry)
    try:
        await TldwCli.handle_tts_message_speech_request_event(app, event)
        tasks = tuple(handler._active_tasks)
        assert tasks
        await asyncio.wait_for(asyncio.gather(*tasks), 5)
        assert len(requests) > 1
        assert b"".join(_SinkRegistry.instances[0].fed) == b"".join(
            _payload_for(request.input) for request in requests
        )
        completed = [
            event
            for event in app.posted
            if isinstance(event, tts_events.TTSCompleteEvent)
        ]
        assert len(completed) == 1
        assert completed[0].error is None
        assert completed[0].message_id == message.id
        assert (
            not outcomes
        )  # Successful Console playback settles through its lifecycle.
        assert states == ["playing", "stopped"]
    finally:
        await handler.cleanup_tts_resources()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_oversized_spaced_clause_keeps_words_whole(monkeypatch):
    text = " ".join(["nevertheless"] * 40) + ", speech ends here."
    service, requests = _service()
    handler = _Handler(service)
    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", _SinkRegistry)

    await _run(handler, service, text, public=True)

    assert len(requests) > 1
    assert max(len(request.input) for request in requests) <= 200
    assert " ".join(request.input for request in requests).split() == text.split()
