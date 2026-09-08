"""Decode actual backend bytes; transport fixtures replace only provider inference."""

import asyncio
import io
import json
import threading
import wave
from types import SimpleNamespace

import httpx
import pytest

from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.audio_service import AudioService
from tldw_chatbook.TTS.backends.alltalk import AllTalkTTSBackend
from tldw_chatbook.TTS.backends.elevenlabs import ElevenLabsTTSBackend
from tldw_chatbook.TTS.backends.higgs import HiggsAudioTTSBackend
from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend

np = pytest.importorskip("numpy")
AudioSegment = pytest.importorskip("pydub").AudioSegment


def _pcm(rate=24000):
    times = np.arange(rate, dtype=np.float64) / rate
    return (np.sin(2 * np.pi * 440 * times) * 8000).astype("<i2").tobytes()


def _wav(pcm, rate=24000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm)
    return buffer.getvalue()


def _assert_duration(audio, audio_format, seconds=1.0):
    if audio_format == "pcm":
        assert len(audio) == pytest.approx(int(seconds * 24000 * 2), abs=2)
        assert not audio.startswith(b"RIFF")
    else:
        # Opus is carried in Ogg; ffmpeg has no "opus" input demuxer.
        decoded = AudioSegment.from_file(
            io.BytesIO(audio), format="ogg" if audio_format == "opus" else audio_format
        )
        assert len(decoded) / 1000 == pytest.approx(seconds, abs=0.10)
        assert decoded.rms > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_format", ["pcm", "aac", "wav", "mp3", "flac", "opus"])
async def test_audio_service_converts_one_complete_utterance(audio_format):
    pcm = _pcm()
    result = await AudioService().convert_audio(
        _wav(pcm), audio_format, source_format="wav", sample_rate=24000
    )
    _assert_duration(result, audio_format)
    if audio_format == "pcm":
        assert result == pcm


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_format", ["wav", "pcm", "aac", "flac", "opus", "mp3"])
async def test_elevenlabs_delivers_the_requested_format_repeatedly(audio_format):
    """The fake server uses documented wire codecs, not the requested UI label."""

    async def handle(request):
        wire_format = request.url.params["output_format"]
        if wire_format.startswith("pcm_"):
            data = _pcm(int(wire_format.split("_")[1]))
        elif wire_format.startswith("mp3_"):
            data = await AudioService().convert_audio(
                _wav(_pcm()), "mp3", source_format="wav"
            )
        elif wire_format.startswith("opus_48000_"):
            data = await AudioService().convert_audio(
                _wav(_pcm()), "opus", source_format="wav"
            )
        else:
            return httpx.Response(422, json={"detail": "unsupported output format"})
        return httpx.Response(200, content=data)

    backend = ElevenLabsTTSBackend({"ELEVENLABS_API_KEY": "test-only"})
    await backend.client.aclose()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        for _ in range(2):
            request = OpenAISpeechRequest(
                model="eleven_multilingual_v2",
                voice="21m00Tcm4TlvDq8ikWAM",
                input="An entire sentence.",
                response_format=audio_format,
            )
            data = b"".join(
                [chunk async for chunk in backend.generate_speech_stream(request)]
            )
            _assert_duration(data, audio_format)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_alltalk_pcm_is_raw_audio_at_the_declared_rate():
    def handle(request):
        assert json.loads(request.content)["response_format"] == "wav"
        return httpx.Response(200, content=_wav(_pcm(22050), 22050))

    backend = AllTalkTTSBackend({"ALLTALK_TTS_URL": "http://127.0.0.1:7851"})
    await backend.client.aclose()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        for _ in range(2):
            request = OpenAISpeechRequest(
                input="Complete reply.", voice="female_01.wav", response_format="pcm"
            )
            data = b"".join(
                [chunk async for chunk in backend.generate_speech_stream(request)]
            )
            _assert_duration(data, "pcm")
    finally:
        await backend.close()


@pytest.fixture
def higgs_backend(tmp_path):
    pytest.importorskip("torch")
    backend = HiggsAudioTTSBackend(
        {
            "HIGGS_DEVICE": "cpu",
            "HIGGS_VOICE_SAMPLES_DIR": str(tmp_path),
            "HIGGS_TRACK_PERFORMANCE": False,
        }
    )
    backend.model_loaded = True
    backend._make_chat_ml_sample = lambda messages: messages
    backend._prepare_messages = lambda *args: []
    return backend


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["exception", "missing", "empty", "nonfinite"])
async def test_higgs_generation_failure_never_becomes_audio(higgs_backend, result):
    def infer(*args, **kwargs):
        if result == "exception":
            raise RuntimeError("private provider diagnostic")
        samples = {
            "missing": None,
            "empty": np.array([], dtype=np.float32),
            "nonfinite": np.array([np.nan], dtype=np.float32),
        }[result]
        return SimpleNamespace(audio=samples)

    higgs_backend._invoke_serve_engine_generate = infer
    chunks = []
    try:
        with pytest.raises((ValueError, RuntimeError)):
            async for chunk in higgs_backend.generate_speech_stream(
                OpenAISpeechRequest(
                    input="Complete reply.",
                    voice="professional_female",
                    response_format="wav",
                )
            ):
                chunks.append(chunk)
        assert chunks == []
    finally:
        await higgs_backend.close()


@pytest.mark.asyncio
async def test_higgs_completed_generation_leaves_no_shutdown_waiters(higgs_backend):
    higgs_backend._invoke_serve_engine_generate = lambda *a, **k: SimpleNamespace(
        audio=np.frombuffer(_pcm(), dtype="<i2").astype(np.float32) / 32768
    )
    before = asyncio.all_tasks()
    try:
        for _ in range(2):
            audio = b"".join(
                [
                    part
                    async for part in higgs_backend.generate_speech_stream(
                        OpenAISpeechRequest(
                            input="Hello.",
                            voice="professional_female",
                            response_format="wav",
                        )
                    )
                ]
            )
            _assert_duration(audio, "wav")
        assert not [task for task in asyncio.all_tasks() - before if not task.done()]
    finally:
        await higgs_backend.close()


@pytest.mark.asyncio
async def test_higgs_cancel_keeps_inference_owned_until_it_finishes(higgs_backend):
    started = threading.Event()
    finish = threading.Event()

    def infer(*args, **kwargs):
        started.set()
        assert finish.wait(5)
        return SimpleNamespace(audio=np.zeros(24000, dtype=np.float32))

    higgs_backend._invoke_serve_engine_generate = infer

    async def drain():
        return [
            part
            async for part in higgs_backend.generate_speech_stream(
                OpenAISpeechRequest(
                    input="Hello.", voice="professional_female", response_format="wav"
                )
            )
        ]

    task = asyncio.create_task(drain())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert higgs_backend._active_tasks
        assert not task.done()
    finally:
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await higgs_backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["elevenlabs", "alltalk"])
async def test_remote_buffer_limit_preserves_typed_nonretryable_failure(
    provider, monkeypatch
):
    monkeypatch.setattr("tldw_chatbook.TTS.audio_limits.MAX_BUFFERED_AUDIO_BYTES", 1000)
    backend = (
        ElevenLabsTTSBackend({"ELEVENLABS_API_KEY": "test-only"})
        if provider == "elevenlabs"
        else AllTalkTTSBackend({"ALLTALK_TTS_URL": "http://127.0.0.1:7851"})
    )
    await backend.client.aclose()
    backend.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=_wav(_pcm()))
        )
    )
    req = OpenAISpeechRequest(
        input="Hello.",
        voice="default",
        response_format="wav" if provider == "elevenlabs" else "pcm",
    )
    try:
        with pytest.raises(TTSOperationError) as error:
            _ = [part async for part in backend.generate_speech_stream(req)]
        assert not error.value.retryable
        assert error.value.recovery_action == "shorten_text"
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["elevenlabs", "alltalk", "openai"])
async def test_empty_remote_success_is_rejected(provider):
    backend = (
        ElevenLabsTTSBackend({"ELEVENLABS_API_KEY": "test-only"})
        if provider == "elevenlabs"
        else OpenAITTSBackend({"OPENAI_API_KEY": "test-only"})
        if provider == "openai"
        else AllTalkTTSBackend({"ALLTALK_TTS_URL": "http://127.0.0.1:7851"})
    )
    await backend.client.aclose()
    backend.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b""))
    )
    try:
        with pytest.raises(ValueError):
            _ = [
                part
                async for part in backend.generate_speech_stream(
                    OpenAISpeechRequest(
                        input="Hello.", voice="default", response_format="mp3"
                    )
                )
            ]
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_section", [False, True])
async def test_higgs_multi_speaker_delivers_all_sections_or_fails(
    higgs_backend, bad_section
):
    calls = 0

    def infer(*args, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            audio=None
            if bad_section and calls == 2
            else (np.frombuffer(_pcm(), dtype="<i2").astype(np.float32) / 32768)
        )

    higgs_backend._invoke_serve_engine_generate = infer

    async def generate():
        return b"".join(
            [
                part
                async for part in higgs_backend.generate_speech_stream(
                    OpenAISpeechRequest(
                        input="Speaker1|||Hello. Speaker2|||Goodbye.",
                        voice="professional_female",
                        response_format="wav",
                    )
                )
            ]
        )

    try:
        if bad_section:
            with pytest.raises(ValueError):
                await generate()
        else:
            _assert_duration(await generate(), "wav", 2.0)
    finally:
        await higgs_backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_higgs_model_load_retains_thread_and_has_no_shutdown_waiter(
    higgs_backend, cancel
):
    entered = threading.Event()
    finish = threading.Event()

    class Engine:
        def __init__(self, **kwargs):
            entered.set()
            assert finish.wait(5)

    higgs_backend.model_loaded = False
    higgs_backend._boson_multimodal = SimpleNamespace()
    higgs_backend._higgs_serve_engine = Engine
    before = asyncio.all_tasks()
    task = asyncio.create_task(higgs_backend.load_model())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert higgs_backend._active_tasks
            assert not task.done()
    finally:
        finish.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        await higgs_backend.close()
    assert not [task for task in asyncio.all_tasks() - before if not task.done()]


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_format", ["wav", "mp3", "aac", "flac", "opus", "pcm"])
async def test_higgs_repeated_complete_formats(higgs_backend, audio_format):
    higgs_backend._invoke_serve_engine_generate = lambda *a, **k: SimpleNamespace(
        audio=np.frombuffer(_pcm(), dtype="<i2").astype(np.float32) / 32768
    )
    try:
        for _ in range(2):
            data = b"".join(
                [
                    part
                    async for part in higgs_backend.generate_speech_stream(
                        OpenAISpeechRequest(
                            input="Hello.",
                            voice="professional_female",
                            response_format=audio_format,
                        )
                    )
                ]
            )
            _assert_duration(data, audio_format)
    finally:
        await higgs_backend.close()
