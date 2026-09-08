"""Encoded Kokoro requests must stop at the audio budget without partial output."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends import kokoro


@pytest.fixture
def make_backend(tmp_path, monkeypatch):
    # One second makes boundary checks cheap without synthesizing a long reply.
    monkeypatch.setattr(
        kokoro, "KOKORO_MAX_ENCODED_AUDIO_SAMPLES", 24000, raising=False
    )

    def build(engine, chunks, *, voice_chunks=None):
        backend = kokoro.KokoroTTSBackend(
            {
                "KOKORO_USE_ONNX": engine == "onnx",
                "KOKORO_VOICE_BLENDS_DIR": str(tmp_path / "blends"),
            }
        )
        backend.model_loaded = True
        backend.audio_service = SimpleNamespace(
            convert_audio=AsyncMock(return_value=b"complete encoded file")
        )
        calls = []
        closed = []

        async def create_stream(_text, *, voice, **_kwargs):
            source = chunks if voice_chunks is None else voice_chunks[voice]
            try:
                for samples in source:
                    calls.append(voice)
                    yield samples, 24000
            finally:
                closed.append(voice)

        if engine == "onnx":
            backend.kokoro_instance = SimpleNamespace(create_stream=create_stream)
        else:
            backend.kokoro_model_pt = object()
            backend._torch = SimpleNamespace(Tensor=type("UnusedTensor", (), {}))
            monkeypatch.setattr(backend, "_download_voice_if_needed", AsyncMock())
            monkeypatch.setattr(backend, "_load_voice_pack", lambda _voice: object())

            def generate(_model, _text, _voice, **_kwargs):
                samples = chunks[len(calls)]
                calls.append("af_heart")
                return samples, "fixture phonemes"

            backend._kokoro_pt_modules = {"generate": generate}
        request = OpenAISpeechRequest(
            input="word " * (150 * len(chunks)),
            model="kokoro",
            voice="af_heart",
            response_format="wav",
        )
        return backend, request, calls, closed

    return build


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["onnx", "pytorch"])
async def test_encoded_audio_at_limit_is_complete(make_backend, engine):
    samples = np.linspace(-0.2, 0.2, 24000, dtype=np.float32)
    backend, request, calls, closed = make_backend(
        engine, [samples[:12000], samples[12000:]]
    )
    output = [chunk async for chunk in backend.generate_speech_stream(request)]
    assert output == [b"complete encoded file"]
    assert len(calls) == 2
    actual = backend.audio_service.convert_audio.await_args.args[0]
    np.testing.assert_array_equal(actual, samples)
    if engine == "onnx":
        assert closed == ["af_heart"]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["onnx", "pytorch"])
async def test_encoded_audio_over_limit_stops_without_partial_file(
    make_backend, engine
):
    backend, request, calls, closed = make_backend(
        engine,
        [np.zeros(24000, dtype=np.float32), np.zeros(1), np.zeros(24000)],
    )
    output = []
    with pytest.raises(TTSOperationError, match="Shorten.*PCM") as caught:
        async for chunk in backend.generate_speech_stream(request):
            output.append(chunk)
    assert caught.value.code == "request_invalid"
    assert caught.value.retryable is False
    assert output == []
    assert len(calls) == 2
    backend.audio_service.convert_audio.assert_not_awaited()
    if engine == "onnx":
        assert closed == ["af_heart"]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["onnx", "pytorch"])
async def test_pcm_above_encoded_limit_stays_incremental(make_backend, engine):
    samples = np.full(24000, 0.2, dtype=np.float32)
    backend, request, calls, _ = make_backend(engine, [samples, samples])
    request.response_format = "pcm"
    output = [chunk async for chunk in backend.generate_speech_stream(request)]
    assert len(output) == len(calls) == 2
    assert b"".join(output) == np.int16(np.tile(samples, 2) * 32767).tobytes()
    backend.audio_service.convert_audio.assert_not_awaited()


@pytest.mark.asyncio
async def test_mixed_voice_limit_stops_before_next_voice(make_backend):
    chunks = [np.zeros(24000, dtype=np.float32), np.zeros(1)]
    backend, request, calls, closed = make_backend("onnx", chunks)
    backend.enable_voice_mixing = True
    request.voice = "af_heart:0.5,af_bella:0.5"
    output = []
    with pytest.raises(TTSOperationError, match="Shorten.*PCM"):
        async for chunk in backend.generate_speech_stream(request):
            output.append(chunk)
    assert output == []
    assert calls == ["af_heart", "af_heart"]
    assert closed == ["af_heart"]
    backend.audio_service.convert_audio.assert_not_awaited()


@pytest.mark.asyncio
async def test_mixed_voice_budget_is_output_length_and_retains_padding(make_backend):
    voices = {
        "af_heart": [np.full(24000, 0.2, dtype=np.float32)],
        "af_bella": [np.full(12000, 0.4, dtype=np.float32)],
        "af_nicole": [np.full(18000, 0.6, dtype=np.float32)],
    }
    backend, request, _, closed = make_backend(
        "onnx", [np.zeros(1)], voice_chunks=voices
    )
    backend.enable_voice_mixing = True
    request.voice = "af_heart:1,af_bella:1,af_nicole:1"
    output = [chunk async for chunk in backend.generate_speech_stream(request)]
    assert output == [b"complete encoded file"]
    assert closed == list(voices)
    actual = backend.audio_service.convert_audio.await_args.args[0]
    expected = np.full(24000, 0.2 / 3, dtype=np.float32)
    expected[:12000] += 0.4 / 3
    expected[:18000] += 0.6 / 3
    np.testing.assert_allclose(actual, expected, atol=1e-7)
