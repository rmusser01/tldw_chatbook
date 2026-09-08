"""Kokoro file responses must decode to the whole generated utterance."""

import io
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from tldw_chatbook.TTS import audio_service
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend


@pytest.fixture
def generated_audio(tmp_path):
    sample_rate = 24000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = np.concatenate(
        [0.2 * np.sin(2 * np.pi * frequency * time) for frequency in (440, 660)]
    ).astype(np.float32)

    async def create_stream(*args, **kwargs):
        for start in range(0, len(samples), 3000):
            yield samples[start : start + 3000], sample_rate

    backend = KokoroTTSBackend(
        {"KOKORO_USE_ONNX": True, "KOKORO_VOICE_BLENDS_DIR": str(tmp_path / "blends")}
    )
    backend.kokoro_instance = SimpleNamespace(create_stream=create_stream)
    backend.model_loaded = True
    return backend, samples, sample_rate


@pytest.mark.asyncio
@pytest.mark.parametrize("format_name", ["wav", "flac", "mp3"])
async def test_encoded_response_decodes_every_generated_sample(
    generated_audio, monkeypatch, format_name
):
    sf = pytest.importorskip("soundfile")
    if format_name == "mp3":
        pytest.importorskip("pydub")
        if shutil.which("ffmpeg") is None:
            pytest.skip("MP3 encoding requires ffmpeg")
    else:
        # Exercise the installed soundfile encoder independently of pydub.
        monkeypatch.setattr(audio_service, "PYDUB_AVAILABLE", False)
    backend, expected, sample_rate = generated_audio
    request = OpenAISpeechRequest(
        input="A complete two-part utterance.",
        model="kokoro",
        voice="af_heart",
        response_format=format_name,
    )
    encoded = b"".join(
        [chunk async for chunk in backend.generate_speech_stream(request)]
    )
    actual, rate = sf.read(io.BytesIO(encoded), dtype="float32")
    assert rate == sample_rate
    assert len(actual) == len(expected)
    if format_name != "mp3":
        np.testing.assert_allclose(actual, expected, atol=2 / 32767)
    else:
        # Lossy encoding may change amplitudes, but must retain both halves.
        assert np.corrcoef(actual[:sample_rate], expected[:sample_rate])[0, 1] > 0.99
        assert np.corrcoef(actual[sample_rate:], expected[sample_rate:])[0, 1] > 0.99


@pytest.mark.asyncio
async def test_pcm_response_remains_incremental_and_complete(generated_audio):
    backend, expected, _ = generated_audio
    request = OpenAISpeechRequest(
        input="A complete two-part utterance.",
        model="kokoro",
        voice="af_heart",
        response_format="pcm",
    )
    chunks = [chunk async for chunk in backend.generate_speech_stream(request)]
    assert len(chunks) > 1
    actual = np.frombuffer(b"".join(chunks), dtype=np.int16)
    np.testing.assert_array_equal(actual, np.int16(expected * 32767))


@pytest.mark.asyncio
async def test_failed_encoding_does_not_emit_a_partial_audio_file(generated_audio):
    backend, _, _ = generated_audio
    backend.audio_service = SimpleNamespace(
        convert_audio=AsyncMock(side_effect=RuntimeError("encoder failed"))
    )
    request = OpenAISpeechRequest(
        input="A complete two-part utterance.",
        model="kokoro",
        voice="af_heart",
        response_format="mp3",
    )
    chunks = [chunk async for chunk in backend.generate_speech_stream(request)]
    assert not any(chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize("format_name", ["wav", "flac", "mp3", "pcm"])
async def test_pytorch_response_preserves_every_text_chunk(
    generated_audio, monkeypatch, format_name
):
    if format_name != "pcm":
        sf = pytest.importorskip("soundfile")
        if format_name == "mp3":
            pytest.importorskip("pydub")
            if shutil.which("ffmpeg") is None:
                pytest.skip("MP3 encoding requires ffmpeg")
        else:
            monkeypatch.setattr(audio_service, "PYDUB_AVAILABLE", False)
    backend, expected, sample_rate = generated_audio
    backend.use_onnx = False
    backend.kokoro_instance = None
    backend.kokoro_model_pt = object()
    backend._torch = SimpleNamespace(Tensor=type("UnusedTensor", (), {}))
    monkeypatch.setattr(backend, "_download_voice_if_needed", AsyncMock())
    monkeypatch.setattr(backend, "_load_voice_pack", lambda _voice: object())
    generated_text = []

    def generate(_model, text, _voice, **_kwargs):
        start = len(generated_text) * sample_rate
        generated_text.append(text)
        return expected[start : start + sample_rate], "fixture phonemes"

    backend._kokoro_pt_modules = {"generate": generate}
    request = OpenAISpeechRequest(
        input="sample " * 300,
        model="kokoro",
        voice="af_heart",
        response_format=format_name,
    )
    chunks = [chunk async for chunk in backend.generate_speech_stream(request)]
    # Exercise the actual 150-word PyTorch splitter, replacing model inference.
    assert [len(text.split()) for text in generated_text] == [150, 150]
    if format_name == "pcm":
        assert len(chunks) == 2
        actual = np.frombuffer(b"".join(chunks), dtype=np.int16)
        np.testing.assert_array_equal(actual, np.int16(expected * 32767))
        return
    actual, rate = sf.read(io.BytesIO(b"".join(chunks)), dtype="float32")
    assert rate == sample_rate
    assert len(actual) == len(expected)
    if format_name != "mp3":
        np.testing.assert_allclose(actual, expected, atol=2 / 32767)
    else:
        assert np.corrcoef(actual[:sample_rate], expected[:sample_rate])[0, 1] > 0.99
        assert np.corrcoef(actual[sample_rate:], expected[sample_rate:])[0, 1] > 0.99
