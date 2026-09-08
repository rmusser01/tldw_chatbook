"""Exercise real Chatterbox audio handling with only inference/IPC replaced."""

import asyncio
import base64
import io
import json
import subprocess
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends.chatterbox import ChatterboxTTSBackend

torch = pytest.importorskip("torch")
AudioSegment = pytest.importorskip("pydub").AudioSegment


def tone(frequency=440):
    return torch.sin(torch.arange(12000) * (2 * torch.pi * frequency / 24000)) * 0.2


class Model:
    sr = 24000

    def __init__(self):
        self.calls = 0
        self.stream_calls = 0

    def generate(self, text, **kwargs):
        self.calls += 1
        return tone()

    def generate_stream(self, text, **kwargs):
        self.stream_calls += 1
        for index in range(3):
            yield tone(220 * (index + 1)), {"chunk_index": index}


@pytest.fixture
def backend(tmp_path):
    owner = ChatterboxTTSBackend(
        {
            "CHATTERBOX_DEVICE": "cpu",
            "CHATTERBOX_VOICE_DIR": str(tmp_path),
            "CHATTERBOX_STREAMING": False,
            "CHATTERBOX_NORMALIZE_AUDIO": False,
            "CHATTERBOX_ENABLE_CROSSFADE": False,
            "CHATTERBOX_PREPROCESS_TEXT": False,
        }
    )
    owner.model = Model()
    owner._initialized = True
    return owner


def request(text="Hello there.", audio_format="wav", stream=True):
    return OpenAISpeechRequest(
        input=text,
        voice="default",
        model="chatterbox",
        response_format=audio_format,
        stream=stream,
    )


async def drain(backend, req=None):
    return b"".join(
        [chunk async for chunk in backend.generate_speech_stream(req or request())]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_format", ["wav", "mp3", "flac", "opus", "aac", "pcm"])
@pytest.mark.parametrize("mode", ["long", "stream", "single"])
async def test_complete_chatterbox_utterance_decodes_in_every_format(
    backend, mode, audio_format
):
    text = "Hello there."
    if mode == "long":
        backend.max_chunk_size = 20
        text = "Sentence alpha. Sentence beta. Sentence gamma."
    backend.streaming_enabled = mode == "stream"
    expected_seconds = 0.5 if mode == "single" else 1.5
    for _ in range(2):
        data = await drain(backend, request(text, audio_format))
        if audio_format == "pcm":
            assert len(data) == int(24000 * 2 * expected_seconds)
            assert not data.startswith(b"RIFF")
        else:
            decoded = AudioSegment.from_file(
                io.BytesIO(data),
                format="ogg" if audio_format == "opus" else audio_format,
            )
            assert len(decoded) / 1000 == pytest.approx(expected_seconds, abs=0.08)
            assert decoded.rms > 0
            if audio_format == "wav":
                with wave.open(io.BytesIO(data)) as wav:
                    assert wav.getnframes() == 24000 * expected_seconds


@pytest.mark.asyncio
async def test_stream_false_uses_single_inference(backend):
    backend.streaming_enabled = True
    await drain(backend, request(stream=False))
    assert backend.model.stream_calls == 0
    assert backend.model.calls == 1


@pytest.mark.asyncio
async def test_complete_audio_limit_is_not_retried(backend, monkeypatch):
    monkeypatch.setattr("tldw_chatbook.TTS.audio_limits.MAX_BUFFERED_AUDIO_BYTES", 1000)
    with pytest.raises(TTSOperationError) as error:
        await drain(backend)
    assert error.value.retryable is False
    assert backend.model.calls == 1


@pytest.mark.asyncio
async def test_crossfade_error_does_not_return_concatenated_containers(backend):
    good = backend._tensor_to_wav_bytes(tone(), 24000)
    with pytest.raises((ValueError, wave.Error, EOFError)):
        await backend._combine_audio_with_crossfade([good, b"corrupt WAV"])


@pytest.mark.asyncio
async def test_fallback_cancel_preserves_backend_and_request_settings(backend):
    entered = asyncio.Event()
    original = (backend.exaggeration, backend.cfg_weight, backend.num_candidates)
    calls = 0

    async def infer(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient model failure")
        entered.set()
        await asyncio.Event().wait()

    backend._generate_single = infer
    req = request()
    task = asyncio.create_task(drain(backend, req))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (
        backend.exaggeration,
        backend.cfg_weight,
        backend.num_candidates,
    ) == original
    assert not getattr(req, "_is_fallback", False)


class Process:
    def __init__(self):
        self.returncode = None
        self.stdout = asyncio.StreamReader()
        self.sent = asyncio.Event()
        self.commands = []
        self.stdin = SimpleNamespace(write=self.write, drain=self.drain)
        self.waited = False

    def write(self, data):
        self.commands.append(json.loads(data))
        self.sent.set()

    async def drain(self):
        pass

    def terminate(self):
        self.returncode = -15
        self.stdout.feed_eof()

    def kill(self):
        self.terminate()

    async def wait(self):
        self.waited = True
        return self.returncode


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "timeout"])
async def test_interrupted_ipc_is_reaped_before_another_request(backend, failure):
    process = Process()
    backend.process = process
    backend.model = "process"
    if failure == "timeout":

        async def time_out(*args, **kwargs):
            raise TimeoutError("test deadline")

        backend._read_chunked_audio = time_out
    task = asyncio.create_task(
        backend._generate_single_isolated("Request A", None, 0.5, 0.5)
    )
    await process.sent.wait()
    if failure == "cancel":
        task.cancel()
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else TimeoutError):
        await task
    assert process.waited
    assert process.returncode is not None
    assert backend.process is None
    assert not backend._initialized
    assert backend.model is None


@pytest.mark.asyncio
async def test_subprocess_transfer_chunks_are_reassembled_once(backend):
    process = Process()
    backend.process = process
    backend.model = "process"
    audio = backend._tensor_to_wav_bytes(tone(), 24000)
    encoded = base64.b64encode(audio).decode()
    for index, part in enumerate((encoded[:16000], encoded[16000:])):
        process.stdout.feed_data(
            (
                json.dumps(
                    {
                        "type": "audio_chunk",
                        "chunk_id": index,
                        "total_chunks": 2,
                        "data": part,
                    }
                )
                + "\n"
            ).encode()
        )
    process.stdout.feed_data(b'{"type":"audio_complete","total_chunks":2}\n')
    assert await drain(backend) == audio


@pytest.mark.asyncio
async def test_subprocess_transfer_is_bounded_before_reassembly(backend, monkeypatch):
    monkeypatch.setattr("tldw_chatbook.TTS.audio_limits.MAX_BUFFERED_AUDIO_BYTES", 1000)
    process = Process()
    backend.process = process
    process.stdout.feed_data(
        (json.dumps({"type": "audio", "data": "A" * 2000}) + "\n").encode()
    )
    with pytest.raises(TTSOperationError):
        await backend._read_chunked_audio()


@pytest.mark.asyncio
async def test_pcm_failure_after_delivery_is_not_retried(backend):
    backend.streaming_enabled = True

    def partial_stream(*args, **kwargs):
        backend.model.stream_calls += 1
        yield tone(), {}
        raise RuntimeError("inference failed after the first audio")

    backend.model.generate_stream = partial_stream
    delivered = []
    with pytest.raises(RuntimeError):
        async for chunk in backend.generate_speech_stream(request(audio_format="pcm")):
            delivered.append(chunk)
    assert len(delivered) == 1
    assert backend.model.stream_calls == 1


@pytest.mark.asyncio
async def test_empty_pcm_stream_uses_fallback_instead_of_succeeding_silently(backend):
    backend.streaming_enabled = True
    backend.model.generate_stream = lambda *args, **kwargs: iter(())
    audio = await drain(backend, request(audio_format="pcm"))
    assert len(audio) == 12000 * 2
    assert backend.model.calls > 0


@pytest.mark.asyncio
async def test_stream_cancel_waits_for_inference_thread(backend):
    entered = threading.Event()
    finish = threading.Event()

    def generate_stream(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        yield tone(), {}

    backend.model.generate_stream = generate_stream
    backend.streaming_enabled = True
    task = asyncio.create_task(drain(backend))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_isolated_worker_encodes_without_optional_torchcodec():
    """Run the actual subprocess protocol; replace neural inference only."""
    script = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/TTS/backends/chatterbox_process.py"
    )
    bootstrap = """
import runpy, sys, types, torch
model = types.SimpleNamespace(sr=24000, generate=lambda *a, **k: torch.ones(12000) * .1)
tts = types.ModuleType('chatterbox.tts')
tts.ChatterboxTTS = types.SimpleNamespace(from_pretrained=lambda **k: model)
sys.modules['chatterbox'] = types.ModuleType('chatterbox')
sys.modules['chatterbox.tts'] = tts
def missing_codec(*a, **k):
    raise ImportError('TorchCodec is required by torchaudio.save')
sys.modules['torchaudio'] = types.SimpleNamespace(save=missing_codec)
runpy.run_path(sys.argv[1], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", bootstrap, str(script)],
        input=(
            '{"command":"initialize","device":"cpu"}\n'
            '{"command":"generate","text":"Hello"}\n'
            '{"command":"shutdown"}\n'
        ),
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert not [message for message in messages if message["type"] == "error"]
    audio = next(message for message in messages if message["type"] == "audio")
    with wave.open(io.BytesIO(base64.b64decode(audio["data"]))) as wav:
        assert wav.getnframes() == 12000
        assert wav.getframerate() == 24000
        assert wav.getsampwidth() == 2
