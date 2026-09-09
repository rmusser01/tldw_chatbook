"""ONNX stream cancellation must join executor work, not only its producer."""

import asyncio
import threading
from contextlib import suppress
from types import SimpleNamespace

import numpy as np
import pytest

from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend


class ExecutorStream:
    """Model boundary with the cancellation behavior of kokoro-onnx 0.6.1."""

    def __init__(self, native):
        self.native = native

    async def create_stream(self, text, *, voice, speed, lang):
        queue = asyncio.Queue(maxsize=1)

        async def produce():
            for part in text.split("|"):
                audio = await asyncio.get_running_loop().run_in_executor(
                    None, self.native, part
                )
                await queue.put((audio, 24000))
            await queue.put(None)

        task = asyncio.create_task(produce())
        try:
            while (chunk := await queue.get()) is not None:
                yield chunk
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def make_backend(tmp_path, runtime):
    backend = KokoroTTSBackend(
        {"KOKORO_USE_ONNX": True, "KOKORO_VOICE_BLENDS_DIR": str(tmp_path / "blends")}
    )
    backend.kokoro_instance = runtime
    backend.model_loaded = True
    return backend


async def collect(backend, mode, text="first|second"):
    if mode == "timestamps":
        return await backend.generate_with_timestamps(text, voice="af_heart")
    request = OpenAISpeechRequest(
        input=text,
        model="kokoro",
        voice="af_heart:1,bf_emma:1" if mode == "mixed" else "af_heart",
        response_format="wav" if mode == "mixed" else mode,
    )
    return b"".join([chunk async for chunk in backend.generate_speech_stream(request)])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pcm", "wav", "timestamps"])
async def test_language_frontend_reaches_onnx_in_every_output_path(
    tmp_path, monkeypatch, mode
):
    from tldw_chatbook.TTS import kokoro_languages

    loop_thread = threading.get_ident()
    prepared, calls = [], []

    def prepare(text, language, cache):
        assert threading.get_ident() != loop_thread
        prepared.append((text, language))
        return "ɕi phonemes", "ja", True

    async def stream(text, **kwargs):
        calls.append((text, kwargs))
        yield np.ones(240, dtype=np.float32) * 0.1, 24000

    monkeypatch.setattr(kokoro_languages, "prepare_onnx_text", prepare)
    backend = make_backend(tmp_path, SimpleNamespace(create_stream=stream))
    try:
        if mode == "timestamps":
            audio, _ = await backend.generate_with_timestamps("漢字", "jf_alpha")
        else:
            request = OpenAISpeechRequest(
                model="kokoro", input="漢字", voice="jf_alpha", response_format=mode
            )
            audio = b"".join(
                [chunk async for chunk in backend.generate_speech_stream(request)]
            )
        assert audio
        assert prepared == [("漢字", "ja")]
        assert calls == [
            (
                "ɕi phonemes",
                {"voice": "jf_alpha", "speed": 1.0, "lang": "ja", "is_phonemes": True},
            )
        ]
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_stop_joins_frontend_without_starting_inference(tmp_path, monkeypatch):
    from tldw_chatbook.TTS import kokoro_languages

    entered, release = threading.Event(), threading.Event()
    calls = []

    def prepare(text, language, cache):
        entered.set()
        assert release.wait(5)
        return "phonemes", "ja", True

    async def stream(*args, **kwargs):
        calls.append(args)
        yield np.zeros(10), 24000

    monkeypatch.setattr(kokoro_languages, "prepare_onnx_text", prepare)
    backend = make_backend(tmp_path, SimpleNamespace(create_stream=stream))
    task = asyncio.create_task(collect(backend, "pcm"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert backend._onnx_tasks
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == []
        assert not backend._onnx_tasks
    finally:
        release.set()
        await backend.close()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_explicit_onnx_phonemes_bypass_text_frontend(tmp_path, monkeypatch):
    from tldw_chatbook.TTS import kokoro_languages

    def prepare(*args):
        pytest.fail("Phonemes must not be converted as text")

    calls = []

    async def stream(text, **kwargs):
        calls.append((text, kwargs))
        yield np.zeros(10), 24000

    monkeypatch.setattr(kokoro_languages, "prepare_onnx_text", prepare)
    backend = make_backend(tmp_path, SimpleNamespace(create_stream=stream))
    try:
        assert [
            chunk
            async for chunk in backend._create_onnx_stream(
                "ɕi", voice="jf_alpha", lang="ja", is_phonemes=True
            )
        ]
        assert calls == [
            ("ɕi", {"voice": "jf_alpha", "lang": "ja", "is_phonemes": True})
        ]
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_direct_phoneme_failure_is_structured_and_retryable(tmp_path):
    def native(*args, **kwargs):
        raise RuntimeError("PRIVATE native failure")

    backend = make_backend(tmp_path, SimpleNamespace(generate_from_phonemes=native))
    try:
        with pytest.raises(TTSOperationError) as error:
            await backend.generate_from_phonemes("hello")
        assert error.value.code == "generation_failed"
        assert error.value.retryable
        assert "PRIVATE" not in str(error.value)
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pcm", "wav", "mixed", "timestamps"])
async def test_cancel_and_close_wait_for_native_inference(tmp_path, mode):
    # Releasing the model when only the upstream producer has stopped breaks
    # every public path, including mixing and timestamps.
    entered, release, finished = (threading.Event() for _ in range(3))
    calls = []

    def native(part):
        calls.append(part)
        entered.set()
        assert release.wait(5)
        finished.set()
        return np.ones(240, dtype=np.float32) * 0.1

    runtime = ExecutorStream(native)
    backend = make_backend(tmp_path, runtime)
    generation = asyncio.create_task(collect(backend, mode))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        generation.cancel()
        await asyncio.sleep(0.02)
        generation.cancel()
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0.03)
        assert not closing.done(), "Close returned with native inference still active"
        assert backend.kokoro_instance is runtime
        assert not generation.done(), "Cancellation abandoned native inference"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await generation
        await closing
        assert finished.is_set()
        assert calls == ["first"]
        assert backend.kokoro_instance is None
    finally:
        release.set()
        await asyncio.gather(
            generation, *([closing] if closing else []), return_exceptions=True
        )
        await backend.close()


@pytest.mark.asyncio
async def test_real_onnx_producer_cancellation_keeps_its_native_worker_owned(tmp_path):
    # Characterize the actual optional dependency, replacing only model setup
    # and batch compute. No ONNX model, phonemizer, download or audio is used.
    kokoro = pytest.importorskip("kokoro_onnx")
    entered, release, finished = (threading.Event() for _ in range(3))
    runtime = kokoro.Kokoro.__new__(kokoro.Kokoro)
    runtime._prepare = lambda *args: (np.ones(1), "first", [("first", 0.0)])

    def native(*args):
        entered.set()
        assert release.wait(5)
        finished.set()
        return np.ones(240, dtype=np.float32) * 0.1, None

    runtime._create_batch = native
    backend = make_backend(tmp_path, runtime)
    generation = asyncio.create_task(collect(backend, "wav"))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        generation.cancel()
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0.03)
        assert not closing.done(), (
            "Real ONNX producer cancellation abandoned its executor"
        )
        assert backend.kokoro_instance is runtime
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await generation
        await closing
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(
            generation, *([closing] if closing else []), return_exceptions=True
        )
        await backend.close()


@pytest.mark.asyncio
async def test_abandoning_pcm_stream_joins_prefetched_work(tmp_path):
    entered, release, finished = (threading.Event() for _ in range(3))
    calls = []

    def native(part):
        calls.append(part)
        if part == "second":
            entered.set()
            assert release.wait(5)
            finished.set()
        return np.ones(240, dtype=np.float32) * 0.1

    backend = make_backend(tmp_path, ExecutorStream(native))
    request = OpenAISpeechRequest(
        input="first|second|third",
        model="kokoro",
        voice="af_heart",
        response_format="pcm",
    )
    stream = backend.generate_speech_stream(request)
    closing = None
    try:
        assert len(await anext(stream)) == 480
        assert await asyncio.to_thread(entered.wait, 2)
        closing = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0.03)
        assert not closing.done(), "Generator close abandoned prefetched native work"
        release.set()
        await closing
        assert finished.is_set()
        assert calls == ["first", "second"]
        assert await collect(backend, "pcm", "successor")
        assert calls[-1] == "successor"
    finally:
        release.set()
        if closing:
            await asyncio.gather(closing, return_exceptions=True)
        await stream.aclose()
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pcm", "wav", "mixed", "timestamps"])
async def test_native_failure_is_retryable_instead_of_error_text_audio(tmp_path, mode):
    async def create_stream(*args, **kwargs):
        raise RuntimeError("PRIVATE native failure")
        yield  # Make this the same async-generator boundary as the runtime.

    backend = make_backend(tmp_path, SimpleNamespace(create_stream=create_stream))
    try:
        with pytest.raises(TTSOperationError) as caught:
            await collect(backend, mode)
        assert caught.value.code == "generation_failed" and caught.value.retryable
        assert "PRIVATE" not in str(caught.value)
        assert not backend._onnx_tasks
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pcm", "wav", "mixed", "timestamps"])
async def test_close_alone_cancels_request_after_joining_native_work(tmp_path, mode):
    entered, release = threading.Event(), threading.Event()

    def native(part):
        entered.set()
        assert release.wait(5)
        return np.ones(240, dtype=np.float32)

    backend = make_backend(tmp_path, ExecutorStream(native))
    generation = asyncio.create_task(collect(backend, mode))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0.03)
        assert not closing.done()
        release.set()
        await closing
        with pytest.raises(asyncio.CancelledError):
            await generation
    finally:
        release.set()
        await asyncio.gather(
            generation, *([closing] if closing else []), return_exceptions=True
        )
        await backend.close()


@pytest.mark.asyncio
async def test_loading_is_responsive_and_retained_through_cancel_and_close(tmp_path):
    entered, release, finished = (threading.Event() for _ in range(3))

    def load(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        finished.set()
        return object()

    backend = make_backend(tmp_path, None)
    backend._kokoro_onnx = SimpleNamespace(Kokoro=load, EspeakConfig=dict)
    for attribute, filename in [
        ("model_path", "model.onnx"),
        ("voices_json", "voices.bin"),
    ]:
        path = tmp_path / filename
        path.touch()
        setattr(backend, attribute, str(path))
    timer = threading.Timer(1.5, release.set)
    timer.start()
    loading = asyncio.create_task(backend._initialize_onnx())
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert not finished.is_set(), "ONNX construction blocked the event loop"
        loading.cancel()
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0.03)
        assert not loading.done() and not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await loading
        await closing
        assert finished.is_set() and backend.kokoro_instance is None
        assert not backend._native_tasks
    finally:
        release.set()
        timer.cancel()
        timer.join()
        await asyncio.gather(
            loading, *([closing] if closing else []), return_exceptions=True
        )
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asset", ["model", "voices"])
async def test_cancelled_asset_download_joins_writer_and_removes_temporary_file(
    tmp_path, monkeypatch, asset
):
    from tldw_chatbook.TTS.backends import kokoro

    entered, release, finished = (threading.Event() for _ in range(3))
    temporary = tmp_path / "owned-download.partial"
    real_mkstemp = kokoro.tempfile.mkstemp
    backend = make_backend(tmp_path, None)
    backend._kokoro_onnx = SimpleNamespace(
        Kokoro=lambda *a, **kw: object(), EspeakConfig=dict
    )
    backend.model_path = str(tmp_path / "model.onnx")
    backend.voices_json = str(tmp_path / "voices.bin")
    if asset == "voices":
        (tmp_path / "model.onnx").touch()

    def mkstemp(**kwargs):
        fd, path = real_mkstemp(dir=tmp_path)
        import os

        os.rename(path, temporary)
        return fd, str(temporary)

    def download(url, path, **kwargs):
        entered.set()
        assert release.wait(5)
        from pathlib import Path

        Path(path).write_bytes(b"finished controlled download")
        finished.set()

    monkeypatch.setattr(kokoro.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(kokoro, "_kokoro_stream_download", download)
    monkeypatch.setattr(kokoro, "REQUESTS_AVAILABLE", True)
    loading = asyncio.create_task(backend._initialize_onnx())
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        loading.cancel()
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0.03)
        assert not closing.done(), "Close abandoned a native asset writer"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await loading
        await closing
        assert finished.is_set()
        assert not temporary.exists()
        assert not (
            tmp_path / ("model.onnx" if asset == "model" else "voices.bin")
        ).exists()
    finally:
        release.set()
        await asyncio.gather(
            loading, *([closing] if closing else []), return_exceptions=True
        )
        await backend.close()
