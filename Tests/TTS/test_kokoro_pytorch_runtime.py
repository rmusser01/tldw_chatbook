"""Kokoro must use its real runtime contract, never placeholder waveforms.

The heavyweight external model/pipeline is replaced here; waveform delivery,
language selection, local voice loading and mixing remain production code.
Real checkpoint and intelligibility checks are recorded separately in QA.
"""

import asyncio
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tldw_chatbook.TTS import kokoro_pytorch
from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    """Strict upstream boundary; local files mirror official component/pack shapes."""
    checkpoint = tmp_path / "kokoro-v1_0.pth"
    torch.save({"decoder": {"weight": torch.tensor([7.0])}}, checkpoint)
    (tmp_path / "config.json").write_text(json.dumps({"vocab": {"a": 1}}))
    voice = torch.full((510, 1, 256), 0.25)
    torch.save(voice, tmp_path / "af_heart.pt")
    state = SimpleNamespace(
        language="a",
        speed=1.0,
        text="First sentence. Last sentence.",
        voices=voice,
        chunks=[torch.tensor([0.1, 0.2]), torch.tensor([-0.3, 0.4])],
        failure=None,
        pipelines=[],
        pipeline_exit=False,
        pipeline_import_error=False,
        phoneme_count=20,
    )

    class Model:
        def __init__(self, *, repo_id, config, model):
            assert repo_id == "hexgrad/Kokoro-82M"
            assert json.loads(Path(config).read_text()) == {"vocab": {"a": 1}}
            assert torch.load(model, weights_only=True)["decoder"]["weight"].item() == 7
            self.device = "cpu"
            self.decoder = SimpleNamespace(
                generator=SimpleNamespace(stft=torch.nn.Identity())
            )

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    class Pipeline:
        def __init__(self, *, lang_code, model, repo_id):
            if state.pipeline_exit:
                raise SystemExit(1)
            if state.pipeline_import_error:
                raise ModuleNotFoundError("PRIVATE missing language package")
            assert repo_id == "hexgrad/Kokoro-82M"
            assert isinstance(model, Model)
            assert lang_code == state.language
            state.pipelines.append(self)
            self.lang_code = lang_code

        def g2p(self, text):
            return "a" * state.phoneme_count, []

        def __call__(self, text, *, voice, speed):
            assert text == state.text
            assert speed == state.speed
            assert voice.dtype == torch.float32 and voice.device.type == "cpu"
            torch.testing.assert_close(voice, state.voices)
            for index, samples in enumerate(state.chunks):
                yield SimpleNamespace(audio=samples, phonemes=f"part{index}")
            if state.failure:
                raise state.failure

    monkeypatch.setitem(
        sys.modules, "kokoro", SimpleNamespace(KModel=Model, KPipeline=Pipeline)
    )
    return checkpoint, state


@pytest.mark.parametrize(
    "language,code",
    [
        ("a", "a"),
        ("en", "a"),
        ("en-US", "a"),
        ("en_gb", "b"),
        ("fr", "f"),
        ("es", "e"),
        ("hi", "h"),
        ("it", "i"),
        ("pt-br", "p"),
        ("ja", "j"),
        ("zh", "z"),
    ],
)
def test_official_pipeline_retains_all_segments_and_requested_language(
    upstream, language, code
):
    checkpoint, state = upstream
    state.language = code
    state.speed = 1.25
    model = kokoro_pytorch.build_model(str(checkpoint), device="cpu")
    audio, phonemes = kokoro_pytorch.generate(
        model,
        state.text,
        state.voices.double(),
        lang=language,
        speed=1.25,
    )
    np.testing.assert_array_equal(
        audio, np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32)
    )
    assert phonemes == "part0 part1"
    assert audio.dtype == np.float32


def test_named_voice_uses_local_pack_and_reuses_language_pipeline(upstream):
    checkpoint, state = upstream
    model = kokoro_pytorch.build_model(str(checkpoint))
    for _ in range(2):
        audio, _ = kokoro_pytorch.generate(
            model, state.text, "af_heart", voice_dir=str(checkpoint.parent)
        )
        assert len(audio) == 4
    assert len(state.pipelines) == 1


@pytest.mark.parametrize(
    "escape", ["traversal", "absolute", "absolute_file", "symlink"]
)
def test_named_voice_cannot_read_outside_configured_directory(upstream, escape):
    checkpoint, state = upstream
    directory = checkpoint.parent / "voices"
    directory.mkdir()
    outside = checkpoint.parent / "af_outside.pt"
    torch.save(state.voices, outside)
    if escape == "symlink":
        (directory / "af_link.pt").symlink_to(outside)
        name = "af_link"
    else:
        name = {
            "traversal": "../af_outside",
            "absolute": str(outside.with_suffix("")),
            "absolute_file": str(outside),
        }[escape]
    model = kokoro_pytorch.build_model(str(checkpoint))
    with pytest.raises(ValueError):
        kokoro_pytorch.generate(model, state.text, name, voice_dir=str(directory))


def test_explicit_voice_load_uses_normalized_validated_path(upstream, monkeypatch):
    checkpoint, state = upstream
    voice_path = checkpoint.parent / "af_heart.pt"
    alias = checkpoint.parent / "af_alias.pt"
    alias.symlink_to(voice_path)
    loaded_paths = []
    original_load = torch.load

    def load(path, **kwargs):
        loaded_paths.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    actual = kokoro_pytorch.load_voice(str(alias))
    torch.testing.assert_close(actual, state.voices)
    assert loaded_paths == [voice_path.resolve()]


def test_available_voices_excludes_links_outside_configured_directory(upstream):
    checkpoint, state = upstream
    directory = checkpoint.parent / "voices"
    directory.mkdir()
    torch.save(state.voices, directory / "af_safe.pt")
    (directory / "af_link.pt").symlink_to(checkpoint.parent / "af_heart.pt")
    assert kokoro_pytorch.get_available_voices(str(directory)) == ["af_safe"]


def test_pipeline_failure_after_segment_does_not_return_partial_success(upstream):
    checkpoint, state = upstream
    state.failure = RuntimeError("inference failed")
    model = kokoro_pytorch.build_model(str(checkpoint))
    with pytest.raises(RuntimeError, match="inference failed"):
        kokoro_pytorch.generate(model, state.text, state.voices)


@pytest.mark.parametrize(
    "chunks", [[], [torch.tensor([float("nan")])], [torch.ones(2, 3)]]
)
def test_invalid_pipeline_audio_fails_without_replacement_noise(upstream, chunks):
    checkpoint, state = upstream
    state.chunks = chunks
    model = kokoro_pytorch.build_model(str(checkpoint))
    with pytest.raises(ValueError, match="audio"):
        kokoro_pytorch.generate(model, state.text, state.voices)


def test_unknown_language_is_rejected_before_inference(upstream):
    checkpoint, state = upstream
    model = kokoro_pytorch.build_model(str(checkpoint))
    with pytest.raises(ValueError, match="language"):
        kokoro_pytorch.generate(model, state.text, state.voices, lang="not-a-language")


def test_upstream_language_installer_exit_becomes_recoverable_error(upstream):
    checkpoint, state = upstream
    state.pipeline_exit = True
    model = kokoro_pytorch.build_model(str(checkpoint))
    with pytest.raises(RuntimeError, match=r"(?s)language.*spacy"):
        kokoro_pytorch.generate(model, state.text, state.voices)


def test_oversized_non_english_phonemes_fail_instead_of_truncating(upstream):
    checkpoint, state = upstream
    state.language = "e"
    state.phoneme_count = 511
    model = kokoro_pytorch.build_model(str(checkpoint))
    with pytest.raises(TTSOperationError, match=r"(?s)phoneme.*[Ss]plit"):
        kokoro_pytorch.generate(model, state.text, state.voices, lang="es")


def test_mixing_real_voice_pack_dimensions_preserves_shape():
    mixed = kokoro_pytorch.mix_voices(
        [
            (torch.ones(510, 1, 256), 1.0),
            (torch.full((510, 1, 256), 4.0), 2.0),
        ]
    )
    assert mixed.shape == (510, 1, 256)
    torch.testing.assert_close(mixed, torch.full((510, 1, 256), 3.0))


@pytest.mark.parametrize("weights", [(0.0, 0.0), (-1.0, 2.0), (float("nan"), 1.0)])
def test_invalid_blend_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weight"):
        kokoro_pytorch.mix_voices(
            [(torch.ones(510, 1, 256), weight) for weight in weights]
        )


def test_missing_runtime_reports_installation_and_python_compatibility(
    upstream, monkeypatch
):
    checkpoint, _ = upstream
    monkeypatch.setitem(sys.modules, "kokoro", None)
    with pytest.raises(TTSOperationError, match=r"(?s)Kokoro.*Python.*ONNX"):
        kokoro_pytorch.build_model(str(checkpoint))


@pytest.mark.parametrize("device", ["cpu", "mps", "mps:0"])
def test_mps_wraps_only_fourier_operations_after_checkpoint_load(upstream, device):
    checkpoint, _ = upstream
    model = kokoro_pytorch.build_model(str(checkpoint), device=device)
    stft = model.model.decoder.generator.stft
    assert model.model.device == device
    if device == "cpu":
        assert isinstance(stft, torch.nn.Identity)
    else:
        assert isinstance(stft, kokoro_pytorch._CpuSTFT)
        assert isinstance(stft.stft, torch.nn.Identity)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires real MPS")
def test_mps_fourier_fallback_preserves_upstream_cpu_math():
    TorchSTFT = pytest.importorskip("kokoro.istftnet").TorchSTFT

    upstream_stft = TorchSTFT(filter_length=32, hop_length=8, win_length=32)
    signal = torch.sin(torch.linspace(0, 20, 256)).unsqueeze(0)
    expected_magnitude, expected_phase = upstream_stft.transform(signal)
    expected_audio = upstream_stft.inverse(expected_magnitude, expected_phase)
    fallback = kokoro_pytorch._CpuSTFT(upstream_stft).to("mps")
    magnitude, phase = fallback.transform(signal.to("mps"))
    actual_audio = fallback.inverse(magnitude, phase)
    assert (
        magnitude.device.type == phase.device.type == actual_audio.device.type == "mps"
    )
    torch.testing.assert_close(magnitude.cpu(), expected_magnitude, rtol=0, atol=0)
    torch.testing.assert_close(phase.cpu(), expected_phase, rtol=0, atol=0)
    torch.testing.assert_close(actual_audio.cpu(), expected_audio, rtol=0, atol=0)


@pytest.mark.parametrize(
    "failure,words",
    [
        ("runtime", ["Kokoro", "Python", "ONNX"]),
        ("language", ["Kokoro", "spacy", "en_core_web_sm"]),
        ("length", ["Kokoro", "510", "newlines"]),
        ("japanese", ["Kokoro", "misaki[ja]"]),
        ("chinese", ["Kokoro", "misaki[zh]"]),
    ],
)
def test_known_runtime_failure_guidance_survives_both_ui_boundaries(
    upstream, monkeypatch, failure, words
):
    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSEventHandler
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler

    checkpoint, state = upstream
    model = kokoro_pytorch.build_model(str(checkpoint))
    if failure == "runtime":
        monkeypatch.setitem(sys.modules, "kokoro", None)
    elif failure == "language":
        state.pipeline_exit = True
    elif failure == "length":
        state.language = "e"
        state.phoneme_count = 511
    else:
        state.language = "j" if failure == "japanese" else "z"
        state.pipeline_import_error = True
    with pytest.raises(TTSOperationError) as captured:
        kokoro_pytorch.generate(model, state.text, state.voices, lang=state.language)
    for copy in (
        STTSEventHandler._generation_error_copy,
        TTSEventHandler._tts_error_copy,
    ):
        message = copy(captured.value)
        assert all(word in message for word in words), message
    # Console must use allowlisted recovery copy, not upstream exception text.
    private_error = TTSOperationError(
        code=captured.value.code,
        message="PRIVATE upstream details",
        retryable=captured.value.retryable,
        operation_id="fixture",
        recovery_action=captured.value.recovery_action,
    )
    assert "PRIVATE" not in TTSEventHandler._tts_error_copy(private_error)


@pytest.mark.asyncio
async def test_backend_blends_full_local_packs_before_official_inference(
    upstream, backend, monkeypatch
):
    checkpoint, state = upstream
    torch.save(torch.ones(510, 1, 256), checkpoint.parent / "af_sky.pt")
    state.voices = torch.full((510, 1, 256), 0.8125)
    backend.voice_dir = str(checkpoint.parent)
    backend.enable_voice_mixing = True
    backend.model_loaded = True
    backend.kokoro_model_pt = kokoro_pytorch.build_model(str(checkpoint))
    backend._kokoro_pt_modules = {
        "generate": kokoro_pytorch.generate,
        "load_voice": kokoro_pytorch.load_voice,
        "mix_voices": kokoro_pytorch.mix_voices,
    }

    async def installed_voice(name):
        assert (checkpoint.parent / f"{name}.pt").is_file(), name

    monkeypatch.setattr(backend, "_download_voice_if_needed", installed_voice)
    request = OpenAISpeechRequest(
        input=state.text,
        model="kokoro",
        voice="af_heart:1,af_sky:3",
        response_format="pcm",
    )
    data = b"".join([chunk async for chunk in backend.generate_speech_stream(request)])
    np.testing.assert_array_equal(
        np.frombuffer(data, dtype=np.int16), [3276, 6553, -9830, 13106]
    )


def test_pytorch_without_model_path_does_not_select_the_onnx_file(
    tmp_path, monkeypatch
):
    from tldw_chatbook.TTS.backends import kokoro

    def setting(section, key, default=None):
        return (
            "/models/onnx.onnx" if key == "KOKORO_ONNX_MODEL_PATH_DEFAULT" else default
        )

    monkeypatch.setattr(kokoro, "get_cli_setting", setting)
    backend = KokoroTTSBackend(
        {"KOKORO_USE_ONNX": False, "KOKORO_VOICE_BLENDS_DIR": str(tmp_path / "blend")}
    )
    assert backend.model_path is None


def test_pytorch_splitter_preserves_explicit_language_boundaries(backend):
    assert backend._split_text_for_pytorch("Primera frase.\nSegunda frase.") == [
        "Primera frase.",
        "Segunda frase.",
    ]


@pytest.mark.asyncio
async def test_onnx_failure_uses_configured_pytorch_checkpoint(backend, monkeypatch):
    backend.use_onnx = True
    expected = backend.model_path
    backend.config["KOKORO_PT_MODEL_PATH_DEFAULT"] = expected

    async def fail_onnx():
        backend.use_onnx = False

    async def initialize_pytorch():
        assert backend.model_path == expected

    monkeypatch.setattr(backend, "_initialize_onnx", fail_onnx)
    monkeypatch.setattr(backend, "_initialize_pytorch", initialize_pytorch)
    await backend.load_model()


@pytest.fixture
def backend(tmp_path):
    return KokoroTTSBackend(
        {
            "KOKORO_USE_ONNX": False,
            "KOKORO_MODEL_PATH": str(tmp_path / "model.pth"),
            "KOKORO_VOICE_BLENDS_DIR": str(tmp_path / "blends"),
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route", ["download", "load", "load_fallback", "stream", "timestamps"]
)
@pytest.mark.parametrize("escape", ["traversal", "absolute", "symlink"])
async def test_backend_named_voice_rejects_escape_before_loading(
    upstream, backend, monkeypatch, route, escape
):
    checkpoint, state = upstream
    directory = checkpoint.parent / "voices"
    directory.mkdir()
    outside = checkpoint.parent / "af_heart.pt"
    if escape == "symlink":
        (directory / "af_link.pt").symlink_to(outside)
        voice = "af_link"
    else:
        voice = "../af_heart" if escape == "traversal" else str(outside.with_suffix(""))
    backend.voice_dir = str(directory)
    backend.model_loaded = True
    backend.kokoro_model_pt = kokoro_pytorch.build_model(str(checkpoint))
    backend._kokoro_pt_modules = {
        "generate": kokoro_pytorch.generate,
        "load_voice": kokoro_pytorch.load_voice,
    }
    loaded_paths = []
    original_load = torch.load

    def load(path, **kwargs):
        loaded_paths.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    with pytest.raises((ValueError, TTSOperationError)):
        if route == "download":
            await backend._download_voice_if_needed(voice)
        elif route in {"load", "load_fallback"}:
            if route == "load_fallback":
                backend._kokoro_pt_modules = None
            backend._load_voice_pack(voice)
        elif route == "timestamps":
            await backend.generate_with_timestamps(state.text, voice)
        else:
            request = OpenAISpeechRequest(
                input=state.text, model="kokoro", voice=voice, response_format="pcm"
            )
            _ = [chunk async for chunk in backend.generate_speech_stream(request)]
    assert loaded_paths == [], "An untrusted named voice reached tensor loading"


@pytest.mark.asyncio
async def test_backend_missing_voice_rejects_escape_before_download(
    backend, monkeypatch, tmp_path
):
    from tldw_chatbook.TTS.backends import kokoro

    downloads = []

    def download(url, destination, **kwargs):
        downloads.append(destination)
        return destination

    backend.voice_dir = str(tmp_path / "voices")
    monkeypatch.setattr(kokoro, "_kokoro_stream_download", download)
    with pytest.raises(ValueError):
        await backend._download_voice_if_needed("../af_missing")
    assert downloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["onnx", "pytorch"])
@pytest.mark.parametrize("timestamps", [False, True])
@pytest.mark.parametrize(
    "voice,language", [("hf_alpha", "hi"), ("im_nicola", "it"), ("pf_dora", "pt-br")]
)
async def test_backend_voice_language_is_consistent_for_all_generation_paths(
    backend, monkeypatch, engine, timestamps, voice, language
):
    calls = []
    samples = np.array([0.25, -0.25], dtype=np.float32)

    async def create_stream(text, *, voice, speed, lang):
        calls.append((voice, lang))
        yield samples, 24000

    def generate(model, text, voice_pack, *, lang, speed, voice_dir=None):
        calls.append((voice, lang))
        return samples, "speech"

    backend.model_loaded = True
    backend.use_onnx = engine == "onnx"
    backend.kokoro_instance = SimpleNamespace(create_stream=create_stream)
    backend.kokoro_model_pt = object()
    backend._kokoro_pt_modules = {"generate": generate}
    monkeypatch.setattr(backend, "_download_voice_if_needed", AsyncMock())
    monkeypatch.setattr(backend, "_load_voice_pack", lambda _: torch.ones(510, 1, 256))
    if timestamps:
        data, timings = await backend.generate_with_timestamps("Test speech", voice)
        assert data.startswith(b"RIFF") and timings
    else:
        request = OpenAISpeechRequest(
            input="Test speech", model="kokoro", voice=voice, response_format="pcm"
        )
        data = b"".join(
            [chunk async for chunk in backend.generate_speech_stream(request)]
        )
        np.testing.assert_array_equal(
            np.frombuffer(data, dtype=np.int16), [8191, -8191]
        )
    assert calls == [(voice, language)]


@pytest.mark.asyncio
async def test_existing_model_initialization_runs_off_event_loop(backend, monkeypatch):
    from pathlib import Path

    Path(backend.model_path).touch()
    loop_thread = threading.get_ident()
    threads = []

    def load():
        threads.append(threading.get_ident())
        backend.kokoro_model_pt = object()

    # Old unrelated tokenizer preparation is isolated to expose loader blocking.
    backend._nltk = SimpleNamespace(data=SimpleNamespace(find=lambda _: True))
    backend._transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: object())
    )
    monkeypatch.setattr(backend, "_load_pytorch_model", load)
    await backend._initialize_pytorch()
    assert backend.kokoro_model_pt is not None
    assert threads and loop_thread not in threads


@pytest.mark.asyncio
async def test_missing_torch_dependency_is_actionable_during_initialization(
    backend, monkeypatch
):
    Path(backend.model_path).touch()

    def missing_dependency():
        raise ImportError("PRIVATE missing torch dependency")

    monkeypatch.setattr(backend, "_load_pytorch_model", missing_dependency)
    with pytest.raises(TTSOperationError) as captured:
        await backend._initialize_pytorch()
    assert captured.value.code == "dependency_missing"
    assert captured.value.recovery_action == "install_kokoro_pytorch"
    assert "ONNX" in str(captured.value) and "PRIVATE" not in str(captured.value)


@pytest.mark.asyncio
async def test_first_generation_reaches_deferred_model_download(backend, monkeypatch):
    backend.model_loaded = True
    monkeypatch.setattr(backend, "_download_voice_if_needed", AsyncMock())
    monkeypatch.setattr(backend, "_load_voice_pack", lambda _: torch.ones(510, 1, 256))

    async def download():
        backend.kokoro_model_pt = object()
        backend._kokoro_pt_modules = {
            "generate": lambda *args, **kwargs: (np.array([0.25, -0.25]), "speech")
        }

    monkeypatch.setattr(backend, "_download_model_if_needed", download)
    request = OpenAISpeechRequest(
        input="First speech", model="kokoro", voice="af_heart", response_format="pcm"
    )
    data = b"".join([chunk async for chunk in backend.generate_speech_stream(request)])
    np.testing.assert_array_equal(np.frombuffer(data, dtype=np.int16), [8191, -8191])


@pytest.mark.asyncio
async def test_cancelled_initialization_and_close_join_the_real_worker(
    backend, monkeypatch
):
    from pathlib import Path

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    Path(backend.model_path).touch()

    def build_model(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        finished.set()
        return object()

    backend._kokoro_pt_modules = {"build_model": build_model}
    task = asyncio.create_task(backend._initialize_pytorch())
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0.03)
        assert not task.done(), "Cancellation released an active native loader"
        assert not closing.done(), "Close returned before the native loader finished"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await closing
        assert finished.is_set()
        assert backend.kokoro_model_pt is None and not backend.model_loaded
    finally:
        release.set()
        await asyncio.gather(
            task, *([closing] if closing else []), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_inference_failure_is_retryable_not_a_settings_error(
    backend, monkeypatch
):
    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSEventHandler
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler

    def fail(*args, **kwargs):
        raise NotImplementedError("PRIVATE unsupported native operation")

    backend.model_loaded = True
    backend.kokoro_model_pt = object()
    backend._kokoro_pt_modules = {"generate": fail}
    monkeypatch.setattr(backend, "_download_voice_if_needed", AsyncMock())
    monkeypatch.setattr(backend, "_load_voice_pack", lambda _: torch.ones(510, 1, 256))
    request = OpenAISpeechRequest(
        input="Test speech", model="kokoro", voice="af_heart", response_format="pcm"
    )
    with pytest.raises(TTSOperationError) as captured:
        _ = [chunk async for chunk in backend.generate_speech_stream(request)]
    assert captured.value.code == "generation_failed"
    assert captured.value.retryable
    for copy in (
        STTSEventHandler._generation_error_copy,
        TTSEventHandler._tts_error_copy,
    ):
        message = copy(captured.value)
        assert "PRIVATE" not in message and "not configured" not in message


@pytest.mark.asyncio
async def test_timestamp_cancellation_joins_inference_before_close(
    backend, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def generate(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        finished.set()
        return np.ones(2400, dtype=np.float32), "speech"

    backend.kokoro_model_pt = object()
    backend._kokoro_pt_modules = {"generate": generate}
    monkeypatch.setattr(backend, "_download_voice_if_needed", AsyncMock())
    monkeypatch.setattr(backend, "_load_voice_pack", lambda _: torch.ones(510, 1, 256))
    task = asyncio.create_task(
        backend.generate_with_timestamps("Test speech", voice="af_heart")
    )
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0.03)
        assert not task.done() and not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await closing
        assert finished.is_set() and backend.kokoro_model_pt is None
        assert not backend._pytorch_tasks
    finally:
        release.set()
        await asyncio.gather(
            task, *([closing] if closing else []), return_exceptions=True
        )
