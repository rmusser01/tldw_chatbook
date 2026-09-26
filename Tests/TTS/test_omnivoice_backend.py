"""OmniVoice ONNX backend — resolution, errors, single-chunk generation.

Fake-session tests per the plan: no model files, no onnxruntime sessions.
The LM/decoder/tokenizer seams are monkeypatched exactly as the engine's
lazy loaders would build them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tldw_chatbook.TTS.backends.omnivoice as omnivoice_module
from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.TTS.backends.omnivoice import (
    OmniVoiceOnnxTTSBackend,
    resolve_model_root,
)

_REQUIRED_TREE = (
    "omnivoice_lm_int8_hq/model.onnx",
    "omnivoice_lm_int8_hq/model.onnx_data",
    "audio_tokenizer_decoder_int8/model.onnx",
    "audio_tokenizer_decoder_int8/model.onnx_data",
    "audio_tokenizer_encoder_int8/model.onnx",
    "audio_tokenizer_encoder_int8/model.onnx_data",
    "tokenizer.json",
    "config.json",
)


def _make_tree(root: Path) -> None:
    for rel in _REQUIRED_TREE:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


class _Tok:
    """Fake tokenizers.Tokenizer: one id per whitespace word."""

    def __init__(self) -> None:
        self.encodes: list[str] = []

    def encode(self, text, add_special_tokens=False):
        self.encodes.append(text)
        return [i % 900 + 10 for i, _ in enumerate(text.split())]


class _StaticLM:
    """Deterministic LM: every position wants token (position_index % 1025)."""

    def run(self, input_ids, audio_mask, attention_mask):
        b, c, s = input_ids.shape
        want = np.tile(np.arange(s, dtype=np.float32) % 1025, (b, c, 1))
        logits = np.zeros((b, c, s, 1025), dtype=np.float32)
        np.put_along_axis(logits, want.astype(np.int64)[..., None], 5.0, axis=-1)
        return logits


class _FakeDecoder:
    """(1, 8, T) codes -> [zeros(T * 320)] float32 at 24 kHz."""

    def __init__(self) -> None:
        self.seen_shapes: list[tuple[int, ...]] = []

    def run(self, codes):
        self.seen_shapes.append(codes.shape)
        t = codes.shape[2]
        return [np.zeros(t * 320, dtype=np.float32)]


def _make_backend(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_config: dict | None = None,
) -> tuple[OmniVoiceOnnxTTSBackend, _Tok, _FakeDecoder]:
    config = {"OMNIVOICE_MODEL_ROOT": str(root), "OMNIVOICE_NUM_STEPS": 2}
    if extra_config:
        config.update(extra_config)
    backend = OmniVoiceOnnxTTSBackend(config)
    tokenizer = _Tok()
    decoder = _FakeDecoder()
    monkeypatch.setattr(backend, "_create_lm_runner", lambda: _StaticLM())
    monkeypatch.setattr(backend, "_create_decoder_session", lambda: decoder)
    monkeypatch.setattr(backend, "_load_tokenizer", lambda path: tokenizer)
    return backend, tokenizer, decoder


# --- resolution ---------------------------------------------------------------


def test_resolve_prefers_explicit_root(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    _make_tree(explicit)
    assert (
        resolve_model_root({"OMNIVOICE_MODEL_ROOT": str(explicit)}, managed=lambda: None)
        == explicit
    )


def test_resolve_uses_managed_root_when_no_explicit(tmp_path: Path) -> None:
    managed_root = tmp_path / "managed"
    _make_tree(managed_root)
    assert resolve_model_root({}, managed=lambda: managed_root) == managed_root


def test_resolve_invalid_layout_raises_model_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(TTSOperationError, match="layout") as err:
        resolve_model_root({"OMNIVOICE_MODEL_ROOT": str(bad)}, managed=lambda: None)
    assert err.value.code == "model_invalid"


def test_resolve_without_any_source_raises_not_configured() -> None:
    with pytest.raises(TTSOperationError, match="not_configured") as err:
        resolve_model_root({}, managed=lambda: None)
    assert err.value.code == "not_configured"


def _install_omnivoice_like_artifact(tmp_path: Path) -> Any:
    """Install/activate an omnivoice-reference artifact through the real service."""
    import hashlib

    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactDescriptor,
        ArtifactFile,
        ArtifactFormat,
        ArtifactRole,
        ModelArtifactService,
        ProvenanceClass,
    )
    from tldw_chatbook.TTS.omnivoice_artifact_catalog import omnivoice_onnx_reference

    core = ModelArtifactService(tmp_path / "root")
    reference = omnivoice_onnx_reference()
    payload = b"tiny-config-bytes"
    descriptor = ArtifactDescriptor(
        reference=reference,
        model_id="omnivoice-onnx-int8hq",
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="tts",
        model_family="omnivoice",
        upstream_repository="ct03/omnivoice-onnx-int8hq",
        upstream_revision=reference.revision,
        source_url="https://example.test/model",
        precision=reference.variant,
        license_id="test-license",
        license_url="https://example.test/license",
        usage_notice="Test model",
        runtime_name="onnx-tts",
        runtime_version_constraint="none",
        supported_os=("darwin",),
        supported_architectures=("arm64",),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
        files=(
            ArtifactFile("config.json", len(payload), hashlib.sha256(payload).hexdigest()),
        ),
        expected_installed_bytes=len(payload),
        dependencies=(),
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_bytes(payload)
    core.install(descriptor, source_dir)
    core.activate(reference)
    return core, reference


def test_managed_root_finds_installed_active_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.Model_Artifacts import store as artifact_store

    core, reference = _install_omnivoice_like_artifact(tmp_path)
    monkeypatch.setattr(artifact_store, "managed_service", lambda root=None: core)
    assert omnivoice_module._managed_root() == core.artifact_path(reference)


def test_managed_root_returns_none_when_nothing_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.Model_Artifacts import store as artifact_store
    from tldw_chatbook.Model_Artifacts.service import ModelArtifactService

    core = ModelArtifactService(tmp_path / "root")
    monkeypatch.setattr(artifact_store, "managed_service", lambda root=None: core)
    assert omnivoice_module._managed_root() is None


# --- lifecycle errors ---------------------------------------------------------


async def test_initialize_dependency_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    monkeypatch.setattr(omnivoice_module, "_dependency_available", lambda: False)
    backend = OmniVoiceOnnxTTSBackend({"OMNIVOICE_MODEL_ROOT": str(root)})
    with pytest.raises(TTSOperationError, match="dependency_missing") as err:
        await backend.initialize()
    assert err.value.code == "dependency_missing"


async def test_generate_with_invalid_layout_maps_model_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    backend = OmniVoiceOnnxTTSBackend({"OMNIVOICE_MODEL_ROOT": str(bad)})
    with pytest.raises(TTSOperationError, match="layout") as err:
        async for _ in backend.generate_speech_stream(text="hello", voice=""):
            pass
    assert err.value.code == "model_invalid"


async def test_generate_with_zero_steps_maps_configuration_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, _, _ = _make_backend(tmp_path / "m", monkeypatch)
    _make_tree(tmp_path / "m")
    backend.config["OMNIVOICE_NUM_STEPS"] = 0
    with pytest.raises(TTSOperationError) as err:
        async for _ in backend.generate_speech_stream(text="hello", voice=""):
            pass
    assert err.value.code == "configuration_invalid"


# --- generation ---------------------------------------------------------------


async def test_generate_single_chunk_with_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, decoder = _make_backend(root, monkeypatch)
    chunks = [c async for c in backend.generate_speech_stream(text="hello world", voice="")]
    assert len(chunks) == 1
    # stdlib-wave 16-bit PCM mono: 44-byte header plus payload
    assert chunks[0][:4] == b"RIFF"
    assert chunks[0][8:12] == b"WAVE"
    assert len(chunks[0]) > 44
    assert (len(chunks[0]) - 44) % 2 == 0
    # decoder received exactly one (1, 8, T) codes batch
    assert len(decoder.seen_shapes) == 1
    assert decoder.seen_shapes[0][0] == 1 and decoder.seen_shapes[0][1] == 8


async def test_generate_passes_prompt_audio_mask_to_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)
    captured: dict = {}

    def fake_sampling(
        lm, prompt_ids, target_len, *, config, cancel_check=None, progress=None,
        num_codebook=8, prompt_audio_mask=None,
    ):
        captured["prompt_ids"] = np.array(prompt_ids, copy=True)
        captured["target_len"] = target_len
        captured["prompt_audio_mask"] = (
            None if prompt_audio_mask is None else np.array(prompt_audio_mask, copy=True)
        )
        captured["steps"] = config.num_step
        return np.zeros((num_codebook, target_len), dtype=np.int64)

    monkeypatch.setattr(omnivoice_module, "run_diffusion_sampling", fake_sampling)
    chunks = [c async for c in backend.generate_speech_stream(text="hello world", voice="")]
    assert len(chunks) == 1

    # LOAD-BEARING: the sampler must receive the prompt's audio mask.
    mask = captured["prompt_audio_mask"]
    assert mask is not None
    assert mask.shape == captured["prompt_ids"].shape
    assert mask.dtype == np.bool_
    # no reference: only the target tail is audio-marked
    target_len = captured["target_len"]
    assert mask[:, : mask.shape[1] - target_len].sum() == 0
    assert mask[:, mask.shape[1] - target_len :].all()
    assert captured["steps"] == 2


async def test_generate_cloning_marks_reference_codes_as_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, tokenizer, _ = _make_backend(root, monkeypatch)
    ref_codes = np.full((8, 50), 7, dtype=np.int64)
    monkeypatch.setattr(
        backend, "_encode_reference", lambda path, max_duration=None: ref_codes
    )
    captured: dict = {}

    def fake_sampling(
        lm, prompt_ids, target_len, *, config, cancel_check=None, progress=None,
        num_codebook=8, prompt_audio_mask=None,
    ):
        captured["prompt_ids"] = np.array(prompt_ids, copy=True)
        captured["target_len"] = target_len
        captured["prompt_audio_mask"] = np.array(prompt_audio_mask, copy=True)
        return np.zeros((num_codebook, target_len), dtype=np.int64)

    monkeypatch.setattr(omnivoice_module, "run_diffusion_sampling", fake_sampling)
    chunks = [
        c
        async for c in backend.generate_speech_stream(
            text="hello world",
            voice="",
            reference_audio="ref.wav",
            reference_text="the reference transcript",
        )
    ]
    assert len(chunks) == 1

    ids = captured["prompt_ids"]
    mask = captured["prompt_audio_mask"]
    target_len = captured["target_len"]
    ref_len = 50
    prefix_len = ids.shape[1] - ref_len - target_len
    # reference codes are embedded verbatim on each codebook row ...
    assert np.array_equal(ids[:, prefix_len : prefix_len + ref_len], ref_codes)
    # ... and marked as AUDIO, not text, on the sampler's mask
    assert mask[:, prefix_len : prefix_len + ref_len].all()
    assert mask[:, :prefix_len].sum() == 0
    # the reference transcript is folded into the text segment
    assert any("the reference transcript" in text for text in tokenizer.encodes)


async def test_generate_reports_step_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)
    reports: list[dict] = []

    async def callback(info: dict) -> None:
        reports.append(info)

    backend.set_progress_callback(callback)
    chunks = [c async for c in backend.generate_speech_stream(text="hello world", voice="")]
    assert len(chunks) == 1
    assert [r["step"] for r in reports] == [1, 2]
    assert all(r["total_steps"] == 2 for r in reports)
    assert all(r["elapsed"] >= 0 for r in reports)


async def test_close_cancel_raises_cancelled_and_reload_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A close()-driven cancel surfaces as CancelledError (never a silent
    zero-chunk stream), releases the lock, and a reload clears the flag."""
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)
    chunks = [c async for c in backend.generate_speech_stream(text="hello world", voice="")]
    assert len(chunks) == 1
    backend._cancel.set()  # what close() does to an in-flight generation
    with pytest.raises(asyncio.CancelledError):
        async for _ in backend.generate_speech_stream(text="hello world", voice=""):
            pass
    assert not backend._generation_lock.locked()
    await backend.close()
    chunks = [c async for c in backend.generate_speech_stream(text="hello world", voice="")]
    assert len(chunks) == 1


async def test_timeout_raises_generation_timeout_and_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(
        root, monkeypatch, extra_config={"OMNIVOICE_TIMEOUT_FACTOR": 1e-9}
    )
    with pytest.raises(TTSOperationError) as err:
        async for _ in backend.generate_speech_stream(text="hello world", voice=""):
            pass
    assert err.value.code == "generation_timeout"
    assert not backend._generation_lock.locked()


async def test_decoder_failure_maps_generation_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)

    class ExplodingDecoder:
        def run(self, codes):
            raise RuntimeError("decoder exploded")

    monkeypatch.setattr(backend, "_create_decoder_session", lambda: ExplodingDecoder())
    with pytest.raises(TTSOperationError) as err:
        async for _ in backend.generate_speech_stream(text="hello world", voice=""):
            pass
    assert err.value.code == "generation_failed"


async def test_second_generation_while_locked_waits_single_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, decoder = _make_backend(root, monkeypatch)
    first = backend.generate_speech_stream(text="hello world", voice="")
    second = backend.generate_speech_stream(text="second pass", voice="")
    chunks = [chunk async for chunk in first]
    assert len(chunks) == 1
    chunks2 = [chunk async for chunk in second]
    assert len(chunks2) == 1
    # single-flight: exactly two decoder batches, strictly serialized
    assert len(decoder.seen_shapes) == 2


# --- request-shaped entry (sibling contract) ----------------------------------


async def test_generate_accepts_openai_speech_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest

    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)
    request = OpenAISpeechRequest(
        model="omnivoice", input="hello from request", voice="", response_format="wav"
    )
    chunks = [c async for c in backend.generate_speech_stream(request)]
    assert len(chunks) == 1
    assert chunks[0][:4] == b"RIFF"


async def test_generate_pcm_format_yields_raw_int16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)
    chunks = [
        c
        async for c in backend.generate_speech_stream(
            text="hello world", voice="", response_format="pcm"
        )
    ]
    assert len(chunks) == 1
    assert (len(chunks[0]) % 2) == 0
    assert chunks[0][:4] != b"RIFF"


# --- response-format routing --------------------------------------------------


class _StubConvertService:
    """audio_service stand-in recording the conversion request."""

    def __init__(self, payload: bytes = b"fake-encoded-bytes") -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def convert_audio(self, audio_data, target_format, source_format=None, sample_rate=None):
        self.calls.append(
            {
                "samples": getattr(audio_data, "size", None),
                "dtype": getattr(audio_data, "dtype", None),
                "target_format": target_format,
                "source_format": source_format,
                "sample_rate": sample_rate,
            }
        )
        return self.payload


async def test_generate_mp3_request_converts_via_audio_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The schema-default mp3 request must convert, never WAV-labeled mp3."""
    from tldw_chatbook.TTS import audio_service as audio_service_module
    from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest

    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)
    stub = _StubConvertService(b"ID3-fake-mp3")
    monkeypatch.setattr(backend, "audio_service", stub)
    monkeypatch.setattr(audio_service_module, "PYDUB_AVAILABLE", True)

    request = OpenAISpeechRequest(
        model="omnivoice", input="hello world", voice=""
        # response_format intentionally left at the schema default: "mp3"
    )
    chunks = [c async for c in backend.generate_speech_stream(request)]
    assert chunks == [b"ID3-fake-mp3"]
    assert chunks[0][:4] != b"RIFF"
    # the engine handed the float32 waveform to the converter as 24 kHz pcm
    (call,) = stub.calls
    assert call["target_format"] == "mp3"
    assert call["source_format"] == "pcm"
    assert call["sample_rate"] == 24000
    assert call["samples"] > 0
    assert str(call["dtype"]) == "float32"


async def test_generate_conversion_failure_maps_generation_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.TTS import audio_service as audio_service_module

    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)
    monkeypatch.setattr(audio_service_module, "PYDUB_AVAILABLE", True)

    class ExplodingService:
        async def convert_audio(self, *args, **kwargs):
            raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(backend, "audio_service", ExplodingService())
    with pytest.raises(TTSOperationError) as err:
        async for _ in backend.generate_speech_stream(
            text="hello world", voice="", response_format="mp3"
        ):
            pass
    assert err.value.code == "generation_failed"


async def test_unsupported_format_maps_request_invalid() -> None:
    backend = OmniVoiceOnnxTTSBackend({})
    with pytest.raises(TTSOperationError) as err:
        await backend._convert_format(np.zeros(8, dtype=np.float32), "wma")
    assert err.value.code == "request_invalid"
    assert "mp3" in str(err.value)
    assert "wav" in str(err.value)


async def test_generate_flac_converts_with_real_audio_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.TTS import audio_service as audio_service_module

    if audio_service_module.PYDUB_AVAILABLE or not audio_service_module.SOUNDFILE_AVAILABLE:
        pytest.skip("requires the soundfile-only conversion path")
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)
    chunks = [
        c
        async for c in backend.generate_speech_stream(
            text="hello world", voice="", response_format="flac"
        )
    ]
    assert len(chunks) == 1
    assert chunks[0][:4] == b"fLaC"


async def test_generate_mp3_without_converter_maps_dependency_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.TTS import audio_service as audio_service_module

    if audio_service_module.PYDUB_AVAILABLE:
        pytest.skip("requires the soundfile-only conversion path")
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)
    with pytest.raises(TTSOperationError) as err:
        async for _ in backend.generate_speech_stream(
            text="hello world", voice="", response_format="mp3"
        ):
            pass
    assert err.value.code == "dependency_missing"


def test_capabilities_advertise_supported_formats() -> None:
    backend = OmniVoiceOnnxTTSBackend({})
    formats = backend.get_capabilities()["formats"]
    assert set(formats) == {"mp3", "wav", "opus", "aac", "flac", "pcm"}


# --- lifecycle hygiene --------------------------------------------------------


async def test_close_releases_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)
    chunks = [c async for c in backend.generate_speech_stream(text="hi", voice="")]
    assert len(chunks) == 1
    assert backend._lm is not None
    await backend.close()
    assert backend._lm is None
    assert backend._decoder is None
    assert backend._tokenizer is None
    assert backend._cancel.is_set()
    assert not backend.model_loaded
    await asyncio.sleep(0)
