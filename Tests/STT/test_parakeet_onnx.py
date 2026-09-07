"""Focused tests for the offline executor-native Parakeet ONNX runtime."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
    PARAKEET_V2_MODEL,
    PARAKEET_V3_MODEL,
)
from tldw_chatbook.Model_Artifacts.leases import ArtifactLeaseKey
from tldw_chatbook.STT.contracts import (
    BufferAudioSource,
    TimestampGranularity,
    TranscriptionWarningCode,
)


class _FakeAsr:
    def __init__(self, texts: list[str]) -> None:
        self._texts = iter(texts)
        self.calls: list[tuple[object, object, dict[str, object]]] = []

    def _get_sample_rate(self) -> int:
        return 16_000

    def recognize_batch(self, waveforms, lengths, **kwargs):
        self.calls.append((waveforms, lengths, kwargs))
        return iter((SimpleNamespace(text=next(self._texts)),))


class _FakeModel:
    def __init__(
        self,
        *,
        short_text: str = "short text",
        recognize_texts: tuple[str, ...] = (),
        segments=(),
    ) -> None:
        self.short_text = short_text
        self._recognize_texts = iter(recognize_texts)
        self.short_calls: list[tuple[object, dict[str, object]]] = []
        self.asr = _FakeAsr(list(segments))
        self.resampler = lambda waveforms, lengths, sample_rate: (waveforms, lengths)

    def recognize(self, audio_path, **kwargs):
        self.short_calls.append((audio_path, kwargs))
        return next(self._recognize_texts, self.short_text)


class _FakeVad:
    def __init__(self, ranges: tuple[tuple[int, int], ...]) -> None:
        self.ranges = ranges
        self.calls: list[tuple[object, object, int]] = []

    def segment_batch(self, waveforms, lengths, sample_rate):
        self.calls.append((waveforms, lengths, sample_rate))
        return iter((iter(self.ranges),))


def _runtime(
    *,
    model_id: str = PARAKEET_V2_MODEL,
    precision: str = "int8",
    duration: float = 10.0,
    short_text: str = "short text",
    segment_texts: tuple[str, ...] = (),
    segment_ranges: tuple[tuple[int, int], ...] = (),
):
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime

    model = _FakeModel(short_text=short_text, segments=segment_texts)
    vad = _FakeVad(segment_ranges)
    root = ArtifactLeaseKey("parakeet-v2", "root-revision", precision)
    dependency = ArtifactLeaseKey("silero-vad", "vad-revision", "f32")
    runtime = ParakeetOnnxRuntime(
        model=model,
        vad=vad,
        model_id=model_id,
        precision=precision,
        artifact_root=root,
        artifact_dependencies=(dependency,),
        model_load_seconds=0.25,
        audio_reader=lambda path, channel=None: ([[0.0] * 64_000], [64_000], 16_000),
        pad_list=lambda chunks: (chunks, [len(chunk) for chunk in chunks]),
        duration_reader=lambda path: duration,
    )
    return runtime, model, vad, root, dependency


@pytest.mark.parametrize(
    ("precision", "expected_quantization"),
    [("int8", "int8"), ("f32", None)],
)
def test_load_uses_explicit_local_cpu_paths_and_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
    expected_quantization: str | None,
) -> None:
    from tldw_chatbook.STT import parakeet_onnx

    model_root = tmp_path / "model"
    vad_root = tmp_path / "vad"
    model_root.mkdir()
    vad_root.mkdir()
    calls = []
    fake_model = _FakeModel()
    fake_vad = _FakeVad(())
    api = SimpleNamespace(
        load_model=lambda *args, **kwargs: calls.append(("model", args, kwargs))
        or fake_model,
        load_vad=lambda *args, **kwargs: calls.append(("vad", args, kwargs))
        or fake_vad,
    )
    monkeypatch.setattr(
        parakeet_onnx,
        "_onnx_asr_api",
        lambda: (api, lambda *args, **kwargs: None, lambda chunks: None),
    )

    runtime = parakeet_onnx.ParakeetOnnxRuntime.load(
        model_root=model_root,
        vad_root=vad_root,
        model_id=PARAKEET_V3_MODEL,
        precision=precision,
        artifact_root=None,
        artifact_dependencies=(),
    )

    assert calls[0] == (
        "model",
        (PARAKEET_V3_MODEL,),
        {
            "path": model_root,
            "quantization": expected_quantization,
            "providers": ["CPUExecutionProvider"],
            "preprocessor_config": {
                "use_numpy_preprocessors": True,
                "max_concurrent_workers": 1,
            },
        },
    )
    assert calls[1] == (
        "vad",
        ("silero",),
        {"path": vad_root, "providers": ["CPUExecutionProvider"]},
    )
    runtime.close()


def test_native_runtime_import_obeys_optional_dependency_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.STT import parakeet_onnx
    from tldw_chatbook.Utils import optional_deps

    monkeypatch.setattr(
        optional_deps,
        "parakeet_onnx_deps_installed",
        lambda: False,
    )

    with pytest.raises(ModuleNotFoundError, match="onnx-asr"):
        parakeet_onnx._onnx_asr_api()


def test_short_v2_result_is_normalized_with_exact_artifact_provenance(
    tmp_path: Path,
) -> None:
    runtime, model, _vad, root, dependency = _runtime()

    result = runtime.transcribe(
        audio_path=tmp_path / "short.wav",
        attempt_id="attempt-1",
        batch_id="batch-1",
        job_id="job-1",
        language="en",
        timestamps=True,
    )

    assert result.text == "short text"
    assert [(item.start_seconds, item.end_seconds, item.text) for item in result.segments] == [
        (0.0, 10.0, "short text")
    ]
    assert model.short_calls == [(tmp_path / "short.wav", {})]
    assert result.provenance.artifact_root == root
    assert result.provenance.artifact_dependencies == (dependency,)
    assert result.provenance.requested_language == "en"
    assert result.provenance.effective_language == "en"
    assert result.provenance.detected_language is None
    assert result.warnings == ()


def test_external_runtime_provenance_has_only_managed_vad_identity(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime

    dependency = ArtifactLeaseKey("silero-vad", "vad-revision", "f32")
    runtime = ParakeetOnnxRuntime(
        model=_FakeModel(short_text="external text"),
        vad=_FakeVad(()),
        model_id=PARAKEET_V2_MODEL,
        precision="int8",
        artifact_root=None,
        artifact_dependencies=(dependency,),
        model_load_seconds=0.1,
        audio_reader=lambda *_args, **_kwargs: None,
        pad_list=lambda _chunks: None,
        duration_reader=lambda _path: 1.0,
    )

    result = runtime.transcribe(
        audio_path=tmp_path / "external.wav",
        attempt_id="external-attempt",
        language="en",
        timestamps=False,
    )

    assert result.provenance.artifact_root is None
    assert result.provenance.artifact_dependencies == (dependency,)


def test_resident_reuse_reports_model_load_only_for_first_attempt(
    tmp_path: Path,
) -> None:
    runtime, _model, _vad, _root, _dependency = _runtime()

    first = runtime.transcribe(
        audio_path=tmp_path / "first.wav",
        attempt_id="attempt-1",
        language="en",
        timestamps=False,
    )
    second = runtime.transcribe(
        audio_path=tmp_path / "second.wav",
        attempt_id="attempt-2",
        language="en",
        timestamps=False,
    )

    assert first.timings.model_load_seconds == 0.25
    assert first.timings.total_seconds == pytest.approx(
        0.25 + first.timings.inference_seconds
    )
    assert second.timings.model_load_seconds == 0.0
    assert second.timings.total_seconds == pytest.approx(
        second.timings.inference_seconds
    )


def test_v3_records_routing_assertion_without_decoder_language_constraint(
    tmp_path: Path,
) -> None:
    runtime, model, _vad, _root, _dependency = _runtime(
        model_id=PARAKEET_V3_MODEL,
        short_text="bonjour",
    )

    result = runtime.transcribe(
        audio_path=tmp_path / "short.wav",
        attempt_id="attempt-v3",
        language="fr",
        timestamps=False,
    )

    assert model.short_calls == [(tmp_path / "short.wav", {})]
    assert result.provenance.requested_language == "fr"
    assert result.provenance.effective_language == "auto"
    assert result.provenance.detected_language is None
    assert result.warnings == (
        TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,
    )


def test_long_form_uses_one_vad_segment_per_asr_batch(
    tmp_path: Path,
) -> None:
    runtime, model, vad, _root, _dependency = _runtime(
        duration=40.0,
        segment_texts=("one", "two"),
        segment_ranges=((0, 16_000), (32_000, 64_000)),
    )

    result = runtime.transcribe(
        audio_path=tmp_path / "long.wav",
        attempt_id="attempt-long",
        language="en",
        timestamps=True,
    )

    assert result.text == "one two"
    assert [(item.start_seconds, item.end_seconds, item.text) for item in result.segments] == [
        (0.0, 1.0, "one"),
        (2.0, 4.0, "two"),
    ]
    assert len(model.asr.calls) == 2
    assert all(len(call[0]) == 1 for call in model.asr.calls)
    assert all(call[2] == {} for call in model.asr.calls)
    assert len(vad.calls) == 1
    assert result.produced_capabilities.vad is True
    assert result.produced_capabilities.timestamps is TimestampGranularity.SEGMENT


def test_cancellation_before_second_segment_prevents_second_asr_call(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxCancelled

    runtime, model, _vad, _root, _dependency = _runtime(
        duration=40.0,
        segment_texts=("one", "must-not-run"),
        segment_ranges=((0, 16_000), (16_000, 32_000)),
    )

    with pytest.raises(ParakeetOnnxCancelled):
        runtime.transcribe(
            audio_path=tmp_path / "long.wav",
            attempt_id="attempt-cancel",
            language="en",
            timestamps=True,
            is_cancelled=lambda: len(model.asr.calls) == 1,
        )

    assert len(model.asr.calls) == 1


def test_long_direct_local_model_without_managed_vad_fails_with_retry_action(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxFailure

    runtime, _model, _vad, _root, _dependency = _runtime(duration=40.0)
    runtime._vad = None

    with pytest.raises(ParakeetOnnxFailure) as raised:
        runtime.transcribe(
            audio_path=tmp_path / "long.wav",
            attempt_id="attempt-no-vad",
            language="en",
            timestamps=True,
        )

    assert raised.value.error_detail == {
        "category": "stt_failure",
        "code": "artifact_incompatible",
        "message": "Long-form Parakeet requires the managed VAD dependency. "
        "Retry with faster-whisper.",
        "actions": ["retry_faster_whisper"],
    }
    assert raised.value.stt_failure_provenance == {
        "attempt_id": "attempt-no-vad",
        "batch_id": None,
        "job_id": None,
        "provider_id": "parakeet-onnx",
        "model_id": PARAKEET_V2_MODEL,
        "artifact_root": {
            "artifact_id": "parakeet-v2",
            "revision": "root-revision",
            "variant": "int8",
        },
        "artifact_dependencies": [
            {
                "artifact_id": "silero-vad",
                "revision": "vad-revision",
                "variant": "f32",
            }
        ],
        "precision": "int8",
        "requested_device": "cpu",
        "effective_device": "cpu",
        "requested_language": "en",
        "effective_language": "en",
        "detected_language": None,
        "task": "transcribe",
        "error_code": "artifact_incompatible",
    }


@pytest.mark.parametrize(
    ("channels", "pcm", "expected"),
    [
        (1, (-32768, 0, 16384, 32767), (-1.0, 0.0, 0.5, 32767 / 32768)),
        (2, (-32768, 0, 16384, 32767), (-0.5, (16384 + 32767) / 2 / 32768)),
    ],
)
def test_buffer_pcm_is_little_endian_mono_float32_with_exact_duration_and_no_staging(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
    pcm: tuple[int, ...],
    expected: tuple[float, ...],
) -> None:
    from tldw_chatbook.STT import parakeet_onnx
    from tldw_chatbook.Utils import optional_deps

    runtime, model, _vad, _root, _dependency = _runtime(short_text="ordinary text")
    dependency_requests: list[tuple[str, str]] = []

    def require_dependency(module_name: str, feature_name: str):
        dependency_requests.append((module_name, feature_name))
        return np

    monkeypatch.setattr(optional_deps, "require_dependency", require_dependency)
    source = BufferAudioSource(
        np.asarray(pcm, dtype="<i2").tobytes(),
        sample_rate=8_000,
        channels=channels,
    )
    monkeypatch.setattr(
        parakeet_onnx,
        "_prepared_wav",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("buffer PCM must not use file staging")
        ),
    )
    monkeypatch.setattr(
        parakeet_onnx.tempfile,
        "NamedTemporaryFile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("buffer PCM must not create a temporary file")
        ),
    )
    monkeypatch.setattr(
        parakeet_onnx.wave,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("buffer PCM must not stage a WAV")
        ),
    )

    result = runtime.transcribe_buffer(
        source=source,
        segment_end_frames=(),
        attempt_id="buffer-attempt",
        language="en",
        job_id=None,
    )

    waveform, kwargs = model.short_calls[0]
    np.testing.assert_allclose(waveform, np.asarray(expected, dtype=np.float32))
    assert waveform.dtype == np.float32
    assert kwargs == {"sample_rate": 8_000}
    assert result.normalized.text == "ordinary text"
    assert result.logical_segments == ("ordinary text",)
    assert result.normalized.duration_seconds == len(expected) / 8_000
    assert result.normalized.provenance.attempt_id == "buffer-attempt"
    assert result.normalized.provenance.job_id is None
    assert result.normalized.provenance.requested_language == "en"
    assert result.normalized.provenance.effective_language == "en"
    assert result.normalized.warnings == ()
    assert result.normalized.produced_capabilities.vad is False
    assert dependency_requests == [("numpy", "transcription_parakeet_onnx")]


def test_v3_buffer_preserves_routing_warning_semantics() -> None:
    runtime, _model, _vad, _root, _dependency = _runtime(
        model_id=PARAKEET_V3_MODEL,
        short_text="bonjour",
    )

    result = runtime.transcribe_buffer(
        source=BufferAudioSource(b"\x00\x00\x01\x00", 16_000),
        segment_end_frames=(),
        attempt_id="buffer-v3",
        language="fr",
    )

    assert result.normalized.provenance.requested_language == "fr"
    assert result.normalized.provenance.effective_language == "auto"
    assert result.normalized.provenance.detected_language is None
    assert result.normalized.warnings == (
        TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,
    )


def test_buffer_preserves_two_ordered_logical_segment_texts() -> None:
    runtime, model, _vad, _root, _dependency = _runtime()
    model._recognize_texts = iter(("ordinary text", "console stop"))
    source = BufferAudioSource(
        np.asarray((100, 200, 300, 400), dtype="<i2").tobytes(),
        sample_rate=16_000,
    )

    result = runtime.transcribe_buffer(
        source=source,
        segment_end_frames=(2, 4),
        attempt_id="two-logical-segments",
        language="en",
    )

    assert result.logical_segments == ("ordinary text", "console stop")
    assert result.normalized.text == "ordinary text console stop"
    assert len(model.short_calls) == 2
    np.testing.assert_allclose(
        model.short_calls[0][0],
        np.asarray((100, 200), dtype=np.float32) / 32768.0,
    )
    np.testing.assert_allclose(
        model.short_calls[1][0],
        np.asarray((300, 400), dtype=np.float32) / 32768.0,
    )


def test_managed_long_buffer_uses_resident_vad_in_memory() -> None:
    runtime, model, vad, _root, _dependency = _runtime(
        segment_texts=("managed long text",),
        segment_ranges=((0, 100),),
    )
    source = BufferAudioSource(
        np.arange(310, dtype="<i2").tobytes(),
        sample_rate=10,
    )

    result = runtime.transcribe_buffer(
        source=source,
        segment_end_frames=(),
        attempt_id="managed-long",
        language="en",
    )

    assert result.normalized.duration_seconds == 31.0
    assert result.normalized.produced_capabilities.vad is True
    assert result.logical_segments == ("managed long text",)
    assert len(vad.calls) == 1
    assert len(model.asr.calls) == 1
    assert model.short_calls == []


def test_verified_legacy_long_buffer_without_vad_recognizes_directly() -> None:
    runtime, model, _vad, _root, _dependency = _runtime(short_text="legacy long text")
    runtime._vad = None
    source = BufferAudioSource(bytes(310 * 2), sample_rate=10)

    result = runtime.transcribe_buffer(
        source=source,
        segment_end_frames=(),
        attempt_id="legacy-long",
        language="en",
    )

    assert result.normalized.duration_seconds == 31.0
    assert result.normalized.produced_capabilities.vad is False
    assert result.logical_segments == ("legacy long text",)
    assert len(model.short_calls) == 1


def test_v3_long_buffer_without_managed_vad_fails_before_native_inference() -> None:
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxFailure

    runtime, model, _vad, _root, _dependency = _runtime(
        model_id=PARAKEET_V3_MODEL,
        short_text="must-not-run",
    )
    runtime._vad = None
    source = BufferAudioSource(bytes(310 * 2), sample_rate=10)

    with pytest.raises(ParakeetOnnxFailure) as raised:
        runtime.transcribe_buffer(
            source=source,
            segment_end_frames=(),
            attempt_id="v3-long-no-vad",
            language="fr",
        )

    assert raised.value.error_detail == {
        "category": "stt_failure",
        "code": "artifact_incompatible",
        "message": "Long-form Parakeet v3 requires the managed VAD dependency. "
        "Retry with faster-whisper.",
        "actions": ["retry_faster_whisper"],
    }
    assert raised.value.stt_failure_provenance["attempt_id"] == "v3-long-no-vad"
    assert raised.value.stt_failure_provenance["job_id"] is None
    assert raised.value.stt_failure_provenance["requested_language"] == "fr"
    assert raised.value.stt_failure_provenance["effective_language"] == "auto"
    assert model.short_calls == []


def test_buffer_cancellation_before_second_logical_segment_prevents_native_call() -> None:
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxCancelled

    runtime, model, _vad, _root, _dependency = _runtime()
    model._recognize_texts = iter(("first", "must-not-run"))
    source = BufferAudioSource(bytes(8), sample_rate=16_000)

    with pytest.raises(ParakeetOnnxCancelled):
        runtime.transcribe_buffer(
            source=source,
            segment_end_frames=(2, 4),
            attempt_id="cancel-direct",
            language="en",
            is_cancelled=lambda: len(model.short_calls) == 1,
        )

    assert len(model.short_calls) == 1


def test_buffer_cancellation_checks_before_each_logical_vad_inference() -> None:
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxCancelled

    runtime, model, vad, _root, _dependency = _runtime(
        segment_texts=("first", "must-not-run"),
        segment_ranges=((0, 100),),
    )
    source = BufferAudioSource(bytes(640), sample_rate=10)

    with pytest.raises(ParakeetOnnxCancelled):
        runtime.transcribe_buffer(
            source=source,
            segment_end_frames=(160, 320),
            attempt_id="cancel-vad",
            language="en",
            is_cancelled=lambda: len(model.asr.calls) == 1,
        )

    assert len(vad.calls) == 1
    assert len(model.asr.calls) == 1
