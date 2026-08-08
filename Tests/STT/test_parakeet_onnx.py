"""Focused tests for the offline executor-native Parakeet ONNX runtime."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
    PARAKEET_V2_MODEL,
    PARAKEET_V3_MODEL,
)
from tldw_chatbook.Model_Artifacts.leases import ArtifactLeaseKey
from tldw_chatbook.STT.contracts import (
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
    def __init__(self, *, short_text: str = "short text", segments=()) -> None:
        self.short_text = short_text
        self.short_calls: list[tuple[object, dict[str, object]]] = []
        self.asr = _FakeAsr(list(segments))
        self.resampler = lambda waveforms, lengths, sample_rate: (waveforms, lengths)

    def recognize(self, audio_path, **kwargs):
        self.short_calls.append((audio_path, kwargs))
        return self.short_text


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
