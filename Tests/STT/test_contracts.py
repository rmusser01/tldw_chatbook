from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_chatbook.Model_Artifacts.leases import ArtifactLeaseKey
from tldw_chatbook.STT import (
    MAX_BUFFER_AUDIO_BYTES,
    BufferAudioSource,
    CancellationGranularity,
    CancellationToken,
    ExecutionDevice,
    FileAudioSource,
    InputKind,
    LanguageInputMode,
    PipelineCapabilities,
    PrivacyRequirements,
    ProducedCapabilities,
    ProgressSink,
    TimestampGranularity,
    TranscriptionPhase,
    TranscriptionProgress,
    TranscriptionProvenance,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTask,
    TranscriptionTimings,
    TranscriptionWarningCode,
)


@pytest.fixture(autouse=True)
def isolate_test_environment() -> None:
    """Keep dependency-free contract tests independent of application config."""


def _provenance(**overrides: object) -> TranscriptionProvenance:
    values: dict[str, object] = {
        "schema_version": 1,
        "attempt_id": "attempt-1",
        "batch_id": "batch-1",
        "job_id": "job-1",
        "retry_of_attempt_id": None,
        "retry_of_job_id": None,
        "provider_id": "parakeet-onnx",
        "model_id": "parakeet-v2",
        "artifact_root": ArtifactLeaseKey("parakeet", "rev-2", "int8"),
        "artifact_dependencies": (
            ArtifactLeaseKey("silero-vad", "rev-1", "fp32"),
        ),
        "precision": "int8",
        "requested_device": ExecutionDevice.AUTO,
        "effective_device": ExecutionDevice.CPU,
        "requested_language": "en",
        "effective_language": "en",
        "detected_language": None,
        "task": TranscriptionTask.TRANSCRIBE,
    }
    values.update(overrides)
    return TranscriptionProvenance(**values)  # type: ignore[arg-type]


def _produced(**overrides: object) -> ProducedCapabilities:
    values: dict[str, object] = {
        "timestamps": TimestampGranularity.SEGMENT,
        "punctuation": True,
        "capitalization": True,
        "vad": False,
        "diarization": False,
    }
    values.update(overrides)
    return ProducedCapabilities(**values)  # type: ignore[arg-type]


def _result(**overrides: object) -> TranscriptionResult:
    values: dict[str, object] = {
        "text": "hello",
        "segments": (TranscriptionSegment(0.0, 1.0, "hello"),),
        "provenance": _provenance(),
        "produced_capabilities": _produced(),
        "duration_seconds": 1.0,
        "timings": TranscriptionTimings(total_seconds=0.5),
        "warnings": (),
    }
    values.update(overrides)
    return TranscriptionResult(**values)  # type: ignore[arg-type]


def test_enums_have_exact_stable_string_values() -> None:
    expected = {
        TranscriptionTask: ("transcribe", "translate"),
        InputKind: ("file", "buffer"),
        TimestampGranularity: ("none", "segment", "word"),
        CancellationGranularity: (
            "none",
            "before_execution",
            "segment_boundary",
            "active",
        ),
        TranscriptionPhase: (
            "queued",
            "preparing",
            "loading",
            "transcribing",
            "post-processing",
            "saving",
            "complete",
        ),
        LanguageInputMode: (
            "enforced",
            "routing_assertion",
            "automatic",
            "automatic_only",
        ),
        ExecutionDevice: ("auto", "cpu", "cuda", "metal"),
        TranscriptionWarningCode: ("requested_language_not_enforced",),
    }

    for enum_type, values in expected.items():
        assert tuple(member.value for member in enum_type) == values
        assert all(enum_type(member.value) is member for member in enum_type)
        with pytest.raises(ValueError):
            enum_type(values[0].upper())


def test_file_audio_source_requires_a_path_and_is_frozen_and_slotted() -> None:
    source = FileAudioSource(Path("/tmp/example.wav"))

    assert source.path == Path("/tmp/example.wav")
    assert not hasattr(source, "__dict__")
    with pytest.raises(FrozenInstanceError):
        source.path = Path("/tmp/other.wav")  # type: ignore[misc]
    with pytest.raises(TypeError):
        FileAudioSource("/tmp/example.wav")  # type: ignore[arg-type]


def test_buffer_audio_source_accepts_bounded_pcm_metadata() -> None:
    source = BufferAudioSource(
        audio=b"\x00\x01",
        sample_rate=16_000,
        channels=2,
        sample_width=2,
    )

    assert source.audio == b"\x00\x01"
    assert MAX_BUFFER_AUDIO_BYTES > 0
    assert not hasattr(source, "__dict__")
    with pytest.raises(FrozenInstanceError):
        source.audio = b"changed"  # type: ignore[misc]


@pytest.mark.parametrize("audio", [b"", bytearray(b"x"), memoryview(b"x")])
def test_buffer_audio_source_rejects_invalid_audio_objects(audio: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BufferAudioSource(audio=audio, sample_rate=16_000)  # type: ignore[arg-type]


def test_buffer_audio_source_rejects_oversized_audio() -> None:
    with pytest.raises(ValueError):
        BufferAudioSource(
            audio=b"x" * (MAX_BUFFER_AUDIO_BYTES + 1),
            sample_rate=16_000,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sample_rate", True),
        ("sample_rate", 0),
        ("sample_rate", 16_000.0),
        ("channels", False),
        ("channels", 0),
        ("channels", 1.0),
        ("sample_width", True),
        ("sample_width", 0),
        ("sample_width", 5),
        ("sample_width", 2.0),
    ],
)
def test_buffer_audio_source_rejects_invalid_integer_fields(
    field_name: str,
    value: object,
) -> None:
    values = {
        "audio": b"x",
        "sample_rate": 16_000,
        "channels": 1,
        "sample_width": 2,
    }
    values[field_name] = value

    with pytest.raises((TypeError, ValueError)):
        BufferAudioSource(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("allow_remote_processing", 0),
        ("allow_remote_processing", None),
        ("allow_disk_staging", 1),
        ("allow_disk_staging", "yes"),
    ],
)
def test_privacy_requirements_reject_non_boolean_flags(
    field_name: str,
    value: object,
) -> None:
    values = {
        "allow_remote_processing": False,
        "allow_disk_staging": True,
    }
    values[field_name] = value

    with pytest.raises(TypeError):
        PrivacyRequirements(**values)  # type: ignore[arg-type]


def test_protocols_accept_structural_token_and_progress_sink() -> None:
    class Token:
        def is_cancelled(self) -> bool:
            return False

    class Sink:
        def __call__(self, event: TranscriptionProgress) -> None:
            del event

    assert isinstance(Token(), CancellationToken)
    assert isinstance(Sink(), ProgressSink)


@pytest.mark.parametrize("identifier", ["", " ", "\t"])
def test_progress_rejects_empty_attempt_id(identifier: str) -> None:
    with pytest.raises(ValueError):
        TranscriptionProgress(
            attempt_id=identifier,
            batch_id=None,
            job_id=None,
            phase=TranscriptionPhase.QUEUED,
        )


@pytest.mark.parametrize("field_name", ["batch_id", "job_id"])
@pytest.mark.parametrize("identifier", ["", " "])
def test_progress_rejects_empty_optional_ids(
    field_name: str,
    identifier: str,
) -> None:
    values = {
        "attempt_id": "attempt-1",
        "batch_id": None,
        "job_id": None,
        "phase": TranscriptionPhase.QUEUED,
    }
    values[field_name] = identifier

    with pytest.raises(ValueError):
        TranscriptionProgress(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("fraction", [-0.1, 1.1, math.inf, -math.inf, math.nan, True])
def test_progress_rejects_invalid_fraction(fraction: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TranscriptionProgress(
            attempt_id="attempt-1",
            batch_id=None,
            job_id=None,
            phase=TranscriptionPhase.TRANSCRIBING,
            fraction=fraction,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "detail_code",
    ["", "Has Spaces", "UPPER_CASE", "../unsafe", "x" * 129, 1],
)
def test_progress_rejects_unsafe_detail_codes(detail_code: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TranscriptionProgress(
            attempt_id="attempt-1",
            batch_id=None,
            job_id=None,
            phase=TranscriptionPhase.TRANSCRIBING,
            detail_code=detail_code,  # type: ignore[arg-type]
        )


def test_progress_accepts_stable_detail_code_and_exact_phase() -> None:
    progress = TranscriptionProgress(
        attempt_id="attempt-1",
        batch_id=None,
        job_id=None,
        phase=TranscriptionPhase.TRANSCRIBING,
        fraction=0.25,
        detail_code="decode.segment_1-ready",
    )

    assert progress.fraction == 0.25
    with pytest.raises(TypeError):
        TranscriptionProgress(
            attempt_id="attempt-1",
            batch_id=None,
            job_id=None,
            phase="transcribing",  # type: ignore[arg-type]
        )


class _Token:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def is_cancelled(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"_Token({self.secret})"


class _Sink:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def __call__(self, event: TranscriptionProgress) -> None:
        del event

    def __repr__(self) -> str:
        return f"_Sink({self.secret})"


def test_request_excludes_callback_and_token_from_repr_and_comparison() -> None:
    first = TranscriptionRequest(
        attempt_id="attempt-1",
        source=BufferAudioSource(b"x", 16_000),
        cancellation=_Token("TOKEN-SECRET"),
        progress=_Sink("CALLBACK-SECRET"),
    )
    second = TranscriptionRequest(
        attempt_id="attempt-1",
        source=BufferAudioSource(b"x", 16_000),
        cancellation=_Token("OTHER-TOKEN"),
        progress=_Sink("OTHER-CALLBACK"),
    )

    assert first == second
    rendered = repr(first)
    assert "TOKEN-SECRET" not in rendered
    assert "CALLBACK-SECRET" not in rendered


@pytest.mark.parametrize(
    "language",
    [None, "", "auto", "en", "zh-cn", "pt-br", "es-419"],
)
def test_request_accepts_canonical_languages(language: str | None) -> None:
    request = TranscriptionRequest(
        attempt_id="attempt-1",
        source=BufferAudioSource(b"x", 16_000),
        language=language,
    )

    assert request.language == language


@pytest.mark.parametrize(
    "language",
    [" ", "EN", "en-US", "en_US", "-en", "en-", "auto-detect", 1],
)
def test_request_rejects_noncanonical_languages(language: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TranscriptionRequest(
            attempt_id="attempt-1",
            source=BufferAudioSource(b"x", 16_000),
            language=language,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("attempt_id", ""),
        ("batch_id", " "),
        ("job_id", ""),
        ("retry_of_attempt_id", " "),
        ("retry_of_job_id", ""),
        ("provider_id", ""),
        ("model_id", " "),
        ("precision", ""),
    ],
)
def test_request_rejects_empty_identity_fields(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "attempt_id": "attempt-1",
        "source": BufferAudioSource(b"x", 16_000),
    }
    values[field_name] = value

    with pytest.raises(ValueError):
        TranscriptionRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source", b"audio"),
        ("task", "transcribe"),
        ("device", "cpu"),
        ("timestamps", "segment"),
        ("diarization", 0),
        ("vad", 1),
        ("privacy", None),
        ("cancellation", object()),
        ("progress", object()),
    ],
)
def test_request_rejects_wrong_contract_types(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "attempt_id": "attempt-1",
        "source": BufferAudioSource(b"x", 16_000),
    }
    values[field_name] = value

    with pytest.raises(TypeError):
        TranscriptionRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("start_seconds", -0.1),
        ("start_seconds", math.inf),
        ("start_seconds", math.nan),
        ("start_seconds", True),
        ("end_seconds", -0.1),
        ("end_seconds", -math.inf),
        ("text", None),
        ("speaker", 1),
    ],
)
def test_segment_rejects_invalid_values(field_name: str, value: object) -> None:
    values: dict[str, object] = {
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "text": "hello",
        "speaker": None,
    }
    values[field_name] = value

    with pytest.raises((TypeError, ValueError)):
        TranscriptionSegment(**values)  # type: ignore[arg-type]


def test_segment_rejects_end_before_start() -> None:
    with pytest.raises(ValueError):
        TranscriptionSegment(1.0, 0.9, "out of order")


def test_provenance_preserves_complete_artifact_identity_in_tuples() -> None:
    provenance = _provenance()

    assert provenance.artifact_root == ArtifactLeaseKey(
        "parakeet",
        "rev-2",
        "int8",
    )
    assert provenance.artifact_dependencies == (
        ArtifactLeaseKey("silero-vad", "rev-1", "fp32"),
    )
    assert not hasattr(provenance, "__dict__")
    with pytest.raises(FrozenInstanceError):
        provenance.provider_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        _provenance(artifact_dependencies=[])


def test_provenance_rejects_an_artifact_key_lookalike() -> None:
    class ArtifactLeaseKeyLookalike:
        artifact_id = "parakeet"
        revision = "rev-2"
        variant = "int8"

    with pytest.raises(TypeError):
        _provenance(artifact_root=ArtifactLeaseKeyLookalike())


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("schema_version", True),
        ("schema_version", 0),
        ("attempt_id", ""),
        ("batch_id", " "),
        ("job_id", ""),
        ("retry_of_attempt_id", " "),
        ("retry_of_job_id", ""),
        ("provider_id", ""),
        ("model_id", " "),
        ("precision", ""),
        ("artifact_root", object()),
        ("artifact_dependencies", (object(),)),
        ("requested_device", "auto"),
        ("effective_device", "cpu"),
        ("requested_language", "EN"),
        ("effective_language", ""),
        ("detected_language", "pt-BR"),
        ("task", "transcribe"),
    ],
)
def test_provenance_rejects_invalid_contract_values(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _provenance(**{field_name: value})


@pytest.mark.parametrize(
    "field_name",
    ["punctuation", "capitalization", "vad", "diarization"],
)
def test_produced_capabilities_require_strict_booleans(field_name: str) -> None:
    with pytest.raises(TypeError):
        _produced(**{field_name: 1})


def test_produced_capabilities_require_exact_timestamp_enum() -> None:
    with pytest.raises(TypeError):
        _produced(timestamps="segment")


def test_pipeline_capabilities_are_frozen_and_use_an_immutable_set() -> None:
    capabilities = PipelineCapabilities(
        timestamps=frozenset(
            {TimestampGranularity.SEGMENT, TimestampGranularity.WORD}
        ),
        vad=True,
        diarization=False,
        requires_disk_staging_for_buffer=True,
    )

    assert capabilities.timestamps == frozenset(
        {TimestampGranularity.SEGMENT, TimestampGranularity.WORD}
    )
    assert not hasattr(capabilities, "__dict__")
    with pytest.raises(TypeError):
        PipelineCapabilities(timestamps={TimestampGranularity.SEGMENT})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PipelineCapabilities(timestamps=frozenset({"segment"}))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    ["vad", "diarization", "requires_disk_staging_for_buffer"],
)
def test_pipeline_capabilities_require_strict_booleans(field_name: str) -> None:
    with pytest.raises(TypeError):
        PipelineCapabilities(**{field_name: 1})


@pytest.mark.parametrize(
    "field_name",
    [
        "preparation_seconds",
        "model_load_seconds",
        "inference_seconds",
        "postprocess_seconds",
        "total_seconds",
    ],
)
@pytest.mark.parametrize("value", [-0.1, math.inf, -math.inf, math.nan, True, "1"])
def test_timings_reject_invalid_values(field_name: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TranscriptionTimings(**{field_name: value})  # type: ignore[arg-type]


def test_result_requires_ordered_immutable_segments_and_warning_tuple() -> None:
    first = TranscriptionSegment(0.0, 2.0, "first")
    overlapping = TranscriptionSegment(1.0, 3.0, "overlapping")
    result = _result(
        segments=(first, overlapping),
        warnings=(TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,),
    )

    assert result.segments == (first, overlapping)
    assert result.warnings == (
        TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,
    )
    assert not hasattr(result, "__dict__")
    with pytest.raises(TypeError):
        _result(segments=[first])
    with pytest.raises(ValueError):
        _result(segments=(overlapping, first))
    with pytest.raises(TypeError):
        _result(warnings=["requested_language_not_enforced"])


def test_result_rejects_timestamp_and_segment_contradictions() -> None:
    with pytest.raises(ValueError):
        _result(produced_capabilities=_produced(timestamps=TimestampGranularity.NONE))
    with pytest.raises(ValueError):
        _result(
            text="",
            segments=(),
            produced_capabilities=_produced(
                timestamps=TimestampGranularity.SEGMENT
            ),
        )


def test_result_rejects_speaker_when_diarization_was_not_produced() -> None:
    with pytest.raises(ValueError):
        _result(
            segments=(TranscriptionSegment(0.0, 1.0, "hello", speaker="A"),),
            produced_capabilities=_produced(diarization=False),
        )


@pytest.mark.parametrize("duration", [-0.1, math.inf, -math.inf, math.nan, True, "1"])
def test_result_rejects_invalid_duration(duration: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _result(duration_seconds=duration)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("text", None),
        ("provenance", object()),
        ("produced_capabilities", object()),
        ("timings", object()),
        ("warnings", (object(),)),
    ],
)
def test_result_rejects_wrong_contract_types(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(TypeError):
        _result(**{field_name: value})
