from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import get_args, get_type_hints

import pytest

from tldw_chatbook.Model_Artifacts.leases import ArtifactLeaseKey
from tldw_chatbook.STT import (
    MAX_BUFFER_AUDIO_BYTES,
    TRANSCRIPTION_FAILURE_CONTRACT,
    BufferAudioSource,
    CancellationGranularity,
    CancellationToken,
    DeviceFailureOrigin,
    DeviceRetryPolicy,
    ExecutionDevice,
    FileAudioSource,
    InputKind,
    LanguageInputMode,
    PipelineCapabilities,
    PrivacyRequirements,
    ProducedCapabilities,
    ProgressSink,
    TimestampGranularity,
    TranscriptionAction,
    TranscriptionFailure,
    TranscriptionFailureCode,
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
        "artifact_dependencies": (ArtifactLeaseKey("silero-vad", "rev-1", "fp32"),),
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
    expected: dict[type[Enum], tuple[str, ...]] = {
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
        DeviceFailureOrigin: (
            "execution_provider_initialization",
            "inference",
            "engine_crash",
        ),
        ExecutionDevice: ("auto", "cpu", "cuda", "metal"),
        TranscriptionFailureCode: (
            "model_not_installed",
            "artifact_corrupt",
            "artifact_incompatible",
            "provider_unavailable",
            "provider_removed",
            "unsupported_language",
            "unsupported_capability",
            "insufficient_disk_space",
            "insufficient_memory",
            "inference_failed",
            "engine_crashed",
            "cancelled",
        ),
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


def test_file_audio_source_repr_redacts_local_path() -> None:
    path = Path("/private/customer-secret-path.wav")
    source = FileAudioSource(path)

    assert source.path == path
    assert source == FileAudioSource(path)
    assert "customer-secret-path" not in repr(source)


def test_buffer_audio_source_accepts_bounded_pcm_metadata() -> None:
    source = BufferAudioSource(
        audio=b"\x00\x01\x02\x03",
        sample_rate=16_000,
        channels=2,
        sample_width=2,
    )

    assert source.audio == b"\x00\x01\x02\x03"
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
    ("audio", "channels", "sample_width"),
    [
        (b"\x00", 1, 2),
        (b"\x00\x00", 3, 1),
    ],
)
def test_buffer_audio_source_rejects_incomplete_interleaved_pcm_frames(
    audio: bytes,
    channels: int,
    sample_width: int,
) -> None:
    with pytest.raises(ValueError):
        BufferAudioSource(
            audio=audio,
            sample_rate=16_000,
            channels=channels,
            sample_width=sample_width,
        )


def test_buffer_audio_source_payload_is_excluded_from_representations() -> None:
    marker = "PCM-SECRET-PAYLOAD"
    source = BufferAudioSource(marker.encode(), 16_000)
    request = TranscriptionRequest(attempt_id="attempt-1", source=source)

    assert marker not in repr(source)
    assert marker not in repr(request)


def test_request_defaults_to_no_timestamps_for_default_parakeet_route() -> None:
    request = TranscriptionRequest(
        attempt_id="attempt-default",
        source=BufferAudioSource(b"\x00\x00", 16_000),
    )

    assert request.timestamps is TimestampGranularity.NONE


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
    values: dict[str, object] = {
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


def test_request_rejects_noncallable_cancellation_member() -> None:
    class InvalidToken:
        is_cancelled = 0

    with pytest.raises(TypeError):
        TranscriptionRequest(
            attempt_id="attempt-1",
            source=BufferAudioSource(b"\x00\x00", 16_000),
            cancellation=InvalidToken(),  # type: ignore[arg-type]
        )


def test_request_rejects_noncallable_progress_member() -> None:
    class InvalidSink:
        __call__ = 0

    with pytest.raises(TypeError):
        TranscriptionRequest(
            attempt_id="attempt-1",
            source=BufferAudioSource(b"\x00\x00", 16_000),
            progress=InvalidSink(),  # type: ignore[arg-type]
        )


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
        source=BufferAudioSource(b"\x00\x00", 16_000),
        cancellation=_Token("TOKEN-SECRET"),
        progress=_Sink("CALLBACK-SECRET"),
    )
    second = TranscriptionRequest(
        attempt_id="attempt-1",
        source=BufferAudioSource(b"\x00\x00", 16_000),
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
        source=BufferAudioSource(b"\x00\x00", 16_000),
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
            source=BufferAudioSource(b"\x00\x00", 16_000),
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
        "source": BufferAudioSource(b"\x00\x00", 16_000),
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
        "source": BufferAudioSource(b"\x00\x00", 16_000),
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


def test_transcription_segment_repr_redacts_text_and_speaker() -> None:
    segment = TranscriptionSegment(
        0.0,
        1.0,
        "customer-secret-transcript",
        speaker="customer-secret-speaker",
    )

    assert segment.text == "customer-secret-transcript"
    assert segment.speaker == "customer-secret-speaker"
    assert segment == TranscriptionSegment(
        0.0,
        1.0,
        "customer-secret-transcript",
        speaker="customer-secret-speaker",
    )
    assert "customer-secret-transcript" not in repr(segment)
    assert "customer-secret-speaker" not in repr(segment)


def test_segment_accepts_finite_nonnegative_arbitrary_precision_integers() -> None:
    timestamp = 10**1000

    segment = TranscriptionSegment(timestamp, timestamp, "far future")

    assert segment.start_seconds == timestamp
    assert segment.end_seconds == timestamp


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


def test_public_artifact_identity_annotations_resolve_at_runtime() -> None:
    failure_hints = get_type_hints(TranscriptionFailure)
    provenance_hints = get_type_hints(TranscriptionProvenance)

    artifact_identity_types = set(get_args(failure_hints["artifact_root"])) - {
        type(None)
    }
    assert len(artifact_identity_types) == 1
    artifact_identity = artifact_identity_types.pop()
    assert getattr(artifact_identity, "_is_protocol", False)
    assert get_args(provenance_hints["artifact_root"]) == (
        artifact_identity,
        type(None),
    )
    assert get_args(provenance_hints["artifact_dependencies"]) == (
        artifact_identity,
        Ellipsis,
    )


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
        timestamps=frozenset({TimestampGranularity.SEGMENT, TimestampGranularity.WORD}),
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
        PipelineCapabilities(**{field_name: 1})  # type: ignore[arg-type]


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


def test_timings_accept_finite_nonnegative_arbitrary_precision_integer() -> None:
    duration = 10**1000

    timings = TranscriptionTimings(total_seconds=duration)

    assert timings.total_seconds == duration


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


def test_result_rejects_segments_without_produced_timestamps() -> None:
    with pytest.raises(ValueError):
        _result(produced_capabilities=_produced(timestamps=TimestampGranularity.NONE))


@pytest.mark.parametrize(
    "timestamps",
    [TimestampGranularity.SEGMENT, TimestampGranularity.WORD],
)
def test_result_accepts_silence_with_produced_timestamp_capability(
    timestamps: TimestampGranularity,
) -> None:
    result = _result(
        text="",
        segments=(),
        produced_capabilities=_produced(timestamps=timestamps),
    )

    assert result.text == ""
    assert result.segments == ()
    assert result.produced_capabilities.timestamps is timestamps


@pytest.mark.parametrize(
    "timestamps",
    [TimestampGranularity.SEGMENT, TimestampGranularity.WORD],
)
def test_result_rejects_nonempty_timestamped_transcript_without_segments(
    timestamps: TimestampGranularity,
) -> None:
    with pytest.raises(ValueError):
        _result(
            text="hello",
            segments=(),
            produced_capabilities=_produced(timestamps=timestamps),
        )


def test_result_rejects_speaker_when_diarization_was_not_produced() -> None:
    with pytest.raises(ValueError):
        _result(
            segments=(TranscriptionSegment(0.0, 1.0, "hello", speaker="A"),),
            produced_capabilities=_produced(diarization=False),
        )


def test_transcription_result_repr_redacts_text_and_segments() -> None:
    segment = TranscriptionSegment(
        0.0,
        1.0,
        "customer-secret-segment",
        speaker="customer-secret-speaker",
    )
    result = _result(
        text="customer-secret-result",
        segments=(segment,),
        produced_capabilities=_produced(diarization=True),
    )

    assert result.text == "customer-secret-result"
    assert result.segments == (segment,)
    rendered = repr(result)
    assert "customer-secret-result" not in rendered
    assert "customer-secret-segment" not in rendered
    assert "customer-secret-speaker" not in rendered


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


_FAILURE_CASES = (
    (
        TranscriptionFailureCode.MODEL_NOT_INSTALLED,
        "The selected speech-to-text model is not installed.",
        False,
    ),
    (
        TranscriptionFailureCode.ARTIFACT_CORRUPT,
        "The installed speech-to-text model failed integrity verification.",
        False,
    ),
    (
        TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
        "The installed speech-to-text model is incompatible with this runtime.",
        False,
    ),
    (
        TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
        "The selected speech-to-text provider is unavailable.",
        True,
    ),
    (
        TranscriptionFailureCode.PROVIDER_REMOVED,
        "The selected speech-to-text provider is no longer supported.",
        False,
    ),
    (
        TranscriptionFailureCode.UNSUPPORTED_LANGUAGE,
        "The selected speech-to-text model does not support the requested language.",
        False,
    ),
    (
        TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
        "The selected speech-to-text model does not support the requested capability.",
        False,
    ),
    (
        TranscriptionFailureCode.INSUFFICIENT_DISK_SPACE,
        "There is not enough disk space to prepare this transcription.",
        False,
    ),
    (
        TranscriptionFailureCode.INSUFFICIENT_MEMORY,
        "There is not enough memory to run this transcription.",
        False,
    ),
    (
        TranscriptionFailureCode.INFERENCE_FAILED,
        "Speech-to-text inference failed.",
        False,
    ),
    (
        TranscriptionFailureCode.ENGINE_CRASHED,
        "The speech-to-text engine stopped unexpectedly.",
        True,
    ),
    (
        TranscriptionFailureCode.CANCELLED,
        "The transcription was cancelled.",
        True,
    ),
)


def test_transcription_action_values_are_exact_and_closed() -> None:
    assert tuple(action.value for action in TranscriptionAction) == (
        "install_model",
        "choose_installed_model",
        "retry_same_configuration",
        "retry_with_faster_whisper",
        "change_language_to_auto",
    )
    assert all(
        TranscriptionAction(action.value) is action for action in TranscriptionAction
    )
    with pytest.raises(ValueError):
        TranscriptionAction("adapter-supplied-action")


def _failure(**overrides: object) -> TranscriptionFailure:
    values: dict[str, object] = {
        "code": TranscriptionFailureCode.INFERENCE_FAILED,
        "attempt_id": "attempt-TOKEN-SECRET",
        "batch_id": "batch-TOKEN-SECRET",
        "job_id": "job-TOKEN-SECRET",
        "phase": TranscriptionPhase.TRANSCRIBING,
        "provider_id": "provider-TOKEN-SECRET",
        "model_id": "model-TOKEN-SECRET",
        "artifact_root": ArtifactLeaseKey(
            "artifact-TOKEN-SECRET",
            "revision-TOKEN-SECRET",
            "variant-TOKEN-SECRET",
        ),
        "precision": "precision-TOKEN-SECRET",
        "requested_device": ExecutionDevice.AUTO,
        "effective_device": ExecutionDevice.CUDA,
    }
    values.update(overrides)
    return TranscriptionFailure(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(("code", "message", "retryable"), _FAILURE_CASES)
def test_failure_contract_is_fixed_typed_and_sanitized(
    code: TranscriptionFailureCode,
    message: str,
    retryable: bool,
) -> None:
    failure = _failure(code=code)

    assert failure.code is code
    assert type(failure.phase) is TranscriptionPhase
    assert type(failure.requested_device) is ExecutionDevice
    assert type(failure.effective_device) is ExecutionDevice
    assert type(failure.artifact_root) is ArtifactLeaseKey
    assert failure.attempt_id == "attempt-TOKEN-SECRET"
    assert failure.batch_id == "batch-TOKEN-SECRET"
    assert failure.job_id == "job-TOKEN-SECRET"
    assert failure.provider_id == "provider-TOKEN-SECRET"
    assert failure.model_id == "model-TOKEN-SECRET"
    assert failure.precision == "precision-TOKEN-SECRET"
    assert failure.message == message
    assert failure.retryable is retryable
    assert str(failure) == message
    assert repr(failure) == (
        f"TranscriptionFailure(code={code.value!r}, phase='transcribing')"
    )
    assert "TOKEN-SECRET" not in repr(failure)
    assert not hasattr(failure, "__dict__")
    with pytest.raises(FrozenInstanceError):
        failure.provider_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        _failure(code=code, exception_text="raw TOKEN-SECRET traceback")
    with pytest.raises(TypeError):
        _failure(code=code, message="raw TOKEN-SECRET traceback")


def test_failure_contract_mapping_is_complete_and_immutable() -> None:
    assert isinstance(TRANSCRIPTION_FAILURE_CONTRACT, MappingProxyType)
    assert tuple(TRANSCRIPTION_FAILURE_CONTRACT) == tuple(TranscriptionFailureCode)
    assert tuple(TRANSCRIPTION_FAILURE_CONTRACT.values()) == tuple(
        (message, retryable) for _, message, retryable in _FAILURE_CASES
    )
    with pytest.raises(TypeError):
        TRANSCRIPTION_FAILURE_CONTRACT[  # type: ignore[index]
            TranscriptionFailureCode.CANCELLED
        ] = (
            "unsafe",
            False,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("code", "inference_failed"),
        ("attempt_id", ""),
        ("batch_id", " "),
        ("job_id", ""),
        ("phase", "transcribing"),
        ("provider_id", ""),
        ("model_id", " "),
        ("artifact_root", object()),
        ("precision", ""),
        ("requested_device", "auto"),
        ("effective_device", "cuda"),
    ],
)
def test_failure_rejects_untyped_or_empty_provenance_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _failure(**{field_name: value})


def test_failure_allows_absent_artifact_and_effective_device() -> None:
    failure = _failure(artifact_root=None, effective_device=None)

    assert failure.artifact_root is None
    assert failure.effective_device is None


def test_progress_repr_and_str_exclude_caller_controlled_detail_and_identity() -> None:
    progress = TranscriptionProgress(
        attempt_id="attempt-TOKEN-SECRET",
        batch_id="batch-TOKEN-SECRET",
        job_id="job-TOKEN-SECRET",
        phase=TranscriptionPhase.TRANSCRIBING,
        fraction=0.25,
        detail_code="customer-secret-token",
    )

    assert progress.detail_code == "customer-secret-token"
    assert (
        repr(progress) == "TranscriptionProgress(phase='transcribing', fraction=0.25)"
    )
    assert str(progress) == "transcribing: 25%"
    assert "TOKEN-SECRET" not in repr(progress)
    assert "TOKEN-SECRET" not in str(progress)
    assert "customer-secret-token" not in repr(progress)
    assert "customer-secret-token" not in str(progress)


@pytest.mark.parametrize(
    ("requested_device", "failed_device"),
    [
        (ExecutionDevice.CUDA, ExecutionDevice.CUDA),
        (ExecutionDevice.METAL, ExecutionDevice.METAL),
        (ExecutionDevice.AUTO, ExecutionDevice.CUDA),
        (ExecutionDevice.AUTO, ExecutionDevice.METAL),
    ],
)
def test_device_retry_policy_allows_one_recycled_same_provider_cpu_retry(
    requested_device: ExecutionDevice,
    failed_device: ExecutionDevice,
) -> None:
    policy = DeviceRetryPolicy.for_failure(
        requested_device=requested_device,
        failed_device=failed_device,
        origin=DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
        retry_device=ExecutionDevice.CPU,
        worker_will_recycle=True,
    )

    assert policy.retry_device is ExecutionDevice.CPU
    assert policy.max_retries == 1
    assert policy.requires_worker_recycling
    assert policy.same_provider_model_only


@pytest.mark.parametrize(
    (
        "requested_device",
        "failed_device",
        "origin",
        "retry_device",
        "worker_will_recycle",
    ),
    [
        (
            ExecutionDevice.CPU,
            ExecutionDevice.CPU,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.AUTO,
            ExecutionDevice.CPU,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.AUTO,
            ExecutionDevice.AUTO,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.CUDA,
            ExecutionDevice.AUTO,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.CUDA,
            ExecutionDevice.METAL,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.CUDA,
            ExecutionDevice.CUDA,
            DeviceFailureOrigin.INFERENCE,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.METAL,
            ExecutionDevice.METAL,
            DeviceFailureOrigin.ENGINE_CRASH,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.CUDA,
            ExecutionDevice.CUDA,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CUDA,
            True,
        ),
        (
            ExecutionDevice.METAL,
            ExecutionDevice.METAL,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CPU,
            False,
        ),
    ],
)
def test_device_retry_policy_fails_closed_for_other_failures(
    requested_device: ExecutionDevice,
    failed_device: ExecutionDevice,
    origin: DeviceFailureOrigin,
    retry_device: ExecutionDevice,
    worker_will_recycle: bool,
) -> None:
    policy = DeviceRetryPolicy.for_failure(
        requested_device=requested_device,
        failed_device=failed_device,
        origin=origin,
        retry_device=retry_device,
        worker_will_recycle=worker_will_recycle,
    )

    assert policy == DeviceRetryPolicy.no_retry()
    assert policy.retry_device is None
    assert policy.max_retries == 0
    assert not policy.requires_worker_recycling
    assert not policy.same_provider_model_only


@pytest.mark.parametrize(
    "values",
    [
        {
            "retry_device": ExecutionDevice.CUDA,
            "max_retries": 1,
            "requires_worker_recycling": True,
            "same_provider_model_only": True,
        },
        {
            "retry_device": ExecutionDevice.CPU,
            "max_retries": 2,
            "requires_worker_recycling": True,
            "same_provider_model_only": True,
        },
        {
            "retry_device": ExecutionDevice.CPU,
            "max_retries": 1,
            "requires_worker_recycling": False,
            "same_provider_model_only": True,
        },
        {
            "retry_device": ExecutionDevice.CPU,
            "max_retries": 1,
            "requires_worker_recycling": True,
            "same_provider_model_only": False,
        },
    ],
)
def test_device_retry_policy_cannot_represent_unsafe_retry(
    values: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        DeviceRetryPolicy(**values)  # type: ignore[arg-type]
