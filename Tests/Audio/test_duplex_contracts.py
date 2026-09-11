"""Dependency-free contracts for the speculative duplex audio boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import subprocess
import sys
from typing import Protocol, get_type_hints

import pytest

from tldw_chatbook.Audio.duplex_contracts import (
    AcousticDemotionReason,
    AcousticIsolationObservation,
    AcousticSafetyPath,
    AcousticSafetySnapshot,
    AecHealth,
    AudioFrame,
    DrainReceipt,
    DuplexMode,
    DuplexAudioPort,
    NearEndDisposition,
)


def test_aec_health_values_are_stable_strings() -> None:
    assert {health.name: health.value for health in AecHealth} == {
        "WARMING": "warming",
        "HEALTHY": "healthy",
        "DEGRADED": "degraded",
    }
    assert all(isinstance(health, str) for health in AecHealth)


@pytest.mark.parametrize("onset", [True, -1, 1.5, "20", 9])
def test_audio_and_admitted_frames_validate_optional_speech_onset(onset):
    from tldw_chatbook.Chat.console_speculative_voice import AdmittedSpeechFrame

    with pytest.raises((TypeError, ValueError)):
        AudioFrame(0, 10, 20, bytes(960), speech_started_ns=onset)
    with pytest.raises((TypeError, ValueError)):
        AdmittedSpeechFrame(0, 10, 20, 0, speech_started_ns=onset)


def test_preroll_speech_onset_can_follow_its_pcm_end():
    from tldw_chatbook.Chat.console_speculative_voice import AdmittedSpeechFrame

    assert (
        AudioFrame(0, 10, 20, bytes(960), speech_started_ns=30).speech_started_ns == 30
    )
    assert (
        AdmittedSpeechFrame(0, 10, 20, 0, speech_started_ns=30).speech_started_ns == 30
    )


def test_acoustic_safety_path_values_are_stable_strings() -> None:
    assert {path.name: path.value for path in AcousticSafetyPath} == {
        "WARMING": "warming",
        "AEC": "aec",
        "ACOUSTIC_ISOLATION": "acoustic-isolation",
        "HALF_DUPLEX": "half-duplex",
    }


def test_acoustic_demotion_reason_values_are_closed_stable_strings() -> None:
    assert {reason.name: reason.value for reason in AcousticDemotionReason} == {
        "AMBIGUOUS_CORRELATION": "ambiguous-correlation",
        "CAPTURE_SEQUENCE_GAP": "capture-sequence-gap",
        "CORRELATED_RENDER": "correlated-render",
        "INAUDIBLE_RENDER": "inaudible-render",
        "INVALID_CAPTURE_PCM": "invalid-capture-pcm",
        "INVALID_RENDER_PCM": "invalid-render-pcm",
        "MISSING_RENDER_REFERENCE": "missing-render-reference",
        "MISSING_TIMING": "missing-timing",
        "NATIVE_PROCESSOR_FAILURE": "native-processor-failure",
        "RENDER_REFERENCE_FAILURE": "render-reference-failure",
        "RENDER_REFERENCE_GAP": "render-reference-gap",
        "RENDER_REFERENCE_OVERFLOW": "render-reference-overflow",
        "RENDER_SEQUENCE_GAP": "render-sequence-gap",
        "ROUTE_MISMATCH": "route-mismatch",
        "SATURATION": "saturation",
        "TIMING_DISCONTINUITY": "timing-discontinuity",
        "UNBOUNDED_TIMING": "unbounded-timing",
    }
    assert all(isinstance(reason, str) for reason in AcousticDemotionReason)


def test_near_end_disposition_values_are_stable_strings() -> None:
    assert {item.name: item.value for item in NearEndDisposition} == {
        "PENDING": "pending",
        "ADMIT": "admit",
        "FENCE": "fence",
    }


def test_isolation_snapshot_is_the_only_non_aec_full_duplex_path() -> None:
    snapshot = AcousticSafetySnapshot(
        path=AcousticSafetyPath.ACOUSTIC_ISOLATION,
        mode=DuplexMode.FULL_DUPLEX,
        route_generation=4,
        admission_open=True,
        demotion_reason=None,
    )

    assert snapshot.mode is DuplexMode.FULL_DUPLEX


@pytest.mark.parametrize(
    "path",
    [AcousticSafetyPath.WARMING, AcousticSafetyPath.HALF_DUPLEX],
)
def test_closed_paths_cannot_claim_full_duplex(path: AcousticSafetyPath) -> None:
    with pytest.raises(ValueError):
        AcousticSafetySnapshot(path, DuplexMode.FULL_DUPLEX, 0, True, None)


@pytest.mark.parametrize(
    "path",
    [AcousticSafetyPath.AEC, AcousticSafetyPath.ACOUSTIC_ISOLATION],
)
def test_full_duplex_paths_require_open_admission_and_no_demotion(
    path: AcousticSafetyPath,
) -> None:
    with pytest.raises(ValueError):
        AcousticSafetySnapshot(path, DuplexMode.FULL_DUPLEX, 0, False, None)
    with pytest.raises(ValueError):
        AcousticSafetySnapshot(
            path,
            DuplexMode.FULL_DUPLEX,
            0,
            True,
            AcousticDemotionReason.CORRELATED_RENDER,
        )


@pytest.mark.parametrize(
    "path",
    [AcousticSafetyPath.AEC, AcousticSafetyPath.ACOUSTIC_ISOLATION],
)
def test_full_duplex_paths_cannot_claim_half_duplex(
    path: AcousticSafetyPath,
) -> None:
    with pytest.raises(ValueError):
        AcousticSafetySnapshot(path, DuplexMode.HALF_DUPLEX, 0, False, None)


def test_closed_snapshot_rejects_open_admission_and_negative_generation() -> None:
    with pytest.raises(ValueError):
        AcousticSafetySnapshot(
            AcousticSafetyPath.HALF_DUPLEX,
            DuplexMode.HALF_DUPLEX,
            0,
            True,
            AcousticDemotionReason.CORRELATED_RENDER,
        )
    with pytest.raises(ValueError):
        AcousticSafetySnapshot(
            AcousticSafetyPath.WARMING,
            DuplexMode.HALF_DUPLEX,
            -1,
            False,
            None,
        )


def test_acoustic_safety_snapshot_is_frozen_and_slotted() -> None:
    snapshot = AcousticSafetySnapshot(
        AcousticSafetyPath.WARMING,
        DuplexMode.HALF_DUPLEX,
        0,
        False,
        None,
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.admission_open = True  # type: ignore[misc]
    assert not hasattr(snapshot, "__dict__")


@pytest.mark.parametrize(
    ("path", "mode"),
    [("warming", DuplexMode.HALF_DUPLEX), (AcousticSafetyPath.WARMING, "half-duplex")],
)
def test_acoustic_safety_snapshot_requires_enum_instances(
    path: object,
    mode: object,
) -> None:
    with pytest.raises(TypeError):
        AcousticSafetySnapshot(path, mode, 0, False, None)  # type: ignore[arg-type]


def test_acoustic_safety_snapshot_requires_closed_demotion_reason() -> None:
    with pytest.raises(TypeError):
        AcousticSafetySnapshot(
            AcousticSafetyPath.HALF_DUPLEX,
            DuplexMode.HALF_DUPLEX,
            0,
            False,
            "correlated-render",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("route_generation", [True, 1.5])
def test_acoustic_safety_snapshot_requires_exact_integer_route_generation(
    route_generation: object,
) -> None:
    with pytest.raises(TypeError):
        AcousticSafetySnapshot(
            AcousticSafetyPath.WARMING,
            DuplexMode.HALF_DUPLEX,
            route_generation,  # type: ignore[arg-type]
            False,
            None,
        )


def test_acoustic_isolation_observation_is_frozen_slotted_and_typed() -> None:
    safety = AcousticSafetySnapshot(
        AcousticSafetyPath.WARMING,
        DuplexMode.HALF_DUPLEX,
        0,
        False,
        None,
    )
    observation = AcousticIsolationObservation(safety, NearEndDisposition.PENDING)

    with pytest.raises(FrozenInstanceError):
        observation.near_end_disposition = None  # type: ignore[misc]
    assert not hasattr(observation, "__dict__")
    with pytest.raises(TypeError):
        AcousticIsolationObservation("warming", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AcousticIsolationObservation(safety, "pending")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "path",
    [AcousticSafetyPath.AEC, AcousticSafetyPath.ACOUSTIC_ISOLATION],
)
def test_fence_disposition_rejects_every_open_snapshot(
    path: AcousticSafetyPath,
) -> None:
    safety = AcousticSafetySnapshot(
        path,
        DuplexMode.FULL_DUPLEX,
        0,
        True,
        None,
    )

    with pytest.raises(ValueError):
        AcousticIsolationObservation(safety, NearEndDisposition.FENCE)


def test_audio_frame_reports_positive_duration() -> None:
    frame = AudioFrame(
        sequence=1,
        started_ns=10,
        ended_ns=20,
        pcm16=b"\x00\x00",
    )

    assert frame.duration_ns == 10


def test_audio_frame_is_frozen_and_slotted() -> None:
    frame = AudioFrame(sequence=1, started_ns=10, ended_ns=20, pcm16=b"")

    with pytest.raises(FrozenInstanceError):
        frame.sequence = 2  # type: ignore[misc]
    assert not hasattr(frame, "__dict__")


def test_audio_frame_rejects_negative_sequence() -> None:
    with pytest.raises(ValueError, match="sequence"):
        AudioFrame(sequence=-1, started_ns=10, ended_ns=20, pcm16=b"")


@pytest.mark.parametrize("started_ns,ended_ns", [(10, 10), (20, 10)])
def test_audio_frame_rejects_non_positive_duration(
    started_ns: int,
    ended_ns: int,
) -> None:
    with pytest.raises(ValueError, match="duration"):
        AudioFrame(
            sequence=0,
            started_ns=started_ns,
            ended_ns=ended_ns,
            pcm16=b"",
        )


def test_drain_receipt_exposes_each_pipeline_watermark() -> None:
    receipt = DrainReceipt(
        capture_watermark_ns=100,
        capture_sequence=7,
        dsp_sequence=6,
        vad_sequence=5,
    )

    assert (
        receipt.capture_watermark_ns,
        receipt.capture_sequence,
        receipt.dsp_sequence,
        receipt.vad_sequence,
    ) == (100, 7, 6, 5)


def test_drain_receipt_is_frozen_and_slotted() -> None:
    receipt = DrainReceipt(
        capture_watermark_ns=100,
        capture_sequence=7,
        dsp_sequence=6,
        vad_sequence=5,
    )

    with pytest.raises(FrozenInstanceError):
        receipt.capture_sequence = 8  # type: ignore[misc]
    assert not hasattr(receipt, "__dict__")


def test_duplex_audio_port_is_a_plain_protocol() -> None:
    assert DuplexAudioPort.__bases__ == (Protocol,)
    assert getattr(DuplexAudioPort, "_is_runtime_protocol", False) is False


def test_duplex_audio_port_methods_are_async() -> None:
    assert inspect.iscoroutinefunction(DuplexAudioPort.abort_output)
    assert inspect.iscoroutinefunction(DuplexAudioPort.drain_capture_through)


def test_duplex_audio_port_abort_signature_is_stable() -> None:
    signature = inspect.signature(DuplexAudioPort.abort_output)

    assert list(signature.parameters) == ["self"]
    assert get_type_hints(DuplexAudioPort.abort_output)["return"] is type(None)


def test_duplex_audio_port_drain_signature_is_stable() -> None:
    signature = inspect.signature(DuplexAudioPort.drain_capture_through)
    type_hints = get_type_hints(DuplexAudioPort.drain_capture_through)

    assert list(signature.parameters) == ["self", "render_boundary_ns"]
    assert type_hints["render_boundary_ns"] is int
    assert type_hints["return"] is DrainReceipt


def test_contract_modules_import_without_native_audio_dependencies() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import tldw_chatbook.Audio.duplex_contracts; "
                "import tldw_chatbook.Chat.console_voice_settings; "
                "assert 'sounddevice' not in sys.modules; "
                "assert 'pyaudio' not in sys.modules; "
                "assert 'tldw_voice_aec' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr
