from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_chatbook.STT.contracts import (
    DeviceFailureOrigin,
    ExecutionDevice,
    TranscriptionFailureCode,
)
from tldw_chatbook.STT.executor import (
    ExecutorEvent,
    ExecutorFailure,
    ExecutorRequest,
    ExecutorResult,
    LocalSourceSnapshot,
    ModelIdentity,
    WorkerPhase,
    _AttemptTerminalGuard,
)


def _identity(**overrides: object) -> ModelIdentity:
    values: dict[str, object] = {
        "provider_id": "parakeet-onnx",
        "model_id": "nemo-parakeet-tdt-0.6b-v2",
        "root_revision": "revision-a",
        "closure_fingerprint": "fingerprint-a",
        "precision": "int8",
        "device": ExecutionDevice.CPU,
        "local_snapshot_token": "private-snapshot-token",
    }
    values.update(overrides)
    return ModelIdentity(**values)


def _request() -> ExecutorRequest:
    return ExecutorRequest(
        generation=3,
        attempt_id="attempt-1",
        job_id="job-1",
        source_path=Path("/private/media/interview.wav"),
        identity=_identity(),
        options={"transcription_model_dir": "/private/models/parakeet"},
        local_source=LocalSourceSnapshot(
            token="private-snapshot-token",
            paths=(Path("/private/models/parakeet/encoder-model.int8.onnx"),),
            identities=((7, 11, 1024, 123456),),
        ),
        managed_store_root=Path("/private/models/managed"),
    )


def test_protocol_objects_are_frozen_slotted_and_picklable() -> None:
    request = _request()
    envelopes = (
        request,
        ExecutorEvent(3, "attempt-1", WorkerPhase.LOADING),
        ExecutorResult(3, "attempt-1", {"content": "hello"}),
        ExecutorFailure(
            generation=3,
            attempt_id="attempt-1",
            code=TranscriptionFailureCode.ENGINE_CRASHED,
            recovery_actions=("retry_faster_whisper",),
            failed_attempt={"attempt_id": "attempt-1"},
            device_failure_origin=DeviceFailureOrigin.ENGINE_CRASH,
        ),
    )

    assert all(pickle.loads(pickle.dumps(value)) == value for value in envelopes)
    assert all(hasattr(type(value), "__slots__") for value in envelopes)
    with pytest.raises(FrozenInstanceError):
        request.generation = 4  # type: ignore[misc]


def test_model_identity_equality_includes_every_residency_component() -> None:
    baseline = _identity()

    changed = (
        _identity(provider_id="transcribe-cpp"),
        _identity(model_id="local-gguf:whisper"),
        _identity(root_revision="revision-b"),
        _identity(closure_fingerprint="fingerprint-b"),
        _identity(precision="f32"),
        _identity(device=ExecutionDevice.METAL),
        _identity(local_snapshot_token="replacement-snapshot"),
    )

    assert all(candidate != baseline for candidate in changed)


def test_protocol_repr_redacts_private_paths_options_and_snapshot_tokens() -> None:
    request = _request()
    failure = ExecutorFailure(
        generation=3,
        attempt_id="attempt-1",
        code=TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
        recovery_actions=("choose_another_gguf", "retry_faster_whisper"),
        failed_attempt={"private": "/private/models/secret.gguf"},
    )

    rendered = repr(request) + repr(request.identity) + repr(request.local_source)
    assert "/private/" not in rendered
    assert "private-snapshot-token" not in rendered
    assert "/private/" not in repr(failure)


def test_worker_phase_is_restricted_to_worker_owned_transitions() -> None:
    assert {phase.value for phase in WorkerPhase} == {
        "preparing",
        "loading",
        "transcribing",
        "post-processing",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"generation": 0},
        {"attempt_id": ""},
    ],
)
def test_protocol_rejects_empty_or_invalid_required_identity(
    kwargs: dict[str, object],
) -> None:
    values = {
        "generation": 3,
        "attempt_id": "attempt-1",
        "job_id": "job-1",
        "source_path": Path("media.wav"),
        "identity": _identity(),
        "options": {},
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        ExecutorRequest(**values)


def test_model_identity_rejects_empty_provider() -> None:
    with pytest.raises(ValueError, match="provider_id"):
        _identity(provider_id="")


def test_failure_requires_stable_typed_code_and_bounded_actions() -> None:
    with pytest.raises(TypeError):
        ExecutorFailure(  # type: ignore[arg-type]
            3,
            "attempt-1",
            "engine_crashed",
        )
    with pytest.raises(ValueError):
        ExecutorFailure(
            3,
            "attempt-1",
            TranscriptionFailureCode.ENGINE_CRASHED,
            recovery_actions=tuple(f"action-{index}" for index in range(9)),
        )


def test_terminal_guard_accepts_exactly_one_matching_terminal_envelope() -> None:
    guard = _AttemptTerminalGuard(generation=3, attempt_id="attempt-1")
    matching = ExecutorResult(3, "attempt-1", {"content": "hello"})

    assert guard.accept(matching) is True
    assert guard.accept(matching) is False
    assert guard.accept(ExecutorResult(2, "attempt-1", {})) is False
    assert guard.accept(ExecutorResult(3, "attempt-2", {})) is False


def test_terminal_guard_does_not_consume_slot_for_stale_envelope() -> None:
    guard = _AttemptTerminalGuard(generation=3, attempt_id="attempt-1")

    assert guard.accept(ExecutorResult(2, "attempt-1", {})) is False
    assert guard.accept(ExecutorResult(3, "attempt-1", {})) is True
