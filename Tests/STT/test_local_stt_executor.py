from __future__ import annotations

import pickle
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_chatbook.STT.contracts import (
    DeviceFailureOrigin,
    ExecutionDevice,
    TranscriptionFailureCode,
)
from tldw_chatbook.STT.executor import (
    ExecutorBusyError,
    ExecutorEvent,
    ExecutorFailure,
    ExecutorRequest,
    ExecutorResult,
    ExecutorUnavailableError,
    LocalSTTExecutor,
    LocalSourceSnapshot,
    ModelIdentity,
    WorkerPhase,
    _AttemptTerminalGuard,
)
from Tests.STT.executor_test_support import fake_executor_worker


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


class _Callbacks:
    def __init__(self) -> None:
        self.events: list[ExecutorEvent] = []
        self.results: list[ExecutorResult] = []
        self.failures: list[ExecutorFailure] = []
        self.terminal = threading.Event()

    def on_event(self, event: ExecutorEvent) -> None:
        self.events.append(event)

    def on_result(self, result: ExecutorResult) -> None:
        self.results.append(result)
        self.terminal.set()

    def on_failure(self, failure: ExecutorFailure) -> None:
        self.failures.append(failure)
        self.terminal.set()


def _executor(*, completed_job_limit: int = 20) -> LocalSTTExecutor:
    return LocalSTTExecutor(
        worker_target=fake_executor_worker,
        completed_job_limit=completed_job_limit,
        startup_timeout=5.0,
        graceful_shutdown_timeout=0.2,
        force_stop_timeout=2.0,
    )


def _submit(
    executor: LocalSTTExecutor,
    callbacks: _Callbacks,
    *,
    attempt_id: str,
    identity: ModelIdentity | None = None,
    mode: str = "succeed",
    explicit_retry: bool = False,
) -> int:
    return executor.submit(
        attempt_id=attempt_id,
        job_id=f"job-{attempt_id}",
        source_path=Path("fixture.wav"),
        identity=identity or _identity(root_revision=None, closure_fingerprint=None),
        options={"test_mode": mode},
        on_event=callbacks.on_event,
        on_result=callbacks.on_result,
        on_failure=callbacks.on_failure,
        explicit_retry=explicit_retry,
    )


def _wait_for_terminal(callbacks: _Callbacks) -> None:
    assert callbacks.terminal.wait(10.0)


def _wait_until(predicate: object, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:  # type: ignore[operator]
        time.sleep(0.01)
    assert predicate()  # type: ignore[operator]


def test_controller_starts_lazily_and_reuses_same_worker_for_same_identity() -> None:
    executor = _executor()
    first = _Callbacks()
    second = _Callbacks()
    try:
        assert executor.generation == 0
        first_generation = _submit(executor, first, attempt_id="one")
        _wait_for_terminal(first)
        second_generation = _submit(executor, second, attempt_id="two")
        _wait_for_terminal(second)

        assert first_generation == second_generation
        assert (
            first.results[0].payload["worker_pid"]
            == second.results[0].payload["worker_pid"]
        )
        assert executor.resident_identity == _identity(
            root_revision=None,
            closure_fingerprint=None,
        )
    finally:
        executor.close()


def test_controller_recycles_idle_worker_when_identity_changes() -> None:
    executor = _executor()
    first = _Callbacks()
    second = _Callbacks()
    try:
        first_generation = _submit(executor, first, attempt_id="one")
        _wait_for_terminal(first)
        changed = _identity(
            model_id="local-gguf:whisper",
            root_revision=None,
            closure_fingerprint=None,
        )
        second_generation = _submit(
            executor,
            second,
            attempt_id="two",
            identity=changed,
        )
        _wait_for_terminal(second)

        assert second_generation > first_generation
        assert (
            first.results[0].payload["worker_pid"]
            != second.results[0].payload["worker_pid"]
        )
    finally:
        executor.close()


def test_controller_recycles_after_completed_job_bound() -> None:
    executor = _executor(completed_job_limit=1)
    first = _Callbacks()
    second = _Callbacks()
    try:
        first_generation = _submit(executor, first, attempt_id="one")
        _wait_for_terminal(first)
        second_generation = _submit(executor, second, attempt_id="two")
        _wait_for_terminal(second)

        assert second_generation > first_generation
    finally:
        executor.close()


def test_controller_has_one_active_request_and_cooperative_cancel_is_attempt_scoped() -> (
    None
):
    executor = _executor()
    held = _Callbacks()
    successor = _Callbacks()
    try:
        _submit(executor, held, attempt_id="held", mode="hold")
        _wait_until(
            lambda: any(
                event.phase is WorkerPhase.TRANSCRIBING for event in held.events
            )
        )
        with pytest.raises(ExecutorBusyError):
            _submit(executor, successor, attempt_id="blocked")
        assert executor.cancel("wrong-attempt") is False
        assert executor.cancel("held") is True
        _wait_for_terminal(held)
        assert held.failures[0].code is TranscriptionFailureCode.CANCELLED

        _submit(executor, successor, attempt_id="successor")
        _wait_for_terminal(successor)
        assert successor.results
    finally:
        executor.close()


@pytest.mark.parametrize("mode", ["stale_then_succeed", "duplicate"])
def test_controller_drops_stale_and_duplicate_terminals(mode: str) -> None:
    executor = _executor()
    callbacks = _Callbacks()
    try:
        _submit(executor, callbacks, attempt_id="one", mode=mode)
        _wait_for_terminal(callbacks)
        time.sleep(0.05)

        assert len(callbacks.results) == 1
        assert callbacks.results[0].payload["content"] == "transcript"
        assert callbacks.failures == []
    finally:
        executor.close()


def test_force_stop_detaches_before_kill_and_cleans_generation_scratch() -> None:
    executor = _executor()
    callbacks = _Callbacks()
    try:
        _submit(executor, callbacks, attempt_id="held", mode="ignore_cancel")
        _wait_until(
            lambda: any(
                event.phase is WorkerPhase.TRANSCRIBING for event in callbacks.events
            )
        )
        scratch = executor._scratch_path
        assert scratch is not None and scratch.is_dir()

        assert executor.force_stop("held") is True
        _wait_for_terminal(callbacks)
        assert callbacks.failures[0].code is TranscriptionFailureCode.CANCELLED
        _wait_until(lambda: executor.retiring is False)
        assert scratch.exists() is False
        assert executor.busy is False
    finally:
        executor.close()


def test_failed_force_stop_quarantines_executor_and_prevents_second_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor()
    callbacks = _Callbacks()
    try:
        _submit(executor, callbacks, attempt_id="held", mode="ignore_cancel")
        _wait_until(
            lambda: any(
                event.phase is WorkerPhase.TRANSCRIBING for event in callbacks.events
            )
        )
        tree = executor._tree
        assert tree is not None
        original = tree.terminate_tree
        monkeypatch.setattr(tree, "terminate_tree", lambda **_kwargs: False)

        assert executor.force_stop("held") is True
        _wait_until(lambda: executor.retiring is False)
        assert executor.unavailable is True
        with pytest.raises(ExecutorUnavailableError):
            _submit(executor, _Callbacks(), attempt_id="blocked")

        monkeypatch.setattr(tree, "terminate_tree", original)
        assert original(term_timeout=1.0, kill_timeout=1.0) is True
    finally:
        executor.close()


def test_loading_crash_marks_only_active_identity_unhealthy() -> None:
    executor = _executor()
    callbacks = _Callbacks()
    identity = _identity(root_revision=None, closure_fingerprint=None)
    try:
        _submit(
            executor,
            callbacks,
            attempt_id="crash",
            identity=identity,
            mode="crash_loading",
        )
        _wait_for_terminal(callbacks)

        assert callbacks.failures[0].code is TranscriptionFailureCode.ENGINE_CRASHED
        assert executor.unhealthy_identity == identity
        with pytest.raises(ExecutorUnavailableError):
            _submit(executor, _Callbacks(), attempt_id="blocked", identity=identity)
        assert executor.clear_unhealthy_identity(identity) is True
    finally:
        executor.close()


def test_typed_device_failure_retries_once_on_cpu_in_fresh_generation() -> None:
    executor = _executor()
    callbacks = _Callbacks()
    accelerated = _identity(
        root_revision=None,
        closure_fingerprint=None,
        device=ExecutionDevice.METAL,
    )
    try:
        first_generation = _submit(
            executor,
            callbacks,
            attempt_id="retry",
            identity=accelerated,
            mode="device_failure",
        )
        _wait_for_terminal(callbacks)

        assert callbacks.failures == []
        assert callbacks.results[0].generation > first_generation
        assert callbacks.results[0].attempt_id == "retry"
        assert callbacks.results[0].payload["device"] == "cpu"
    finally:
        executor.close()


def test_close_is_idempotent_and_removes_idle_generation_scratch() -> None:
    executor = _executor()
    callbacks = _Callbacks()
    _submit(executor, callbacks, attempt_id="one")
    _wait_for_terminal(callbacks)
    scratch = executor._scratch_path
    assert scratch is not None and scratch.exists()

    executor.close()
    executor.close()

    assert scratch.exists() is False
    assert executor.busy is False
