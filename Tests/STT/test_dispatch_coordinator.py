"""Behavior tests for the bounded dictation-next dispatch coordinator."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.STT.contracts import (
    BufferAudioSource,
    ExecutionDevice,
    FileAudioSource,
    TranscriptionFailureCode,
)
from tldw_chatbook.STT.executor import (
    ExecutorBusyError,
    ExecutorEvent,
    ExecutorFailure,
    ExecutorResult,
    ExecutorUnavailableError,
    ModelIdentity,
    WorkerPhase,
)
from tldw_chatbook.STT.parakeet_dispatch import ParakeetDispatch


@pytest.fixture
def coordinator_module() -> Any:
    """Import inside the test so a missing implementation is an intentional RED."""

    return importlib.import_module("tldw_chatbook.STT.dispatch_coordinator")


class FakeExecutor:
    """Queue-less executor fake with explicit, deterministic terminal controls."""

    def __init__(self, order: list[str] | None = None) -> None:
        self._condition = threading.Condition()
        self._generation = 0
        self.active: dict[str, Any] | None = None
        self.submissions: list[dict[str, Any]] = []
        self.cancelled_attempts: list[str] = []
        self.force_stopped_attempts: list[str] = []
        self.closed = False
        self.order = order if order is not None else []
        self.before_submit: Callable[[str], None] | None = None

    def submit(
        self,
        *,
        attempt_id: str,
        job_id: str | None,
        source: FileAudioSource | BufferAudioSource,
        identity: ModelIdentity,
        options: dict[str, Any],
        segment_end_frames: tuple[int, ...] = (),
        local_source: Any = None,
        managed_store_root: Path | None = None,
        managed_artifact_ref: tuple[str, str, str] | None = None,
        managed_dependency_refs: tuple[tuple[str, str, str], ...] = (),
        on_event: Callable[[Any], None] = lambda _event: None,
        on_result: Callable[[ExecutorResult], None] = lambda _result: None,
        on_failure: Callable[[ExecutorFailure], None] = lambda _failure: None,
        explicit_retry: bool = False,
    ) -> int:
        kind = "library" if job_id is not None else "dictation"
        if self.before_submit is not None:
            self.before_submit(kind)
        with self._condition:
            if self.active is not None:
                raise ExecutorBusyError("fake executor already has active work")
            self._generation += 1
            submission = {
                "generation": self._generation,
                "attempt_id": attempt_id,
                "job_id": job_id,
                "source": source,
                "identity": identity,
                "options": dict(options),
                "segment_end_frames": segment_end_frames,
                "local_source": local_source,
                "managed_store_root": managed_store_root,
                "managed_artifact_ref": managed_artifact_ref,
                "managed_dependency_refs": managed_dependency_refs,
                "on_event": on_event,
                "on_result": on_result,
                "on_failure": on_failure,
                "explicit_retry": explicit_retry,
                "submit_thread": threading.current_thread().name,
            }
            self.active = submission
            self.submissions.append(submission)
            self.order.append(f"submit-{kind}")
            self._condition.notify_all()
            return self._generation

    def wait_for_submissions(self, count: int, timeout: float = 1.0) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: len(self.submissions) >= count,
                timeout,
            )

    def succeed(
        self,
        payload: dict[str, Any] | None = None,
        *,
        thread_name: str | None = None,
    ) -> None:
        submission = self._take_active()
        result = ExecutorResult(
            submission["generation"],
            submission["attempt_id"],
            payload
            or {
                "text": "ok",
                "logical_segments": ("ok",),
                "duration": 0.1,
                "transcription_model": submission["identity"].model_id,
                "transcription_provenance": {},
            },
        )
        self._deliver(submission["on_result"], result, thread_name)

    def fail(
        self,
        code: TranscriptionFailureCode,
        *,
        recovery_actions: tuple[str, ...] = (),
        failed_attempt: dict[str, Any] | None = None,
        thread_name: str | None = None,
    ) -> None:
        submission = self._take_active()
        failure = ExecutorFailure(
            submission["generation"],
            submission["attempt_id"],
            code,
            recovery_actions=recovery_actions,
            failed_attempt=failed_attempt,
        )
        self._deliver(submission["on_failure"], failure, thread_name)

    def cancel(self, attempt_id: str) -> bool:
        with self._condition:
            if self.active is None or self.active["attempt_id"] != attempt_id:
                return False
            self.cancelled_attempts.append(attempt_id)
            return True

    def force_stop(self, attempt_id: str) -> bool:
        with self._condition:
            if self.active is None or self.active["attempt_id"] != attempt_id:
                return False
            self.force_stopped_attempts.append(attempt_id)
        self.fail(TranscriptionFailureCode.CANCELLED)
        return True

    def close(self) -> None:
        self.closed = True

    def deliver_duplicate_result(self, submission_index: int = -1) -> None:
        submission = self.submissions[submission_index]
        submission["on_result"](
            ExecutorResult(
                submission["generation"],
                submission["attempt_id"],
                {"text": "duplicate", "logical_segments": ("duplicate",)},
            )
        )

    def deliver_stale_result(self, submission_index: int = -1) -> None:
        submission = self.submissions[submission_index]
        submission["on_result"](
            ExecutorResult(
                submission["generation"] + 100,
                submission["attempt_id"],
                {"text": "stale", "logical_segments": ("stale",)},
            )
        )

    def emit_event(
        self,
        generation: int,
        *,
        attempt_id: str | None = None,
        phase: WorkerPhase = WorkerPhase.PREPARING,
    ) -> None:
        with self._condition:
            assert self.active is not None
            submission = self.active
        submission["on_event"](
            ExecutorEvent(
                generation,
                attempt_id or submission["attempt_id"],
                phase,
            )
        )

    def _take_active(self) -> dict[str, Any]:
        with self._condition:
            assert self.active is not None
            submission = self.active
            self.active = None
            return submission

    @staticmethod
    def _deliver(
        callback: Callable[[Any], None],
        envelope: Any,
        thread_name: str | None,
    ) -> None:
        if thread_name is None:
            callback(envelope)
            return
        thread = threading.Thread(
            target=callback,
            args=(envelope,),
            name=thread_name,
        )
        thread.start()
        thread.join(1.0)
        assert not thread.is_alive()


def _identity(model_id: str = "parakeet-tdt-0.6b-v2") -> ModelIdentity:
    return ModelIdentity(
        provider_id="parakeet-onnx",
        model_id=model_id,
        root_revision=f"revision-{model_id}",
        closure_fingerprint=f"fingerprint-{model_id}",
        precision="int8",
        device=ExecutionDevice.CPU,
    )


def _dispatch(model_id: str = "parakeet-tdt-0.6b-v2") -> ParakeetDispatch:
    return ParakeetDispatch(
        identity=_identity(model_id),
        local_source=None,
        managed_store_root=None,
        managed_artifact_ref=None,
        option_updates={"provider_option": "copied"},
    )


def test_dictation_forwards_exact_managed_dependency_refs(
    coordinator_module: Any,
) -> None:
    dependency_refs = (("silero-vad", "vad-revision", "f32"),)
    dispatch = ParakeetDispatch(
        identity=_identity(),
        local_source=None,
        managed_store_root=Path("managed-store"),
        managed_artifact_ref=None,
        managed_dependency_refs=dependency_refs,
        option_updates={},
    )
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)

    handle = _begin(coordinator_module, coordinator, dispatch=dispatch)
    handle.append_segment(b"\x01\x00")

    assert executor.submissions[0]["managed_dependency_refs"] == dependency_refs


def _library_kwargs(
    *,
    attempt_id: str = "library-attempt",
    model_id: str = "parakeet-tdt-0.6b-v3",
    on_event: Callable[[ExecutorEvent], None] = lambda _event: None,
    on_result: Callable[[ExecutorResult], None] = lambda _result: None,
    on_failure: Callable[[ExecutorFailure], None] = lambda _failure: None,
) -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "job_id": f"job-{attempt_id}",
        "source": FileAudioSource(Path(f"/tmp/{attempt_id}.wav")),
        "identity": _identity(model_id),
        "options": {"language": "en"},
        "on_event": on_event,
        "on_result": on_result,
        "on_failure": on_failure,
    }


def _begin(
    module: Any,
    coordinator: Any,
    *,
    capture_generation: int = 7,
    dispatch: ParakeetDispatch | None = None,
    sample_rate: int = 8_000,
    channels: int = 1,
    sample_width: int = 2,
    callback: Callable[[int, str], None] = lambda _sequence, _text: None,
) -> Any:
    return coordinator.begin_dictation(
        capture_generation=capture_generation,
        dispatch=dispatch or _dispatch(),
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        language="en",
        on_logical_segment=callback,
    )


def _wait_for(event: threading.Event, timeout: float = 1.0) -> None:
    assert event.wait(timeout), "timed out waiting for event-controlled transition"


def test_pcm_byte_limit_is_derived_from_the_single_sixty_second_ceiling(
    coordinator_module: Any,
) -> None:
    assert coordinator_module.DICTATION_MAX_SECONDS == 60.0
    assert coordinator_module.pcm_byte_limit(
        sample_rate=16_000,
        channels=2,
        sample_width=2,
    ) == 3_840_000


def test_idle_dictation_dispatches_immediately_and_delivers_ordered_text(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    delivered: list[tuple[int, str]] = []
    handle = _begin(
        coordinator_module,
        coordinator,
        callback=lambda sequence, text: delivered.append((sequence, text)),
    )

    assert handle.waiting_for_executor is False
    assert handle.append_segment(b"\x01\x00\x02\x00") is coordinator_module.DictationAppendStatus.ACCEPTED
    assert executor.wait_for_submissions(1)
    assert executor.submissions[0]["source"].audio == b"\x01\x00\x02\x00"
    assert executor.submissions[0]["segment_end_frames"] == (2,)
    assert executor.submissions[0]["options"] == {
        "provider_option": "copied",
        "language": "en",
    }

    handle.finish()
    executor.succeed({"text": "hello", "logical_segments": ("hello",)})
    handle.wait()

    assert delivered == [(0, "hello")]
    assert coordinator.dictation_reserved is False


def test_batch_terminal_submits_pending_dictation_before_original_callback(
    coordinator_module: Any,
) -> None:
    order: list[str] = []
    executor = FakeExecutor(order)
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    coordinator.submit_library(
        **_library_kwargs(on_result=lambda _result: order.append("library-result"))
    )
    handle = _begin(coordinator_module, coordinator)
    assert handle.waiting_for_executor is True
    handle.append_segment(b"\x01\x00")
    handle.finish()

    with pytest.raises(ExecutorBusyError):
        coordinator.submit_library(**_library_kwargs(attempt_id="later"))

    executor.succeed()
    assert executor.wait_for_submissions(2)

    assert order[:3] == ["submit-library", "submit-dictation", "library-result"]
    assert coordinator.dictation_reserved is True


def test_batch_failure_also_submits_dictation_before_original_callback(
    coordinator_module: Any,
) -> None:
    order: list[str] = []
    failure_delivered = threading.Event()
    executor = FakeExecutor(order)

    def on_failure(_failure: ExecutorFailure) -> None:
        order.append("library-failure")
        failure_delivered.set()

    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    coordinator.submit_library(**_library_kwargs(on_failure=on_failure))
    handle = _begin(coordinator_module, coordinator)
    handle.append_segment(b"\x01\x00")
    handle.finish()

    executor.fail(TranscriptionFailureCode.ENGINE_CRASHED)
    assert executor.wait_for_submissions(2)
    _wait_for(failure_delivered)

    assert order[:3] == ["submit-library", "submit-dictation", "library-failure"]


def test_synchronous_library_submit_failure_still_dispatches_reserved_dictation(
    coordinator_module: Any,
) -> None:
    submit_entered = threading.Event()
    release_submit = threading.Event()
    submit_finished = threading.Event()
    caught: list[Exception] = []
    executor = FakeExecutor()

    def fail_library_submit(kind: str) -> None:
        if kind != "library":
            return
        submit_entered.set()
        release_submit.wait()
        raise ExecutorUnavailableError("planned Library submit failure")

    executor.before_submit = fail_library_submit
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)

    def submit_library() -> None:
        try:
            coordinator.submit_library(**_library_kwargs())
        except Exception as error:
            caught.append(error)
        finally:
            submit_finished.set()

    library_thread = threading.Thread(target=submit_library, name="failing-library")
    library_thread.start()
    _wait_for(submit_entered)
    handle = _begin(coordinator_module, coordinator)
    handle.append_segment(b"\x01\x00")
    handle.finish()

    release_submit.set()
    _wait_for(submit_finished)
    library_thread.join(1.0)

    assert len(caught) == 1
    assert isinstance(caught[0], ExecutorUnavailableError)
    assert executor.wait_for_submissions(1)
    assert executor.submissions[0]["job_id"] is None
    executor.succeed({"text": "kept", "logical_segments": ("kept",)})
    handle.wait()
    assert coordinator.dictation_reserved is False


def test_dictation_terminal_delivers_callback_then_clears_gate_and_tops_up(
    coordinator_module: Any,
) -> None:
    order: list[str] = []
    idle = threading.Event()
    executor = FakeExecutor(order)

    def on_idle() -> None:
        order.append("top-up")
        idle.set()

    coordinator = coordinator_module.LocalSTTDispatchCoordinator(
        executor,
        on_dictation_idle=on_idle,
    )
    handle = _begin(
        coordinator_module,
        coordinator,
        callback=lambda sequence, text: order.append(f"segment-{sequence}-{text}"),
    )
    handle.append_segment(b"\x01\x00")
    handle.finish()
    executor.succeed({"text": "hello", "logical_segments": ("hello",)})
    handle.wait()
    _wait_for(idle)

    assert order == ["submit-dictation", "segment-0-hello", "top-up"]
    assert coordinator.dictation_reserved is False


def test_user_terminal_callback_runs_outside_the_coordinator_lock(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    callback_finished = threading.Event()
    blocked: list[bool] = []

    def on_result(_result: ExecutorResult) -> None:
        reader = threading.Thread(
            target=lambda: coordinator.dictation_reserved,
            name="callback-lock-probe",
        )
        reader.start()
        reader.join(0.05)
        blocked.append(reader.is_alive())
        callback_finished.set()

    coordinator.submit_library(**_library_kwargs(on_result=on_result))
    executor.succeed()

    _wait_for(callback_finished)
    assert blocked == [False]


def test_empty_mic_reservation_blocks_only_library_then_cancel_wakes_waiter(
    coordinator_module: Any,
) -> None:
    idle = threading.Event()
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(
        executor,
        on_dictation_idle=idle.set,
    )
    handle = _begin(coordinator_module, coordinator)

    assert coordinator.dictation_reserved is True
    with pytest.raises(ExecutorBusyError):
        coordinator.submit_library(**_library_kwargs())
    assert handle.cancel() is True
    with pytest.raises(RuntimeError) as caught:
        handle.wait()
    _wait_for(idle)

    assert caught.value.args == (TranscriptionFailureCode.CANCELLED,)
    assert coordinator.dictation_reserved is False
    assert executor.submissions == []


def test_waiting_segments_coalesce_into_one_source_with_ordered_boundaries(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    coordinator.submit_library(**_library_kwargs())
    handle = _begin(coordinator_module, coordinator)

    handle.append_segment(b"\x01\x00")
    handle.append_segment(b"\x02\x00\x03\x00")
    handle.append_segment(b"\x04\x00")
    handle.finish()
    assert len(executor.submissions) == 1

    executor.succeed()
    assert executor.wait_for_submissions(2)
    pending = executor.submissions[1]
    assert pending["source"].audio == b"\x01\x00\x02\x00\x03\x00\x04\x00"
    assert pending["segment_end_frames"] == (1, 3, 4)


def test_active_dictation_allows_exactly_one_coalesced_next_request(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    delivered: list[tuple[int, str]] = []
    handle = _begin(
        coordinator_module,
        coordinator,
        callback=lambda sequence, text: delivered.append((sequence, text)),
    )

    handle.append_segment(b"\x01\x00")
    handle.append_segment(b"\x02\x00")
    handle.append_segment(b"\x03\x00")
    handle.finish()
    assert len(executor.submissions) == 1

    executor.succeed({"text": "one", "logical_segments": ("one",)})
    assert executor.wait_for_submissions(2)
    assert len(executor.submissions) == 2
    assert executor.submissions[1]["source"].audio == b"\x02\x00\x03\x00"
    assert executor.submissions[1]["segment_end_frames"] == (1, 2)
    executor.succeed({"text": "two three", "logical_segments": ("two", "three")})
    handle.wait()

    assert delivered == [(0, "one"), (1, "two"), (2, "three")]


def test_rapid_next_terminal_cannot_overtake_prior_segment_callback(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    first_callback_entered = threading.Event()
    release_first_callback = threading.Event()
    wait_done = threading.Event()
    delivered: list[tuple[int, str, str]] = []

    def on_segment(sequence: int, text: str) -> None:
        if sequence == 0:
            first_callback_entered.set()
            release_first_callback.wait()
        delivered.append((sequence, text, threading.current_thread().name))

    handle = _begin(coordinator_module, coordinator, callback=on_segment)
    handle.append_segment(b"\x01\x00")
    handle.append_segment(b"\x02\x00")
    handle.finish()
    waiter = threading.Thread(
        target=lambda: (handle.wait(), wait_done.set()),
        name="ordered-callback-waiter",
    )
    waiter.start()

    executor.succeed({"text": "first", "logical_segments": ("first",)})
    assert executor.wait_for_submissions(2)
    _wait_for(first_callback_entered)
    executor.succeed({"text": "second", "logical_segments": ("second",)})

    try:
        assert delivered == []
        assert not wait_done.is_set()
    finally:
        release_first_callback.set()
    _wait_for(wait_done)
    waiter.join(1.0)

    assert [(sequence, text) for sequence, text, _thread in delivered] == [
        (0, "first"),
        (1, "second"),
    ]
    assert len({thread for _sequence, _text, thread in delivered}) == 1


@pytest.mark.parametrize(
    "changed",
    ["capture_generation", "identity", "sample_rate", "channels", "sample_width"],
)
def test_mismatched_capture_is_visibly_busy_and_never_combines_pcm(
    coordinator_module: Any,
    changed: str,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    first = _begin(coordinator_module, coordinator)
    first.append_segment(b"\x01\x00")
    kwargs: dict[str, Any] = {}
    if changed == "capture_generation":
        kwargs[changed] = 8
    elif changed == "identity":
        kwargs["dispatch"] = _dispatch("parakeet-tdt-0.6b-v3")
    elif changed == "sample_rate":
        kwargs[changed] = 16_000
    elif changed == "channels":
        kwargs[changed] = 2
    else:
        kwargs[changed] = 1

    with pytest.raises(ExecutorBusyError, match="dictation"):
        _begin(coordinator_module, coordinator, **kwargs)

    assert executor.submissions[0]["source"].audio == b"\x01\x00"


def test_exact_limit_is_frame_aligned_retained_and_reported_once(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    handle = _begin(
        coordinator_module,
        coordinator,
        sample_rate=2,
        channels=1,
        sample_width=2,
    )
    exact = b"\x01\x00" * 120

    assert handle.append_segment(exact) is coordinator_module.DictationAppendStatus.LIMIT_REACHED
    assert executor.wait_for_submissions(1)
    assert executor.submissions[0]["source"].audio == exact
    with pytest.raises(RuntimeError, match="finished"):
        handle.append_segment(b"\x02\x00")


def test_one_frame_over_limit_retains_every_accepted_complete_frame(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    handle = _begin(
        coordinator_module,
        coordinator,
        sample_rate=2,
        channels=1,
        sample_width=2,
    )
    over = b"\x01\x00" * 121

    assert handle.append_segment(over) is coordinator_module.DictationAppendStatus.LIMIT_REACHED
    assert executor.wait_for_submissions(1)
    assert executor.submissions[0]["source"].audio == over[:-2]
    assert executor.submissions[0]["segment_end_frames"] == (120,)


def test_append_rejects_partial_pcm_frames_without_retaining_them(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    handle = _begin(coordinator_module, coordinator, channels=2, sample_width=2)

    with pytest.raises(ValueError, match="complete interleaved PCM frames"):
        handle.append_segment(b"\x00\x00")
    assert executor.submissions == []


def test_stale_and_duplicate_terminals_are_ignored_once(
    coordinator_module: Any,
) -> None:
    delivered: list[str] = []
    delivered_event = threading.Event()
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    def on_result(result: ExecutorResult) -> None:
        delivered.append(result.payload["text"])
        delivered_event.set()

    coordinator.submit_library(**_library_kwargs(on_result=on_result))

    executor.deliver_stale_result()
    assert delivered == []
    executor.succeed({"text": "real"})
    _wait_for(delivered_event)
    executor.deliver_duplicate_result()

    assert delivered == ["real"]


def test_current_preparing_event_advances_library_generation_once_outside_lock(
    coordinator_module: Any,
) -> None:
    events: list[ExecutorEvent] = []
    results: list[str] = []
    result_delivered = threading.Event()
    callback_blocked: list[bool] = []
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)

    def on_event(event: ExecutorEvent) -> None:
        reader = threading.Thread(
            target=lambda: coordinator.dictation_reserved,
            name="event-callback-lock-probe",
        )
        reader.start()
        reader.join(0.05)
        callback_blocked.append(reader.is_alive())
        events.append(event)

    def on_result(result: ExecutorResult) -> None:
        results.append(result.payload["text"])
        result_delivered.set()

    assert coordinator.submit_library(
        **_library_kwargs(on_event=on_event, on_result=on_result)
    ) == 1
    submission = executor.submissions[0]

    executor.emit_event(3, attempt_id="different-attempt")
    executor.emit_event(2)
    submission["on_result"](
        ExecutorResult(1, submission["attempt_id"], {"text": "stale"})
    )
    assert results == []
    submission["on_result"](
        ExecutorResult(2, submission["attempt_id"], {"text": "cpu retry"})
    )
    _wait_for(result_delivered)
    executor.emit_event(2)

    assert [(event.generation, event.phase) for event in events] == [
        (2, WorkerPhase.PREPARING)
    ]
    assert callback_blocked == [False]
    assert results == ["cpu retry"]


def test_nonretryable_worker_failure_clears_capture_with_sanitized_category(
    coordinator_module: Any,
) -> None:
    idle = threading.Event()
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(
        executor,
        on_dictation_idle=idle.set,
    )
    handle = _begin(coordinator_module, coordinator)
    handle.append_segment(b"\x01\x00")
    handle.finish()
    executor.fail(
        TranscriptionFailureCode.INFERENCE_FAILED,
        failed_attempt={"private_path": "/Users/alice/secret/model.onnx"},
    )

    with pytest.raises(RuntimeError) as caught:
        handle.wait()
    _wait_for(idle)

    assert caught.value.args == (TranscriptionFailureCode.INFERENCE_FAILED,)
    assert "/Users/alice" not in str(caught.value)
    assert handle.take_retry_buffer() is None
    assert coordinator.dictation_reserved is False


def test_pending_cancel_clears_gate_once_without_preempting_batch(
    coordinator_module: Any,
) -> None:
    idle_count = 0
    idle = threading.Event()

    def on_idle() -> None:
        nonlocal idle_count
        idle_count += 1
        idle.set()

    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(
        executor,
        on_dictation_idle=on_idle,
    )
    coordinator.submit_library(**_library_kwargs())
    handle = _begin(coordinator_module, coordinator)
    handle.append_segment(b"\x01\x00")
    handle.finish()

    assert handle.cancel() is True
    with pytest.raises(RuntimeError) as caught:
        handle.wait()
    _wait_for(idle)
    assert handle.cancel() is False
    executor.succeed()

    assert executor.cancelled_attempts == []
    assert caught.value.args == (TranscriptionFailureCode.CANCELLED,)
    assert len(executor.submissions) == 1
    assert idle_count == 1


def test_active_cooperative_cancel_targets_attempt_and_waits_for_terminal(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    handle = _begin(coordinator_module, coordinator)
    handle.append_segment(b"\x01\x00")
    handle.finish()
    attempt_id = executor.active["attempt_id"]

    assert handle.cancel() is True
    assert executor.cancelled_attempts == [attempt_id]
    executor.fail(TranscriptionFailureCode.CANCELLED)
    with pytest.raises(RuntimeError) as caught:
        handle.wait()

    assert caught.value.args == (TranscriptionFailureCode.CANCELLED,)
    assert coordinator.dictation_reserved is False


def test_force_cancel_uses_executor_force_stop_and_resolves_once(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    handle = _begin(coordinator_module, coordinator)
    handle.append_segment(b"\x01\x00")
    handle.finish()
    attempt_id = executor.active["attempt_id"]

    assert handle.cancel(force=True) is True
    with pytest.raises(RuntimeError) as caught:
        handle.wait()

    assert caught.value.args == (TranscriptionFailureCode.CANCELLED,)
    assert executor.force_stopped_attempts == [attempt_id]
    assert coordinator.dictation_reserved is False


def test_force_cancel_before_submit_residency_is_replayed_as_force_stop(
    coordinator_module: Any,
) -> None:
    submit_entered = threading.Event()
    release_submit = threading.Event()
    append_finished = threading.Event()
    executor = FakeExecutor()

    def hold_dictation_submit(kind: str) -> None:
        if kind == "dictation":
            submit_entered.set()
            release_submit.wait()

    executor.before_submit = hold_dictation_submit
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    handle = _begin(coordinator_module, coordinator)
    appender = threading.Thread(
        target=lambda: (handle.append_segment(b"\x01\x00"), append_finished.set()),
        name="dictation-submit-race",
    )
    appender.start()
    _wait_for(submit_entered)

    try:
        assert handle.cancel(force=True) is True
    finally:
        release_submit.set()
    _wait_for(append_finished)
    appender.join(1.0)

    with pytest.raises(RuntimeError) as caught:
        handle.wait()
    assert caught.value.args == (TranscriptionFailureCode.CANCELLED,)
    assert len(executor.force_stopped_attempts) == 1
    assert executor.cancelled_attempts == []


def test_close_releases_pending_audio_without_closing_or_preempting_executor(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    coordinator.submit_library(**_library_kwargs())
    handle = _begin(coordinator_module, coordinator)
    handle.append_segment(b"\x01\x00")
    handle.finish()

    coordinator.close()
    coordinator.close()
    with pytest.raises(RuntimeError) as caught:
        handle.wait()
    executor.succeed()

    assert caught.value.args == (TranscriptionFailureCode.CANCELLED,)
    assert executor.closed is False
    assert executor.cancelled_attempts == []
    assert len(executor.submissions) == 1
    assert coordinator.dictation_reserved is False
    with pytest.raises(RuntimeError, match="closed"):
        _begin(coordinator_module, coordinator, capture_generation=9)


def test_close_cooperatively_cancels_active_dictation_without_owning_executor(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    handle = _begin(coordinator_module, coordinator)
    handle.append_segment(b"\x01\x00")
    handle.finish()
    attempt_id = executor.active["attempt_id"]

    coordinator.close()
    assert executor.cancelled_attempts == [attempt_id]
    assert executor.closed is False
    executor.fail(TranscriptionFailureCode.CANCELLED)
    with pytest.raises(RuntimeError) as caught:
        handle.wait()

    assert caught.value.args == (TranscriptionFailureCode.CANCELLED,)
    assert coordinator.dictation_reserved is False


def test_retry_retains_failed_pending_and_later_pcm_but_not_earlier_success(
    coordinator_module: Any,
) -> None:
    delivered: list[tuple[int, str]] = []
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    handle = _begin(
        coordinator_module,
        coordinator,
        callback=lambda sequence, text: delivered.append((sequence, text)),
    )

    handle.append_segment(b"\x01\x00")
    executor.succeed({"text": "released", "logical_segments": ("released",)})
    assert executor.wait_for_submissions(1)
    handle.append_segment(b"\x02\x00")
    assert executor.wait_for_submissions(2)
    handle.append_segment(b"\x03\x00")
    executor.fail(
        TranscriptionFailureCode.INFERENCE_FAILED,
        recovery_actions=("retry_faster_whisper",),
        failed_attempt={"native": "secret /private/model.onnx"},
    )
    handle.append_segment(b"\x04\x00")
    handle.finish()

    with pytest.raises(coordinator_module.RetryableDictationFailure) as caught:
        handle.wait()
    retry = caught.value.retry_buffer

    assert str(caught.value) == "Parakeet transcription failed."
    assert "/private/model.onnx" not in repr(caught.value)
    assert retry.source.audio == b"\x02\x00\x03\x00\x04\x00"
    assert retry.segment_end_frames == (1, 2, 3)
    assert delivered == [(0, "released")]
    assert handle.take_retry_buffer() is retry
    assert handle.take_retry_buffer() is None


def test_cancel_releases_a_completed_retry_buffer(coordinator_module: Any) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    handle = _begin(coordinator_module, coordinator)
    handle.append_segment(b"\x01\x00")
    handle.finish()
    executor.fail(
        TranscriptionFailureCode.INFERENCE_FAILED,
        recovery_actions=("retry_faster_whisper",),
    )

    with pytest.raises(coordinator_module.RetryableDictationFailure):
        handle.wait()
    assert handle.cancel() is True
    assert handle.take_retry_buffer() is None
    with pytest.raises(RuntimeError) as caught:
        handle.wait()
    assert caught.value.args == (TranscriptionFailureCode.CANCELLED.value,)


def test_close_releases_a_completed_untransferred_retry_buffer(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    handle = _begin(coordinator_module, coordinator)
    handle.append_segment(b"\x01\x00")
    handle.finish()
    executor.fail(
        TranscriptionFailureCode.INFERENCE_FAILED,
        recovery_actions=("retry_faster_whisper",),
    )

    with pytest.raises(coordinator_module.RetryableDictationFailure):
        handle.wait()
    assert coordinator.dictation_reserved is False
    coordinator.close()

    assert handle.take_retry_buffer() is None


def test_second_retryable_capture_does_not_invalidate_first_retry_buffer(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    first = _begin(coordinator_module, coordinator, capture_generation=11)
    first.append_segment(b"\x01\x00")
    first.finish()
    executor.fail(
        TranscriptionFailureCode.INFERENCE_FAILED,
        recovery_actions=("retry_faster_whisper",),
    )
    with pytest.raises(coordinator_module.RetryableDictationFailure) as first_error:
        first.wait()

    second = _begin(coordinator_module, coordinator, capture_generation=12)
    second.append_segment(b"\x02\x00")
    second.finish()
    executor.fail(
        TranscriptionFailureCode.INFERENCE_FAILED,
        recovery_actions=("retry_faster_whisper",),
    )
    with pytest.raises(coordinator_module.RetryableDictationFailure) as second_error:
        second.wait()

    assert first.take_retry_buffer() is first_error.value.retry_buffer
    assert second.take_retry_buffer() is second_error.value.retry_buffer


def test_close_releases_all_live_handles_untransferred_retry_buffers(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    handles = []
    for generation, audio in ((21, b"\x01\x00"), (22, b"\x02\x00")):
        handle = _begin(
            coordinator_module,
            coordinator,
            capture_generation=generation,
        )
        handle.append_segment(audio)
        handle.finish()
        executor.fail(
            TranscriptionFailureCode.INFERENCE_FAILED,
            recovery_actions=("retry_faster_whisper",),
        )
        with pytest.raises(coordinator_module.RetryableDictationFailure):
            handle.wait()
        handles.append(handle)

    coordinator.close()

    assert [handle.take_retry_buffer() for handle in handles] == [None, None]


def test_identity_change_resubmits_off_reader_then_preserves_callback_order(
    coordinator_module: Any,
) -> None:
    order: list[str] = []
    idle = threading.Event()
    executor = FakeExecutor(order)
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(
        executor,
        on_dictation_idle=lambda: (order.append("top-up"), idle.set()),
    )
    coordinator.submit_library(
        **_library_kwargs(
            model_id="parakeet-tdt-0.6b-v3",
            on_result=lambda _result: order.append("batch-callback"),
        )
    )
    handle = _begin(
        coordinator_module,
        coordinator,
        dispatch=_dispatch("parakeet-tdt-0.6b-v2"),
        callback=lambda sequence, _text: order.append(f"dictation-callback-{sequence}"),
    )
    handle.append_segment(b"\x01\x00")
    handle.finish()

    executor.succeed(thread_name="fake-reader-v3")
    assert executor.wait_for_submissions(2)
    dictation_submission = executor.submissions[1]

    assert dictation_submission["submit_thread"] != "fake-reader-v3"
    assert dictation_submission["submit_thread"].startswith("local-stt-dispatch-")
    assert order[:3] == ["submit-library", "submit-dictation", "batch-callback"]
    executor.succeed({"text": "hello", "logical_segments": ("hello",)})
    handle.wait()
    _wait_for(idle)
    assert order[-2:] == ["dictation-callback-0", "top-up"]


def test_processing_thread_exits_within_join_bound_while_dictation_waits(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    coordinator.submit_library(**_library_kwargs())
    handle = _begin(coordinator_module, coordinator)
    processing_done = threading.Event()
    release_batch = threading.Event()
    batch_done = threading.Event()
    wait_done = threading.Event()

    def process_audio() -> None:
        handle.append_segment(b"\x01\x00")
        handle.finish()
        processing_done.set()

    processing = threading.Thread(target=process_audio, name="dictation-processing")
    batch = threading.Thread(
        target=lambda: (
            release_batch.wait(),
            executor.succeed(),
            batch_done.set(),
        ),
        name="event-delayed-batch",
    )
    waiter = threading.Thread(
        target=lambda: (handle.wait(), wait_done.set()),
        name="dictation-waiter",
    )
    batch.start()
    processing.start()
    processing.join(0.05)
    waiter.start()

    assert processing_done.is_set()
    assert not processing.is_alive()
    assert len(executor.submissions) == 1
    assert not batch_done.is_set()
    assert not wait_done.is_set()

    release_batch.set()
    _wait_for(batch_done)
    assert executor.wait_for_submissions(2)
    assert not wait_done.is_set()
    executor.succeed({"text": "kept", "logical_segments": ("kept",)})
    _wait_for(wait_done)
    batch.join(1.0)
    waiter.join(1.0)


def test_blocking_one_shot_returns_exact_executor_payload(coordinator_module: Any) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    expected = {
        "text": "one shot",
        "logical_segments": ("one shot",),
        "duration": 0.25,
        "transcription_model": "parakeet-tdt-0.6b-v2",
        "transcription_provenance": {"schema_version": 1},
    }
    returned: list[dict[str, Any]] = []
    completed = threading.Event()

    def call() -> None:
        returned.append(
            coordinator.transcribe_buffer(
                source=BufferAudioSource(b"\x01\x00", 8_000),
                dispatch=_dispatch(),
                language="en",
            )
        )
        completed.set()

    caller = threading.Thread(target=call, name="one-shot-caller")
    caller.start()
    assert executor.wait_for_submissions(1)
    assert not completed.is_set()
    executor.succeed(expected)
    _wait_for(completed)
    caller.join(1.0)

    assert returned == [expected]


def test_blocking_one_shot_accepts_exact_pcm_limit(coordinator_module: Any) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)
    exact = b"\x01\x00" * 120
    expected = {"text": "exact", "logical_segments": ("exact",)}
    returned: list[dict[str, Any]] = []
    completed = threading.Event()

    def call() -> None:
        returned.append(
            coordinator.transcribe_buffer(
                source=BufferAudioSource(exact, 2),
                dispatch=_dispatch(),
                language="en",
            )
        )
        completed.set()

    caller = threading.Thread(target=call, name="exact-limit-one-shot-caller")
    caller.start()
    assert executor.wait_for_submissions(1)
    assert executor.submissions[0]["source"].audio == exact
    executor.succeed(expected)
    _wait_for(completed)
    caller.join(1.0)

    assert returned == [expected]


def test_blocking_one_shot_rejects_pcm_over_limit_before_submit(
    coordinator_module: Any,
) -> None:
    executor = FakeExecutor()
    coordinator = coordinator_module.LocalSTTDispatchCoordinator(executor)

    def unexpected_submit(**_kwargs: Any) -> int:
        raise AssertionError("oversized one-shot reached the executor")

    executor.submit = unexpected_submit

    with pytest.raises(ValueError, match="exceeds the 60-second PCM limit"):
        coordinator.transcribe_buffer(
            source=BufferAudioSource(b"\x01\x00" * 121, 2),
            dispatch=_dispatch(),
            language="en",
        )
