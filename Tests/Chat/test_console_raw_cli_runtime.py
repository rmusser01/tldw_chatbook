"""Runtime ownership tests for the process-lifetime raw CLI gate."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Tools.raw_cli_executor import RawCliRequest, RawCliResult


@pytest.fixture
def raw_runtime_module() -> Any:
    """Load lazily so RED evidence is an assertion, not a collection error."""
    module_name = "tldw_chatbook.Chat.console_raw_cli"
    if importlib.util.find_spec(module_name) is None:
        return SimpleNamespace()
    return importlib.import_module(module_name)


def _required(module: Any, name: str) -> Any:
    value = getattr(module, name, None)
    assert value is not None, f"raw CLI runtime contract {name} is missing"
    return value


def _request(
    directory: Path, invocation_id: str = "raw-1", **overrides: Any
) -> RawCliRequest:
    values: dict[str, Any] = {
        "invocation_id": invocation_id,
        "caller": "user",
        "command": "printf hello",
        "shell": "auto",
        "initial_directory": directory,
        "timeout_seconds": 30.0,
        "console_session_id": "console-1",
        "transcript_anchor_id": None,
    }
    values.update(overrides)
    return RawCliRequest(**values)


def _result(request: RawCliRequest, terminal_state: str) -> RawCliResult:
    return RawCliResult(
        invocation_id=request.invocation_id,
        caller=request.caller,
        resolved_shell=request.shell,
        initial_directory=request.initial_directory,
        elapsed_seconds=0.0,
        stdout_preview="",
        stderr_preview="",
        record_output="",
        exit_code=0 if terminal_state == "exited" else None,
        terminal_state=terminal_state,
        truncated=False,
        cleanup_proven=True,
    )


class _UnexpectedExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, *_args: Any, **_kwargs: Any) -> RawCliResult:
        self.calls += 1
        raise AssertionError("refused raw CLI request reached the executor")


class _WaitingAdmissionExecutor:
    """Expose the gap between runtime registration and worker admission."""

    def __init__(self) -> None:
        self.waiting = threading.Event()
        self.release = threading.Event()
        self.cancel_event: threading.Event | None = None
        self.admitted = False
        self.committed = False

    def execute(
        self,
        request: RawCliRequest,
        *,
        cancel_event: threading.Event,
        on_event: Any,
        admit_worker: Any,
    ) -> RawCliResult:
        del on_event
        self.cancel_event = cancel_event
        self.waiting.set()
        assert self.release.wait(2.0), "test did not release admission"
        owner = self

        class Tree:
            def admit(self) -> None:
                owner.admitted = True

        def commit_launch() -> float:
            owner.committed = True
            return 1.0

        admit_worker(Tree(), commit_launch)
        state = "exited" if self.committed else "containment_unavailable"
        return _result(request, state)


class _BlockingTreeAdmissionExecutor:
    """Block inside containment admission until the test releases it."""

    def __init__(self) -> None:
        self.admission_started = threading.Event()
        self.release_admission = threading.Event()
        self.cancel_event: threading.Event | None = None
        self.committed = False
        self.calls = 0

    def execute(
        self,
        request: RawCliRequest,
        *,
        cancel_event: threading.Event,
        on_event: Any,
        admit_worker: Any,
    ) -> RawCliResult:
        del on_event
        self.calls += 1
        self.cancel_event = cancel_event
        owner = self

        class Tree:
            def admit(self) -> None:
                owner.admission_started.set()
                assert owner.release_admission.wait(2.0), (
                    "test did not release containment admission"
                )

        def commit_launch() -> float:
            owner.committed = True
            return 1.0

        admit_worker(Tree(), commit_launch)
        if self.committed:
            state = "exited"
        elif cancel_event.is_set():
            state = "cancelled"
        else:
            state = "containment_unavailable"
        return _result(request, state)


class _BlockingLaunchCommitExecutor:
    """Expose the authority-decision to launch-commit race window."""

    def __init__(self) -> None:
        self.commit_started = threading.Event()
        self.release_commit = threading.Event()
        self.committed = False

    def execute(
        self,
        request: RawCliRequest,
        *,
        cancel_event: threading.Event,
        on_event: Any,
        admit_worker: Any,
    ) -> RawCliResult:
        del on_event
        owner = self

        class Tree:
            def admit(self) -> None:
                return None

        def commit_launch() -> float:
            owner.commit_started.set()
            assert owner.release_commit.wait(2.0), "test did not release launch commit"
            owner.committed = True
            return 1.0

        admit_worker(Tree(), commit_launch)
        assert cancel_event.wait(2.0), "active invocation was not cancelled"
        return _result(request, "cancelled")


class _ActiveExecutor:
    """Admit each invocation and remain active until cancellation arrives."""

    def __init__(self, expected: int) -> None:
        self.expected = expected
        self._lock = threading.Lock()
        self._started_count = 0
        self.all_started = threading.Event()
        self.cancel_events: dict[str, threading.Event] = {}

    def execute(
        self,
        request: RawCliRequest,
        *,
        cancel_event: threading.Event,
        on_event: Any,
        admit_worker: Any,
    ) -> RawCliResult:
        del on_event
        committed = False

        class Tree:
            def admit(self) -> None:
                return None

        def commit_launch() -> float:
            nonlocal committed
            committed = True
            return 1.0

        admit_worker(Tree(), commit_launch)
        assert committed is True
        with self._lock:
            self.cancel_events[request.invocation_id] = cancel_event
            self._started_count += 1
            if self._started_count == self.expected:
                self.all_started.set()
        assert cancel_event.wait(2.0), "active invocation was not cancelled"
        return _result(request, "cancelled")


class _HangingExecutor:
    """Ignore cancellation until the test releases the execution thread."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel_event: threading.Event | None = None
        self.calls = 0

    def execute(
        self,
        request: RawCliRequest,
        *,
        cancel_event: threading.Event,
        on_event: Any,
        admit_worker: Any,
    ) -> RawCliResult:
        del on_event
        self.calls += 1
        self.cancel_event = cancel_event

        class Tree:
            def admit(self) -> None:
                return None

        admit_worker(Tree(), lambda: 1.0)
        self.started.set()
        assert self.release.wait(2.0), "test did not release hanging executor"
        return _result(request, "exited")


class _ImmediateExecutor:
    def execute(
        self,
        request: RawCliRequest,
        *,
        cancel_event: threading.Event,
        on_event: Any,
        admit_worker: Any,
    ) -> RawCliResult:
        del cancel_event, on_event

        class Tree:
            def admit(self) -> None:
                return None

        admit_worker(Tree(), lambda: 1.0)
        return _result(request, "exited")


class _StartedBoundaryExecutor:
    """Report a deterministic launch timestamp only after tree admission."""

    def __init__(self) -> None:
        self.admitted = False
        self.callback_saw_admission = False

    def execute(
        self,
        request: RawCliRequest,
        *,
        cancel_event: threading.Event,
        on_event: Any,
        admit_worker: Any,
    ) -> RawCliResult:
        del cancel_event, on_event
        owner = self

        class Tree:
            def admit(self) -> None:
                owner.admitted = True

        def commit_launch() -> float:
            assert owner.admitted is True
            return 123.5

        admit_worker(Tree(), commit_launch)
        return _result(request, "exited")


def test_runtime_starts_unarmed_even_when_persistent_unlock_is_true(
    raw_runtime_module: Any,
    tmp_path: Path,
) -> None:
    executor = _UnexpectedExecutor()
    runtime = _required(raw_runtime_module, "RawCliRuntime")(
        lambda: True,
        executor=executor,
    )

    assert runtime.permitted is True
    assert runtime.armed is False
    assert runtime.execute(_request(tmp_path), lambda _event: None).terminal_state == (
        "refused"
    )
    assert executor.calls == 0


def test_runtime_notifies_actual_start_only_after_containment_admission(
    raw_runtime_module: Any,
    tmp_path: Path,
) -> None:
    executor = _StartedBoundaryExecutor()
    runtime = _required(raw_runtime_module, "RawCliRuntime")(
        lambda: True,
        executor=executor,
    )
    runtime.arm()
    registered: list[str] = []
    started: list[float] = []

    result = runtime.execute(
        _request(tmp_path),
        lambda _event: None,
        on_registered=lambda: registered.append("registered"),
        on_started=lambda timestamp: (
            setattr(executor, "callback_saw_admission", executor.admitted),
            started.append(timestamp),
        ),
    )

    assert result.terminal_state == "exited"
    assert registered == ["registered"]
    assert started == [123.5]
    assert executor.callback_saw_admission is True


def test_arm_refuses_until_saved_unlock_is_strictly_true(
    raw_runtime_module: Any,
) -> None:
    saved: dict[str, object] = {"value": False}
    runtime = _required(raw_runtime_module, "RawCliRuntime")(
        lambda: saved["value"],
        executor=_UnexpectedExecutor(),
    )

    assert runtime.arm() == _required(raw_runtime_module, "RawCliArmResult")(
        armed=False,
        reason="locked",
    )
    saved["value"] = "true"
    assert runtime.arm().armed is False
    saved["value"] = 1
    assert runtime.arm().armed is False
    saved["value"] = True
    assert runtime.arm() == _required(raw_runtime_module, "RawCliArmResult")(
        armed=True,
        reason="armed",
    )
    assert runtime.armed is True


def test_execute_validates_request_before_authority_refusal(
    raw_runtime_module: Any,
    tmp_path: Path,
) -> None:
    executor = _UnexpectedExecutor()
    runtime = _required(raw_runtime_module, "RawCliRuntime")(
        lambda: False,
        executor=executor,
    )

    with pytest.raises(ValueError, match="empty|whitespace"):
        runtime.execute(_request(tmp_path, command="  "), lambda _event: None)

    assert executor.calls == 0


def test_execute_rechecks_latest_unlock_before_admission(
    raw_runtime_module: Any,
    tmp_path: Path,
) -> None:
    saved = {"value": True}
    executor = _WaitingAdmissionExecutor()
    runtime = _required(raw_runtime_module, "RawCliRuntime")(
        lambda: saved["value"],
        executor=executor,
    )
    assert runtime.arm().armed is True
    results: list[RawCliResult] = []
    thread = threading.Thread(
        target=lambda: results.append(
            runtime.execute(_request(tmp_path), lambda _event: None)
        )
    )

    thread.start()
    assert executor.waiting.wait(2.0)
    saved["value"] = False
    executor.release.set()
    thread.join(2.0)

    assert thread.is_alive() is False
    assert executor.admitted is False
    assert executor.committed is False
    assert results[0].terminal_state == "containment_unavailable"


def test_disarm_before_admission_keeps_launch_gate_closed(
    raw_runtime_module: Any,
    tmp_path: Path,
) -> None:
    executor = _WaitingAdmissionExecutor()
    runtime = _required(raw_runtime_module, "RawCliRuntime")(
        lambda: True,
        executor=executor,
    )
    runtime.arm()
    results: list[RawCliResult] = []
    thread = threading.Thread(
        target=lambda: results.append(
            runtime.execute(_request(tmp_path), lambda _event: None)
        )
    )

    thread.start()
    assert executor.waiting.wait(2.0)
    assert runtime.disarm() == ("raw-1",)
    assert executor.cancel_event is not None and executor.cancel_event.is_set()
    executor.release.set()
    thread.join(2.0)

    assert thread.is_alive() is False
    assert executor.admitted is False
    assert executor.committed is False
    assert results[0].terminal_state == "containment_unavailable"


def test_disarm_during_stalled_admission_returns_and_prevents_launch_commit(
    raw_runtime_module: Any,
    tmp_path: Path,
) -> None:
    executor = _BlockingTreeAdmissionExecutor()
    runtime = _required(raw_runtime_module, "RawCliRuntime")(
        lambda: True,
        executor=executor,
    )
    runtime.arm()
    results: list[RawCliResult] = []
    execute_thread = threading.Thread(
        target=lambda: results.append(
            runtime.execute(_request(tmp_path), lambda _event: None)
        )
    )
    execute_thread.start()
    assert executor.admission_started.wait(2.0)

    disarm_results: list[tuple[str, ...]] = []
    disarm_thread = threading.Thread(
        target=lambda: disarm_results.append(runtime.disarm())
    )
    disarm_thread.start()
    disarm_thread.join(0.2)
    returned_before_release = not disarm_thread.is_alive()
    cancellation_signalled = bool(
        executor.cancel_event is not None and executor.cancel_event.is_set()
    )

    executor.release_admission.set()
    disarm_thread.join(2.0)
    execute_thread.join(2.0)

    assert returned_before_release is True
    assert disarm_results == [("raw-1",)]
    assert cancellation_signalled is True
    assert executor.committed is False
    assert results[0].terminal_state == "cancelled"


def test_disarm_cannot_finish_between_authority_decision_and_launch_commit(
    raw_runtime_module: Any,
    tmp_path: Path,
) -> None:
    executor = _BlockingLaunchCommitExecutor()
    runtime = _required(raw_runtime_module, "RawCliRuntime")(
        lambda: True,
        executor=executor,
    )
    runtime.arm()
    results: list[RawCliResult] = []
    execute_thread = threading.Thread(
        target=lambda: results.append(
            runtime.execute(_request(tmp_path), lambda _event: None)
        )
    )
    execute_thread.start()
    assert executor.commit_started.wait(2.0)

    disarm_results: list[tuple[str, ...]] = []
    disarm_started = threading.Event()
    disarm_finished = threading.Event()

    def disarm() -> None:
        disarm_started.set()
        disarm_results.append(runtime.disarm())
        disarm_finished.set()

    disarm_thread = threading.Thread(target=disarm)
    disarm_thread.start()
    assert disarm_started.wait(2.0)
    disarm_finished_before_commit = disarm_finished.wait(0.2)

    executor.release_commit.set()
    disarm_thread.join(2.0)
    execute_thread.join(2.0)

    assert disarm_finished_before_commit is False
    assert executor.committed is True
    assert disarm_results == [("raw-1",)]
    assert disarm_finished.is_set()
    assert results[0].terminal_state == "cancelled"


def test_disarm_clears_session_grants_and_cancels_every_active_invocation(
    raw_runtime_module: Any,
    tmp_path: Path,
) -> None:
    grant_clears: list[str] = []
    runtime_class = _required(raw_runtime_module, "RawCliRuntime")

    class RuntimeWithGrantProbe(runtime_class):
        def _clear_model_session_grants_locked(self) -> None:
            grant_clears.append("cleared")

    executor = _ActiveExecutor(expected=2)
    runtime = RuntimeWithGrantProbe(lambda: True, executor=executor)
    runtime.arm()
    results: list[RawCliResult] = []
    threads = [
        threading.Thread(
            target=lambda invocation_id=invocation_id: results.append(
                runtime.execute(
                    _request(tmp_path, invocation_id=invocation_id),
                    lambda _event: None,
                )
            )
        )
        for invocation_id in ("raw-a", "raw-b")
    ]
    for thread in threads:
        thread.start()
    assert executor.all_started.wait(2.0)

    cancelled = runtime.disarm()

    assert cancelled == ("raw-a", "raw-b")
    assert runtime.armed is False
    assert grant_clears == ["cleared"]
    assert all(event.is_set() for event in executor.cancel_events.values())
    for thread in threads:
        thread.join(2.0)
        assert thread.is_alive() is False
    assert sorted(result.terminal_state for result in results) == [
        "cancelled",
        "cancelled",
    ]


def test_cancel_session_signals_only_that_sessions_active_invocations(
    raw_runtime_module: Any,
    tmp_path: Path,
) -> None:
    executor = _ActiveExecutor(expected=3)
    runtime = _required(raw_runtime_module, "RawCliRuntime")(
        lambda: True,
        executor=executor,
    )
    runtime.arm()
    requests = (
        _request(tmp_path, "raw-a", console_session_id="session-a"),
        _request(tmp_path, "raw-b", console_session_id="session-a"),
        _request(tmp_path, "raw-c", console_session_id="session-b"),
    )
    results: list[RawCliResult] = []
    threads = [
        threading.Thread(
            target=lambda request=request: results.append(
                runtime.execute(request, lambda _event: None)
            )
        )
        for request in requests
    ]
    for thread in threads:
        thread.start()
    assert executor.all_started.wait(2.0)

    assert runtime.cancel_session("session-a") == ("raw-a", "raw-b")
    assert executor.cancel_events["raw-a"].is_set()
    assert executor.cancel_events["raw-b"].is_set()
    assert not executor.cancel_events["raw-c"].is_set()
    assert runtime.cancel_session("missing") == ()

    assert runtime.cancel("raw-c") is True
    for thread in threads:
        thread.join(2.0)
        assert not thread.is_alive()
    assert sorted(result.terminal_state for result in results) == [
        "cancelled",
        "cancelled",
        "cancelled",
    ]


def test_shutdown_is_idempotent_bounded_and_prevents_new_work(
    raw_runtime_module: Any,
    tmp_path: Path,
) -> None:
    executor = _HangingExecutor()
    runtime = _required(raw_runtime_module, "RawCliRuntime")(
        lambda: True,
        executor=executor,
        shutdown_timeout_seconds=0.03,
    )
    runtime.arm()
    results: list[RawCliResult] = []
    thread = threading.Thread(
        target=lambda: results.append(
            runtime.execute(_request(tmp_path), lambda _event: None)
        )
    )
    thread.start()
    assert executor.started.wait(2.0)

    started_at = time.monotonic()
    first = runtime.shutdown()
    elapsed = time.monotonic() - started_at
    second = runtime.shutdown()

    assert elapsed < 0.5
    assert first == second
    assert first == _required(raw_runtime_module, "RawCliShutdownResult")(
        cancelled_invocation_ids=("raw-1",),
        unfinished_invocation_ids=("raw-1",),
    )
    assert executor.cancel_event is not None and executor.cancel_event.is_set()
    assert runtime.armed is False
    assert runtime.arm().reason == "shutdown"
    assert (
        runtime.execute(
            _request(tmp_path, invocation_id="raw-after-shutdown"),
            lambda _event: None,
        ).terminal_state
        == "refused"
    )
    assert executor.calls == 1

    executor.release.set()
    thread.join(2.0)
    assert thread.is_alive() is False
    assert results[0].terminal_state == "exited"


def test_shutdown_is_bounded_while_containment_admission_is_stalled(
    raw_runtime_module: Any,
    tmp_path: Path,
) -> None:
    executor = _BlockingTreeAdmissionExecutor()
    runtime = _required(raw_runtime_module, "RawCliRuntime")(
        lambda: True,
        executor=executor,
        shutdown_timeout_seconds=0.03,
    )
    runtime.arm()
    results: list[RawCliResult] = []
    execute_thread = threading.Thread(
        target=lambda: results.append(
            runtime.execute(_request(tmp_path), lambda _event: None)
        )
    )
    execute_thread.start()
    assert executor.admission_started.wait(2.0)

    shutdown_results: list[Any] = []
    shutdown_thread = threading.Thread(
        target=lambda: shutdown_results.append(runtime.shutdown())
    )
    shutdown_thread.start()
    shutdown_thread.join(0.2)
    returned_before_release = not shutdown_thread.is_alive()
    if returned_before_release:
        arm_reason = runtime.arm().reason
        refused_state = runtime.execute(
            _request(tmp_path, invocation_id="raw-after-shutdown"),
            lambda _event: None,
        ).terminal_state
    else:
        arm_reason = None
        refused_state = None
    cancellation_signalled = bool(
        executor.cancel_event is not None and executor.cancel_event.is_set()
    )

    executor.release_admission.set()
    shutdown_thread.join(2.0)
    execute_thread.join(2.0)

    assert returned_before_release is True
    assert shutdown_results == [
        _required(raw_runtime_module, "RawCliShutdownResult")(
            cancelled_invocation_ids=("raw-1",),
            unfinished_invocation_ids=("raw-1",),
        )
    ]
    assert arm_reason == "shutdown"
    assert refused_state == "refused"
    assert cancellation_signalled is True
    assert executor.calls == 1
    assert executor.committed is False
    assert results[0].terminal_state == "cancelled"


def test_terminal_result_wins_over_late_disarm_and_cancel(
    raw_runtime_module: Any,
    tmp_path: Path,
) -> None:
    runtime = _required(raw_runtime_module, "RawCliRuntime")(
        lambda: True,
        executor=_ImmediateExecutor(),
    )
    runtime.arm()

    result = runtime.execute(_request(tmp_path), lambda _event: None)

    assert result.terminal_state == "exited"
    assert runtime.cancel("raw-1") is False
    assert runtime.disarm() == ()
    assert result.terminal_state == "exited"
