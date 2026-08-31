from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest

from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
from tldw_chatbook.Terminal.contracts import (
    AdmissionGate,
    BackendIdentity,
    CleanupAttempt,
    CleanupProof,
    CleanupSchedule,
    MAX_SESSION_RECORDS,
    TerminalLaunchRequest,
    TerminalLifecycle,
    TerminalReason,
)
from tldw_chatbook.Terminal.io_actors import InputEventKind, TerminalOutputActor
from tldw_chatbook.Terminal.session_manager import (
    ManagedProcessIdentity,
    TerminalArmResult,
    TerminalCleanupDeadline,
    TerminalCreateResult,
    TerminalSessionView,
    TerminalSessionManager,
    TerminalSubscriptionToken,
    TerminalViewState,
    TerminalViewToken,
)
from tldw_chatbook.Terminal.screen_model import (
    TerminalScreenModel,
    TerminalScreenSnapshot,
)


def launch_request(name: str) -> TerminalLaunchRequest:
    """Return one already-validated launch request for manager tests."""
    return TerminalLaunchRequest(
        name=name,
        shell="default",
        start_directory=str(Path.cwd()),
        columns=80,
        rows=24,
    )


class RecordingBackend:
    """Narrow complete backend used to observe manager coordination."""

    def __init__(
        self,
        *,
        on_start: Callable[[TerminalLaunchRequest, AdmissionGate], None] | None = None,
        on_cleanup: Callable[[CleanupAttempt], CleanupProof] | None = None,
        on_raw_cleanup_drain: Callable[[CleanupAttempt], CleanupProof] | None = None,
        cleanup_proof: CleanupProof = CleanupProof(True, True, True),
    ) -> None:
        self.on_start = on_start
        self.on_cleanup = on_cleanup
        self.on_raw_cleanup_drain = on_raw_cleanup_drain
        self.cleanup_proof = cleanup_proof
        self.started: list[tuple[TerminalLaunchRequest, AdmissionGate]] = []
        self.writes: list[bytes] = []
        self.resizes: list[tuple[int, int]] = []
        self.priority_close_requests = 0
        self.cleanup_attempts: list[CleanupAttempt] = []
        self.raw_cleanup_attempts: list[CleanupAttempt] = []

    def start(
        self, request: TerminalLaunchRequest, admission: AdmissionGate
    ) -> BackendIdentity:
        self.started.append((request, admission))
        if self.on_start is not None:
            self.on_start(request, admission)
        if admission.admitted is not True:
            raise RuntimeError("backend received a refused admission")
        return BackendIdentity(session_id=admission.token)

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def resize(self, columns: int, rows: int) -> None:
        self.resizes.append((columns, rows))

    def request_priority_close(self) -> None:
        self.priority_close_requests += 1

    def cleanup(self, attempt: CleanupAttempt) -> CleanupProof:
        self.cleanup_attempts.append(attempt)
        if self.on_cleanup is not None:
            return self.on_cleanup(attempt)
        return self.cleanup_proof

    def cleanup_raw_drain(self, attempt: CleanupAttempt) -> CleanupProof:
        self.raw_cleanup_attempts.append(attempt)
        if self.on_raw_cleanup_drain is not None:
            return self.on_raw_cleanup_drain(attempt)
        return self.cleanup_proof


class ManualClock:
    """Thread-safe manually advanced monotonic clock."""

    def __init__(self, value: float) -> None:
        self._value = value
        self._lock = Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value


def create_running_session(
    terminal: TerminalSessionManager,
    name: str,
) -> str:
    """Create one running session and return its opaque ID."""
    result = terminal.create_session(launch_request(name))
    assert result.admitted is True
    assert result.projection is not None
    return result.projection.session_id


def test_create_session_docstring_documents_parameter_and_result() -> None:
    docstring = inspect.getdoc(TerminalSessionManager.create_session)

    assert docstring is not None
    assert "Args:" in docstring
    assert "Returns:" in docstring


def test_terminal_arm_is_independent_and_resets_per_manager() -> None:
    raw = RawCliRuntime(lambda: True)
    terminal = TerminalSessionManager(
        read_permitted=lambda: True,
        backend_factory=RecordingBackend,
    )

    assert terminal.discoverable is True
    assert terminal.permitted is True
    assert terminal.armed is False
    assert terminal.disclosure_acknowledged is False
    assert raw.arm().armed is True
    assert terminal.armed is False

    first_arm = terminal.arm()
    assert first_arm == TerminalArmResult(
        armed=False,
        reason=None,
        disclosure_required=True,
    )
    assert terminal.arm(acknowledge_disclosure=True).armed is True
    assert terminal.disclosure_acknowledged is True

    raw.disarm()
    assert terminal.armed is True
    terminal.disarm()
    assert terminal.armed is False
    assert terminal.disclosure_acknowledged is True
    assert terminal.arm().armed is True

    fresh = TerminalSessionManager(lambda: True, RecordingBackend)
    assert fresh.armed is False
    assert fresh.disclosure_acknowledged is False
    assert fresh.arm().disclosure_required is True


@pytest.mark.parametrize("saved_value", [False, None, "true", "1", 1, object()])
def test_terminal_arm_requires_the_literal_persisted_boolean_true(
    saved_value: object,
) -> None:
    terminal = TerminalSessionManager(lambda: saved_value, RecordingBackend)

    result = terminal.arm(acknowledge_disclosure=True)

    assert result == TerminalArmResult(
        armed=False,
        reason=TerminalReason.LOCKED,
        disclosure_required=False,
    )
    assert terminal.discoverable is True
    assert terminal.permitted is False
    assert terminal.armed is False
    assert terminal.disclosure_acknowledged is False


def test_terminal_permission_reader_failure_is_fail_closed() -> None:
    def broken_reader() -> bool:
        raise RuntimeError("unreadable setting")

    terminal = TerminalSessionManager(broken_reader, RecordingBackend)

    assert terminal.permitted is False
    assert terminal.arm(acknowledge_disclosure=True).reason is TerminalReason.LOCKED
    assert terminal.armed is False


def test_create_requires_current_unlock_and_launch_local_arm() -> None:
    permitted = True
    terminal = TerminalSessionManager(lambda: permitted, RecordingBackend)

    unarmed = terminal.create_session(launch_request("unarmed"))
    assert unarmed == TerminalCreateResult(reason=TerminalReason.UNARMED)

    terminal.arm(acknowledge_disclosure=True)
    permitted = False
    locked = terminal.create_session(launch_request("locked"))
    assert locked == TerminalCreateResult(reason=TerminalReason.LOCKED)
    assert terminal.armed is False
    assert terminal.projections() == ()


def test_persisted_unlock_revocation_disarms_cleans_and_requires_rearming() -> None:
    permitted = True
    clock = ManualClock(25.0)
    backends: list[RecordingBackend] = []

    def backend_factory() -> RecordingBackend:
        backend = RecordingBackend()
        backends.append(backend)
        return backend

    terminal = TerminalSessionManager(
        lambda: permitted,
        backend_factory,
        monotonic_clock=clock,
    )
    terminal.arm(acknowledge_disclosure=True)
    create_running_session(terminal, "revoked-one")
    create_running_session(terminal, "revoked-two")

    permitted = False
    assert terminal.permitted is False
    assert terminal.armed is False
    assert all(
        terminal.wait_for_cleanup(projection.session_id, timeout_seconds=1)
        for projection in terminal.projections()
    )
    assert [backend.cleanup_attempts for backend in backends] == [
        [CleanupAttempt(25.0)],
        [CleanupAttempt(25.0)],
    ]
    assert terminal.projections() == ()

    permitted = True
    assert terminal.permitted is True
    assert terminal.armed is False
    assert terminal.create_session(launch_request("still-unarmed")) == (
        TerminalCreateResult(reason=TerminalReason.UNARMED)
    )


def test_create_reserves_atomically_before_calling_the_backend() -> None:
    worker_count = MAX_SESSION_RECORDS + 3
    race = Barrier(worker_count + 1)
    observation_lock = Lock()
    backend_calls = 0
    terminal: TerminalSessionManager

    def observe_start(request: TerminalLaunchRequest, admission: AdmissionGate) -> None:
        nonlocal backend_calls
        with observation_lock:
            backend_calls += 1
        assert admission.admitted is True
        assert any(
            item.session_id == admission.token
            and item.name == request.name
            and item.lifecycle
            in {
                TerminalLifecycle.RESERVED,
                TerminalLifecycle.CREATING,
                TerminalLifecycle.ADMITTING,
            }
            for item in terminal.projections()
        )

    def backend_factory() -> RecordingBackend:
        return RecordingBackend(on_start=observe_start)

    terminal = TerminalSessionManager(lambda: True, backend_factory)
    assert terminal.arm(acknowledge_disclosure=True).armed is True

    def create(index: int) -> TerminalCreateResult:
        race.wait()
        return terminal.create_session(launch_request(f"session-{index}"))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(create, index) for index in range(worker_count)]
        race.wait()
        results = [future.result(timeout=2) for future in futures]

    admitted = [result for result in results if result.admitted]
    refused = [result for result in results if not result.admitted]
    assert len(admitted) == MAX_SESSION_RECORDS
    assert len(refused) == worker_count - MAX_SESSION_RECORDS
    assert {result.reason for result in refused} == {TerminalReason.SESSION_LIMIT}
    assert backend_calls == MAX_SESSION_RECORDS
    assert len(terminal.projections()) == MAX_SESSION_RECORDS


def test_prelaunch_failure_releases_its_reservation() -> None:
    calls = 0

    def backend_factory() -> RecordingBackend:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("backend unavailable before launch")
        return RecordingBackend()

    terminal = TerminalSessionManager(lambda: True, backend_factory)
    terminal.arm(acknowledge_disclosure=True)

    failed = terminal.create_session(launch_request("reusable"))
    succeeded = terminal.create_session(launch_request("reusable"))

    assert failed == TerminalCreateResult(reason=TerminalReason.BACKEND_UNAVAILABLE)
    assert succeeded.admitted is True
    assert len(terminal.projections()) == 1


def test_casefolded_unicode_names_are_unique_across_retained_records() -> None:
    terminal = TerminalSessionManager(lambda: True, RecordingBackend)
    terminal.arm(acknowledge_disclosure=True)

    first = terminal.create_session(launch_request("  Straße  "))
    duplicate = terminal.create_session(launch_request("STRASSE"))

    assert first.admitted is True
    assert first.projection is not None
    assert first.projection.name == "Straße"
    assert duplicate == TerminalCreateResult(reason=TerminalReason.INVALID_NAME)
    assert len(terminal.projections()) == 1


def test_public_manager_results_and_projections_are_immutable() -> None:
    terminal = TerminalSessionManager(lambda: True, RecordingBackend)
    arm_result = terminal.arm(acknowledge_disclosure=True)
    create_result = terminal.create_session(launch_request("immutable"))

    with pytest.raises(FrozenInstanceError):
        arm_result.armed = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        create_result.reason = TerminalReason.IO_FAILED  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        create_result.projection.lifecycle = TerminalLifecycle.CLOSED  # type: ignore[misc,union-attr]


def test_shell_exit_uses_one_absolute_attempt_and_retains_exited_record() -> None:
    clock = ManualClock(100.0)
    cleanup_started = Event()
    finish_cleanup = Event()
    backend = RecordingBackend()

    def cleanup(attempt: CleanupAttempt) -> CleanupProof:
        cleanup_started.set()
        assert finish_cleanup.wait(1)
        return CleanupProof(True, True, True)

    backend.on_cleanup = cleanup
    terminal = TerminalSessionManager(
        lambda: True,
        lambda: backend,
        monotonic_clock=clock,
    )
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "natural-exit")

    receipt = terminal.shell_exited(session_id, exit_code=7)

    assert receipt is not None
    assert receipt.attempt == CleanupAttempt(100.0)
    assert receipt.action == "shell_exit"
    assert cleanup_started.wait(1)
    draining = terminal.projection(session_id)
    assert draining is not None
    assert draining.lifecycle is TerminalLifecycle.DRAINING
    assert draining.exit_code == 7
    deadline = terminal.cleanup_deadline(session_id)
    assert deadline == TerminalCleanupDeadline(
        attempt=CleanupAttempt(100.0),
        hangup_at=100.75,
        terminate_at=102.25,
        force_kill_at=103.75,
        deadline_at=105.0,
    )

    clock.set(1_000.0)
    finish_cleanup.set()
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
    exited = terminal.projection(session_id)
    assert exited is not None
    assert exited.lifecycle is TerminalLifecycle.EXITED
    assert exited.exit_code == 7
    assert exited.stream_closed is True
    assert exited.output_complete is True
    assert backend.cleanup_attempts == [CleanupAttempt(100.0)]


def test_disarm_starts_one_parallel_cleanup_cohort_after_clearing_arm() -> None:
    clock = ManualClock(200.0)
    cleanup_started = Barrier(3)
    finish_cleanup = Event()
    backends: list[RecordingBackend] = []
    terminal: TerminalSessionManager

    def backend_factory() -> RecordingBackend:
        def cleanup(attempt: CleanupAttempt) -> CleanupProof:
            assert terminal.armed is False
            cleanup_started.wait()
            assert finish_cleanup.wait(1)
            return CleanupProof(True, True, True)

        backend = RecordingBackend(on_cleanup=cleanup)
        backends.append(backend)
        return backend

    terminal = TerminalSessionManager(
        lambda: True,
        backend_factory,
        monotonic_clock=clock,
    )
    terminal.arm(acknowledge_disclosure=True)
    session_ids = [
        create_running_session(terminal, "parallel-one"),
        create_running_session(terminal, "parallel-two"),
    ]

    terminal.disarm()

    assert terminal.armed is False
    cleanup_started.wait(timeout=1)
    assert {
        terminal.projection(session_id).lifecycle  # type: ignore[union-attr]
        for session_id in session_ids
    } == {TerminalLifecycle.CLOSING}
    assert [backend.cleanup_attempts for backend in backends] == [
        [CleanupAttempt(200.0)],
        [CleanupAttempt(200.0)],
    ]
    finish_cleanup.set()
    for session_id in session_ids:
        assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
    assert terminal.projections() == ()


def test_joined_disarm_does_not_extend_or_duplicate_an_existing_cleanup() -> None:
    clock = ManualClock(300.0)
    cleanup_started = Event()
    finish_cleanup = Event()

    def cleanup(attempt: CleanupAttempt) -> CleanupProof:
        cleanup_started.set()
        assert finish_cleanup.wait(1)
        return CleanupProof(True, True, True)

    backend = RecordingBackend(on_cleanup=cleanup)
    terminal = TerminalSessionManager(
        lambda: True,
        lambda: backend,
        monotonic_clock=clock,
    )
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "joined")
    terminal.shell_exited(session_id, exit_code=0)
    assert cleanup_started.wait(1)

    clock.set(301.0)
    terminal.disarm()

    receipt = terminal.cleanup_receipt(session_id)
    assert receipt is not None
    assert receipt.attempt == CleanupAttempt(300.0)
    assert backend.cleanup_attempts == [CleanupAttempt(300.0)]
    finish_cleanup.set()
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)


def test_only_explicit_retry_gets_a_fresh_cleanup_attempt() -> None:
    clock = ManualClock(400.0)
    attempts = 0

    def cleanup(attempt: CleanupAttempt) -> CleanupProof:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return CleanupProof(False, False, False)
        return CleanupProof(True, True, True)

    backend = RecordingBackend(on_cleanup=cleanup)
    permitted = True
    terminal = TerminalSessionManager(
        lambda: permitted,
        lambda: backend,
        monotonic_clock=clock,
    )
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "retry")
    terminal.disarm()
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
    retained = terminal.projection(session_id)
    assert retained is not None
    assert retained.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
    assert terminal.cleanup_receipt(session_id).attempt == CleanupAttempt(400.0)  # type: ignore[union-attr]

    permitted = False
    clock.set(500.0)
    view = terminal.attach_view()
    retry_receipt = terminal.retry_cleanup(session_id, view=view)

    assert retry_receipt is not None
    assert retry_receipt.action == "retry"
    assert retry_receipt.attempt == CleanupAttempt(500.0)
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
    assert terminal.projection(session_id) is None
    assert backend.cleanup_attempts == [CleanupAttempt(400.0), CleanupAttempt(500.0)]


@pytest.mark.asyncio
async def test_cleanup_unproven_is_never_retried_by_other_actions() -> None:
    clock = ManualClock(525.0)
    backend = RecordingBackend(cleanup_proof=CleanupProof())
    terminal = TerminalSessionManager(
        lambda: True,
        lambda: backend,
        monotonic_clock=clock,
    )
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "explicit-retry-only")
    terminal.disarm()
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
    original_receipt = terminal.cleanup_receipt(session_id)
    assert original_receipt is not None

    clock.set(550.0)
    terminal.disarm()
    view = terminal.attach_view()
    assert terminal.close_session(session_id, view=view) == original_receipt
    assert await terminal.shutdown(deadline_seconds=0.01) is False
    assert terminal.shell_exited(session_id, exit_code=9) == original_receipt

    retained = terminal.projection(session_id)
    assert retained is not None
    assert retained.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
    assert retained.exit_code == 9
    assert backend.cleanup_attempts == [CleanupAttempt(525.0)]


def test_shell_exit_joins_cleanup_already_in_progress() -> None:
    clock = ManualClock(575.0)
    cleanup_started = Event()
    finish_cleanup = Event()

    def cleanup(attempt: CleanupAttempt) -> CleanupProof:
        cleanup_started.set()
        assert finish_cleanup.wait(1)
        return CleanupProof(True, True, True)

    backend = RecordingBackend(on_cleanup=cleanup)
    terminal = TerminalSessionManager(
        lambda: True,
        lambda: backend,
        monotonic_clock=clock,
    )
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "late-shell-exit")
    view = terminal.attach_view()
    close_receipt = terminal.close_session(session_id, view=view)
    assert close_receipt is not None
    assert cleanup_started.wait(1)

    clock.set(580.0)
    shell_receipt = terminal.shell_exited(session_id, exit_code=7)

    assert shell_receipt == close_receipt
    closing = terminal.projection(session_id)
    assert closing is not None
    assert closing.lifecycle is TerminalLifecycle.CLOSING
    assert closing.exit_code == 7
    assert backend.cleanup_attempts == [CleanupAttempt(575.0)]
    finish_cleanup.set()
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)


@pytest.mark.asyncio
async def test_shutdown_is_bounded_and_uses_one_global_attempt() -> None:
    clock = ManualClock(600.0)
    backends: list[RecordingBackend] = []

    def backend_factory() -> RecordingBackend:
        backend = RecordingBackend()
        backends.append(backend)
        return backend

    terminal = TerminalSessionManager(
        lambda: True,
        backend_factory,
        monotonic_clock=clock,
    )
    terminal.arm(acknowledge_disclosure=True)
    create_running_session(terminal, "shutdown-one")
    create_running_session(terminal, "shutdown-two")

    assert await terminal.shutdown(deadline_seconds=0.5) is True

    assert terminal.armed is False
    assert terminal.projections() == ()
    assert [backend.cleanup_attempts for backend in backends] == [
        [CleanupAttempt(600.0)],
        [CleanupAttempt(600.0)],
    ]


def test_parser_failure_disables_input_and_raw_drains_only_after_death() -> None:
    clock = ManualClock(700.0)
    cleanup_started = Event()
    finish_cleanup = Event()
    raw_drain_process_dead: list[bool] = []

    def cleanup(attempt: CleanupAttempt) -> CleanupProof:
        cleanup_started.set()
        assert finish_cleanup.wait(1)
        return CleanupProof(True, False, False)

    def raw_cleanup_drain(attempt: CleanupAttempt) -> CleanupProof:
        raw_drain_process_dead.append(True)
        return CleanupProof(True, True, False)

    backend = RecordingBackend(
        on_cleanup=cleanup,
        on_raw_cleanup_drain=raw_cleanup_drain,
    )

    class FailingScreen:
        def feed(self, data: bytes) -> None:
            del data
            raise RuntimeError("parser invariant failed")

        def resize(self, *, columns: int, rows: int) -> None:
            del columns, rows

        def snapshot(self) -> object:
            return object()

    terminal = TerminalSessionManager(
        lambda: True,
        lambda: backend,
        monotonic_clock=clock,
        screen_model_factory=lambda _columns, _rows: FailingScreen(),
    )
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "parser-failure")
    assert terminal.accepts_input(session_id) is True
    assert terminal.offer_output(session_id, b"not projected").accepted is True

    assert terminal.process_output(session_id, visible=True) is None

    failed = terminal.projection(session_id)
    assert failed is not None
    assert failed.lifecycle is TerminalLifecycle.CLOSING
    assert failed.reason is TerminalReason.TERMINAL_PROTOCOL_FAILED
    assert failed.parser_failed is True
    assert terminal.accepts_input(session_id) is False
    assert backend.priority_close_requests == 1
    assert cleanup_started.wait(1)
    assert backend.raw_cleanup_attempts == []

    finish_cleanup.set()
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
    assert raw_drain_process_dead == [True]
    assert backend.raw_cleanup_attempts == [CleanupAttempt(700.0)]
    assert terminal.projection(session_id) is None


def test_parser_failure_never_raw_drains_without_process_death_proof() -> None:
    backend = RecordingBackend(cleanup_proof=CleanupProof(False, False, False))
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "no-raw-drain")

    terminal.parser_failed(session_id)

    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
    assert backend.raw_cleanup_attempts == []
    retained = terminal.projection(session_id)
    assert retained is not None
    assert retained.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
    assert retained.reason is TerminalReason.CLEANUP_UNPROVEN


def test_screen_model_reported_failure_closes_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = RecordingBackend(cleanup_proof=CleanupProof(False, False, False))
    model = TerminalScreenModel(columns=80, rows=24)

    def fail_without_exposing_output(_: str) -> None:
        raise RuntimeError("private terminal output")

    monkeypatch.setattr(model._stream, "feed", fail_without_exposing_output)
    terminal = TerminalSessionManager(
        lambda: True,
        lambda: backend,
        screen_model_factory=lambda _columns, _rows: model,
    )
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "reported-parser-failure")
    assert terminal.offer_output(session_id, b"private terminal output").accepted

    assert terminal.process_output(session_id, visible=True) is None
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)

    retained = terminal.projection(session_id)
    assert retained is not None
    assert retained.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
    assert retained.parser_failed is True
    assert backend.priority_close_requests == 1


def test_malformed_terminal_reply_fails_closed_before_input_queueing() -> None:
    backend = RecordingBackend(cleanup_proof=CleanupProof(False, False, False))

    class MalformedReplyScreen(TerminalScreenModel):
        def take_pending_replies(self) -> tuple[object, ...]:
            return (object(),)

    terminal = TerminalSessionManager(
        lambda: True,
        lambda: backend,
        screen_model_factory=lambda columns, rows: MalformedReplyScreen(
            columns=columns,
            rows=rows,
        ),
    )
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "malformed-reply")
    assert terminal.offer_output(session_id, b"safe output").accepted

    assert terminal.process_output(session_id, visible=True) is None
    assert terminal.take_input(session_id) is None
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)

    retained = terminal.projection(session_id)
    assert retained is not None
    assert retained.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
    assert retained.parser_failed is True
    assert backend.priority_close_requests == 1


def test_cleanup_schedule_defaults_are_not_mutated_by_waits() -> None:
    assert CleanupSchedule() == CleanupSchedule(
        deadline_seconds=5.0,
        hangup_no_later_than=0.75,
        terminate_no_later_than=2.25,
        force_kill_no_later_than=3.75,
        proof_reserve_seconds=1.25,
    )


def test_view_state_contains_only_immutable_safe_projections() -> None:
    backend = RecordingBackend()
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "safe-view")
    token = terminal.attach_view()
    terminal.offer_output(session_id, b"safe output")
    terminal.process_output(session_id, visible=True)

    state = terminal.view_state(token)

    assert isinstance(token, TerminalViewToken)
    assert isinstance(state, TerminalViewState)
    assert state.selected_session_id == session_id
    assert len(state.sessions) == 1
    session = state.sessions[0]
    assert isinstance(session, TerminalSessionView)
    assert session.projection.session_id == session_id
    assert session.screen.lines[0].text.startswith("safe output")
    assert set(TerminalSessionView.__slots__) == {
        "projection",
        "screen",
        "shell",
        "start_directory",
        "columns",
        "rows",
        "cleanup_receipt",
    }
    assert session.shell == "default"
    assert session.start_directory == str(Path.cwd())
    assert (session.columns, session.rows) == (80, 24)
    assert not hasattr(session, "backend")
    assert not hasattr(session, "model")
    assert not hasattr(session, "environment")
    with pytest.raises(FrozenInstanceError):
        state.selected_session_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        session.projection.name = "changed"  # type: ignore[misc]


def test_screen_snapshot_waits_for_the_active_parser_turn() -> None:
    parser_entered = Event()
    release_parser = Event()
    concurrent_snapshot = Event()

    class CoordinatedScreen:
        def feed(self, data: bytes) -> None:
            del data
            parser_entered.set()
            assert release_parser.wait(1)

        def resize(self, *, columns: int, rows: int) -> None:
            del columns, rows

        def snapshot(self) -> TerminalScreenSnapshot:
            if parser_entered.is_set() and not release_parser.is_set():
                concurrent_snapshot.set()
            return TerminalScreenSnapshot(lines=())

        def take_pending_replies(self) -> tuple[bytes, ...]:
            return ()

    terminal = TerminalSessionManager(
        lambda: True,
        RecordingBackend,
        screen_model_factory=lambda _columns, _rows: CoordinatedScreen(),
    )
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "serialized-screen")
    view = terminal.attach_view()
    assert terminal.offer_output(session_id, b"one parser turn").accepted is True

    with ThreadPoolExecutor(max_workers=2) as executor:
        parsing = executor.submit(terminal.process_output, session_id, visible=True)
        assert parser_entered.wait(1)
        snapshot = executor.submit(terminal.view_state, view)

        assert concurrent_snapshot.wait(0.1) is False
        release_parser.set()
        assert parsing.result(timeout=1) is not None
        assert snapshot.result(timeout=1) is not None

    assert concurrent_snapshot.is_set() is False


def test_stale_or_detached_view_tokens_cannot_mutate_or_repaint() -> None:
    cleanup_started = Event()
    finish_cleanup = Event()

    def cleanup(attempt: CleanupAttempt) -> CleanupProof:
        cleanup_started.set()
        assert finish_cleanup.wait(1)
        return CleanupProof(True, True, True)

    backends = [RecordingBackend(on_cleanup=cleanup), RecordingBackend()]
    terminal = TerminalSessionManager(lambda: True, lambda: backends.pop(0))
    terminal.arm(acknowledge_disclosure=True)
    first_id = create_running_session(terminal, "first")
    second_id = create_running_session(terminal, "second")
    stale = terminal.attach_view()
    current = terminal.attach_view()

    assert terminal.view_state(stale) is None
    assert terminal.resize_session(first_id, columns=100, rows=30, view=stale) is False
    assert terminal.focus_session(second_id, view=stale) is False
    assert terminal.close_session(first_id, view=stale) is None
    assert terminal.selected_session_id == first_id
    assert terminal.projection(first_id).lifecycle is TerminalLifecycle.RUNNING  # type: ignore[union-attr]

    assert terminal.resize_session(first_id, columns=100, rows=30, view=current)
    assert terminal.focus_session(second_id, view=current)
    assert terminal.selected_session_id == second_id
    receipt = terminal.close_session(first_id, view=current)
    assert receipt is not None
    assert cleanup_started.wait(1)
    assert terminal.projection(first_id).lifecycle is TerminalLifecycle.CLOSING  # type: ignore[union-attr]

    terminal.detach_view(current)
    assert terminal.view_state(current) is None
    assert (
        terminal.resize_session(second_id, columns=90, rows=25, view=current) is False
    )
    assert terminal.focus_session(first_id, view=current) is False
    finish_cleanup.set()
    assert terminal.wait_for_cleanup(first_id, timeout_seconds=1)


@pytest.mark.asyncio
async def test_resize_is_coalesced_and_applied_outside_the_view_callback() -> None:
    backend = RecordingBackend()
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "coalesced-resize")
    view = terminal.attach_view()

    assert terminal.resize_session(session_id, columns=90, rows=25, view=view)
    assert terminal.resize_session(session_id, columns=100, rows=30, view=view)
    assert backend.resizes == []

    assert await terminal.apply_pending_resize(session_id, view=view) is True
    assert backend.resizes == [(100, 30)]
    state = terminal.view_state(view)
    assert state is not None
    assert (state.sessions[0].columns, state.sessions[0].rows) == (100, 30)


@pytest.mark.asyncio
async def test_concurrent_resize_workers_never_enter_backend_out_of_order() -> None:
    first_resize_started = Event()
    second_resize_started = Event()
    release_first_resize = Event()
    call_lock = Lock()

    class OrderedResizeBackend(RecordingBackend):
        def resize(self, columns: int, rows: int) -> None:
            with call_lock:
                call_number = len(self.resizes) + 1
                self.resizes.append((columns, rows))
            if call_number == 1:
                first_resize_started.set()
                assert release_first_resize.wait(1)
            else:
                second_resize_started.set()

    backend = OrderedResizeBackend()
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "ordered-resize")
    view = terminal.attach_view()
    assert terminal.resize_session(session_id, columns=90, rows=25, view=view)

    first = asyncio.create_task(terminal.apply_pending_resize(session_id, view=view))
    assert await asyncio.to_thread(first_resize_started.wait, 1)
    assert terminal.resize_session(session_id, columns=100, rows=30, view=view)
    second = asyncio.create_task(terminal.apply_pending_resize(session_id, view=view))

    second_entered_while_first_blocked = await asyncio.to_thread(
        second_resize_started.wait,
        0.2,
    )
    release_first_resize.set()

    assert second_entered_while_first_blocked is False
    assert await first is True
    assert await second is True
    assert backend.resizes == [(90, 25), (100, 30)]


@pytest.mark.asyncio
async def test_model_resize_failure_starts_cleanup_instead_of_leaving_split_state() -> (
    None
):
    class ResizeFailingScreen:
        failure_reason = None

        def feed(self, data: bytes) -> None:
            del data

        def take_pending_replies(self) -> tuple[bytes, ...]:
            return ()

        def resize(self, *, columns: int, rows: int) -> None:
            del columns, rows
            raise RuntimeError("model resize failed")

    backend = RecordingBackend(cleanup_proof=CleanupProof())
    terminal = TerminalSessionManager(
        lambda: True,
        lambda: backend,
        screen_model_factory=lambda _columns, _rows: ResizeFailingScreen(),
    )
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "failed-model-resize")
    view = terminal.attach_view()
    assert terminal.resize_session(session_id, columns=100, rows=30, view=view)

    assert await terminal.apply_pending_resize(session_id, view=view) is False
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)

    retained = terminal.projection(session_id)
    assert backend.resizes == [(100, 30)]
    assert len(backend.cleanup_attempts) == 1
    assert retained is not None
    assert retained.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN


@pytest.mark.asyncio
async def test_blocked_backend_resize_cannot_delay_disarm_cleanup() -> None:
    resize_started = Event()
    release_resize = Event()
    cleanup_started = Event()

    class BlockingResizeBackend(RecordingBackend):
        def resize(self, columns: int, rows: int) -> None:
            del columns, rows
            resize_started.set()
            assert release_resize.wait(1)

        def cleanup(self, attempt: CleanupAttempt) -> CleanupProof:
            del attempt
            cleanup_started.set()
            return CleanupProof(True, True, True)

    terminal = TerminalSessionManager(lambda: True, BlockingResizeBackend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "blocking-resize")
    view = terminal.attach_view()
    assert terminal.resize_session(session_id, columns=100, rows=30, view=view)

    resize_task = asyncio.create_task(
        terminal.apply_pending_resize(session_id, view=view)
    )
    assert await asyncio.to_thread(resize_started.wait, 1)
    disarm_task = asyncio.create_task(asyncio.to_thread(terminal.disarm))
    cleanup_was_not_blocked = await asyncio.to_thread(cleanup_started.wait, 0.2)

    release_resize.set()
    await disarm_task
    assert await resize_task is False
    assert cleanup_was_not_blocked is True


def test_stale_view_cannot_retry_retained_cleanup() -> None:
    backend = RecordingBackend(cleanup_proof=CleanupProof())
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "stale-retry")
    stale = terminal.attach_view()

    assert terminal.close_session(session_id, view=stale) is not None
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
    assert (
        terminal.projection(session_id).lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
    )  # type: ignore[union-attr]
    current = terminal.attach_view()

    with pytest.raises(TypeError):
        terminal.retry_cleanup(session_id)  # type: ignore[call-arg]
    assert terminal.retry_cleanup(session_id, view=stale) is None
    assert len(backend.cleanup_attempts) == 1
    assert terminal.retry_cleanup(session_id, view=current) is not None
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
    assert len(backend.cleanup_attempts) == 2


def test_destroy_and_remount_preserves_backend_identity_while_output_continues() -> (
    None
):
    backend = RecordingBackend()
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "remount")
    first_view = terminal.attach_view()
    terminal.offer_output(session_id, b"before ")
    terminal.process_output(session_id, visible=True)
    first_state = terminal.view_state(first_view)
    terminal.detach_view(first_view)

    terminal.offer_output(session_id, b"after")
    terminal.process_output(session_id, visible=False)
    second_view = terminal.attach_view()
    second_state = terminal.view_state(second_view)

    assert first_state is not None
    assert second_state is not None
    assert first_state.sessions[0].projection.session_id == session_id
    assert second_state.sessions[0].projection.session_id == session_id
    assert second_state.sessions[0].screen.lines[0].text.startswith("before after")
    assert len(backend.started) == 1
    assert backend.started[0][1].token == session_id


def test_healthy_output_drain_remains_available_during_cleanup() -> None:
    cleanup_started = Event()
    finish_cleanup = Event()

    def cleanup(attempt: CleanupAttempt) -> CleanupProof:
        cleanup_started.set()
        assert finish_cleanup.wait(1)
        return CleanupProof(True, True, True)

    backend = RecordingBackend(on_cleanup=cleanup)
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "healthy-drain")

    terminal.disarm()
    assert cleanup_started.wait(1)
    assert terminal.offer_output(session_id, b"final bytes").accepted is True
    turn = terminal.process_output(session_id, visible=False)
    assert turn is not None
    assert turn.processed_bytes == len(b"final bytes")
    finish_cleanup.set()
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)


@pytest.mark.asyncio
async def test_shutdown_returns_at_its_wait_bound_when_cleanup_is_still_running() -> (
    None
):
    cleanup_started = Event()
    finish_cleanup = Event()

    def cleanup(attempt: CleanupAttempt) -> CleanupProof:
        cleanup_started.set()
        assert finish_cleanup.wait(1)
        return CleanupProof(True, True, True)

    backend = RecordingBackend(on_cleanup=cleanup)
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "bounded-shutdown")

    try:
        async with asyncio.timeout(0.2):
            assert await terminal.shutdown(deadline_seconds=0.01) is False
        assert cleanup_started.wait(1)
        assert terminal.armed is False
        assert terminal.projection(session_id) is not None
    finally:
        finish_cleanup.set()
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)


def test_managed_process_inventory_is_test_only_and_content_free() -> None:
    class InventoryBackend(RecordingBackend):
        def managed_process_inventory_for_tests(
            self,
        ) -> tuple[ManagedProcessIdentity, ...]:
            return (
                ManagedProcessIdentity(pid=101, birth_identity="birth-a"),
                ManagedProcessIdentity(pid=202, birth_identity="birth-b"),
            )

    terminal = TerminalSessionManager(lambda: True, InventoryBackend)
    terminal.arm(acknowledge_disclosure=True)
    create_running_session(terminal, "inventory")
    view = terminal.attach_view()

    inventory = terminal.managed_process_inventory_for_tests()
    state = terminal.view_state(view)

    assert inventory == (
        ManagedProcessIdentity(pid=101, birth_identity="birth-a"),
        ManagedProcessIdentity(pid=202, birth_identity="birth-b"),
    )
    assert ManagedProcessIdentity.__slots__ == ("pid", "birth_identity")
    assert state is not None
    assert not hasattr(state, "managed_processes")
    assert all(not hasattr(session, "managed_processes") for session in state.sessions)


def test_rename_is_normalized_unique_and_generation_scoped() -> None:
    terminal = TerminalSessionManager(lambda: True, RecordingBackend)
    terminal.arm(acknowledge_disclosure=True)
    first_id = create_running_session(terminal, "First")
    second_id = create_running_session(terminal, "Second")
    stale = terminal.attach_view()
    current = terminal.attach_view()

    assert terminal.rename_session(second_id, "Renamed", view=stale) is False
    assert terminal.rename_session(second_id, "  FIRST  ", view=current) is False
    assert terminal.rename_session(second_id, "  Résumé  ", view=current) is True

    assert terminal.projection(first_id).name == "First"  # type: ignore[union-attr]
    assert terminal.projection(second_id).name == "Résumé"  # type: ignore[union-attr]


def test_view_input_uses_the_owned_bounded_actor_and_stale_calls_are_ignored() -> None:
    terminal = TerminalSessionManager(lambda: True, RecordingBackend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "input")
    stale = terminal.attach_view()
    current = terminal.attach_view()

    assert terminal.send_key(session_id, b"a", view=stale).accepted is False
    assert terminal.send_key(session_id, b"a", view=current).accepted is True
    assert (
        terminal.send_paste(
            session_id,
            "line\n",
            bracketed=False,
            view=current,
        ).accepted
        is True
    )

    key = terminal.take_input(session_id)
    paste = terminal.take_input(session_id)
    assert key is not None
    assert key.kind is InputEventKind.KEY
    assert key.data == b"a"
    assert paste is not None
    assert paste.kind is InputEventKind.PASTE
    assert paste.data == b"line\n"
    assert terminal.take_input(session_id) is None

    terminal.disarm()
    assert terminal.send_key(session_id, b"b", view=current).accepted is False


def test_change_subscriptions_are_content_free_and_unsubscribable() -> None:
    notifications: list[str] = []
    terminal = TerminalSessionManager(lambda: True, RecordingBackend)
    subscription = terminal.subscribe(lambda: notifications.append("changed"))
    assert isinstance(subscription, TerminalSubscriptionToken)

    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "subscription")
    terminal.offer_output(session_id, b"output")
    terminal.process_output(session_id, visible=True)

    assert notifications
    before_unsubscribe = len(notifications)
    assert terminal.unsubscribe(subscription) is True
    view = terminal.attach_view()
    assert terminal.rename_session(session_id, "renamed", view=view) is True
    assert len(notifications) == before_unsubscribe


def test_explicit_retry_notifies_before_cleanup_settles() -> None:
    retry_started = Event()
    finish_retry = Event()
    attempts = 0

    def cleanup(attempt: CleanupAttempt) -> CleanupProof:
        nonlocal attempts
        del attempt
        attempts += 1
        if attempts == 1:
            return CleanupProof()
        retry_started.set()
        assert finish_retry.wait(1)
        return CleanupProof(True, True, True)

    backend = RecordingBackend(on_cleanup=cleanup)
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "retry-notification")
    terminal.disarm()
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
    notifications: list[str] = []
    terminal.subscribe(lambda: notifications.append("changed"))
    view = terminal.attach_view()

    receipt = terminal.retry_cleanup(session_id, view=view)

    assert receipt is not None
    assert retry_started.wait(1)
    assert terminal.projection(session_id).lifecycle is TerminalLifecycle.CLOSING  # type: ignore[union-attr]
    assert notifications == ["changed"]
    finish_retry.set()
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)


def test_disarm_during_backend_factory_cancels_before_backend_start() -> None:
    factory_started = Event()
    release_factory = Event()
    backend = RecordingBackend()

    def backend_factory() -> RecordingBackend:
        factory_started.set()
        assert release_factory.wait(1)
        return backend

    terminal = TerminalSessionManager(lambda: True, backend_factory)
    terminal.arm(acknowledge_disclosure=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        creating = executor.submit(
            terminal.create_session,
            launch_request("cancel-factory"),
        )
        assert factory_started.wait(1)
        terminal.disarm()
        release_factory.set()
        result = creating.result(timeout=1)

    assert result.admitted is False
    assert terminal.armed is False
    assert terminal.projections() == ()
    assert backend.started == []


def test_disarm_during_backend_start_never_resurrects_a_running_record() -> None:
    start_entered = Event()
    release_start = Event()
    cleanup_started = Event()
    release_cleanup = Event()

    def on_start(
        request: TerminalLaunchRequest,
        admission: AdmissionGate,
    ) -> None:
        del request, admission
        start_entered.set()
        assert release_start.wait(1)

    def on_cleanup(attempt: CleanupAttempt) -> CleanupProof:
        del attempt
        cleanup_started.set()
        assert release_cleanup.wait(1)
        return CleanupProof(True, True, True)

    backend = RecordingBackend(on_start=on_start, on_cleanup=on_cleanup)
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        creating = executor.submit(
            terminal.create_session,
            launch_request("cancel-start"),
        )
        assert start_entered.wait(1)
        terminal.disarm()
        assert cleanup_started.wait(0.2) is False
        release_start.set()
        result = creating.result(timeout=1)

    assert result.admitted is False
    assert cleanup_started.wait(1)
    assert all(
        projection.lifecycle is not TerminalLifecycle.RUNNING
        for projection in terminal.projections()
    )
    release_cleanup.set()
    assert terminal.wait_for_cleanup(
        backend.started[0][1].token,
        timeout_seconds=1,
    )
    assert terminal.projections() == ()


def test_start_failure_after_disarm_cannot_erase_cleanup_uncertainty() -> None:
    start_entered = Event()
    release_start = Event()
    cleanup_finished = Event()

    def on_start(
        request: TerminalLaunchRequest,
        admission: AdmissionGate,
    ) -> None:
        del request, admission
        start_entered.set()
        assert release_start.wait(1)
        raise RuntimeError("launch failed after cleanup began")

    def on_cleanup(attempt: CleanupAttempt) -> CleanupProof:
        del attempt
        cleanup_finished.set()
        return CleanupProof()

    backend = RecordingBackend(on_start=on_start, on_cleanup=on_cleanup)
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        creating = executor.submit(
            terminal.create_session,
            launch_request("failed-start-cleanup"),
        )
        assert start_entered.wait(1)
        terminal.disarm()
        assert cleanup_finished.wait(0.2) is False
        release_start.set()
        result = creating.result(timeout=1)

    assert result == TerminalCreateResult(reason=TerminalReason.SPAWN_FAILED)
    assert cleanup_finished.wait(1)
    assert len(terminal.projections()) == 1
    retained = terminal.projections()[0]
    assert retained.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
    assert retained.reason is TerminalReason.CLEANUP_UNPROVEN


@pytest.mark.parametrize("mismatched_identity", [False, True])
def test_attempted_start_failure_runs_cleanup_before_releasing_ownership(
    mismatched_identity: bool,
) -> None:
    cleanup_finished = Event()

    class FailedStartBackend(RecordingBackend):
        def start(
            self,
            request: TerminalLaunchRequest,
            admission: AdmissionGate,
        ) -> BackendIdentity:
            self.started.append((request, admission))
            if mismatched_identity:
                return BackendIdentity(session_id="wrong-session")
            raise RuntimeError("failed after partial spawn")

        def cleanup(self, attempt: CleanupAttempt) -> CleanupProof:
            self.cleanup_attempts.append(attempt)
            cleanup_finished.set()
            return CleanupProof(True, True, True)

    backend = FailedStartBackend()
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)

    result = terminal.create_session(launch_request("failed-start"))

    assert result == TerminalCreateResult(reason=TerminalReason.SPAWN_FAILED)
    assert cleanup_finished.wait(1)
    session_id = backend.started[0][1].token
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
    assert len(backend.cleanup_attempts) == 1
    assert terminal.projections() == ()


def test_shell_exit_without_eof_retains_cleanup_unproven_not_exited() -> None:
    backend = RecordingBackend(cleanup_proof=CleanupProof(True, False, False))
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "missing-eof")

    terminal.shell_exited(session_id, exit_code=0)

    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
    retained = terminal.projection(session_id)
    assert retained is not None
    assert retained.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
    assert retained.stream_closed is False
    assert retained.output_complete is False


def test_eof_settlement_drains_admitted_output_and_finalizes_the_screen() -> None:
    backend = RecordingBackend(cleanup_proof=CleanupProof(True, True, True))
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "final-output")
    view = terminal.attach_view()
    assert terminal.offer_output(session_id, b"final output").accepted

    terminal.shell_exited(session_id, exit_code=0)
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)

    exited = terminal.projection(session_id)
    assert exited is not None
    assert exited.lifecycle is TerminalLifecycle.EXITED
    assert exited.stream_closed is True
    assert exited.output_complete is True
    state = terminal.view_state(view)
    assert state is not None
    assert state.sessions[0].screen.lines[0].text.startswith("final output")


def test_eof_finalizer_failure_fails_closed_instead_of_claiming_completeness() -> None:
    backend = RecordingBackend(cleanup_proof=CleanupProof(True, True, True))

    class FailingFinalizerScreen:
        failure_reason = None

        def feed(self, data: bytes) -> None:
            del data

        def take_pending_replies(self) -> tuple[bytes, ...]:
            return ()

        def finish(self) -> None:
            raise RuntimeError("decoder finalization failed")

        def resize(self, *, columns: int, rows: int) -> None:
            del columns, rows

    terminal = TerminalSessionManager(
        lambda: True,
        lambda: backend,
        screen_model_factory=lambda _columns, _rows: FailingFinalizerScreen(),
    )
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "failing-finalizer")
    assert terminal.offer_output(session_id, b"partial").accepted

    terminal.shell_exited(session_id, exit_code=0)
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)

    assert terminal.projection(session_id) is None
    assert backend.priority_close_requests == 1


def test_output_is_refused_after_stream_closure_is_proven() -> None:
    backend = RecordingBackend(cleanup_proof=CleanupProof(True, True, True))
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "closed-stream")

    terminal.shell_exited(session_id, exit_code=0)
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)

    exited = terminal.projection(session_id)
    assert exited is not None
    assert exited.stream_closed is True
    assert terminal.offer_output(session_id, b"late bytes").accepted is False


def test_offer_delayed_after_manager_check_is_refused_by_actor_eof_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offer_reached_actor = Event()
    release_offer = Event()
    original_offer = TerminalOutputActor.offer_output

    def delayed_offer(actor: TerminalOutputActor, data: bytes):
        offer_reached_actor.set()
        assert release_offer.wait(1)
        return original_offer(actor, data)

    monkeypatch.setattr(TerminalOutputActor, "offer_output", delayed_offer)
    backend = RecordingBackend(cleanup_proof=CleanupProof(True, True, True))
    terminal = TerminalSessionManager(lambda: True, lambda: backend)
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "eof-offer-race")

    with ThreadPoolExecutor(max_workers=1) as executor:
        offering = executor.submit(terminal.offer_output, session_id, b"late bytes")
        assert offer_reached_actor.wait(1)
        terminal.shell_exited(session_id, exit_code=0)
        assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)
        release_offer.set()
        result = offering.result(timeout=1)

    assert result.accepted is False
    exited = terminal.projection(session_id)
    assert exited is not None
    assert exited.stream_closed is True
    assert exited.output_complete is True


def test_view_snapshot_failure_is_contained_and_closes_the_session() -> None:
    backend = RecordingBackend(cleanup_proof=CleanupProof(False, False, False))

    class SnapshotFailingScreen:
        failure_reason = None

        def feed(self, data: bytes) -> None:
            del data

        def resize(self, *, columns: int, rows: int) -> None:
            del columns, rows

        def snapshot(self) -> TerminalScreenSnapshot:
            raise RuntimeError("private screen state")

        def take_pending_replies(self) -> tuple[bytes, ...]:
            return ()

    terminal = TerminalSessionManager(
        lambda: True,
        lambda: backend,
        screen_model_factory=lambda _columns, _rows: SnapshotFailingScreen(),
    )
    terminal.arm(acknowledge_disclosure=True)
    session_id = create_running_session(terminal, "snapshot-failure")
    view = terminal.attach_view()

    assert terminal.view_state(view) is None
    assert terminal.wait_for_cleanup(session_id, timeout_seconds=1)

    retained = terminal.projection(session_id)
    assert retained is not None
    assert retained.parser_failed is True
    assert retained.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
    assert backend.priority_close_requests == 1
