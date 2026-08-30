from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace

import pytest

from tldw_chatbook.Terminal.backend import TerminalBackend
from tldw_chatbook.Terminal.contracts import (
    MAX_IO_CHUNK_BYTES,
    MAX_PARSER_TURN_BYTES,
    MAX_PARSER_TURN_SECONDS,
    MAX_PASTE_BYTES,
    MAX_PENDING_INPUT_BYTES,
    MAX_PENDING_OUTPUT_BYTES,
    MAX_SCROLLBACK_BYTES,
    MAX_SCROLLBACK_LINES,
    MAX_SESSION_RECORDS,
    MAX_COLUMNS,
    MAX_ROWS,
    MIN_COLUMNS,
    MIN_ROWS,
    AdmissionGate,
    BackendIdentity,
    CleanupAttempt,
    CleanupProof,
    CleanupSchedule,
    TerminalEvent,
    TerminalLaunchRequest,
    TerminalLifecycle,
    TerminalProjection,
    TerminalReason,
    TerminalReceipt,
    apply_event,
    join_cleanup,
    retry_cleanup,
    slot_held,
    validate_transition,
)


def running_projection() -> TerminalProjection:
    return TerminalProjection(lifecycle=TerminalLifecycle.RUNNING)


def test_terminal_limits_match_adr_099() -> None:
    assert MAX_SESSION_RECORDS == 4
    assert (MIN_COLUMNS, MAX_COLUMNS) == (5, 300)
    assert (MIN_ROWS, MAX_ROWS) == (2, 120)
    assert MAX_SCROLLBACK_LINES == 5_000
    assert MAX_SCROLLBACK_BYTES == 4 * 1024 * 1024
    assert MAX_PENDING_INPUT_BYTES == 512 * 1024
    assert MAX_PENDING_OUTPUT_BYTES == 512 * 1024
    assert MAX_PASTE_BYTES == 256 * 1024
    assert MAX_IO_CHUNK_BYTES == 64 * 1024
    assert MAX_PARSER_TURN_BYTES == 256 * 1024
    assert MAX_PARSER_TURN_SECONDS == 0.008
    schedule = CleanupSchedule()
    assert schedule.deadline_seconds == 5.0
    assert schedule.hangup_no_later_than == 0.75
    assert schedule.terminate_no_later_than == 2.25
    assert schedule.force_kill_no_later_than == 3.75
    assert schedule.proof_reserve_seconds == 1.25


def test_lifecycle_and_terminal_reason_vocabularies_match_the_design() -> None:
    assert {item.value for item in TerminalLifecycle} == {
        "reserved",
        "creating",
        "admitting",
        "running",
        "draining",
        "exited",
        "closing",
        "closed",
        "cleanup_unproven",
    }
    assert {item.value for item in TerminalReason} == {
        "locked",
        "unarmed",
        "session_limit",
        "invalid_name",
        "invalid_start_directory",
        "shell_unavailable",
        "backend_unavailable",
        "admission_failed",
        "spawn_failed",
        "input_backpressure",
        "terminal_protocol_failed",
        "io_failed",
        "worker_failed",
        "output_incomplete",
        "cleanup_unproven",
    }


@pytest.mark.parametrize(
    "value_type",
    [
        CleanupSchedule,
        TerminalLaunchRequest,
        AdmissionGate,
        BackendIdentity,
        CleanupAttempt,
        CleanupProof,
        TerminalEvent,
        TerminalProjection,
        TerminalReceipt,
    ],
)
def test_terminal_value_contracts_are_frozen_and_slotted(value_type: type) -> None:
    assert value_type.__dataclass_params__.frozen is True
    assert isinstance(value_type.__slots__, tuple)
    assert "__dict__" not in value_type.__slots__


def test_terminal_backend_protocol_signatures_match_the_brief() -> None:
    parameter = inspect.Parameter
    positional = parameter.POSITIONAL_OR_KEYWORD
    expected = {
        "start": inspect.Signature(
            [
                parameter("self", positional),
                parameter("request", positional, annotation=TerminalLaunchRequest),
                parameter("admission", positional, annotation=AdmissionGate),
            ],
            return_annotation=BackendIdentity,
        ),
        "write": inspect.Signature(
            [
                parameter("self", positional),
                parameter("data", positional, annotation=bytes),
            ],
            return_annotation=None,
        ),
        "resize": inspect.Signature(
            [
                parameter("self", positional),
                parameter("columns", positional, annotation=int),
                parameter("rows", positional, annotation=int),
            ],
            return_annotation=None,
        ),
        "request_priority_close": inspect.Signature(
            [parameter("self", positional)], return_annotation=None
        ),
        "cleanup": inspect.Signature(
            [
                parameter("self", positional),
                parameter("attempt", positional, annotation=CleanupAttempt),
            ],
            return_annotation=CleanupProof,
        ),
    }

    assert {
        name: inspect.signature(getattr(TerminalBackend, name)) for name in expected
    } == expected


def test_exited_does_not_claim_stream_or_output_completion() -> None:
    projection = replace(
        running_projection(),
        lifecycle=TerminalLifecycle.EXITED,
        exit_code=0,
        stream_closed=False,
        output_complete=False,
    )
    assert projection.exit_code == 0
    assert projection.stream_closed is False
    assert projection.output_complete is False


def test_contract_skeleton_is_immutable_and_transitions_are_not_placeholders() -> None:
    projection = running_projection()
    with pytest.raises(FrozenInstanceError):
        projection.lifecycle = TerminalLifecycle.CLOSING  # type: ignore[misc]
    assert validate_transition(TerminalLifecycle.RUNNING, TerminalLifecycle.DRAINING)


def test_launch_failure_releases_reservation() -> None:
    assert slot_held(TerminalLifecycle.CREATING)
    assert not slot_held(TerminalLifecycle.CLOSED)
    assert validate_transition(TerminalLifecycle.CREATING, TerminalLifecycle.CLOSED)


def test_admission_failure_closes_and_releases_reservation() -> None:
    admitting = replace(running_projection(), lifecycle=TerminalLifecycle.ADMITTING)

    projection = apply_event(admitting, TerminalEvent("admission_failure"))

    assert projection.lifecycle is TerminalLifecycle.CLOSED
    assert projection.reason is TerminalReason.ADMISSION_FAILED
    assert not slot_held(projection.lifecycle)


@pytest.mark.parametrize(
    ("event_kind", "authorized_source", "authorized_reason"),
    [
        pytest.param(
            "admission_failure",
            TerminalLifecycle.ADMITTING,
            TerminalReason.ADMISSION_FAILED,
            id="admission-failure",
        ),
        pytest.param(
            "cleanup_proven",
            TerminalLifecycle.CLOSING,
            TerminalReason.WORKER_FAILED,
            id="cleanup-proven",
        ),
    ],
)
@pytest.mark.parametrize(
    "source", list(TerminalLifecycle), ids=lambda source: source.value
)
def test_apply_event_requires_an_event_authorized_source(
    event_kind: str,
    authorized_source: TerminalLifecycle,
    authorized_reason: TerminalReason,
    source: TerminalLifecycle,
) -> None:
    original = TerminalProjection(
        session_id="session-1",
        name="shell",
        lifecycle=source,
        reason=TerminalReason.WORKER_FAILED,
        exit_code=7,
        stream_closed=True,
        output_complete=True,
    )

    cleanup_proof = (
        CleanupProof(
            process_dead=True,
            stream_closed=True,
            output_complete=True,
        )
        if event_kind == "cleanup_proven"
        else None
    )
    projection = apply_event(
        original,
        TerminalEvent(event_kind, cleanup_proof=cleanup_proof),
    )

    if source is authorized_source:
        assert projection == replace(
            original,
            lifecycle=TerminalLifecycle.CLOSED,
            reason=authorized_reason,
        )
    else:
        assert projection == original


def test_shell_exit_drains_and_nonzero_exit_is_ordinary() -> None:
    projection = apply_event(
        running_projection(), TerminalEvent("shell_exit", exit_code=23)
    )
    assert projection.lifecycle is TerminalLifecycle.DRAINING
    assert projection.exit_code == 23
    assert projection.reason is None


def test_shell_exit_records_code_after_cleanup_already_started() -> None:
    closing = replace(running_projection(), lifecycle=TerminalLifecycle.CLOSING)

    projection = apply_event(closing, TerminalEvent("shell_exit", exit_code=17))

    assert projection.lifecycle is TerminalLifecycle.CLOSING
    assert projection.exit_code == 17


@pytest.mark.parametrize(
    "supplied_reason",
    [None, TerminalReason.IO_FAILED],
    ids=["no-reason", "mismatched-reason"],
)
def test_parser_failure_always_has_terminal_protocol_reason(
    supplied_reason: TerminalReason | None,
) -> None:
    projection = apply_event(
        running_projection(),
        TerminalEvent("parser_failure", reason=supplied_reason),
    )
    assert projection.reason is TerminalReason.TERMINAL_PROTOCOL_FAILED
    assert projection.output_complete is False


@pytest.mark.parametrize(
    "lifecycle",
    [TerminalLifecycle.CLEANUP_UNPROVEN, TerminalLifecycle.CLOSED],
)
def test_late_parser_failure_cannot_rewrite_terminal_outcome(
    lifecycle: TerminalLifecycle,
) -> None:
    original = TerminalProjection(
        lifecycle=lifecycle,
        reason=TerminalReason.CLEANUP_UNPROVEN,
        stream_closed=True,
        output_complete=False,
    )

    assert apply_event(original, TerminalEvent("parser_failure")) == original


def test_output_completion_cannot_reverse_parser_failure() -> None:
    parser_failed = apply_event(running_projection(), TerminalEvent("parser_failure"))

    projection = apply_event(parser_failed, TerminalEvent("output_complete"))

    assert projection.output_complete is False


def test_cleanup_closes_only_with_proof() -> None:
    closing = replace(running_projection(), lifecycle=TerminalLifecycle.CLOSING)
    projection = apply_event(
        closing,
        TerminalEvent(
            "cleanup_proven",
            cleanup_proof=CleanupProof(
                process_dead=True,
                stream_closed=True,
                output_complete=False,
            ),
        ),
    )

    assert projection.lifecycle is TerminalLifecycle.CLOSED
    assert projection.stream_closed is True
    assert projection.output_complete is False
    assert (
        apply_event(closing, TerminalEvent("cleanup_failed")).lifecycle
        is TerminalLifecycle.CLEANUP_UNPROVEN
    )


def test_cleanup_proven_event_without_backend_proof_does_not_close() -> None:
    closing = replace(running_projection(), lifecycle=TerminalLifecycle.CLOSING)

    assert apply_event(closing, TerminalEvent("cleanup_proven")) == closing


@pytest.mark.parametrize(
    "proof",
    [
        CleanupProof(process_dead=False, stream_closed=True),
        CleanupProof(process_dead=True, stream_closed=False),
    ],
)
def test_incomplete_backend_proof_does_not_close(proof: CleanupProof) -> None:
    closing = replace(running_projection(), lifecycle=TerminalLifecycle.CLOSING)

    projection = apply_event(
        closing,
        TerminalEvent("cleanup_proven", cleanup_proof=proof),
    )

    assert projection == closing


@pytest.mark.parametrize(
    "event",
    [TerminalEvent("close"), TerminalEvent("parser_failure")],
    ids=["close", "parser-failure"],
)
def test_ordinary_events_cannot_retry_cleanup_unproven(event: TerminalEvent) -> None:
    retained = replace(
        running_projection(), lifecycle=TerminalLifecycle.CLEANUP_UNPROVEN
    )

    projection = apply_event(retained, event)

    assert projection.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN


def test_retry_cleanup_atomically_reenters_closing_with_a_fresh_t0() -> None:
    retained = replace(
        running_projection(), lifecycle=TerminalLifecycle.CLEANUP_UNPROVEN
    )

    result = retry_cleanup(retained, 20.0)

    assert result == (
        replace(retained, lifecycle=TerminalLifecycle.CLOSING),
        TerminalReceipt(CleanupAttempt(20.0), "retry"),
    )


def test_retry_cleanup_refuses_a_lifecycle_without_retained_authority() -> None:
    with pytest.raises(ValueError, match="cleanup_unproven"):
        retry_cleanup(running_projection(), 20.0)


def test_join_cleanup_retains_the_existing_attempt_t0() -> None:
    receipt = TerminalReceipt(CleanupAttempt(10.0), "close")

    assert join_cleanup(receipt, 20.0) is receipt
    assert join_cleanup(receipt, 20.0).attempt.t0 == 10.0


def test_join_cleanup_adopts_an_earlier_global_t0() -> None:
    receipt = TerminalReceipt(CleanupAttempt(20.0), "close")

    assert join_cleanup(receipt, 10.0).attempt.t0 == 10.0


def test_terminal_event_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="terminal event kind"):
        TerminalEvent("cleanup_typo")


def test_lifecycle_transition_matrix_matches_the_approved_design() -> None:
    allowed_pairs = {
        (TerminalLifecycle.RESERVED, TerminalLifecycle.CREATING),
        (TerminalLifecycle.RESERVED, TerminalLifecycle.CLOSED),
        (TerminalLifecycle.CREATING, TerminalLifecycle.ADMITTING),
        (TerminalLifecycle.CREATING, TerminalLifecycle.CLOSED),
        (TerminalLifecycle.ADMITTING, TerminalLifecycle.RUNNING),
        (TerminalLifecycle.ADMITTING, TerminalLifecycle.CLOSED),
        (TerminalLifecycle.RUNNING, TerminalLifecycle.DRAINING),
        (TerminalLifecycle.RUNNING, TerminalLifecycle.CLOSING),
        (TerminalLifecycle.DRAINING, TerminalLifecycle.EXITED),
        (TerminalLifecycle.DRAINING, TerminalLifecycle.CLOSING),
        (TerminalLifecycle.EXITED, TerminalLifecycle.CLOSING),
        (TerminalLifecycle.CLOSING, TerminalLifecycle.CLOSED),
        (TerminalLifecycle.CLOSING, TerminalLifecycle.CLEANUP_UNPROVEN),
        (TerminalLifecycle.CLEANUP_UNPROVEN, TerminalLifecycle.CLOSING),
    }

    for current in TerminalLifecycle:
        for target in TerminalLifecycle:
            assert validate_transition(current, target) is (
                (current, target) in allowed_pairs
            ), f"unexpected transition result for {current.value} -> {target.value}"
