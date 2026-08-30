from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

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
    CleanupSchedule,
    TerminalEvent,
    TerminalLifecycle,
    TerminalReason,
    apply_event,
    join_cleanup,
    retry_cleanup,
    running_projection,
    slot_held,
    validate_transition,
)


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


def test_shell_exit_drains_and_nonzero_exit_is_ordinary() -> None:
    projection = apply_event(
        running_projection(), TerminalEvent("shell_exit", exit_code=23)
    )
    assert projection.lifecycle is TerminalLifecycle.DRAINING
    assert projection.exit_code == 23
    assert projection.reason is None


def test_parser_failure_has_content_free_reason() -> None:
    projection = apply_event(
        running_projection(),
        TerminalEvent("parser_failure", reason=TerminalReason.TERMINAL_PROTOCOL_FAILED),
    )
    assert projection.reason is TerminalReason.TERMINAL_PROTOCOL_FAILED
    assert projection.output_complete is False


def test_cleanup_closes_only_with_proof() -> None:
    closing = replace(running_projection(), lifecycle=TerminalLifecycle.CLOSING)
    assert (
        apply_event(closing, TerminalEvent("cleanup_proven")).lifecycle
        is TerminalLifecycle.CLOSED
    )
    assert (
        apply_event(closing, TerminalEvent("cleanup_failed")).lifecycle
        is TerminalLifecycle.CLEANUP_UNPROVEN
    )


def test_retry_is_the_only_event_that_creates_a_new_cleanup_t0() -> None:
    from tldw_chatbook.Terminal.contracts import CleanupAttempt, TerminalReceipt

    receipt = TerminalReceipt(CleanupAttempt(10.0), "close")
    assert join_cleanup(receipt, 20.0).attempt.t0 == 10.0
    assert retry_cleanup(receipt, 20.0).attempt.t0 == 20.0


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (TerminalLifecycle.CLOSED, TerminalLifecycle.RUNNING),
        (TerminalLifecycle.RUNNING, TerminalLifecycle.CLOSED),
        (TerminalLifecycle.RESERVED, TerminalLifecycle.RUNNING),
        (TerminalLifecycle.DRAINING, TerminalLifecycle.CREATING),
    ],
)
def test_forbidden_lifecycle_transitions_are_rejected(current, target) -> None:
    assert not validate_transition(current, target)
