"""Behavioral tests for the bounded terminal protocol pre-parser."""

from __future__ import annotations

import logging

import pytest

from tldw_chatbook.Terminal.protocol_gate import TerminalProtocolGate


def _feed(gate: TerminalProtocolGate, *chunks: bytes) -> bytes:
    return b"".join(gate.feed(chunk) for chunk in chunks)


def test_plain_text_and_admitted_csi_recover_across_chunk_boundaries() -> None:
    gate = TerminalProtocolGate()

    admitted = _feed(gate, b"before\x1b[3", b"1mafter")

    assert admitted == b"before\x1b[31mafter"
    assert gate.snapshot().buffered_bytes == 0


@pytest.mark.parametrize(
    ("sequence", "admitted"),
    [
        (b"\x1b[" + b"1;" * 31 + b"1m", True),
        (b"\x1b[" + b"1;" * 32 + b"1m", False),
        (b"\x1b[9999m", True),
        (b"\x1b[10000m", False),
        (b"\x1b[00001m", False),
    ],
)
def test_csi_parameter_and_private_intermediate_caps(
    sequence: bytes, admitted: bool
) -> None:
    gate = TerminalProtocolGate()

    output = _feed(gate, sequence, b"tail")

    assert output == (sequence if admitted else b"") + b"tail"
    assert gate.snapshot().rejected_sequences == (0 if admitted else 1)


@pytest.mark.parametrize(
    ("private_bytes", "rejected"),
    [(16, False), (17, True)],
)
def test_csi_private_intermediate_cap_precedes_unknown_sequence_filtering(
    private_bytes: int, rejected: bool
) -> None:
    gate = TerminalProtocolGate()
    sequence = b"\x1b[" + b"?" * private_bytes + b"h"

    output = _feed(gate, sequence, b"tail")

    assert output == b"tail"
    assert gate.snapshot().rejected_sequences == int(rejected)
    assert gate.snapshot().ignored_sequences == int(not rejected)


def test_csi_limit_crossing_discards_until_the_late_final_byte() -> None:
    gate = TerminalProtocolGate()

    first = gate.feed(b"\x1b[" + b"1;" * 33)
    status = gate.snapshot()
    second = gate.feed(b"mrecovered")

    assert first == b""
    assert status.discarding is True
    assert status.buffered_bytes == 0
    assert second == b"recovered"
    assert gate.snapshot().discarding is False


@pytest.mark.parametrize(
    ("intermediate_count", "rejected"),
    [(14, False), (15, True)],
)
def test_non_csi_escape_sequence_has_a_sixteen_byte_total_cap(
    intermediate_count: int, rejected: bool
) -> None:
    gate = TerminalProtocolGate()
    sequence = b"\x1b" + b" " * intermediate_count + b"7"

    output = _feed(gate, sequence, b"tail")

    # The bounded sequence is unsupported and therefore ignored. The
    # distinction under test is whether it crossed the safety cap.
    assert output == b"tail"
    assert gate.snapshot().rejected_sequences == int(rejected)


@pytest.mark.parametrize("introducer", [b"\x1b]", b"\x1bP", b"\x1b^", b"\x1b_"])
@pytest.mark.parametrize("terminator", [b"\x07", b"\x1b\\", b"\x18", b"\x1a", b"\x9c"])
def test_string_controls_discard_payload_and_recover_on_late_terminator(
    introducer: bytes,
    terminator: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate = TerminalProtocolGate()
    secret = b"QODO_SECRET_PAYLOAD"

    with caplog.at_level(logging.DEBUG):
        before = gate.feed(b"before" + introducer + secret)
        during = gate.snapshot()
        after = gate.feed(terminator + b"after")

    assert before == b"before"
    assert during.buffered_bytes <= 4096
    assert after == b"after"
    assert secret.decode() not in repr(during)
    assert secret.decode() not in caplog.text


def test_incomplete_string_retains_only_classification_state() -> None:
    gate = TerminalProtocolGate()
    secret = b"LIVE_OBJECT_SECRET"

    gate.feed(b"\x1b]52;c;" + secret)

    assert gate.snapshot().buffered_bytes == 0
    assert secret not in bytes(gate._buffer)


@pytest.mark.parametrize("introducer", [b"\x1b]", b"\x1bP", b"\x1b^", b"\x1b_"])
def test_oversized_string_control_retains_no_payload_while_discarding(
    introducer: bytes,
) -> None:
    gate = TerminalProtocolGate()

    assert gate.feed(introducer + b"S" * 4096) == b""
    status = gate.snapshot()
    recovered = gate.feed(b"\x1b\\safe")

    assert status.discarding is True
    assert status.buffered_bytes == 0
    assert "SSSS" not in repr(status)
    assert recovered == b"safe"
    assert gate.snapshot().rejected_sequences == 1


def test_parser_reset_terminates_string_discard_and_is_admitted() -> None:
    gate = TerminalProtocolGate()

    output = _feed(gate, b"\x1b]discarded", b"\x1bc", b"safe")

    assert output == b"\x1bcsafe"
    assert gate.snapshot().buffered_bytes == 0


def test_finish_discards_an_incomplete_sequence_without_exposing_it() -> None:
    gate = TerminalProtocolGate()
    secret = b"INCOMPLETE_SECRET"

    assert gate.feed(b"\x1b]" + secret) == b""
    status = gate.finish()

    assert status.buffered_bytes == 0
    assert status.discarding is False
    assert status.rejected_sequences == 1
    assert secret.decode() not in repr(status)


def test_unknown_controls_are_ignored_without_hiding_following_text() -> None:
    gate = TerminalProtocolGate()

    output = _feed(gate, b"a\x1b[1;2z", b"b\x1b X", b"c")

    assert output == b"abc"
    assert gate.snapshot().ignored_sequences == 2


@pytest.mark.parametrize(
    "sequence",
    [
        b"\x1b[1;2A",
        b"\x1b[?5n",
        b"\x1b[1;2c",
        b"\x1b[>1m",
        b"\x1b[1 q",
    ],
)
def test_unsupported_csi_argument_shapes_are_ignored(sequence: bytes) -> None:
    gate = TerminalProtocolGate()

    output = _feed(gate, b"before", sequence, b"after")

    assert output == b"beforeafter"
    assert gate.snapshot().ignored_sequences == 1


def test_raw_c1_bytes_never_enter_the_admitted_stream() -> None:
    gate = TerminalProtocolGate()

    output = gate.feed(b"before\x9b31mafter")

    assert output == b"beforeafter"
    assert gate.snapshot().ignored_sequences == 1


def test_raw_c1_string_and_st_are_discarded_without_exposing_payload() -> None:
    gate = TerminalProtocolGate()

    output = gate.feed(b"before\x9dsecret\x9cafter")

    assert output == b"beforeafter"


def test_utf8_continuation_equal_to_c1_st_does_not_end_a_string() -> None:
    gate = TerminalProtocolGate()

    output = gate.feed(b"before\x1b]discard-\xe2\x9c\x93\x07after")

    assert output == b"beforeafter"


def test_raw_st_after_incomplete_utf8_terminates_a_bounded_string() -> None:
    gate = TerminalProtocolGate()

    output = gate.feed(b"before\x1b]discard-\xe2\x9cafter")

    assert output == b"beforeafter"


def test_utf8_between_escape_and_backslash_does_not_form_string_terminator() -> None:
    gate = TerminalProtocolGate()

    output = gate.feed(b"before\x1b]discard\x1b\xe2\x9c\x93\\still-discarded\x07after")

    assert output == b"beforeafter"


def test_discard_mode_requires_adjacent_escape_and_backslash_terminator() -> None:
    gate = TerminalProtocolGate()

    first = gate.feed(b"\x1b]" + b"S" * 4096)
    second = gate.feed(b"\x1b\xe2\x9c\x93\\still-discarded\x07after")

    assert first == b""
    assert second == b"after"


def test_raw_st_after_incomplete_utf8_at_string_cap_recovers_safe_text() -> None:
    gate = TerminalProtocolGate()

    before = gate.feed(b"\x1b]" + b"S" * 4094 + b"\xe2")
    after = gate.feed(b"\x9csafe")

    assert before == b""
    assert after == b"safe"
    assert gate.snapshot().discarding is False


def test_valid_utf8_crossing_string_cap_does_not_act_as_raw_st() -> None:
    gate = TerminalProtocolGate()

    before = gate.feed(b"\x1b]" + b"S" * 4094 + "✓".encode())
    after = gate.feed(b"still-discarded\x07safe")

    assert before == b""
    assert after == b"safe"
