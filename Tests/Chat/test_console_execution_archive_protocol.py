"""Archive checkpoints carry only exact original archive metadata."""

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_execution_protocol import (
    ConsoleProtocolError,
    ConsoleWireMessage,
    WireIdentity,
    decode_message,
    encode_message,
)


def test_archive_checkpoint_and_ack_are_closed_and_phase_bound():
    identity = WireIdentity(
        "profile", "service", 1, "received", "session", "incarnation", 2
    )
    body = {
        "turn_id": "turn",
        "ordinal": 1,
        "stage": "prepare",
        "conversation_id": "conversation",
        "archived": None,
    }
    for archived in (None, False, True):
        finish = dict(body, ordinal=2, stage="finish", archived=archived)
        message = ConsoleWireMessage(
            identity, "archive_checkpoint", "ReceiveTurn", finish
        )
        assert decode_message(encode_message(message)) == message
    for invalid in (
        dict(body, ordinal=2),
        dict(body, archived=False),
        dict(body, ordinal=True),
        dict(body, body="not permitted"),
        dict(body, conversation_id=""),
    ):
        with pytest.raises(ConsoleProtocolError):
            ConsoleWireMessage(identity, "archive_checkpoint", "ReceiveTurn", invalid)
    with pytest.raises(ConsoleProtocolError):
        ConsoleWireMessage(
            replace(identity, session_id="", incarnation_id=""),
            "archive_checkpoint",
            "ReceiveTurn",
            body,
        )
    ack = {
        "received_request_id": "received",
        "turn_id": "turn",
        "ordinal": 1,
        "stage": "prepare",
        "conversation_id": "conversation",
        "generation": 0,
        "refusal": None,
    }
    for refusal in (None, "archive_inflight"):
        message = ConsoleWireMessage(
            identity, "command", "ArchiveSendCheckpointAck", dict(ack, refusal=refusal)
        )
        assert decode_message(encode_message(message)) == message
    for invalid in (
        dict(ack, generation=True),
        dict(ack, generation=-1),
        dict(ack, refusal="archived"),
        dict(ack, refusal="arbitrary text"),
        dict(ack, ordinal=2),
        dict(ack, authority=True),
    ):
        with pytest.raises(ConsoleProtocolError):
            ConsoleWireMessage(identity, "command", "ArchiveSendCheckpointAck", invalid)


def test_archive_checkpoint_ack_has_separate_applied_and_retired_outcomes():
    identity = WireIdentity("profile", "service", 1, "ack", "session", "incarnation", 2)
    applied = ConsoleWireMessage(
        identity, "applied", "ArchiveSendCheckpointAck", {"applied": True}
    )
    retired = ConsoleWireMessage(
        identity,
        "retired",
        "ArchiveSendCheckpointAck",
        {"operation_id": "ack", "retired": True},
    )
    assert decode_message(encode_message(applied)) == applied
    assert decode_message(encode_message(retired)) == retired
    with pytest.raises(ConsoleProtocolError):
        ConsoleWireMessage(
            identity,
            "applied",
            "ArchiveSendCheckpointAck",
            {"applied": True, "retired": True},
        )
