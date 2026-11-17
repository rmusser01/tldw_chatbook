"""Exact projection references remain inert and bound to their original key."""

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_control_dispatch import ConsoleProjection
from tldw_chatbook.Chat.console_execution_projection_result import (
    decode_projection_payload,
    encode_projection_payload,
)
from tldw_chatbook.Chat.console_execution_projection_values import (
    encode_message_projection,
)
from tldw_chatbook.Chat.console_execution_protocol import (
    ConsoleProtocolError,
    ConsoleWireMessage,
    WireIdentity,
    decode_message,
    encode_message,
)


def test_reference_reuses_exact_transcript_and_refreshes_run_values():
    original = ConsoleProjection(
        3,
        "session",
        "incarnation",
        7,
        (ConsoleChatMessage(ConsoleMessageRole.USER, "original"),),
        ConsoleRunState(ConsoleRunStatus.IDLE, "Ready"),
        None,
    )
    body = encode_projection_payload(
        original, messages=[encode_message_projection(original.messages[0])]
    )
    identity = WireIdentity(
        "profile", "service", 1, "request", "session", "incarnation", 3
    )
    wire = decode_message(
        encode_message(ConsoleWireMessage(identity, "applied", "ReadProjection", body))
    )
    first = decode_projection_payload(encode_message(wire)["payload"])
    fresh = replace(
        original, run_state=ConsoleRunState(ConsoleRunStatus.IDLE, "Changed copy")
    )
    reference = encode_projection_payload(fresh, messages=None)
    second = decode_projection_payload(reference, previous=first)
    assert second.messages is first.messages
    assert second.run_state.visible_copy == "Changed copy"
    assert first.messages[0] is not original.messages[0]
    original.messages[0].content = "later"
    assert first.messages[0].content == "original"
    for previous in (
        None,
        replace(first, source_revision=4),
        replace(first, incarnation_id="replacement"),
        replace(first, display_revision=8),
    ):
        with pytest.raises(ConsoleProtocolError, match="snapshot_not_owned"):
            decode_projection_payload(reference, previous=previous)


def test_projection_envelope_and_nested_original_fields_remain_checked():
    original = ConsoleProjection(
        0,
        "session",
        "incarnation",
        0,
        (ConsoleChatMessage(ConsoleMessageRole.USER, "text"),),
        ConsoleRunState(ConsoleRunStatus.IDLE, "Ready"),
        None,
    )
    body = encode_projection_payload(
        original, messages=[encode_message_projection(original.messages[0])]
    )
    with pytest.raises(ConsoleProtocolError, match="identity changed"):
        ConsoleWireMessage(
            WireIdentity("profile", "service", 1, "request", "other", "incarnation"),
            "applied",
            "ReadProjection",
            body,
        )
    body["messages"][0]["sibling_count"] = 0
    with pytest.raises(ConsoleProtocolError, match="sibling"):
        ConsoleWireMessage(
            WireIdentity("profile", "service", 1, "request", "session", "incarnation"),
            "applied",
            "ReadProjection",
            body,
        )
