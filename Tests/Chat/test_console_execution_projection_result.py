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


def test_session_view_keeps_current_inputs_and_declines_missing_or_mixed_frames():
    from copy import deepcopy
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
    from tldw_chatbook.Chat.console_execution_projection_result import (
        decode_projection_view,
    )
    from tldw_chatbook.Chat.console_execution_projection_values import (
        encode_session_projection,
    )
    from tldw_chatbook.Chat.console_execution_projection_view import (
        ConsoleProjectionUnavailable,
    )

    session = ConsoleChatSession(id="current", draft="pending", draft_revision=3)
    other = ConsoleChatSession(id="other")
    frame = ConsoleProjection(
        0,
        session.id,
        session.incarnation_id,
        1,
        (ConsoleChatMessage(ConsoleMessageRole.USER, "saved"),),
        ConsoleRunState(ConsoleRunStatus.IDLE, "Ready"),
        None,
    )
    body = encode_projection_payload(
        frame, messages=[encode_message_projection(frame.messages[0])]
    )
    body["session_view"] = {
        "sessions": [encode_session_projection(row) for row in (session, other)],
        "active_session_id": session.id,
        "fingerprints": [
            {"session_id": session.id, "count": 1, "last_content_length": 5},
            {"session_id": other.id, "count": 0, "last_content_length": 0},
        ],
        "pressed_inputs": {
            "draft": session.draft,
            "authored_token": None,
            "settings": None,
            "settings_revision": 0,
            "attachment_revision": 0,
            "attachment_ids": [],
            "attachments": [],
            "one_shot_prefill": None,
            "prefill_revision": 0,
            "session_id": session.id,
            "incarnation_id": session.incarnation_id,
        },
    }
    identity = WireIdentity(
        "profile", "service", 1, "request", session.id, session.incarnation_id
    )
    wire = decode_message(
        encode_message(ConsoleWireMessage(identity, "applied", "ReadProjection", body))
    )
    decoded = encode_message(wire)["payload"]
    projection = decode_projection_payload(decoded)
    view = decode_projection_view(decoded, projection)
    assert view.session(session.id).draft_revision == 3
    assert view.session_fingerprint(session.id) == (1, 5)
    assert view.messages_for_session(session.id)[0].content == "saved"
    assert decoded["session_view"]["pressed_inputs"]["authored_token"] is None
    from tldw_chatbook.Chat.console_execution_protocol import _PRESSED_INPUTS

    with pytest.raises(ConsoleProtocolError):
        _PRESSED_INPUTS(decoded["session_view"]["pressed_inputs"], "pressed_inputs")
    with pytest.raises(ConsoleProjectionUnavailable):
        view.messages_for_session(other.id)
    for key, value in (
        ("draft", "later"),
        ("settings_revision", 1),
        ("incarnation_id", "replacement"),
    ):
        changed = deepcopy(body)
        changed["session_view"]["pressed_inputs"][key] = value
        with pytest.raises(ConsoleProtocolError):
            ConsoleWireMessage(identity, "applied", "ReadProjection", changed)
    changed = deepcopy(body)
    changed["session_view"]["sessions"].append(changed["session_view"]["sessions"][0])
    with pytest.raises(ConsoleProtocolError):
        ConsoleWireMessage(identity, "applied", "ReadProjection", changed)
    for key, value in (
        ("active_session_id", "unknown"),
        ("fingerprints", body["session_view"]["fingerprints"][:1]),
        ("fingerprints", body["session_view"]["fingerprints"] * 2),
    ):
        changed = deepcopy(body)
        changed["session_view"][key] = value
        with pytest.raises(ConsoleProtocolError):
            ConsoleWireMessage(identity, "applied", "ReadProjection", changed)
    assert view.session(session.id).draft == "pending"
