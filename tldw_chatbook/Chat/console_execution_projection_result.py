"""Closed values for the original ConsoleProjection, with exact snapshot reuse.

This is the existing seven-field projection, not an attached-screen model or
authority. Session rails, ancillary display and named parent checkpoints have
their own capture boundaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .console_execution_protocol import ConsoleProtocolError

if TYPE_CHECKING:
    from .console_control_dispatch import ConsoleProjection
    from .console_chat_store import ConsoleChatSession
    from .console_received_intent import ConsoleSessionInputSnapshot
    from .console_execution_projection_view import ConsoleProjectionView


def projection_key(value: ConsoleProjection) -> tuple[int, str, str, int]:
    """Return the original source and transcript invalidation facts."""
    return (
        value.source_revision,
        value.session_id,
        value.incarnation_id,
        value.display_revision,
    )


def encode_projection_payload(
    value: ConsoleProjection, *, messages: list[dict] | None
) -> dict:
    """Encode fresh run/decision values beside full or referenced messages."""
    from .console_control_dispatch import ConsoleProjection
    from .console_execution_decision_values import encode_pending_decision
    from .console_execution_projection_values import encode_run_projection

    if type(value) is not ConsoleProjection:
        raise ConsoleProtocolError("invalid_original_projection")
    return {
        "source_revision": value.source_revision,
        "session_id": value.session_id,
        "incarnation_id": value.incarnation_id,
        "display_revision": value.display_revision,
        "messages": messages,
        "run_state": encode_run_projection(value.run_state),
        "decision": encode_pending_decision(value.decision),
    }


def _shape(value: object) -> None:
    fields = {
        "source_revision",
        "session_id",
        "incarnation_id",
        "display_revision",
        "messages",
        "run_state",
        "decision",
    }
    if type(value) is not dict or set(value) not in (fields, fields | {"session_view"}):
        raise ConsoleProtocolError("invalid_projection_fields")
    for name in ("source_revision", "display_revision"):
        if type(value[name]) is not int or value[name] < 0:
            raise ConsoleProtocolError("invalid_projection_revision")
    for name in ("session_id", "incarnation_id"):
        if type(value[name]) is not str or not value[name]:
            raise ConsoleProtocolError("invalid_projection_session")
    messages = value["messages"]
    if messages is not None and type(messages) is not list:
        raise ConsoleProtocolError("invalid_projection_messages")


def validate_projection_payload(value: object) -> None:
    """Check finite values without constructing transient transcript messages."""
    from .console_execution_decision_values import decode_pending_decision
    from .console_execution_projection_values import (
        validate_message_projection,
        decode_run_projection,
    )

    _shape(value)
    messages = value["messages"]
    if messages is not None:
        for message in messages:
            validate_message_projection(message)
    decode_run_projection(value["run_state"])
    decision = decode_pending_decision(value["decision"])
    if decision is not None and decision.session_id != value["session_id"]:
        raise ConsoleProtocolError("projection_decision_session_changed")
    if "session_view" in value:
        _session_view(value)


def _session_view(payload: dict) -> tuple:
    from .console_execution_projection_values import (
        decode_session_projection,
        _settings,
    )
    from .console_execution_protocol import _PROJECTED_INPUTS

    value = payload["session_view"]
    if (
        type(value) is not dict
        or set(value)
        != {"sessions", "active_session_id", "fingerprints", "pressed_inputs"}
        or type(value["sessions"]) is not list
        or type(value["fingerprints"]) is not list
    ):
        raise ConsoleProtocolError("invalid_projection_view_fields")
    rows = tuple(decode_session_projection(row) for row in value["sessions"])
    identities = {}
    for row in rows:
        if not row.id or not row.incarnation_id or row.id in identities:
            raise ConsoleProtocolError("invalid_projection_view_session")
        identities[row.id] = row.incarnation_id
    active = value["active_session_id"]
    if active is not None and (type(active) is not str or active not in identities):
        raise ConsoleProtocolError("invalid_projection_view_active")
    fingerprints = []
    seen = set()
    for item in value["fingerprints"]:
        if type(item) is not dict or set(item) != {
            "session_id",
            "count",
            "last_content_length",
        }:
            raise ConsoleProtocolError("invalid_projection_view_fingerprint")
        session_id = item["session_id"]
        if (
            type(session_id) is not str
            or session_id not in identities
            or session_id in seen
        ):
            raise ConsoleProtocolError("invalid_projection_view_fingerprint")
        numbers = (item["count"], item["last_content_length"])
        if any(type(number) is not int or number < 0 for number in numbers):
            raise ConsoleProtocolError("invalid_projection_view_fingerprint")
        seen.add(session_id)
        fingerprints.append((session_id, numbers))
    if (
        seen != identities.keys()
        or identities.get(payload["session_id"]) != payload["incarnation_id"]
    ):
        raise ConsoleProtocolError("projection_view_session_changed")
    pressed = value["pressed_inputs"]
    _PROJECTED_INPUTS(pressed, "payload.session_view.pressed_inputs")
    if (
        pressed["session_id"] != payload["session_id"]
        or pressed["incarnation_id"] != payload["incarnation_id"]
        or [item["attachment_id"] for item in pressed["attachments"]]
        != pressed["attachment_ids"]
    ):
        raise ConsoleProtocolError("projection_pressed_inputs_changed")
    current_row = next(row for row in rows if row.id == payload["session_id"])
    if (
        pressed["draft"] != current_row.draft
        or pressed["settings_revision"] != current_row.settings_revision
        or pressed["settings"]
        != (
            None
            if current_row.settings is None
            else _settings(current_row.settings, False)
        )
    ):
        raise ConsoleProtocolError("projection_input_row_changed")
    return rows, active, tuple(fingerprints)


def encode_pressed_inputs(
    snapshot: ConsoleSessionInputSnapshot, session: ConsoleChatSession
) -> dict:
    """Detach the original coherent input snapshot without transferring its proof."""
    import base64
    from .attachment_core import PendingAttachment
    from .console_received_intent import ConsoleSessionInputSnapshot
    from .console_chat_store import ConsoleChatSession
    from .console_execution_projection_values import _settings
    from .console_execution_protocol import _PROJECTED_INPUTS

    if (
        type(snapshot) is not ConsoleSessionInputSnapshot
        or type(session) is not ConsoleChatSession
    ):
        raise ConsoleProtocolError("invalid_original_input_snapshot")
    if (
        snapshot.session_id != session.id
        or snapshot.incarnation_id != session.incarnation_id
    ):
        raise ConsoleProtocolError("projection_pressed_inputs_changed")
    attachments = []
    for item in snapshot.attachments:
        if type(item) is not PendingAttachment:
            raise ConsoleProtocolError("invalid_original_attachment")
        body = {
            name: getattr(item, name)
            for name in (
                "attachment_id",
                "file_path",
                "display_name",
                "file_type",
                "insert_mode",
                "mime_type",
                "text_content",
                "original_size",
                "processed_size",
            )
        }
        body["data"] = (
            None if item.data is None else base64.b64encode(item.data).decode("ascii")
        )
        attachments.append(body)
    body = {
        "draft": snapshot.draft,
        "authored_token": None
        if session._draft_authored_token is None
        else list(session._draft_authored_token),
        "settings": None
        if snapshot._settings is None
        else _settings(snapshot._settings, False),
        "settings_revision": snapshot.settings_revision,
        "attachment_revision": snapshot.attachment_revision,
        "attachment_ids": list(snapshot._attachment_ids),
        "attachments": attachments,
        "one_shot_prefill": snapshot.one_shot_prefill,
        "prefill_revision": snapshot.prefill_revision,
        "session_id": snapshot.session_id,
        "incarnation_id": snapshot.incarnation_id,
    }
    _PROJECTED_INPUTS(body, "payload.session_view.pressed_inputs")
    return body


def decode_projection_view(
    payload: dict, projection: ConsoleProjection
) -> ConsoleProjectionView:
    """Build the finite read-only view from this exact retired frame and rows."""
    from .console_execution_projection_view import ConsoleProjectionView

    if "session_view" not in payload:
        raise ConsoleProtocolError("projection_view_missing")
    rows, active, fingerprints = _session_view(payload)
    return ConsoleProjectionView(
        projection.source_revision, rows, active, fingerprints, (projection,)
    )


def decode_projection_payload(
    value: object, *, previous: ConsoleProjection | None = None
) -> ConsoleProjection:
    """Make disposable render values, requiring exact prior snapshot references."""
    from .console_control_dispatch import ConsoleProjection
    from .console_execution_decision_values import decode_pending_decision
    from .console_execution_projection_values import (
        decode_message_projection,
        decode_run_projection,
    )

    _shape(value)
    key = tuple(
        value[name]
        for name in (
            "source_revision",
            "session_id",
            "incarnation_id",
            "display_revision",
        )
    )
    if value["messages"] is None:
        if type(previous) is not ConsoleProjection or projection_key(previous) != key:
            raise ConsoleProtocolError("projection_snapshot_not_owned")
        messages = previous.messages
    else:
        messages = tuple(decode_message_projection(item) for item in value["messages"])
    decision = decode_pending_decision(value["decision"])
    if decision is not None and decision.session_id != value["session_id"]:
        raise ConsoleProtocolError("projection_decision_session_changed")
    return ConsoleProjection(
        *key,
        messages,
        decode_run_projection(value["run_state"]),
        decision,
    )
