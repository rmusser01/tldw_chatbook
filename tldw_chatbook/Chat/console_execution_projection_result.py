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


def projection_key(value: ConsoleProjection) -> tuple[int, str, str, int, int | None]:
    """Return the original source and transcript invalidation facts."""
    return (
        value.source_revision,
        value.session_id,
        value.incarnation_id,
        value.display_revision,
        value.marker_revision,
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
    body = {
        "source_revision": value.source_revision,
        "session_id": value.session_id,
        "incarnation_id": value.incarnation_id,
        "display_revision": value.display_revision,
        "marker_revision": value.marker_revision,
        "messages": messages,
        "run_state": encode_run_projection(value.run_state),
        "decision": encode_pending_decision(value.decision),
    }
    if value.run_history is not None:
        body["run_history"] = list(value.run_history)
    return body


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
    if (type(value) is not dict or not fields <= set(value)
        or set(value) - fields - {"session_view", "marker_revision", "run_history"}):
        raise ConsoleProtocolError("invalid_projection_fields")
    if "run_history" in value:
        from .console_backend_run_history import validate_run_history

        validate_run_history(value["run_history"])
    marker = value.get("marker_revision")
    if marker is not None and (type(marker) is not int or marker < 0):
        raise ConsoleProtocolError("invalid_projection_marker_revision")
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
    from .console_execution_protocol import _PROJECTED_INPUTS, _detach

    value = payload["session_view"]
    fields = {"sessions", "active_session_id", "fingerprints", "pressed_inputs"}
    if (
        type(value) is not dict
        or set(value) not in (fields, fields | {"readiness", "composition", "workspace", "session_surface"})
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
    path = "payload.session_view.pressed_inputs"
    _PROJECTED_INPUTS(_detach(pressed, path, 4, set()), path)
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
    if "readiness" in value:
        _readiness(value["readiness"], current_row)
        from .console_execution_composition_values import decode_session_composition

        if active is None:
            if value["composition"] is not None:
                raise ConsoleProtocolError("projection_composition_without_active_session")
        else:
            decode_session_composition(
                value["composition"], next(row for row in rows if row.id == active)
            )
        from .console_execution_workspace_values import decode_workspace_surface
        from .console_execution_projection_view import ConsoleProjectionView
        from .console_execution_session_surface import decode_session_surface_projection

        surface = decode_workspace_surface(value["workspace"])
        if tuple(item.session_id for item in surface.sessions) != tuple(row.id for row in rows):
            raise ConsoleProtocolError("workspace_projection_sessions_changed")
        # This validation-only row view explicitly contains no transcript frame.
        row_view = ConsoleProjectionView(
            payload["source_revision"], rows, active, tuple(fingerprints), (),
        )
        decode_session_surface_projection(row_view, value["session_surface"])
    return rows, active, tuple(fingerprints)


def _readiness(value, row) -> tuple:
    import math
    from dataclasses import replace
    from .console_execution_projection_values import _settings
    from .console_execution_readiness_values import decode_settings_readiness
    from .console_execution_context_summary_values import decode_context_summary

    readiness_fields = {
        "effective_settings", "state", "provider_label", "control_provider_label", "model",
        "remaining_seconds", "captured_at_monotonic", "context_summary", "agent_rail", "prompt_history",
    }
    if (
        type(value) is not dict
        or set(value) not in (
            readiness_fields, readiness_fields | {"send_selection"},
            readiness_fields | {"send_selection", "project_head"},
        )
        or type(value["provider_label"]) is not str
        or type(value["control_provider_label"]) is not str
        or value["model"] is not None and type(value["model"]) is not str
        or row.settings is None
    ):
        raise ConsoleProtocolError("invalid_projection_readiness")
    for name in ("remaining_seconds", "captured_at_monotonic"):
        number = value[name]
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ConsoleProtocolError("invalid_projection_readiness_clock")
    if value["remaining_seconds"] > 1.0:
        raise ConsoleProtocolError("invalid_projection_readiness_clock")
    effective = _settings(value["effective_settings"], True)
    readiness = decode_settings_readiness(value["state"])
    if (
        effective.model != value["model"]
        or replace(row.settings, model=effective.model, base_url=effective.base_url)
        != effective
    ):
        raise ConsoleProtocolError("projection_readiness_settings_changed")
    from .console_execution_agent_rail_values import decode_agent_rail_projection
    from .console_execution_prompt_history import decode_prompt_history
    from .console_execution_protocol import _SELECTION, _detach

    send_selection = value.get("send_selection")
    if "send_selection" in value:
        _SELECTION(_detach(send_selection, "send_selection", 0, set()), "send_selection")

    result = (
        effective, readiness, value["provider_label"], value["model"],
        value["remaining_seconds"], value["captured_at_monotonic"],
        value["control_provider_label"], decode_context_summary(value["context_summary"]),
        decode_agent_rail_projection(value["agent_rail"]),
        decode_prompt_history(value["prompt_history"]),
        send_selection,
    )
    if "project_head" not in value:
        return result
    from .console_execution_project_values import decode_project_head

    head = decode_project_head(value["project_head"])
    if head is not None and (
        head.session_id != row.id or head.incarnation_id != row.incarnation_id
    ):
        raise ConsoleProtocolError("project_head_session_changed")
    return (*result, head)


def decode_projection_readiness(payload: dict, view: ConsoleProjectionView) -> tuple:
    """Decode only readiness captured with this exact session publication."""
    row = view.session(payload["session_id"])
    body = payload.get("session_view")
    if type(body) is not dict or "readiness" not in body:
        raise ConsoleProtocolError("projection_readiness_missing")
    return _readiness(body["readiness"], row)


def encode_pressed_inputs(
    snapshot: ConsoleSessionInputSnapshot, session: ConsoleChatSession
) -> dict:
    """Detach the original coherent input snapshot without transferring its proof."""
    import base64
    from .attachment_core import PendingAttachment
    from .console_received_intent import ConsoleSessionInputSnapshot
    from .console_chat_store import ConsoleChatSession
    from .console_execution_projection_values import _settings
    from .console_execution_protocol import _PROJECTED_INPUTS, _detach

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
    path = "payload.session_view.pressed_inputs"
    _PROJECTED_INPUTS(_detach(body, path, 4, set()), path)
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
        if type(previous) is not ConsoleProjection or projection_key(previous) != (*key, value.get("marker_revision")):
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
        value.get("marker_revision"),
        tuple(value["run_history"]) if "run_history" in value else None,
    )
