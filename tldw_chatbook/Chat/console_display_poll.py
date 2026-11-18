"""Changed sections of an immutable, checked runtime display publication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .console_execution_protocol import (
    ConsoleProtocolError, ConsoleWireMessage, WireIdentity, _REFERENCE, _SEQUENCE, _detach,
)


_FIELDS = frozenset({"display_revision", "marker_revision", "messages", "run_state", "decision", "run_history"})
_SECTIONS = frozenset({
    "sessions", "active_session_id", "fingerprints", "pressed_inputs",
    "readiness", "workspace", "composition", "session_surface",
})
_READINESS = frozenset({
    "effective_settings", "state", "provider_label", "control_provider_label", "model",
    "remaining_seconds", "captured_at_monotonic", "context_summary", "agent_rail",
    "prompt_history", "send_selection", "project_head",
})


@dataclass(frozen=True, slots=True)
class DisplayPublication:
    """The complete checked result retained by one side of the runtime pipe."""

    attachment_id: str
    visit: int
    message: ConsoleWireMessage

    def matches(self, identity: WireIdentity, poll: Mapping) -> bool:
        prior = self.message.identity
        return (
            self.message.kind == "applied" and self.message.operation == "ReadProjection"
            and self.attachment_id == poll["attachment_id"] and self.visit == poll["visit"]
            and prior.request_id == poll["after"] and prior.request_id != identity.request_id
            and prior.sequence < identity.sequence
            and prior.profile_epoch == identity.profile_epoch
            and prior.service_epoch == identity.service_epoch
            and prior.source_revision == identity.source_revision
            and prior.session_id == identity.session_id and prior.incarnation_id == identity.incarnation_id
        )


def validate_delta(payload: Mapping) -> None:
    """Validate the delta envelope. Only an expanded, checked result can be applied."""
    required = {"after", "attachment_id", "visit", "fields", "sections"}
    if payload.keys() not in (required, required | {"readiness"}):
        raise ConsoleProtocolError("invalid_display_delta_fields")
    _REFERENCE(payload["after"], "payload.after")
    _REFERENCE(payload["attachment_id"], "payload.attachment_id")
    _SEQUENCE(payload["visit"], "payload.visit")
    for name, allowed in (("fields", _FIELDS), ("sections", _SECTIONS)):
        value = payload[name]
        if type(value) is not MappingProxyType or not value.keys() <= allowed:
            raise ConsoleProtocolError("invalid_display_delta_section")
    if "readiness" in payload and (
        type(payload["readiness"]) is not MappingProxyType
        or not payload["readiness"].keys() <= _READINESS or "readiness" in payload["sections"]
    ):
        raise ConsoleProtocolError("invalid_display_delta_readiness")


def _same_value(left, right) -> bool:
    # Wire equality preserves bool/int, int/float and negative zero.
    if left is right:
        return True
    kind = type(left)
    if kind is not type(right):
        return False
    if kind is MappingProxyType:
        return left.keys() == right.keys() and all(_same_value(v, right[k]) for k, v in left.items())
    if kind is tuple:
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right, strict=True))
    if kind is float:
        return left.hex() == right.hex()
    return left == right


def _thaw(value):
    return _detach(value, "payload", 2, set(), thaw=True)


def poll_response(
    current: ConsoleWireMessage, poll: Mapping | None, previous: DisplayPublication | None,
) -> ConsoleWireMessage:
    """Send a complete first poll, then send only changes to the acknowledged base."""
    if poll is None or previous is None or not previous.matches(current.identity, poll):
        return current
    before, after = previous.message.payload, current.payload
    if (
        before.keys() != after.keys() or "session_view" not in after
        or before["session_view"].keys() != after["session_view"].keys()
    ):
        return current
    fields = {
        key: _thaw(after[key]) for key in after.keys() & _FIELDS
        if not _same_value(before[key], after[key])
    }
    old_view, new_view = before["session_view"], after["session_view"]
    sections = {
        key: _thaw(value) for key, value in new_view.items()
        if key != "readiness" and not _same_value(old_view[key], value)
    }
    delta = {**poll, "fields": fields, "sections": sections}
    if "readiness" in new_view:
        old, new = old_view["readiness"], new_view["readiness"]
        if old.keys() != new.keys():
            sections["readiness"] = _thaw(new)
        else:
            changed = {key: _thaw(value) for key, value in new.items() if not _same_value(old[key], value)}
            if changed:
                delta["readiness"] = changed
    return ConsoleWireMessage(current.identity, "projection_delta", current.operation, delta)


def expand_projection_delta(
    delta: ConsoleWireMessage, poll: Mapping | None, previous: DisplayPublication | None,
) -> ConsoleWireMessage:
    """Recheck the complete display and its cross-section invariants before Applied."""
    if (
        previous is None or poll is None or not previous.matches(delta.identity, poll)
        or any(delta.payload[k] != poll[k] for k in ("after", "attachment_id", "visit"))
    ):
        raise ConsoleProtocolError("display_poll_base_changed")
    body = _thaw(previous.message.payload)
    if (
        not delta.payload["fields"].keys() <= body.keys() or "session_view" not in body
        or not delta.payload["sections"].keys() <= body["session_view"].keys()
    ):
        raise ConsoleProtocolError("display_poll_shape_changed")
    body.update(_thaw(delta.payload["fields"]))
    body["session_view"].update(_thaw(delta.payload["sections"]))
    if "readiness" in delta.payload:
        readiness = body["session_view"].get("readiness")
        if readiness is None or not delta.payload["readiness"].keys() <= readiness.keys():
            raise ConsoleProtocolError("display_poll_readiness_shape_changed")
        readiness.update(_thaw(delta.payload["readiness"]))
    return ConsoleWireMessage(delta.identity, "applied", "ReadProjection", body)
