"""Fixed mixed-head presentation values; original decision owners stay local."""

from __future__ import annotations

from types import MappingProxyType

from .console_chat_controller import ConsolePendingDecisionProjection
from .console_execution_approval_values import (
    _arguments,
    decode_approval_view,
    encode_approval_view,
)
from .console_execution_projection_values import (
    _boolean,
    _count,
    _duration,
    _enum,
    _fail,
    _number,
    _optional,
    _text,
)


def _list(codec):
    def transform(value, decode):
        if type(value) is not list:
            _fail("expected original decision array")
        return [codec(item, decode) for item in value]

    return transform


def _fields(value, schema, required, decode, *, excluded=frozenset()):
    containers = (dict,) if decode else (dict, MappingProxyType)
    if (
        type(value) not in containers
        or any(type(key) is not str for key in value)
        or not required <= value.keys()
        or not value.keys() <= schema.keys() | excluded
    ):
        _fail("unexpected or missing decision fields")
    return {
        key: schema[key](item, decode) for key, item in value.items() if key in schema
    }


def _call_arguments(value, decode):
    result = _arguments(value, decode)
    # Original approval calls carry a dict; the separate view keeps its original
    # top-level mapping proxy. Both use the same strict embedded-JSON contract.
    return dict(result) if decode else result


_CALL = {
    "llm_name": _text,
    "server_key": _text,
    "tool_name": _text,
    "server_label": _text,
    "arguments": _call_arguments,
    "reason": _text,
    "options": _list(_text),
    "effects": _list(_text),
    "execution_policy": _enum("bounded_abandonable", "definitive_after_start"),
    "path_precheck_failed": _boolean,
    "call_id": _text,
    "full_command": _text,
    "warning": _text,
    "scope_notice": _text,
    "rationale": _text,
    "description": _text,
}


def _call(value, decode):
    return _fields(value, _CALL, _CALL.keys(), decode)


def _view(value, decode):
    return decode_approval_view(value) if decode else encode_approval_view(value)


_COMMON = {
    "session_id": _text,
    "run_id": _text,
    "timeout_seconds": _duration,
    "phase": _enum("approval", "finishing"),
}
_SCHEMAS = {
    "approval": {
        **_COMMON,
        "calls": _list(_call),
        "round_id": _text,
        "summary": _optional(_text),
        "view": _optional(_view),
        "presentation_revision": _optional(_count),
    },
    "skill_install": {**_COMMON, "request_id": _text, "url": _text},
    "skill_script": {
        **_COMMON,
        "request_id": _text,
        "skill_name": _text,
        "script_path": _text,
        "mechanism": _text,
        "interpreter": _text,
        "is_binary": _boolean,
        "args": _list(_text),
    },
    "hook_review": {
        "session_id": _text,
        "timeout_seconds": _duration,
        "phase": _enum("approval", "finishing"),
        "generation": _count,
        "waiting_for_send": _boolean,
        "busy": _boolean,
    },
}
_REQUIRED = {
    "approval": frozenset(("calls", "round_id", "timeout_seconds")),
    "skill_install": frozenset(("request_id", "url", "timeout_seconds")),
    "skill_script": frozenset(
        (
            "request_id",
            "skill_name",
            "script_path",
            "mechanism",
            "interpreter",
            "is_binary",
            "args",
            "timeout_seconds",
        )
    ),
    "hook_review": frozenset(
        (
            "generation",
            "waiting_for_send",
            "busy",
            "timeout_seconds",
        )
    ),
}
_LOCAL_FIELDS = frozenset(
    (
        "_decision_type",
        "_decision_id",
        "_decision_order",
        "deadline_monotonic",
    )
)
_OUTER_FIELDS = frozenset(
    (
        "decision_type",
        "session_id",
        "decision_id",
        "remaining_active_seconds",
        "payload",
    )
)


def _payload(kind, session_id, decision_id, value, decode):
    excluded = frozenset()
    if not decode:
        excluded = _LOCAL_FIELDS
        if kind == "approval":
            excluded |= {"presentation_state"}
        elif kind == "hook_review":
            excluded |= {"snapshot"}
    result = _fields(value, _SCHEMAS[kind], _REQUIRED[kind], decode, excluded=excluded)
    for key, expected in (
        ("session_id", session_id),
        ("round_id", decision_id),
        ("request_id", decision_id),
    ):
        if key in result and result[key] != expected:
            _fail("decision presentation identity mismatch")
    if not decode:
        for key, expected in (("_decision_type", kind), ("_decision_id", decision_id)):
            if key in value and _text(value[key], False) != expected:
                _fail("original mixed-head identity mismatch")
        if "_decision_order" in value:
            _count(value["_decision_order"], False)
        if "deadline_monotonic" in value:
            _optional(_number)(value["deadline_monotonic"], False)
        if (
            "presentation_state" in value
            and type(value["presentation_state"]) is not dict
        ):
            _fail("expected original local remount draft")
        if "snapshot" in value:
            from ..Agents.hook_permissions import HookReviewSnapshot

            if type(value["snapshot"]) is not HookReviewSnapshot:
                _fail("expected original local hook-review snapshot")
            # Only routing facts cross here. The actual modal snapshot and its
            # claim/release/actions require the separate parent checkpoint.
    return result


def encode_pending_decision(
    value: ConsolePendingDecisionProjection | None,
) -> dict | None:
    """Detach the original mixed FIFO head without moving its owner.

    Args:
        value: Original five-field pending projection, or no head.

    Returns:
        Fixed presentation values with owner-local state explicitly excluded.

    Raises:
        ConsoleProtocolError: The variant, identity or value shape is unsupported.
    """
    if value is None:
        return None
    if type(value) is not ConsolePendingDecisionProjection:
        _fail("expected original pending-decision projection")
    kind = _enum(*_SCHEMAS)(value.decision_type, False)
    session_id = _text(value.session_id, False)
    decision_id = _text(value.decision_id, False)
    return {
        "decision_type": kind,
        "session_id": session_id,
        "decision_id": decision_id,
        "remaining_active_seconds": _optional(_duration)(
            value.remaining_active_seconds, False
        ),
        "payload": _payload(kind, session_id, decision_id, value.payload, False),
    }


def decode_pending_decision(value: object) -> ConsolePendingDecisionProjection | None:
    """Restore an inert original head without clocks, remount drafts or authority.

    Args:
        value: Plain JSON projection produced by the fixed decision codec.

    Returns:
        Original display-only pending projection, or no head.

    Raises:
        ConsoleProtocolError: The variant, identity or value shape is unsupported.
    """
    if value is None:
        return None
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or value.keys() != _OUTER_FIELDS
    ):
        _fail("unexpected or missing pending-head fields")
    kind = _enum(*_SCHEMAS)(value["decision_type"], True)
    session_id = _text(value["session_id"], True)
    decision_id = _text(value["decision_id"], True)
    return ConsolePendingDecisionProjection(
        decision_type=kind,
        session_id=session_id,
        decision_id=decision_id,
        remaining_active_seconds=_optional(_duration)(
            value["remaining_active_seconds"], True
        ),
        payload=_payload(kind, session_id, decision_id, value["payload"], True),
    )
