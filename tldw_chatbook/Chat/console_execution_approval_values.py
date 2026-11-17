"""Detached original approval-card values, never permission inputs or grants."""

from __future__ import annotations

import json
from types import MappingProxyType

from .approval_presentation import (
    ApprovalAuthority,
    ApprovalBatchView,
    ApprovalRowView,
    _DECISIONS,
    _MAX_ARGUMENT_DEPTH,
)
from .console_execution_projection_values import (
    _boolean,
    _count,
    _enum,
    _fail,
    _number,
    _optional,
    _record,
    _text,
    _tuple,
)
from .console_execution_protocol import _unique_pairs


def _json(value: object, decode: bool, depth: int, active: set[int]):
    # Original approval depth includes scalar leaves and starts at the root
    # argument mapping, independently of any eventual message envelope.
    if depth > _MAX_ARGUMENT_DEPTH:
        _fail("approval arguments exceed the original nesting limit")
    kind = type(value)
    if kind is str:
        return _text(value, decode)
    if value is None or kind in (bool, int):
        return value
    if kind is float:
        return _number(value, decode)
    mappings = (dict,) if decode else (dict, MappingProxyType)
    arrays = (list,) if decode else (list, tuple)
    if kind not in (*mappings, *arrays):
        _fail("expected captured JSON argument data")
    identity = id(value)
    if identity in active:
        _fail("cyclic approval argument value")
    active.add(identity)
    try:
        if kind in arrays:
            return [_json(item, decode, depth + 1, active) for item in value]
        return {
            _text(key, decode): _json(item, decode, depth + 1, active)
            for key, item in value.items()
        }
    finally:
        active.remove(identity)


def _arguments(value: object, decode: bool):
    if decode:
        _text(value, True)
        try:
            value = json.loads(value, object_pairs_hook=_unique_pairs)
        except (ValueError, RecursionError):
            _fail("invalid embedded approval arguments")
    expected = (dict,) if decode else (dict, MappingProxyType)
    if type(value) not in expected:
        _fail("expected captured argument mapping")
    copied = _json(value, decode, 1, set())
    # Preserve the original capture's top-level read-only mapping. Nested JSON
    # containers remain independent copies consumed by existing Details code.
    if decode:
        return MappingProxyType(copied)
    # The original argument depth is independent of the framed envelope depth.
    # Reversible JSON retains those values without adding wrapper nesting to it.
    try:
        return json.dumps(
            copied, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
    except (ValueError, TypeError, OverflowError, RecursionError):
        _fail("approval arguments cannot be encoded as JSON")


_AUTHORITY = {
    "provider_kind": _enum(
        "mcp", "builtin", "local", "virtual_cli", "raw_shell", "runtime"
    ),
    "profile_id": _optional(_text),
    "profile_label": _text,
    "location_label": _text,
    "grant_domain": _enum("profile", "console_chat", "none"),
    "stamp_domain": _enum("call", "tool_name", "shared_group"),
    "revocation_label": _text,
    "inherits_default": _boolean,
}


def _authority(value: object, decode: bool):
    return _record(value, ApprovalAuthority, _AUTHORITY, decode)


_ROW = {
    "verdict_key": _text,
    "call_count": _count,
    "action_label": _text,
    "targets": _tuple(_text),
    "authority": _authority,
    "legal_decisions": _tuple(_enum(*_DECISIONS)),
    "requires_review": _boolean,
    "withheld_scope_copy": _text,
    "argument_sets": _tuple(_arguments),
}


def _row(value: object, decode: bool):
    return _record(value, ApprovalRowView, _ROW, decode)


_VIEW = {
    "round_id": _text,
    "session_id": _text,
    "run_id": _text,
    "revision": _count,
    "rows": _tuple(_row),
    "call_count": _count,
    "bulk_once": _boolean,
    "bulk_deny": _boolean,
}


def encode_approval_view(view: ApprovalBatchView) -> dict:
    """Detach the exact original approval display into plain JSON values.

    Args:
        view: Original captured approval batch, including detailed arguments.

    Returns:
        An independent dictionary with one embedded JSON string per argument set.

    Raises:
        ConsoleProtocolError: A value is outside the declared display contract.
    """
    return _record(view, ApprovalBatchView, _VIEW, False)


def decode_approval_view(value: object) -> ApprovalBatchView:
    """Restore original display records from independently copied JSON values.

    Args:
        value: Plain JSON dictionary produced by the fixed approval-view codec.

    Returns:
        Original display-only records with no permission or dispatch authority.

    Raises:
        ConsoleProtocolError: A value is outside the declared display contract.
    """
    return _record(value, ApprovalBatchView, _VIEW, True)
