"""The existing detached Agent run fields, independent of display readiness."""

import json

from .console_execution_protocol import (
    ConsoleProtocolError,
    _array,
    _boolean,
    _detach,
    _integer,
    _optional,
    _record,
    _text,
    _unique_pairs,
)

_TEXT = _text()
_REQUEST = _record(
    conversation_id=_text(minimum=1),
    include_superseded=_boolean,
    limit=_optional(_integer()),
    agent_kind=_optional(_TEXT),
)
_FIELDS = (
    "id",
    "conversation_id",
    "parent_run_id",
    "agent_kind",
    "status",
    "created_at",
    "updated_at",
    "assistant_message_id",
)
_ROW = _record(
    **{
        name: _optional(_TEXT)
        if name in {"parent_run_id", "assistant_message_id"}
        else _TEXT
        for name in _FIELDS
    },
    steps=_TEXT,
)


def validate_run_history(value):
    frozen = _detach(value, "run_history", 0, set(), thaw=False)
    _array(_ROW)(frozen, "run_history")
    for row in value:
        try:
            steps = json.loads(row["steps"], object_pairs_hook=_unique_pairs)
        except (ValueError, RecursionError):
            raise ConsoleProtocolError("invalid_run_history_steps") from None
        if type(steps) is not list or any(type(item) is not dict for item in steps):
            raise ConsoleProtocolError("invalid_run_history_steps")
        _detach(steps, "run_history.steps", 0, set(), thaw=False)


def decode_run_history(value):
    validate_run_history(value)
    return [{**row, "steps": json.loads(row["steps"])} for row in value]
