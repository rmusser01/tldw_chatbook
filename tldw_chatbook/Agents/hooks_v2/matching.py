"""Structured matcher and one-pass typed MCP input expansion."""

from __future__ import annotations

import fnmatch
import re
from typing import Any

from .models import HookEvent, HookHandler
from .validation import INPUT_BYTES, _json_tree

_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_.]*)\}")


class UnsupportedEventField(ValueError):
    """A required host projection is unavailable for this occurrence."""


def _field(event: HookEvent, path: str) -> Any:
    parts = path.split(".")
    if parts[0] == "data":
        if len(parts) != 2 or parts[1] not in event.data:
            raise UnsupportedEventField(path)
        return event.data[parts[1]]
    if len(parts) != 1:
        raise UnsupportedEventField(path)
    value = getattr(event, parts[0], None)
    if value is None:
        raise UnsupportedEventField(path)
    return value


def matches_handler(handler: HookHandler, event: HookEvent) -> bool:
    """Evaluate AND keys and OR patterns against qualified host fields.

    An unavailable canonical operation is unsupported for an operation matcher,
    while another matcher using only the available tool ID still works.
    """
    if handler.event != event.event:
        return False
    if not handler.match:
        return True
    # Check support for the whole conjunction first. An absent operation must
    # not be hidden by an earlier mismatching tool ID on a controlling handler.
    values = {}
    for key in handler.match:
        value = event.data.get(key)
        if type(value) is not str:
            raise UnsupportedEventField(key)
        values[key] = value
    for key, patterns in handler.match.items():
        if not any(fnmatch.fnmatchcase(values[key], pattern) for pattern in patterns):
            return False
    return True


def expand_input(handler: HookHandler, event: HookEvent) -> dict[str, Any]:
    """Expand one typed MCP template pass over documented host event fields."""
    if handler.type != "mcp_tool":
        raise ValueError("only MCP handlers have input templates")

    def expand(value: str) -> object:
        match = _REFERENCE.fullmatch(value)
        if match:
            return _field(event, match.group(1))
        parts: list[str] = []
        size = 0
        end = 0
        for found in _REFERENCE.finditer(value):
            literal = value[end : found.start()]
            projected = _field(event, found.group(1))
            if type(projected) not in (str, int, float, bool):
                raise ValueError("embedded template path must be scalar")
            rendered = (
                str(projected).lower() if type(projected) is bool else str(projected)
            )
            size += len(literal.encode("utf-8")) + len(rendered.encode("utf-8"))
            if size > INPUT_BYTES:
                raise ValueError("template output byte limit exceeded")
            parts.extend((literal, rendered))
            end = found.end()
        if not parts:
            return value
        tail = value[end:]
        if size + len(tail.encode("utf-8")) > INPUT_BYTES:
            raise ValueError("template output byte limit exceeded")
        parts.append(tail)
        return "".join(parts)

    return _json_tree(handler.input or {}, string_transform=expand, allow_frozen=True)
