"""Repair model tool arguments that arrive JSON-encoded as strings.

A common small-model failure is emitting ``"[\\"a\\"]"`` where the schema
declares an array. Handed straight to the provider that fails validation and
costs a whole turn, so it is repaired once at the dispatch choke point.

Coercion is driven by the schema, never by the value. Guessing from the value
alone would corrupt a legitimately string-typed field that happens to contain
brackets -- a note, a regex, a snippet of prose -- which is a worse failure than
the one being fixed: silent data corruption instead of a loud validation error.

Only the string-to-container direction is repaired, and only when the decoded
value actually matches the declared type. Anything else is left exactly as it
arrived so the existing validation reports it (TASK-26005 AC#4).
"""

from __future__ import annotations

import json
from typing import Any

#: Bounded so a pathologically nested string cannot drive a long decode loop.
#: Two covers the observed cases: encoded once, and double-encoded.
_MAX_DECODE_PASSES = 2

_FENCE_PREFIXES = ("```json", "```")
_FENCE_SUFFIX = "```"

_CONTAINER_TYPES: dict[str, type | tuple[type, ...]] = {
    "array": list,
    "object": dict,
}


def _strip_code_fence(text: str) -> str:
    """Undo a markdown fence a model wrapped around a JSON value."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    for prefix in _FENCE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break
    if stripped.endswith(_FENCE_SUFFIX):
        stripped = stripped[: -len(_FENCE_SUFFIX)]
    return stripped.strip()


def _decode_to_type(raw: str, expected: type | tuple[type, ...]) -> Any | None:
    """Decode ``raw`` until it is ``expected``, or give up.

    Returns None when the value cannot be decoded, or decodes to something
    other than the declared type -- both cases must fall through to normal
    validation rather than being substituted.
    """
    candidate: Any = _strip_code_fence(raw)
    for _ in range(_MAX_DECODE_PASSES):
        if not isinstance(candidate, str):
            break
        text = candidate.strip()
        if not text:
            return None
        try:
            candidate = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
    return candidate if isinstance(candidate, expected) else None


def _properties_of(schema: Any) -> dict:
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _coerce_value(value: Any, schema: Any, path: str, coerced: list[str]) -> Any:
    """Coerce one value against its subschema, recursing into containers."""
    if not isinstance(schema, dict):
        return value

    declared = schema.get("type")
    expected = _CONTAINER_TYPES.get(declared) if isinstance(declared, str) else None

    if isinstance(value, str) and expected is not None:
        decoded = _decode_to_type(value, expected)
        if decoded is None:
            return value
        coerced.append(path)
        value = decoded

    if isinstance(value, dict):
        properties = _properties_of(schema)
        if not properties:
            return value
        return {
            key: _coerce_value(
                item,
                properties.get(key),
                f"{path}.{key}" if path else str(key),
                coerced,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        items_schema = schema.get("items")
        if not isinstance(items_schema, dict):
            return value
        return [
            _coerce_value(item, items_schema, f"{path}[{index}]", coerced)
            for index, item in enumerate(value)
        ]

    return value


def coerce_tool_args(
    args: dict[str, Any] | None, parameters: Any
) -> tuple[dict[str, Any], list[str]]:
    """Return repaired arguments plus the paths that were repaired.

    Args:
        args: Arguments as the model emitted them.
        parameters: The tool's JSON-Schema ``parameters`` object.

    Returns:
        ``(arguments, coerced_paths)``. The input is never mutated, and
        ``coerced_paths`` is empty when nothing needed repair -- callers use it
        to report a systematically malformed model instead of masking it.
    """
    if not isinstance(args, dict) or not args:
        return ({} if args is None else args), []

    properties = _properties_of(parameters)
    if not properties:
        return dict(args), []

    coerced: list[str] = []
    repaired = {
        key: _coerce_value(value, properties.get(key), str(key), coerced)
        for key, value in args.items()
    }
    return repaired, coerced
