"""Shared sampling-param keys, validation, and layer-merge helpers (ADR-147).

Single source of truth for the param names a preset, a registry entry, or
the agent resolver may carry. Transport-level keys (``streaming``) are
deliberately excluded: child runs keep the run loop's streaming policy.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, overload

KNOWN_SAMPLING_PARAM_KEYS: frozenset[str] = frozenset({
    "temperature", "top_p", "min_p", "top_k", "max_tokens", "seed",
    "presence_penalty", "frequency_penalty",
    "reasoning_effort", "reasoning_summary", "verbosity",
    "thinking_effort", "thinking_budget_tokens",
})
_FLOAT_KEYS = frozenset({
    "temperature", "top_p", "min_p", "presence_penalty", "frequency_penalty",
})
_INT_KEYS = frozenset({"top_k", "max_tokens", "seed", "thinking_budget_tokens"})
_STRING_KEYS = frozenset({
    "reasoning_effort", "reasoning_summary", "verbosity", "thinking_effort",
})

def validate_sampling_params(params: Mapping[str, Any]) -> list[str]:
    """Return validation errors for ``params``; empty list means valid."""
    errors: list[str] = []
    for key, value in params.items():
        if key not in KNOWN_SAMPLING_PARAM_KEYS:
            errors.append(f"unknown sampling param '{key}'")
            continue
        if isinstance(value, bool):
            errors.append(f"'{key}' must not be a boolean")
        elif key in _FLOAT_KEYS and not isinstance(value, (int, float)):
            errors.append(f"'{key}' must be a number")
        elif key in _INT_KEYS and not isinstance(value, int):
            errors.append(f"'{key}' must be an integer")
        elif key in _STRING_KEYS and not isinstance(value, str):
            errors.append(f"'{key}' must be a string")
    return errors

def params_to_tuple(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(params.items()))

def params_to_dict(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    return dict(pairs)


# Layer-merge helpers moved verbatim from console_session_settings.py
# (TASK-32477): same precedence walk (first non-blank source wins), same
# coercion rules. console_session_settings re-imports them under their
# original underscore names so existing call sites are unchanged.


def _is_blank_value(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _parse_optional_int(value: object) -> int | None:
    if _is_blank_value(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdecimal():
            return int(stripped)
        if stripped.startswith("-") and stripped[1:].isdecimal():
            return int(stripped)
    return None


def _string_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def float_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
    default: float,
) -> float:
    for source in sources:
        if key not in source:
            continue
        value = source.get(key)
        if _is_blank_value(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def optional_float_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
) -> float | None:
    for source in sources:
        if key not in source:
            continue
        value = source.get(key)
        if _is_blank_value(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def optional_int_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
) -> int | None:
    for source in sources:
        if key not in source:
            continue
        value = source.get(key)
        if _is_blank_value(value):
            continue
        parsed = _parse_optional_int(value)
        if parsed is not None:
            return parsed
    return None


def optional_string_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
) -> str | None:
    for source in sources:
        value = source.get(key)
        text = _string_value(value)
        if text:
            return text
    return None


@overload
def bool_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
    default: bool,
) -> bool: ...


@overload
def bool_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
    default: None,
) -> bool | None: ...


def bool_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
    default: bool | None,
) -> bool | None:
    for source in sources:
        if key not in source:
            continue
        value = source.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                return True
            if normalized in {"false", "0"}:
                return False
    return default
