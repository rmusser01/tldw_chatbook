"""Shared sampling-param keys, validation, and layer-merge helpers (ADR-147).

Single source of truth for the param names a preset, a registry entry, or
the agent resolver may carry. Transport-level keys (``streaming``) are
deliberately excluded: child runs keep the run loop's streaming policy.
"""
from __future__ import annotations

import math
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

# Numeric bounds and string enumerations shared with
# ``validate_console_session_settings`` (console_session_settings.py aliases
# the four enum sets below): presets, registry entries, and ordinary Console
# settings are held to ONE contract, so an out-of-range value is refused at
# save time instead of reaching a provider request.
TEMPERATURE_RANGE: tuple[float, float] = (0.0, 2.0)
TOP_P_RANGE: tuple[float, float] = (0.0, 1.0)
MIN_P_RANGE: tuple[float, float] = (0.0, 1.0)
#: Shared by ``presence_penalty`` and ``frequency_penalty``.
PENALTY_RANGE: tuple[float, float] = (-2.0, 2.0)
MIN_TOP_K = 0
MIN_MAX_TOKENS = 1
MIN_SEED = 0
MIN_THINKING_BUDGET_TOKENS = 1024
REASONING_EFFORT_VALUES: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)
REASONING_SUMMARY_VALUES: frozenset[str] = frozenset(
    {"auto", "concise", "detailed", "none"}
)
VERBOSITY_VALUES: frozenset[str] = frozenset({"low", "medium", "high"})
THINKING_EFFORT_VALUES: frozenset[str] = frozenset(
    {"off", "low", "medium", "high", "xhigh", "max"}
)
_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "temperature": TEMPERATURE_RANGE,
    "top_p": TOP_P_RANGE,
    "min_p": MIN_P_RANGE,
    "presence_penalty": PENALTY_RANGE,
    "frequency_penalty": PENALTY_RANGE,
}
_INT_MINIMA: dict[str, int] = {
    "top_k": MIN_TOP_K,
    "max_tokens": MIN_MAX_TOKENS,
    "seed": MIN_SEED,
    "thinking_budget_tokens": MIN_THINKING_BUDGET_TOKENS,
}
_STRING_ENUMS: dict[str, frozenset[str]] = {
    "reasoning_effort": REASONING_EFFORT_VALUES,
    "reasoning_summary": REASONING_SUMMARY_VALUES,
    "verbosity": VERBOSITY_VALUES,
    "thinking_effort": THINKING_EFFORT_VALUES,
}


def validate_sampling_params(params: Mapping[str, Any]) -> list[str]:
    """Return validation errors for ``params``; empty list means valid.

    Checks key membership, primitive types (booleans are never valid
    sampling values), numeric bounds and finiteness, integer minima, and
    string enumerations — the same contract
    ``validate_console_session_settings`` enforces for ordinary Console
    settings.

    Args:
        params: Candidate sampling parameters from a preset, registry
            entry, or ad-hoc overlay.

    Returns:
        Human-readable error strings, one per offending key, in input
        order; ``[]`` when every key is known and in range.
    """
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
        else:
            if key in _FLOAT_RANGES:
                low, high = _FLOAT_RANGES[key]
                if not math.isfinite(value) or not (low <= value <= high):
                    errors.append(f"'{key}' must be between {low} and {high}")
            elif key in _INT_MINIMA:
                minimum = _INT_MINIMA[key]
                if value < minimum:
                    errors.append(f"'{key}' must be at least {minimum}")
            elif key in _STRING_ENUMS and value not in _STRING_ENUMS[key]:
                allowed = ", ".join(sorted(_STRING_ENUMS[key]))
                errors.append(f"'{key}' must be one of {allowed}")
    return errors


def params_to_tuple(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Canonical immutable form of a params mapping.

    Args:
        params: Sampling params keyed by name.

    Returns:
        ``(key, value)`` pairs sorted by key, so two mappings with the same
        contents always produce equal tuples (dataclass equality and JSON
        snapshots rely on this ordering).
    """
    return tuple(sorted(params.items()))


def params_to_dict(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Inverse of :func:`params_to_tuple`.

    Args:
        pairs: ``(key, value)`` pairs from a preset/registry snapshot.

    Returns:
        A plain dict; later duplicate keys overwrite earlier ones.
    """
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
    """First non-blank ``key`` across ``sources`` as a float.

    Args:
        sources: Precedence-ordered mappings; earlier wins.
        key: Setting name to look up.
        default: Returned when no source yields a coercible value.

    Returns:
        The first value ``float()`` accepts, else ``default``.
    """
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
    """Optional variant of :func:`float_setting_from_sources`.

    Args:
        sources: Precedence-ordered mappings; earlier wins.
        key: Setting name to look up.

    Returns:
        The first coercible value, or ``None`` when every source is
        missing, blank, or uncoercible.
    """
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
    """First non-blank ``key`` across ``sources`` as an int.

    Args:
        sources: Precedence-ordered mappings; earlier wins.
        key: Setting name to look up.

    Returns:
        The first value :func:`_parse_optional_int` accepts (ints, integral
        floats, decimal strings), or ``None``.
    """
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
    """First non-blank string ``key`` across ``sources``, stripped.

    Args:
        sources: Precedence-ordered mappings; earlier wins.
        key: Setting name to look up.

    Returns:
        The first non-empty stripped string, or ``None``.
    """
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
    """First explicit boolean ``key`` across ``sources``.

    Args:
        sources: Precedence-ordered mappings; earlier wins.
        key: Setting name to look up.
        default: Returned when no source holds a real bool or a recognized
            string; pass ``None`` to distinguish "unset" from ``False``.

    Returns:
        Real bools as-is; ``"true"/"1"`` and ``"false"/"0"``
        (case-insensitive, stripped) mapped; otherwise ``default``.
    """
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
