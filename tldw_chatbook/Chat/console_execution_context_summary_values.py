"""Finite original context estimates and summary rows for ordinary presentation."""

from __future__ import annotations

from dataclasses import dataclass

from .console_execution_projection_values import (
    _boolean,
    _integer,
    _optional,
    _record,
    _text,
)
from .console_execution_readiness_values import (
    decode_settings_readiness,
    encode_settings_readiness,
)
from .console_session_settings import (
    ConsoleSettingsContextEstimate,
    ConsoleSettingsSummaryState,
)


@dataclass(frozen=True, slots=True)
class ConsoleContextSummaryProjection:
    estimate: ConsoleSettingsContextEstimate
    summary: ConsoleSettingsSummaryState
    compaction_pause_expected: bool
    compaction_forecast_unavailable: bool


# The original constructors impose no nonnegative bound. The estimate builder
# also preserves supplied token-counter/resolver and staged-count integers.
_ESTIMATE = {
    "used_tokens": _optional(_integer),
    "token_limit": _optional(_integer),
    "label": _text,
    "staged_source_count": _integer,
    "staged_context_summary": _text,
    "token_limit_verified": _optional(_boolean),
    "token_limit_source": _text,
}


def _estimate(value: object, decode: bool):
    return _record(value, ConsoleSettingsContextEstimate, _ESTIMATE, decode)


def _readiness(value: object, decode: bool):
    return decode_settings_readiness(value) if decode else encode_settings_readiness(value)


_SUMMARY = {
    "model_row": _text,
    "context_row": _text,
    "sampling_row": _text,
    "identity_row": _text,
    "temperature": _text,
    "max_tokens": _text,
    "streaming": _text,
    "readiness_label": _text,
    "provider_row": _text,
    "endpoint_row": _text,
    "credential_row": _text,
    "transport_row": _text,
    "action_label": _text,
    "action_tooltip": _text,
    "readiness": _optional(_readiness),
}


def _summary(value: object, decode: bool):
    return _record(value, ConsoleSettingsSummaryState, _SUMMARY, decode)


_PROJECTION = {
    "estimate": _estimate,
    "summary": _summary,
    "compaction_pause_expected": _boolean,
    "compaction_forecast_unavailable": _boolean,
}


def encode_context_summary(
    estimate: ConsoleSettingsContextEstimate,
    summary: ConsoleSettingsSummaryState,
    *,
    compaction_pause_expected: bool,
    compaction_forecast_unavailable: bool,
) -> dict:
    """Detach exact original display values; flags are supplied, never inferred."""
    value = ConsoleContextSummaryProjection(
        estimate, summary, compaction_pause_expected, compaction_forecast_unavailable
    )
    return _record(value, ConsoleContextSummaryProjection, _PROJECTION, False)


def decode_context_summary(payload: dict) -> ConsoleContextSummaryProjection:
    """Restore original constructors from a closed, exact-primitive record."""
    return _record(payload, ConsoleContextSummaryProjection, _PROJECTION, True)
