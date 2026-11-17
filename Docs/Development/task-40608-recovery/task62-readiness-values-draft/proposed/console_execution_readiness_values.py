"""Closed readiness display values; decoded evidence grants no authority."""

from __future__ import annotations

from datetime import datetime

from .console_execution_projection_values import (
    _boolean,
    _count,
    _fail,
    _optional,
    _record,
    _text,
    _tuple,
)
from .console_session_settings import ConsoleSettingsReadiness
from .provider_test_evidence import ProviderDraftIdentity, ProviderTestEvidence


def _timestamp(value: object, decode: bool) -> datetime | str:
    if decode:
        try:
            return datetime.fromisoformat(_text(value, True))
        except (ValueError, OverflowError):
            _fail("invalid observation time")
    if type(value) is not datetime:
        _fail("expected original observation time")
    try:
        return value.isoformat()
    except (ValueError, TypeError, OverflowError):
        _fail("invalid observation time")


_IDENTITY = {
    "provider_key": _text,
    "connection_identity": _tuple(_text),
    "credential_source": _text,
    "credential_revision": _count,
    "draft_generation": _count,
    "custom_endpoint_id": _optional(_text),
}


def _identity(value: object, decode: bool):
    return _record(value, ProviderDraftIdentity, _IDENTITY, decode)


_EVIDENCE = {
    "identity": _identity,
    "endpoint": _text,
    "model_ids": _tuple(_text),
    "category": _optional(_text),
    "credential": _text,
    "generation": _text,
    "generation_category": _optional(_text),
    "observed_at": _optional(_timestamp),
    "generation_observed_at": _optional(_timestamp),
    "generation_model": _optional(_text),
}


def _evidence(value: object, decode: bool):
    return _record(value, ProviderTestEvidence, _EVIDENCE, decode)


# Original constructors own the finite vocabulary and cross-field rules. In
# particular, top-level observations need not accompany connection/evidence;
# only ProviderTestEvidence requires timezone-aware observation values.
_READINESS = {
    "label": _text,
    "detail": _text,
    "native_send_supported": _boolean,
    "operability": _optional(_text),
    "blocker": _optional(_text),
    "recovery_action": _optional(_text),
    "provider_display_name": _text,
    "configuration": _optional(_text),
    "configuration_issue": _optional(_text),
    "credential": _optional(_text),
    "credential_source": _optional(_text),
    "endpoint": _optional(_text),
    "endpoint_category": _optional(_text),
    "model": _optional(_text),
    "generation": _optional(_text),
    "generation_category": _optional(_text),
    "subscription_status": _optional(_text),
    "connection": _optional(_identity),
    "observed_at": _optional(_timestamp),
    "generation_observed_at": _optional(_timestamp),
    "evidence": _optional(_evidence),
    "invalid_endpoint": _optional(_text),
}


def encode_settings_readiness(value: ConsoleSettingsReadiness) -> dict:
    """Encode only original readiness facts as strict plain JSON.

    Args:
        value: Original readiness record, with no live credential holder.

    Returns:
        Detached fields, preserving independent observations and None.

    Raises:
        ConsoleProtocolError: A type, field or original constructor is invalid.
    """
    return _record(value, ConsoleSettingsReadiness, _READINESS, False)


def decode_settings_readiness(value: dict) -> ConsoleSettingsReadiness:
    """Reconstruct the original readiness value without reading live state.

    Args:
        value: Exact plain JSON record with the closed readiness fields.

    Returns:
        An original readiness record containing only inert display evidence.

    Raises:
        ConsoleProtocolError: A type, field or original constructor is invalid.
    """
    return _record(value, ConsoleSettingsReadiness, _READINESS, True)
