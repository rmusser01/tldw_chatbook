"""Privacy-governed export projections for one stored Console exchange."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    ExchangeCapture,
    build_request_capture,
    sanitize_capture_value,
)
from tldw_chatbook.Chat.console_project_instructions import (
    canonical_provider_endpoint_identity,
)
from tldw_chatbook.Chat.trajectory_export import TraceExportProfile

__all__ = [
    "ExchangeExportProjection",
    "ExchangeExportUnavailable",
    "project_exchange_export",
]


_FULL_UNAVAILABLE = (
    "Full trace is unavailable because this call was captured in Safe mode."
)


class ExchangeExportUnavailable(ValueError):
    """The requested disclosure profile cannot be produced from stored data."""


@dataclass(frozen=True, slots=True)
class ExchangeExportProjection:
    """One immutable, JSON-ready disclosure projection."""

    profile: TraceExportProfile
    payload: Mapping[str, Any]
    json_text: str
    full_available: bool
    disabled_reason: str | None


def _endpoint(endpoint: str | None) -> str | None:
    if endpoint is None:
        return None
    try:
        return canonical_provider_endpoint_identity(endpoint)
    except ValueError:
        return "[invalid endpoint]"


def _metadata(capture: ExchangeCapture) -> dict[str, Any]:
    try:
        usage = json.loads(capture.usage_json) if capture.usage_json else None
    except (TypeError, json.JSONDecodeError):
        usage = None
    return {
        "run_tag": capture.run_tag,
        "seq": capture.seq,
        "created_at": capture.created_at,
        "provider": capture.provider,
        "model": capture.model,
        "endpoint": _endpoint(capture.endpoint),
        "status": capture.status,
        "usage": sanitize_capture_value(usage),
        "capture_detail": capture.capture_detail.value,
        "omitted_keys": list(capture.omitted_keys),
        "truncation_inventory": {
            "request": list(capture.request.get("truncation_inventory") or ()),
            "response": list(capture.response.get("truncation_inventory") or ()),
        },
    }


def project_exchange_export(
    capture: ExchangeCapture,
    profile: TraceExportProfile,
) -> ExchangeExportProjection:
    """Project one stored call through the selected existing Trace profile."""
    if not isinstance(profile, TraceExportProfile):
        raise TypeError("profile must be TraceExportProfile")
    full_available = capture.capture_detail is CaptureDetail.FULL
    if profile is TraceExportProfile.FULL_TRACE and not full_available:
        raise ExchangeExportUnavailable(_FULL_UNAVAILABLE)

    payload = _metadata(capture)
    if profile is not TraceExportProfile.SAFE_SUMMARY:
        detail = (
            CaptureDetail.FULL
            if profile is TraceExportProfile.FULL_TRACE
            else CaptureDetail.SAFE
        )
        request, redacted_paths = build_request_capture(
            {
                key: value
                for key, value in capture.request.items()
                if key != "truncation_inventory"
            },
            capture_detail=detail,
        )
        request["truncation_inventory"] = list(
            capture.request.get("truncation_inventory") or ()
        )
        payload["request"] = request
        payload["response"] = sanitize_capture_value(capture.response)
        payload["omitted_keys"] = sorted(
            set(capture.omitted_keys).union(redacted_paths)
        )

    payload = sanitize_capture_value(payload)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return ExchangeExportProjection(
        profile=profile,
        payload=payload,
        json_text=json_text,
        full_available=full_available,
        disabled_reason=None if full_available else _FULL_UNAVAILABLE,
    )
