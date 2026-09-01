"""Read-only normalized-first projection for Console Inspector calls.

The normalized reader is deliberately injected and disabled by default.  This
foundation therefore changes no writer behavior and keeps the shipping path on
the existing ``message_exchanges`` rows until a later rollout enables verified
normalized reconstruction.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

from loguru import logger

from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    ExchangeCapture,
    capture_from_storage,
)
if TYPE_CHECKING:
    from tldw_chatbook.Chat.trace_export_profiles import TraceViewerProfile


@dataclass(frozen=True, slots=True)
class NormalizedTraceCall:
    """One reconstructed normalized call and its verification result."""

    call_id: str
    capture: ExchangeCapture
    abandoned: bool
    verification_status: Literal["verified", "unverified"]
    provenance: Literal["native", "legacy_snapshot"] = "native"
    chronology: Literal["known", "recorded_call_only"] = "known"
    uncertainty_codes: tuple[str, ...] = ()
    source: Literal["normalized"] = field(default="normalized", init=False)

    @property
    def verified(self) -> bool:
        """Whether equivalence verification permits normalized preference."""

        return self.verification_status == "verified"


@dataclass(frozen=True, slots=True)
class LegacyExchangeCall:
    """One decoded legacy ``message_exchanges`` call."""

    capture: ExchangeCapture
    abandoned: bool
    provenance: Literal["legacy_blob"] = field(default="legacy_blob", init=False)
    chronology: Literal["recorded_call_only"] = field(
        default="recorded_call_only", init=False
    )
    uncertainty_codes: tuple[str, ...] = field(
        default=("legacy_blob_unmigrated",), init=False
    )
    source: Literal["legacy"] = field(default="legacy", init=False)


ProjectedTraceCall: TypeAlias = NormalizedTraceCall | LegacyExchangeCall
NormalizedCallReader: TypeAlias = Callable[[str], Sequence[NormalizedTraceCall]]
LegacyExchangeReader: TypeAlias = Callable[[str], Sequence[Mapping[str, object]]]
_TraceCallT = TypeVar("_TraceCallT", NormalizedTraceCall, LegacyExchangeCall)
_VALID_CAPTURE_STATUSES = frozenset({"complete", "stopped", "error"})
_USAGE_COUNT_FIELDS = frozenset(
    {
        "uncached_input",
        "cache_read",
        "cache_write",
        "output",
        "audio_input",
        "audio_output",
    }
)
_USAGE_STRING_FIELDS = frozenset({"provider", "model"})
_USAGE_FIELDS = (
    _USAGE_COUNT_FIELDS
    | _USAGE_STRING_FIELDS
    | {
        "transcription_seconds",
        "partial",
    }
)
_SAFE_BODY_OMISSION = "[hidden by Safe trace viewer]"


def project_capture_for_viewer(
    capture: ExchangeCapture,
    profile: TraceViewerProfile,
) -> ExchangeCapture:
    """Return a credential-safe disclosure projection of one stored call.

    Safe keeps structural facts but never returns sensitive provider-only
    bodies. Full returns every persisted non-credential field available in
    the stored call; legacy Safe captures remain irrecoverably reduced.

    Args:
        capture: Stored legacy exchange capture to disclose.
        profile: Explicit local Safe or Full viewer profile.

    Returns:
        A credential-safe capture containing only fields allowed by ``profile``.

    Raises:
        TypeError: If ``profile`` is not a :class:`TraceViewerProfile`.
    """

    # Keep the Inspector's first-paint import closure small. Export projection is
    # needed only when a call body is actually opened.
    from tldw_chatbook.Chat.console_exchange_export import project_exchange_export
    from tldw_chatbook.Chat.trace_export_profiles import (
        TraceExportProfile,
        TraceViewerProfile,
    )

    if not isinstance(profile, TraceViewerProfile):
        raise TypeError("profile")
    export_profile = (
        TraceExportProfile.FULL_TRACE
        if profile is TraceViewerProfile.FULL
        and capture.capture_detail is CaptureDetail.FULL
        else TraceExportProfile.REDACTED_DIAGNOSTIC
    )
    payload = project_exchange_export(capture, export_profile).payload
    request = payload.get("request")
    response = payload.get("response")
    request_mapping = dict(request) if isinstance(request, Mapping) else {}
    response_mapping = dict(response) if isinstance(response, Mapping) else {}
    if "truncation_inventory" not in capture.request:
        request_mapping.pop("truncation_inventory", None)
    if "truncation_inventory" not in capture.response:
        response_mapping.pop("truncation_inventory", None)
    if profile is TraceViewerProfile.SAFE:
        messages = request_mapping.get("messages_payload")
        message_count = len(messages) if isinstance(messages, list) else 0
        tools = request_mapping.get("tools")
        tool_count = len(tools) if isinstance(tools, list) else 0
        tool_calls = response_mapping.get("tool_calls")
        tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
        request_mapping = {
            "system_message": (
                _SAFE_BODY_OMISSION
                if request_mapping.get("system_message")
                else ""
            ),
            "messages_payload": [
                {"role": "hidden", "content": _SAFE_BODY_OMISSION}
                for _index in range(message_count)
            ],
            "tools": [
                {"schema": _SAFE_BODY_OMISSION} for _index in range(tool_count)
            ],
            "truncation_inventory": list(
                request_mapping.get("truncation_inventory") or ()
            ),
        }
        response_mapping = {
            "content": (
                _SAFE_BODY_OMISSION if response_mapping.get("content") else ""
            ),
            "tool_calls": [
                {"call": _SAFE_BODY_OMISSION}
                for _index in range(tool_call_count)
            ],
            "synthetic_fallback": bool(
                response_mapping.get("synthetic_fallback", False)
            ),
            "truncation_inventory": list(
                response_mapping.get("truncation_inventory") or ()
            ),
        }
    projected = replace(
        capture,
        provider=str(payload.get("provider") or ""),
        model=str(payload.get("model") or ""),
        endpoint=(
            str(payload["endpoint"]) if payload.get("endpoint") is not None else None
        ),
        request=request_mapping,
        response=response_mapping,
        omitted_keys=tuple(str(item) for item in payload.get("omitted_keys") or ()),
    )
    return capture if projected == capture else projected


def _semantic_key(call: ProjectedTraceCall) -> tuple[str, int]:
    return call.capture.run_tag, call.capture.seq


def _semantic_order(call: ProjectedTraceCall) -> tuple[str, int, str, str]:
    stable_id = call.call_id if isinstance(call, NormalizedTraceCall) else ""
    return (
        call.capture.created_at,
        call.capture.seq,
        call.capture.run_tag,
        stable_id,
    )


def _usage_validation_error(raw: object) -> str | None:
    if raw is None:
        return None
    if type(raw) is not str or not raw:
        return "usage_json"
    try:
        usage = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return "usage_json"
    if type(usage) is not dict or not set(usage).issubset(_USAGE_FIELDS):
        return "usage_json"
    for field_name in _USAGE_COUNT_FIELDS:
        if field_name in usage:
            value = usage[field_name]
            if type(value) is not int or value < 0:
                return "usage_json"
    for field_name in _USAGE_STRING_FIELDS:
        if field_name in usage and type(usage[field_name]) is not str:
            return "usage_json"
    if "partial" in usage and type(usage["partial"]) is not bool:
        return "usage_json"
    if "transcription_seconds" in usage:
        seconds = usage["transcription_seconds"]
        if type(seconds) not in {int, float} or seconds < 0:
            return "usage_json"
        try:
            finite = math.isfinite(seconds)
        except (TypeError, OverflowError):
            return "usage_json"
        if not finite:
            return "usage_json"
    return None


def _mapping_validation_error(value: object, field_name: str) -> str | None:
    if not isinstance(value, Mapping):
        return field_name
    try:
        if any(type(key) is not str for key in value):
            return field_name
    except Exception:
        return field_name
    return None


def _capture_validation_error(capture: object) -> str | None:
    if not isinstance(capture, ExchangeCapture):
        return "capture_type"
    if type(capture.run_tag) is not str or not capture.run_tag:
        return "run_tag"
    if type(capture.seq) is not int or capture.seq < 0:
        return "seq"
    if type(capture.status) is not str or capture.status not in _VALID_CAPTURE_STATUSES:
        return "status"
    if type(capture.created_at) is not str or not capture.created_at:
        return "created_at"
    if type(capture.provider) is not str:
        return "provider"
    if type(capture.model) is not str:
        return "model"
    if capture.endpoint is not None and type(capture.endpoint) is not str:
        return "endpoint"
    request_error = _mapping_validation_error(capture.request, "request")
    if request_error is not None:
        return request_error
    response_error = _mapping_validation_error(capture.response, "response")
    if response_error is not None:
        return response_error
    usage_error = _usage_validation_error(capture.usage_json)
    if usage_error is not None:
        return usage_error
    if type(capture.omitted_keys) is not tuple or any(
        type(key) is not str for key in capture.omitted_keys
    ):
        return "omitted_keys"
    if type(capture.capture_detail) is not CaptureDetail:
        return "capture_detail"
    return None


def _normalized_validation_error(call: object) -> str | None:
    if not isinstance(call, NormalizedTraceCall):
        return "call_type"
    if type(call.call_id) is not str or not call.call_id:
        return "call_id"
    if type(call.abandoned) is not bool:
        return "abandoned"
    if type(call.verification_status) is not str or call.verification_status not in {
        "verified",
        "unverified",
    }:
        return "verification_status"
    if call.provenance not in {"native", "legacy_snapshot"}:
        return "provenance"
    if call.chronology not in {"known", "recorded_call_only"}:
        return "chronology"
    if type(call.uncertainty_codes) is not tuple or any(
        type(item) is not str or not item for item in call.uncertainty_codes
    ):
        return "uncertainty_codes"
    return _capture_validation_error(call.capture)


def _append_unique_claim(
    groups: dict[tuple[str, int], list[_TraceCallT]],
    claim: _TraceCallT,
) -> None:
    key = _semantic_key(claim)
    claims = groups.setdefault(key, [])
    if claim not in claims:
        claims.append(claim)


def _warn_rejected(*, source: str, reason: str) -> None:
    logger.warning(
        f"console_trace_projection_candidate_rejected: source={source}: reason={reason}"
    )


def _warn_ambiguous(*, source: str) -> None:
    logger.warning(f"console_trace_projection_ambiguous: source={source}")


class ConsoleTraceProjection:
    """Merge verified normalized calls with legacy exchange fallbacks.

    Readers are synchronous because the store invokes this boundary through
    ``asyncio.to_thread``.  Neither this class nor its reader contract opens a
    transaction or claims a write lock.
    """

    def __init__(
        self,
        *,
        legacy_reader: LegacyExchangeReader,
        normalized_reader: NormalizedCallReader | None = None,
        normalized_reads_enabled: bool = False,
    ) -> None:
        self._legacy_reader = legacy_reader
        self._normalized_reader = normalized_reader
        self._normalized_reads_enabled = bool(normalized_reads_enabled)

    @property
    def normalized_writes_enabled(self) -> Literal[False]:
        """Return the hard-off writer state for this rollout foundation."""

        return False

    @property
    def normalized_reads_enabled(self) -> bool:
        """Whether the injected normalized reader participates in reads."""

        return self._normalized_reads_enabled and self._normalized_reader is not None

    def read_calls(self, message_id: str) -> tuple[ProjectedTraceCall, ...]:
        """Return normalized-first calls for one persisted assistant message."""

        if not isinstance(message_id, str) or not message_id:
            return ()

        legacy_by_key: dict[tuple[str, int], list[LegacyExchangeCall]] = {}
        for row in self._legacy_reader(message_id):
            legacy = self._decode_legacy_row(row)
            if legacy is not None:
                _append_unique_claim(legacy_by_key, legacy)

        normalized_by_key: dict[tuple[str, int], list[NormalizedTraceCall]] = {}
        if self.normalized_reads_enabled:
            assert self._normalized_reader is not None
            for call in self._normalized_reader(message_id):
                error = _normalized_validation_error(call)
                if error is not None:
                    _warn_rejected(source="normalized", reason=error)
                    continue
                assert isinstance(call, NormalizedTraceCall)
                _append_unique_claim(normalized_by_key, call)

        selected: list[ProjectedTraceCall] = []
        for key in legacy_by_key.keys() | normalized_by_key.keys():
            normalized_claims = normalized_by_key.get(key, [])
            legacy_claims = legacy_by_key.get(key, [])
            if len(normalized_claims) == 1 and normalized_claims[0].verified:
                selected.append(normalized_claims[0])
                continue
            if len(normalized_claims) > 1:
                _warn_ambiguous(source="normalized")
                if len(legacy_claims) == 1:
                    selected.append(legacy_claims[0])
                elif len(legacy_claims) > 1:
                    _warn_ambiguous(source="legacy")
                continue
            if len(legacy_claims) == 1:
                selected.append(legacy_claims[0])
            elif len(legacy_claims) > 1:
                _warn_ambiguous(source="legacy")

        return tuple(sorted(selected, key=_semantic_order))

    @staticmethod
    def _decode_legacy_row(
        row: Mapping[str, object],
    ) -> LegacyExchangeCall | None:
        try:
            run_tag = row["run_tag"]
            seq = row["seq"]
            status = row["status"]
            created_at = row["created_at"]
            if type(run_tag) is not str or not run_tag:
                raise ValueError("run_tag")
            if type(seq) is not int or seq < 0:
                raise ValueError("seq")
            if type(status) is not str or status not in _VALID_CAPTURE_STATUSES:
                raise ValueError("status")
            if type(created_at) is not str or not created_at:
                raise ValueError("created_at")
            abandoned = row["abandoned"]
            if type(abandoned) is not bool:
                raise ValueError("abandoned")
            blob = row["capture_blob"]
            if not isinstance(blob, (bytes, bytearray, memoryview)):
                raise TypeError("capture_blob")
            detail = row.get("capture_detail", "safe")
            if type(detail) is not str or detail not in {"safe", "full"}:
                raise ValueError("capture_detail")
            capture = capture_from_storage(bytes(blob), detail)
            capture_error = _capture_validation_error(capture)
            if capture_error is not None:
                raise ValueError(capture_error)
            if (
                capture.run_tag != run_tag
                or capture.seq != seq
                or capture.status != status
                or capture.created_at != created_at
            ):
                raise ValueError("capture_authority_mismatch")
            return LegacyExchangeCall(
                capture=capture,
                abandoned=abandoned,
            )
        except Exception as exc:
            # Do not attach a traceback: this frame owns decoded/raw capture
            # content, and Loguru diagnose output may expose local values.
            logger.warning(
                f"exchange_blob_decode_failed: error_type={type(exc).__name__}"
            )
            return None


__all__ = [
    "ConsoleTraceProjection",
    "LegacyExchangeCall",
    "NormalizedTraceCall",
    "ProjectedTraceCall",
]
