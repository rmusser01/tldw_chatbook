"""Dual-read projection tests for Console semantic traces."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
from loguru import logger

from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    ExchangeCapture,
    capture_to_blob,
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_trace_metrics import TraceCompatibilityMetrics
from tldw_chatbook.Chat.console_trace_projection import (
    ConsoleTraceProjection,
    LegacyExchangeCall,
    NormalizedTraceCall,
    project_capture_for_viewer,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.trace_export_profiles import TraceViewerProfile


def _capture(*, run_tag: str, seq: int, created_at: str, model: str) -> ExchangeCapture:
    return ExchangeCapture(
        run_tag=run_tag,
        seq=seq,
        created_at=created_at,
        provider="openai",
        model=model,
        endpoint=None,
        request={"model": model},
        response={"content": model},
        status="complete",
        usage_json=None,
        omitted_keys=(),
    )


def _legacy_row(capture: ExchangeCapture, *, abandoned: bool = False) -> dict:
    return {
        "run_tag": capture.run_tag,
        "seq": capture.seq,
        "status": capture.status,
        "abandoned": abandoned,
        "capture_detail": capture.capture_detail.value,
        "capture_blob": capture_to_blob(capture),
        "created_at": capture.created_at,
    }


class _Readers:
    def __init__(
        self,
        *,
        normalized: Sequence[NormalizedTraceCall] = (),
        legacy: Sequence[dict] = (),
    ) -> None:
        self.normalized = normalized
        self.legacy = legacy
        self.normalized_calls: list[str] = []
        self.legacy_calls: list[str] = []

    def read_normalized(self, message_id: str) -> Sequence[NormalizedTraceCall]:
        self.normalized_calls.append(message_id)
        return self.normalized

    def read_legacy(self, message_id: str) -> Sequence[dict]:
        self.legacy_calls.append(message_id)
        return self.legacy


def _normalized(
    capture: ExchangeCapture,
    *,
    call_id: str,
    verified: bool,
    abandoned: bool = False,
) -> NormalizedTraceCall:
    return NormalizedTraceCall(
        call_id=call_id,
        capture=capture,
        abandoned=abandoned,
        verification_status="verified" if verified else "unverified",
    )


def test_safe_and_full_are_credential_safe_views_of_the_same_trace() -> None:
    source = replace(
        _capture(
            run_tag="viewer-run",
            seq=0,
            created_at="2026-08-30T00:00:00Z",
            model="viewer-model",
        ),
        capture_detail=CaptureDetail.FULL,
        endpoint="https://user:password@example.test/v1?api_key=secret-value",
        request={
            "system_message": "provider-only system body",
            "messages_payload": [
                {"role": "user", "content": "provider-only message body"}
            ],
            "tools": [{"name": "private-tool", "token": "secret-value"}],
        },
        response={"content": "provider-only response body"},
    )

    safe = project_capture_for_viewer(source, TraceViewerProfile.SAFE)
    full = project_capture_for_viewer(source, TraceViewerProfile.FULL)

    assert "provider-only message body" not in repr(safe)
    assert "provider-only response body" not in repr(safe)
    assert "provider-only message body" in repr(full)
    assert "provider-only response body" in repr(full)
    assert "secret-value" not in repr(safe)
    assert "secret-value" not in repr(full)
    assert source.request["messages_payload"][0]["content"] == (
        "provider-only message body"
    )


def test_both_viewer_profiles_preserve_frozen_pii_masks() -> None:
    source = replace(
        _capture(
            run_tag="masked-run",
            seq=0,
            created_at="2026-08-30T00:00:00Z",
            model="viewer-model",
        ),
        capture_detail=CaptureDetail.FULL,
        request={
            "system_message": "",
            "messages_payload": [
                {"role": "user", "content": "Email [REDACTED PII]"}
            ],
            "tools": [],
        },
        response={"content": "[REDACTED PII]"},
    )

    safe = project_capture_for_viewer(source, TraceViewerProfile.SAFE)
    full = project_capture_for_viewer(source, TraceViewerProfile.FULL)

    assert "person@example.test" not in repr(safe)
    assert "person@example.test" not in repr(full)
    assert "[REDACTED PII]" in repr(full)


def test_verified_normalized_call_replaces_matching_legacy_exchange() -> None:
    legacy = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="legacy"
    )
    normalized = _capture(
        run_tag="run-1",
        seq=0,
        created_at="2026-08-28T10:00:00Z",
        model="normalized",
    )
    readers = _Readers(
        normalized=(_normalized(normalized, call_id="call-1", verified=True),),
        legacy=(_legacy_row(legacy),),
    )
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    result = projection.read_calls("message-1")

    assert result == (
        NormalizedTraceCall(
            call_id="call-1",
            capture=normalized,
            abandoned=False,
            verification_status="verified",
        ),
    )


def test_unverified_normalized_call_falls_back_to_matching_legacy_exchange() -> None:
    legacy = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="legacy"
    )
    unverified = _capture(
        run_tag="run-1",
        seq=0,
        created_at="2026-08-28T10:00:00Z",
        model="unverified",
    )
    readers = _Readers(
        normalized=(_normalized(unverified, call_id="call-1", verified=False),),
        legacy=(_legacy_row(legacy, abandoned=True),),
    )
    metrics = TraceCompatibilityMetrics()
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
        compatibility_metrics=metrics,
    )

    result = projection.read_calls("message-1")

    assert result == (LegacyExchangeCall(capture=legacy, abandoned=True),)
    assert dict(metrics.snapshot()) == {
        "normalized_write": 0,
        "normalized_read": 0,
        "legacy_read": 1,
        "fallback_read": 1,
        "incomplete": 1,
    }


def test_missing_normalized_call_falls_back_to_legacy_exchange() -> None:
    legacy = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="legacy"
    )
    readers = _Readers(legacy=(_legacy_row(legacy),))
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    assert projection.read_calls("message-1") == (
        LegacyExchangeCall(capture=legacy, abandoned=False),
    )


def test_read_calls_snapshots_normalized_gate_for_fallback_metrics() -> None:
    legacy = _capture(
        run_tag="run-1",
        seq=0,
        created_at="2026-08-28T10:00:00Z",
        model="legacy",
    )
    reads = 0

    def changing_gate() -> bool:
        nonlocal reads
        reads += 1
        return reads == 1

    metrics = TraceCompatibilityMetrics()
    projection = ConsoleTraceProjection(
        normalized_reader=lambda _message_id: (),
        legacy_reader=lambda _message_id: (_legacy_row(legacy),),
        normalized_reads_enabled=changing_gate,
        compatibility_metrics=metrics,
    )

    assert projection.read_calls("message-1") == (LegacyExchangeCall(legacy, False),)
    assert reads == 1
    assert dict(metrics.snapshot())["fallback_read"] == 1


def test_mixed_calls_have_stable_semantic_order_without_duplicates() -> None:
    first = _capture(
        run_tag="run-a", seq=0, created_at="2026-08-28T10:00:00Z", model="first"
    )
    duplicate_legacy = _capture(
        run_tag="run-a",
        seq=1,
        created_at="2026-08-28T10:00:02Z",
        model="legacy-duplicate",
    )
    normalized = _capture(
        run_tag="run-a",
        seq=1,
        created_at="2026-08-28T10:00:02Z",
        model="normalized",
    )
    last = _capture(
        run_tag="run-b", seq=0, created_at="2026-08-28T10:00:03Z", model="last"
    )
    readers = _Readers(
        normalized=(_normalized(normalized, call_id="call-2", verified=True),),
        legacy=(
            _legacy_row(last),
            _legacy_row(duplicate_legacy),
            _legacy_row(first),
        ),
    )
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    result = projection.read_calls("message-1")

    assert [item.capture.model for item in result] == ["first", "normalized", "last"]
    assert [item.source for item in result] == ["legacy", "normalized", "legacy"]


def test_normalized_read_gate_is_off_by_default() -> None:
    normalized = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="normalized"
    )
    legacy = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="legacy"
    )
    readers = _Readers(
        normalized=(_normalized(normalized, call_id="call-1", verified=True),),
        legacy=(_legacy_row(legacy),),
    )
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
    )

    result = projection.read_calls("message-1")

    assert result == (LegacyExchangeCall(capture=legacy, abandoned=False),)
    assert readers.normalized_calls == []
    assert projection.normalized_writes_enabled is False


def test_normalized_write_gate_is_independent_of_read_gate() -> None:
    projection = ConsoleTraceProjection(
        legacy_reader=lambda _message_id: (),
        normalized_reader=lambda _message_id: (),
        normalized_reads_enabled=False,
        normalized_writes_enabled=True,
    )

    assert projection.normalized_reads_enabled is False
    assert projection.normalized_writes_enabled is True


def test_projection_reports_content_free_compatibility_paths() -> None:
    normalized = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="normalized"
    )
    legacy = _capture(
        run_tag="run-2", seq=0, created_at="2026-08-28T10:00:01Z", model="legacy"
    )
    readers = _Readers(
        normalized=(_normalized(normalized, call_id="call-1", verified=True),),
        legacy=(_legacy_row(legacy),),
    )
    metrics = TraceCompatibilityMetrics()
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
        compatibility_metrics=metrics,
    )

    projection.read_calls("message-1")

    assert dict(metrics.snapshot()) == {
        "normalized_write": 0,
        "normalized_read": 1,
        "legacy_read": 1,
        "fallback_read": 1,
        "incomplete": 0,
    }


def test_corrupt_legacy_row_is_isolated_from_valid_siblings() -> None:
    valid = _capture(
        run_tag="run-1", seq=1, created_at="2026-08-28T10:00:01Z", model="valid"
    )
    readers = _Readers(
        legacy=(
            {
                "run_tag": "run-1",
                "seq": 0,
                "capture_detail": "safe",
                "capture_blob": b"corrupt",
                "created_at": "2026-08-28T10:00:00Z",
            },
            _legacy_row(valid),
        )
    )
    projection = ConsoleTraceProjection(legacy_reader=readers.read_legacy)

    assert projection.read_calls("message-1") == (
        LegacyExchangeCall(capture=valid, abandoned=False),
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("run_tag", "other-run"),
        ("seq", True),
        ("status", "error"),
        ("created_at", "2026-08-28T09:59:59Z"),
    ),
)
def test_malformed_legacy_identity_or_order_field_isolated_from_valid_sibling(
    field: str,
    invalid_value: object,
) -> None:
    malformed_capture = _capture(
        run_tag="run-bad",
        seq=0,
        created_at="2026-08-28T10:00:00Z",
        model="must-not-render",
    )
    malformed_row = _legacy_row(malformed_capture)
    malformed_row[field] = invalid_value
    valid = _capture(
        run_tag="run-good",
        seq=1,
        created_at="2026-08-28T10:00:01Z",
        model="valid",
    )
    projection = ConsoleTraceProjection(
        legacy_reader=lambda _message_id: (malformed_row, _legacy_row(valid))
    )

    assert projection.read_calls("message-1") == (
        LegacyExchangeCall(capture=valid, abandoned=False),
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("run_tag", ""),
        ("seq", True),
        ("status", ""),
        ("created_at", ""),
    ),
)
def test_malformed_normalized_identity_or_order_field_isolated_from_valid_sibling(
    field: str,
    invalid_value: object,
) -> None:
    malformed_capture = replace(
        _capture(
            run_tag="run-bad",
            seq=0,
            created_at="2026-08-28T10:00:00Z",
            model="must-not-render",
        ),
        **{field: invalid_value},
    )
    valid = _capture(
        run_tag="run-good",
        seq=1,
        created_at="2026-08-28T10:00:01Z",
        model="valid",
    )
    readers = _Readers(
        normalized=(
            _normalized(malformed_capture, call_id="call-bad", verified=True),
            _normalized(valid, call_id="call-good", verified=True),
        )
    )
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    assert projection.read_calls("message-1") == (
        _normalized(valid, call_id="call-good", verified=True),
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("provider", 7),
        ("model", 7),
        ("endpoint", 7),
        ("request", []),
        ("request", {7: "not-a-string-key"}),
        ("response", []),
        ("response", {7: "not-a-string-key"}),
        ("usage_json", 7),
        ("usage_json", "[]"),
        ("usage_json", '{"uncached_input": "1"}'),
        ("omitted_keys", ["api_key"]),
        ("omitted_keys", ("api_key", 7)),
        ("capture_detail", "safe"),
    ),
)
def test_malformed_normalized_inspector_field_isolated_from_valid_sibling(
    field: str,
    invalid_value: object,
) -> None:
    malformed = replace(
        _capture(
            run_tag="run-bad",
            seq=0,
            created_at="2026-08-28T10:00:00Z",
            model="must-not-render",
        ),
        **{field: invalid_value},
    )
    valid = _capture(
        run_tag="run-good",
        seq=1,
        created_at="2026-08-28T10:00:01Z",
        model="valid",
    )
    readers = _Readers(
        normalized=(
            _normalized(malformed, call_id="call-bad", verified=True),
            _normalized(valid, call_id="call-good", verified=True),
        )
    )
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    assert projection.read_calls("message-1") == (
        _normalized(valid, call_id="call-good", verified=True),
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("provider", 7),
        ("model", 7),
        ("endpoint", 7),
        ("request", []),
        ("response", []),
        ("usage_json", 7),
        ("usage_json", "[]"),
        ("usage_json", '{"uncached_input": "1"}'),
        ("omitted_keys", ("api_key", 7)),
    ),
)
def test_malformed_legacy_inspector_field_isolated_from_valid_sibling(
    field: str,
    invalid_value: object,
) -> None:
    malformed = replace(
        _capture(
            run_tag="run-bad",
            seq=0,
            created_at="2026-08-28T10:00:00Z",
            model="must-not-render",
        ),
        **{field: invalid_value},
    )
    valid = _capture(
        run_tag="run-good",
        seq=1,
        created_at="2026-08-28T10:00:01Z",
        model="valid",
    )
    projection = ConsoleTraceProjection(
        legacy_reader=lambda _message_id: (
            _legacy_row(malformed),
            _legacy_row(valid),
        )
    )

    assert projection.read_calls("message-1") == (
        LegacyExchangeCall(capture=valid, abandoned=False),
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (("abandoned", "false"), ("capture_detail", "invalid")),
)
def test_malformed_legacy_row_contract_isolated_from_valid_sibling(
    field: str,
    invalid_value: object,
) -> None:
    malformed = _capture(
        run_tag="run-bad",
        seq=0,
        created_at="2026-08-28T10:00:00Z",
        model="must-not-render",
    )
    malformed_row = _legacy_row(malformed)
    malformed_row[field] = invalid_value
    valid = _capture(
        run_tag="run-good",
        seq=1,
        created_at="2026-08-28T10:00:01Z",
        model="valid",
    )
    projection = ConsoleTraceProjection(
        legacy_reader=lambda _message_id: (malformed_row, _legacy_row(valid))
    )

    assert projection.read_calls("message-1") == (
        LegacyExchangeCall(capture=valid, abandoned=False),
    )


@pytest.mark.parametrize("missing_field", ("abandoned",))
def test_incomplete_legacy_row_contract_isolated_from_valid_sibling(
    missing_field: str,
) -> None:
    malformed = _capture(
        run_tag="run-bad",
        seq=0,
        created_at="2026-08-28T10:00:00Z",
        model="must-not-render",
    )
    malformed_row = _legacy_row(malformed)
    malformed_row.pop(missing_field)
    valid = _capture(
        run_tag="run-good",
        seq=1,
        created_at="2026-08-28T10:00:01Z",
        model="valid",
    )
    projection = ConsoleTraceProjection(
        legacy_reader=lambda _message_id: (malformed_row, _legacy_row(valid))
    )

    assert projection.read_calls("message-1") == (
        LegacyExchangeCall(capture=valid, abandoned=False),
    )


def test_string_abandoned_normalized_call_isolated_from_valid_sibling() -> None:
    malformed = _capture(
        run_tag="run-bad",
        seq=0,
        created_at="2026-08-28T10:00:00Z",
        model="must-not-render",
    )
    valid = _capture(
        run_tag="run-good",
        seq=1,
        created_at="2026-08-28T10:00:01Z",
        model="valid",
    )
    malformed_call = NormalizedTraceCall(
        call_id="call-bad",
        capture=malformed,
        abandoned="false",
        verification_status="verified",
    )
    valid_call = _normalized(valid, call_id="call-good", verified=True)
    readers = _Readers(normalized=(malformed_call, valid_call))
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    assert projection.read_calls("message-1") == (valid_call,)


@pytest.mark.parametrize("source", ("normalized", "legacy"))
def test_valid_nonempty_usage_shape_projects(source: str) -> None:
    usage_json = ProviderUsage(
        uncached_input=1,
        output=2,
        provider="openai",
        model="valid",
    ).to_json()
    capture = replace(
        _capture(
            run_tag="run-1",
            seq=0,
            created_at="2026-08-28T10:00:00Z",
            model="valid",
        ),
        usage_json=usage_json,
        omitted_keys=("api_key",),
    )
    normalized = _normalized(capture, call_id="call-1", verified=True)
    readers = _Readers(
        normalized=(normalized,) if source == "normalized" else (),
        legacy=(_legacy_row(capture),) if source == "legacy" else (),
    )
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    expected = (
        normalized if source == "normalized" else LegacyExchangeCall(capture, False)
    )
    assert projection.read_calls("message-1") == (expected,)


def test_exact_repeated_claimants_are_deduplicated() -> None:
    legacy = _capture(
        run_tag="run-legacy",
        seq=0,
        created_at="2026-08-28T10:00:00Z",
        model="legacy",
    )
    normalized = _capture(
        run_tag="run-normalized",
        seq=0,
        created_at="2026-08-28T10:00:01Z",
        model="normalized",
    )
    normalized_call = _normalized(normalized, call_id="call-1", verified=True)
    legacy_row = _legacy_row(legacy)
    readers = _Readers(
        normalized=(normalized_call, normalized_call),
        legacy=(legacy_row, dict(legacy_row)),
    )
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    assert projection.read_calls("message-1") == (
        LegacyExchangeCall(capture=legacy, abandoned=False),
        normalized_call,
    )


@pytest.mark.parametrize("reverse", (False, True))
def test_distinct_legacy_claimants_for_same_key_fail_closed(reverse: bool) -> None:
    first = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="first"
    )
    second = replace(first, model="second", response={"content": "second"})
    rows = [_legacy_row(first), _legacy_row(second)]
    if reverse:
        rows.reverse()
    projection = ConsoleTraceProjection(legacy_reader=lambda _message_id: rows)

    assert projection.read_calls("message-1") == ()


@pytest.mark.parametrize("reverse", (False, True))
def test_ambiguous_normalized_fallback_preserves_stable_order(
    reverse: bool,
) -> None:
    legacy = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="legacy"
    )
    first = replace(legacy, model="normalized-a", response={"content": "a"})
    second = replace(legacy, model="normalized-b", response={"content": "b"})
    later = _capture(
        run_tag="run-2", seq=0, created_at="2026-08-28T10:00:01Z", model="later"
    )
    later_call = _normalized(later, call_id="call-later", verified=True)
    normalized = [
        _normalized(first, call_id="call-a", verified=True),
        _normalized(second, call_id="call-b", verified=True),
        later_call,
    ]
    if reverse:
        normalized.reverse()
    readers = _Readers(normalized=normalized, legacy=(_legacy_row(legacy),))
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    assert projection.read_calls("message-1") == (
        LegacyExchangeCall(capture=legacy, abandoned=False),
        later_call,
    )


def test_ambiguous_normalized_and_legacy_claimants_omit_semantic_key() -> None:
    base = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="base"
    )
    normalized = (
        _normalized(replace(base, model="n-a"), call_id="call-a", verified=True),
        _normalized(replace(base, model="n-b"), call_id="call-b", verified=True),
    )
    legacy = (
        _legacy_row(replace(base, model="l-a")),
        _legacy_row(replace(base, model="l-b")),
    )
    readers = _Readers(normalized=normalized, legacy=legacy)
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    assert projection.read_calls("message-1") == ()


def test_one_verified_normalized_claim_wins_over_ambiguous_legacy_claims() -> None:
    base = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="base"
    )
    normalized = _normalized(
        replace(base, model="normalized"), call_id="call-1", verified=True
    )
    readers = _Readers(
        normalized=(normalized,),
        legacy=(
            _legacy_row(replace(base, model="legacy-a")),
            _legacy_row(replace(base, model="legacy-b")),
        ),
    )
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    assert projection.read_calls("message-1") == (normalized,)


def test_ambiguous_verified_normalized_claim_without_legacy_is_omitted() -> None:
    base = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="base"
    )
    readers = _Readers(
        normalized=(
            _normalized(replace(base, model="a"), call_id="call-a", verified=True),
            _normalized(replace(base, model="b"), call_id="call-b", verified=True),
        )
    )
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    assert projection.read_calls("message-1") == ()


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(
    "verification_states",
    (("verified", "unverified"), ("unverified", "unverified")),
)
def test_mixed_verification_collision_falls_back_to_unambiguous_legacy(
    reverse: bool,
    verification_states: tuple[str, str],
) -> None:
    legacy = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="legacy"
    )
    calls = [
        NormalizedTraceCall(
            call_id="call-a",
            capture=replace(legacy, model="normalized-a"),
            abandoned=False,
            verification_status=verification_states[0],
        ),
        NormalizedTraceCall(
            call_id="call-b",
            capture=replace(legacy, model="normalized-b"),
            abandoned=False,
            verification_status=verification_states[1],
        ),
    ]
    if reverse:
        calls.reverse()
    readers = _Readers(normalized=calls, legacy=(_legacy_row(legacy),))
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    assert projection.read_calls("message-1") == (
        LegacyExchangeCall(capture=legacy, abandoned=False),
    )


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(
    "verification_states",
    (("verified", "unverified"), ("unverified", "unverified")),
)
def test_mixed_verification_collision_without_legacy_is_omitted(
    reverse: bool,
    verification_states: tuple[str, str],
) -> None:
    base = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="base"
    )
    calls = [
        NormalizedTraceCall(
            call_id="call-a",
            capture=replace(base, model="normalized-a"),
            abandoned=False,
            verification_status=verification_states[0],
        ),
        NormalizedTraceCall(
            call_id="call-b",
            capture=replace(base, model="normalized-b"),
            abandoned=False,
            verification_status=verification_states[1],
        ),
    ]
    if reverse:
        calls.reverse()
    readers = _Readers(normalized=calls)
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )

    assert projection.read_calls("message-1") == ()


def test_ambiguity_diagnostic_does_not_log_trace_identity_or_content() -> None:
    canary = "secret-trace-content-canary"
    base = _capture(
        run_tag="private-run-tag",
        seq=0,
        created_at="2026-08-28T10:00:00Z",
        model=canary,
    )
    readers = _Readers(
        normalized=(
            _normalized(base, call_id="private-call-a", verified=True),
            _normalized(
                replace(base, response={"content": "different"}),
                call_id="private-call-b",
                verified=True,
            ),
        )
    )
    projection = ConsoleTraceProjection(
        normalized_reader=readers.read_normalized,
        legacy_reader=readers.read_legacy,
        normalized_reads_enabled=True,
    )
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        assert projection.read_calls("private-message-id") == ()
    finally:
        logger.remove(sink_id)

    diagnostic = "\n".join(messages)
    assert "console_trace_projection_ambiguous: source=normalized" in diagnostic
    assert canary not in diagnostic
    assert "private-run-tag" not in diagnostic
    assert "private-call" not in diagnostic
    assert "private-message-id" not in diagnostic


def test_store_exposes_injected_projection_without_database_access() -> None:
    capture = _capture(
        run_tag="run-1", seq=0, created_at="2026-08-28T10:00:00Z", model="legacy"
    )
    projection = ConsoleTraceProjection(
        legacy_reader=lambda message_id: (
            (_legacy_row(capture),) if message_id == "persisted-1" else ()
        )
    )
    store = ConsoleChatStore(trace_projection=projection)

    assert store.projected_trace_calls("persisted-1") == (
        LegacyExchangeCall(capture=capture, abandoned=False),
    )


def test_runtime_injects_legacy_projection_with_rollout_write_default_on() -> None:
    class _DB:
        def get_message_exchanges(self, message_id: str) -> Sequence[dict]:
            assert message_id == "persisted-1"
            return ()

    runtime = ConsoleRuntime(
        SimpleNamespace(
            chachanotes_db=_DB(),
            citation_trace_repository=None,
            workspace_registry_service=None,
            persona_buddy_controller=None,
        )
    )

    store = runtime.ensure_chat_store()

    assert isinstance(store.trace_projection, ConsoleTraceProjection)
    assert store.trace_projection.normalized_reads_enabled is False
    assert store.trace_projection.normalized_writes_enabled is True
    assert store.projected_trace_calls("persisted-1") == ()


def test_runtime_builds_normalized_readers_only_on_first_normalized_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DB:
        def transaction(self) -> None:
            raise AssertionError("normalizer construction must remain lazy")

        def get_message_exchanges(self, message_id: str) -> Sequence[dict]:
            assert message_id == "persisted-1"
            return ()

    constructed: list[tuple[str, object]] = []

    class _Normalizer:
        def __init__(self, database: object) -> None:
            constructed.append(("legacy", database))

        def read_calls(self, message_id: str) -> tuple[()]:
            assert message_id == "persisted-1"
            return ()

    module = ModuleType("tldw_chatbook.Chat.console_trace_legacy")
    module.LegacyTraceNormalizer = _Normalizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class _NativeReader:
        def __init__(self, database: object, **_kwargs: object) -> None:
            constructed.append(("native", database))

        def read_calls(self, message_id: str) -> tuple[()]:
            assert message_id == "persisted-1"
            return ()

    native_module = ModuleType("tldw_chatbook.Chat.console_trace_native_reader")
    native_module.ConsoleTraceNativeReader = _NativeReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, native_module.__name__, native_module)
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime.recover_console_trace_calls",
        lambda _database: (),
    )
    database = _DB()
    runtime = ConsoleRuntime(
        SimpleNamespace(
            _ui_ready=False,
            chachanotes_db=database,
            citation_trace_repository=None,
            workspace_registry_service=None,
            persona_buddy_controller=None,
        )
    )

    store = runtime.ensure_chat_store()

    assert constructed == []
    assert store.projected_trace_calls("persisted-1") == ()
    assert constructed == [("native", database), ("legacy", database)]


@pytest.mark.asyncio
async def test_runtime_starts_legacy_maintenance_only_after_ui_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[object] = []
    provider_states: list[bool] = []

    class _Maintenance:
        def __init__(self, database: object, **kwargs: object) -> None:
            started.append(database)
            provider_active = kwargs["provider_active"]
            assert callable(provider_active)
            provider_states.append(provider_active())

        def run_batch(self) -> SimpleNamespace:
            return SimpleNamespace(logical_complete=True, admitted=True)

    module = ModuleType("tldw_chatbook.Chat.console_trace_maintenance")
    module.LegacyTraceMaintenance = _Maintenance  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime."
        "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS",
        0.01,
    )
    app = SimpleNamespace(_ui_ready=False, persona_buddy_controller=None)
    runtime = ConsoleRuntime(app)
    runtime._chat_controller = SimpleNamespace(_active_stream_tasks={"run": object()})
    normalizer = object()

    runtime._schedule_legacy_trace_maintenance(object(), lambda: normalizer)
    await asyncio.sleep(0.01)
    assert started == []

    app._ui_ready = True
    await asyncio.sleep(0)
    assert started == []
    await asyncio.sleep(0.07)
    assert len(started) == 1
    assert provider_states == [True]

    await runtime.dispose()


def test_legacy_trace_maintenance_keeps_a_post_mount_settling_delay() -> None:
    """Slow mount settling cannot pull migration imports onto first paint."""

    from tldw_chatbook.Chat.console_runtime import (
        LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS,
    )

    assert LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS >= 5.0


@pytest.mark.asyncio
async def test_runtime_runs_gc_then_physical_maintenance_after_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _Maintenance:
        def __init__(self, _database: object, **_kwargs: object) -> None:
            pass

        def run_batch(self) -> SimpleNamespace:
            return SimpleNamespace(logical_complete=True, admitted=True)

    class _Collector:
        def __init__(self, _database: object) -> None:
            pass

        def current_graph_epoch(self) -> int:
            return 7

        def collect(self, *, request_id: str) -> object:
            assert request_id.startswith("auto-")
            events.append("gc")
            return SimpleNamespace(marked_epoch=7)

    class _Compactor:
        def __init__(self, _database: object, **kwargs: object) -> None:
            assert callable(kwargs["pause_dispatch"])
            assert callable(kwargs["resume_dispatch"])
            assert kwargs["idle_seconds"]() == 47.0  # type: ignore[operator]
            events.append("compactor")

        def run_after_gc(self, _result: object) -> SimpleNamespace:
            events.append("vacuum")
            return SimpleNamespace(completed=True, reason_code="complete")

    module = ModuleType("tldw_chatbook.Chat.console_trace_maintenance")
    module.LegacyTraceMaintenance = _Maintenance  # type: ignore[attr-defined]
    module.TraceGarbageCollector = _Collector  # type: ignore[attr-defined]
    module.PhysicalTraceCompactor = _Compactor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        "tldw_chatbook.config.resolve_trace_compaction_policy",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime."
        "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS",
        0.0,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime."
        "TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS",
        0.0,
    )
    controller = SimpleNamespace(
        _active_stream_tasks={},
        pause_trace_maintenance_dispatch=lambda: events.append("pause"),
        resume_trace_maintenance_dispatch=lambda: events.append("resume"),
        trace_maintenance_idle_seconds=lambda: 47.0,
    )
    runtime = ConsoleRuntime(
        SimpleNamespace(
            _ui_ready=True,
            persona_buddy_controller=None,
            app_config={"console": {}},
        )
    )
    runtime._chat_controller = controller

    runtime._schedule_legacy_trace_maintenance(object(), object)
    await asyncio.sleep(1.08)

    assert events[:3] == ["gc", "compactor", "vacuum"]
    assert events.count("gc") == 1
    await runtime.dispose()


@pytest.mark.asyncio
async def test_runtime_retries_legacy_maintenance_after_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class _Maintenance:
        def __init__(self, _database: object, **_kwargs: object) -> None:
            pass

        def run_batch(self) -> SimpleNamespace:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected")
            return SimpleNamespace(logical_complete=True, admitted=True)

    module = ModuleType("tldw_chatbook.Chat.console_trace_maintenance")
    module.LegacyTraceMaintenance = _Maintenance  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime."
        "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS",
        0.0,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime."
        "LEGACY_TRACE_MAINTENANCE_RETRY_DELAY_SECONDS",
        0.01,
    )
    runtime = ConsoleRuntime(
        SimpleNamespace(_ui_ready=True, persona_buddy_controller=None)
    )

    runtime._schedule_legacy_trace_maintenance(object(), object)
    await asyncio.sleep(0.08)

    assert attempts >= 2
    await runtime.dispose()
