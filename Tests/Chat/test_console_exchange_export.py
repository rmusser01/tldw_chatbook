"""Privacy-governed export projections for one Console exchange capture."""

from __future__ import annotations

import json

import pytest
from loguru import logger

from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    ExchangeCapture,
    build_request_capture,
    build_response_capture,
    capture_from_storage,
    capture_to_blob,
)
from tldw_chatbook.Chat.console_exchange_export import (
    ExchangeExportUnavailable,
    project_exchange_export,
)
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Chat.trajectory_export import TraceExportProfile
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _capture(detail: CaptureDetail) -> ExchangeCapture:
    return ExchangeCapture(
        run_tag="run-1",
        seq=1,
        created_at="2026-08-26T00:00:00Z",
        provider="anthropic",
        model="claude-test",
        endpoint="https://user:pass@example.test/v1?api_key=leak#fragment",
        request={
            "system_message": "SYSTEM BODY",
            "messages_payload": [
                {
                    "role": "system",
                    "content": "AGENTS BODY",
                    EPHEMERAL_ORIGIN_KEY: "project_instructions",
                },
                {"role": "user", "content": "ordinary semantic secret"},
            ],
            "tools": [
                {
                    "name": "lookup",
                    "api_key": "sk-STRUCTURED",
                    "arguments": json.dumps(
                        {"token": "structured-tool-token", "query": "hello"}
                    ),
                }
            ],
            "truncation_inventory": ("tools[9]",),
        },
        response={
            "content": "answer",
            "tool_calls": [
                {
                    "name": "lookup",
                    "result": json.dumps(
                        {
                            "password": "structured-result-password",
                            "data": "A" * 5000,
                        }
                    ),
                }
            ],
            "truncation_inventory": ("content",),
        },
        status="complete",
        usage_json='{"input_tokens": 12, "output_tokens": 7}',
        omitted_keys=("api_key",),
        capture_detail=detail,
    )


def test_safe_summary_contains_only_metadata_and_inventories() -> None:
    projection = project_exchange_export(
        _capture(CaptureDetail.FULL), TraceExportProfile.SAFE_SUMMARY
    )

    assert projection.profile is TraceExportProfile.SAFE_SUMMARY
    assert projection.payload["capture_detail"] == "full"
    assert projection.payload["provider"] == "anthropic"
    assert projection.payload["omitted_keys"] == ["api_key"]
    assert projection.payload["truncation_inventory"] == {
        "request": ["tools[9]"],
        "response": ["content"],
    }
    assert "request" not in projection.payload
    assert "response" not in projection.payload
    assert "AGENTS BODY" not in projection.json_text


def test_redacted_diagnostic_reapplies_safe_instruction_redaction() -> None:
    projection = project_exchange_export(
        _capture(CaptureDetail.FULL), TraceExportProfile.REDACTED_DIAGNOSTIC
    )

    assert "AGENTS BODY" not in projection.json_text
    assert "ordinary semantic secret" in projection.json_text
    assert "messages_payload[0].content" in projection.payload["omitted_keys"]


def test_full_trace_is_unavailable_when_capture_was_safe() -> None:
    with pytest.raises(ExchangeExportUnavailable):
        project_exchange_export(
            _capture(CaptureDetail.SAFE), TraceExportProfile.FULL_TRACE
        )


def test_full_trace_keeps_semantic_bodies_but_blocks_structured_secrets() -> None:
    projection = project_exchange_export(
        _capture(CaptureDetail.FULL), TraceExportProfile.FULL_TRACE
    )

    assert projection.full_available is True
    assert projection.disabled_reason is None
    assert "AGENTS BODY" in projection.json_text
    assert "ordinary semantic secret" in projection.json_text
    assert "sk-STRUCTURED" not in projection.json_text
    assert "structured-tool-token" not in projection.json_text
    assert "structured-result-password" not in projection.json_text
    assert "user:pass" not in projection.json_text
    assert "api_key=leak" not in projection.json_text
    assert "sha256:" in projection.json_text


def test_non_full_profiles_report_why_full_is_unavailable_for_safe_capture() -> None:
    projection = project_exchange_export(
        _capture(CaptureDetail.SAFE), TraceExportProfile.REDACTED_DIAGNOSTIC
    )

    assert projection.full_available is False
    assert projection.disabled_reason == (
        "Full trace is unavailable because this call was captured in Safe mode."
    )


def test_production_shaped_anthropic_sentinels_across_storage_and_exports(
    tmp_path,
) -> None:
    semantic = {
        "system": "SYSTEM-SENTINEL-22507",
        "instructions": "AGENTS-WORKSPACE-SENTINEL-22507",
        "rag": "RAG-SENTINEL-22507",
        "schema": "TOOL-SCHEMA-SENTINEL-22507",
        "arguments": "TOOL-ARGS-SENTINEL-22507",
        "result": "TOOL-RESULT-SENTINEL-22507",
        "ordinary_secret": "ORDINARY-SEMANTIC-SECRET-22507",
    }
    structured = (
        "STRUCTURED-API-KEY-22507",
        "STRUCTURED-TOOL-KEY-22507",
        "endpoint-user-22507",
        "endpoint-pass-22507",
        "endpoint-query-22507",
        "endpoint-fragment-22507",
        "QUJD" * 2000,
    )
    raw_request = {
        "api_endpoint": "anthropic",
        "api_base_url": (
            "https://endpoint-user-22507:endpoint-pass-22507@example.test/v1"
            "?api_key=endpoint-query-22507#endpoint-fragment-22507"
        ),
        "api_key": structured[0],
        "system_message": semantic["system"],
        "messages_payload": [
            {
                "role": "system",
                "content": semantic["instructions"],
                EPHEMERAL_ORIGIN_KEY: "project_instructions",
            },
            {
                "role": "user",
                "content": f'{semantic["rag"]} {semantic["ordinary_secret"]}',
            },
        ],
        "tools": [
            {
                "name": "lookup",
                "description": semantic["schema"],
                "input_schema": {"api_key": structured[1], "type": "object"},
            }
        ],
    }
    raw_tool_calls = [
        {
            "name": "lookup",
            "arguments": json.dumps(
                {"query": semantic["arguments"], "api_key": structured[1]}
            ),
            "result": json.dumps(
                {
                    "value": semantic["result"],
                    "password": structured[1],
                    "nested_image": "data:image/png;base64," + structured[-1],
                }
            ),
        }
    ]

    captures: list[ExchangeCapture] = []
    for seq, detail in enumerate((CaptureDetail.SAFE, CaptureDetail.FULL)):
        request, omitted = build_request_capture(
            raw_request, capture_detail=detail
        )
        captures.append(
            ExchangeCapture(
                run_tag=f"sentinel-{detail.value}",
                seq=seq,
                created_at="2026-08-26T00:00:00Z",
                provider="anthropic",
                model="claude-test",
                endpoint=raw_request["api_base_url"],
                request=request,
                response=build_response_capture(
                    content="answer", tool_calls=raw_tool_calls
                ),
                status="complete",
                usage_json='{"input_tokens": 12, "output_tokens": 7}',
                omitted_keys=omitted,
                capture_detail=detail,
            )
        )

    log_path = tmp_path / "capture-inspection.log"
    sink_id = logger.add(log_path, diagnose=False)
    db = CharactersRAGDB(":memory:", client_id="capture-sentinel-inspection")
    try:
        conversation_id = db.add_conversation({"title": "sentinel inspection"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "sentinel inspection",
            }
        )
        db.append_message_exchanges_local(
            message_id,
            [
                {
                    "run_tag": capture.run_tag,
                    "seq": capture.seq,
                    "status": capture.status,
                    "abandoned": False,
                    "capture_detail": capture.capture_detail.value,
                    "capture_blob": capture_to_blob(capture),
                    "created_at": capture.created_at,
                }
                for capture in captures
            ],
        )
        decoded = tuple(
            capture_from_storage(row["capture_blob"], row["capture_detail"])
            for row in db.get_message_exchanges(message_id)
        )
        memory_owner = tuple(decoded)
        cache_owner = {capture.run_tag: capture for capture in memory_owner}

        safe_text = json.dumps(cache_owner["sentinel-safe"].request)
        full_text = json.dumps(cache_owner["sentinel-full"].request)
        redacted = project_exchange_export(
            cache_owner["sentinel-full"],
            TraceExportProfile.REDACTED_DIAGNOSTIC,
        ).json_text
        full_export = project_exchange_export(
            cache_owner["sentinel-full"], TraceExportProfile.FULL_TRACE
        ).json_text

        assert semantic["instructions"] not in safe_text
        assert semantic["instructions"] not in redacted
        assert semantic["instructions"] in full_text
        assert semantic["instructions"] in full_export
        for value in semantic.values():
            if value != semantic["instructions"]:
                assert value in full_export
        for value in structured:
            assert value not in safe_text
            assert value not in full_text
            assert value not in redacted
            assert value not in full_export
        assert "sha256:" in full_export
    finally:
        db.close_connection()
        logger.remove(sink_id)

    log_text = log_path.read_text(encoding="utf-8")
    for value in (*semantic.values(), *structured):
        assert value not in log_text
