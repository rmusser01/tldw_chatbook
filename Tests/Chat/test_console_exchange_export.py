"""Privacy-governed export projections for one Console exchange capture."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
from loguru import logger

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    ExchangeCapture,
    capture_from_storage,
)
from tldw_chatbook.Chat.console_exchange_export import (
    ExchangeExportUnavailable,
    project_exchange_export,
)
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Chat.trajectory_export import TraceExportProfile
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)


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


@pytest.mark.asyncio
async def test_real_gateway_controller_store_sentinels_across_all_owners(
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
    endpoint = (
        "https://endpoint-user-22507:endpoint-pass-22507@example.test/v1"
        "?api_key=endpoint-query-22507#endpoint-fragment-22507"
    )
    messages = [
        {"role": "system", "content": semantic["system"]},
        {
            "role": "system",
            "content": semantic["instructions"],
            EPHEMERAL_ORIGIN_KEY: "project_instructions",
        },
        {
            "role": "user",
            "content": f'{semantic["rag"]} {semantic["ordinary_secret"]}',
        },
        {
            "role": "tool",
            "content": semantic["result"],
            "tool_call_id": "call-1",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": semantic["schema"],
                "parameters": {
                    "api_key": structured[1],
                    "type": "object",
                    "example_image": "data:image/png;base64," + structured[-1],
                },
            },
        }
    ]

    def provider_call(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": "answer",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": json.dumps(
                                        {
                                            "query": semantic["arguments"],
                                            "api_key": structured[1],
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    gateway = ConsoleProviderGateway(chat_api_call_fn=provider_call)
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url=endpoint,
        model="gpt-test",
        ready=True,
        execution_key="openai",
        api_key=structured[0],
        streaming=False,
    )

    class Persistence:
        def __init__(self) -> None:
            self.rows: list[dict] = []
            self._message_count = 0

        def create_conversation(self, **_kwargs):
            return "conversation-sentinel"

        def create_message(self, **_kwargs):
            self._message_count += 1
            return f"message-{self._message_count}"

        def update_message_content(self, **_kwargs):
            return True

        def append_message_exchanges(self, *, message_id, rows):
            assert message_id == "message-1"
            self.rows = list(rows)
            return True

    persistence = Persistence()
    store = ConsoleChatStore(persistence=persistence)
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.ensure_session(title="sentinel inspection")
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(assistant.id, "answer")
    store.mark_message_complete(assistant.id)

    log_path = tmp_path / "capture-inspection.log"
    sink_id = logger.add(log_path, diagnose=False)
    try:
        signal_owners = []
        for detail in (CaptureDetail.SAFE, CaptureDetail.FULL):
            signals = ConsoleProviderStreamSignals(
                exchange_capture_enabled=True,
                capture_detail=detail,
            )
            _ = [
                chunk
                async for chunk in gateway.stream_chat(
                    resolution, messages, tools=tools, signals=signals
                )
            ]
            signal_owners.extend(signals.exchange_captures())
            controller._attach_stream_usage(
                assistant.id, signals, resolution, partial=False
            )

        store_owner = store.get_message(assistant.id).exchanges
        cache_owner = tuple(store._exchange_blob_cache[assistant.id].values())
        storage_owner = tuple(
            capture_from_storage(row["capture_blob"], row["capture_detail"])
            for row in persistence.rows
        )
        assert signal_owners and store_owner and cache_owner and storage_owner
        assert signal_owners is not store_owner
        safe = next(c for c in store_owner if c.capture_detail is CaptureDetail.SAFE)
        full = next(c for c in store_owner if c.capture_detail is CaptureDetail.FULL)

        safe_text = json.dumps(asdict(safe), default=str)
        full_text = json.dumps(asdict(full), default=str)
        redacted = project_exchange_export(
            full,
            TraceExportProfile.REDACTED_DIAGNOSTIC,
        ).json_text
        full_export = project_exchange_export(
            full, TraceExportProfile.FULL_TRACE
        ).json_text

        assert semantic["instructions"] not in safe_text
        assert semantic["instructions"] not in redacted
        assert semantic["instructions"] in full_text
        assert semantic["instructions"] in full_export
        for value in semantic.values():
            if value != semantic["instructions"]:
                assert value in full_export
        owner_texts = [
            *(json.dumps(asdict(c), default=str) for c in signal_owners),
            *(json.dumps(asdict(c), default=str) for c in store_owner),
            *(json.dumps(asdict(c), default=str) for c in storage_owner),
            redacted,
            full_export,
        ]
        for value in structured:
            for owner_index, owner in enumerate(owner_texts):
                assert value not in owner, (value, owner_index)
            assert all(value.encode() not in blob for blob in cache_owner)
        assert "sha256:" in full_export
    finally:
        logger.remove(sink_id)

    log_text = log_path.read_text(encoding="utf-8")
    for value in (*semantic.values(), *structured):
        assert value not in log_text
