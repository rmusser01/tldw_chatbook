"""Exact provider grammar for ephemeral nested project-instruction rows."""

from __future__ import annotations

import copy
import json

import httpx
import pytest

import tldw_chatbook.Chat.Chat_Functions as chat_functions
import tldw_chatbook.Chat.console_agent_bridge as console_agent_bridge
from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.project_instruction_runtime import (
    build_project_instruction_deferral_rows,
)
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY

_CONTEXT = "NESTED_GRAMMAR_SENTINEL"
_CANONICAL_NATIVE_ROWS = [
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-a",
                "type": "function",
                "function": {"name": "fs_read", "arguments": "{}"},
            },
            {
                "id": "call-b",
                "type": "function",
                "function": {"name": "fs_list", "arguments": "{}"},
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call-a",
        "content": "deferred-a",
    },
    {
        "role": "tool",
        "tool_call_id": "call-b",
        "content": "deferred-b",
    },
    {
        "role": "user",
        "content": _CONTEXT,
        EPHEMERAL_ORIGIN_KEY: "project_instructions",
    },
]


def test_native_transport_keeps_each_tool_result_before_separate_context_row():
    original = copy.deepcopy(_CANONICAL_NATIVE_ROWS)

    result = console_agent_bridge._serialize_project_instruction_rows_for_transport(
        _CANONICAL_NATIVE_ROWS, native_tools=True
    )

    assert [row["role"] for row in result] == ["assistant", "tool", "tool", "user"]
    assert result[-1][EPHEMERAL_ORIGIN_KEY] == "project_instructions"
    assert all(_CONTEXT not in str(row.get("content", "")) for row in result[1:3])
    assert _CANONICAL_NATIVE_ROWS == original


def test_native_adapter_keeps_native_context_grammar_with_empty_tool_catalog(
    monkeypatch,
):
    """Native transport mode must not be inferred from a non-empty tool list."""

    class ExpectedStop(Exception):
        pass

    seen: list[bool] = []

    def capture_mode(messages, *, native_tools):
        seen.append(native_tools)
        raise ExpectedStop

    monkeypatch.setattr(
        console_agent_bridge,
        "_serialize_project_instruction_rows_for_transport",
        capture_mode,
    )
    adapter = console_agent_bridge._StreamingModelAdapter(
        store=None,
        provider_gateway=None,
        resolution=None,
        assistant_message_id="grammar-test-assistant",
        should_cancel=lambda: False,
        loop=None,
        native_tools=True,
    )

    with pytest.raises(ExpectedStop):
        adapter.chat_call(messages_payload=_CANONICAL_NATIVE_ROWS, tools=None)

    assert seen == [True]


def test_fenced_transport_closes_complete_result_section_before_labeled_context():
    calls = (
        ToolCall("fs_read", {"path": "a"}, ""),
        ToolCall("fs_list", {"path": "b"}, ""),
    )
    rows = [
        {"role": "assistant", "content": "tool-call"},
        *build_project_instruction_deferral_rows(calls),
        {
            "role": "user",
            "content": _CONTEXT,
            EPHEMERAL_ORIGIN_KEY: "project_instructions",
        },
    ]

    result = console_agent_bridge._serialize_project_instruction_rows_for_transport(
        rows, native_tools=False
    )

    assert result[0] == rows[0]
    assert len(result) == 2
    combined = result[1]
    assert combined[EPHEMERAL_ORIGIN_KEY] == "project_instructions"
    assert combined["content"].index("fs_read") < combined["content"].index("fs_list")
    assert combined["content"].index("fs_list") < combined["content"].index(
        "Project instruction context:\n"
    )
    assert combined["content"].count("Deferred because project instructions") == 2
    assert combined["content"].endswith(_CONTEXT)


@pytest.mark.parametrize("endpoint", sorted(chat_functions.API_CALL_HANDLERS))
def test_final_handler_boundary_never_exposes_internal_marker(monkeypatch, endpoint):
    seen = {}

    def handler(**kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setitem(chat_functions.API_CALL_HANDLERS, endpoint, handler)
    marker_aware = endpoint in chat_functions.EPHEMERAL_GROUPING_ENDPOINTS
    chat_functions.chat_api_call(
        endpoint,
        messages_payload=copy.deepcopy(_CANONICAL_NATIVE_ROWS),
    )
    payload_key = chat_functions.PROVIDER_PARAM_MAP[endpoint]["messages_payload"]
    sent = seen[payload_key]
    assert all(marker_aware or EPHEMERAL_ORIGIN_KEY not in row for row in sent)
    assert _CANONICAL_NATIVE_ROWS[-1][EPHEMERAL_ORIGIN_KEY] == "project_instructions"


@pytest.mark.asyncio
async def test_native_review_read_deferral_resumes_through_strict_hosted_payload():
    """A context-only retry must reach the next request without dispatching early."""
    from tldw_chatbook.Agents.agent_models import (
        RUN_DONE,
        AgentConfig,
        ModelTurn,
        ToolResult,
    )
    from tldw_chatbook.Agents.agent_runtime import (
        LoopDeps,
        ToolBatchPreparation,
        run_agent_loop,
    )
    from tldw_chatbook.Agents.project_instruction_runtime import (
        PROJECT_INSTRUCTION_ROW_KEY,
        InstructionDeliveryReceipt,
    )
    from tldw_chatbook.Chat.console_prepared_request import thaw_json
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )
    from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
        HostedProviderResolution,
        build_hosted_chat_payload,
    )
    from tldw_chatbook.provider_registry import CUSTOM_HOSTED

    def no_network(_request):
        raise AssertionError("offline regression must not dispatch HTTP")

    client = httpx.AsyncClient(transport=httpx.MockTransport(no_network))
    gateway = ConsoleProviderGateway(
        http_client=client, config_provider=dict, environ={}
    )
    resolution = ConsoleProviderResolution(
        provider="custom-ep:ac24-local",
        execution_key="custom-hosted",
        model="qwen",
        base_url="http://127.0.0.1:8010/v1",
        ready=True,
        max_tokens=1024,
    )
    hosted = HostedProviderResolution(
        provider="custom-hosted",
        model="qwen",
        api_key="",
        base_url=resolution.base_url,
        timeout=1.0,
        retries=0,
        retry_delay=0.0,
        streaming=True,
    )
    context = {
        "role": "user",
        "content": _CONTEXT,
        EPHEMERAL_ORIGIN_KEY: "project_instructions",
        PROJECT_INSTRUCTION_ROW_KEY: "review-source-context",
    }
    receipt = InstructionDeliveryReceipt(
        receipt_id="review-source-receipt",
        chain_id="verification-continuation",
        through_revision=1,
        source_digests=(),
        outcome_keys=(),
        row_keys=("review-source-context",),
    )

    def native_turn(ids):
        calls = tuple(
            ToolCall("code_review_read", {"view": "source"}, cid) for cid in ids
        )
        return ModelTurn(
            tool_calls=calls,
            assistant_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": c.call_id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": json.dumps(c.args),
                        },
                    }
                    for c in calls
                ],
            },
        )

    turns = iter(
        [
            native_turn(("deferred-a", "deferred-b")),
            native_turn(("accepted-a",)),
            ModelTurn(text="done"),
        ]
    )
    payloads, reviewed, executed = [], [], []

    def call_model(messages, _active):
        original = copy.deepcopy(messages)
        rows = console_agent_bridge._serialize_project_instruction_rows_for_transport(
            messages, native_tools=True
        )
        prepared = gateway.prepare_chat_request(
            resolution,
            rows,
            context_window_override_tokens=32768,
            apply_safety_window=False,
        )
        handler_kwargs = chat_functions.project_chat_handler_kwargs(
            "custom-hosted",
            {"messages_payload": thaw_json(prepared.messages_payload)},
        )
        payloads.append(
            build_hosted_chat_payload(
                CUSTOM_HOSTED,
                resolution=hosted,
                messages_payload=handler_kwargs["input_data"],
                system_message=prepared.system_message,
            )
        )
        assert messages == original
        return next(turns)

    def prepare(calls):
        if calls[0].call_id == "deferred-a":
            return ToolBatchPreparation("retry_with_context", (context,), receipt)
        return ToolBatchPreparation("proceed")

    try:
        outcome = run_agent_loop(
            AgentConfig(
                model="qwen", system_prompt="", allowed_tools=("code_review_read",)
            ),
            [{"role": "user", "content": "Continue the admitted verification"}],
            [],
            LoopDeps(
                call_model=call_model,
                prepare_tool_calls=prepare,
                review_tool_calls=lambda calls: (
                    reviewed.extend(c.call_id for c in calls) or {}
                ),
                invoke_tool=lambda call: (
                    executed.append(call.call_id)
                    or ToolResult(ok=True, content="captured source")
                ),
                spawn=lambda _task: ToolResult(
                    ok=False, error="no child in this control"
                ),
                find_tools=lambda _query: [],
                load_schemas=lambda _ids: [],
                should_cancel=lambda: False,
                clock=lambda: 0.0,
            ),
        )
        assert outcome.status == RUN_DONE and outcome.final_text == "done"
        assert reviewed == executed == ["accepted-a"]
        deferred = payloads[1]["messages"][-3:]
        assert [row["role"] for row in deferred] == ["tool", "tool", "user"]
        assert [row["tool_call_id"] for row in deferred[:2]] == [
            "deferred-a",
            "deferred-b",
        ]
        assert all(
            set(row) == {"role", "tool_call_id", "content"} for row in deferred[:2]
        )
        assert all(_CONTEXT not in row["content"] for row in deferred[:2])
        assert deferred[-1] == {"role": "user", "content": _CONTEXT}
        assert payloads[-1]["messages"][-1] == {
            "role": "tool",
            "tool_call_id": "accepted-a",
            "content": "captured source",
        }
    finally:
        await gateway.aclose()
        await client.aclose()
