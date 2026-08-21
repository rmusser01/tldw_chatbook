"""Exact provider grammar for ephemeral nested project-instruction rows."""

from __future__ import annotations

import copy

import pytest

import tldw_chatbook.Chat.Chat_Functions as chat_functions
import tldw_chatbook.Chat.console_agent_bridge as console_agent_bridge
from tldw_chatbook.Agents.agent_models import FENCE_TOOL_RESULT_PREFIX
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
        "name": "fs_read",
        "content": "deferred-a",
    },
    {
        "role": "tool",
        "tool_call_id": "call-b",
        "name": "fs_list",
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


def test_fenced_transport_closes_complete_result_section_before_labeled_context():
    rows = [
        {"role": "assistant", "content": "tool-call"},
        {"role": "user", "content": f"{FENCE_TOOL_RESULT_PREFIX}fs_read: deferred-a"},
        {"role": "user", "content": f"{FENCE_TOOL_RESULT_PREFIX}fs_list: deferred-b"},
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
    assert combined["content"].index("fs_read: deferred-a") < combined["content"].index(
        "fs_list: deferred-b"
    )
    assert combined["content"].index("fs_list: deferred-b") < combined["content"].index(
        "Project instruction context:\n"
    )
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
