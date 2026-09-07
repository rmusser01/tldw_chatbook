"""TASK-334: code_audit_tool._request_llm_analysis repointed to chat_api_call."""

import asyncio
from pathlib import Path
from unittest.mock import patch

from tldw_chatbook.Tools.code_audit_tool import FileAuditSystem


def test_request_llm_analysis_calls_chat_api_call_and_extracts_content():
    # _request_llm_analysis lives on FileAuditSystem, not the CodeAuditTool
    # Tool subclass (which has no such method).
    tool = FileAuditSystem()
    captured = {}

    def fake_chat_api_call(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "RISK: HIGH — hardcoded return"}}]}

    with patch(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call", side_effect=fake_chat_api_call
    ):
        result = asyncio.run(tool._request_llm_analysis("analyze this code"))

    assert result == "RISK: HIGH — hardcoded return"
    assert captured["api_endpoint"] == "anthropic"
    assert captured["messages_payload"] == [
        {"role": "user", "content": "analyze this code"}
    ]
    # TASK-19048: the analysis model must be currently served. The former
    # hardcoded claude-3-haiku-20240307 is RETIRED on the wire (404
    # not_found_error, probe req_011CeEDXZ8iS29MZCgyySwQa); claude-haiku-4-5
    # is its served successor in the same cheap-fast lineage (TASK-19020
    # precedent), preserving the site's "fast model for quick analysis" intent.
    assert captured["model"] == "claude-haiku-4-5"
    assert captured["streaming"] is False
    assert "timeout" not in captured  # dropped — chat_api_call has no timeout param


def test_no_dead_import_in_code_audit():
    import tldw_chatbook.Tools.code_audit_tool as m
    src = Path(m.__file__).read_text(encoding="utf-8")
    assert "import chat_with_provider" not in src
